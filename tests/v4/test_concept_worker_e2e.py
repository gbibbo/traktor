"""
PURPOSE: Prueba de punta a punta del Worker de conceptos (cloudflare/concept-ai-worker.js, el código real, en
         `wrangler dev` con una D1 local) y del agente del ordenador (conceptos_agente.py, con un codificador falso).
         No toca Cloudflare ni necesita cuenta. Se activa con TRAKTOR_WORKER_E2E=1 porque arranca Node y wrangler:
           TRAKTOR_WORKER_E2E=1 python -m pytest tests/v4/test_concept_worker_e2e.py -q
         Cubre: crear un trabajo con la firma real, idempotencia al guardar, reclamar y completar, el contrato de
         resultados, job_id para no consumir conceptos reales, lease y tope de intentos, generación del reclamo.
CHANGELOG:
  - 2026-11-09: Creación inicial.
"""
import hashlib
import hmac
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.traktor_live import conceptos_agente as ca  # noqa: E402
from tools.traktor_live import recomendador as rc  # noqa: E402

pytestmark = pytest.mark.skipif(os.environ.get("TRAKTOR_WORKER_E2E") != "1", reason="TRAKTOR_WORKER_E2E=1 para correrla")

WORKER_DIR = REPO_ROOT / "cloudflare"
APP = "clave-app-de-prueba"
DEVICE = "token-ordenador-de-prueba"
LEASE_MS = 1200
WRANGLER = "wrangler@4.159.0"
DUMMY_ID = "00000000-0000-0000-0000-000000000000"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    npx = shutil.which("npx")
    if npx is None:
        pytest.skip("hace falta Node (npx)")
    work = tmp_path_factory.mktemp("worker")
    state = work / "state"
    # Sin el binding AI: en local, wrangler lo manda a la cuenta de Cloudflare y esta prueba no la usa.
    config = {"name": "traktor-concept-ai-e2e", "main": os.path.relpath(WORKER_DIR / "concept-ai-worker.js", work),
              "compatibility_date": "2026-11-01",
              "d1_databases": [{"binding": "DB", "database_name": "traktor-concepts", "database_id": DUMMY_ID}]}
    config_path = work / "wrangler.jsonc"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    env = {**os.environ, "CI": "1", "WRANGLER_SEND_METRICS": "false", "NO_COLOR": "1"}
    subprocess.run([npx, "--yes", WRANGLER, "d1", "execute", "traktor-concepts", "--local", "--persist-to", str(state),
                    "--config", str(config_path), "--file", str(WORKER_DIR / "schema.sql")],
                   check=True, cwd=work, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                   timeout=240)
    port = free_port()
    log = open(work / "wrangler.log", "w", encoding="utf-8", errors="replace")
    process = subprocess.Popen([npx, "--yes", WRANGLER, "dev", "--local", "--ip", "127.0.0.1", "--port", str(port),
                                "--persist-to", str(state), "--config", str(config_path),
                                "--var", f"APP_PASSWORD:{APP}", "--var", f"DEVICE_TOKEN:{DEVICE}",
                                "--var", f"LEASE_MS:{LEASE_MS}", "--show-interactive-dev-session=false"],
                               cwd=work, env=env, stdout=log, stderr=log, stdin=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(240):
            try:
                urllib.request.urlopen(url + "/", timeout=2).read()
                break
            except OSError:
                time.sleep(0.5)
        else:
            log.flush()
            pytest.fail("wrangler dev no arrancó: " + (work / "wrangler.log").read_text(encoding="utf-8", errors="replace")[-1500:])
        yield url
    finally:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True)
        else:
            process.terminate()
        process.wait(timeout=20)
        log.close()


def call(url, method, path, headers=None, body=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url + path, data=data, method=method, headers={
        "Content-Type": "application/json", "User-Agent": "e2e", **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=20) as r:
            return r.status, json.loads(r.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def app(url, method, path, body=None):
    return call(url, method, path, {"X-App-Password": APP}, body)


def device(url, path, body=None):
    return call(url, "POST", path, {"X-Device-Token": DEVICE}, {} if body is None else body)


counter = {"n": 0}


def concept(name=None):
    counter["n"] += 1
    return {"version": 1, "name": name or f"Concepto {counter['n']}", "description": "Una descripción suficientemente larga.",
            "positive_prompts": ["Dark hypnotic techno with a deep bassline"], "negative_prompts": [],
            "constraints": {"energy": "unspecified", "vocals": "absent", "bpm": "unspecified"}}


def signed(c, expires=None):
    """La firma que arma /api/draft: HMAC de JSON.stringify({concept, expires}) con APP_PASSWORD."""
    expires = expires or int(time.time() * 1000) + 10 * 60 * 1000
    payload = json.dumps({"concept": c, "expires": expires}, ensure_ascii=False, separators=(",", ":"))
    return {"concept": c, "expires": expires, "confirmed": True,
            "signature": hmac.new(APP.encode(), payload.encode("utf-8"), hashlib.sha256).hexdigest()}


def create_job(url, c=None):
    status, data = app(url, "POST", "/api/jobs", signed(c or concept()))
    assert status == 201, data
    return data["id"]


def drain(url):
    """Deja la cola vacía: la D1 es compartida entre tests."""
    while True:
        _, data = device(url, "/api/device/claim")
        if data["job"] is None:
            return
        device(url, "/api/device/complete", {"id": data["job"]["id"], "attempts": data["job"]["attempts"],
                                             "status": "failed", "error": "limpieza"})


def test_page_is_served_with_its_security_headers(server):
    with urllib.request.urlopen(server + "/", timeout=10) as r:
        html = r.read().decode("utf-8")
        assert "Nuevo concepto" in html and "default-src 'none'" in r.headers["Content-Security-Policy"]
        assert r.headers["X-Frame-Options"] == "DENY"


def test_auth_is_required(server):
    assert call(server, "GET", "/api/health")[0] == 401
    assert call(server, "POST", "/api/device/claim", {"X-Device-Token": "mal"}, {})[0] == 401
    assert call(server, "GET", "/api/device/claim", {"X-Device-Token": DEVICE})[0] == 405


def test_bad_signature_and_expired_draft_are_rejected(server):
    body = signed(concept())
    body["signature"] = "0" * 64
    assert app(server, "POST", "/api/jobs", body)[0] == 400
    assert app(server, "POST", "/api/jobs", signed(concept(), expires=int(time.time() * 1000) - 1000))[0] == 400
    tampered = signed(concept())
    tampered["concept"] = {**tampered["concept"], "name": "Otro nombre"}
    assert app(server, "POST", "/api/jobs", tampered)[0] == 400


def test_saving_the_same_draft_twice_creates_one_job(server):
    drain(server)
    before = app(server, "GET", "/api/health")[1]["jobs"]
    body = signed(concept())
    first = app(server, "POST", "/api/jobs", body)
    second = app(server, "POST", "/api/jobs", body)
    assert first[0] == second[0] == 201
    assert first[1]["id"] == second[1]["id"] and first[1]["status"] == "queued"
    assert app(server, "GET", "/api/health")[1]["jobs"] == before + 1
    drain(server)


def test_agent_completes_a_job_end_to_end(server, tmp_path):
    """Trabajo real -> agente real -> done -> el teléfono lo lee. Con caracteres que no son ASCII."""
    drain(server)
    job_id = create_job(server, concept("Techno oscuro"))
    items = [rc.Item(uid="u1", rel="Musica\\a.mp3", original="o\\a.mp3", artist="Ben Böhmer", title="Beyond Beliefs — Ñandú",
                     bpm=124.0, key="8A", vec=[1.0, 0.0]),
             rc.Item(uid="u2", rel="Musica\\b.mp3", original="o\\b.mp3", artist="", title="Sin artista", bpm=None, key="?", vec=[0.6, 0.8]),
             rc.Item(uid="u3", rel="Musica\\c.mp3", original="o\\c.mp3", artist="Vocal", title="Con voz", bpm=128.0, key="9A", vec=[1.0, 0.0])]
    bib = ca.Biblioteca(items=items, matrix=np.array([i.vec for i in items], dtype=np.float32),
                        vocal={"u1": False, "u2": False, "u3": True}, recommender=rc.Recommender(items, {}, 30))
    client = ca.WorkerClient(server, DEVICE)
    agent = ca.Agente(client, lambda *a: None, cargar=lambda: bib, codificar=lambda _: np.array([[1.0, 0.0]]),
                      sesion=ca.Sesion, nml=lambda b, c: tmp_path / "x.nml", results_log=None)
    assert ca.bucle(agent, threading.Event(), una_vez=True) == 0
    status, job = app(server, "GET", f"/api/jobs/{job_id}")
    assert status == 200 and job["status"] == "done" and job["error"] is None
    assert job["concept"]["name"] == "Techno oscuro"
    tracks = job["result"]["tracks"]
    assert [t["track_id"] for t in tracks] == ["u1", "u2"]          # sin el Vocal
    assert tracks[0] == {"track_id": "u1", "title": "Beyond Beliefs — Ñandú", "artist": "Ben Böhmer", "score": 1.0,
                         "bpm": 124.0, "key": "8A", "vocal": False}
    assert "bpm" not in tracks[1] and "key" not in tracks[1]
    assert job["result"]["pool"] == 3 and job["result"]["passed"] == 2
    assert any("Sin voces" in n for n in job["result"]["notes"])


def test_claim_with_job_id_leaves_older_jobs_queued(server):
    drain(server)
    older = create_job(server)
    newer = create_job(server)
    status, data = device(server, "/api/device/claim", {"job_id": newer})
    assert status == 200 and data["job"]["id"] == newer
    assert app(server, "GET", f"/api/jobs/{older}")[1]["status"] == "queued"
    assert device(server, "/api/device/claim", {"job_id": newer})[1]["job"] is None   # ya no está en cola
    assert device(server, "/api/device/claim", {"job_id": "x"})[0] == 400
    device(server, "/api/device/complete", {"id": newer, "attempts": 1, "status": "failed", "error": "fin"})
    drain(server)


def test_claim_is_fifo_and_empty_body_is_accepted(server):
    drain(server)
    first = create_job(server)
    time.sleep(0.01)
    second = create_job(server)
    assert call(server, "POST", "/api/device/claim", {"X-Device-Token": DEVICE})[1]["job"]["id"] == first
    assert device(server, "/api/device/claim")[1]["job"]["id"] == second
    drain(server)


def test_v3_result_shape_is_still_accepted(server):
    drain(server)
    job_id = create_job(server)
    job = device(server, "/api/device/claim", {"job_id": job_id})[1]["job"]
    code, _ = device(server, "/api/device/complete", {"id": job_id, "attempts": job["attempts"], "status": "done",
                                                      "result": {"tracks": [{"track_id": "a", "title": "T", "artist": "", "score": 0.5}]}})
    assert code == 200
    assert app(server, "GET", f"/api/jobs/{job_id}")[1]["result"] == {"tracks": [{"track_id": "a", "title": "T", "artist": "", "score": 0.5}]}


@pytest.mark.parametrize("bad", [
    {"tracks": [{"track_id": "a", "title": "T", "artist": "", "score": 1.5}]},
    {"tracks": [{"track_id": "a", "title": "T", "artist": "", "score": 0.5, "bpm": -3}]},
    {"tracks": [{"track_id": "a", "title": "T", "artist": "", "score": 0.5, "vocal": "si"}]},
    {"tracks": [], "notes": ["x" * 201]},
    {"tracks": [], "pool": -1},
    {"tracks": [{"track_id": "a", "title": "T", "artist": "", "score": 0.5}] * 31},
])
def test_invalid_results_are_rejected_and_the_job_stays_open(server, bad):
    drain(server)
    job_id = create_job(server)
    job = device(server, "/api/device/claim", {"job_id": job_id})[1]["job"]
    assert device(server, "/api/device/complete", {"id": job_id, "attempts": job["attempts"], "status": "done", "result": bad})[0] == 400
    assert app(server, "GET", f"/api/jobs/{job_id}")[1]["status"] == "processing"
    device(server, "/api/device/complete", {"id": job_id, "attempts": job["attempts"], "status": "failed", "error": "fin"})


def test_abandoned_job_returns_to_the_queue_and_a_late_complete_is_refused(server):
    drain(server)
    job_id = create_job(server)
    first = device(server, "/api/device/claim", {"job_id": job_id})[1]["job"]
    assert first["attempts"] == 1
    time.sleep(LEASE_MS / 1000 + 0.4)
    second = device(server, "/api/device/claim", {"job_id": job_id})[1]["job"]
    assert second is not None and second["attempts"] == 2
    result = {"tracks": []}
    assert device(server, "/api/device/complete", {"id": job_id, "attempts": 1, "status": "done", "result": result})[0] == 409
    assert device(server, "/api/device/complete", {"id": job_id, "attempts": 2, "status": "done", "result": result})[0] == 200
    assert app(server, "GET", f"/api/jobs/{job_id}")[1]["status"] == "done"


def test_a_processing_job_within_its_lease_is_not_taken_again(server):
    drain(server)
    job_id = create_job(server)
    assert device(server, "/api/device/claim", {"job_id": job_id})[1]["job"]["attempts"] == 1
    assert device(server, "/api/device/claim")[1]["job"] is None
    assert app(server, "GET", f"/api/jobs/{job_id}")[1]["status"] == "processing"
    device(server, "/api/device/complete", {"id": job_id, "attempts": 1, "status": "failed", "error": "fin"})


def test_job_fails_after_max_attempts(server):
    drain(server)
    job_id = create_job(server)
    for attempt in (1, 2, 3):
        assert device(server, "/api/device/claim", {"job_id": job_id})[1]["job"]["attempts"] == attempt
        time.sleep(LEASE_MS / 1000 + 0.4)
    assert device(server, "/api/device/claim", {"job_id": job_id})[1]["job"] is None
    job = app(server, "GET", f"/api/jobs/{job_id}")[1]
    assert job["status"] == "failed" and "no terminó" in job["error"]


def test_completing_a_finished_job_twice_is_refused(server):
    drain(server)
    job_id = create_job(server)
    job = device(server, "/api/device/claim", {"job_id": job_id})[1]["job"]
    body = {"id": job_id, "attempts": job["attempts"], "status": "done", "result": {"tracks": []}}
    assert device(server, "/api/device/complete", body)[0] == 200
    assert device(server, "/api/device/complete", body)[0] == 409
