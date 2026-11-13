"""
PURPOSE: Prueba de punta a punta del circuito de conceptos: el Worker real (cloudflare/concept-ai-worker.js) en `wrangler dev`
         con una D1 local y el agente del ordenador (conceptos_agente.py, con un codificador falso). El LLM es un doble que
         se inyecta envolviendo al Worker (no hay código de prueba en producción) y responde según palabras del pedido, para
         forzar cada camino: JSON roto, reintento, juez que rechaza, juez caído, textos prohibidos, ajuste.
         No toca Cloudflare ni necesita cuenta. Se activa con TRAKTOR_WORKER_E2E=1 porque arranca Node y wrangler:
           TRAKTOR_WORKER_E2E=1 python -m pytest tests/v4/test_concept_worker_e2e.py -q
CHANGELOG:
  - 2026-11-09: Creación inicial (Worker v4: firma, idempotencia, lease, contrato de resultados).
  - 2026-11-09: Worker v5: circuito con el LLM (pedido, sondeo, decisión, final), validadores, ajuste, conceptos viejos.
  - 2026-11-13: Worker v6: presets sin LLM (formulario en el trabajo, versión 3), presets y texto juntos, la ruta /api/estado
                con lo que suena y el catálogo que manda el ordenador, resultados con pasos, lo ignorado y candidatos.
"""
import hashlib
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

# El doble del LLM. Interpretar: lee el pedido; elegir: lee lo que pidió el DJ. Palabras clave en el pedido:
#   ROTO (nunca JSON), MALO_UNA (JSON roto la primera vez), PROHIBIDO (un texto con envoltura y otro con ocasión),
#   RECHAZA (el juez dice mal siempre), QUEDA_A (el juez deja solo A), JUEZ_CAIDO (el juez nunca responde bien).
STUB = r"""
import worker from "%(worker)s";
const counts = {};
function texts(prefix) {
  return [prefix + " dark techno with a rolling bassline", prefix + " hypnotic techno with deep kick drums", prefix + " minimal techno with metallic percussion"];
}
let llmCalls = 0;
const AI = { async run(model, input) {
  llmCalls++;
  const system = input.messages[0].content, user = input.messages[1].content;
  const key = (user.match(/Request: (.*)/) || user.match(/The DJ asked: (.*)/) || [null, user])[1];
  counts[key + system.length] = (counts[key + system.length] || 0) + 1;
  const first = counts[key + system.length] === 1;
  const json = o => ({ choices: [{ message: { content: JSON.stringify(o) } }] });
  if (system.startsWith("You rewrite")) {
    if (user.includes("ROTO") || (user.includes("MALO_UNA") && first)) return { choices: [{ message: { content: "no es json" } }] };
    let textos = texts("round one");
    if (user.includes("POCA_VOZ")) textos = ["mellow house with minimal vocal snippets", "soft house with very few vocals", "warm house with female vocals"];
    if (user.includes("PROHIBIDO")) textos = ["this is a track of dark techno", "dark techno for a birthday party", "ambient pads with no percussion", "hypnotic techno with deep kick drums", "minimal techno with metallic percussion"];
    const entendi = user.includes("Now the DJ says") ? "ajuste: " + user.split("Now the DJ says: ")[1].split("\n")[0] : "Techno oscuro";
    return json({ entendi, no_se_oye: ["after"], nombre: "After oscuro", textos });
  }
  if (user.includes("JUEZ_CAIDO")) return { choices: [{ message: { content: "{" } }] };
  if (user.includes("RECHAZA")) return json({ veredicto: "mal", problema: "Salió house", textos: texts("round two") });
  if (user.includes("QUEDA_A")) return json({ veredicto: "ok", quedan: ["A", "Z"], resumen: "Solo la primera" });
  return json({ veredicto: "ok", quedan: ["A", "C"], resumen: "Techno oscuro con bajos" });
} };
export default { fetch(request, env, ctx) {
  if (new URL(request.url).pathname === "/__llm_calls") return Response.json({ calls: llmCalls });
  return worker.fetch(request, { ...env, AI }, ctx);
} };
"""


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
    (work / "stub.js").write_text(STUB % {"worker": (WORKER_DIR / "concept-ai-worker.js").as_posix()}, encoding="utf-8")
    config = {"name": "traktor-concept-ai-e2e", "main": "stub.js", "compatibility_date": "2026-11-09",
              "d1_databases": [{"binding": "DB", "database_name": "traktor-concepts", "database_id": DUMMY_ID}]}
    config_path = work / "wrangler.jsonc"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    env = {**os.environ, "CI": "1", "WRANGLER_SEND_METRICS": "false", "NO_COLOR": "1"}
    base = [npx, "--yes", WRANGLER]
    d1 = base + ["d1", "execute", "traktor-concepts", "--local", "--persist-to", str(state), "--config", str(config_path)]
    subprocess.run(d1 + ["--file", str(WORKER_DIR / "schema.sql")], check=True, cwd=work, env=env, capture_output=True, timeout=240)
    port = free_port()
    log = open(work / "wrangler.log", "w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(base + ["dev", "--local", "--ip", "127.0.0.1", "--port", str(port), "--persist-to", str(state),
                                       "--config", str(config_path), "--var", f"APP_PASSWORD:{APP}", "--var", f"DEVICE_TOKEN:{DEVICE}",
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
        yield {"url": url, "d1": d1, "cwd": work, "env": env}
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
        with urllib.request.urlopen(request, timeout=30) as r:
            return r.status, json.loads(r.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def app(s, method, path, body=None):
    return call(s["url"], method, path, {"X-App-Password": APP}, body)


def device(s, path, body=None):
    return call(s["url"], "POST", path, {"X-Device-Token": DEVICE}, {} if body is None else body)


def pedido(s, texto="algo oscuro tipo after", anterior=None):
    body = {"texto": texto}
    if anterior:
        body["anterior"] = anterior
    status, data = app(s, "POST", "/api/conceptos", body)
    assert status == 201, data
    return data


def drain(s):
    """Deja la cola vacía: la D1 es compartida entre tests."""
    while True:
        _, data = device(s, "/api/device/claim")
        if data["job"] is None:
            return
        device(s, "/api/device/complete", {"id": data["job"]["id"], "attempts": data["job"]["attempts"], "status": "failed", "error": "limpieza"})


def fake_codificar(frases):
    """Un vector 2D estable por frase."""
    out = []
    for f in frases:
        h = int(hashlib.md5(f.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF * 6.283
        out.append([np.cos(h), np.sin(h)])
    return np.array(out)


@pytest.fixture
def agent(tmp_path):
    items = [rc.Item(uid=f"u{i}", rel=f"Musica\\{i}.mp3", original=f"o\\{i}.mp3", artist="Ben Böhmer" if i == 0 else f"Artista {i}",
                     title="Beyond — Ñandú" if i == 0 else f"Tema {i}", bpm=120.0 + i, key="8A", vec=list(fake_codificar([f"t{i}"])[0]),
                     genre="Techno" if i % 2 else "House") for i in range(12)]
    bib = ca.Biblioteca(items=items, matrix=np.array([i.vec for i in items], dtype=np.float32), vocal={f"u{i}": i % 3 == 0 for i in range(12)})
    escritas = []

    def nml(b, chosen, nombre):
        escritas.append((nombre, [it.uid for it in chosen]))
        return ca.nombre_playlist(nombre)

    def make(s):
        ag = ca.Agente(ca.WorkerClient(s["url"], DEVICE), lambda *a: None, cargar=lambda: bib, codificar=fake_codificar, nml=nml, results_log=None)
        ag.escritas = escritas
        return ag
    return make


def correr(ag, job_id):
    assert ca.bucle(ag, threading.Event(), job_id=job_id, una_vez=True, poll_s=0.05, poll_activo_s=0.05, activo_s=3.0) == 0


# ---------------------------------------------------------------- página y acceso

def test_page_is_served_with_its_security_headers(server):
    with urllib.request.urlopen(server["url"] + "/", timeout=10) as r:
        html = r.read().decode("utf-8")
        assert "Buscar en mi música" in html and "default-src 'none'" in r.headers["Content-Security-Policy"]
        assert r.headers["X-Frame-Options"] == "DENY"


def test_auth_is_required(server):
    assert call(server["url"], "GET", "/api/health")[0] == 401
    assert call(server["url"], "POST", "/api/conceptos", {}, {"texto": "hola"})[0] == 401
    assert call(server["url"], "POST", "/api/device/claim", {"X-Device-Token": "mal"}, {})[0] == 401
    assert call(server["url"], "GET", "/api/device/claim", {"X-Device-Token": DEVICE})[0] == 405
    assert app(server, "POST", "/api/chat", {"messages": []})[0] == 404          # las rutas de la v4 ya no existen


# ---------------------------------------------------------------- el pedido

def test_request_is_interpreted_and_queued_for_searches(server):
    drain(server)
    job = pedido(server)
    c = job["concept"]
    assert job["status"] == "queued" and c["version"] == 3 and c["fase"] == "sondeo" and c["ronda"] == 1
    assert c["formulario"] == ca.fm.vacio() and c["presets"] == []
    assert c["original"] == "algo oscuro tipo after" and c["entendi"] == "Techno oscuro" and c["nombre"] == "After oscuro"
    assert [t["id"] for t in c["textos"]] == ["A", "B", "C"] and c["no_se_oye"] == ["after"]
    drain(server)


def test_empty_request_and_broken_llm(server):
    assert app(server, "POST", "/api/conceptos", {"texto": " "})[0] == 400
    status, data = app(server, "POST", "/api/conceptos", {"texto": "ROTO"})
    assert status == 502 and "interpretar" in data["error"]
    assert pedido(server, "MALO_UNA")["concept"]["textos"]                    # un reintento alcanza
    drain(server)


def test_texts_with_wrappers_or_occasions_are_dropped(server):
    drain(server)
    textos = [t["texto"] for t in pedido(server, "PROHIBIDO")["concept"]["textos"]]
    assert textos == ["hypnotic techno with deep kick drums", "minimal techno with metallic percussion"]
    drain(server)


def test_less_voice_is_always_written_as_without_vocals(server):
    drain(server)
    textos = [t["texto"] for t in pedido(server, "POCA_VOZ")["concept"]["textos"]]
    assert textos == ["mellow house without vocals", "soft house without vocals", "warm house with female vocals"]
    drain(server)


# ---------------------------------------------------------------- el circuito

def test_full_circuit_ends_in_a_playlist(server, agent):
    drain(server)
    ag = agent(server)
    job = pedido(server)
    correr(ag, job["id"])
    status, final = app(server, "GET", f"/api/jobs/{job['id']}")
    c = final["concept"]
    assert final["status"] == "done" and c["fase"] == "final" and final["error"] is None
    assert [t["id"] for t in c["textos"]] == ["A", "C"] and c["resumen"] == "Techno oscuro con bajos"
    ronda = c["rondas"][0]
    assert ronda["decision"]["veredicto"] == "ok" and set(ronda["busquedas"]) == {"A", "B", "C"}
    b = ronda["busquedas"]["A"]
    assert b["temas"] and b["suena_a"] and sum(b["generos"].values()) == 12 and 0 <= b["voz"] <= 1
    r = final["result"]
    assert r["playlist"].startswith("Concepto ") and r["playlist"].endswith("After oscuro") and len(r["tracks"]) == 12
    assert any(t["title"] == "Beyond — Ñandú" and t["artist"] == "Ben Böhmer" for t in r["tracks"])
    assert ag.escritas[-1][0] == "After oscuro"


def test_judge_rejects_twice_then_the_second_round_is_used(server, agent):
    drain(server)
    job = pedido(server, "RECHAZA")
    correr(agent(server), job["id"])
    c = app(server, "GET", f"/api/jobs/{job['id']}")[1]["concept"]
    assert c["fase"] == "final" and c["ronda"] == 2 and len(c["rondas"]) == 2 and c["problema"] == "Salió house"
    assert all(t["texto"].startswith("round two") for t in c["textos"]) and len(c["textos"]) == 3


def test_judge_keeps_only_valid_ids(server, agent):
    drain(server)
    job = pedido(server, "QUEDA_A")
    correr(agent(server), job["id"])
    c = app(server, "GET", f"/api/jobs/{job['id']}")[1]["concept"]
    assert [t["id"] for t in c["textos"]] == ["A"]


def test_broken_judge_falls_back_to_all_texts(server, agent):
    drain(server)
    job = pedido(server, "JUEZ_CAIDO")
    correr(agent(server), job["id"])
    final = app(server, "GET", f"/api/jobs/{job['id']}")[1]
    assert final["status"] == "done" and len(final["concept"]["textos"]) == 3 and final["concept"]["rondas"][0]["decision"] is None


def test_adjusting_reuses_the_previous_concept(server, agent):
    drain(server)
    primero = pedido(server)
    segundo = pedido(server, "más oscuro", anterior=primero["id"])
    c = segundo["concept"]
    assert c["original"] == "algo oscuro tipo after / más oscuro" and c["entendi"] == "ajuste: más oscuro"
    assert app(server, "POST", "/api/conceptos", {"texto": "x y", "anterior": "nope"})[0] == 400
    drain(server)


def test_invalid_search_report_is_rejected_and_the_job_stays_open(server):
    drain(server)
    job = pedido(server)
    claimed = device(server, "/api/device/claim", {"job_id": job["id"]})[1]["job"]
    malo = {"busquedas": {"A": {"generos": {}, "bpm": None, "voz": 2, "suena_a": [], "temas": []}}}
    assert device(server, "/api/device/complete", {"id": job["id"], "attempts": claimed["attempts"], "status": "done", "result": malo})[0] == 400
    assert app(server, "GET", f"/api/jobs/{job['id']}")[1]["status"] == "processing"
    drain(server)


def test_a_concept_saved_before_the_circuit_still_becomes_a_playlist(server, agent):
    drain(server)
    old = {"version": 1, "name": "Techno viejo", "description": "Concepto guardado con la versión anterior.",
           "positive_prompts": ["heavy industrial techno"], "negative_prompts": [], "constraints": {"energy": "unspecified"}}
    job_id = "4810286b-c784-4634-b8df-b5c71342a2d4"
    now = int(time.time() * 1000)
    sql = f"INSERT INTO jobs (id, created_at, updated_at, status, concept_json) VALUES ('{job_id}', {now}, {now}, 'queued', '{json.dumps(old)}')"
    subprocess.run(server["d1"] + ["--command", sql], check=True, cwd=server["cwd"], env=server["env"], capture_output=True, timeout=120)
    correr(agent(server), job_id)
    final = app(server, "GET", f"/api/jobs/{job_id}")[1]
    assert final["status"] == "done" and final["result"]["playlist"].endswith("Techno viejo")


# ---------------------------------------------------------------- la cola

def test_claim_with_job_id_leaves_older_jobs_queued(server):
    drain(server)
    older, newer = pedido(server)["id"], pedido(server)["id"]
    assert device(server, "/api/device/claim", {"job_id": newer})[1]["job"]["id"] == newer
    assert app(server, "GET", f"/api/jobs/{older}")[1]["status"] == "queued"
    assert device(server, "/api/device/claim", {"job_id": newer})[1]["job"] is None
    assert device(server, "/api/device/claim", {"job_id": "x"})[0] == 400
    drain(server)


def test_claim_is_fifo_and_empty_body_is_accepted(server):
    drain(server)
    first = pedido(server)["id"]
    time.sleep(0.01)
    second = pedido(server)["id"]
    assert call(server["url"], "POST", "/api/device/claim", {"X-Device-Token": DEVICE})[1]["job"]["id"] == first
    assert device(server, "/api/device/claim")[1]["job"]["id"] == second
    drain(server)


def test_after_a_search_round_the_claim_count_starts_again(server, agent):
    drain(server)
    job = pedido(server)
    ag = agent(server)
    claimed = device(server, "/api/device/claim", {"job_id": job["id"]})[1]["job"]
    ag.atender(claimed)
    again = device(server, "/api/device/claim", {"job_id": job["id"]})[1]["job"]
    assert again["attempts"] == 1 and again["concept"]["fase"] == "final"
    drain(server)


def test_abandoned_job_returns_to_the_queue_and_a_late_complete_is_refused(server):
    drain(server)
    job_id = pedido(server)["id"]
    assert device(server, "/api/device/claim", {"job_id": job_id})[1]["job"]["attempts"] == 1
    time.sleep(LEASE_MS / 1000 + 0.4)
    assert device(server, "/api/device/claim", {"job_id": job_id})[1]["job"]["attempts"] == 2
    assert device(server, "/api/device/complete", {"id": job_id, "attempts": 1, "status": "failed", "error": "tarde"})[0] == 409
    assert device(server, "/api/device/complete", {"id": job_id, "attempts": 2, "status": "failed", "error": "fin"})[0] == 200


def test_a_processing_job_within_its_lease_is_not_taken_again(server):
    drain(server)
    job_id = pedido(server)["id"]
    assert device(server, "/api/device/claim", {"job_id": job_id})[1]["job"]["attempts"] == 1
    assert device(server, "/api/device/claim")[1]["job"] is None
    device(server, "/api/device/complete", {"id": job_id, "attempts": 1, "status": "failed", "error": "fin"})


def test_job_fails_after_max_attempts(server):
    drain(server)
    job_id = pedido(server)["id"]
    for attempt in (1, 2, 3):
        assert device(server, "/api/device/claim", {"job_id": job_id})[1]["job"]["attempts"] == attempt
        time.sleep(LEASE_MS / 1000 + 0.4)
    assert device(server, "/api/device/claim", {"job_id": job_id})[1]["job"] is None
    job = app(server, "GET", f"/api/jobs/{job_id}")[1]
    assert job["status"] == "failed" and "no terminó" in job["error"]


@pytest.mark.parametrize("bad", [
    {"tracks": [{"track_id": "a", "title": "T", "artist": "", "score": 1.5}]},
    {"tracks": [{"track_id": "a", "title": "T", "artist": "", "score": 0.5, "bpm": -3}]},
    {"tracks": [], "playlist": "x" * 121},
    {"tracks": [{"track_id": "a", "title": "T", "artist": "", "score": 0.5}] * 31},
])
def test_invalid_final_results_are_rejected(server, agent, bad):
    drain(server)
    job = pedido(server)
    ag = agent(server)
    ag.atender(device(server, "/api/device/claim", {"job_id": job["id"]})[1]["job"])     # sondeo
    claimed = device(server, "/api/device/claim", {"job_id": job["id"]})[1]["job"]      # final
    assert device(server, "/api/device/complete", {"id": job["id"], "attempts": claimed["attempts"], "status": "done", "result": bad})[0] == 400
    device(server, "/api/device/complete", {"id": job["id"], "attempts": claimed["attempts"], "status": "failed", "error": "fin"})


def test_completing_a_finished_job_twice_is_refused(server):
    drain(server)
    job_id = pedido(server)["id"]
    claimed = device(server, "/api/device/claim", {"job_id": job_id})[1]["job"]
    body = {"id": job_id, "attempts": claimed["attempts"], "status": "failed", "error": "fin"}
    assert device(server, "/api/device/complete", body)[0] == 200
    assert device(server, "/api/device/complete", body)[0] == 409


# ---------------------------------------------------------------- v6: presets, estado y explicación

def llm_calls(s):
    with urllib.request.urlopen(s["url"] + "/__llm_calls", timeout=10) as r:
        return json.loads(r.read())["calls"]


def catalogo_de_prueba(s):
    device(s, "/api/device/claim", {"catalogo": {"version": 1, "temas": 12, "sin_genero": 0,
                                                 "generos": [{"nombre": "Techno", "temas": 6}, {"nombre": "House", "temas": 6}]}})


def test_page_asks_the_password_once_and_shows_what_is_playing(server):
    with urllib.request.urlopen(server["url"] + "/", timeout=10) as r:
        html = r.read().decode("utf-8")
    assert "localStorage" in html and 'id="now"' in html and "/api/estado" in html and "Buscar en mi música" in html


def test_state_lists_the_presets(server):
    assert call(server["url"], "GET", "/api/estado")[0] == 401
    status, e = app(server, "GET", "/api/estado")
    assert status == 200 and {p["id"] for p in e["presets"]} >= {"sonando", "mas_rapido", "mas_lento", "sin_voces", "pico"}
    assert all(p["pendiente"] for p in e["presets"] if p["grupo"] == "momento")


def test_the_computer_reports_what_is_playing_and_the_catalog(server):
    drain(server)
    sonando = {"titulo": "Beyond — Ñandú", "artista": "Ben Böhmer", "bpm": 122.5, "clave": "8A", "deck": "B"}
    catalogo = {"version": 1, "temas": 12, "sin_genero": 0, "generos": [{"nombre": "Techno", "temas": 6}, {"nombre": "House", "temas": 6}],
                "bpm": {"min": 120.0, "max": 131.0}, "voz": {"con": 4, "sin": 8}, "campos": {"genero": 12}}
    assert device(server, "/api/device/claim", {"sonando": sonando, "catalogo": catalogo})[1]["job"] is None
    e = app(server, "GET", "/api/estado")[1]
    assert e["sonando"] == sonando and e["ordenador_conectado"] and e["ordenador_hace_s"] <= 5
    assert e["generos_preset"] == ["Techno", "House"] and e["catalogo"]["temas"] == 12
    # lo que no valida se ignora y el reclamo sigue andando
    assert device(server, "/api/device/claim", {"sonando": {"titulo": 3}, "catalogo": {"version": 9}})[0] == 200
    assert app(server, "GET", "/api/estado")[1]["sonando"] == sonando
    assert device(server, "/api/device/claim", {"sonando": None})[0] == 200
    e = app(server, "GET", "/api/estado")[1]
    assert e["sonando"] is None and e["ordenador_conectado"]


def test_presets_skip_the_llm_and_become_a_list(server, agent):
    drain(server)
    catalogo_de_prueba(server)
    antes = llm_calls(server)
    status, job = app(server, "POST", "/api/conceptos", {"presets": ["sin_voces"], "generos": ["House"]})
    assert status == 201 and llm_calls(server) == antes                         # ni una neurona
    c = job["concept"]
    assert c["version"] == 3 and c["fase"] == "final" and c["textos"] == [] and c["entendi"] == "Sin voces · House"
    assert c["formulario"]["filtros"]["voz"] == "sin" and c["formulario"]["filtros"]["generos"] == {"incluir": ["House"], "excluir": [], "modo": "duro"}
    correr(agent(server), job["id"])
    final = app(server, "GET", f"/api/jobs/{job['id']}")[1]
    r = final["result"]
    assert final["status"] == "done" and [p["quedan"] for p in r["pasos"]] == [12, 6, 4] and len(r["tracks"]) == 4
    assert r["aplicado"] == ["House", "Sin voces"] and r["playlist"].endswith("Sin voces, House")


def test_presets_and_text_together_keep_the_presets_through_the_llm(server, agent):
    drain(server)
    status, job = app(server, "POST", "/api/conceptos", {"texto": "algo oscuro tipo after", "presets": ["sin_voces"]})
    assert status == 201 and job["concept"]["fase"] == "sondeo" and job["concept"]["formulario"]["filtros"]["voz"] == "sin"
    correr(agent(server), job["id"])
    final = app(server, "GET", f"/api/jobs/{job['id']}")[1]
    assert final["status"] == "done" and final["concept"]["formulario"]["filtros"]["voz"] == "sin"
    assert final["result"]["pasos"][-1]["paso"] == "Sin voces" and len(final["result"]["tracks"]) == 8


def test_bad_presets_are_refused(server):
    catalogo_de_prueba(server)
    assert app(server, "POST", "/api/conceptos", {"presets": ["volar"]})[0] == 400
    assert app(server, "POST", "/api/conceptos", {"presets": "sin_voces"})[0] == 400
    assert app(server, "POST", "/api/conceptos", {"generos": ["Bossa Nova"]})[0] == 400       # no está en el catálogo
    assert app(server, "POST", "/api/conceptos", {"presets": [], "generos": []})[0] == 400
    assert app(server, "POST", "/api/conceptos", {"referencia_uid": "../x"})[0] == 400


def test_results_with_steps_are_kept_and_bad_ones_refused(server):
    drain(server)
    job = app(server, "POST", "/api/conceptos", {"presets": ["sin_voces"]})[1]
    claimed = device(server, "/api/device/claim", {"job_id": job["id"]})[1]["job"]
    malo = {"tracks": [], "pasos": [{"paso": "x" * 121, "quedan": 1}]}
    assert device(server, "/api/device/complete", {"id": job["id"], "attempts": claimed["attempts"], "status": "done", "result": malo})[0] == 400
    bueno = {"tracks": [], "pasos": [{"paso": "Biblioteca", "quedan": 9}, {"paso": "Sin voces", "quedan": 0}],
             "aplicado": ["Sin voces"], "ignorado": [{"casilla": "momento", "motivo": "todavía no"}],
             "referencia": {"titulo": "Uno", "artista": ""}, "candidatos": [{"uid": "abc", "titulo": "Dos", "artista": "X"}]}
    assert device(server, "/api/device/complete", {"id": job["id"], "attempts": claimed["attempts"], "status": "done", "result": bueno})[0] == 200
    assert app(server, "GET", f"/api/jobs/{job['id']}")[1]["result"]["candidatos"][0]["uid"] == "abc"


def test_choosing_a_candidate_sends_its_uid(server):
    drain(server)
    status, job = app(server, "POST", "/api/conceptos", {"referencia_uid": "u3", "referencia_titulo": "Tema 3"})
    assert status == 201 and job["concept"]["formulario"]["referencia"] == {"tipo": "tema", "uid": "u3"}
    assert job["concept"]["entendi"] == "Como «Tema 3»"
    drain(server)
