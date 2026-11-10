"""
PURPOSE: Tests del agente local de conceptos (tools/traktor_live/conceptos_agente.py) sin red real, sin Cloudflare y
         sin CLAP: un servidor HTTP de prueba con el contrato del Worker y un codificador falso. Cubren el orden por
         parecido, cada condición (y las notas de lo que no se aplica), el registro de la sesión, la playlist de
         Traktor, los códigos del Worker y que a Cloudflare no vayan rutas ni el token.
CHANGELOG:
  - 2026-11-09: Creación inicial.
"""
import json
import math
import os
import sys
import threading
import time
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.traktor_live import conceptos_agente as ca  # noqa: E402
from tools.traktor_live import recomendador as rc  # noqa: E402


def unit(*values):
    v = np.array(values, dtype=np.float32)
    return (v / np.linalg.norm(v)).tolist()


def item(uid, vec, bpm=125.0, key="8A", title=None, artist="Artista"):
    return rc.Item(uid=uid, rel=f"Musica\\{uid}.mp3", original=f"orig\\{uid}.mp3", artist=artist,
                   title=title or f"Tema {uid}", bpm=bpm, key=key, vec=vec)


def biblioteca(items, vocal=None, version=None):
    return ca.Biblioteca(items=items, matrix=np.array([it.vec for it in items], dtype=np.float32),
                         vocal=vocal or {}, recommender=rc.Recommender(items, {}, 30), nml_version=version)


ITEMS = [item("a", unit(1, 0, 0)), item("b", unit(0.8, 0.6, 0)), item("c", unit(0, 1, 0)),
         item("d", unit(-1, 0, 0), bpm=170.0)]
VOCAL = {"a": True, "b": False, "c": False, "d": True}


def concepto(**kw):
    return ca.Concepto(name="prueba", positive=("una frase larga",), **kw)


def codificar(_frases):
    return np.array([[1.0, 0.0, 0.0]])


# ---------------------------------------------------------------- ranking y condiciones

def test_ranking_by_cosine_with_raw_scores_in_range():
    result, chosen = ca.resolver(concepto(), biblioteca(ITEMS), codificar, ca.Sesion())
    assert [t["track_id"] for t in result["tracks"]] == ["a", "b", "c", "d"]
    assert result["tracks"][0]["score"] == pytest.approx(1.0)
    assert result["tracks"][-1]["score"] == pytest.approx(-1.0)
    assert all(-1 <= t["score"] <= 1 for t in result["tracks"])
    assert (result["pool"], result["passed"]) == (4, 4)
    assert [it.uid for it in chosen] == ["a", "b", "c", "d"]


def test_query_is_the_mean_of_the_normalized_phrases():
    bib = biblioteca(ITEMS)
    two = lambda _: np.array([unit(1, 0, 0), unit(0, 1, 0)])      # noqa: E731
    result, _ = ca.resolver(concepto(), bib, two, ca.Sesion())
    assert result["tracks"][0]["track_id"] == "b"                  # el del medio entre las dos frases
    assert result["tracks"][0]["score"] == pytest.approx((0.8 + 0.6) / math.sqrt(2), abs=1e-4)


def test_vocals_filter_uses_the_vocal_tag_and_says_so():
    result, _ = ca.resolver(concepto(vocals="absent"), biblioteca(ITEMS, VOCAL), codificar, ca.Sesion())
    assert [t["track_id"] for t in result["tracks"]] == ["b", "c"]
    assert result["passed"] == 2 and any("Sin voces" in n and "2" in n for n in result["notes"])
    assert all(t["vocal"] is False for t in result["tracks"])
    result, _ = ca.resolver(concepto(vocals="present"), biblioteca(ITEMS, VOCAL), codificar, ca.Sesion())
    assert [t["track_id"] for t in result["tracks"]] == ["a", "d"]


def test_vocals_filter_without_the_tag_is_not_applied_and_says_so():
    result, _ = ca.resolver(concepto(vocals="absent"), biblioteca(ITEMS), codificar, ca.Sesion())
    assert result["passed"] == 4 and any("sin aplicar" in n for n in result["notes"])


def test_track_missing_from_the_vocal_tag_does_not_pass_either_filter():
    result, _ = ca.resolver(concepto(vocals="absent"), biblioteca(ITEMS, {"b": False}), codificar, ca.Sesion())
    assert [t["track_id"] for t in result["tracks"]] == ["b"]


def test_bpm_near_current_uses_the_pitched_bpm_of_the_main_track():
    sesion = ca.Sesion(principal={"path": "X:/otra/cosa.mp3", "bpm": 150.0, "tempo": 1.1})   # 165
    result, _ = ca.resolver(concepto(bpm="near_current"), biblioteca(ITEMS), codificar, sesion)
    assert [t["track_id"] for t in result["tracks"]] == ["d"]
    assert any("165" in n for n in result["notes"])


def test_bpm_near_current_without_a_playing_track_is_not_applied_and_says_so():
    result, _ = ca.resolver(concepto(bpm="near_current"), biblioteca(ITEMS), codificar, ca.Sesion())
    assert result["passed"] == 4 and any("no hay un tema sonando" in n for n in result["notes"])


def test_main_and_played_tracks_are_left_out():
    sesion = ca.Sesion(principal={"path": "C:\\TRAKTOR ML\\Musica\\a.mp3"}, tocados=["D:\\x\\orig\\b.mp3"])
    result, _ = ca.resolver(concepto(), biblioteca(ITEMS), codificar, sesion)
    assert [t["track_id"] for t in result["tracks"]] == ["c", "d"]
    assert any("2 ya tocados" in n for n in result["notes"])


def test_energy_and_negatives_are_reported_as_not_applied_without_changing_the_list():
    plain, _ = ca.resolver(concepto(), biblioteca(ITEMS), codificar, ca.Sesion())
    result, _ = ca.resolver(concepto(energy="increase", negative=("distorted screaming guitars",)),
                            biblioteca(ITEMS), codificar, ca.Sesion())
    assert result["tracks"] == plain["tracks"]
    assert any("Energía (subir): sin aplicar" in n for n in result["notes"])
    assert any("Exclusiones sin aplicar" in n and "guitars" in n for n in result["notes"])


def test_voice_negative_is_covered_by_the_vocals_filter():
    result, _ = ca.resolver(concepto(vocals="absent", negative=("singing vocals",)), biblioteca(ITEMS, VOCAL),
                            codificar, ca.Sesion())
    assert not any("Exclusiones" in n for n in result["notes"])


def test_no_track_passes():
    result, chosen = ca.resolver(concepto(bpm="near_current"), biblioteca(ITEMS), codificar,
                                 ca.Sesion(principal={"path": "x", "bpm": 60.0, "tempo": 1.0}))
    assert result["tracks"] == [] and result["passed"] == 0 and chosen == []


def test_result_limits_and_title_fallback():
    long = item("e", unit(1, 0, 0), bpm=None, key="?", title="x" * 400, artist="y" * 400)
    nameless = item("f", unit(1, 0, 0), title="", artist="")
    nameless.title = ""
    result, _ = ca.resolver(concepto(), biblioteca([long, nameless]), codificar, ca.Sesion(), n=30)
    by_id = {t["track_id"]: t for t in result["tracks"]}
    assert len(by_id["e"]["title"]) == 160 and len(by_id["e"]["artist"]) == 120
    assert "bpm" not in by_id["e"] and "key" not in by_id["e"]
    assert by_id["f"]["title"] == "f"


def test_n_tracks_is_capped():
    items = [item(f"t{i}", unit(1, i / 100, 0)) for i in range(40)]
    result, _ = ca.resolver(concepto(), biblioteca(items), codificar, ca.Sesion())
    assert len(result["tracks"]) == 30 and result["passed"] == 40


@pytest.mark.parametrize("raw", [None, [], {"version": 2, "positive_prompts": ["hola mundo"]},
                                 {"version": 1, "positive_prompts": []}, {"version": 1},
                                 {"version": 1, "positive_prompts": ["ok ok ok"], "negative_prompts": "x"}])
def test_parse_rejects_unusable_concepts(raw):
    with pytest.raises(ca.ConceptoInvalido):
        ca.parse_concepto(raw)


def test_parse_reads_the_worker_contract():
    c = ca.parse_concepto({"version": 1, "name": "N", "positive_prompts": [" uno dos tres "], "negative_prompts": ["x y z w"],
                           "constraints": {"energy": "decrease", "vocals": "absent", "bpm": "near_current"}})
    assert c == ca.Concepto("N", ("uno dos tres",), ("x y z w",), "decrease", "absent", "near_current")


# ---------------------------------------------------------------- sesión

def test_session_reads_last_main_track_and_all_played(tmp_path):
    (tmp_path / "sugerencias_vivo_20260101_000000.jsonl").write_text("{}\n", encoding="utf-8")
    log = tmp_path / "sugerencias_vivo_20260102_000000.jsonl"
    lines = [
        {"t": "1", "kind": "principal", "target": "A", "data": {"path": "uno.mp3", "bpm": 120}},
        {"t": "2", "kind": "fader", "target": "A", "data": {"canal": 1}},
        {"t": "3", "kind": "tocado", "target": "A", "data": {"path": "uno.mp3"}},
        {"t": "4", "kind": "principal", "target": "B", "data": {"path": "dos.mp3", "bpm": 124}},
    ]
    log.write_text("\n".join(json.dumps(x) for x in lines) + '\n{"t": "5", "kind": "princ', encoding="utf-8")
    now = log.stat().st_mtime
    os.utime(tmp_path / "sugerencias_vivo_20260101_000000.jsonl", (now - 1000, now - 1000))
    s = ca.leer_sesion(tmp_path, running=lambda: True, now=lambda: now + 5)
    assert s.principal["path"] == "dos.mp3" and s.tocados == ["uno.mp3"]


def test_session_is_empty_when_traktor_is_closed_or_the_log_is_old(tmp_path):
    log = tmp_path / "sugerencias_vivo_20260102_000000.jsonl"
    log.write_text(json.dumps({"t": "1", "kind": "principal", "target": "A", "data": {"path": "x"}}) + "\n", encoding="utf-8")
    assert ca.leer_sesion(tmp_path, running=lambda: False).principal is None
    now = log.stat().st_mtime
    assert ca.leer_sesion(tmp_path, running=lambda: True, now=lambda: now + ca.SESSION_FRESH_S + 1).principal is None
    assert ca.leer_sesion(tmp_path / "vacio", running=lambda: True).principal is None


# ---------------------------------------------------------------- playlist de Traktor

def entry(artist, title, directory, filename):
    e = ET.Element("ENTRY", TITLE=title, ARTIST=artist)
    ET.SubElement(e, "LOCATION", DIR=directory, FILE=filename, VOLUME="C:", VOLUMEID="C:")
    return e


def test_nml_lists_the_entries_in_order_and_rewrites_itself(tmp_path):
    a = item("a", unit(1, 0, 0)); a.entry = entry("A", "Uno", "/:Musica/:", "uno.mp3")
    b = item("b", unit(0, 1, 0)); b.entry = entry("B", "Dos", "/:Musica/:", "dos.mp3")
    bib = biblioteca([a, b], version="20")
    out = ca.escribir_nml(bib, [b, a], tmp_path)
    assert out.name == "TRAKTOR ML Concepto NML.nml"
    root = ET.parse(out).getroot()
    assert [e.get("TITLE") for e in root.find("COLLECTION")] == ["Dos", "Uno"]
    keys = [k.get("KEY") for k in root.iter("PRIMARYKEY")]
    assert keys[0].endswith("dos.mp3") and keys[1].endswith("uno.mp3")
    assert root.find(".//NODE[@TYPE='PLAYLIST']").get("NAME") == "TRAKTOR ML Concepto NML"
    assert ca.escribir_nml(bib, [a], tmp_path) == out                # la segunda lista pisa la primera
    assert [e.get("TITLE") for e in ET.parse(out).getroot().find("COLLECTION")] == ["Uno"]


def test_nml_is_skipped_without_collection_entries(tmp_path):
    assert ca.escribir_nml(biblioteca(ITEMS, version=None), ITEMS, tmp_path) is None
    assert ca.escribir_nml(biblioteca(ITEMS, version="20"), ITEMS, tmp_path) is None


# ---------------------------------------------------------------- cliente HTTP

class FakeWorker:
    """El contrato de las rutas del ordenador, en memoria."""

    def __init__(self):
        self.requests, self.jobs, self.reply = [], [], None
        worker = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                worker.requests.append((self.path, dict(self.headers), body))
                if worker.reply is not None:
                    code, payload = worker.reply
                elif self.headers.get("X-Device-Token") != "secreto":
                    code, payload = 401, {"error": "No autorizado"}
                elif self.path == "/api/device/claim":
                    code, payload = 200, {"job": worker.jobs.pop(0) if worker.jobs else None}
                else:
                    code, payload = 200, {"id": body.get("id"), "status": body.get("status")}
                data = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def bodies(self, path):
        return [r[2] for r in self.requests if r[0] == path]


@pytest.fixture
def worker():
    w = FakeWorker()
    yield w
    w.close()


def test_client_sends_token_and_user_agent(worker):
    client = ca.WorkerClient(worker.url, "secreto")
    worker.jobs.append({"id": "x", "attempts": 1, "concept": {}})
    assert client.claim()["id"] == "x"
    assert client.claim() is None
    path, headers, body = worker.requests[0]
    assert path == "/api/device/claim" and body == {}
    assert headers["X-Device-Token"] == "secreto" and "Python-urllib" not in headers["User-Agent"]


def test_accented_titles_travel_as_utf8_not_as_escapes(worker):
    ca.WorkerClient(worker.url, "secreto").complete("x", 1, "done", result={"tracks": [{"title": "Ñandú — Böhmer"}]})
    assert worker.bodies("/api/device/complete")[0]["result"]["tracks"][0]["title"] == "Ñandú — Böhmer"
    assert ca.json.dumps("Ñ", ensure_ascii=False).encode("utf-8") == '"Ñ"'.encode("utf-8")


def test_client_claims_only_the_requested_job(worker):
    ca.WorkerClient(worker.url, "secreto").claim("abc")
    assert worker.bodies("/api/device/claim") == [{"job_id": "abc"}]


def test_client_maps_worker_codes(worker):
    bad = ca.WorkerClient(worker.url, "mal")
    with pytest.raises(ca.NoAutorizado):
        bad.claim()
    client = ca.WorkerClient(worker.url, "secreto")
    worker.reply = (409, {"error": "ya no"})
    assert client.complete("x", 1, "failed", error="e") is False
    worker.reply = (400, {"error": "Resultados inválidos"})
    with pytest.raises(ca.ResultadoRechazado):
        client.complete("x", 1, "done", result={"tracks": []})
    worker.reply = (500, {"error": "boom"})
    with pytest.raises(OSError):
        client.complete("x", 1, "done", result={"tracks": []})
    with pytest.raises(OSError):
        client.claim()


# ---------------------------------------------------------------- agente y bucle

CONCEPT = {"version": 1, "name": "Prueba", "positive_prompts": ["una frase larga"], "negative_prompts": [],
           "constraints": {"energy": "unspecified", "vocals": "unspecified", "bpm": "unspecified"}}
JOB = {"id": "1234", "attempts": 2, "concept": CONCEPT}


def agente(client, tmp_path, said=None, **kw):
    said = [] if said is None else said
    defaults = dict(cargar=lambda: biblioteca(ITEMS), codificar=codificar, sesion=ca.Sesion, nml=lambda b, c: tmp_path / "x.nml",
                    results_log=tmp_path / "res.jsonl", sleep=lambda s: None)
    defaults.update(kw)
    return ca.Agente(client, lambda lvl, txt: said.append((lvl, txt)), **defaults)


def test_job_completes_done_with_the_claim_generation(worker, tmp_path):
    ag = agente(ca.WorkerClient(worker.url, "secreto"), tmp_path)
    ag.atender(dict(JOB))
    body = worker.bodies("/api/device/complete")[0]
    assert body["id"] == "1234" and body["attempts"] == 2 and body["status"] == "done"
    assert [t["track_id"] for t in body["result"]["tracks"]] == ["a", "b", "c", "d"]
    logged = json.loads((tmp_path / "res.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert logged["job"] == "1234" and logged["uids"][0] == "a"


def test_nothing_local_goes_to_cloudflare(worker, tmp_path):
    """Por el contrato, al Worker solo van título, artista, parecido, BPM, tonalidad y Vocal."""
    it = item("a", unit(1, 0, 0))
    it.path = Path("C:/TRAKTOR ML/Musica/secreto.mp3")
    ag = agente(ca.WorkerClient(worker.url, "secreto"), tmp_path, cargar=lambda: biblioteca([it], {"a": True}))
    ag.atender(dict(JOB))
    body = worker.bodies("/api/device/complete")[0]
    text = json.dumps(body)
    assert "secreto.mp3" not in text and "Musica" not in text and "orig" not in text
    assert set(body["result"]["tracks"][0]) <= {"track_id", "title", "artist", "score", "bpm", "key", "vocal"}


def test_invalid_concept_fails_with_a_public_message(worker, tmp_path):
    ag = agente(ca.WorkerClient(worker.url, "secreto"), tmp_path)
    ag.atender({"id": "1", "attempts": 1, "concept": {"version": 9}})
    body = worker.bodies("/api/device/complete")[0]
    assert body["status"] == "failed" and "formato" in body["error"]


def test_internal_error_fails_without_leaking_details(worker, tmp_path, capsys):
    def boom(_):
        raise RuntimeError(r"C:\ruta\privada\archivo.mp3 no se pudo")
    ag = agente(ca.WorkerClient(worker.url, "secreto"), tmp_path, codificar=boom)
    ag.atender(dict(JOB))
    body = worker.bodies("/api/device/complete")[0]
    assert body["status"] == "failed" and "privada" not in body["error"] and "registro" in body["error"]
    assert "privada" in capsys.readouterr().err                   # el detalle queda en el registro local


def test_missing_index_tells_the_phone(worker, tmp_path):
    def missing():
        raise ca.BibliotecaNoDisponible("El ordenador no tiene el índice de la biblioteca.")
    ag = agente(ca.WorkerClient(worker.url, "secreto"), tmp_path, cargar=missing)
    ag.atender(dict(JOB))
    assert worker.bodies("/api/device/complete")[0]["status"] == "failed"
    assert "índice" in worker.bodies("/api/device/complete")[0]["error"]


def test_playlist_failure_still_returns_the_list(worker, tmp_path):
    def no_nml(_b, _c):
        raise FileExistsError("existe y no lo escribió este script")
    ag = agente(ca.WorkerClient(worker.url, "secreto"), tmp_path, nml=no_nml)
    ag.atender(dict(JOB))
    body = worker.bodies("/api/device/complete")[0]
    assert body["status"] == "done" and any("playlist" in n for n in body["result"]["notes"])


def test_no_collection_says_no_playlist(worker, tmp_path):
    ag = agente(ca.WorkerClient(worker.url, "secreto"), tmp_path, nml=lambda b, c: None)
    ag.atender(dict(JOB))
    assert any("colección" in n for n in worker.bodies("/api/device/complete")[0]["result"]["notes"])


def test_rejected_result_becomes_a_failed_job(worker, tmp_path):
    class Client(ca.WorkerClient):
        calls = []

        def complete(self, job_id, attempts, status, result=None, error=None):
            self.calls.append(status)
            if status == "done":
                raise ca.ResultadoRechazado("Resultados inválidos")
            return True

    client = Client(worker.url, "secreto")
    agente(client, tmp_path).atender(dict(JOB))
    assert client.calls == ["done", "failed"]


def test_complete_retries_transient_errors_then_gives_up_to_the_lease(worker, tmp_path):
    class Flaky(ca.WorkerClient):
        tries = 0

        def complete(self, *a, **k):
            type(self).tries += 1
            raise OSError("sin red")

    said = []
    agente(Flaky(worker.url, "secreto"), tmp_path, said).atender(dict(JOB))
    assert Flaky.tries == 5 and any("lease" in t for _, t in said)


def test_loop_handles_a_job_and_stops_with_una_vez(worker, tmp_path):
    worker.jobs.append(dict(JOB))
    ag = agente(ca.WorkerClient(worker.url, "secreto"), tmp_path)
    assert ca.bucle(ag, threading.Event(), poll_s=0.01, una_vez=True) == 0
    assert len(worker.bodies("/api/device/complete")) == 1


def test_loop_exits_on_rejected_token(worker, tmp_path):
    said = []
    ag = agente(ca.WorkerClient(worker.url, "mal"), tmp_path, said)
    assert ca.bucle(ag, threading.Event(), poll_s=0.01) == 2
    assert any(lvl == "error" for lvl, _ in said) and not any("mal" in t.split() for _, t in said)


def test_loop_frees_memory_only_after_clap_was_loaded():
    class Client:
        def claim(self, _job):
            return None

    stop = threading.Event()
    clock = iter(range(0, 100000, 1000))
    ag = ca.Agente(Client(), lambda *a: None, results_log=None, sleep=lambda s: None)
    ag.bib = None
    timer = threading.Timer(0.3, stop.set)
    timer.start()
    assert ca.bucle(ag, stop, poll_s=0.01, idle_exit_s=10.0, clock=lambda: next(clock)) == 0   # liviano: sigue hasta stop
    timer.cancel()
    assert stop.is_set()

    stop2 = threading.Event()
    clock2 = iter(range(0, 100000, 1000))
    ag.bib = object()                                              # CLAP cargado
    t0 = time.monotonic()
    assert ca.bucle(ag, stop2, poll_s=0.01, idle_exit_s=10.0, clock=lambda: next(clock2)) == 0
    assert not stop2.is_set() and time.monotonic() - t0 < 1


def test_loop_backs_off_when_the_network_is_down_and_survives():
    class Client:
        calls = 0

        def claim(self, _job):
            type(self).calls += 1
            if type(self).calls < 5:
                raise OSError("sin red")
            return None

    stop = threading.Event()
    said = []
    ag = ca.Agente(Client(), lambda lvl, txt: said.append((lvl, txt)), results_log=None, sleep=lambda s: None)
    timer = threading.Timer(1.2, stop.set)
    timer.start()
    assert ca.bucle(ag, stop, poll_s=0.01) == 0
    timer.cancel()
    assert Client.calls >= 5
    assert [lvl for lvl, _ in said] == ["aviso", "ok"]


# ---------------------------------------------------------------- ajustes y servicio

def test_settings_from_file_and_environment(tmp_path):
    path = tmp_path / "c.json"
    assert ca.load_ajustes(path, env={}) is None
    path.write_text(json.dumps({"device_token": "t", "url": "https://x.dev/"}), encoding="utf-8")
    assert ca.load_ajustes(path, env={}) == ca.Ajustes("https://x.dev", "t")
    assert ca.load_ajustes(path, env={"TRAKTOR_CONCEPT_DEVICE_TOKEN": "e"}).token == "e"
    path.write_text(json.dumps({"device_token": "t"}), encoding="utf-8")
    assert ca.load_ajustes(path, env={}).url == ca.DEFAULT_URL


def test_child_session_runs_low_priority_and_ends_with_traktor():
    class Child:
        def __init__(self):
            self.terminated, self.code = False, None

        def poll(self):
            return self.code

        def terminate(self):
            self.terminated, self.code = True, 0

        def wait(self, timeout=None):
            return 0

    child = Child()
    seen = {}

    def popen(command, **kwargs):
        seen.update(kwargs, command=command)
        return child

    stop = threading.Event()
    thread = threading.Thread(target=lambda: seen.setdefault("code", ca.sesion_hija(None, stop, popen=popen)))
    thread.start()
    time.sleep(0.1)
    stop.set()
    thread.join(timeout=5)
    assert child.terminated and seen["code"] == 0
    assert seen["command"][2:4] == ["correr", "--salir-sin-uso"]
    if sys.platform == "win32":
        assert seen["creationflags"] & ca.BELOW_NORMAL_PRIORITY_CLASS


def test_child_session_returns_the_child_code():
    class Child:
        def poll(self):
            return 2

        def terminate(self):
            raise AssertionError("ya terminó")

    assert ca.sesion_hija(None, threading.Event(), popen=lambda *a, **k: Child()) == 2
