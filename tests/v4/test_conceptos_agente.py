"""
PURPOSE: Tests del ordenador en el circuito de conceptos (tools/traktor_live/conceptos_agente.py) sin red real, sin Cloudflare
         y sin CLAP: un servidor HTTP de prueba con el contrato del Worker y un codificador falso. Cubren el formato de los
         trabajos, buscar(), el informe de un sondeo, la lista final, la playlist nueva por concepto, los códigos del Worker
         y que a Cloudflare no vayan rutas ni el token.
CHANGELOG:
  - 2026-11-09: Creación inicial.
  - 2026-11-09: Circuito con el LLM: sondeo y final, playlist nueva por concepto, sin filtros.
"""
import datetime as dt
import json
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


def item(uid, vec, bpm=125.0, key="8A", title=None, artist="Artista", genre="Techno"):
    return rc.Item(uid=uid, rel=f"Musica\\{uid}.mp3", original=f"orig\\{uid}.mp3", artist=artist,
                   title=title or f"Tema {uid}", bpm=bpm, key=key, vec=vec, genre=genre)


def biblioteca(items, vocal=None, version=None):
    return ca.Biblioteca(items=items, matrix=np.array([it.vec for it in items], dtype=np.float32),
                         vocal=vocal or {}, nml_version=version)


ITEMS = [item("a", unit(1, 0, 0)), item("b", unit(0.8, 0.6, 0), genre="House"), item("c", unit(0, 1, 0), genre="House"),
         item("d", unit(-1, 0, 0), bpm=170.0)]
VOCAL = {"a": True, "b": False, "c": False, "d": True}
WORDS = {"trompeta": [1.0, 0.0, 0.0], "cuerdas": [0.0, 1.0, 0.0], "ruido": [-1.0, 0.0, 0.0]}


def codificar(frases):
    """Un codificador de juguete: conoce las palabras de WORDS; lo demás es el primer eje."""
    return np.array([unit(*WORDS.get(f, [1.0, 0.0, 0.0])) for f in frases])


def oido(bib):
    return ca.Oido(bib, codificar, vocabulario=list(WORDS))


def sondeo_job(textos=("trompeta", "cuerdas")):
    return {"version": 2, "fase": "sondeo", "ronda": 1, "nombre": "Fiesta", "textos": [{"id": c, "texto": t} for c, t in zip("ABC", textos)]}


def final_job(textos=("trompeta",)):
    return {"version": 2, "fase": "final", "nombre": "Fiesta", "textos": [{"id": c, "texto": t} for c, t in zip("ABC", textos)]}


# ---------------------------------------------------------------- formato de los trabajos

def test_parse_reads_sondeo_and_final():
    t = ca.parse_tarea(sondeo_job())
    assert (t.fase, t.nombre, t.textos) == ("sondeo", "Fiesta", (("A", "trompeta"), ("B", "cuerdas")))
    assert ca.parse_tarea(final_job()).fase == "final"


def test_parse_reads_concepts_saved_before_the_circuit_as_final():
    t = ca.parse_tarea({"version": 1, "name": "Viejo", "positive_prompts": [" una frase "], "negative_prompts": [], "constraints": {}})
    assert (t.fase, t.nombre, t.textos) == ("final", "Viejo", (("A", "una frase"),))


@pytest.mark.parametrize("raw", [None, [], {"version": 9}, {"version": 2, "fase": "raro", "textos": [{"id": "A", "texto": "x"}]},
                                 {"version": 2, "fase": "final", "textos": []}, {"version": 2, "fase": "final", "textos": [{"id": "A"}]},
                                 {"version": 1, "positive_prompts": []}])
def test_parse_rejects_unusable_jobs(raw):
    with pytest.raises(ca.TareaInvalida):
        ca.parse_tarea(raw)


# ---------------------------------------------------------------- buscar, sondeo y lista

def test_buscar_orders_by_cosine_and_is_the_only_ranking():
    indices, sims = ca.buscar(biblioteca(ITEMS), np.array(unit(1, 0, 0), dtype=np.float64), n=3)
    assert indices == [0, 1, 2] and sims[0] == pytest.approx(1.0)


def test_query_is_the_mean_of_the_phrases():
    bib = biblioteca(ITEMS)
    dos = lambda frases: np.array([unit(1, 0, 0), unit(0, 1, 0)])    # noqa: E731
    result, _ = ca.crear_lista(ca.Tarea("final", "x", (("A", "uno"), ("B", "dos"))), bib, dos)
    assert result["tracks"][0]["track_id"] == "b"                     # el que queda entre las dos frases


def test_oido_describes_what_is_near_the_group_and_not_the_library():
    bib = biblioteca(ITEMS)
    assert ca.Oido(bib, codificar, vocabulario=list(WORDS)).suena_a([0], k=1) == ["trompeta"]
    assert ca.Oido(bib, codificar, vocabulario=list(WORDS)).suena_a([3], k=1) == ["ruido"]


def test_vocabulary_excludes_words_that_do_not_describe_a_sound():
    bib = biblioteca(ITEMS)
    o = ca.Oido(bib, lambda f: np.array([unit(1, 0, 0)] * len(f)))
    assert "birthday" not in o.palabras and "brass instruments" in o.palabras and "breakdown" not in o.palabras


def test_sondeo_reports_each_text_separately():
    bib = biblioteca(ITEMS, VOCAL)
    r = ca.sondear(ca.parse_tarea(sondeo_job()), bib, codificar, oido(bib))
    assert set(r["busquedas"]) == {"A", "B"} and r["pool"] == 4
    a = r["busquedas"]["A"]
    assert a["generos"] == {"Techno": 2, "House": 2} and a["voz"] == 0.5
    assert a["temas"][0] == "Artista - Tema a" and len(a["temas"]) == 4
    assert a["bpm"][0] <= a["bpm"][1] <= a["bpm"][2]
    assert a["suena_a"] and all(isinstance(w, str) for w in a["suena_a"])
    assert "score" not in json.dumps(r)


def test_sondeo_without_tags_says_unknown():
    bib = biblioteca([item("x", unit(1, 0, 0), bpm=None, genre=None)])
    r = ca.sondear(ca.parse_tarea(sondeo_job(("trompeta",))), bib, codificar, oido(bib))
    b = r["busquedas"]["A"]
    assert b["bpm"] is None and b["voz"] is None and b["generos"] == {}


def test_final_list_order_and_track_fields():
    result, chosen = ca.crear_lista(ca.parse_tarea(final_job()), biblioteca(ITEMS, VOCAL), codificar)
    assert [t["track_id"] for t in result["tracks"]] == ["a", "b", "c", "d"] and [it.uid for it in chosen] == ["a", "b", "c", "d"]
    assert result["tracks"][0]["score"] == pytest.approx(1.0) and all(-1 <= t["score"] <= 1 for t in result["tracks"])
    assert set(result["tracks"][0]) == {"track_id", "title", "artist", "score", "bpm", "key", "vocal"}


def test_final_list_is_capped_and_titles_are_clipped():
    items = [item(f"t{i}", unit(1, i / 100, 0)) for i in range(40)]
    assert len(ca.crear_lista(ca.parse_tarea(final_job()), biblioteca(items), codificar)[0]["tracks"]) == 30
    largo = item("e", unit(1, 0, 0), bpm=None, key="?", title="x" * 400, artist="y" * 400)
    sin_nombre = item("f", unit(1, 0, 0), title="", artist="")
    sin_nombre.title = ""
    tracks = {t["track_id"]: t for t in ca.crear_lista(ca.parse_tarea(final_job()), biblioteca([largo, sin_nombre]), codificar)[0]["tracks"]}
    assert len(tracks["e"]["title"]) == 160 and len(tracks["e"]["artist"]) == 120 and "bpm" not in tracks["e"] and "key" not in tracks["e"]
    assert tracks["f"]["title"] == "f"


# ---------------------------------------------------------------- playlist de Traktor

def entry(artist, title, directory, filename):
    e = ET.Element("ENTRY", TITLE=title, ARTIST=artist)
    ET.SubElement(e, "LOCATION", DIR=directory, FILE=filename, VOLUME="C:", VOLUMEID="C:")
    return e


@pytest.mark.parametrize("nombre,esperado", [
    ("Fiesta con trompetas", "Concepto 1109-2315 Fiesta con trompetas"),
    ('Mal: nom/bre\\con*cosas?"<>|', "Concepto 1109-2315 Mal nom bre con cosas"),
    ("   ", "Concepto 1109-2315 Sin nombre"),
    ("x" * 300, "Concepto 1109-2315 " + "x" * 60),
])
def test_playlist_name_is_safe_for_a_windows_file(nombre, esperado):
    assert ca.nombre_playlist(nombre, dt.datetime(2026, 11, 9, 23, 15)) == esperado


def test_each_concept_gets_its_own_new_playlist(tmp_path):
    a = item("a", unit(1, 0, 0)); a.entry = entry("A", "Uno", "/:Musica/:", "uno.mp3")
    b = item("b", unit(0, 1, 0)); b.entry = entry("B", "Dos", "/:Musica/:", "dos.mp3")
    bib = biblioteca([a, b], version="20")
    primero = ca.escribir_nml(bib, [b, a], "Fiesta", tmp_path)
    segundo = ca.escribir_nml(bib, [a], "Oscuro", tmp_path)
    assert primero != segundo and len(list(tmp_path.glob("*.nml"))) == 2
    root = ET.parse(tmp_path / f"{primero}.nml").getroot()
    assert root.find(".//NODE[@TYPE='PLAYLIST']").get("NAME") == primero
    assert [e.get("TITLE") for e in root.find("COLLECTION")] == ["Dos", "Uno"]
    keys = [k.get("KEY") for k in root.iter("PRIMARYKEY")]
    assert keys[0].endswith("dos.mp3") and keys[1].endswith("uno.mp3")


def test_no_playlist_without_collection_entries(tmp_path):
    assert ca.escribir_nml(biblioteca(ITEMS, version=None), ITEMS, "x", tmp_path) is None
    assert ca.escribir_nml(biblioteca(ITEMS, version="20"), ITEMS, "x", tmp_path) is None


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


def test_accented_titles_travel_as_utf8(worker):
    ca.WorkerClient(worker.url, "secreto").complete("x", 1, "done", result={"tracks": [{"title": "Ñandú — Böhmer"}]})
    assert worker.bodies("/api/device/complete")[0]["result"]["tracks"][0]["title"] == "Ñandú — Böhmer"


def test_client_claims_only_the_requested_job(worker):
    ca.WorkerClient(worker.url, "secreto").claim("abc")
    assert worker.bodies("/api/device/claim") == [{"job_id": "abc"}]


def test_client_maps_worker_codes(worker):
    with pytest.raises(ca.NoAutorizado):
        ca.WorkerClient(worker.url, "mal").claim()
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

def job(concept=None, attempts=2):
    return {"id": "1234", "attempts": attempts, "concept": concept or final_job()}


def agente(client, tmp_path, said=None, **kw):
    said = [] if said is None else said
    bib = kw.pop("bib", None) or biblioteca(ITEMS)
    defaults = dict(cargar=lambda: bib, codificar=codificar, nml=lambda b, c, n: f"Concepto {n}",
                    results_log=tmp_path / "res.jsonl", sleep=lambda s: None)
    defaults.update(kw)
    ag = ca.Agente(client, lambda lvl, txt: said.append((lvl, txt)), **defaults)
    return ag


def patch_oido(monkeypatch):
    monkeypatch.setattr(ca, "VOCABULARIO", {"instrumentos": list(WORDS)})


def test_sondeo_job_reports_searches_with_the_claim_generation(worker, tmp_path, monkeypatch):
    patch_oido(monkeypatch)
    agente(ca.WorkerClient(worker.url, "secreto"), tmp_path).atender(job(sondeo_job()))
    body = worker.bodies("/api/device/complete")[0]
    assert body["attempts"] == 2 and body["status"] == "done" and set(body["result"]["busquedas"]) == {"A", "B"}


def test_final_job_writes_the_playlist_and_reports_its_name(worker, tmp_path, monkeypatch):
    patch_oido(monkeypatch)
    agente(ca.WorkerClient(worker.url, "secreto"), tmp_path).atender(job())
    body = worker.bodies("/api/device/complete")[0]
    assert body["result"]["playlist"] == "Concepto Fiesta" and [t["track_id"] for t in body["result"]["tracks"]] == ["a", "b", "c", "d"]
    logged = json.loads((tmp_path / "res.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert logged["fase"] == "final" and logged["job"] == "1234"


def test_nothing_local_goes_to_cloudflare(worker, tmp_path, monkeypatch):
    patch_oido(monkeypatch)
    it = item("a", unit(1, 0, 0))
    it.path = Path("C:/TRAKTOR ML/Musica/secreto.mp3")
    ag = agente(ca.WorkerClient(worker.url, "secreto"), tmp_path, bib=biblioteca([it], {"a": True}))
    ag.atender(job(sondeo_job(("trompeta",))))
    ag.atender(job())
    text = json.dumps([r[2] for r in worker.requests])
    assert "secreto.mp3" not in text and "Musica" not in text and "orig" not in text and worker.requests[0][1]["X-Device-Token"] not in text


def test_invalid_job_fails_with_a_public_message(worker, tmp_path, monkeypatch):
    patch_oido(monkeypatch)
    agente(ca.WorkerClient(worker.url, "secreto"), tmp_path).atender({"id": "1", "attempts": 1, "concept": {"version": 9}})
    body = worker.bodies("/api/device/complete")[0]
    assert body["status"] == "failed" and "formato" in body["error"]


def test_internal_error_fails_without_leaking_details(worker, tmp_path, capsys, monkeypatch):
    patch_oido(monkeypatch)
    def boom(_):
        raise RuntimeError(r"C:\ruta\privada\archivo.mp3 no se pudo")
    agente(ca.WorkerClient(worker.url, "secreto"), tmp_path, codificar=boom).atender(job())
    body = worker.bodies("/api/device/complete")[0]
    assert body["status"] == "failed" and "privada" not in body["error"] and "registro" in body["error"]
    assert "privada" in capsys.readouterr().err                   # el detalle queda en el registro local


def test_missing_index_tells_the_phone(worker, tmp_path):
    def missing():
        raise ca.BibliotecaNoDisponible("El ordenador no tiene el índice de la biblioteca.")
    agente(ca.WorkerClient(worker.url, "secreto"), tmp_path, cargar=missing).atender(job())
    body = worker.bodies("/api/device/complete")[0]
    assert body["status"] == "failed" and "índice" in body["error"]


def test_playlist_failure_still_returns_the_list(worker, tmp_path, monkeypatch):
    patch_oido(monkeypatch)
    def no_nml(_b, _c, _n):
        raise FileExistsError("existe y no lo escribió este script")
    agente(ca.WorkerClient(worker.url, "secreto"), tmp_path, nml=no_nml).atender(job())
    body = worker.bodies("/api/device/complete")[0]
    assert body["status"] == "done" and "playlist" not in body["result"] and "playlist" in body["result"]["notes"][0]


def test_no_collection_says_no_playlist(worker, tmp_path, monkeypatch):
    patch_oido(monkeypatch)
    agente(ca.WorkerClient(worker.url, "secreto"), tmp_path, nml=lambda b, c, n: None).atender(job())
    assert "colección" in worker.bodies("/api/device/complete")[0]["result"]["notes"][0]


def test_rejected_result_becomes_a_failed_job(worker, tmp_path, monkeypatch):
    patch_oido(monkeypatch)
    class Client(ca.WorkerClient):
        calls = []

        def complete(self, job_id, attempts, status, result=None, error=None):
            self.calls.append(status)
            if status == "done":
                raise ca.ResultadoRechazado("Resultados inválidos")
            return True

    client = Client(worker.url, "secreto")
    agente(client, tmp_path).atender(job())
    assert client.calls == ["done", "failed"]


def test_complete_retries_transient_errors_then_gives_up_to_the_lease(worker, tmp_path, monkeypatch):
    patch_oido(monkeypatch)
    class Flaky(ca.WorkerClient):
        tries = 0

        def complete(self, *a, **k):
            type(self).tries += 1
            raise OSError("sin red")

    said = []
    agente(Flaky(worker.url, "secreto"), tmp_path, said).atender(job())
    assert Flaky.tries == 5 and any("lease" in t for _, t in said)


def test_loop_stops_after_one_job_with_una_vez(worker, tmp_path, monkeypatch):
    patch_oido(monkeypatch)
    worker.jobs.append(job())
    assert ca.bucle(agente(ca.WorkerClient(worker.url, "secreto"), tmp_path), threading.Event(), poll_s=0.01, una_vez=True) == 0
    assert len(worker.bodies("/api/device/complete")) == 1


def test_loop_with_a_job_id_follows_that_concept_to_the_end(worker, tmp_path, monkeypatch):
    patch_oido(monkeypatch)
    worker.jobs.extend([job(sondeo_job()), job()])
    ag = agente(ca.WorkerClient(worker.url, "secreto"), tmp_path)
    assert ca.bucle(ag, threading.Event(), poll_s=0.01, job_id="1234", una_vez=True, poll_activo_s=0.01) == 0
    assert len(worker.bodies("/api/device/complete")) == 2
    assert all(b == {"job_id": "1234"} for b in worker.bodies("/api/device/claim"))


def test_loop_with_an_unknown_job_id_gives_up():
    class Client:
        def claim(self, _job):
            return None

    ag = ca.Agente(Client(), lambda *a: None, results_log=None, sleep=lambda s: None)
    t = iter(range(0, 100000, 100))
    assert ca.bucle(ag, threading.Event(), job_id="x", una_vez=True, clock=lambda: next(t), poll_activo_s=0.001) == 0


def test_loop_exits_on_rejected_token(worker, tmp_path):
    said = []
    assert ca.bucle(agente(ca.WorkerClient(worker.url, "mal"), tmp_path, said), threading.Event(), poll_s=0.01) == 2
    assert any(lvl == "error" for lvl, _ in said) and not any("mal" in t.split() for _, t in said)


def test_loop_frees_memory_only_after_clap_was_loaded():
    class Client:
        def claim(self, _job):
            return None

    stop = threading.Event()
    clock = iter(range(0, 100000, 1000))
    ag = ca.Agente(Client(), lambda *a: None, results_log=None, sleep=lambda s: None)
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
