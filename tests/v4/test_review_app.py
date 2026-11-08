"""
PURPOSE: Tests de src/v4/ui/review_app.py (app local) con una biblioteca sintética: página con token,
         rechazo de pedidos sin token o con otro Host, audio con rangos y sin salir de la biblioteca,
         fusiones (borradores, aplicación con organización mantenida, las incompletas quedan),
         deshacer, descripción de carpetas (dentro/fuera de la biblioteca), orden a mano y eliminar
         fusiones.
CHANGELOG:
  - 2026-09-29: Creación inicial.
  - 2026-09-29: /api/reorder y /api/remove-fusion.
  - 2026-09-29: /cover/ (carátula del archivo o de la carpeta).
  - 2026-11-02: /api/ping y --exit-on-stdin-eof (pestaña «Organización» de la app de descargas).
  - 2026-11-04: Las copias idénticas y las descartadas por dedupe cuentan como ya organizadas; la música
                agregada se etiqueta Vocal con clap_probe (app y organize.py ingest).
  - 2026-11-04: Sin la carpeta Música/: la música sale del paquete (por track_uid) y la app es de solo lectura.
"""
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.pipeline.organize import Library, OrgStore, build  # noqa: E402
from src.v4.ui import review_app  # noqa: E402
from src.v4.ui.review_app import App, serve  # noqa: E402
from tests.v4.test_organize import PARAMS, _library, _playlists  # noqa: E402


@pytest.fixture()
def running_app(tmp_path):
    art = _library(tmp_path)
    lib_dir = tmp_path / "lib"
    (lib_dir / "A").mkdir(parents=True)
    (lib_dir / "A" / "t0.mp3").write_bytes(bytes(range(256)) * 4)
    store = OrgStore(art, "t")
    build(store, Library(art, "clap_full"), dict(PARAMS), [""])
    app = App.__new__(App)
    app.config, app.dataset, app.org_name = {}, "u", "t"
    app.artifacts, app.library, app.store = art, lib_dir.resolve(), store
    app.token, app.jobs, app.lock, app.port = "tok", {}, threading.Lock(), 0
    httpd = serve(app, port=0, open_browser=False)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield app, f"http://127.0.0.1:{app.port}"
    httpd.shutdown()


def _req(url, body=None, token="tok", headers=None):
    data = json.dumps(body or {}).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET",
                                 headers={"Content-Type": "application/json", **({"X-App-Token": token} if token else {}),
                                          **(headers or {})})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def _wait(base, job_id, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        code, body, _ = _req(f"{base}/api/job/{job_id}")
        j = json.loads(body)
        if j["status"] != "running":
            return j
        time.sleep(0.2)
    raise TimeoutError


def test_page_and_security(running_app):
    app, base = running_app
    code, body, _ = _req(base + "/")
    assert code == 200 and b'"token":"tok"' in body and b"TRAKTOR ML" in body
    assert _req(base + "/", headers={"Host": "evil.example"})[0] == 403
    assert _req(base + "/api/drafts", {"fusions": []}, token=None)[0] == 403
    assert _req(base + "/api/drafts", {"fusions": []}, headers={"Origin": "http://evil.example"})[0] == 403
    assert _req(base + "/api/drafts", {"fusions": []})[0] == 200


def test_audio_ranges_and_traversal(running_app):
    app, base = running_app
    code, body, headers = _req(base + "/audio/A/t0.mp3", headers={"Range": "bytes=10-19"})
    assert code == 206 and body == bytes(range(10, 20)) and headers["Content-Range"] == "bytes 10-19/1024"
    assert _req(base + "/audio/A/t0.mp3")[0] == 200
    assert _req(base + "/audio/..%2F..%2Fsecret.txt")[0] == 404


def test_cover_embedded_folder_and_missing(running_app):
    import numpy as np
    import soundfile as sf
    from mutagen.id3 import APIC, ID3
    app, base = running_app
    jpg = b"\xff\xd8\xff\xe0" + bytes(64)
    png = b"\x89PNG\r\n\x1a\n" + bytes(32)
    # sin carátula en el archivo ni en la carpeta
    assert _req(base + "/cover/A/t0.mp3")[0] == 404
    # imagen de la carpeta
    (app.library / "A" / "Folder.png").write_bytes(png)
    code, body, headers = _req(base + "/cover/A/t0.mp3")
    assert code == 200 and body == png and headers["Content-Type"] == "image/png"
    # imagen incluida en el MP3: gana la portada (tipo 3) sobre otras
    mp3 = app.library / "B" / "t1.mp3"
    mp3.parent.mkdir()
    try:
        sf.write(str(mp3), np.zeros((4410, 2), dtype=np.float32), 44100, format="MP3")
    except Exception:  # noqa: BLE001  (libsndfile sin MP3)
        pytest.skip("soundfile no puede escribir MP3")
    tags = ID3()
    tags.add(APIC(encoding=3, mime="image/png", type=4, desc="back", data=png))
    tags.add(APIC(encoding=3, mime="image/jpeg", type=3, desc="front", data=jpg))
    tags.save(str(mp3))
    code, body, headers = _req(base + "/cover/B/t1.mp3")
    assert code == 200 and body == jpg and headers["Content-Type"] == "image/jpeg"
    assert _req(base + "/cover/..%2F..%2Fsecret.txt")[0] == 404


def test_fusions_apply_keep_and_undo(running_app):
    app, base = running_app
    a, _, _ = app.store.load()
    p = _playlists(a)
    k = sorted(p)
    u1, u2, u3 = p[k[0]][0], p[k[-1]][0], p[k[1]][0]
    drafts = [{"name": "Para abrir", "color": 0, "tracks": [u1[:16], u2[:16]]},
              {"name": "Sola", "color": 1, "tracks": [u3[:16]]}]
    assert _req(base + "/api/drafts", {"fusions": drafts})[0] == 200
    code, body, _ = _req(base + "/api/apply-fusions", {"keep": True})
    assert code == 200
    j = _wait(base, json.loads(body)["id"])
    assert j["status"] == "done" and j["result"]["keep"] and j["result"]["moved"] == 1
    assert [f["name"] for f in app.store.fusions()] == ["Para abrir"]
    assert [d["name"] for d in app.store.drafts()] == ["Sola"]  # la incompleta sigue en armado
    v = app.store.meta()["current_version"]
    code, body, _ = _req(base + "/api/undo", {})
    assert code == 200 and json.loads(body)["version"] == v - 1 and app.store.fusions() == []


def test_describe_folder_inside_and_outside(running_app, tmp_path):
    app, base = running_app
    code, body, _ = _req(base + "/api/describe-folder", {"path": str(app.library / "A")})
    info = json.loads(body)
    assert code == 200 and info["inside"] and info["n_audio"] == 1 and not info["is_library"]
    outside = tmp_path / "Nueva"
    outside.mkdir()
    (outside / "x.mp3").write_bytes(b"0" * 10)
    info = json.loads(_req(base + "/api/describe-folder", {"path": str(outside)})[1])
    assert not info["inside"] and Path(info["dest"]).parent == app.library and Path(info["dest"]).name == "Nueva"


def test_describe_folder_counts_copies_as_already_organized(running_app):
    """Una copia idéntica (duplicates.csv) o descartada por dedupe no es música nueva: «Agregar» no la suma."""
    import pandas as pd
    from src.v4.common.duplicates import add_decision
    app, base = running_app
    a0, a1 = "a0000" + "0" * 56, "a0001" + "0" * 56
    copia = app.library / "Copia"
    copia.mkdir()
    for name in ("t0.mp3", "vieja.mp3", "nuevo.mp3"):
        (copia / name).write_bytes(b"0" * 10)
    pd.DataFrame([{"track_uid": a0, "rel_path": "Copia/t0.mp3", "kept_rel_path": "A/t0.mp3"}]).to_csv(
        app.artifacts / "duplicates.csv", index=False)
    cat = pd.read_parquet(app.artifacts / "catalog.parquet")
    gone = "c" * 64
    pd.concat([cat, pd.DataFrame([{"track_uid": gone, "rel_path": "Copia/vieja.mp3"}])]).to_parquet(
        app.artifacts / "catalog.parquet", index=False)
    add_decision(app.artifacts, "mismo", [{"track_uid": gone, "rel_path": "Copia/vieja.mp3"},
                                          {"track_uid": a1, "rel_path": "A/t1.mp3"}], keep=a1)
    info = json.loads(_req(base + "/api/describe-folder", {"path": str(copia)})[1])
    assert info["n_audio"] == 3 and info["n_known"] == 2


def test_added_music_gets_the_current_vocal_rule(running_app, monkeypatch):
    """Agregar música etiqueta Vocal con clap_probe: 'clap' reescribía la marca de toda la biblioteca
    con la regla vieja. También organize.py ingest."""
    from src.v4.pipeline import organize
    app, _ = running_app
    calls = []

    def fake_run(args, job, step):
        calls.append([str(a) for a in args])
        if args[0].endswith("tag_vocals.py"):
            raise RuntimeError("fin de la prueba")
    monkeypatch.setattr(review_app, "run_script", fake_run)
    job = app.add_collection(str(app.library / "A"), keep=True)
    t0 = time.time()
    while job.status == "running" and time.time() - t0 < 30:
        time.sleep(0.05)
    monkeypatch.setattr(organize, "_run", lambda cmd: calls.append([str(a) for a in cmd[1:]]))
    organize.ingest("u", "A", "clap_full", vocals=True)
    vocal = [c for c in calls if c[0].endswith("tag_vocals.py")]
    assert len(vocal) == 2 and all(c[c.index("--method") + 1] == "clap_probe" and "--write-tags" in c for c in vocal)


def test_added_music_gets_traktor_bpm_key_and_genre_without_stopping_on_failures(running_app, monkeypatch):
    """BPM y tonalidad también de Traktor; género y subgénero con los scripts de siempre. La búsqueda de Beatport
    de la carpeta se suma a features/beatport sin tocar el resto, Beatport bloqueado pasa a --offline y lo que
    falla queda como aviso sin cortar el agregado."""
    import pandas as pd
    from src.v4.common import beatport
    app, base = running_app
    features = app.artifacts / "features"
    cat = pd.read_parquet(app.artifacts / "catalog.parquet")
    a_uids = cat.loc[cat["rel_path"].str.startswith("A/"), "track_uid"].tolist()
    b_uid = cat.loc[cat["rel_path"].str.startswith("B/"), "track_uid"].iloc[0]
    pd.DataFrame({"track_uid": [a_uids[0], b_uid], "level": ["D", "A"], "bp_genre": [None, "Techno"]}).to_parquet(
        features / "beatport.parquet", index=False)
    calls = []

    def fake_run(args, job, step):
        calls.append([str(a) for a in args])
        name = Path(args[0]).name
        if name == "beatport_lookup.py":
            out = args[args.index("--output") + 1]
            pd.DataFrame({"track_uid": a_uids[:2], "level": ["A", "D"], "bp_genre": ["House", None]}).to_parquet(
                features / f"{out}.parquet", index=False)
        if name == "genre_model.py":
            raise RuntimeError("otro conjunto de track_uids\nmás detalle")
    monkeypatch.setattr(review_app, "run_script", fake_run)
    monkeypatch.setattr(beatport, "available", lambda: (False, "pide la verificación anti-robots de Cloudflare"))
    monkeypatch.setattr(review_app.organize, "add", lambda store, lib, scope: (7, {"added": 0}))
    monkeypatch.setattr("src.v4.pipeline.dedupe.auto_resolve", lambda *a: {"resolved": [], "pending": []})
    monkeypatch.setattr(review_app.organize, "Library", lambda *a: None)
    monkeypatch.setattr(App, "package_dir", lambda self: None)
    job = app.add_collection(str(app.library / "A"), keep=True)
    t0 = time.time()
    while job.status == "running" and time.time() - t0 < 30:
        time.sleep(0.05)
    assert job.status == "done", job.error
    assert "Género y subgénero" in job.steps
    by = {Path(c[0]).name: c for c in calls}
    assert "--traktor" in by["phase1_tags.py"] and "--estimate-missing" in by["phase1_tags.py"]
    maest = [c for c in calls if c[0].endswith("extract_representations.py") and "maesthf" in c]
    assert maest and maest[0][maest[0].index("--folder") + 1] == "A"
    # solo la carpeta: ensamblado incremental y voces de la carpeta (no toda la biblioteca)
    assert all("--incremental" in c for c in calls if c[0].endswith("extract_representations.py"))
    assert by["tag_vocals.py"][by["tag_vocals.py"].index("--scope") + 1] == "A"
    assert by["beatport_lookup.py"][by["beatport_lookup.py"].index("--folder") + 1] == "A"
    assert "--offline" in by["beatport_lookup.py"] and "--offline" in by["beatport_subgenres.py"]
    assert "--predict" in by["genre_model.py"]
    bp = pd.read_parquet(features / "beatport.parquet").set_index("track_uid")
    assert len(bp) == 3 and bp.at[a_uids[0], "bp_genre"] == "House" and bp.at[b_uid, "bp_genre"] == "Techno"
    assert not list(features.glob(f"{review_app.BEATPORT_PART}*"))
    notes = job.result["notes"]
    assert any(n.startswith("Beatport pide la verificación") for n in notes)
    assert any(n.startswith("Género del modelo: no se pudo (otro conjunto de track_uids)") for n in notes)
    # sin tonalidad en la carpeta: el aviso dice que falta analizarla en Traktor
    table = pd.read_parquet(features / "bpm_key.parquet")
    table.loc[table["track_uid"] == a_uids[0], "key"] = None
    table.to_parquet(features / "bpm_key.parquet", index=False)
    # una carpeta ya agregada se completa sin tocar las playlists; fuera de la biblioteca, no
    calls.clear()
    code, body, _ = _req(base + "/api/complete-metadata", {"path": str(app.library / "A")})
    j = _wait(base, json.loads(body)["id"])
    assert code == 200 and j["status"] == "done" and j["steps"] == review_app.COMPLETE_STEPS
    assert [Path(c[0]).name for c in calls] == ["phase1_tags.py", "extract_representations.py", "beatport_lookup.py",
                                                "beatport_subgenres.py", "genre_model.py"]
    assert any("1 tema quedó sin tonalidad" in n for n in j["result"]["notes"])
    assert _req(base + "/api/complete-metadata", {"path": str(app.library.parent)})[0] == 400


def test_the_app_watches_its_stdin_without_blocking_it(tmp_path):
    """Lanzada desde la app de descargas, la app se cierra cuando le cierran la entrada estándar. Antes la leía con
    un hilo bloqueado, y en Windows eso trabó la app dos veces el 2026-11-07: un script que no arrancaba y la app
    entera al importar dedupe a mitad de una tarea (en una prueba, importar con la lectura pendiente quedó colgado
    más de 200 s). stdin_closed mira sin bloquear: con la entrada abierta se importa y se lanzan scripts, y el
    cierre se ve."""
    import subprocess
    script = tmp_path / "paso.py"
    script.write_text("print('1/1 temas', flush=True)" + chr(10), encoding="utf-8")
    driver = tmp_path / "app.py"
    driver.write_text(chr(10).join([
        "import os, sys, threading, time",
        f"sys.path.insert(0, {str(REPO_ROOT)!r})",
        "import src.v4.ui.review_app as ra",
        "closed = threading.Event()",
        "def watch():",
        "    while not ra.stdin_closed():",
        "        time.sleep(0.2)",
        "    closed.set()",
        "threading.Thread(target=watch, daemon=True).start()",
        "time.sleep(1)",
        "import src.v4.pipeline.tag_vocals, src.v4.pipeline.genre_model  # importar a mitad de la vigilancia",
        "job = ra.Job('add', ['paso'])",
        f"ra.run_script([{str(script)!r}], job, 0)",
        "print('frac', job.frac, 'abierta' if not closed.is_set() else 'cerrada', flush=True)",
        "print('cierre visto' if closed.wait(30) else 'sin cierre', flush=True)",
        "os._exit(0)", ""]), encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(driver)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    first = proc.stdout.readline()
    deadline = time.time() + 120
    while "frac" not in first and time.time() < deadline and first:
        first = proc.stdout.readline()
    assert "frac 1.0 abierta" in first, first
    proc.stdin.close()   # la app de descargas se cierra
    try:
        rest, _ = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail("no vio que se cerró la entrada")
    assert "cierre visto" in rest, rest


def test_reorder_and_remove_fusion(running_app):
    app, base = running_app
    a, _, _ = app.store.load()
    p = _playlists(a)
    key, order = max(p.items(), key=lambda kv: len(kv[1]))
    new = order[1:] + order[:1]
    code, body, _ = _req(base + "/api/reorder", {"l1": int(key[0]), "l2": int(key[1]), "tracks": [u[:16] for u in new]})
    r = json.loads(body)
    assert code == 200 and r["can_undo"] and "orden cambiado a mano" in r["history"][-1]
    assert _playlists(app.store.load()[0])[key] == new
    assert _req(base + "/api/reorder", {"l1": int(key[0]), "l2": int(key[1]), "tracks": ["zzzz"]})[0] == 400
    # fusión aplicada: se quita con una versión nueva; un borrador con el mismo nombre también se va
    k = sorted(p)
    u1, u2, u3 = p[k[0]][0], p[k[-1]][0], p[k[1]][0]
    _req(base + "/api/drafts", {"fusions": [{"name": "F", "color": 0, "tracks": [u1[:16], u2[:16]]}]})
    _wait(base, json.loads(_req(base + "/api/apply-fusions", {"keep": True})[1])["id"])
    _req(base + "/api/drafts", {"fusions": [{"name": "F", "color": 0, "tracks": [u1[:16], u3[:16]]},
                                            {"name": "Solo borrador", "color": 1, "tracks": [u3[:16]]}]})
    code, body, _ = _req(base + "/api/remove-fusion", {"name": "F"})
    assert code == 200 and json.loads(body)["applied"] and app.store.fusions() == []
    assert [d["name"] for d in app.store.drafts()] == ["Solo borrador"]
    code, body, _ = _req(base + "/api/remove-fusion", {"name": "Solo borrador"})
    assert code == 200 and json.loads(body) == {"version": None, "applied": False} and app.store.drafts() == []
    assert _req(base + "/api/remove-fusion", {"name": "No existe"})[0] == 400


def test_ping_says_whether_a_task_is_running(running_app):
    app, base = running_app
    code, body, _ = _req(base + "/api/ping")
    assert code == 200 and json.loads(body) == {"app": "traktor-ml", "org": "t", "busy": False, "readonly": False}
    app.jobs["x"] = type("J", (), {"status": "running"})()
    assert json.loads(_req(base + "/api/ping")[1])["busy"] is True


def test_closes_when_the_app_that_opened_it_closes(running_app, monkeypatch):
    """Fin de la entrada estándar = la app de descargas se cerró: termina, pero después de la tarea en curso."""
    import io
    app, _ = running_app
    job = type("J", (), {"status": "running"})()
    app.jobs["x"] = job
    calls = []
    httpd = type("H", (), {"shutdown": lambda self: calls.append(time.time())})()
    monkeypatch.setattr(review_app.sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(review_app.time, "sleep", lambda s: setattr(job, "status", "done"))
    review_app.exit_with_parent(app, httpd)
    assert len(calls) == 1 and job.status == "done"  # esperó a la tarea



def _package_app(tmp_path, monkeypatch):
    """Los datos de la organización sin la carpeta de música, y un paquete de usb_package.py con otros nombres."""
    import pandas as pd
    import yaml
    root = tmp_path / "datasets"
    root.mkdir()
    art = _library(tmp_path).rename(root / "sinmusica")
    build(OrgStore(art, "t"), Library(art, "clap_full"), dict(PARAMS), [""])
    cat = pd.read_parquet(art / "catalog.parquet")
    pkg = tmp_path / "pen" / "TRAKTOR ML"
    rows = []
    for i, uid in enumerate(cat["track_uid"]):
        if i == 0:
            continue  # uno que no está en el paquete
        (pkg / "Musica" / "P1").mkdir(parents=True, exist_ok=True)
        (pkg / "Musica" / "P1" / f"Otro nombre {i}.mp3").write_bytes(bytes([i % 256]) * 64)
        rows.append({"carpeta": "A", "playlist": "P1", "posicion": str(i), "track_uid": uid,
                     "archivo": rf"Musica\P1\Otro nombre {i}.mp3"})  # rutas de Windows, como usb_package.py
    rows.append({"carpeta": "A", "playlist": "P1", "posicion": "999", "track_uid": "fuera",
                 "archivo": r"..\..\secreto.txt"})  # una ruta que sale del paquete no se usa
    pd.DataFrame(rows).to_csv(pkg / "_organizacion.csv", index=False, encoding="utf-8-sig")
    cfg = tmp_path / "v4.yaml"
    cfg.write_text(yaml.safe_dump({"paths": {"artifacts_root": str(root)},
                                   "datasets": {"sinmusica": {"audio_root": str(tmp_path / "no existe")}}}),
                   encoding="utf-8")
    monkeypatch.delenv("TRAKTOR_ARTIFACTS_ROOT", raising=False)
    monkeypatch.delenv("TRAKTOR_AUDIO_ROOTS", raising=False)
    return cfg, pkg, cat


def test_without_the_music_folder_it_plays_from_the_package_read_only(tmp_path, monkeypatch):
    cfg, pkg, cat = _package_app(tmp_path, monkeypatch)
    app = App("sinmusica", "t", str(cfg), package=str(pkg))
    assert app.readonly and app.package.public() == {"root": str(pkg.resolve()), "found": len(cat) - 1, "total": len(cat)}
    app.token = "tok"  # el de _req
    httpd = serve(app, port=0, open_browser=False)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{app.port}"
    try:
        code, body, _ = _req(base + "/")
        assert code == 200 and b'"readonly":"En este ordenador' in body and b'"found":' in body
        rel1, rel0 = cat["rel_path"][1], cat["rel_path"][0]
        code, body, _ = _req(base + "/audio/" + urllib.request.quote(rel1))
        assert code == 200 and body == bytes([1]) * 64              # por track_uid, con otro nombre
        assert _req(base + "/audio/" + urllib.request.quote(rel0))[0] == 404  # no está en el paquete
        assert _req(base + "/audio/..%2F..%2Fsecreto.txt")[0] == 404
        for path, body in (("/api/drafts", {"fusions": []}), ("/api/undo", {}), ("/api/describe-folder", {"path": "x"}),
                           ("/api/export", {})):
            code, out, _ = _req(base + path, body)
            assert code == 400 and "para ver y escuchar" in json.loads(out)["error"]
        assert json.loads(_req(base + "/api/ping")[1])["readonly"] is True
    finally:
        httpd.shutdown()


def test_without_the_music_folder_nor_a_package_it_says_so(tmp_path, monkeypatch):
    cfg, pkg, _ = _package_app(tmp_path, monkeypatch)
    monkeypatch.setattr(review_app, "find_package", lambda prefer_near=None: None)
    with pytest.raises(FileNotFoundError, match="Audio root for dataset"):  # el aviso de la pestaña lo reconoce
        App("sinmusica", "t", str(cfg))


def test_the_package_is_looked_for_first_where_the_organization_data_is():
    kinds = {"C:\\": 3, "D:\\": 2, "E:\\": 3, "R:\\": 5, "Z:\\": 4}  # fijo, extraíble, fijo, lectora, red
    assert review_app.drive_order(kinds) == ["C:\\", "E:\\", "D:\\"]
    assert review_app.drive_order(kinds, "D:\\") == ["D:\\", "C:\\", "E:\\"]   # los datos en el pendrive
    assert review_app.drive_order(kinds, "Z:\\") == ["C:\\", "E:\\", "D:\\"]   # nunca una unidad de red


def test_a_cut_or_repeated_copy_does_not_make_a_second_folder(running_app, tmp_path):
    """Gabriel, 2026-11-07: «Agregar» se trabó, al reintentar copió la carpeta otra vez («Workout Delft (2)») y la
    primera copia quedó suelta. Una copia a medias se completa; lo que ya está en la biblioteca no se vuelve a copiar."""
    import pandas as pd
    app, base = running_app
    src = tmp_path / "Nueva"
    src.mkdir()
    (src / "x.mp3").write_bytes(b"x" * 300)
    (src / "y.mp3").write_bytes(b"y" * 400)
    describe = lambda: json.loads(_req(base + "/api/describe-folder", {"path": str(src)})[1])  # noqa: E731
    info = describe()
    assert info["dest_kind"] == "new" and Path(info["dest"]) == app.library / "Nueva"
    (app.library / "Nueva").mkdir()                       # el intento anterior se cortó copiando x.mp3
    (app.library / "Nueva" / "x.mp3").write_bytes(b"x" * 120)
    info = describe()
    assert info["dest_kind"] == "resume" and Path(info["dest"]) == app.library / "Nueva"
    job = review_app.Job("add", ["copia"])
    review_app.copy_tree(src, Path(info["dest"]), job, 0)
    assert (app.library / "Nueva" / "x.mp3").read_bytes() == b"x" * 300 and (app.library / "Nueva" / "y.mp3").exists()
    assert not list((app.library / "Nueva").glob("*.copiando"))
    (app.library / "Nueva" / "z.mp3").write_bytes(b"z")  # otra carpeta con el mismo nombre: no es esta copia
    assert Path(describe()["dest"]) == app.library / "Nueva (2)"
    # x.mp3 ya está organizado en la biblioteca (mismo nombre y tamaño): se copia solo y.mp3, junto a él
    uid = app.store.load()[0]["track_uid"].iloc[0]
    cat = pd.read_parquet(app.artifacts / "catalog.parquet")
    cat.loc[cat["track_uid"] == uid, ["rel_path", "filesize_bytes"]] = ["A/x.mp3", 300]
    cat.to_parquet(app.artifacts / "catalog.parquet", index=False)
    info = describe()
    assert info["dest_kind"] == "home" and Path(info["dest"]) == app.library / "A"
    assert info["n_known"] == 1 and info["skip"] == ["x.mp3"]


def test_a_stuck_script_is_cut_and_retried_and_cancel_stops_it(tmp_path):
    pytest.importorskip("psutil")
    count = tmp_path / "intentos.txt"
    stuck = tmp_path / "trabado.py"
    stuck.write_text(f"open({str(count)!r}, 'a').write('x')\nimport time\ntime.sleep(60)\n", encoding="utf-8")
    job = review_app.Job("add", ["paso"])
    t0 = time.time()
    with pytest.raises(RuntimeError, match="se trabó"):
        review_app.run_script([str(stuck)], job, 0, stall_s=1.5)
    assert count.read_text() == "xx" and time.time() - t0 < 30   # se cortó y se lanzó una vez más
    busy = tmp_path / "trabajando.py"                              # sin escribir pero usando CPU: no está trabado
    busy.write_text("import time\nt = time.time()\nwhile time.time() - t < 4: pass\nprint('1/1 temas')\n",
                    encoding="utf-8")
    review_app.run_script([str(busy)], job, 0, stall_s=1.5)
    assert job.frac == 1.0
    job = review_app.Job("add", ["paso"])
    errors = []
    worker = threading.Thread(target=lambda: errors.append(
        pytest.raises(review_app.Cancelled, review_app.run_script, [str(stuck)], job, 0)))
    worker.start()
    time.sleep(1.5)
    job.cancel()                                                   # «Cancelar»
    worker.join(timeout=15)
    assert not worker.is_alive() and errors and job.proc is None


def test_the_package_only_rewrites_what_moved(running_app):
    """«Actualizar el pendrive» reescribía los 1760 archivos en cada agregado; ahora solo lo nuevo o lo movido."""
    app, base = running_app
    since = app.store.meta()["current_version"]
    assert review_app.changed_places(app.store, since) == set()
    a, _, _ = app.store.load()
    key, order = max(_playlists(a).items(), key=lambda kv: len(kv[1]))
    new = order[1:] + order[:1]                                    # el primero pasa al final: se mueven todos
    _req(base + "/api/reorder", {"l1": int(key[0]), "l2": int(key[1]), "tracks": [u[:16] for u in new]})
    assert review_app.changed_places(app.store, since) == set(order)
