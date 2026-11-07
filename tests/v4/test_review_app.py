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
    assert code == 200 and j["status"] == "done" and j["steps"] == ["BPM y tonalidad", "Género y subgénero"]
    assert [Path(c[0]).name for c in calls] == ["phase1_tags.py", "extract_representations.py", "beatport_lookup.py",
                                                "beatport_subgenres.py", "genre_model.py"]
    assert any("1 tema quedó sin tonalidad" in n for n in j["result"]["notes"])
    assert _req(base + "/api/complete-metadata", {"path": str(app.library.parent)})[0] == 400


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
