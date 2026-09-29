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
