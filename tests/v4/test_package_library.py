"""
PURPOSE: Tests de la biblioteca en el pendrive (src/v4/pipeline/package_library.py y
         usb_package.refresh_from_org): un paquete sintético armado con build_package, `adopt` (catálogo a los
         archivos del paquete, respaldo y junction), un cambio de orden en la organización que llega a las
         playlists del paquete sin mover ni tocar el audio, y la app (review_app) que lo hace al «Preparar».
CHANGELOG:
  - 2026-11-04: Creación.
"""
import hashlib
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.pipeline import file_embeddings as fe  # noqa: E402
from src.v4.pipeline import organize  # noqa: E402
from src.v4.pipeline.organize import Library, OrgStore, build  # noqa: E402
from src.v4.pipeline.package_library import adopt, is_package_library  # noqa: E402
from src.v4.pipeline.usb_package import build_package, refresh_from_org  # noqa: E402
from tests.v4.test_organize import PARAMS, _library  # noqa: E402
from tests.v4.test_usb_package import tiny_mp3  # noqa: E402


def _audio_digest(folder: Path) -> dict:
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob("*.mp3")}


@pytest.fixture()
def package(tmp_path, monkeypatch):
    """Organización sintética 't' (dataset 'art') con su música en tmp/orig y un paquete armado de ahí."""
    art = _library(tmp_path)  # tmp/art = artifacts del dataset 'art'
    monkeypatch.setenv("TRAKTOR_ARTIFACTS_ROOT", str(tmp_path))
    cat = pd.read_parquet(art / "catalog.parquet")
    orig = tmp_path / "orig"
    paths = []
    for rel in cat["rel_path"]:
        f = orig / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        tiny_mp3(f)
        paths.append(str(f))
    cat.assign(source_path=paths, duration_s=300.0).to_parquet(art / "catalog.parquet", index=False)
    build(OrgStore(art, "t"), Library(art, "clap_full"), dict(PARAMS), [""])
    assert build_package("art", "t", tmp_path / "pen", config={}) == 0
    return tmp_path, art, tmp_path / "pen" / "TRAKTOR ML"


def test_adopt_points_the_catalog_at_the_package(package):
    tmp, art, pkg = package
    before = pd.read_parquet(art / "catalog.parquet")
    assert adopt(pkg, "art", {}, dry_run=True, link=tmp / "Música") == {
        "catalogo": len(before), "en_el_paquete": len(before), "sin_archivo": 0}
    assert not (tmp / "Música").exists()                                            # el ensayo no escribe nada
    out = adopt(pkg, "art", {}, link=tmp / "Música")
    cat = pd.read_parquet(art / "catalog.parquet")
    music = (pkg / "Musica").resolve()
    assert out["enlace"] == "creado" and (tmp / "Música").resolve() == music
    assert is_package_library(tmp / "Música")
    assert all(music in Path(p).parents and Path(p).is_file() for p in cat["source_path"])
    assert set(cat["orig_rel_path"]) <= set(before["rel_path"])                     # la ruta original queda
    assert (Path(out["respaldo"]) / "catalog.parquet").is_file()                    # respaldo antes de escribir
    assert adopt(pkg, "art", {}, link=tmp / "Música")["enlace"] == "ya estaba"      # se puede repetir
    (tmp / "otra").mkdir()
    snapshot = (art / "catalog.parquet").read_bytes()
    with pytest.raises(RuntimeError):                                               # una carpeta real no se toca
        adopt(pkg, "art", {}, link=tmp / "otra")
    assert (art / "catalog.parquet").read_bytes() == snapshot                       # y no se escribió nada


def test_a_new_order_reaches_the_package_without_touching_audio(package):
    tmp, art, pkg = package
    adopt(pkg, "art", {}, link=tmp / "Música")
    audio = _audio_digest(pkg / "Musica")
    first = pd.read_csv(pkg / "_organizacion.csv", encoding="utf-8-sig", dtype=str).set_index("track_uid")
    store = OrgStore(art, "t")
    assign, _, _ = store.load()
    (l1, l2), group = max(assign.groupby(["l1", "l2"]), key=lambda kv: len(kv[1]))
    order = list(group.sort_values("position")["track_uid"])
    organize.reorder(store, int(l1), int(l2), order[::-1])                      # al revés
    assert refresh_from_org(pkg, "art", "t", {}) == 0
    index = pd.read_csv(pkg / "_organizacion.csv", encoding="utf-8-sig", dtype=str)
    rows = index[index["track_uid"].isin(order)].sort_values("posicion", key=lambda s: s.astype(int))
    assert list(rows["track_uid"]) == order[::-1]                               # la playlist, en el orden nuevo
    assert _audio_digest(pkg / "Musica") == audio                               # el audio, igual y en su lugar
    after = index.set_index("track_uid").loc[first.index]
    for col in ("origen", "nombre_original", "carpeta_original"):                # Preparar Traktor los necesita
        assert list(after[col]) == list(first[col]), col
    plan, rep = fe.package_plan(pkg, art, "t")
    fe.write_package(plan, rep)
    _, place = fe.read_track(pkg / PurePath(rows.iloc[0]["archivo"]), rep)
    assert place["track_uid"] == order[-1] and place["position"] == 0           # el lugar guardado en el archivo


def test_the_app_updates_the_package_when_preparing(package):
    from src.v4.ui.review_app import App
    tmp, art, pkg = package
    adopt(pkg, "art", {}, link=tmp / "Música")
    app = App.__new__(App)
    app.config, app.dataset, app.org_name = {}, "art", "t"
    app.artifacts, app.library, app.store = art, (tmp / "Música").resolve(), OrgStore(art, "t")
    app.token, app.jobs, app.lock, app.port = "tok", {}, threading.Lock(), 0
    assert app.package_dir() == pkg.resolve() and not app.readonly
    (pkg / "_organizacion.csv").unlink()  # se tiene que volver a escribir
    (pkg / "_organizacion.csv").write_text("carpeta,playlist,posicion,track_uid,archivo\n", encoding="utf-8")
    job = app.export()
    while job.status == "running":
        time.sleep(0.2)
    assert job.status == "done", job.error
    assert job.result["package"] == str(pkg.resolve()) and job.steps[-1] == "Actualizar el pendrive"
    assert len(pd.read_csv(pkg / "_organizacion.csv", encoding="utf-8-sig")) == len(OrgStore(art, "t").load()[0])


from pathlib import PurePath  # noqa: E402  (al final: solo para armar rutas de Windows del CSV)
