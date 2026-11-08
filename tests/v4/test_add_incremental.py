"""
PURPOSE: Tests de lo que hace que «Agregar música nueva» trabaje solo con lo nuevo (2026-11-07): la caché de tags
         (src/v4/common/tag_cache.py), el ensamblado incremental de representaciones
         (extract_representations.assemble_incremental) y el lugar en el archivo sin la versión de la organización
         (file_embeddings.same).
CHANGELOG:
  - 2026-11-07: Creación.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.tag_cache import TagCache  # noqa: E402
from src.v4.pipeline import file_embeddings  # noqa: E402
from src.v4.pipeline.extract_representations import assemble_incremental  # noqa: E402


def test_tag_cache_reads_a_file_once_until_it_changes(tmp_path):
    song = tmp_path / "tema.mp3"
    song.write_bytes(b"x" * 10)
    calls = []

    def reader(path):
        calls.append(path.name)
        return {"tag_comment": f"leído {len(calls)}", "tag_bpm": float("nan")}

    cache = TagCache(tmp_path / "tag_cache.json")
    first = cache.read(song, "tags", reader)
    assert first == {"tag_comment": "leído 1", "tag_bpm": None}   # NaN no es JSON: queda vacío
    assert cache.read(song, "tags", reader)["tag_comment"] == "leído 1" and calls == ["tema.mp3"]
    cache.save()
    again = TagCache(tmp_path / "tag_cache.json")                     # otra corrida: sigue sin abrirlo
    assert again.read(song, "tags", reader)["tag_comment"] == "leído 1" and len(calls) == 1
    time.sleep(0.01)
    song.write_bytes(b"x" * 11)                                       # se escribió un tag: cambió
    assert again.read(song, "tags", reader)["tag_comment"] == "leído 2"
    assert again.read(tmp_path / "no_esta.mp3", "tags", reader)["tag_comment"] == "leído 3"  # sin guardar


def _cache(root: Path, uid: str, l7: float, styles: float) -> None:
    root.mkdir(parents=True, exist_ok=True)
    np.savez(root / f"{uid}.npz", _l7=np.full(4, l7, np.float32), _styles=np.full(3, styles, np.float32),
             _layers=np.zeros((2, 4), np.float32))


def test_incremental_assembly_adds_the_new_and_drops_the_gone(tmp_path):
    rep_root = tmp_path / "representations"
    cache = rep_root / "maesthf_full" / "cache"
    for uid, v in (("a", 1), ("b", 2), ("c", 3)):
        _cache(cache, uid, v, v / 10)
    for suffix, dim in (("_l7", 4), ("_styles", 3)):
        out = rep_root / f"maesthf_full{suffix}"
        out.mkdir(parents=True)
        mat = np.array([[1.0] * dim, [2.0] * dim], np.float32) * (1 if suffix == "_l7" else 0.1)
        np.save(out / "embeddings.npy", mat)
        (out / "track_uids.json").write_text(json.dumps(["a", "b"]))
        (out / "manifest.json").write_text(json.dumps({"variant": out.name}))
    catalog = pd.DataFrame({"track_uid": ["b", "c"]})               # «a» salió del catálogo, «c» es nueva
    assert assemble_incremental(rep_root, "maesthf", "full", catalog, cache, {})
    for suffix, scale in (("_l7", 1.0), ("_styles", 0.1)):
        out = rep_root / f"maesthf_full{suffix}"
        uids = json.loads((out / "track_uids.json").read_text())
        mat = np.load(out / "embeddings.npy")
        assert uids == ["b", "c"] and mat.shape[0] == 2
        assert np.allclose(mat[:, 0], np.array([2, 3]) * scale)      # cada fila con su track_uid
        assert json.loads((out / "manifest.json").read_text())["assembly"] == "incremental"
    assert not (rep_root / "maesthf_full_layers").exists()          # las capas quedan solo en la caché
    # una variante que todavía no existe: no se puede sumar, se ensambla todo como antes
    assert not assemble_incremental(rep_root, "clap", "full", catalog, rep_root / "clap_full" / "cache", {})


def test_the_place_in_the_file_does_not_change_with_the_version():
    """Cada agregado sube la versión de la organización; un tema que no se movió no se reescribe."""
    vec = np.arange(4, dtype="<f4")
    place = {"org": "biblioteca", "version": 11, "track_uid": "u", "l1": 1, "l2": 2, "position": 3,
             "x": 0.5, "y": 0.25, "carpeta": "A", "playlist": "A1"}
    assert file_embeddings.same(vec, place, vec, {**place, "version": 12})
    assert not file_embeddings.same(vec, place, vec, {**place, "position": 4})
    assert not file_embeddings.same(None, place, vec, place)
