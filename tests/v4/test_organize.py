"""
PURPOSE: Tests de src/v4/pipeline/organize.py con datos sintéticos (sin audio ni modelos): build por
         alcance, add congelado (nada existente cambia de playlist, orden relativo ni mapa), link
         congelado (solo se mueve el grupo) y con --rebuild (semilla junta), fusión de semillas,
         inserción de menor costo, catálogo por alcance y ensamblado de embeddings con --folder.
CHANGELOG:
  - 2026-09-29: Creación inicial.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.pipeline.organize import (  # noqa: E402
    DEFAULT_PARAMS, Library, OrgStore, add, build, components, link,
)
from src.v4.pipeline.phase4_order import insert_position  # noqa: E402

PARAMS = dict(DEFAULT_PARAMS, rep="clap_full", pca_dim=8, n_l1=3, l2_target_size=10, min_new_playlist=4)


def _library(tmp: Path, n_a: int = 60, n_b: int = 24, seed: int = 0) -> Path:
    """Artifacts falsos: 3 estilos en carpeta A y B (B incluye un estilo nuevo, lejano)."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((4, 16)) * 3
    rows, embs = [], []
    for i in range(n_a + n_b):
        folder = "A" if i < n_a else "B"
        style = i % 3 if folder == "A" else (i % 2) * 3  # B: estilo 0 (conocido) y 3 (nuevo)
        embs.append(centers[style] + rng.standard_normal(16) * 0.3)
        rows.append({"track_uid": f"{folder.lower()}{i:04d}" + "0" * 56, "rel_path": f"{folder}/t{i}.mp3",
                     "filename": f"t{i}.mp3", "artist": f"Art{i}", "title": f"Tit{i}",
                     "beatport_genre_norm": ["Techno", "House", "Deep House", "Trance"][style]})
    art = tmp / "art"
    (art / "features").mkdir(parents=True)
    rep = art / "representations" / "clap_full"
    rep.mkdir(parents=True)
    cat = pd.DataFrame(rows)
    cat.to_parquet(art / "catalog.parquet", index=False)
    pd.DataFrame({"track_uid": cat["track_uid"], "bpm": rng.uniform(120, 130, len(cat)).round(1),
                  "key": ["8A"] * len(cat), "energy": rng.integers(4, 8, len(cat)).astype(float)}
                 ).to_parquet(art / "features" / "bpm_key.parquet", index=False)
    np.save(rep / "embeddings.npy", np.array(embs, dtype=np.float32))
    (rep / "track_uids.json").write_text(json.dumps(cat["track_uid"].tolist()))
    return art


def _playlists(df):
    return {k: g.sort_values("position")["track_uid"].tolist() for k, g in df.groupby(["l1", "l2"])}


def test_components_merge_overlapping():
    assert components([["a", "b"], ["c", "d"], ["b", "c"], ["e", "e"]]) == [["a", "b", "c", "d"]]


def test_insert_position_keeps_relative_order():
    rng = np.random.default_rng(1)
    E = rng.standard_normal((6, 4)); E /= np.linalg.norm(E, axis=1, keepdims=True)
    bpm, keys = np.array([120., 121, 122, 123, 124, 121.5]), ["8A"] * 6
    order = [0, 1, 2, 3, 4]
    pos = insert_position(order, 5, E, bpm, keys)
    new = order[:pos] + [5] + order[pos:]
    assert [x for x in new if x != 5] == order and 0 <= pos <= 5


def test_build_scope_add_frozen_and_link(tmp_path):
    art = _library(tmp_path)
    store = OrgStore(art, "t")
    lib = Library(art, "clap_full")
    v1 = build(store, lib, dict(PARAMS), ["A"])
    a1, _, _ = store.load()
    assert v1 == 1 and len(a1) == 60 and a1["track_uid"].str.startswith("a").all()

    v2, info = add(store, lib, "B")
    a2, names2, _ = store.load()
    assert v2 == 2 and info["added"] == 24 and len(a2) == 84
    m = a1.merge(a2, on="track_uid", suffixes=("_1", "_2"))
    assert ((m.l1_1 == m.l1_2) & (m.l2_1 == m.l2_2)).all()  # nadie cambió de playlist
    assert np.allclose(m.x_1, m.x_2) and np.allclose(m.y_1, m.y_2)  # ni de lugar en el mapa
    p1, p2 = _playlists(a1), _playlists(a2)
    for k, old in p1.items():
        assert [u for u in p2[k] if u in set(old)] == old  # orden relativo intacto
    # el estilo nuevo (lejano) forma playlists nuevas; el conocido entra en las existentes
    assert info.get("add-new", 0) > 0 and info.get("add", 0) > 0
    assert any("(nuevos)" in n for n in names2.values())
    assert "B" in store.meta()["added_scopes"]

    # link congelado: dos temas de playlists distintas
    keys = sorted(p2)
    ua, ub = p2[keys[0]][0], p2[keys[-1]][0]
    v3, info = link(store, lib, [ua, ub])
    a3, _, _ = store.load()
    moved = set(info["moved"])
    assert v3 == 3 and len(moved) == 1
    assert a3[a3.track_uid.isin([ua, ub])][["l1", "l2"]].drop_duplicates().shape[0] == 1
    m = a2.merge(a3, on="track_uid", suffixes=("_2", "_3"))
    changed = m[(m.l1_2 != m.l1_3) | (m.l2_2 != m.l2_3)]["track_uid"]
    assert set(changed) == moved
    assert np.allclose(m.x_2, m.x_3)

    # rebuild: todo desde cero sobre A + B, con la semilla junta
    v4, _ = link(store, lib, [ua, ub], rebuild=True)
    a4, _, _ = store.load()
    assert v4 == 4 and len(a4) == 84 and (a4["origin"] == "build").all()
    assert a4[a4.track_uid.isin([ua, ub])][["l1", "l2"]].drop_duplicates().shape[0] == 1


def test_link_rejects_ambiguous_query(tmp_path):
    art = _library(tmp_path)
    store = OrgStore(art, "t")
    lib = Library(art, "clap_full")
    build(store, lib, dict(PARAMS), [""])
    with pytest.raises(ValueError, match="coincide con"):
        link(store, lib, ["Art1", "Art2"])  # "Art1" matchea Art1, Art10..Art19


def test_catalog_scope_keeps_rest(tmp_path, monkeypatch):
    import soundfile as sf
    from src.v4.common.catalog import build_catalog
    root = tmp_path / "lib"
    for folder in ("Old", "New"):
        (root / folder).mkdir(parents=True)
    rng = np.random.default_rng(0)
    for name in ("Old/a.flac", "New/b.flac"):
        sf.write(str(root / name), (rng.standard_normal((44100, 2)) * 0.1).astype(np.float32), 44100, format="FLAC")
    monkeypatch.setenv("TRAKTOR_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    build_catalog(root, "u", {}, recursive=True, hash_mode="audio")
    sf.write(str(root / "New" / "c.flac"), (rng.standard_normal((44100, 2)) * 0.1).astype(np.float32), 44100, format="FLAC")
    cat = build_catalog(root, "u", {}, recursive=True, hash_mode="audio", scope="New")
    assert sorted(cat["rel_path"]) == ["New/b.flac", "New/c.flac", "Old/a.flac"]


def test_extract_assembly_with_folder_uses_full_catalog(tmp_path, monkeypatch):
    from src.v4.pipeline.extract_representations import run
    art = tmp_path / "artifacts" / "u"
    cache = art / "representations" / "clap_full" / "cache"
    cache.mkdir(parents=True)
    cat = pd.DataFrame({"track_uid": ["x1", "x2"], "rel_path": ["A/1.mp3", "B/2.mp3"], "filename": ["1.mp3", "2.mp3"],
                        "source_path": ["-", "-"], "duration_s": [300.0, 300.0]})
    cat.to_parquet(art / "catalog.parquet", index=False)
    for u in ("x1", "x2"):
        np.savez(cache / f"{u}.npz", _=np.ones(4, dtype=np.float32))
    monkeypatch.setenv("TRAKTOR_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    run("u", {}, ["clap"], ["full"], None, None, Path("."), None, folder="A", assemble_only=True)
    assert json.loads((art / "representations" / "clap_full" / "track_uids.json").read_text()) == ["x1", "x2"]


def test_read_seed_file(tmp_path):
    from src.v4.pipeline.organize import read_seed_file
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"org": "biblioteca", "groups": [{"tracks": ["a1", "b2"], "labels": ["A", "B"]}, {"tracks": ["c3"]}]}))
    assert read_seed_file(f, "biblioteca") == [{"tracks": ["a1", "b2"]}]
    with pytest.raises(ValueError, match="organización"):
        read_seed_file(f, "otra")



def test_fusions_versioned_named_and_undo(tmp_path):
    from src.v4.pipeline.organize import link_groups, merge_fusions
    art = _library(tmp_path)
    store = OrgStore(art, "t")
    lib = Library(art, "clap_full")
    build(store, lib, dict(PARAMS), [""])
    a, _, _ = store.load()
    p = _playlists(a)
    k = sorted(p)
    t1, t2, t3 = p[k[0]][0], p[k[-1]][0], p[k[1]][0]
    v2, _ = link_groups(store, lib, [{"name": "Para abrir", "color": 2, "tracks": [t1, t2]}])
    assert store.fusions() == [{"name": "Para abrir", "color": 2, "tracks": sorted([t1, t2])}]
    # una fusión nueva que comparte un tema se une a la existente y conserva su nombre y color
    v3, _ = link_groups(store, lib, [{"name": "Otra", "color": 5, "tracks": [t2, t3]}])
    assert [(f["name"], f["color"], len(f["tracks"])) for f in store.fusions()] == [("Para abrir", 2, 3)]
    # deshacer vuelve a v2 con su fusión de 2 temas; v3 no se borra
    assert store.undo() == v2 and store.meta()["current_version"] == v2
    assert len(store.fusions()[0]["tracks"]) == 2 and (store.dir / f"v{v3}").exists()
    assert merge_fusions([], [{"name": "A", "color": 0, "tracks": ["x", "y"]}, {"name": "B", "color": 1, "tracks": ["z", "w"]}]) == [
        {"name": "A", "color": 0, "tracks": ["x", "y"]}, {"name": "B", "color": 1, "tracks": ["w", "z"]}]



def test_undo_add_restores_scopes(tmp_path):
    art = _library(tmp_path)
    store = OrgStore(art, "t")
    lib = Library(art, "clap_full")
    build(store, lib, dict(PARAMS), ["A"])
    add(store, lib, "B")
    assert store.meta()["added_scopes"] == ["B"]
    store.undo()
    assert store.meta()["added_scopes"] == [] and len(store.load()[0]) == 60


def test_reorder_by_hand_survives_add(tmp_path):
    from src.v4.pipeline.organize import reorder
    art = _library(tmp_path)
    store = OrgStore(art, "t")
    lib = Library(art, "clap_full")
    v1 = build(store, lib, dict(PARAMS), ["A"])
    a, _, _ = store.load()
    key, order = max(_playlists(a).items(), key=lambda kv: len(kv[1]))
    new = [order[-1]] + order[:-1]  # el último pasa a ser el primero (por prefijo de uid)
    v2 = reorder(store, key[0], key[1], [u[:16] for u in new])
    assert v2 == v1 + 1 and _playlists(store.load()[0])[key] == new
    assert store.meta()["history"][-1]["action"] == "reorder"
    assert reorder(store, key[0], key[1], new) == v2  # mismo orden: sin versión nueva
    with pytest.raises(ValueError):
        reorder(store, key[0], key[1], new[:-1])  # faltan temas
    # agregar música no rompe el orden a mano: los que ya estaban conservan su orden relativo
    add(store, lib, "B")
    after = _playlists(store.load()[0])[key]
    assert [u for u in after if u in set(new)] == new
    assert store.undo() == v2 and store.undo() == v1


def test_remove_fusion_keeps_tracks_in_place(tmp_path):
    from src.v4.pipeline.organize import link_groups, remove_fusion
    art = _library(tmp_path)
    store = OrgStore(art, "t")
    lib = Library(art, "clap_full")
    build(store, lib, dict(PARAMS), [""])
    p = _playlists(store.load()[0])
    k = sorted(p)
    t1, t2 = p[k[0]][0], p[k[-1]][0]
    v2, _ = link_groups(store, lib, [{"name": "Para abrir", "color": 1, "tracks": [t1, t2]}])
    before = store.load()[0].set_index("track_uid")[["l1", "l2", "position"]]
    v3 = remove_fusion(store, "Para abrir")
    assert v3 == v2 + 1 and store.fusions() == []
    assert store.load()[0].set_index("track_uid")[["l1", "l2", "position"]].equals(before)
    assert store.meta()["history"][-1]["action"] == "unlink"
    with pytest.raises(ValueError):
        remove_fusion(store, "Para abrir")
    assert store.undo() == v2 and [f["name"] for f in store.fusions()] == ["Para abrir"]
