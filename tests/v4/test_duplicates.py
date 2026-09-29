"""
PURPOSE: Tests de la regla de temas repetidos (src/v4/common/duplicates.py), de organize.remove_tracks y
         de la exclusión de copias descartadas en organize.Library.
CHANGELOG:
  - 2026-09-29: Creación inicial.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common import duplicates as dup  # noqa: E402
from src.v4.pipeline.organize import Library, OrgStore, build, link_groups, remove_tracks  # noqa: E402
from tests.v4.test_organize import PARAMS, _library, _playlists  # noqa: E402


def q(fmt, kbps, lossless=False):
    return {"fmt": fmt, "kbps": kbps, "lossless": lossless}


def test_keeper_prefers_mp3_320_then_quality():
    # Regla de Gabriel: MP3 320 antes que todo (grupo 3: MP3 320 sobre el WAV)
    assert dup.choose_keeper([q("wav", 1411, True), q("mp3", 320)]) == 1
    # Entre los demás, la mejor calidad: sin pérdida antes que MP3 de 264 (grupo 13)
    assert dup.choose_keeper([q("mp3", 264), q("flac", 1411, True)]) == 1
    assert dup.choose_keeper([q("mp3", 128), q("mp3", 192), q("m4a", 256)]) == 2
    # Empate de calidad: no se decide solo
    assert dup.choose_keeper([q("mp3", 320), q("mp3", 320)]) is None
    assert dup.choose_keeper([q("wav", 1411, True), q("wav", 1411, True)]) is None
    assert dup.choose_keeper([q("mp3", 320), q("mp3", 320), q("mp3", 128)]) is None


def test_names_normalize():
    assert dup.norm_text("Iñaky García feat. Someone") == "inaky garcia"
    assert dup.norm_text("Benoit  Sergio") == dup.norm_text("Benoit & Sergio")
    assert dup.base_title("FISHER - Oh Sugar (Extended Mix)") == "oh sugar"
    assert dup.norm_mix("Extended Mix") == dup.norm_mix("Original Mix") == dup.norm_mix("") == ""
    assert dup.norm_mix("Oliver Huntemann Remix") == "oliver huntemann remix"
    assert dup.bracket_mix("Untitled (Human Pt.II) [LT029.5]", "x (Dub Mix)") == "Dub Mix"


def _cands(**kw):
    rng = np.random.default_rng(0)
    base = rng.standard_normal((5, 32))
    emb = np.stack([base[0], base[0] + 0.01 * rng.standard_normal(32),   # 0-1: mismo audio
                    base[1], base[1] + 0.25 * rng.standard_normal(32),   # 2-3: mismo nombre, otro master
                    base[1] + 0.25 * rng.standard_normal(32)])            # 4: otro corte de 2-3
    uids = ["u0", "u1", "u2", "u3", "u4"]
    dur = [400.0, 401.5, 300.0, 300.5, 380.0]
    artist = ["A", "Other name", "Wally Lopez", "Wally  Lopez", "Wally Lopez"]
    title = ["Uno", "Otro título", "American Icon", "American Icon (Original Mix)", "American Icon"]
    mix = ["", "", "", "Original Mix", "Extended Mix"]
    return dup.find_candidates(uids, emb, dur, artist, title, mix, **kw)


def test_find_candidates_layers_and_skip():
    groups = _cands()
    layers = {tuple(g["members"]): g["layer"] for g in groups}
    assert layers[(0, 1)] == "sonido"  # nombres distintos, mismo audio (Trommelmaschine / Trommel Machine)
    assert layers[(2, 3, 4)] == "corte"  # grupo con un corte distinto: la capa más dudosa
    assert all(g["layer"] != "sonido" or g["sim"] >= dup.SOUND_COS for g in groups)
    # pares ya decididos no vuelven; focus: solo grupos que tocan la música nueva
    skip = {frozenset(("u0", "u1"))}
    assert [g["members"] for g in _cands(skip=skip)] == [[2, 3, 4]]
    assert [g["members"] for g in _cands(focus={"u1"})] == [[0, 1]]


def test_resolve_by_layer():
    g = {"layer": "sonido", "members": [0, 1]}
    assert dup.resolve(g, [q("mp3", 192), q("mp3", 320)]) == (1, True)
    assert dup.resolve(g, [q("mp3", 320), q("mp3", 320)]) == (None, False)  # empate sin rutas: no decide
    assert dup.resolve({"layer": "nombre", "members": [0, 1]}, [q("mp3", 128), q("mp3", 320)]) == (1, True)
    assert dup.resolve({"layer": "parecido", "members": [0, 1]}, [q("mp3", 128), q("mp3", 320)]) == (1, False)


def test_decisions_roundtrip(tmp_path):
    a, b, c = ({"track_uid": u, "rel_path": f"{u}.mp3"} for u in ("a", "b", "c"))
    dup.add_decision(tmp_path, "mismo", [a, b], keep="a", layer="sonido")
    assert dup.dropped(tmp_path) == {"b": "a"}
    with pytest.raises(ValueError):
        dup.add_decision(tmp_path, "mismo", [a, c], keep="zz")
    dup.add_decision(tmp_path, "distintos", [b, c])
    assert dup.decided_pairs(dup.load_decisions(tmp_path)) == {frozenset("bc")}  # la nueva reemplazó a la de b
    assert dup.dropped(tmp_path) == {}


def test_remove_tracks_and_library_exclusion(tmp_path):
    art = _library(tmp_path)
    store = OrgStore(art, "t")
    lib = Library(art, "clap_full")
    build(store, lib, dict(PARAMS), [""])
    p = _playlists(store.load()[0])
    k = sorted(p)
    keep, copy, other = p[k[0]][0], p[k[0]][1], p[k[-1]][0]
    v2, _ = link_groups(store, lib, [{"name": "F", "color": 0, "tracks": [copy, other]}])
    before = _playlists(store.load()[0])
    dup.add_decision(art, "mismo", [{"track_uid": keep, "rel_path": "k.mp3"}, {"track_uid": copy, "rel_path": "c.mp3"}],
                     keep=keep, layer="sonido")
    v3 = remove_tracks(store, dup.dropped(art))
    after = _playlists(store.load()[0])
    assert v3 == v2 + 1 and copy not in {u for o in after.values() for u in o}
    # el resto no se mueve: mismo orden relativo en cada playlist
    assert all([u for u in before[key] if u != copy] == after[key] for key in after)
    # en la fusión, la copia se reemplaza por la que se queda
    assert sorted(store.fusions()[0]["tracks"]) == sorted([keep, other])
    assert store.meta()["history"][-1]["action"] == "dedupe"
    assert remove_tracks(store, dup.dropped(art)) == v3  # nada más para sacar
    # un build desde cero ya no la incluye
    assert copy not in Library(art, "clap_full").in_scope([""])
    assert store.undo() == v2 and copy in set(store.load()[0]["track_uid"])


def test_band_between_097_and_098_is_asked():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(64)
    a /= np.linalg.norm(a)
    o = rng.standard_normal(64)
    o -= (o @ a) * a
    o /= np.linalg.norm(o)
    b = 0.975 * a + np.sqrt(1 - 0.975 ** 2) * o  # coseno 0.975: casi igual pero no tanto
    c = float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    assert dup.LIKE_COS <= c < dup.SOUND_COS
    g = dup.find_candidates(["x", "y"], np.stack([a, b]), [400.0, 402.0], ["Benno Blome", "Abotha"],
                            ["Abotha", "Mihai Popoviciu Rmx"], ["", ""])
    assert [(x["layer"], x["members"]) for x in g] == [("parecido", [0, 1])]
    assert dup.resolve(g[0], [q("mp3", 192), q("mp3", 320)]) == (1, False)  # se pregunta aunque la calidad decida



def test_location_rank_and_tie_break():
    # Gabriel: nunca Milo; primero lo mejor clasificado (carpetas por año), después el resto, después «copia»
    assert dup.location_rank("#1 BIBO/PRO/Milo 10/Joris Voorn - Goodbye Fly (Original Mix).mp3") == 3
    assert dup.location_rank("#1 BIBO/PRO/Milo/Markus Homm - Dance With Me.mp3") == 3
    assert dup.location_rank("2020 old/para mezclar con Milo/Bora bora/x.mp3") == 0  # carpeta propia
    assert dup.location_rank("2020 new - copia/x.wav") == 2
    assert dup.location_rank("2019/Parte 4 - Techno Comercial/x.mp3") == 0
    assert dup.location_rank("#1 BIBO/PRO/Nuevitas 6/x.mp3") == 1
    # sus 8 elecciones en empates de calidad
    ties = [(["2019/Parte 4 - Techno Comercial/a.mp3", "#1 BIBO/PRO/Nuevitas 6/a.mp3"], 0),
            (["2019/Parte 1 - Tech House/b.mp3", "#1 BIBO/PRO/Nuevitas 9/b.mp3"], 0),
            (["2020 old/old 2/c.mp3", "#1 BIBO/PRO/Milo 11/c.mp3"], 0),
            (["#1 BIBO/PRO/Nuevitas 6/d.mp3", "#1 BIBO/PRO/Milo 7/d.mp3"], 0),
            (["#1 BIBO/PRO/Nuevitas/TROPICALES/e.mp3", "#1 BIBO/PRO/Milo 8/e.mp3"], 0),
            (["#1 BIBO/PRO/Milo 3/f.mp3", "2019/Parte 1 - Afro House/f.mp3"], 1),
            (["2020 old/old 6/g.mp3", "#1 BIBO/PRO/Milo 10/g.mp3"], 0),
            (["#1 BIBO/PRO/Nuevitas 12 (cachengue)/h.wav", "2019/Parte 2 - Misterio Melódico/h.wav",
              "2020 new - copia/h.wav", "2020 new/Vocal/h.wav"], 3)]
    for paths, want in ties:
        assert dup.choose_keeper([q("mp3", 320)] * len(paths), paths) == want, paths
    # la calidad manda antes que la ubicación: MP3 320 de Milo sobre MP3 192 bien ubicado
    assert dup.choose_keeper([q("mp3", 192), q("mp3", 320)], ["2019/x.mp3", "#1 BIBO/PRO/Milo/x.mp3"]) == 1
    assert dup.resolve({"layer": "sonido"}, [q("mp3", 320)] * 2, ["2019/x.mp3", "#1 BIBO/PRO/Milo/x.mp3"]) == (0, True)
    assert dup.resolve({"layer": "nombre"}, [q("mp3", 128), q("mp3", 320)], ["a/x.mp3", "b/x.mp3"]) == (1, True)
    assert dup.resolve({"layer": "corte"}, [q("mp3", 128), q("mp3", 320)], ["a/x.mp3", "b/x.mp3"]) == (1, False)


def _dedupe_library(tmp_path):
    """Biblioteca sintética con audio 'real' en disco: t1 es copia de t0 (Milo, peor ubicada)."""
    import json
    import pandas as pd
    rng = np.random.default_rng(3)
    n = 40
    emb = rng.standard_normal((n, 32))
    emb[1] = emb[0] + 0.001 * rng.standard_normal(32)
    rels = [f"2019/Parte 1/t{i}.mp3" for i in range(n)]
    rels[1] = "#1 BIBO/PRO/Milo 3/t0 copia.mp3"
    audio = tmp_path / "Música"
    for r in rels:
        (audio / r).parent.mkdir(parents=True, exist_ok=True)
        (audio / r).write_bytes(b"x" * 10)
    art = tmp_path / "art"
    (art / "features").mkdir(parents=True)
    rep = art / "representations" / "clap_full"
    rep.mkdir(parents=True)
    uids = [f"{i:04d}" + "0" * 60 for i in range(n)]
    cat = pd.DataFrame({"track_uid": uids, "rel_path": rels, "filename": [Path(r).name for r in rels],
                        "source_path": [str(audio / r) for r in rels], "folder": [str(Path(r).parent.as_posix()) for r in rels],
                        "artist": [f"A{i}" for i in range(n)], "title": [f"T{i}" for i in range(n)],
                        "duration_s": [300.0 + i * 10 for i in range(n)], "beatport_genre_norm": "Techno"})
    cat.loc[1, "duration_s"] = 300.5
    cat.to_parquet(art / "catalog.parquet", index=False)
    pd.DataFrame({"track_uid": uids, "bpm": 125.0, "key": "8A", "energy": 5.0}).to_parquet(art / "features" / "bpm_key.parquet", index=False)
    np.save(rep / "embeddings.npy", emb.astype(np.float32))
    (rep / "track_uids.json").write_text(json.dumps(uids))
    return art, audio, uids


def test_dedupe_auto_apply_move_and_restore(tmp_path):
    import pandas as pd
    from src.v4.pipeline import dedupe
    art, audio, uids = _dedupe_library(tmp_path)
    store = OrgStore(art, "t")
    v1 = build(store, Library(art, "clap_full"), dict(PARAMS, n_l1=2, l2_target_size=10), [""])
    r = dedupe.auto_resolve(art, audio, "t")
    assert [(g["layer"], g["keep"]) for g in r["resolved"]] == [("sonido", uids[0])] and not r["pending"]
    assert dup.dropped(art) == {uids[1]: uids[0]} and dup.load_decisions(art)[0]["by"] == "regla"
    assert dedupe.apply(art, "t") == v1 + 1 and uids[1] not in set(store.load()[0]["track_uid"])
    # mover: la copia va a _copias con su ruta; catálogo y decisiones apuntan ahí
    moved = dedupe.move_copies(art, audio)["moved"]
    assert [m["to"] for m in moved] == ["_copias/#1 BIBO/PRO/Milo 3/t0 copia.mp3"]
    assert (audio / "_copias/#1 BIBO/PRO/Milo 3/t0 copia.mp3").is_file() and not (audio / "#1 BIBO/PRO/Milo 3/t0 copia.mp3").exists()
    cat = pd.read_parquet(art / "catalog.parquet").set_index("track_uid")
    assert cat.loc[uids[1], "rel_path"].startswith("_copias/") and Path(cat.loc[uids[1], "source_path"]).is_file()
    assert any(t["rel_path"].startswith("_copias/") for t in dup.load_decisions(art)[0]["tracks"])
    assert dedupe.move_copies(art, audio)["moved"] == []  # ya estaba movida
    assert dedupe.candidates(art, audio, "t") == []  # nada nuevo para decidir
    # volver atrás
    back = dedupe.restore_copies(art, audio)
    assert len(back["restored"]) == 1 and (audio / "#1 BIBO/PRO/Milo 3/t0 copia.mp3").is_file()
    assert pd.read_parquet(art / "catalog.parquet").set_index("track_uid").loc[uids[1], "rel_path"] == "#1 BIBO/PRO/Milo 3/t0 copia.mp3"


def test_move_copies_protects_other_datasets_and_restores_some(tmp_path):
    from src.v4.pipeline import dedupe
    art, audio, uids = _dedupe_library(tmp_path)
    store = OrgStore(art, "t")
    build(store, Library(art, "clap_full"), dict(PARAMS, n_l1=2, l2_target_size=10), [""])
    dedupe.auto_resolve(art, audio, "t")
    # la carpeta de la copia es la de otro dataset (como test_20 -> «2020 new - copia»): no se mueve
    r = dedupe.move_copies(art, audio, protected=[audio / "#1 BIBO/PRO/Milo 3"])
    assert r["moved"] == [] and "otro dataset" in r["skipped"][0]["error"]
    assert (audio / "#1 BIBO/PRO/Milo 3/t0 copia.mp3").is_file()
    r = dedupe.move_copies(art, audio)
    assert len(r["moved"]) == 1
    assert dedupe.restore_copies(art, audio, only=["otra/ruta.mp3"])["restored"] == []  # no toca las demás
    assert len(dedupe.restore_copies(art, audio, only=["#1 BIBO/PRO/Milo 3/t0 copia.mp3"])["restored"]) == 1
