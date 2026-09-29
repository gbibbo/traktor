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


def test_resolve_only_sound_with_quality_difference():
    g = {"layer": "sonido", "members": [0, 1]}
    assert dup.resolve(g, [q("mp3", 192), q("mp3", 320)]) == (1, True)
    assert dup.resolve(g, [q("mp3", 320), q("mp3", 320)]) == (None, False)  # empate: se pregunta
    assert dup.resolve({"layer": "nombre", "members": [0, 1]}, [q("mp3", 128), q("mp3", 320)]) == (1, False)


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

