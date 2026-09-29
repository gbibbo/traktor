"""
PURPOSE: Tests de tools/playlist_review/build_review_page.py: carpeta de origen abreviada, datos por tema
         (BPM, tonalidad, energía, Vocal desde el comentario), carpetas/playlists en orden con UMAP y
         sugeridas, marcas de temas nuevos/semilla, y que el JSON embebido no pueda cerrar el <script>.
CHANGELOG:
  - 2026-09-27: Creación inicial.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.playlist_review.build_review_page import build_tracks, org_payload, render, short_folder  # noqa: E402


def _catalog():
    return pd.DataFrame({
        "track_uid": ["u1" * 8, "u2" * 8, "u3" * 8],
        "rel_path": ["#1 BIBO/PRO/Nuevitas 5/a.mp3", "2019/Parte 1 - House/b.mp3", "c.mp3"],
        "folder": ["#1 BIBO/PRO/Nuevitas 5", "2019/Parte 1 - House", ""],
        "artist": ["A", None, "C"], "title": ["One", "Two", "Three"],
        "tag_genre": ["Techno", None, "House"],
        "tag_comment": ["8A - Energy 6 - Vocal", "Vocalstation", None],
    })


def _bpm_key():
    return pd.DataFrame({"track_uid": ["u1" * 8, "u2" * 8, "u3" * 8], "bpm": [128.0, None, 122.4],
                         "key": ["8A", None, "5A"], "energy": [6, None, 4]})


def test_short_folder():
    assert short_folder("#1 BIBO/PRO/Nuevitas 6/asdasdsd") == "Nuevitas 6 / asdasdsd"
    assert short_folder("2019/Parte 1 - House") == "2019 / Parte 1 - House"
    assert short_folder("") == ""


def test_build_tracks_fields():
    uids = ["u3" * 8, "u1" * 8, "u2" * 8]
    t = build_tracks(_catalog(), _bpm_key(), uids)
    assert [x["t"] for x in t] == ["Three", "One", "Two"]
    assert t[1] == {"u": ("u1" * 8)[:16], "p": "#1 BIBO/PRO/Nuevitas 5/a.mp3", "a": "A", "t": "One", "g": "Techno",
                    "fs": "Nuevitas 5", "b": 128.0, "k": "8A", "e": 6, "v": True}
    assert t[2]["v"] is False and t[2]["b"] is None and t[2]["a"] == ""  # "Vocalstation" no es la marca


def test_org_payload_order_and_suggested():
    ordered = pd.DataFrame({
        "track_uid": ["a", "b", "c", "d", "e"], "label_l1": [0, 0, 0, 1, 1], "label_l2": [0, 0, 1, 0, 0],
        "position": [1, 0, 0, 1, 0], "umap_x": [0.1, 0.2, 0.3, 0.4, 0.5], "umap_y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    names = {"l1_0": "Techno", "l1_0_l2_0": "A1 Techno", "l1_0_l2_1": "A2", "l1_1": "Group B", "l1_1_l2_0": "B1"}
    idx = {u: i for i, u in enumerate("abcde")}
    p = org_payload({"hash": "h1", "ordered": ordered, "names": names}, idx, "Actual", n_suggested=1)
    assert p["id"] == "h1" and p["folders"][0]["playlists"][0]["id"] == "h1:0:0"
    f0, f1 = p["folders"]
    assert f0["name"] == "A · Techno" and f1["name"] == "B"
    assert f0["playlists"][0]["tracks"] == [idx["b"], idx["a"]]  # por position
    assert [pl["suggested"] for pl in f0["playlists"]] == [True, False] and not f1["playlists"][0]["suggested"]
    assert len(p["xy"]) == len(p["trackIdx"]) == 5


def test_render_escapes_script_close():
    html = render({"run_id": "x", "tracks": [{"t": "</script><b>"}], "orgs": []})
    payload = re.search(r"const DATA = (\{.*?\});\n", html, re.S).group(1)
    assert "</script>" not in payload
    assert json.loads(payload.replace("<\\/", "</"))["tracks"][0]["t"] == "</script><b>"


def test_org_payload_flags_new_and_seed_tracks():
    ordered = pd.DataFrame({"track_uid": ["a", "b", "c"], "label_l1": [0, 0, 0], "label_l2": [0, 0, 0],
                            "position": [0, 1, 2], "umap_x": [0.0, 1.0, 2.0], "umap_y": [0.0, 1.0, 2.0],
                            "origin": ["build", "add-new", "link"]})
    idx = {"a": 0, "b": 1, "c": 2}
    p = org_payload({"hash": "org:x", "ordered": ordered, "names": {}}, idx, "x v2")
    assert p["flags"] == {"1": "nuevo", "2": "semilla"}


def test_history_lines_readable():
    from tools.playlist_review.build_review_page import history_lines
    meta = {"history": [
        {"version": 1, "action": "import", "source_hash": "fb78f2f6", "n_tracks": 10, "n_playlists": 2},
        {"version": 2, "action": "add", "scope": "2026 Octubre", "added": 7, "add": 5, "add-new": 2, "n_tracks": 17, "n_playlists": 3},
        {"version": 3, "action": "link", "groups": [["a", "b"]], "moved": ["b"], "n_tracks": 17, "n_playlists": 3},
        {"version": 4, "action": "link-rebuild", "n_tracks": 17, "n_playlists": 4}]}
    assert history_lines(meta) == [
        "v1 · importada de fb78f2f6",
        "v2 · agregada '2026 Octubre': +7 (5 a playlists existentes, 2 en playlists nuevas)",
        "v3 · semillas: 1 grupo(s), 1 tema(s) movido(s)",
        "v4 · rehecha desde cero con semillas (4 playlists)"]


def test_org_payload_carries_seeds_and_meta():
    ordered = pd.DataFrame({"track_uid": ["a", "b", "c"], "label_l1": [0, 0, 0], "label_l2": [0, 0, 0],
                            "position": [0, 1, 2], "umap_x": [0.0, 1.0, 2.0], "umap_y": [0.0, 1.0, 2.0]})
    org = {"hash": "org:x", "ordered": ordered, "names": {}, "cli": "x", "legacy": ["h1"],
           "meta": {"name": "x", "version": 2, "current": True, "scopes": [""], "history": []},
           "seeds": [["a", "c"], ["zz", "b"]]}
    p = org_payload(org, {"a": 0, "b": 1, "c": 2}, "x v2")
    assert p["cli"] == "x" and p["legacy"] == ["h1"] and p["seeds"] == [[0, 2], [1]]
