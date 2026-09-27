"""
PURPOSE: Tests del MVP de exportación a software de DJ y de las piezas nuevas del pipeline:
         codificación de rutas Rekordbox/Traktor, rekordbox.xml y NML bien formados con conteos
         coherentes, M3U8 absoluto, bpm_key desde tags, arranque del orden por energía, BPM estimado y
         ventanas de 10 s del extractor.
CHANGELOG:
  - 2026-09-27: Creación inicial.
"""
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.dj_export import (  # noqa: E402
    PlaylistSpec, rekordbox_location, traktor_location, write_m3u8, write_rekordbox_xml, write_traktor_nml,
)
from src.v4.pipeline.extract_representations import split_windows  # noqa: E402
from src.v4.pipeline.phase1_tags import bpm_key_from_catalog  # noqa: E402
from src.v4.pipeline.phase4_order import order_cluster_tracks  # noqa: E402


def _tracks():
    return pd.DataFrame({
        "track_uid": ["u1", "u2", "u3"],
        "source_path": [r"C:\VS\traktor\Música\#1 BIBO\PRO\Milo 3\A - One.mp3",
                        r"C:\VS\traktor\Música\2019\Parte 1 - House\B & C - Two.flac",
                        r"C:\VS\traktor\Música\2021\x.wav"],
        "artist": ["A", "B & C", None], "title": ["One", "Two", None],
        "tag_genre": ["Techno", "House", None], "tag_label": [None, "Lbl", None],
        "tag_comment": ["8A - Energy 6 - Vocal", None, None],
        "duration_s": [360.4, 400.0, 300.0], "filesize_bytes": [10, 20, 30],
    }).set_index("track_uid")


def test_rekordbox_location_encoding():
    loc = rekordbox_location(r"C:\VS\traktor\Música\#1 BIBO\a b.mp3")
    assert loc == "file://localhost/C:/VS/traktor/M%C3%BAsica/%231%20BIBO/a%20b.mp3"


def test_traktor_location():
    loc = traktor_location(r"C:\VS\traktor\Música\2019\x.mp3")
    assert loc == {"VOLUME": "C:", "DIR": "/:VS/:traktor/:Música/:2019/:", "FILE": "x.mp3",
                   "KEY": "C:/:VS/:traktor/:Música/:2019/:x.mp3"}


def _specs():
    return [PlaylistSpec("A · Techno", "A1 Techno (126-128)", ["u1", "u2"]),
            PlaylistSpec("A · Techno", "A2", ["u2"]),
            PlaylistSpec("", "Sin grupo", ["u3"])]


def test_rekordbox_xml_structure():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "rekordbox.xml"
        write_rekordbox_xml(out, _tracks(), _specs(), "TRAKTOR ML test")
        root = ET.parse(out).getroot()
        coll = root.find("COLLECTION")
        assert coll.get("Entries") == "3" and len(coll) == 3
        ids = {t.get("TrackID") for t in coll}
        base = root.find("PLAYLISTS/NODE/NODE")
        assert base.get("Name") == "TRAKTOR ML test" and base.get("Count") == "2"
        playlists = [n for n in root.iter("NODE") if n.get("Type") == "1"]
        assert [p.get("Entries") for p in playlists] == ["2", "1", "1"]
        assert all(t.get("Key") in ids for p in playlists for t in p)
        first = coll[0]
        assert first.get("Comments") == "8A - Energy 6 - Vocal" and "AverageBpm" not in first.attrib


def test_traktor_nml_structure():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "traktor.nml"
        write_traktor_nml(out, _tracks(), _specs(), "TRAKTOR ML test")
        root = ET.parse(out).getroot()
        assert root.find("COLLECTION").get("ENTRIES") == "3"
        base = root.find("PLAYLISTS/NODE/SUBNODES/NODE")
        assert base.get("NAME") == "TRAKTOR ML test"
        keys = [pk.get("KEY") for pk in root.iter("PRIMARYKEY")]
        assert keys[0] == "C:/:VS/:traktor/:Música/:#1 BIBO/:PRO/:Milo 3/:A - One.mp3"
        assert len(keys) == 4
        assert base.find("SUBNODES").get("COUNT") == "2"


def test_m3u8_absolute_paths():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "x.m3u8"
        write_m3u8(out, _tracks(), ["u2", "u1"])
        lines = out.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "#EXTM3U"
        assert lines[1] == "#EXTINF:400,B & C - Two"
        assert lines[2].endswith(r"Parte 1 - House\B & C - Two.flac")
        assert lines[4].startswith("C:\\VS\\traktor\\Música\\#1 BIBO")


def test_bpm_key_from_catalog():
    cat = pd.DataFrame({"track_uid": ["a", "b"], "tag_bpm": [128.0, None],
                        "tag_camelot": ["8A", None], "mik_energy": [6, None]})
    t = bpm_key_from_catalog(cat)
    assert t.loc[0, "bpm"] == 128.0 and t.loc[0, "key"] == "8A" and t.loc[0, "energy"] == 6
    assert pd.isna(t.loc[1, "bpm"]) and pd.isna(t.loc[1, "key"]) and pd.isna(t.loc[1, "bpm_source"])


def test_order_starts_at_lowest_energy():
    emb = np.eye(3, dtype=np.float32)
    bpm = np.array([120.0, 125.0, 130.0])
    keys = ["8A", "8A", "8A"]
    energy = np.array([7.0, 4.0, np.nan])
    order = order_cluster_tracks([0, 1, 2], emb, bpm, keys, {"embedding": 0.5, "bpm": 0.3, "key": 0.2},
                                 energy=energy)
    assert order[0] == 1
    assert order_cluster_tracks([0, 1, 2], emb, bpm, keys)[0] == 0  # sin energía: menor BPM


def test_split_windows():
    seg = np.ones(25, dtype=np.float32)
    wins = split_windows(seg, 10)
    assert len(wins) == 3 and all(len(w) == 10 for w in wins)
    assert wins[2][:5].sum() == 5 and wins[2][5:].sum() == 0


def test_estimate_bpm_click_track():
    from src.v4.common.tempo import estimate_bpm
    sr, bpm = 22050, 126.0
    rng = np.random.default_rng(0)
    segs = []
    for _ in range(3):
        x = rng.standard_normal(sr * 30).astype(np.float32) * 0.01
        period = 60.0 / bpm
        for k in range(int(30 / period)):
            i = int(k * period * sr)
            x[i:i + 200] += np.hanning(200).astype(np.float32)
        segs.append(x)
    est = estimate_bpm(segs, sr)
    assert est is not None and abs(est - bpm) < 0.5


def test_vocal_summarize_rules():
    from src.v4.pipeline.tag_vocals import summarize
    probs = {"a": np.array([0.9, 0.1, 0.1]), "b": np.array([0.2, 0.2, 0.2])}
    by_cov = summarize(probs, threshold=0.5, min_coverage=0.34).set_index("track_uid")["is_vocal"]
    assert not by_cov["a"] and not by_cov["b"]
    by_mean = summarize(probs, 0.5, 0.34, mean_threshold=0.15).set_index("track_uid")["is_vocal"]
    assert by_mean["a"] and by_mean["b"]


def test_xml_declarations_use_double_quotes():
    with tempfile.TemporaryDirectory() as tmp:
        rb, nml = Path(tmp) / "r.xml", Path(tmp) / "t.nml"
        write_rekordbox_xml(rb, _tracks(), _specs(), "X")
        write_traktor_nml(nml, _tracks(), _specs(), "X")
        assert rb.read_text(encoding="utf-8").startswith('<?xml version="1.0" encoding="UTF-8"?>\n<DJ_PLAYLISTS')
        assert nml.read_text(encoding="utf-8").startswith('<?xml version="1.0" encoding="UTF-8" standalone="no" ?>\n<NML')


def test_xml_strips_control_chars():
    t = _tracks()
    t.loc["u1", "tag_comment"] = "bad\x1acomment"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "r.xml"
        write_rekordbox_xml(out, t, _specs(), "X")
        root = ET.parse(out).getroot()  # no debe fallar
        assert root.find("COLLECTION")[0].get("Comments") == "badcomment"
        nml = Path(tmp) / "t.nml"
        write_traktor_nml(nml, t, _specs(), "X")
        ET.parse(nml)
