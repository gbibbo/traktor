"""
PURPOSE: Tests de src/v4/pipeline/phase1_tags.py --traktor: BPM y tonalidad que falten en los tags, desde
         la colección de Traktor (collection.nml), emparejando por nombre de archivo y duración.
CHANGELOG:
  - 2026-11-07: Creación.
"""
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.pipeline.phase1_tags import bpm_key_from_catalog, fill_from_traktor  # noqa: E402


def _nml(path: Path, entries) -> Path:
    """entries: (archivo, BPM o None, MUSICAL_KEY o None, duración en s)."""
    body = []
    for name, bpm, key, dur in entries:
        body.append(f'<ENTRY TITLE="x"><LOCATION DIR="/:Music/:" FILE="{name}" VOLUME="C:"/>'
                    f'<INFO PLAYTIME="{int(dur)}" PLAYTIME_FLOAT="{dur}"/>'
                    + (f'<TEMPO BPM="{bpm}" BPM_QUALITY="100.000000"/>' if bpm else "")
                    + (f'<MUSICAL_KEY VALUE="{key}"/>' if key is not None else "") + "</ENTRY>")
    path.write_text('<?xml version="1.0" encoding="UTF-8" standalone="no" ?><NML VERSION="20">'
                    f'<COLLECTION ENTRIES="{len(entries)}">{"".join(body)}</COLLECTION></NML>', encoding="utf-8")
    return path


def _catalog(rows) -> pd.DataFrame:
    """rows: (nombre, duración, BPM de los tags, Camelot de los tags)."""
    return pd.DataFrame([{"track_uid": f"u{i}", "source_path": f"C:/Música/Nueva/{name}", "duration_s": dur,
                          "tag_bpm": bpm, "tag_camelot": key, "mik_energy": None}
                         for i, (name, dur, bpm, key) in enumerate(rows)])


def test_fills_only_what_the_tags_lack_and_only_with_one_match(tmp_path):
    catalog = _catalog([
        ("Nuevo.mp3", 300.5, None, None),          # 0: Traktor lo analizó
        ("ConTags.mp3", 300.0, 125.0, "5A"),       # 1: los tags mandan
        ("Repetido.mp3", 300.0, None, None),       # 2: dos entradas posibles: no se adivina
        ("OtraVersion.mp3", 360.0, None, None),    # 3: mismo nombre, otra duración
        ("Dos.mp3", 240.0, None, None),            # 4: una de dos entradas tiene su duración
        ("SinAnalizar.mp3", 200.0, None, None),    # 5: Traktor no lo analizó
        ("MAYUS.MP3", 280.0, None, "3A"),          # 6: nombre en otra caja; la tonalidad ya está
    ])
    nml = _nml(tmp_path / "collection.nml", [
        ("Nuevo.mp3", 128.0, 21, 300.0),           # 21 = La menor = 8A
        ("ConTags.mp3", 130.0, 0, 300.0),
        ("Repetido.mp3", 126.0, 1, 300.0), ("Repetido.mp3", 127.0, 2, 301.0),
        ("OtraVersion.mp3", 124.0, 3, 200.0),
        ("Dos.mp3", 122.0, 12, 240.4), ("Dos.mp3", 123.0, 13, 420.0),
        ("SinAnalizar.mp3", None, None, 200.0),
        ("mayus.mp3", 131.0, 5, 280.0),
    ])
    table = bpm_key_from_catalog(catalog)
    assert fill_from_traktor(table, catalog, nml) == (3, 2)
    t = table.set_index("track_uid")
    assert (t.at["u0", "bpm"], t.at["u0", "key"], t.at["u0", "bpm_source"], t.at["u0", "key_source"]) == \
        (128.0, "8A", "traktor", "traktor")
    assert (t.at["u1", "bpm"], t.at["u1", "key"], t.at["u1", "bpm_source"]) == (125.0, "5A", "tags")
    for uid in ("u2", "u3", "u5"):
        assert pd.isna(t.at[uid, "bpm"]) and pd.isna(t.at[uid, "key"])
    assert (t.at["u4", "bpm"], t.at["u4", "key"]) == (122.0, "5A")         # 12 = Do menor = 5A
    assert (t.at["u6", "bpm"], t.at["u6", "key"], t.at["u6", "key_source"]) == (131.0, "3A", "tags")
