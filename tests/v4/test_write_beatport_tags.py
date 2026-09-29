"""
PURPOSE: Tests de src/v4/pipeline/write_beatport_tags.py: plan por tema según las reglas del
         2026-09-29 (Beatport en A y B confirmado, modelo solo con confianza, B sin confirmar sin
         tocar, Genre vacío solo con la opción, WAV/AIFF fuera) y escritura real en MP3 (ID3 2.3) y
         FLAC hechos con soundfile: valores releídos, hash del audio sin cambios y reversión.
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

from src.v4.common.tags import audio_payload_hash  # noqa: E402
from src.v4.pipeline.write_beatport_tags import (  # noqa: E402
    FIELDS, apply, build_plan, read_fields, revert, simulate, write_fields,
)


def bp_row(uid, level, confirmed, **new):
    row = {"track_uid": uid, "rel_path": f"x/{uid}.mp3", "level": level, "confirmed": confirmed}
    row.update({f"new_{f}": new.get(f) for f in FIELDS})
    return row


def frames(tmp_path=None, suffix=".mp3"):
    bp = pd.DataFrame([
        bp_row("a", "A", True, artist="Patrice Baumel", remixers="Adana Twins Remix", label="Watergate Records",
               genre="Melodic House & Techno", released="2018-11-12"),
        bp_row("b1", "B", True, artist="Oxia", remixers="Original Mix", label=None, genre="Techno", released=None),
        bp_row("b0", "B", False, artist="Oasis", genre="Rock"),
        bp_row("c", "C", False), bp_row("d", "D", False), bp_row("g", "D", False), bp_row("n", "D", False),
        bp_row("w", "A", True, genre="House"),
    ])
    pred = pd.DataFrame([
        {"track_uid": "c", "model_genre": "Deep House", "model_source": "classifier", "model_conf": 0.79},
        {"track_uid": "d", "model_genre": "House", "model_source": "classifier", "model_conf": 0.41},
        {"track_uid": "g", "model_genre": "Pop", "model_source": "gate_discogs", "model_conf": np.nan},
    ])
    paths = {u: f"C:/m/{u}{'.wav' if u == 'w' else suffix}" for u in bp["track_uid"]}
    cat = pd.DataFrame({"track_uid": list(paths), "source_path": list(paths.values())})
    return bp, pred, cat


def test_build_plan_rules():
    bp, pred, cat = frames()
    plan = build_plan(bp, pred, cat).set_index("track_uid")
    assert plan.loc["a", "action"] == "beatport" and plan.loc["a", "new_remixers"] == "Adana Twins Remix"
    # un campo que Beatport no trae no se toca
    assert plan.loc["b1", "action"] == "beatport" and pd.isna(plan.loc["b1", "new_label"])
    assert plan.loc["b0", "action"] == "keep" and pd.isna(plan.loc["b0", "new_genre"])
    assert plan.loc["c", "action"] == "model" and plan.loc["c", "new_genre"] == "Deep House"
    assert pd.isna(plan.loc["c", "new_artist"])
    for uid in ("d", "g", "n"):   # confianza baja, gate sin validar, sin representaciones
        assert plan.loc[uid, "action"] == "keep" and pd.isna(plan.loc[uid, "new_genre"])
    assert plan.loc["w", "action"] == "skip_format"


def test_build_plan_clear_unreliable():
    bp, pred, cat = frames()
    plan = build_plan(bp, pred, cat, clear_unreliable=True).set_index("track_uid")
    for uid in ("d", "g", "n"):
        assert plan.loc[uid, "action"] == "clear_genre" and plan.loc[uid, "new_genre"] == ""
    assert plan.loc["b0", "action"] == "keep"


def _audio(path: Path, fmt: str) -> bool:
    import soundfile as sf
    data = (np.random.default_rng(0).standard_normal((44100, 2)) * 0.1).astype(np.float32)
    try:
        sf.write(str(path), data, 44100, format=fmt)
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.parametrize("suffix,fmt", [(".mp3", "MP3"), (".flac", "FLAC")])
def test_write_verify_and_revert(tmp_path, suffix, fmt):
    p = tmp_path / f"a{suffix}"
    if not _audio(p, fmt):
        pytest.skip(f"soundfile no puede escribir {fmt}")
    write_fields(p, {"artist": ["Patrice Baumel, Adana Twins"], "genre": ["Techno"], "released": ["2018"]})
    if suffix == ".mp3":   # como los archivos de Beatport: ID3 2.3
        from mutagen.id3 import ID3
        t = ID3(str(p))
        t.save(v2_version=3)
    uid = audio_payload_hash(p)
    bp, pred, cat = frames(suffix=suffix)
    cat.loc[cat.track_uid == "a", "source_path"] = str(p)
    bp.loc[bp.track_uid == "a", "track_uid"] = uid
    cat.loc[cat.source_path == str(p), "track_uid"] = uid
    plan = build_plan(bp, pred, cat)
    plan = plan[plan.track_uid == uid]
    sim = simulate(plan)
    assert set(sim.iloc[0]["changes"].split(", ")) == {"artist", "remixers", "label", "genre", "released"}
    log = apply(plan, tmp_path / "backup.csv")
    assert log.iloc[0]["status"] == "written" and bool(log.iloc[0]["uid_matches_catalog"])
    now = read_fields(p)
    assert now["artist"] == ["Patrice Baumel"] and now["remixers"] == ["Adana Twins Remix"]
    assert now["label"] == ["Watergate Records"] and now["released"] == ["2018-11-12"]
    assert audio_payload_hash(p) == uid
    # segunda pasada: nada que cambiar
    assert apply(plan, tmp_path / "backup2.csv").iloc[0]["status"] == "unchanged"
    assert revert(tmp_path / "backup.csv") == 1
    back = read_fields(p)
    assert back["artist"] == ["Patrice Baumel, Adana Twins"] and back["genre"] == ["Techno"]
    assert back["remixers"] is None and back["label"] is None and back["released"] == ["2018"]
    assert json.loads(log.iloc[0]["old"])["remixers"] is None
