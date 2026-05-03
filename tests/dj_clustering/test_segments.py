"""
PURPOSE: Unit tests for src/dj_clustering/segments.py — segment policy, IDs,
         edge margins, and manifest schema. Pure Python + numpy; no audio or GPU.

CHANGELOG:
  D2.2 - Initial implementation.
"""

import re
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dj_clustering.segments import (
    SEGMENT_MANIFEST_COLUMNS,
    build_segment_manifest,
    classify_track,
    compute_segments,
)


# ---------------------------------------------------------------------------
# classify_track
# ---------------------------------------------------------------------------

def test_classify_long():
    assert classify_track(300.0) == "long"
    assert classify_track(150.0) == "long"


def test_classify_medium():
    assert classify_track(100.0) == "medium"
    assert classify_track(75.0) == "medium"
    assert classify_track(149.9) == "medium"


def test_classify_short():
    assert classify_track(50.0) == "short"
    assert classify_track(35.0) == "short"
    assert classify_track(74.9) == "short"


def test_classify_very_short():
    assert classify_track(20.0) == "very_short"
    assert classify_track(34.9) == "very_short"
    assert classify_track(0.1) == "very_short"


# ---------------------------------------------------------------------------
# Long-track segment policy
# ---------------------------------------------------------------------------

def test_long_track_three_segments():
    segs = compute_segments("abc123", 300.0)
    assert len(segs) == 3
    for seg in segs:
        assert seg["track_class"] == "long"
        assert seg["num_segments"] == 3
        assert pytest.approx(seg["duration_sec"], abs=1e-4) == 30.0


def test_long_track_boundary_150s():
    segs = compute_segments("abc123", 150.0)
    assert len(segs) == 3
    for seg in segs:
        assert seg["start_sec"] >= 30.0 - 1e-6, "start violates edge margin"
        assert seg["end_sec"] <= 120.0 + 1e-6, "end violates edge margin"
        assert pytest.approx(seg["duration_sec"], abs=1e-4) == 30.0


def test_long_track_edge_margins_respected():
    # All long tracks must keep segments inside [30s, duration-30s]
    for dur in [150.0, 200.0, 300.0, 500.0, 760.0]:
        segs = compute_segments("tid", dur)
        for seg in segs:
            assert seg["start_sec"] >= 30.0 - 1e-6, f"dur={dur}: start {seg['start_sec']} < 30s"
            assert seg["end_sec"] <= dur - 30.0 + 1e-6, f"dur={dur}: end {seg['end_sec']} > dur-30s"


def test_long_track_segment_durations_are_30s():
    segs = compute_segments("tid", 400.0)
    for seg in segs:
        assert pytest.approx(seg["end_sec"] - seg["start_sec"], abs=1e-4) == 30.0


# ---------------------------------------------------------------------------
# Medium-track segment policy
# ---------------------------------------------------------------------------

def test_medium_track_one_segment():
    segs = compute_segments("tid", 100.0)
    assert len(segs) == 1
    assert segs[0]["track_class"] == "medium"
    assert segs[0]["num_segments"] == 1
    assert pytest.approx(segs[0]["duration_sec"], abs=1e-4) == 60.0


def test_medium_track_centered():
    dur = 100.0
    segs = compute_segments("tid", dur)
    mid = (segs[0]["start_sec"] + segs[0]["end_sec"]) / 2.0
    assert pytest.approx(mid, abs=0.1) == dur / 2.0


def test_medium_track_at_75s():
    segs = compute_segments("tid", 75.0)
    assert len(segs) == 1
    assert segs[0]["start_sec"] >= 0.0
    assert segs[0]["end_sec"] <= 75.0


# ---------------------------------------------------------------------------
# Short-track segment policy
# ---------------------------------------------------------------------------

def test_short_track_one_segment():
    segs = compute_segments("tid", 50.0)
    assert len(segs) == 1
    assert segs[0]["track_class"] == "short"
    assert pytest.approx(segs[0]["duration_sec"], abs=1e-4) == 30.0


def test_short_track_centered():
    dur = 60.0
    segs = compute_segments("tid", dur)
    mid = (segs[0]["start_sec"] + segs[0]["end_sec"]) / 2.0
    assert pytest.approx(mid, abs=0.1) == dur / 2.0


# ---------------------------------------------------------------------------
# Very-short-track segment policy
# ---------------------------------------------------------------------------

def test_very_short_track_full():
    dur = 20.0
    segs = compute_segments("tid", dur)
    assert len(segs) == 1
    assert segs[0]["start_sec"] == pytest.approx(0.0)
    assert segs[0]["end_sec"] == pytest.approx(dur)
    assert segs[0]["is_short_track"] is True
    assert segs[0]["track_class"] == "very_short"


def test_very_short_track_is_short_track_flag():
    segs = compute_segments("tid", 10.0)
    assert segs[0]["is_short_track"] is True


def test_non_very_short_not_flagged():
    for dur in [150.0, 100.0, 50.0]:
        segs = compute_segments("tid", dur)
        for seg in segs:
            assert seg["is_short_track"] is False


# ---------------------------------------------------------------------------
# Segment IDs — determinism and format
# ---------------------------------------------------------------------------

def test_segment_ids_deterministic():
    segs1 = compute_segments("aabbcc112233", 300.0)
    segs2 = compute_segments("aabbcc112233", 300.0)
    assert [s["segment_id"] for s in segs1] == [s["segment_id"] for s in segs2]


def test_segment_id_format():
    pattern = re.compile(r"^[^_]+_seg\d{2}$")
    for dur in [300.0, 100.0, 50.0, 20.0]:
        segs = compute_segments("abcdef123456", dur)
        for seg in segs:
            assert pattern.match(seg["segment_id"]), (
                f"Unexpected segment_id format: {seg['segment_id']!r}"
            )


def test_segment_index_is_zero_based():
    segs = compute_segments("tid", 300.0)
    for i, seg in enumerate(segs):
        assert seg["segment_index"] == i


def test_segment_ids_unique_per_track():
    segs = compute_segments("tid", 300.0)
    ids = [s["segment_id"] for s in segs]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# build_segment_manifest
# ---------------------------------------------------------------------------

def _make_inventory(rows: list) -> pd.DataFrame:
    defaults = {
        "is_canonical": True,
        "decode_status": "ok",
    }
    full_rows = [{**defaults, **r} for r in rows]
    return pd.DataFrame(full_rows)


def test_segment_manifest_schema():
    inv = _make_inventory([
        {"track_id": "aaa", "duration_seconds": 300.0},
        {"track_id": "bbb", "duration_seconds": 100.0},
    ])
    df = build_segment_manifest(inv)
    for col in SEGMENT_MANIFEST_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"


def test_segment_manifest_excludes_non_canonical():
    inv = _make_inventory([
        {"track_id": "aaa", "duration_seconds": 300.0, "is_canonical": True},
        {"track_id": "bbb", "duration_seconds": 300.0, "is_canonical": False},
    ])
    df = build_segment_manifest(inv)
    assert "bbb" not in df["track_id"].values


def test_segment_manifest_excludes_failed_decode():
    inv = _make_inventory([
        {"track_id": "aaa", "duration_seconds": 300.0, "decode_status": "ok"},
        {"track_id": "bbb", "duration_seconds": 300.0, "decode_status": "failed"},
    ])
    df = build_segment_manifest(inv)
    assert "bbb" not in df["track_id"].values


def test_segment_manifest_row_count():
    # 2 long tracks → 6 segments
    inv = _make_inventory([
        {"track_id": "aaa", "duration_seconds": 300.0},
        {"track_id": "bbb", "duration_seconds": 200.0},
    ])
    df = build_segment_manifest(inv)
    assert len(df) == 6


def test_segment_manifest_segment_ids_unique():
    inv = _make_inventory([
        {"track_id": "aaa", "duration_seconds": 300.0},
        {"track_id": "bbb", "duration_seconds": 200.0},
    ])
    df = build_segment_manifest(inv)
    assert df["segment_id"].nunique() == len(df)
