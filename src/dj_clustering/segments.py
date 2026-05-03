"""
PURPOSE: Deterministic segment generation from track duration for the DJ clustering
         feature extraction pipeline. No audio I/O — uses duration_seconds from inventory.

CHANGELOG:
  D2.2 - Initial implementation.
"""

from __future__ import annotations

from typing import Dict, List, Literal

import pandas as pd

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

TrackClass = Literal["long", "medium", "short", "very_short"]

# ---------------------------------------------------------------------------
# Classification thresholds (mirrors features.yaml segment_policy)
# ---------------------------------------------------------------------------

_LONG_MIN = 150.0      # >= this → long
_MEDIUM_MIN = 75.0     # >= this and < long → medium
_SHORT_MIN = 35.0      # >= this and < medium → short
                       # < short → very_short

_LONG_SEG_DUR = 30.0
_LONG_EDGE_MARGIN = 30.0
_LONG_CENTER_FRACS = (0.25, 0.50, 0.75)

_MEDIUM_SEG_DUR = 60.0
_SHORT_SEG_DUR = 30.0

# Required columns in the segment manifest
SEGMENT_MANIFEST_COLUMNS = [
    "segment_id",
    "track_id",
    "segment_index",
    "start_sec",
    "end_sec",
    "duration_sec",
    "track_duration_sec",
    "track_class",
    "is_short_track",
    "num_segments",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_track(duration_sec: float) -> TrackClass:
    """Return the segment-policy class for a track of the given duration."""
    if duration_sec >= _LONG_MIN:
        return "long"
    if duration_sec >= _MEDIUM_MIN:
        return "medium"
    if duration_sec >= _SHORT_MIN:
        return "short"
    return "very_short"


def compute_segments(track_id: str, duration_sec: float) -> List[Dict]:
    """
    Return a list of segment dicts for one track.

    Segment IDs are of the form ``{track_id}_seg{i:02d}`` (0-indexed).
    All returned dicts have the keys in SEGMENT_MANIFEST_COLUMNS.
    """
    cls = classify_track(duration_sec)

    if cls == "long":
        rows = _long_segments(track_id, duration_sec)
    elif cls == "medium":
        rows = _centered_segments(track_id, duration_sec, _MEDIUM_SEG_DUR, 1)
    elif cls == "short":
        rows = _centered_segments(track_id, duration_sec, _SHORT_SEG_DUR, 1)
    else:  # very_short
        rows = _full_track_segment(track_id, duration_sec)

    return rows


def build_segment_manifest(inventory_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the segment manifest from a library inventory DataFrame.

    Requires columns: track_id, duration_seconds, is_canonical, decode_status.
    Returns a DataFrame with SEGMENT_MANIFEST_COLUMNS.
    """
    canonical = inventory_df[
        inventory_df["is_canonical"] & (inventory_df["decode_status"] != "failed")
    ].copy()

    rows = []
    for _, row in canonical.iterrows():
        rows.extend(compute_segments(str(row["track_id"]), float(row["duration_seconds"])))

    df = pd.DataFrame(rows, columns=SEGMENT_MANIFEST_COLUMNS)
    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_row(
    track_id: str,
    index: int,
    start_sec: float,
    end_sec: float,
    track_duration_sec: float,
    track_class: str,
    num_segments: int,
    is_short_track: bool = False,
) -> Dict:
    return {
        "segment_id": f"{track_id}_seg{index:02d}",
        "track_id": track_id,
        "segment_index": index,
        "start_sec": round(start_sec, 6),
        "end_sec": round(end_sec, 6),
        "duration_sec": round(end_sec - start_sec, 6),
        "track_duration_sec": track_duration_sec,
        "track_class": track_class,
        "is_short_track": is_short_track,
        "num_segments": num_segments,
    }


def _long_segments(track_id: str, duration_sec: float) -> List[Dict]:
    """Three 30 s segments near 25%, 50%, 75%, avoiding first/last 30 s."""
    seg_dur = _LONG_SEG_DUR
    margin = _LONG_EDGE_MARGIN
    rows = []
    for i, cf in enumerate(_LONG_CENTER_FRACS):
        center = cf * duration_sec
        # Clamp so segment fits inside [margin, duration - margin]
        start = max(margin, center - seg_dur / 2.0)
        start = min(start, duration_sec - margin - seg_dur)
        end = start + seg_dur
        rows.append(_make_row(track_id, i, start, end, duration_sec, "long", 3))
    return rows


def _centered_segments(
    track_id: str,
    duration_sec: float,
    seg_dur: float,
    num: int,
) -> List[Dict]:
    """One centered segment of seg_dur seconds (medium or short policy)."""
    cls = classify_track(duration_sec)
    start = max(0.0, (duration_sec - seg_dur) / 2.0)
    end = min(duration_sec, start + seg_dur)
    return [_make_row(track_id, 0, start, end, duration_sec, cls, num)]


def _full_track_segment(track_id: str, duration_sec: float) -> List[Dict]:
    """Full track as one segment (very_short policy)."""
    return [
        _make_row(
            track_id, 0, 0.0, duration_sec, duration_sec,
            "very_short", 1, is_short_track=True,
        )
    ]
