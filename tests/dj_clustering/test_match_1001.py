"""
PURPOSE: Unit tests for src/dj_clustering/match_1001.py — D3.4 1001Tracklists
         metadata-only matching: name/marker/duration normalization, confidence
         policy, accepted/rejected/ambiguous categorization, setlist pair
         weighting, the no-usable-source path, and report privacy. Pure Python;
         no audio, no GPU, no network.

CHANGELOG:
  D3.4 - Initial implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dj_clustering.match_1001 import (
    CONFIDENCE_METADATA,
    INPUT_TEMPLATE_COLUMNS,
    MATCH_COLUMNS,
    SOURCE_STATUS_NONE,
    SOURCE_STATUS_PRESENT,
    STATUS_ACCEPTED,
    STATUS_AMBIGUOUS,
    STATUS_REJECTED,
    build_match_report,
    create_input_template,
    duration_within_tolerance,
    extract_version_markers,
    match_tracklists,
    normalize_name,
    parse_duration,
    setlist_pair_weight,
    usable_input_rows,
    version_markers_compatible,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _inventory(rows):
    """Build a processed-inventory DataFrame (as load_inventory_canonical emits)."""
    recs = []
    for tid, performer, track, dur in rows:
        recs.append(
            {
                "track_id": tid,
                "name_performer_norm": normalize_name(performer),
                "name_track_norm": normalize_name(track),
                "markers": extract_version_markers(track),
                "duration": dur,
            }
        )
    return pd.DataFrame(recs)


def _input(rows):
    """Build a 1001Tracklists input DataFrame."""
    recs = []
    for i, (perf, track, marker, dur) in enumerate(rows, start=1):
        recs.append(
            {
                "set_id": "S1",
                "set_position": str(i),
                "track_artist": perf,
                "track_title": track,
                "version_marker": marker,
                "duration_seconds": dur,
                "source_url": "",
            }
        )
    return pd.DataFrame(recs, columns=list(INPUT_TEMPLATE_COLUMNS))


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_normalize_name_basic():
    assert normalize_name("  The Artist!  ") == "the artist"
    assert normalize_name("A/B - C") == "a b c"
    assert normalize_name(None) == ""


def test_extract_version_markers():
    assert extract_version_markers("Track (Extended Mix)") == frozenset({"extended", "mix"})
    assert extract_version_markers("Plain Track") == frozenset()


def test_version_markers_compatible():
    assert version_markers_compatible(frozenset(), frozenset())
    assert version_markers_compatible(frozenset({"remix"}), frozenset({"remix"}))
    assert not version_markers_compatible(frozenset(), frozenset({"remix"}))


def test_parse_duration():
    assert parse_duration("184") == 184.0
    assert parse_duration("3:04") == 184.0
    assert parse_duration("1:02:03") == 3723.0
    assert parse_duration("") is None
    assert parse_duration("abc") is None
    assert parse_duration(None) is None


def test_duration_within_tolerance():
    assert duration_within_tolerance(180.0, 182.0)
    assert duration_within_tolerance(180.0, 183.0)
    assert not duration_within_tolerance(180.0, 184.0)
    assert not duration_within_tolerance(180.0, None)


def test_setlist_pair_weight():
    assert setlist_pair_weight(1) == 1.0
    assert setlist_pair_weight(2) == 0.6
    assert setlist_pair_weight(3) == 0.6
    assert setlist_pair_weight(5) == 0.2


# ---------------------------------------------------------------------------
# Input template
# ---------------------------------------------------------------------------


def test_create_input_template_header_only(tmp_path):
    out = tmp_path / "input.csv"
    n = create_input_template(out)
    assert n == 0
    df = pd.read_csv(out, dtype=str)
    assert list(df.columns) == list(INPUT_TEMPLATE_COLUMNS)
    assert len(df) == 0


def test_usable_input_rows():
    df = _input(
        [
            ("Performer A", "Track A", "", "180"),
            ("", "Track B", "", "200"),
            ("Performer C", "", "", "210"),
        ]
    )
    usable = usable_input_rows(df)
    assert len(usable) == 1


# ---------------------------------------------------------------------------
# No-usable-source path
# ---------------------------------------------------------------------------


def test_match_no_usable_source_empty():
    inv = _inventory([("t1", "Performer A", "Track A", 180.0)])
    empty = pd.DataFrame(columns=list(INPUT_TEMPLATE_COLUMNS))
    result = match_tracklists(inv, empty)
    assert result.source_status == SOURCE_STATUS_NONE
    assert result.accepted_count == 0
    assert result.rejected_count == 0
    assert result.ambiguous_count == 0
    assert result.matches.empty
    assert list(result.matches.columns) == list(MATCH_COLUMNS)


def test_match_no_usable_source_blank_rows():
    inv = _inventory([("t1", "Performer A", "Track A", 180.0)])
    df = _input([("", "", "", "")])
    result = match_tracklists(inv, df)
    assert result.source_status == SOURCE_STATUS_NONE


# ---------------------------------------------------------------------------
# Accepted / rejected / ambiguous
# ---------------------------------------------------------------------------


def test_match_accepted_metadata_tier():
    inv = _inventory([("t1", "Performer A", "Track A", 180.0)])
    df = _input([("Performer A", "Track A", "", "181")])
    result = match_tracklists(inv, df)
    assert result.source_status == SOURCE_STATUS_PRESENT
    assert result.accepted_count == 1
    row = result.matches.iloc[0]
    assert row["match_status"] == STATUS_ACCEPTED
    assert row["confidence"] == CONFIDENCE_METADATA
    assert row["matched_track_id"] == "t1"


def test_match_rejected_no_name():
    inv = _inventory([("t1", "Performer A", "Track A", 180.0)])
    df = _input([("Performer Z", "Track Z", "", "180")])
    result = match_tracklists(inv, df)
    assert result.rejected_count == 1
    assert result.matches.iloc[0]["match_status"] == STATUS_REJECTED


def test_match_rejected_duration_out_of_tolerance():
    inv = _inventory([("t1", "Performer A", "Track A", 180.0)])
    df = _input([("Performer A", "Track A", "", "200")])
    result = match_tracklists(inv, df)
    assert result.rejected_count == 1
    assert result.matches.iloc[0]["match_status"] == STATUS_REJECTED


def test_match_rejected_marker_conflict():
    inv = _inventory([("t1", "Performer A", "Track A", 180.0)])
    df = _input([("Performer A", "Track A", "Remix", "181")])
    result = match_tracklists(inv, df)
    assert result.rejected_count == 1
    assert result.matches.iloc[0]["match_status"] == STATUS_REJECTED


def test_match_ambiguous_duplicate_name():
    inv = _inventory(
        [
            ("t1", "Performer A", "Track A", 180.0),
            ("t2", "Performer A", "Track A", 181.0),
        ]
    )
    df = _input([("Performer A", "Track A", "", "180")])
    result = match_tracklists(inv, df)
    assert result.ambiguous_count == 1
    assert result.matches.iloc[0]["match_status"] == STATUS_AMBIGUOUS


def test_match_confidence_below_accept_not_accepted():
    inv = _inventory([("t1", "Performer A", "Track A", 180.0)])
    df = _input([("Performer A", "Track A", "", "")])  # no duration → cannot validate
    result = match_tracklists(inv, df)
    assert result.accepted_count == 0
    assert all(c < CONFIDENCE_METADATA for c in result.matches["confidence"])


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_report_no_source_statement():
    inv = _inventory([("t1", "Performer A", "Track A", 180.0)])
    result = match_tracklists(inv, pd.DataFrame(columns=list(INPUT_TEMPLATE_COLUMNS)))
    text = build_match_report(result)
    assert SOURCE_STATUS_NONE in text
    assert "Accepted: 0" in text
    assert "Rejected: 0" in text
    assert "Ambiguous: 0" in text


def test_report_privacy_aggregate_only():
    inv = _inventory([("t1", "Performer A", "Track A", 180.0)])
    result = match_tracklists(inv, pd.DataFrame(columns=list(INPUT_TEMPLATE_COLUMNS)))
    text = build_match_report(result)
    lower = text.lower()
    for pat in ["/mnt/fast", "nobackup", "scratch4weeks", "gb0048", "raw_audio"]:
        assert pat not in text, f"infrastructure pattern in report: {pat}"
    for field in [
        "file_path",
        "file_name",
        "folder_path",
        "folder_hint",
        "artist",
        "title",
        "album",
        "label",
    ]:
        assert field not in lower, f"forbidden token in report: {field}"
