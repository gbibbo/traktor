"""
PURPOSE: D3.4 1001Tracklists matching library. Metadata-only mode. Provides:
  - input template schema + creation for optional, manually provided
    1001Tracklists set metadata (no scraping, no credentials);
  - normalization of source-metadata names, version/remix/edit markers, and
    durations;
  - matching of manually provided set entries against local canonical decoded
    tracks, with a confidence policy (>= 0.7 accept, < 0.7 reject) and an
    ambiguous category for non-unique or marker-conflicting candidates;
  - setlist pair-weight assignment by in-set distance;
  - aggregate-only matching report generation (no private metadata values).

  1001Tracklists evidence is weak evidence and evaluation support only. It is
  never ground truth and never a clustering feature in Regime 1. When no usable
  source is supplied, matching reports `no_usable_source` and writes no matches.

CHANGELOG:
  D3.4 - Initial implementation (metadata-only matching; no-usable-source path).
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns of the manually filled 1001Tracklists input (ignored artifact).
INPUT_TEMPLATE_COLUMNS = (
    "set_id",
    "set_position",
    "track_artist",
    "track_title",
    "version_marker",
    "duration_seconds",
    "source_url",
)

# Columns of the ignored matches artifact (only written when a usable source
# exists).
MATCH_COLUMNS = (
    "set_id",
    "set_position",
    "matched_track_id",
    "confidence",
    "match_status",
    "match_reason",
)

# Confidence policy (plan D3.4).
CONFIDENCE_FINGERPRINT = 1.0
CONFIDENCE_METADATA = 0.7
CONFIDENCE_MIN_ACCEPT = 0.7

# Identity policy: duration agreement tolerance when fingerprinting is absent.
DURATION_TOLERANCE_SEC = 3.0

# Setlist pair-weighting policy (plan D3.4).
PAIR_WEIGHT_DISTANCE_1 = 1.0
PAIR_WEIGHT_DISTANCE_2_3 = 0.6
PAIR_WEIGHT_FARTHER = 0.2

# Recognized version/remix/edit markers (normalized, lowercase).
VERSION_MARKERS = frozenset(
    {
        "remix",
        "edit",
        "mix",
        "dub",
        "instrumental",
        "extended",
        "radio",
        "vip",
        "rework",
        "bootleg",
        "remaster",
        "rerub",
    }
)

# Status labels for matched / unmatched entries.
STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected"
STATUS_AMBIGUOUS = "ambiguous"

SOURCE_STATUS_NONE = "no_usable_source"
SOURCE_STATUS_PRESENT = "usable_source"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MatchResult:
    matches: pd.DataFrame
    source_status: str
    n_input_rows: int
    n_usable_rows: int
    n_inventory: int
    accepted_count: int
    rejected_count: int
    ambiguous_count: int


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_name(value: Optional[str]) -> str:
    """Normalize a source-metadata name: lowercase, strip punctuation, collapse
    whitespace. Returns an empty string for missing input."""
    if value is None:
        return ""
    s = str(value).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def extract_version_markers(value: Optional[str]) -> frozenset:
    """Return the set of recognized version/remix/edit markers in a string."""
    norm = normalize_name(value)
    if not norm:
        return frozenset()
    tokens = set(norm.split())
    return frozenset(tokens & VERSION_MARKERS)


def version_markers_compatible(
    markers_a: frozenset, markers_b: frozenset
) -> bool:
    """Two entries are version-compatible when their recognized marker sets are
    equal. An empty set denotes an original/unmarked version; two empty sets
    are compatible, and an empty set is not compatible with a non-empty one."""
    return set(markers_a) == set(markers_b)


def parse_duration(value: Optional[str]) -> Optional[float]:
    """Parse a duration given as seconds or ``mm:ss`` / ``hh:mm:ss``.
    Returns None when the value is missing or unparseable."""
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    if ":" in s:
        parts = s.split(":")
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            return None
        total = 0.0
        for n in nums:
            total = total * 60.0 + n
        return total
    try:
        return float(s)
    except ValueError:
        return None


def duration_within_tolerance(
    dur_a: Optional[float],
    dur_b: Optional[float],
    tolerance: float = DURATION_TOLERANCE_SEC,
) -> bool:
    """True when both durations are present and differ by <= tolerance seconds."""
    if dur_a is None or dur_b is None:
        return False
    return abs(dur_a - dur_b) <= tolerance


def setlist_pair_weight(distance: int) -> float:
    """Return the pair weight for two accepted tracks at a given in-set distance."""
    if distance == 1:
        return PAIR_WEIGHT_DISTANCE_1
    if distance in (2, 3):
        return PAIR_WEIGHT_DISTANCE_2_3
    return PAIR_WEIGHT_FARTHER


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def create_input_template(out_path: Path) -> int:
    """Write a blank 1001Tracklists input template (header only). Returns the
    number of data rows written (always 0 — the user fills it manually)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=list(INPUT_TEMPLATE_COLUMNS)).to_csv(out_path, index=False)
    return 0


def load_input(input_csv: Path) -> pd.DataFrame:
    """Load a manually provided 1001Tracklists input CSV."""
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"1001 input CSV not found: {input_csv}")
    return pd.read_csv(input_csv, dtype=str).fillna("")


def usable_input_rows(input_df: pd.DataFrame) -> pd.DataFrame:
    """Return rows that carry both a non-empty performer name and track name."""
    if input_df.empty:
        return input_df
    if "track_artist" not in input_df.columns or "track_title" not in input_df.columns:
        return input_df.iloc[0:0]
    has_performer = input_df["track_artist"].astype(str).str.strip() != ""
    has_name = input_df["track_title"].astype(str).str.strip() != ""
    return input_df[has_performer & has_name].copy()


def load_inventory_canonical(inventory_csv: Path) -> pd.DataFrame:
    """Load canonical decoded local tracks with normalized names and durations."""
    inventory_csv = Path(inventory_csv)
    if not inventory_csv.exists():
        raise FileNotFoundError(f"Inventory CSV not found: {inventory_csv}")
    df = pd.read_csv(inventory_csv, dtype=str).fillna("")
    canonical = df[
        df["is_canonical"].astype(str).str.lower().isin({"true", "1"})
        & (df["decode_status"].astype(str).str.lower() == "ok")
    ].copy()
    canonical["name_performer_norm"] = canonical["artist"].map(normalize_name)
    canonical["name_track_norm"] = canonical["title"].map(normalize_name)
    canonical["markers"] = canonical["title"].map(extract_version_markers)
    canonical["duration"] = canonical["duration_seconds"].map(parse_duration)
    return canonical


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _score_entry(
    entry: Dict[str, str],
    inventory_df: pd.DataFrame,
) -> Dict[str, object]:
    """Score one input entry against the canonical inventory.

    Metadata-only mode: fingerprinting is unavailable, so the maximum reachable
    confidence is the metadata tier (0.7). MERT cosine is never used as the sole
    identity rule. Returns a dict with matched_track_id, confidence, status,
    reason.
    """
    performer = normalize_name(entry.get("track_artist"))
    track = normalize_name(entry.get("track_title"))
    in_markers = extract_version_markers(entry.get("track_title")) | extract_version_markers(
        entry.get("version_marker")
    )
    in_duration = parse_duration(entry.get("duration_seconds"))

    name_hits = inventory_df[
        (inventory_df["name_performer_norm"] == performer)
        & (inventory_df["name_track_norm"] == track)
    ]

    if name_hits.empty:
        return {
            "matched_track_id": "",
            "confidence": 0.0,
            "match_status": STATUS_REJECTED,
            "match_reason": "no normalized name match",
        }

    # Apply duration + version-marker identity checks to each name hit.
    validated: List[str] = []
    for _, cand in name_hits.iterrows():
        marker_ok = version_markers_compatible(in_markers, cand["markers"])
        duration_ok = duration_within_tolerance(in_duration, cand["duration"])
        if marker_ok and duration_ok:
            validated.append(str(cand["track_id"]))

    if len(validated) == 1:
        return {
            "matched_track_id": validated[0],
            "confidence": CONFIDENCE_METADATA,
            "match_status": STATUS_ACCEPTED,
            "match_reason": "normalized name + duration + version markers",
        }

    if len(validated) > 1:
        return {
            "matched_track_id": "",
            "confidence": 0.0,
            "match_status": STATUS_AMBIGUOUS,
            "match_reason": "multiple validated candidates",
        }

    # Name matched but identity checks did not pass for any candidate.
    if len(name_hits) > 1:
        return {
            "matched_track_id": "",
            "confidence": 0.0,
            "match_status": STATUS_AMBIGUOUS,
            "match_reason": "non-unique name match, identity unconfirmed",
        }
    return {
        "matched_track_id": "",
        "confidence": 0.0,
        "match_status": STATUS_REJECTED,
        "match_reason": "duration or version markers below acceptance",
    }


def match_tracklists(
    inventory_df: pd.DataFrame,
    input_df: pd.DataFrame,
) -> MatchResult:
    """Match manually provided 1001Tracklists entries against canonical tracks.

    When no usable source rows are present, returns a MatchResult with
    source_status == no_usable_source and zero counts.
    """
    n_input = len(input_df)
    usable = usable_input_rows(input_df)
    n_usable = len(usable)
    n_inventory = len(inventory_df)

    if n_usable == 0:
        return MatchResult(
            matches=pd.DataFrame(columns=list(MATCH_COLUMNS)),
            source_status=SOURCE_STATUS_NONE,
            n_input_rows=n_input,
            n_usable_rows=0,
            n_inventory=n_inventory,
            accepted_count=0,
            rejected_count=0,
            ambiguous_count=0,
        )

    rows: List[Dict[str, object]] = []
    for _, entry in usable.iterrows():
        scored = _score_entry(entry.to_dict(), inventory_df)
        rows.append(
            {
                "set_id": entry.get("set_id", ""),
                "set_position": entry.get("set_position", ""),
                "matched_track_id": scored["matched_track_id"],
                "confidence": scored["confidence"],
                "match_status": scored["match_status"],
                "match_reason": scored["match_reason"],
            }
        )

    matches = pd.DataFrame(rows, columns=list(MATCH_COLUMNS))
    accepted = int((matches["match_status"] == STATUS_ACCEPTED).sum())
    rejected = int((matches["match_status"] == STATUS_REJECTED).sum())
    ambiguous = int((matches["match_status"] == STATUS_AMBIGUOUS).sum())

    return MatchResult(
        matches=matches,
        source_status=SOURCE_STATUS_PRESENT,
        n_input_rows=n_input,
        n_usable_rows=n_usable,
        n_inventory=n_inventory,
        accepted_count=accepted,
        rejected_count=rejected,
        ambiguous_count=ambiguous,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_match_report(result: MatchResult) -> str:
    """Build the committed matching report. Aggregate-only: counts and policy
    statements only, with no private source-metadata values."""
    if result.source_status == SOURCE_STATUS_NONE:
        status_line = (
            "No usable 1001Tracklists source was available. Matching produced "
            "no accepted, rejected, or ambiguous entries."
        )
    else:
        status_line = (
            "A 1001Tracklists source was provided and matched against local "
            "canonical decoded tracks."
        )

    lines = [
        "# 1001Tracklists Matching Report",
        "",
        "## Source Status",
        "",
        f"- Source status: `{result.source_status}`",
        f"- {status_line}",
        f"- Input rows received: {result.n_input_rows}",
        f"- Usable input rows: {result.n_usable_rows}",
        f"- Canonical decoded tracks available for matching: {result.n_inventory}",
        "",
        "## Match Counts",
        "",
        f"- Accepted: {result.accepted_count}",
        f"- Rejected: {result.rejected_count}",
        f"- Ambiguous: {result.ambiguous_count}",
        "",
        "## Identity And Confidence Policy",
        "",
        "- Audio fingerprinting is preferred for identity validation; it is not",
        "  available in this metadata-only run.",
        "- Without fingerprinting, acceptance requires a normalized name match",
        "  (performer and track name), a duration agreement within "
        f"{int(DURATION_TOLERANCE_SEC)} seconds,",
        "  and compatible version/remix/edit markers.",
        "- MERT cosine may be stored as supporting evidence only; it is never the",
        "  sole identity-acceptance rule.",
        f"- Accepted confidence is {CONFIDENCE_METADATA} for the metadata tier and "
        f"{CONFIDENCE_FINGERPRINT}",
        "  for strong audio-identity evidence; entries below "
        f"{CONFIDENCE_MIN_ACCEPT} are rejected.",
        "",
        "## Role And Decision Notes",
        "",
        "- 1001Tracklists evidence is weak evidence and evaluation support only.",
        "- It is never ground truth and never a clustering feature in Regime 1.",
        "- Per the operational plan, the workflow continues without 1001 evidence",
        "  when no usable source is available; Regime 1 is not blocked on it.",
        "",
        "_Last updated: D3.4_",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_matches(matches_df: pd.DataFrame, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    matches_df.to_csv(out_path, index=False)


def write_match_report(report_text: str, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_text, encoding="utf-8")
