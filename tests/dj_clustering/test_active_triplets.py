"""
PURPOSE: Unit tests for the D4.2 active-triplet selection functions in
         src/dj_clustering/triplets.py — top-3 leaderboard config selection
         with the documented accuracy/config_id tie-break and diagnostic-row
         exclusion, per-config rank differences, disagreement detection,
         disagreement_margin, rank-difference variance ranking, de-duplication
         against an already-asked queue, audio-hash-collision exclusion, and
         active-set assembly with disagreement / boundary-fill sources.
         Pure synthetic fixtures; no audio, no GPU, no private data.

CHANGELOG:
  D4.2a - Initial implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dj_clustering.triplets import (
    ACTIVE_BOUNDARY_FILL_SOURCE,
    ACTIVE_DISAGREEMENT_SOURCE,
    build_active_triplet_list,
    candidate_votes,
    config_rank_difference,
    deduplicate_triplets,
    disagreement_margin,
    exclude_existing_triplets,
    existing_triplet_keys,
    is_disagreement,
    rank_active_candidates,
    rank_diff_variance,
    select_top3_configs,
)


# --- top-3 config selection -------------------------------------------------


def _leaderboard():
    return pd.DataFrame(
        [
            {"config_id": "cfg_z", "kind": "grid", "status": "ok",
             "executable": True, "triplet_accuracy": 0.62},
            {"config_id": "cfg_a", "kind": "baseline", "status": "ok",
             "executable": True, "triplet_accuracy": 0.62},
            {"config_id": "cfg_m", "kind": "grid", "status": "ok",
             "executable": True, "triplet_accuracy": 0.62},
            {"config_id": "cfg_low", "kind": "grid", "status": "ok",
             "executable": True, "triplet_accuracy": 0.40},
            {"config_id": "diagnostic.V4", "kind": "diagnostic",
             "status": "diagnostic_only", "executable": False,
             "triplet_accuracy": np.nan},
        ]
    )


def test_select_top3_tie_break_by_config_id():
    top3 = select_top3_configs(_leaderboard(), n=3)
    # all three 0.62 configs, ordered by config_id ascending
    assert [c["config_id"] for c in top3] == ["cfg_a", "cfg_m", "cfg_z"]


def test_select_top3_excludes_diagnostic_rows():
    top3 = select_top3_configs(_leaderboard(), n=3)
    assert all(c["config_id"] != "diagnostic.V4" for c in top3)
    assert all(c["kind"] != "diagnostic" for c in top3)


def test_select_top3_prefers_higher_accuracy():
    top3 = select_top3_configs(_leaderboard(), n=4)
    # cfg_low (0.40) ranks last among the four executable configs
    assert top3[-1]["config_id"] == "cfg_low"


# --- disagreement detection / scoring ---------------------------------------


def test_disagreement_2_1_split_is_disagreement():
    # two configs choose B (rd>0), one chooses C (rd<0)
    rank_diffs = [0.5, 0.3, -0.4]
    assert is_disagreement(rank_diffs) is True
    votes = candidate_votes(rank_diffs)
    assert votes["count_b"] == 2 and votes["count_c"] == 1


def test_disagreement_3_0_split_is_not_disagreement():
    rank_diffs = [0.5, 0.3, 0.1]  # all choose B
    assert is_disagreement(rank_diffs) is False


def test_disagreement_margin_computation():
    # 2-1 split over 3 configs -> min(2,1)/3
    assert disagreement_margin([0.5, 0.3, -0.4]) == 1 / 3
    # unanimous -> margin 0
    assert disagreement_margin([0.5, 0.3, 0.1]) == 0.0


def test_rank_diff_variance_tie_break_orders_candidates():
    # both 2-1 splits (equal margin); higher rank-diff variance ranks first
    low_var = {"id": "low", "rank_diffs": [0.10, 0.10, -0.10]}
    high_var = {"id": "high", "rank_diffs": [3.0, 0.05, -2.5]}
    ranked = rank_active_candidates([low_var, high_var])
    assert [c["id"] for c in ranked] == ["high", "low"]
    assert rank_diff_variance(high_var["rank_diffs"]) > rank_diff_variance(
        low_var["rank_diffs"]
    )


def test_rank_active_candidates_drops_non_disagreements():
    cands = [
        {"id": "agree", "rank_diffs": [0.5, 0.4, 0.3]},
        {"id": "split", "rank_diffs": [0.5, -0.4, 0.3]},
    ]
    ranked = rank_active_candidates(cands)
    assert [c["id"] for c in ranked] == ["split"]


# --- per-config rank difference ---------------------------------------------


def test_config_rank_difference_sign():
    # anchor at origin; B near, C far -> d(A,C)-d(A,B) > 0 -> chooses B
    matrix = np.array([[0.0, 0.0], [0.1, 0.0], [9.0, 0.0]])
    index = {"anchor": 0, "near": 1, "far": 2}
    rd = config_rank_difference(matrix, index, "anchor", "near", "far")
    assert rd is not None and rd > 0


def test_config_rank_difference_missing_track_returns_none():
    matrix = np.array([[0.0, 0.0], [0.1, 0.0]])
    index = {"anchor": 0, "near": 1}
    assert config_rank_difference(matrix, index, "anchor", "near", "absent") is None


# --- de-duplication ---------------------------------------------------------


def _queue_df(rows):
    return pd.DataFrame(
        rows,
        columns=["anchor_track_id", "candidate_b_track_id", "candidate_c_track_id"],
    )


def test_exclude_existing_triplets_against_queue():
    existing = _queue_df([("a", "b", "c")])
    keys = existing_triplet_keys(existing)
    triplets = [
        # same anchor, same unordered {b,c} (order swapped) -> excluded
        {"anchor_track_id": "a", "candidate_b_track_id": "c",
         "candidate_c_track_id": "b"},
        # genuinely new -> kept
        {"anchor_track_id": "a", "candidate_b_track_id": "d",
         "candidate_c_track_id": "e"},
    ]
    kept = exclude_existing_triplets(triplets, keys)
    assert len(kept) == 1
    assert kept[0]["candidate_b_track_id"] == "d"


def test_deduplicate_within_set_and_hash_collisions():
    inventory = pd.DataFrame(
        {
            "track_id": ["a", "b", "c", "d"],
            "audio_content_hash": ["h1", "h2", "h3", "h2"],  # b and d collide
        }
    )
    triplets = [
        {"anchor_track_id": "a", "candidate_b_track_id": "b",
         "candidate_c_track_id": "c"},
        # exact duplicate of the first
        {"anchor_track_id": "a", "candidate_b_track_id": "c",
         "candidate_c_track_id": "b"},
        # hash collision between b and d
        {"anchor_track_id": "a", "candidate_b_track_id": "b",
         "candidate_c_track_id": "d"},
    ]
    out = deduplicate_triplets(triplets, inventory)
    assert len(out) == 1


# --- active-set assembly ----------------------------------------------------


def test_build_active_triplet_list_disagreement_first_then_fill():
    disagreement = [
        {"anchor_track_id": "a", "candidate_b_track_id": "b",
         "candidate_c_track_id": "c"},
        {"anchor_track_id": "d", "candidate_b_track_id": "e",
         "candidate_c_track_id": "f"},
    ]
    boundary = [
        {"anchor_track_id": "g", "candidate_b_track_id": "h",
         "candidate_c_track_id": "i"},
        {"anchor_track_id": "j", "candidate_b_track_id": "k",
         "candidate_c_track_id": "l"},
    ]
    out = build_active_triplet_list(disagreement, boundary, n_questions=3)
    assert len(out) == 3
    sources = [t["selection_source"] for t in out]
    assert sources == [
        ACTIVE_DISAGREEMENT_SOURCE,
        ACTIVE_DISAGREEMENT_SOURCE,
        ACTIVE_BOUNDARY_FILL_SOURCE,
    ]


def test_build_active_triplet_list_boundary_fill_when_disagreement_insufficient():
    # only one disagreement candidate, but three requested
    disagreement = [
        {"anchor_track_id": "a", "candidate_b_track_id": "b",
         "candidate_c_track_id": "c"},
    ]
    boundary = [
        {"anchor_track_id": "g", "candidate_b_track_id": "h",
         "candidate_c_track_id": "i"},
        {"anchor_track_id": "j", "candidate_b_track_id": "k",
         "candidate_c_track_id": "l"},
    ]
    out = build_active_triplet_list(disagreement, boundary, n_questions=3)
    assert len(out) == 3
    assert sum(
        1 for t in out if t["selection_source"] == ACTIVE_BOUNDARY_FILL_SOURCE
    ) == 2


def test_build_active_triplet_list_skips_boundary_duplicate_of_disagreement():
    disagreement = [
        {"anchor_track_id": "a", "candidate_b_track_id": "b",
         "candidate_c_track_id": "c"},
    ]
    boundary = [
        # same key as the disagreement triplet -> must be skipped
        {"anchor_track_id": "a", "candidate_b_track_id": "c",
         "candidate_c_track_id": "b"},
        {"anchor_track_id": "j", "candidate_b_track_id": "k",
         "candidate_c_track_id": "l"},
    ]
    out = build_active_triplet_list(disagreement, boundary, n_questions=5)
    assert len(out) == 2
    assert out[1]["anchor_track_id"] == "j"


def test_build_active_triplet_list_caps_at_n_questions():
    disagreement = [
        {"anchor_track_id": f"a{i}", "candidate_b_track_id": f"b{i}",
         "candidate_c_track_id": f"c{i}"}
        for i in range(10)
    ]
    out = build_active_triplet_list(disagreement, [], n_questions=4)
    assert len(out) == 4
