"""
PURPOSE: Unit tests for src/dj_clustering/metrics.py — manual triplet accuracy
         scoring with the B/C distance direction, skip-row exclusion, unscored
         triplets when a track is absent from the embedding space, cluster
         diagnostics (HDBSCAN-only raw noise rate), exploration-evidence
         status with the 20-triplet winner threshold, and 1001Tracklists
         no-usable-source omission. Pure synthetic fixtures; no private data.

CHANGELOG:
  D4.1a - Initial implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dj_clustering.metrics import (
    NO_USABLE_1001_SOURCE,
    cluster_diagnostics,
    evidence_status,
    filter_non_skip,
    triplet_accuracy,
    tracklists_1001_evidence,
)


def _triplets(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "anchor_track_id",
            "candidate_b_track_id",
            "candidate_c_track_id",
            "answer",
        ],
    )


def test_triplet_accuracy_b_and_c_direction():
    # Anchor at origin; t_near close, t_far distant.
    matrix = np.array([[0.0, 0.0], [0.1, 0.0], [9.0, 0.0]])
    index = {"anchor": 0, "near": 1, "far": 2}
    # answer B: B is near -> correct. answer C: C is near -> correct.
    triplets = _triplets(
        [
            ("anchor", "near", "far", "B"),
            ("anchor", "far", "near", "C"),
        ]
    )
    result = triplet_accuracy(triplets, matrix, index)
    assert result["n_scored"] == 2
    assert result["n_correct"] == 2
    assert result["accuracy"] == 1.0


def test_triplet_accuracy_counts_wrong_answer():
    matrix = np.array([[0.0, 0.0], [0.1, 0.0], [9.0, 0.0]])
    index = {"anchor": 0, "near": 1, "far": 2}
    # answer C while C (far) is actually the more distant -> incorrect.
    triplets = _triplets([("anchor", "near", "far", "C")])
    result = triplet_accuracy(triplets, matrix, index)
    assert result["n_scored"] == 1
    assert result["n_correct"] == 0
    assert result["accuracy"] == 0.0


def test_triplet_accuracy_skips_excluded():
    matrix = np.array([[0.0, 0.0], [0.1, 0.0], [9.0, 0.0]])
    index = {"anchor": 0, "near": 1, "far": 2}
    triplets = _triplets(
        [
            ("anchor", "near", "far", "B"),
            ("anchor", "near", "far", "skip"),
        ]
    )
    assert len(filter_non_skip(triplets)) == 1
    result = triplet_accuracy(triplets, matrix, index)
    assert result["n_total"] == 1
    assert result["n_scored"] == 1


def test_triplet_accuracy_unscored_when_track_missing():
    matrix = np.array([[0.0, 0.0], [0.1, 0.0]])
    index = {"anchor": 0, "near": 1}
    triplets = _triplets([("anchor", "near", "absent", "B")])
    result = triplet_accuracy(triplets, matrix, index)
    assert result["n_scored"] == 0
    assert result["n_unscored"] == 1
    assert result["accuracy"] is None


def test_evidence_status_threshold():
    low = evidence_status(10)
    assert low["classification"] == "exploration"
    assert low["winner_selection_allowed"] is False

    high = evidence_status(25)
    assert high["classification"] == "exploration"
    assert high["winner_selection_allowed"] is True


def test_tracklists_1001_no_usable_source_is_omitted():
    ev = tracklists_1001_evidence(NO_USABLE_1001_SOURCE)
    assert ev["included"] is False
    assert ev["n_positives"] == 0
    assert ev["blocks_regime1"] is False


def test_cluster_diagnostics_hdbscan_reports_noise():
    labels = [0, 0, 0, 1, 1, -1, -1, 2]
    diag = cluster_diagnostics(labels, "hdbscan")
    assert diag["n_tracks"] == 8
    assert diag["cluster_count"] == 3
    assert diag["singleton_count"] == 1  # cluster 2
    assert diag["raw_noise_rate"] == 2 / 8
    assert diag["largest_cluster_share"] == 3 / 8


def test_cluster_diagnostics_kmeans_has_no_noise_rate():
    labels = [0, 0, 1, 1, 2, 2]
    diag = cluster_diagnostics(labels, "kmeans")
    assert diag["raw_noise_rate"] is None
    assert diag["cluster_count"] == 3
