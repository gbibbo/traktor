"""
PURPOSE: Regime 1 sweep evaluation metrics for DJ clustering (Task D4.1).
         Provides manual triplet accuracy scoring, cluster-shape diagnostics,
         exploration-evidence status, and 1001Tracklists evidence handling.

         Triplet semantics: answer 'B' means d(anchor, B) < d(anchor, C);
         answer 'C' means d(anchor, C) < d(anchor, B). Skip rows are excluded.
         Distances are Euclidean in each config's transformed embedding space.
         Until the first full sweep, all answered triplets are exploration
         evidence; with fewer than 20 non-skip answers no winner may be picked.

CHANGELOG:
  D4.1a - Initial implementation (scaffold; sweep not executed).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

# Minimum non-skip triplet count below which no winner may be selected.
MIN_NON_SKIP_FOR_WINNER = 20

# Status string emitted by D3.4 when no usable 1001Tracklists source exists.
NO_USABLE_1001_SOURCE = "no_usable_source"

ANSWER_VALUES = ("B", "C")


def load_manual_triplets(path: Path) -> pd.DataFrame:
    """Load the ingested manual triplet answers as a DataFrame."""
    df = pd.read_csv(path, dtype=str)
    required = {"anchor_track_id", "candidate_b_track_id", "candidate_c_track_id", "answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manual triplets missing columns: {sorted(missing)}")
    return df


def filter_non_skip(triplets: pd.DataFrame) -> pd.DataFrame:
    """Return only rows with a B/C answer; skip rows are excluded defensively.

    The ingested manual_triplets.csv already excludes skips (they live in the
    separate skip log), but this guard keeps scoring correct if a skip row is
    ever present.
    """
    answers = triplets["answer"].astype(str).str.strip()
    return triplets[answers.isin(ANSWER_VALUES)].copy()


def triplet_accuracy(
    triplets: pd.DataFrame,
    transformed_matrix: np.ndarray,
    track_id_index: Mapping[str, int],
) -> Dict:
    """Score manual triplets against a config's transformed embedding space.

    For each non-skip triplet the closer candidate is predicted from Euclidean
    distance to the anchor. A triplet referencing a track absent from the
    embedding matrix is counted as unscored. Exact-tie predictions count as
    incorrect. Returns aggregate counts and accuracy over scored triplets.
    """
    non_skip = filter_non_skip(triplets)
    n_total = int(len(non_skip))
    n_correct = 0
    n_scored = 0
    n_unscored = 0

    for _, row in non_skip.iterrows():
        anchor = str(row["anchor_track_id"]).strip()
        cand_b = str(row["candidate_b_track_id"]).strip()
        cand_c = str(row["candidate_c_track_id"]).strip()
        answer = str(row["answer"]).strip()
        if any(tid not in track_id_index for tid in (anchor, cand_b, cand_c)):
            n_unscored += 1
            continue
        x_a = transformed_matrix[track_id_index[anchor]]
        x_b = transformed_matrix[track_id_index[cand_b]]
        x_c = transformed_matrix[track_id_index[cand_c]]
        dist_b = float(np.linalg.norm(x_a - x_b))
        dist_c = float(np.linalg.norm(x_a - x_c))
        n_scored += 1
        if dist_b < dist_c:
            predicted = "B"
        elif dist_c < dist_b:
            predicted = "C"
        else:
            predicted = None  # exact tie -> incorrect
        if predicted == answer:
            n_correct += 1

    accuracy = (n_correct / n_scored) if n_scored > 0 else None
    return {
        "n_total": n_total,
        "n_scored": n_scored,
        "n_unscored": n_unscored,
        "n_correct": n_correct,
        "accuracy": accuracy,
    }


def cluster_diagnostics(metric_labels: Sequence[int], clusterer: str) -> Dict:
    """Compute cluster-shape diagnostics from metric labels.

    raw_noise_rate is reported only for HDBSCAN (the sole Regime 1 clusterer
    with native noise); for KMeans/agglomerative it is None.
    """
    labels = np.asarray(list(metric_labels), dtype=int)
    n_tracks = int(labels.size)
    noise_mask = labels == -1
    n_noise = int(noise_mask.sum())

    assigned = labels[~noise_mask]
    if assigned.size > 0:
        unique, counts = np.unique(assigned, return_counts=True)
        cluster_count = int(unique.size)
        largest_cluster_share = float(counts.max() / n_tracks) if n_tracks else 0.0
        singleton_count = int((counts == 1).sum())
    else:
        cluster_count = 0
        largest_cluster_share = 0.0
        singleton_count = 0

    raw_noise_rate: Optional[float]
    if clusterer == "hdbscan":
        raw_noise_rate = float(n_noise / n_tracks) if n_tracks else 0.0
    else:
        raw_noise_rate = None

    return {
        "n_tracks": n_tracks,
        "cluster_count": cluster_count,
        "largest_cluster_share": largest_cluster_share,
        "singleton_count": singleton_count,
        "raw_noise_rate": raw_noise_rate,
    }


def evidence_status(n_non_skip: int) -> Dict:
    """Classify the manual triplet evidence available for the sweep.

    Until the first full sweep, all answered triplets are exploration evidence.
    With fewer than 20 non-skip answers a winner may not be selected.
    """
    sufficient = n_non_skip >= MIN_NON_SKIP_FOR_WINNER
    return {
        "n_non_skip": int(n_non_skip),
        "classification": "exploration",
        "winner_selection_allowed": bool(sufficient),
        "min_required_for_winner": MIN_NON_SKIP_FOR_WINNER,
    }


def tracklists_1001_evidence(status: str) -> Dict:
    """Describe the 1001Tracklists contribution to sweep metrics.

    With status 'no_usable_source', 1001 contributes zero external evidence and
    is omitted from metrics; it never blocks Regime 1.
    """
    if status == NO_USABLE_1001_SOURCE:
        return {
            "included": False,
            "reason": NO_USABLE_1001_SOURCE,
            "n_positives": 0,
            "blocks_regime1": False,
        }
    return {
        "included": True,
        "reason": status,
        "n_positives": None,
        "blocks_regime1": False,
    }
