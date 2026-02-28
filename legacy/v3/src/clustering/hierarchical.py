#!/usr/bin/env python3
"""
PURPOSE: Hierarchical two-level clustering for TRAKTOR ML V3.
         L1 produces macro groups (groove/energy), L2 sub-clusters within
         each L1 group (timbre/elements). Outputs composite labels like
         "A", "A1", "A2", "B", "B1", etc.

CHANGELOG:
    2026-02-06: Initial implementation for V3 clustering module.
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.clustering.flat import create_clusterer
from src.clustering.interface import compute_metrics


# ---------------------------------------------------------------------------
# Label helpers (ported from legacy V2 clustering_utils.py)
# ---------------------------------------------------------------------------
def cluster_to_letter(cluster_id: int) -> str:
    """Convert numeric cluster ID to letter. -1 -> 'Noise'."""
    if cluster_id == -1:
        return "Noise"
    if cluster_id > 25:
        print(f"[WARN] cluster_id={cluster_id} exceeds 26 letters, "
              f"label will be non-alphabetic")
    return chr(ord("A") + cluster_id)


def subcluster_label(parent_letter: str, sub_id: int) -> str:
    """Generate sub-label: 'A' + 0 -> 'A1', 'B' + 2 -> 'B3'."""
    return f"{parent_letter}{sub_id + 1}"


# ---------------------------------------------------------------------------
# Hierarchical clusterer
# ---------------------------------------------------------------------------
class HierarchicalClusterer:
    """Two-level clustering: L1 (coarse) -> L2 (fine) within each L1 group.

    Not a BaseClusterer subclass because its output structure is different
    (two label columns + metadata dict instead of a flat labels array).

    Parameters
    ----------
    l1_algorithm : str
        Algorithm name for L1 (macro) clustering.
    l1_params : dict, optional
        Kwargs passed to the L1 clusterer constructor.
    l2_algorithm : str
        Algorithm name for L2 (sub) clustering.
    l2_params : dict, optional
        Kwargs passed to the L2 clusterer constructor.
    l2_min_points : int
        Minimum points in an L1 cluster to attempt L2 sub-clustering.
        Clusters smaller than this get a single sub-label (e.g., "C1").
    """

    def __init__(
        self,
        l1_algorithm: str = "kmeans",
        l1_params: Optional[Dict[str, Any]] = None,
        l2_algorithm: str = "kmeans",
        l2_params: Optional[Dict[str, Any]] = None,
        l2_min_points: int = 10,
    ) -> None:
        self._l1_algorithm = l1_algorithm
        self._l1_params = l1_params or {}
        self._l2_algorithm = l2_algorithm
        self._l2_params = l2_params or {}
        self._l2_min_points = l2_min_points

    def fit_predict(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Run two-level clustering.

        Returns
        -------
        l1_labels : np.ndarray, shape [N,] of str
            Letter labels: "A", "B", ..., "Noise"
        composite_labels : np.ndarray, shape [N,] of str
            Sub-labels: "A1", "A2", "B1", ..., "Noise"
        metadata : dict
            Nested metadata with L1 info, per-cluster L2 info, and metrics.
        """
        N = X.shape[0]

        # --- L1: coarse clustering ---
        l1_clusterer = create_clusterer(self._l1_algorithm, **self._l1_params)
        l1_raw = l1_clusterer.fit_predict(X)
        l1_metrics = compute_metrics(X, l1_raw)

        # Convert integer labels to letters
        l1_labels = np.array([cluster_to_letter(int(lbl)) for lbl in l1_raw],
                             dtype=object)
        composite_labels = np.empty(N, dtype=object)

        # --- L2: sub-clustering within each L1 group ---
        real_labels = sorted(set(l1_raw) - {-1})
        l2_per_cluster = {}

        for raw_lbl in real_labels:
            letter = cluster_to_letter(int(raw_lbl))
            mask = l1_raw == raw_lbl
            X_sub = X[mask]
            n_points = int(mask.sum())
            indices = np.where(mask)[0]

            if n_points < self._l2_min_points:
                # Too few points — assign single sub-label
                for idx in indices:
                    composite_labels[idx] = subcluster_label(letter, 0)
                l2_per_cluster[letter] = {
                    "n_points": n_points,
                    "n_subclusters": 1,
                    "skipped": True,
                    "reason": f"n_points ({n_points}) < l2_min_points ({self._l2_min_points})",
                }
                print(f"[INFO] L2 skip cluster {letter}: "
                      f"{n_points} points < {self._l2_min_points} threshold")
                continue

            # Run L2 clustering on the subset
            l2_clusterer = create_clusterer(self._l2_algorithm, **self._l2_params)
            l2_raw = l2_clusterer.fit_predict(X_sub)
            l2_metrics = compute_metrics(X_sub, l2_raw)

            # Assign composite labels
            l2_real = sorted(set(l2_raw) - {-1})
            for i, idx in enumerate(indices):
                sub_lbl = int(l2_raw[i])
                if sub_lbl == -1:
                    composite_labels[idx] = f"{letter}_noise"
                else:
                    composite_labels[idx] = subcluster_label(letter, sub_lbl)

            n_subclusters = len(l2_real)
            l2_per_cluster[letter] = {
                "n_points": n_points,
                "n_subclusters": n_subclusters,
                "skipped": False,
                "metrics": l2_metrics,
                "params": l2_clusterer.get_params(),
            }
            print(f"[INFO] L2 cluster {letter}: "
                  f"{n_points} points -> {n_subclusters} sub-clusters")

        # Noise at L1 level
        noise_mask = l1_raw == -1
        if noise_mask.any():
            for idx in np.where(noise_mask)[0]:
                composite_labels[idx] = "Noise"
            print(f"[INFO] L1 noise: {int(noise_mask.sum())} points")

        # --- Metadata ---
        metadata = {
            "l1": {
                "algorithm": self._l1_algorithm,
                "params": l1_clusterer.get_params(),
                "metrics": l1_metrics,
                "n_clusters": len(real_labels),
            },
            "l2": {
                "algorithm": self._l2_algorithm,
                "params": self._l2_params,
                "per_cluster": l2_per_cluster,
            },
            "l2_min_points": self._l2_min_points,
        }

        return l1_labels, composite_labels, metadata

    def get_params(self) -> Dict[str, Any]:
        return {
            "mode": "hierarchical",
            "l1_algorithm": self._l1_algorithm,
            "l1_params": self._l1_params,
            "l2_algorithm": self._l2_algorithm,
            "l2_params": self._l2_params,
            "l2_min_points": self._l2_min_points,
        }
