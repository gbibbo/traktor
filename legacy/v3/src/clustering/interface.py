#!/usr/bin/env python3
"""
PURPOSE: Abstract interface for TRAKTOR ML V3 clustering algorithms.
         Defines the contract all clusterers must satisfy, plus shared
         metric computation utilities (silhouette, calinski-harabasz,
         davies-bouldin).

CHANGELOG:
    2026-02-06: Initial implementation for V3 clustering module.
"""
import abc
from typing import Any, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------
class BaseClusterer(abc.ABC):
    """Abstract base class for all clustering algorithms."""

    @abc.abstractmethod
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Cluster the input data and return integer labels.

        Parameters
        ----------
        X : np.ndarray, shape [N, D]
            Feature matrix (e.g., PCA-reduced embeddings).

        Returns
        -------
        labels : np.ndarray, shape [N,]
            Integer cluster labels. -1 indicates noise (for algorithms
            that support it, e.g., HDBSCAN).
        """
        ...

    @abc.abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict of all algorithm parameters."""
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short algorithm name for file naming (e.g., 'kmeans', 'hdbscan')."""
        ...


# ---------------------------------------------------------------------------
# Shared metric computation
# ---------------------------------------------------------------------------
def compute_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """Compute clustering quality metrics.

    Handles degenerate cases gracefully:
    - All points in one cluster -> metrics are None
    - n_clusters < 2 -> metrics are None
    - Noise-only labels (all -1) -> metrics are None

    Parameters
    ----------
    X : np.ndarray, shape [N, D]
    labels : np.ndarray, shape [N,]

    Returns
    -------
    dict with keys: silhouette, calinski_harabasz, davies_bouldin,
                    n_clusters, n_noise, n_total, cluster_sizes
    """
    from sklearn.metrics import (
        silhouette_score,
        calinski_harabasz_score,
        davies_bouldin_score,
    )

    unique_labels = set(labels)
    n_noise = int(np.sum(labels == -1))
    real_labels = unique_labels - {-1}
    n_clusters = len(real_labels)
    n_total = len(labels)

    # Cluster sizes (excluding noise)
    cluster_sizes = {}
    for lbl in sorted(real_labels):
        cluster_sizes[int(lbl)] = int(np.sum(labels == lbl))

    result: Dict[str, Any] = {
        "silhouette": None,
        "calinski_harabasz": None,
        "davies_bouldin": None,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "n_total": n_total,
        "cluster_sizes": cluster_sizes,
    }

    # Need at least 2 clusters and 2 non-noise samples per cluster
    if n_clusters < 2 or n_total < 2:
        return result

    # Filter out noise for metric computation
    if n_noise > 0:
        mask = labels != -1
        X_clean = X[mask]
        labels_clean = labels[mask]
    else:
        X_clean = X
        labels_clean = labels

    # Need at least 2 samples after filtering
    if len(X_clean) < 2:
        return result

    # Verify still >= 2 clusters after filtering
    if len(set(labels_clean)) < 2:
        return result

    try:
        result["silhouette"] = round(float(silhouette_score(X_clean, labels_clean)), 6)
        result["calinski_harabasz"] = round(float(calinski_harabasz_score(X_clean, labels_clean)), 6)
        result["davies_bouldin"] = round(float(davies_bouldin_score(X_clean, labels_clean)), 6)
    except ValueError:
        # Degenerate geometry (e.g., all points identical)
        pass

    return result
