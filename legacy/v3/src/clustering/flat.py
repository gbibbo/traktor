#!/usr/bin/env python3
"""
PURPOSE: Flat clustering algorithms for TRAKTOR ML V3.
         Wraps scikit-learn KMeans, AgglomerativeClustering, and HDBSCAN
         behind the BaseClusterer interface.

CHANGELOG:
    2026-02-06: Initial implementation for V3 clustering module.
"""
from typing import Any, Dict

import numpy as np
from sklearn.cluster import AgglomerativeClustering, HDBSCAN, KMeans

from src.clustering.interface import BaseClusterer


# ---------------------------------------------------------------------------
# KMeans
# ---------------------------------------------------------------------------
class KMeansClusterer(BaseClusterer):
    """KMeans clustering with configurable k and random state."""

    def __init__(
        self,
        n_clusters: int = 5,
        random_state: int = 42,
        n_init: int = 10,
        max_iter: int = 300,
    ) -> None:
        self._n_clusters = n_clusters
        self._random_state = random_state
        self._n_init = n_init
        self._max_iter = max_iter
        self._model = None

    @property
    def name(self) -> str:
        return "kmeans"

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self._model = KMeans(
            n_clusters=self._n_clusters,
            random_state=self._random_state,
            n_init=self._n_init,
            max_iter=self._max_iter,
        )
        return self._model.fit_predict(X)

    def get_params(self) -> Dict[str, Any]:
        return {
            "algorithm": "kmeans",
            "n_clusters": self._n_clusters,
            "random_state": self._random_state,
            "n_init": self._n_init,
            "max_iter": self._max_iter,
        }


# ---------------------------------------------------------------------------
# Agglomerative
# ---------------------------------------------------------------------------
class AgglomerativeClusterer(BaseClusterer):
    """Agglomerative (hierarchical linkage) clustering."""

    def __init__(
        self,
        n_clusters: int = 5,
        linkage: str = "ward",
        metric: str = "euclidean",
    ) -> None:
        self._n_clusters = n_clusters
        self._linkage = linkage
        self._metric = metric
        self._model = None

        # Ward linkage requires euclidean metric
        if self._linkage == "ward" and self._metric != "euclidean":
            print(f"[WARN] linkage='ward' requires metric='euclidean', "
                  f"overriding metric='{self._metric}' -> 'euclidean'")
            self._metric = "euclidean"

    @property
    def name(self) -> str:
        return "agglomerative"

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self._model = AgglomerativeClustering(
            n_clusters=self._n_clusters,
            linkage=self._linkage,
            metric=self._metric,
        )
        return self._model.fit_predict(X)

    def get_params(self) -> Dict[str, Any]:
        return {
            "algorithm": "agglomerative",
            "n_clusters": self._n_clusters,
            "linkage": self._linkage,
            "metric": self._metric,
        }


# ---------------------------------------------------------------------------
# HDBSCAN
# ---------------------------------------------------------------------------
class HDBSCANClusterer(BaseClusterer):
    """HDBSCAN density-based clustering (noise-aware)."""

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: int = 3,
        metric: str = "euclidean",
        cluster_selection_method: str = "eom",
    ) -> None:
        self._min_cluster_size = min_cluster_size
        self._min_samples = min_samples
        self._metric = metric
        self._cluster_selection_method = cluster_selection_method
        self._model = None

    @property
    def name(self) -> str:
        return "hdbscan"

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self._model = HDBSCAN(
            min_cluster_size=self._min_cluster_size,
            min_samples=self._min_samples,
            metric=self._metric,
            cluster_selection_method=self._cluster_selection_method,
        )
        return self._model.fit_predict(X)

    def get_params(self) -> Dict[str, Any]:
        return {
            "algorithm": "hdbscan",
            "min_cluster_size": self._min_cluster_size,
            "min_samples": self._min_samples,
            "metric": self._metric,
            "cluster_selection_method": self._cluster_selection_method,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
ALGORITHMS: Dict[str, type] = {
    "kmeans": KMeansClusterer,
    "agglomerative": AgglomerativeClusterer,
    "hdbscan": HDBSCANClusterer,
}


def create_clusterer(algorithm: str, **kwargs: Any) -> BaseClusterer:
    """Create a clusterer by name with keyword arguments.

    Parameters
    ----------
    algorithm : str
        One of: 'kmeans', 'agglomerative', 'hdbscan'.
    **kwargs
        Passed to the clusterer constructor.

    Raises
    ------
    ValueError
        If algorithm name is unknown.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Choose from: {list(ALGORITHMS.keys())}"
        )
    return ALGORITHMS[algorithm](**kwargs)
