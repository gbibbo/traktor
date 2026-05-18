"""
PURPOSE: Regime 1 clustering primitives for the DJ clustering sweep (Task
         D4.1). Provides embedding loading, L2 / no-op normalization,
         dimensionality reduction (none / PCA / UMAP), the three Regime 1
         clusterer wrappers (HDBSCAN, KMeans, agglomerative with average
         linkage), and HDBSCAN noise-policy handling. Operates purely on
         frozen MERT embedding vectors; no audio, no GPU, no MERT inference.

         hdbscan and umap are optional imports: when unavailable, the affected
         wrappers raise a clear RuntimeError and `*_AVAILABLE` flags let
         callers and tests degrade gracefully.

CHANGELOG:
  D4.1a - Initial implementation (scaffold; sweep not executed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:  # optional runtime dependency
    import hdbscan as _hdbscan

    HDBSCAN_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when dep is absent
    _hdbscan = None
    HDBSCAN_AVAILABLE = False

try:  # optional runtime dependency
    import umap as _umap

    UMAP_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when dep is absent
    _umap = None
    UMAP_AVAILABLE = False

# Cosine-distance threshold for confidence_limited_1nn noise reassignment.
CONFIDENCE_LIMITED_MAX_COSINE_DISTANCE = 0.30

# Seed used for every stochastic clustering / reduction step in the sweep.
DEFAULT_SEED = 13

NOISE_LABEL = -1


@dataclass
class ClusterResult:
    """Outcome of clustering a single config.

    metric_labels   labels used for evaluation (native noise kept as -1 unless
                    a noise policy reassigns it).
    export_labels   labels used for manifest export (may force-assign noise).
    membership_source  'native' for vector clustering; 'medoid_fallback' is
                    reserved for precomputed-distance paths (not used in D4.1).
    """

    metric_labels: np.ndarray
    export_labels: np.ndarray
    membership_source: str = "native"
    noise_policy: str = "not_applicable"
    extra: Dict = field(default_factory=dict)


def load_embedding_source(
    features_dir: Path, embedding_source: str, aggregation: str
) -> Tuple[np.ndarray, List[str]]:
    """Load a frozen MERT embedding matrix and its aligned track_ids.

    Returns (X, track_ids) where X has shape (n_tracks, dim) as float64 and
    track_ids is the row-aligned list read from track_ids.txt.
    """
    base = Path(features_dir) / embedding_source / aggregation
    emb_path = base / "embeddings.npy"
    ids_path = base / "track_ids.txt"
    if not emb_path.exists():
        raise FileNotFoundError(f"missing embeddings: {emb_path}")
    if not ids_path.exists():
        raise FileNotFoundError(f"missing track_ids: {ids_path}")
    matrix = np.load(emb_path).astype(np.float64)
    track_ids = [line.strip() for line in ids_path.read_text().splitlines() if line.strip()]
    if matrix.shape[0] != len(track_ids):
        raise ValueError(
            f"embedding/track_id row mismatch for {embedding_source}/{aggregation}: "
            f"{matrix.shape[0]} rows vs {len(track_ids)} ids"
        )
    return matrix, track_ids


def normalize_embeddings(matrix: np.ndarray, method: str) -> np.ndarray:
    """Apply the requested normalization.

    'l2'  : scale each row to unit L2 norm (zero rows left unchanged).
    'none': return an unmodified copy (diagnostic ablation).
    """
    if method == "none":
        return np.array(matrix, dtype=np.float64, copy=True)
    if method == "l2":
        out = np.array(matrix, dtype=np.float64, copy=True)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        safe = np.where(norms == 0.0, 1.0, norms)
        return out / safe
    raise ValueError(f"unknown normalization method: {method}")


def reduce_dimensions(
    matrix: np.ndarray, method: str, seed: int = DEFAULT_SEED
) -> np.ndarray:
    """Apply dimensionality reduction.

    'none'   : identity.
    'pca_50' : PCA to 50 components (clamped to a valid component count).
    'pca_100': PCA to 100 components (clamped to a valid component count).
    'umap_15': UMAP to 15 components (requires the optional umap dependency).
    """
    if method == "none":
        return np.array(matrix, dtype=np.float64, copy=True)
    if method in ("pca_50", "pca_100"):
        from sklearn.decomposition import PCA

        requested = int(method.split("_")[1])
        n_components = min(requested, matrix.shape[0], matrix.shape[1])
        reducer = PCA(n_components=n_components, random_state=seed)
        return reducer.fit_transform(matrix).astype(np.float64)
    if method == "umap_15":
        if not UMAP_AVAILABLE:
            raise RuntimeError("dim_reduction 'umap_15' requires the umap package")
        n_components = min(15, max(2, matrix.shape[0] - 2))
        reducer = _umap.UMAP(
            n_components=n_components,
            n_neighbors=15,
            random_state=seed,
        )
        return reducer.fit_transform(matrix).astype(np.float64)
    raise ValueError(f"unknown dim_reduction method: {method}")


def cluster_hdbscan(
    matrix: np.ndarray, min_cluster_size: int, min_samples: int
) -> np.ndarray:
    """Cluster with HDBSCAN on Euclidean distance; noise rows get label -1."""
    if not HDBSCAN_AVAILABLE:
        raise RuntimeError("clusterer 'hdbscan' requires the hdbscan package")
    clusterer = _hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        metric="euclidean",
    )
    return clusterer.fit_predict(matrix).astype(int)


def cluster_kmeans(
    matrix: np.ndarray, k: int, seed: int = DEFAULT_SEED
) -> np.ndarray:
    """Cluster with KMeans; every row receives a cluster label (no noise)."""
    from sklearn.cluster import KMeans

    clusterer = KMeans(n_clusters=int(k), random_state=seed, n_init=10)
    return clusterer.fit_predict(matrix).astype(int)


def cluster_agglomerative(
    matrix: np.ndarray, k: int, linkage: str = "average"
) -> np.ndarray:
    """Cluster with agglomerative clustering; every row receives a label.

    Average linkage is enforced so the same wrapper is valid for both vector
    and (future) precomputed-distance inputs; ward linkage is rejected.
    """
    from sklearn.cluster import AgglomerativeClustering

    if linkage == "ward":
        raise ValueError("ward linkage is not permitted for Regime 1 sweep configs")
    clusterer = AgglomerativeClustering(n_clusters=int(k), linkage=linkage)
    return clusterer.fit_predict(matrix).astype(int)


def _cosine_distance_rows(matrix: np.ndarray) -> np.ndarray:
    """Return the dense (n, n) cosine-distance matrix for noise reassignment."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    unit = matrix / safe
    similarity = np.clip(unit @ unit.T, -1.0, 1.0)
    return 1.0 - similarity


def apply_noise_policy(
    labels: np.ndarray,
    matrix: np.ndarray,
    policy: str,
    max_cosine_distance: float = CONFIDENCE_LIMITED_MAX_COSINE_DISTANCE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply an HDBSCAN noise policy and return (metric_labels, export_labels).

    no_reassignment        : both outputs keep native noise (-1).
    confidence_limited_1nn : reassign a noise row only if its nearest non-noise
                             neighbour has cosine distance <= max_cosine_distance;
                             applied to both metric and export labels.
    forced_1nn_export_only : metric labels keep native noise; export labels
                             force every noise row to its nearest non-noise
                             cluster regardless of distance.
    """
    labels = np.asarray(labels, dtype=int)
    metric = labels.copy()
    export = labels.copy()
    noise_idx = np.where(labels == NOISE_LABEL)[0]
    non_noise_idx = np.where(labels != NOISE_LABEL)[0]

    if policy == "no_reassignment":
        return metric, export
    if len(noise_idx) == 0 or len(non_noise_idx) == 0:
        return metric, export

    distances = _cosine_distance_rows(matrix)
    for i in noise_idx:
        nearest = non_noise_idx[np.argmin(distances[i, non_noise_idx])]
        nearest_dist = distances[i, nearest]
        if policy == "confidence_limited_1nn":
            if nearest_dist <= max_cosine_distance:
                metric[i] = labels[nearest]
                export[i] = labels[nearest]
        elif policy == "forced_1nn_export_only":
            export[i] = labels[nearest]
        else:
            raise ValueError(f"unknown noise policy: {policy}")
    return metric, export


def run_clustering(
    matrix: np.ndarray, cfg: Dict, seed: int = DEFAULT_SEED
) -> ClusterResult:
    """Run normalization, dim reduction, clustering and noise policy for a config.

    `cfg` is a run spec produced by src.dj_clustering.sweep. Used by the D4.1b/c
    sweep execution path; not invoked during D4.1a scaffolding.
    """
    normalized = normalize_embeddings(matrix, cfg["normalization"])
    reduced = reduce_dimensions(normalized, cfg["dim_reduction"], seed=seed)
    clusterer = cfg["clusterer"]

    if clusterer == "hdbscan":
        labels = cluster_hdbscan(
            reduced,
            cfg["hdbscan_min_cluster_size"],
            cfg["hdbscan_min_samples"],
        )
        metric, export = apply_noise_policy(labels, reduced, cfg["noise_policy"])
        return ClusterResult(
            metric_labels=metric,
            export_labels=export,
            membership_source="native",
            noise_policy=cfg["noise_policy"],
        )
    if clusterer == "kmeans":
        labels = cluster_kmeans(reduced, cfg["kmeans_k"], seed=seed)
        return ClusterResult(
            metric_labels=labels,
            export_labels=labels.copy(),
            membership_source="native",
            noise_policy="not_applicable",
        )
    if clusterer == "agglomerative":
        labels = cluster_agglomerative(
            reduced, cfg["agglomerative_k"], cfg["agglomerative_linkage"]
        )
        return ClusterResult(
            metric_labels=labels,
            export_labels=labels.copy(),
            membership_source="native",
            noise_policy="not_applicable",
        )
    raise ValueError(f"unknown clusterer: {clusterer}")
