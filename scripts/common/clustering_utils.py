"""
PURPOSE: Clustering utilities for TRAKTOR ML V2.
         Provides UMAP dimensionality reduction and HDBSCAN clustering.

CHANGELOG:
    2026-02-05: Added simplify_genre_name() to strip category prefixes from genre names.
    2025-02-04: Extracted from V1 reduce_and_cluster.py for V2 pipeline.
"""
from typing import Tuple, Optional
import numpy as np
import umap
from sklearn.cluster import HDBSCAN


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """
    Apply L2 normalization to embeddings.

    Args:
        embeddings: Shape (N, D)

    Returns:
        Normalized embeddings with unit L2 norm per row
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)  # Avoid division by zero
    return embeddings / norms


def apply_umap(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
    verbose: bool = True,
) -> np.ndarray:
    """
    Apply UMAP dimensionality reduction.

    Args:
        embeddings: Shape (N, D) high-dimensional embeddings
        n_components: Output dimensions (default 2 for visualization)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: Distance metric to use
        random_state: Random seed for reproducibility
        verbose: Whether to print progress

    Returns:
        np.ndarray: Shape (N, n_components) reduced embeddings
    """
    if verbose:
        print(f"[UMAP] Running (n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric})...")

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=verbose,
    )

    reduced = reducer.fit_transform(embeddings)

    if verbose:
        print(f"[UMAP] Output shape: {reduced.shape}")

    return reduced


def apply_hdbscan(
    embeddings_2d: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    metric: str = "euclidean",
    verbose: bool = True,
) -> np.ndarray:
    """
    Apply HDBSCAN clustering.

    Args:
        embeddings_2d: Shape (N, 2) UMAP-reduced embeddings
        min_cluster_size: Minimum cluster size
        min_samples: Minimum samples for core points
        metric: Distance metric
        verbose: Whether to print stats

    Returns:
        np.ndarray: Shape (N,) cluster labels (-1 for noise)
    """
    if verbose:
        print(f"[HDBSCAN] Running (min_cluster_size={min_cluster_size}, min_samples={min_samples})...")

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    )

    labels = clusterer.fit_predict(embeddings_2d)

    if verbose:
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        print(f"[HDBSCAN] Found {n_clusters} clusters, {n_noise} noise points")

    return labels


def cluster_to_letter(cluster_id: int) -> str:
    """
    Convert numeric cluster ID to letter label.

    Args:
        cluster_id: Numeric cluster ID (0, 1, 2, ...) or -1 for noise

    Returns:
        Letter label ('A', 'B', 'C', ...) or 'Noise'
    """
    if cluster_id == -1:
        return "Noise"
    return chr(ord('A') + cluster_id)


def letter_to_cluster(letter: str) -> int:
    """
    Convert letter label back to numeric cluster ID.

    Args:
        letter: Letter label ('A', 'B', 'C', ...) or 'Noise'

    Returns:
        Numeric cluster ID or -1 for noise
    """
    if letter == "Noise":
        return -1
    return ord(letter) - ord('A')


def subcluster_label(parent_letter: str, sub_id: int) -> str:
    """
    Generate subcluster label.

    Args:
        parent_letter: Parent cluster letter ('A', 'B', etc.)
        sub_id: Subcluster ID (0, 1, 2, ...)

    Returns:
        Subcluster label ('A1', 'A2', 'B1', etc.)
    """
    return f"{parent_letter}{sub_id + 1}"


def get_cluster_stats(labels: np.ndarray) -> dict:
    """
    Get statistics about clustering results.

    Args:
        labels: Cluster labels array

    Returns:
        Dict with cluster stats
    """
    unique, counts = np.unique(labels, return_counts=True)
    stats = {
        "n_clusters": len(unique) - (1 if -1 in unique else 0),
        "n_noise": int((labels == -1).sum()),
        "n_total": len(labels),
        "cluster_sizes": {int(k): int(v) for k, v in zip(unique, counts) if k != -1},
    }
    return stats


def simplify_genre_name(genre: str) -> str:
    """
    Remove category prefix from genre names (e.g., 'Electronic---' prefix).

    Args:
        genre: Full genre name (e.g., "Electronic---Techno", "Rock---Indie Rock")

    Returns:
        Simplified genre name (e.g., "Techno", "Indie Rock")
    """
    if not genre:
        return "Unknown"

    if genre in ["unknown", "error", "Noise"]:
        return genre

    if "---" in genre:
        parts = genre.split("---", 1)
        if len(parts) == 2:
            return parts[1]

    return genre
