"""
PURPOSE: Apply UMAP dimensionality reduction and HDBSCAN clustering to audio embeddings.
         Combines effnet and maest embeddings after L2 normalization.

CHANGELOG:
    2025-02-03: Initial implementation for Phase 1 validation.
"""
from pathlib import Path
from typing import List, Tuple, Optional
import argparse
import json
import sys

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import HDBSCAN


def load_embeddings(embeddings_dir: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """
    Load embeddings and manifest from directory.

    Args:
        embeddings_dir: Directory containing embedding files

    Returns:
        effnet_embeddings: Shape (N, 1280) or None
        maest_embeddings: Shape (N, 768) or None
        filenames: List of track names
    """
    # Load manifest
    manifest_path = embeddings_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    filenames = manifest["filenames"]

    # Load effnet embeddings
    effnet_embeddings = None
    effnet_path = embeddings_dir / "embeddings_effnet.npy"
    if effnet_path.exists():
        effnet_embeddings = np.load(effnet_path)
        print(f"[INFO] Loaded EffNet embeddings: {effnet_embeddings.shape}")

    # Load maest embeddings
    maest_embeddings = None
    maest_path = embeddings_dir / "embeddings_maest.npy"
    if maest_path.exists():
        maest_embeddings = np.load(maest_path)
        print(f"[INFO] Loaded MAEST embeddings: {maest_embeddings.shape}")

    return effnet_embeddings, maest_embeddings, filenames


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


def combine_embeddings(
    effnet_embeddings: Optional[np.ndarray],
    maest_embeddings: Optional[np.ndarray],
    normalize: bool = True,
) -> np.ndarray:
    """
    Combine embeddings from multiple models.

    If both are provided, concatenate after optional L2 normalization.
    If only one is provided, return it (normalized if requested).

    Args:
        effnet_embeddings: EffNet embeddings or None
        maest_embeddings: MAEST embeddings or None
        normalize: Whether to L2 normalize before combining

    Returns:
        Combined embeddings array
    """
    embeddings_list = []

    if effnet_embeddings is not None:
        emb = l2_normalize(effnet_embeddings) if normalize else effnet_embeddings
        embeddings_list.append(emb)

    if maest_embeddings is not None:
        emb = l2_normalize(maest_embeddings) if normalize else maest_embeddings
        embeddings_list.append(emb)

    if not embeddings_list:
        raise ValueError("No embeddings provided!")

    if len(embeddings_list) == 1:
        return embeddings_list[0]

    combined = np.concatenate(embeddings_list, axis=1)
    print(f"[INFO] Combined embeddings shape: {combined.shape}")
    return combined


def apply_umap(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
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

    Returns:
        np.ndarray: Shape (N, n_components) reduced embeddings
    """
    print(f"[INFO] Running UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric})...")

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=True,
    )

    reduced = reducer.fit_transform(embeddings)
    print(f"[INFO] UMAP output shape: {reduced.shape}")
    return reduced


def apply_hdbscan(
    embeddings_2d: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Apply HDBSCAN clustering.

    Args:
        embeddings_2d: Shape (N, 2) UMAP-reduced embeddings
        min_cluster_size: Minimum cluster size
        min_samples: Minimum samples for core points
        metric: Distance metric

    Returns:
        np.ndarray: Shape (N,) cluster labels (-1 for noise)
    """
    print(f"[INFO] Running HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples})...")

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    )

    labels = clusterer.fit_predict(embeddings_2d)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"[INFO] Found {n_clusters} clusters, {n_noise} noise points")

    return labels


def save_results(
    output_dir: Path,
    filenames: List[str],
    umap_2d: np.ndarray,
    cluster_labels: np.ndarray,
    effnet_embeddings: Optional[np.ndarray] = None,
    maest_embeddings: Optional[np.ndarray] = None,
) -> Path:
    """
    Save all results to CSV.

    Args:
        output_dir: Directory to save results
        filenames: Track filenames
        umap_2d: UMAP coordinates
        cluster_labels: Cluster assignments
        effnet_embeddings: Optional raw effnet embeddings
        maest_embeddings: Optional raw maest embeddings

    Returns:
        Path to saved CSV file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build dataframe
    df = pd.DataFrame({
        "track": filenames,
        "cluster": cluster_labels,
        "umap_x": umap_2d[:, 0],
        "umap_y": umap_2d[:, 1],
    })

    # Save CSV
    csv_path = output_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[SAVED] {csv_path}")

    # Save UMAP coordinates separately
    umap_path = output_dir / "umap_2d.npy"
    np.save(umap_path, umap_2d)
    print(f"[SAVED] {umap_path}")

    # Print cluster statistics
    print("\n[INFO] Cluster distribution:")
    cluster_counts = df["cluster"].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        label = "noise" if cluster_id == -1 else f"cluster {cluster_id}"
        print(f"  {label}: {count} tracks")

    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Apply UMAP and HDBSCAN to audio embeddings"
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        required=True,
        help="Directory containing embedding .npy files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save results"
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter (default: 15)"
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter (default: 0.1)"
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="HDBSCAN min_cluster_size parameter (default: 5)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="HDBSCAN min_samples parameter (default: 3)"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip L2 normalization before combining embeddings"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("TRAKTOR ML - Dimensionality Reduction & Clustering")
    print("=" * 60)
    print(f"Embeddings directory: {args.embeddings_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"UMAP: n_neighbors={args.n_neighbors}, min_dist={args.min_dist}")
    print(f"HDBSCAN: min_cluster_size={args.min_cluster_size}, min_samples={args.min_samples}")
    print()

    # Load embeddings
    effnet_emb, maest_emb, filenames = load_embeddings(args.embeddings_dir)

    if effnet_emb is None and maest_emb is None:
        print("[ERROR] No embeddings found!")
        return 1

    # Combine embeddings
    combined = combine_embeddings(
        effnet_emb,
        maest_emb,
        normalize=not args.no_normalize
    )

    # Apply UMAP
    umap_2d = apply_umap(
        combined,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
    )

    # Apply HDBSCAN
    cluster_labels = apply_hdbscan(
        umap_2d,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )

    # Save results
    save_results(
        args.output_dir,
        filenames,
        umap_2d,
        cluster_labels,
        effnet_emb,
        maest_emb,
    )

    print("\n" + "=" * 60)
    print("[SUCCESS] Clustering complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
