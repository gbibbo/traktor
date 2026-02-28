#!/usr/bin/env python3
"""
PURPOSE: CLI orchestrator for TRAKTOR ML V3 clustering.
         Loads PCA features, runs flat or hierarchical clustering,
         saves labeled CSVs and JSON metadata to the dataset artifacts.

CHANGELOG:
    2026-02-06: Initial implementation for V3 clustering module.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.clustering.flat import ALGORITHMS, create_clusterer
from src.clustering.hierarchical import HierarchicalClusterer
from src.clustering.interface import compute_metrics


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_INPUT_FEATURES = "X_pca128.npy"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(
    dataset_dir: Path,
    features_file: str = DEFAULT_INPUT_FEATURES,
) -> tuple:
    """Load feature matrix and manifest, aligned by track_id.

    The manifest may contain more tracks than the feature matrix (e.g.,
    tracks that failed during extraction are in the manifest but have no
    embeddings). This function joins on track_ids to ensure row alignment.

    Returns
    -------
    X : np.ndarray, shape [N, D]
    manifest_df : pd.DataFrame  (filtered and aligned with X)
    """
    features_path = dataset_dir / features_file
    manifest_path = dataset_dir / "manifest.parquet"
    track_ids_path = dataset_dir / "embeddings" / "track_ids.npy"

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not track_ids_path.exists():
        raise FileNotFoundError(f"Track IDs not found: {track_ids_path}")

    X = np.load(features_path)
    manifest_df = pd.read_parquet(manifest_path)
    track_ids = np.load(track_ids_path, allow_pickle=True).tolist()

    if X.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape {X.shape}")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Features contain NaN or Inf values")
    if X.shape[0] != len(track_ids):
        raise ValueError(
            f"Shape mismatch: {X.shape[0]} features vs {len(track_ids)} track_ids"
        )

    # Align manifest to feature rows via track_id
    n_manifest = len(manifest_df)
    manifest_df = manifest_df[manifest_df["track_id"].isin(track_ids)]
    manifest_df = manifest_df.set_index("track_id").loc[track_ids].reset_index()

    if len(manifest_df) != X.shape[0]:
        raise ValueError(
            f"After alignment: {len(manifest_df)} manifest rows "
            f"vs {X.shape[0]} features"
        )

    n_dropped = n_manifest - len(manifest_df)
    if n_dropped > 0:
        print(f"[WARN] Dropped {n_dropped} manifest rows with no features")

    return X, manifest_df


# ---------------------------------------------------------------------------
# Run ID generation
# ---------------------------------------------------------------------------
def generate_run_id(
    algorithm: str, mode: str, user_id: Optional[str] = None
) -> str:
    """Generate a unique run ID.

    Format: <mode>_<algorithm>_<YYYYMMDD_HHMMSS> or user-provided.
    """
    if user_id:
        return user_id
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{mode}_{algorithm}_{ts}"


# ---------------------------------------------------------------------------
# Algo kwargs builders
# ---------------------------------------------------------------------------
def _build_algo_kwargs(args: argparse.Namespace, algorithm: str) -> Dict[str, Any]:
    """Map algorithm name to relevant argparse fields."""
    if algorithm == "kmeans":
        return {
            "n_clusters": args.n_clusters,
            "random_state": args.random_state,
        }
    elif algorithm == "agglomerative":
        return {
            "n_clusters": args.n_clusters,
            "linkage": args.linkage,
        }
    elif algorithm == "hdbscan":
        return {
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
        }
    return {}


def _build_l2_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """Build L2 algorithm kwargs from argparse."""
    if args.l2_algorithm == "kmeans":
        return {
            "n_clusters": args.l2_n_clusters,
            "random_state": args.random_state,
        }
    elif args.l2_algorithm == "agglomerative":
        return {
            "n_clusters": args.l2_n_clusters,
            "linkage": args.linkage,
        }
    elif args.l2_algorithm == "hdbscan":
        return {
            "min_cluster_size": args.l2_min_cluster_size,
            "min_samples": args.min_samples,
        }
    return {}


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------
def save_flat_results(
    dataset_dir: Path,
    run_id: str,
    manifest_df: pd.DataFrame,
    labels: np.ndarray,
    params: dict,
    metrics: dict,
    features_file: str,
    features_shape: tuple,
    dataset_name: str,
    elapsed: float,
) -> Path:
    """Save flat clustering results to CSV + JSON metadata."""
    csv_path = dataset_dir / f"clustering_{run_id}.csv"
    meta_path = dataset_dir / f"clustering_{run_id}_meta.json"

    df = pd.DataFrame({
        "track_id": manifest_df["track_id"].values,
        "cluster_label": labels,
    })
    df.to_csv(csv_path, index=False)

    meta = {
        "run_id": run_id,
        "mode": "flat",
        "params": params,
        "metrics": metrics,
        "features_file": features_file,
        "features_shape": list(features_shape),
        "dataset_name": dataset_name,
        "date": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 3),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Saved: {csv_path}")
    print(f"[INFO] Saved: {meta_path}")
    return csv_path


def save_hierarchical_results(
    dataset_dir: Path,
    run_id: str,
    manifest_df: pd.DataFrame,
    l1_labels: np.ndarray,
    composite_labels: np.ndarray,
    metadata: dict,
    features_file: str,
    features_shape: tuple,
    dataset_name: str,
    elapsed: float,
) -> Path:
    """Save hierarchical clustering results to CSV + JSON metadata."""
    csv_path = dataset_dir / f"clustering_{run_id}.csv"
    meta_path = dataset_dir / f"clustering_{run_id}_meta.json"

    df = pd.DataFrame({
        "track_id": manifest_df["track_id"].values,
        "l1_label": l1_labels,
        "composite_label": composite_labels,
    })
    df.to_csv(csv_path, index=False)

    metadata["run_id"] = run_id
    metadata["features_file"] = features_file
    metadata["features_shape"] = list(features_shape)
    metadata["dataset_name"] = dataset_name
    metadata["date"] = datetime.now().isoformat()
    metadata["elapsed_seconds"] = round(elapsed, 3)

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"[INFO] Saved: {csv_path}")
    print(f"[INFO] Saved: {meta_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------
def print_flat_summary(labels: np.ndarray, metrics: dict, elapsed: float) -> None:
    """Print a formatted summary table for flat clustering."""
    print()
    print("=" * 70)
    print("Clustering Summary (Flat)")
    print("=" * 70)

    # Cluster sizes
    print(f"\n  {'Cluster':<12} {'Size':>8} {'% Total':>10}")
    print(f"  {'-' * 30}")
    n_total = len(labels)
    for lbl, size in sorted(metrics.get("cluster_sizes", {}).items(),
                            key=lambda x: int(x[0])):
        pct = size / n_total * 100
        print(f"  {str(lbl):<12} {size:>8} {pct:>9.1f}%")
    if metrics.get("n_noise", 0) > 0:
        n_noise = metrics["n_noise"]
        pct = n_noise / n_total * 100
        print(f"  {'Noise':<12} {n_noise:>8} {pct:>9.1f}%")

    # Metrics
    print(f"\n  Metrics:")
    for key in ("silhouette", "calinski_harabasz", "davies_bouldin"):
        val = metrics.get(key)
        if val is not None:
            print(f"    {key:<22}: {val:.4f}")
        else:
            print(f"    {key:<22}: N/A")

    print(f"\n  Total tracks : {n_total}")
    print(f"  Clusters     : {metrics.get('n_clusters', 0)}")
    print(f"  Elapsed      : {elapsed:.3f}s")


def print_hierarchical_summary(
    l1_labels: np.ndarray,
    composite_labels: np.ndarray,
    metadata: dict,
    elapsed: float,
) -> None:
    """Print formatted summary for hierarchical clustering."""
    print()
    print("=" * 70)
    print("Clustering Summary (Hierarchical)")
    print("=" * 70)

    n_total = len(l1_labels)
    l1_meta = metadata.get("l1", {})
    l2_meta = metadata.get("l2", {})
    per_cluster = l2_meta.get("per_cluster", {})

    # L1 overview
    print(f"\n  L1 Algorithm : {l1_meta.get('algorithm', '?')}")
    print(f"  L1 Clusters  : {l1_meta.get('n_clusters', '?')}")
    l1_metrics = l1_meta.get("metrics", {})
    sil = l1_metrics.get("silhouette")
    if sil is not None:
        print(f"  L1 Silhouette: {sil:.4f}")

    # Per-cluster L2 breakdown
    print(f"\n  {'L1':<6} {'Size':>6} {'L2 Subs':>8} {'Status':>10}")
    print(f"  {'-' * 32}")
    for letter in sorted(per_cluster.keys()):
        info = per_cluster[letter]
        n_pts = info.get("n_points", 0)
        n_subs = info.get("n_subclusters", 0)
        status = "skipped" if info.get("skipped") else "ok"
        print(f"  {letter:<6} {n_pts:>6} {n_subs:>8} {status:>10}")

    # Noise
    n_noise = int(np.sum(l1_labels == "Noise"))
    if n_noise > 0:
        print(f"  {'Noise':<6} {n_noise:>6}")

    # Composite label distribution
    unique_comp, counts_comp = np.unique(composite_labels, return_counts=True)
    print(f"\n  Composite labels ({len(unique_comp)} unique):")
    for lbl, cnt in sorted(zip(unique_comp, counts_comp), key=lambda x: str(x[0])):
        print(f"    {str(lbl):<12} : {cnt}")

    print(f"\n  Total tracks : {n_total}")
    print(f"  Elapsed      : {elapsed:.3f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRAKTOR ML V3 — Clustering Orchestrator",
    )
    parser.add_argument(
        "--dataset-name", type=str, required=True,
        help="Dataset name (reads artifacts/dataset/<name>/)",
    )
    parser.add_argument(
        "--mode", type=str, default="flat", choices=["flat", "hierarchical"],
        help="Clustering mode (default: flat)",
    )
    parser.add_argument(
        "--algorithm", type=str, default="kmeans",
        choices=list(ALGORITHMS.keys()),
        help="Algorithm for flat mode or L1 in hierarchical (default: kmeans)",
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Custom run ID (default: auto-generated from mode+algorithm+timestamp)",
    )
    parser.add_argument(
        "--features-file", type=str, default=DEFAULT_INPUT_FEATURES,
        help=f"Input features file name (default: {DEFAULT_INPUT_FEATURES})",
    )

    # Algorithm-specific params (flat or L1)
    parser.add_argument(
        "--n-clusters", type=int, default=5,
        help="Number of clusters for KMeans/Agglomerative (default: 5)",
    )
    parser.add_argument(
        "--linkage", type=str, default="ward",
        help="Linkage for Agglomerative (default: ward)",
    )
    parser.add_argument(
        "--min-cluster-size", type=int, default=5,
        help="HDBSCAN min_cluster_size (default: 5)",
    )
    parser.add_argument(
        "--min-samples", type=int, default=3,
        help="HDBSCAN min_samples (default: 3)",
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random seed (default: 42)",
    )

    # Hierarchical-specific params
    parser.add_argument(
        "--l2-algorithm", type=str, default="kmeans",
        choices=list(ALGORITHMS.keys()),
        help="L2 algorithm for hierarchical mode (default: kmeans)",
    )
    parser.add_argument(
        "--l2-n-clusters", type=int, default=3,
        help="L2 n_clusters (default: 3)",
    )
    parser.add_argument(
        "--l2-min-cluster-size", type=int, default=3,
        help="L2 HDBSCAN min_cluster_size (default: 3)",
    )
    parser.add_argument(
        "--l2-min-points", type=int, default=10,
        help="Min points in L1 cluster to attempt L2 (default: 10)",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "artifacts" / "dataset" / args.dataset_name

    # --- Header ---
    print("=" * 70)
    print("TRAKTOR ML V3 — Clustering")
    print("=" * 70)
    print(f"  Dataset     : {args.dataset_name}")
    print(f"  Mode        : {args.mode}")
    print(f"  Algorithm   : {args.algorithm}")
    if args.mode == "hierarchical":
        print(f"  L2 Algorithm: {args.l2_algorithm}")
        print(f"  L2 Min Pts  : {args.l2_min_points}")
    print(f"  Features    : {args.features_file}")
    print()

    # --- Load data ---
    try:
        X, manifest_df = load_data(dataset_dir, args.features_file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    print(f"[INFO] Loaded features: {X.shape}")
    print(f"[INFO] Manifest tracks: {len(manifest_df)} (aligned with features)")

    run_id = generate_run_id(args.algorithm, args.mode, args.run_id)
    print(f"[INFO] Run ID: {run_id}")

    t0 = time.time()

    if args.mode == "flat":
        algo_kwargs = _build_algo_kwargs(args, args.algorithm)
        clusterer = create_clusterer(args.algorithm, **algo_kwargs)

        print(f"[INFO] Running {args.algorithm} clustering...")
        labels = clusterer.fit_predict(X)
        metrics = compute_metrics(X, labels)
        elapsed = time.time() - t0

        save_flat_results(
            dataset_dir, run_id, manifest_df, labels,
            clusterer.get_params(), metrics,
            args.features_file, X.shape, args.dataset_name, elapsed,
        )
        print_flat_summary(labels, metrics, elapsed)

    elif args.mode == "hierarchical":
        l1_params = _build_algo_kwargs(args, args.algorithm)
        l2_params = _build_l2_kwargs(args)

        hc = HierarchicalClusterer(
            l1_algorithm=args.algorithm,
            l1_params=l1_params,
            l2_algorithm=args.l2_algorithm,
            l2_params=l2_params,
            l2_min_points=args.l2_min_points,
        )

        print(f"[INFO] Running hierarchical clustering "
              f"(L1={args.algorithm}, L2={args.l2_algorithm})...")
        l1_labels, composite_labels, metadata = hc.fit_predict(X)
        elapsed = time.time() - t0

        save_hierarchical_results(
            dataset_dir, run_id, manifest_df,
            l1_labels, composite_labels, metadata,
            args.features_file, X.shape, args.dataset_name, elapsed,
        )
        print_hierarchical_summary(l1_labels, composite_labels, metadata, elapsed)

    # --- Footer ---
    print("\n" + "=" * 70)
    print(f"[SUCCESS] Clustering complete! Run ID: {run_id}")
    print(f"[SUCCESS] Output: artifacts/dataset/{args.dataset_name}/clustering_{run_id}.csv")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
