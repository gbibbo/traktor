#!/usr/bin/env python3
"""
PURPOSE: Fit PCA on MERT embeddings for TRAKTOR ML V3 pipeline.
         Applies L2 normalization then reduces dimensionality from 1024 to 128,
         producing the final feature matrix for clustering.

CHANGELOG:
    2026-02-05: Initial implementation for V3 pipeline.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_N_COMPONENTS = 128


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_embeddings(dataset_dir: Path) -> Tuple[np.ndarray, List[str]]:
    """Load MERT embeddings and track IDs from the embeddings directory.

    Returns
    -------
    embeddings : np.ndarray, shape [N, 1024]
    track_ids  : list[str]
    """
    emb_dir = dataset_dir / "embeddings"
    emb_path = emb_dir / "embeddings_mert.npy"
    ids_path = emb_dir / "track_ids.npy"

    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")
    if not ids_path.exists():
        raise FileNotFoundError(f"Track IDs file not found: {ids_path}")

    embeddings = np.load(emb_path)
    track_ids = np.load(ids_path, allow_pickle=True).tolist()

    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
    if embeddings.shape[0] != len(track_ids):
        raise ValueError(
            f"Shape mismatch: {embeddings.shape[0]} embeddings vs {len(track_ids)} track IDs"
        )

    return embeddings, track_ids


# ---------------------------------------------------------------------------
# PCA pipeline
# ---------------------------------------------------------------------------
def fit_and_transform(
    embeddings: np.ndarray,
    n_components: int = DEFAULT_N_COMPONENTS,
) -> Tuple[np.ndarray, PCA, Normalizer]:
    """L2-normalize embeddings and fit PCA.

    Returns
    -------
    X_pca       : np.ndarray, shape [N, n_components]
    pca         : fitted PCA object
    normalizer  : fitted Normalizer object
    """
    normalizer = Normalizer(norm="l2")
    X_normed = normalizer.fit_transform(embeddings)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_normed)

    return X_pca, pca, normalizer


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save_results(
    dataset_dir: Path,
    X_pca: np.ndarray,
    pca: PCA,
    normalizer: Normalizer,
    n_components: int,
    original_dim: int,
) -> Path:
    """Save PCA results and fitted objects."""
    output_path = dataset_dir / f"X_pca{n_components}.npy"
    pca_path = dataset_dir / f"pca_{n_components}.joblib"
    norm_path = dataset_dir / "normalizer.joblib"
    meta_path = dataset_dir / "pca_metadata.json"

    np.save(output_path, X_pca)
    joblib.dump(pca, pca_path)
    joblib.dump(normalizer, norm_path)

    variance_cumulative = float(np.sum(pca.explained_variance_ratio_))
    meta = {
        "n_components": n_components,
        "original_dim": original_dim,
        "n_tracks": int(X_pca.shape[0]),
        "variance_explained": round(variance_cumulative, 6),
        "date": datetime.now().isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit PCA on MERT embeddings for TRAKTOR ML V3",
    )
    parser.add_argument(
        "--dataset-name", type=str, required=True,
        help="Dataset name (reads artifacts/dataset/<name>/embeddings/)",
    )
    parser.add_argument(
        "--n-components", type=int, default=DEFAULT_N_COMPONENTS,
        help=f"Number of PCA dimensions (default: {DEFAULT_N_COMPONENTS})",
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
    print("TRAKTOR ML V3 — Fit PCA")
    print("=" * 70)
    print(f"  Dataset       : {args.dataset_name}")
    print(f"  Input dir     : {dataset_dir / 'embeddings'}")
    print(f"  N components  : {args.n_components}")
    print()

    # --- Load embeddings ---
    try:
        embeddings, track_ids = load_embeddings(dataset_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    print(f"[INFO] Loaded embeddings: {embeddings.shape}")

    # --- Fit PCA ---
    t0 = time.time()
    X_pca, pca, normalizer = fit_and_transform(embeddings, args.n_components)
    elapsed = time.time() - t0

    # --- Validate ---
    if np.any(np.isnan(X_pca)) or np.any(np.isinf(X_pca)):
        print("[ERROR] Output contains NaN or Inf values!")
        return 1

    # --- Save ---
    output_path = save_results(
        dataset_dir, X_pca, pca, normalizer,
        args.n_components, embeddings.shape[1],
    )

    # --- Summary ---
    variance_pct = np.sum(pca.explained_variance_ratio_) * 100
    print()
    print("=" * 70)
    print("PCA Summary")
    print("=" * 70)
    print(f"  Components       : {args.n_components} / {embeddings.shape[1]}")
    print(f"  Variance retained: {variance_pct:.1f}%")
    print(f"  Output shape     : {X_pca.shape}")
    size_kb = output_path.stat().st_size / 1024
    print(f"  Output file      : {output_path} ({size_kb:.1f} KB)")
    print(f"  Elapsed time     : {elapsed:.3f}s")

    # Top-5 components contribution
    top5 = pca.explained_variance_ratio_[:5] * 100
    print(f"\n  Top-5 components : {', '.join(f'{v:.1f}%' for v in top5)}")

    print("\n" + "=" * 70)
    print("[SUCCESS] PCA fitted successfully!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
