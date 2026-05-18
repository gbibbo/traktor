"""
PURPOSE: Driver for the Regime 1 similarity sweep (Task D4.1). Loads the
         committed sweep grid, assembles the deterministic run plan (capped
         sampled configs + fixed baselines + diagnostic-only V2/V4 rows),
         optionally executes clustering on frozen MERT embeddings, scores
         manual triplet accuracy and cluster diagnostics, and writes the
         leaderboard, per-config component metrics, run metadata, and the
         resolved config under the ignored runs/ tree.

         Modes:
           --plan-only       build and print the run plan; write nothing.
           --max-execute N   execute only the first N executable configs
                             (D4.1b validation subset).
           default           execute the full capped sweep (D4.1c).

         CPU-only; no MERT inference, no audio, no GPU. This script is the
         D4.1a scaffold: it is implemented but not executed during D4.1a.

CHANGELOG:
  D4.1a - Initial implementation (scaffold; sweep not executed).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import yaml

from src.dj_clustering.clustering import (
    DEFAULT_SEED,
    apply_noise_policy,
    cluster_agglomerative,
    cluster_hdbscan,
    cluster_kmeans,
    normalize_embeddings,
    reduce_dimensions,
)
from src.dj_clustering.metrics import (
    NO_USABLE_1001_SOURCE,
    cluster_diagnostics,
    evidence_status,
    filter_non_skip,
    load_manual_triplets,
    tracklists_1001_evidence,
    triplet_accuracy,
)
from src.dj_clustering.sweep import assemble_run_plan, load_sweep_config

BANNER = "=" * 70


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DJ clustering D4.1: Regime 1 similarity sweep runner"
    )
    p.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs/dj_clustering/sweep_regime1.yaml",
        help="committed Regime 1 sweep grid config",
    )
    p.add_argument(
        "--features-dir",
        type=Path,
        default=project_root / "artifacts/dj_clustering/features",
        help="root of frozen MERT feature artifacts",
    )
    p.add_argument(
        "--triplets",
        type=Path,
        default=project_root / "artifacts/dj_clustering/triplets/manual_triplets.csv",
        help="ingested manual triplet answers",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=project_root / "runs/dj_clustering/first_sweep",
        help="ignored output directory for sweep artifacts",
    )
    p.add_argument(
        "--tracklists-1001-status",
        type=str,
        default=NO_USABLE_1001_SOURCE,
        help="1001Tracklists source status (D3.4)",
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="clustering / reduction seed"
    )
    p.add_argument(
        "--max-execute",
        type=int,
        default=None,
        help="execute only the first N executable configs (validation subset)",
    )
    p.add_argument(
        "--plan-only",
        action="store_true",
        help="build and print the run plan only; write no artifacts",
    )
    return p.parse_args()


def _transform(matrix: np.ndarray, cfg: Dict, seed: int) -> np.ndarray:
    """Normalization + dim reduction pipeline shared by clustering and scoring."""
    normalized = normalize_embeddings(matrix, cfg["normalization"])
    return reduce_dimensions(normalized, cfg["dim_reduction"], seed=seed)


def _cluster_from_transformed(
    transformed: np.ndarray, cfg: Dict, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the config's clusterer on an already-transformed matrix.

    Returns (metric_labels, export_labels).
    """
    clusterer = cfg["clusterer"]
    if clusterer == "hdbscan":
        labels = cluster_hdbscan(
            transformed,
            cfg["hdbscan_min_cluster_size"],
            cfg["hdbscan_min_samples"],
        )
        return apply_noise_policy(labels, transformed, cfg["noise_policy"])
    if clusterer == "kmeans":
        labels = cluster_kmeans(transformed, cfg["kmeans_k"], seed=seed)
        return labels, labels.copy()
    if clusterer == "agglomerative":
        labels = cluster_agglomerative(
            transformed, cfg["agglomerative_k"], cfg["agglomerative_linkage"]
        )
        return labels, labels.copy()
    raise ValueError(f"unknown clusterer: {clusterer}")


def _execute_config(
    cfg: Dict,
    embedding_cache: Dict[Tuple[str, str], Tuple[np.ndarray, Dict[str, int]]],
    features_dir: Path,
    triplets: pd.DataFrame,
    seed: int,
) -> Dict:
    """Execute a single executable config and return its leaderboard row."""
    from src.dj_clustering.clustering import load_embedding_source

    key = (cfg["embedding_source"], cfg["mert_aggregation"])
    if key not in embedding_cache:
        matrix, track_ids = load_embedding_source(features_dir, *key)
        embedding_cache[key] = (matrix, {tid: i for i, tid in enumerate(track_ids)})
    matrix, track_index = embedding_cache[key]

    transformed = _transform(matrix, cfg, seed)
    metric_labels, _export_labels = _cluster_from_transformed(transformed, cfg, seed)

    acc = triplet_accuracy(triplets, transformed, track_index)
    diag = cluster_diagnostics(metric_labels, cfg["clusterer"])

    return {
        "config_id": cfg["config_id"],
        "kind": cfg["kind"],
        "baseline_name": cfg.get("baseline_name"),
        "embedding_source": cfg["embedding_source"],
        "mert_aggregation": cfg["mert_aggregation"],
        "normalization": cfg["normalization"],
        "dim_reduction": cfg["dim_reduction"],
        "clusterer": cfg["clusterer"],
        "hdbscan_min_cluster_size": cfg.get("hdbscan_min_cluster_size"),
        "hdbscan_min_samples": cfg.get("hdbscan_min_samples"),
        "noise_policy": cfg.get("noise_policy"),
        "kmeans_k": cfg.get("kmeans_k"),
        "agglomerative_k": cfg.get("agglomerative_k"),
        "triplet_accuracy": acc["accuracy"],
        "triplet_n_scored": acc["n_scored"],
        "triplet_n_unscored": acc["n_unscored"],
        "cluster_count": diag["cluster_count"],
        "largest_cluster_share": diag["largest_cluster_share"],
        "singleton_count": diag["singleton_count"],
        "raw_noise_rate": diag["raw_noise_rate"],
        "executable": True,
        "status": "ok",
    }


def _diagnostic_row(cfg: Dict) -> Dict:
    """Build a clearly-marked, non-executed diagnostic reference leaderboard row."""
    return {
        "config_id": cfg["config_id"],
        "kind": "diagnostic",
        "baseline_name": None,
        "embedding_source": None,
        "mert_aggregation": None,
        "normalization": None,
        "dim_reduction": None,
        "clusterer": None,
        "hdbscan_min_cluster_size": None,
        "hdbscan_min_samples": None,
        "noise_policy": None,
        "kmeans_k": None,
        "agglomerative_k": None,
        "triplet_accuracy": None,
        "triplet_n_scored": None,
        "triplet_n_unscored": None,
        "cluster_count": None,
        "largest_cluster_share": None,
        "singleton_count": None,
        "raw_noise_rate": None,
        "executable": False,
        "status": "diagnostic_only",
        "reference_name": cfg.get("reference_name"),
    }


def run_sweep(
    plan: Dict,
    config: Dict,
    features_dir: Path,
    triplets: pd.DataFrame,
    out_dir: Path,
    seed: int,
    tracklists_1001_status: str,
    max_execute: Optional[int] = None,
) -> Dict:
    """Execute the run plan and write all sweep artifacts under out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    executable = list(plan["baselines"]) + list(plan["sampled_configs"])
    if max_execute is not None:
        executable = executable[:max_execute]

    embedding_cache: Dict[Tuple[str, str], Tuple[np.ndarray, Dict[str, int]]] = {}
    rows: List[Dict] = []
    started = time.time()
    for idx, cfg in enumerate(executable, start=1):
        try:
            rows.append(
                _execute_config(cfg, embedding_cache, features_dir, triplets, seed)
            )
        except Exception as exc:  # record and continue; one bad config must not abort
            print(f"[ERROR] config {idx}/{len(executable)} failed: {type(exc).__name__}")
            rows.append(
                {
                    "config_id": cfg.get("config_id"),
                    "kind": cfg.get("kind"),
                    "executable": True,
                    "status": f"failed:{type(exc).__name__}",
                }
            )
        if idx % 25 == 0:
            print(f"[INFO] executed {idx}/{len(executable)} configs")

    for cfg in plan["diagnostic_rows"]:
        rows.append(_diagnostic_row(cfg))

    leaderboard = pd.DataFrame(rows)
    sort_key = leaderboard["triplet_accuracy"].fillna(-1.0)
    leaderboard = leaderboard.assign(_sort=sort_key).sort_values(
        "_sort", ascending=False
    ).drop(columns="_sort")

    leaderboard_path = out_dir / "first_sweep_leaderboard.csv"
    component_path = out_dir / "component_metrics.csv"
    metadata_path = out_dir / "run_metadata.json"
    resolved_path = out_dir / "resolved_sweep_config.yaml"

    leaderboard.to_csv(leaderboard_path, index=False)
    leaderboard.to_csv(component_path, index=False)
    with resolved_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    non_skip = filter_non_skip(triplets)
    metadata = {
        "task": "D4.1",
        "seed": seed,
        "elapsed_seconds": round(time.time() - started, 2),
        "grid_total": plan["grid_total"],
        "n_sampled": plan["n_sampled"],
        "n_baselines": plan["n_baselines"],
        "n_diagnostic": plan["n_diagnostic"],
        "n_executed": len(executable),
        "max_execute": max_execute,
        "n_non_skip_triplets": int(len(non_skip)),
        "evidence_status": evidence_status(len(non_skip)),
        "tracklists_1001": tracklists_1001_evidence(tracklists_1001_status),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return metadata


def main() -> int:
    args = parse_args()
    print(BANNER)
    print("DJ clustering D4.1 - Regime 1 similarity sweep runner")
    print(BANNER)

    config = load_sweep_config(args.config)
    plan = assemble_run_plan(config)
    print(f"[INFO] grid total non-baseline configs: {plan['grid_total']}")
    print(f"[INFO] sampled configs (cap {plan['max_configs']}): {plan['n_sampled']}")
    print(f"[INFO] fixed baselines: {plan['n_baselines']}")
    print(f"[INFO] diagnostic-only references: {plan['n_diagnostic']}")
    print(f"[INFO] executable configs: {plan['n_executable']}")

    if args.plan_only:
        print("[INFO] --plan-only: no clustering, no artifacts written")
        print(BANNER)
        return 0

    if not args.triplets.exists():
        print(f"[ERROR] manual triplets not found: {args.triplets}")
        return 1
    triplets = load_manual_triplets(args.triplets)

    metadata = run_sweep(
        plan=plan,
        config=config,
        features_dir=args.features_dir,
        triplets=triplets,
        out_dir=args.out_dir,
        seed=args.seed,
        tracklists_1001_status=args.tracklists_1001_status,
        max_execute=args.max_execute,
    )
    print(f"[INFO] executed {metadata['n_executed']} configs "
          f"in {metadata['elapsed_seconds']}s")
    print(f"[INFO] non-skip triplets: {metadata['n_non_skip_triplets']} "
          f"({metadata['evidence_status']['classification']})")
    print(f"[INFO] artifacts written under: {args.out_dir}")
    print(BANNER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
