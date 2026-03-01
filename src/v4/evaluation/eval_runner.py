"""
PURPOSE: Runner de evaluación automatizada para clustering V4.
         Carga artifacts de Phase 1 y Phase 2, calcula métricas disponibles,
         guarda JSON de resultados. Funciona sin ground truth (solo cluster stats).
CHANGELOG:
  - 2026-02-28: Creación inicial V4 (Block 3).
"""
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.v4.evaluation.metrics import (
    clustering_ari,
    clustering_nmi,
    noise_rate,
    composite_score,
)
from src.v4.common.path_resolver import resolve_dataset_artifacts


def run_evaluation(
    dataset_name: str,
    clustering_config_hash: str,
    config: dict,
    dev_set_path: Optional[Path] = None,
    dj_pairs_path: Optional[Path] = None,
) -> dict:
    """Evalúa un resultado de clustering. Guarda scores en JSON.

    Args:
        dataset_name: nombre del dataset (ej. "test_20").
        clustering_config_hash: hash del config de clustering (identifica el run).
        config: configuración cargada via config_loader.load_config().
        dev_set_path: CSV con columnas [track_uid, label_true]. None = omite ARI/NMI.
        dj_pairs_path: CSV con columnas [track_uid_a, track_uid_b]. None = omite retrieval.

    Returns:
        dict con todas las métricas calculadas y metadatos del run.
    """
    t0 = time.time()
    artifacts_dir = resolve_dataset_artifacts(dataset_name, config)
    clustering_dir = artifacts_dir / "clustering"
    eval_dir = artifacts_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    results_path = clustering_dir / f"results_{clustering_config_hash}.parquet"
    if not results_path.exists():
        raise FileNotFoundError(
            f"Clustering results not found: {results_path}\n"
            "Run phase2_cluster.py first."
        )

    df = pd.read_parquet(results_path)
    labels_l1 = df["label_l1"].to_numpy()
    labels_l2 = df["label_l2"].to_numpy()

    metrics: dict = {}

    # --- Cluster stats (siempre disponibles) ---
    l1_noise = noise_rate(labels_l1)
    l1_labels_assigned = labels_l1[labels_l1 != -1]
    l1_n_clusters = int(len(set(l1_labels_assigned))) if len(l1_labels_assigned) > 0 else 0
    l1_cluster_sizes = (
        [int(np.sum(labels_l1 == c)) for c in sorted(set(l1_labels_assigned))]
        if l1_n_clusters > 0 else []
    )

    metrics["noise_rate_l1"] = l1_noise
    metrics["n_clusters_l1"] = l1_n_clusters
    metrics["cluster_sizes_l1"] = l1_cluster_sizes
    metrics["n_tracks_total"] = int(len(df))
    metrics["n_tracks_noise_l1"] = int(np.sum(labels_l1 == -1))

    # L2 stats (por cluster L1)
    l2_stats = {}
    for cl1 in sorted(set(l1_labels_assigned)):
        mask = labels_l1 == cl1
        l2_sub = labels_l2[mask]
        l2_assigned = l2_sub[l2_sub != -1]
        l2_stats[int(cl1)] = {
            "n_tracks": int(np.sum(mask)),
            "n_subclusters": int(len(set(l2_assigned))) if len(l2_assigned) > 0 else 1,
            "noise_rate_l2": float(noise_rate(l2_sub)),
        }
    metrics["l2_stats_per_l1"] = l2_stats

    # --- ARI/NMI (solo si dev_set_path disponible) ---
    if dev_set_path is not None and Path(dev_set_path).exists():
        dev_df = pd.read_csv(dev_set_path)
        merged = df.merge(dev_df[["track_uid", "label_true"]], on="track_uid", how="inner")
        if len(merged) > 1:
            metrics["ari_l1"] = clustering_ari(
                merged["label_l1"].to_numpy(),
                merged["label_true"].to_numpy(),
            )
            metrics["nmi_l1"] = clustering_nmi(
                merged["label_l1"].to_numpy(),
                merged["label_true"].to_numpy(),
            )
        else:
            print("[WARN] No tracks matched between clustering results and dev set.")
    else:
        print("[INFO] dev_set_path not provided — skipping ARI/NMI.")

    # --- Composite score (solo métricas escalares en [0,1]) ---
    scalar_metrics = {
        k: v for k, v in metrics.items()
        if isinstance(v, float) and k not in ("noise_rate_l1",)  # noise ya es penalización implícita
    }
    metrics["composite_score"] = composite_score(scalar_metrics)

    elapsed = round(time.time() - t0, 2)
    output = {
        "dataset_name": dataset_name,
        "clustering_config_hash": clustering_config_hash,
        "elapsed_s": elapsed,
        "metrics": metrics,
    }

    out_path = eval_dir / f"{clustering_config_hash}_scores.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[INFO] Evaluation saved to: {out_path}")

    return output


def print_clustering_report(eval_result: dict) -> None:
    """Imprime reporte de clustering para revisión humana."""
    m = eval_result["metrics"]
    print("=" * 70)
    print("=== CLUSTERING REPORT (para revisión humana) ===")
    print(f"Dataset     : {eval_result['dataset_name']}")
    print(f"Config hash : {eval_result['clustering_config_hash']}")
    print(f"Tracks total: {m['n_tracks_total']}")
    print(f"Clusters L1 : {m['n_clusters_l1']}")
    print(f"Noise (L1)  : {m['noise_rate_l1']:.1%} ({m['n_tracks_noise_l1']} tracks)")
    print(f"Tamaños L1  : {m['cluster_sizes_l1']}")
    print()
    for cl1, stats in m.get("l2_stats_per_l1", {}).items():
        print(
            f"  L1={cl1:>2}: {stats['n_tracks']:>3} tracks → "
            f"{stats['n_subclusters']} subcluster(s), "
            f"noise_l2={stats['noise_rate_l2']:.0%}"
        )
    if "ari_l1" in m:
        print(f"\nARI (L1)    : {m['ari_l1']:.4f}")
        print(f"NMI (L1)    : {m['nmi_l1']:.4f}")
    print("=" * 70)
