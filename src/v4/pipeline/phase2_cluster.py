"""
PURPOSE: Phase 2 — Clustering jerárquico L1/L2 sobre embeddings MERT.
         L1: HDBSCAN sobre mert_perc (groove/percusión).
         L2: HDBSCAN sobre mert_full dentro de cada cluster L1 con ≥ 8 tracks.
         UMAP 2D para visualización.
         Guarda clustering/results_<hash>.parquet y clustering/config_<hash>.json.
CHANGELOG:
  - 2026-02-28: Creación inicial V4 (Block 3).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.v4.common.config_loader import load_config
from src.v4.common.path_resolver import resolve_dataset_artifacts


# ---------------------------------------------------------------------------
# Contrato de etiquetas (documentado aquí y en config_<hash>.json):
#   label_l1 = -1  → ruido L1 (HDBSCAN outlier)
#   label_l2 = -1  → si label_l1 == -1 (sin padre) O ruido dentro del L2 HDBSCAN
#   label_l2 = 0   → cluster L1 con < L2_MIN_PARENT_SIZE tracks (subgrupo único trivial)
#   label_l2 >= 0  → subcluster real producido por HDBSCAN L2
# ---------------------------------------------------------------------------

L2_MIN_PARENT_SIZE = 8  # Mínimo de tracks en un cluster L1 para aplicar HDBSCAN L2


def _hash_config(cfg: dict) -> str:
    """Hash MD5 de 8 chars del config de clustering."""
    serialized = json.dumps(cfg, sort_keys=True)
    return hashlib.md5(serialized.encode()).hexdigest()[:8]


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalizar filas."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def _hdbscan_cluster(
    X: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
) -> np.ndarray:
    """Ejecutar HDBSCAN sobre X. Retorna array de labels (int)."""
    try:
        import hdbscan
    except ImportError:
        raise ImportError(
            "hdbscan no encontrado. Instalar: pip install hdbscan"
        )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        core_dist_n_jobs=-1,
    )
    clusterer.fit(X)
    return clusterer.labels_.astype(int)


def _umap_2d(X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """UMAP 2D para visualización. Retorna (N, 2) array."""
    try:
        import umap
    except ImportError:
        raise ImportError(
            "umap-learn no encontrado. Instalar: pip install umap-learn"
        )
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, len(X) - 1),
        min_dist=min_dist,
        random_state=42,
        metric="cosine",
    )
    return reducer.fit_transform(X).astype(np.float32)


def run_clustering(
    dataset_name: str,
    config: dict,
    l1_min_cluster_size: int,
    l1_min_samples: int,
    l2_min_cluster_size: int,
    l2_min_samples: int,
    config_tag: str = "baseline",
    skip_umap: bool = False,
) -> Path:
    """Ejecutar clustering L1/L2 y guardar resultados.

    Args:
        dataset_name: nombre del dataset.
        config: configuración cargada.
        l1_min_cluster_size, l1_min_samples: parámetros HDBSCAN L1.
        l2_min_cluster_size, l2_min_samples: parámetros HDBSCAN L2.
        config_tag: etiqueta legible para el run (no afecta el hash).
        skip_umap: si True, omite UMAP (útil para tests rápidos).

    Returns:
        Path al parquet de resultados.
    """
    t0 = time.time()
    artifacts_dir = resolve_dataset_artifacts(dataset_name, config)
    embeddings_dir = artifacts_dir / "embeddings"
    clustering_dir = artifacts_dir / "clustering"
    clustering_dir.mkdir(parents=True, exist_ok=True)

    # --- Cargar embeddings ---
    uids_path = embeddings_dir / "track_uids.json"
    perc_path = embeddings_dir / "mert_perc.npy"
    full_path = embeddings_dir / "mert_full.npy"

    for p in (uids_path, perc_path, full_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Embedding artifact not found: {p}\n"
                "Run phase1_extract.py + phase1_merge_shards.py first."
            )

    with open(uids_path) as f:
        track_uids: List[str] = json.load(f)

    mert_perc = np.load(perc_path)  # (N, 1024)
    mert_full = np.load(full_path)  # (N, 1024)
    N = len(track_uids)

    if mert_perc.shape[0] != N or mert_full.shape[0] != N:
        raise ValueError(
            f"Shape mismatch: track_uids={N}, mert_perc={mert_perc.shape[0]}, "
            f"mert_full={mert_full.shape[0]}"
        )

    print(f"[INFO] Loaded {N} tracks, embeddings shape: {mert_perc.shape}")

    # --- Cargar catálogo y alinear por track_uid ---
    catalog_path = artifacts_dir / "catalog.parquet"
    if catalog_path.exists():
        catalog = pd.read_parquet(catalog_path)
        uid_to_row = {uid: idx for idx, uid in enumerate(track_uids)}
        # Reindexar catálogo según orden de track_uids.json
        catalog = catalog.set_index("track_uid").reindex(track_uids).reset_index()
        n_missing = catalog["track_uid"].isna().sum()
        if n_missing > 0:
            print(f"[WARN] {n_missing} UIDs from track_uids.json not found in catalog.parquet")
    else:
        catalog = pd.DataFrame({"track_uid": track_uids})

    # --- L2-normalizar ---
    perc_norm = _l2_normalize(mert_perc)
    full_norm = _l2_normalize(mert_full)

    # --- L1 Clustering (sobre mert_perc) ---
    print(f"[INFO] Running L1 HDBSCAN (min_cluster_size={l1_min_cluster_size}, "
          f"min_samples={l1_min_samples}) ...")
    labels_l1 = _hdbscan_cluster(perc_norm, l1_min_cluster_size, l1_min_samples)

    n_clusters_l1 = len(set(labels_l1[labels_l1 != -1]))
    n_noise_l1 = int(np.sum(labels_l1 == -1))
    print(f"[INFO] L1 result: {n_clusters_l1} clusters, {n_noise_l1} noise points "
          f"({n_noise_l1/N:.1%})")

    # --- L2 Clustering (sobre mert_full, dentro de cada cluster L1) ---
    labels_l2 = np.full(N, -1, dtype=int)
    l2_small_cluster_ids = []

    for cl1 in sorted(set(labels_l1[labels_l1 != -1])):
        mask = labels_l1 == cl1
        n_in_cluster = int(np.sum(mask))

        if n_in_cluster < L2_MIN_PARENT_SIZE:
            # Cluster demasiado pequeño: subgrupo único trivial
            labels_l2[mask] = 0
            l2_small_cluster_ids.append(int(cl1))
        else:
            sub_emb = full_norm[mask]
            sub_labels = _hdbscan_cluster(sub_emb, l2_min_cluster_size, l2_min_samples)
            labels_l2[mask] = sub_labels
            n_sub = len(set(sub_labels[sub_labels != -1]))
            print(f"  L1={cl1}: {n_in_cluster} tracks → {n_sub} subclusters")

    # label_l2 de puntos de ruido L1 permanece -1 (ya inicializado)

    # --- UMAP 2D ---
    if skip_umap:
        umap_coords = np.zeros((N, 2), dtype=np.float32)
        print("[INFO] UMAP skipped (--skip-umap)")
    else:
        print("[INFO] Running UMAP 2D ...")
        umap_coords = _umap_2d(full_norm)
        print(f"[INFO] UMAP done: shape={umap_coords.shape}")

    # --- Construir config hash ---
    cluster_cfg = {
        "l1_min_cluster_size": l1_min_cluster_size,
        "l1_min_samples": l1_min_samples,
        "l2_min_cluster_size": l2_min_cluster_size,
        "l2_min_samples": l2_min_samples,
        "l2_min_parent_size": L2_MIN_PARENT_SIZE,
        "config_tag": config_tag,
        "dataset_name": dataset_name,
    }
    config_hash = _hash_config(cluster_cfg)

    # --- Guardar parquet ---
    result_df = pd.DataFrame({
        "track_uid": track_uids,
        "label_l1": labels_l1.tolist(),
        "label_l2": labels_l2.tolist(),
        "umap_x": umap_coords[:, 0].tolist(),
        "umap_y": umap_coords[:, 1].tolist(),
    })
    results_path = clustering_dir / f"results_{config_hash}.parquet"
    result_df.to_parquet(results_path, index=False)
    print(f"[INFO] Saved results: {results_path} ({len(result_df)} rows)")

    # --- Guardar config con label_semantics ---
    config_out = {
        **cluster_cfg,
        "config_hash": config_hash,
        "n_tracks": N,
        "n_clusters_l1": n_clusters_l1,
        "n_noise_l1": n_noise_l1,
        "l2_small_cluster_ids": l2_small_cluster_ids,
        "label_semantics": {
            "label_l1=-1": "noise (HDBSCAN L1 outlier)",
            "label_l2=-1": "noise L2 or no parent (label_l1==-1)",
            "label_l2=0_if_small": (
                f"label_l2=0 when parent L1 cluster has <{L2_MIN_PARENT_SIZE} tracks "
                "(trivial single subgroup — not a real HDBSCAN result)"
            ),
        },
        "elapsed_s": round(time.time() - t0, 2),
    }
    config_path = clustering_dir / f"config_{config_hash}.json"
    with open(config_path, "w") as f:
        json.dump(config_out, f, indent=2)
    print(f"[INFO] Saved config:  {config_path}")

    # --- Resumen ---
    elapsed = round(time.time() - t0, 2)
    print("=" * 70)
    print(f"[INFO] Phase 2 clustering complete in {elapsed}s")
    print(f"[INFO] Config hash: {config_hash}")
    print(f"[INFO] L1 clusters: {n_clusters_l1}, noise: {n_noise_l1/N:.1%}")
    print("=" * 70)

    return results_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 2: HDBSCAN clustering L1/L2 sobre embeddings MERT."
    )
    parser.add_argument("--dataset-name", required=True, help="Nombre del dataset.")
    parser.add_argument("--config", default=None, help="Ruta al config YAML.")
    parser.add_argument(
        "--l1-min-cluster-size", type=int, default=None,
        help="HDBSCAN L1 min_cluster_size (default: del config).",
    )
    parser.add_argument(
        "--l1-min-samples", type=int, default=None,
        help="HDBSCAN L1 min_samples (default: del config).",
    )
    parser.add_argument(
        "--l2-min-cluster-size", type=int, default=None,
        help="HDBSCAN L2 min_cluster_size (default: del config).",
    )
    parser.add_argument(
        "--l2-min-samples", type=int, default=None,
        help="HDBSCAN L2 min_samples (default: del config).",
    )
    parser.add_argument(
        "--config-tag", default="baseline",
        help="Etiqueta legible para este run (no afecta el hash).",
    )
    parser.add_argument(
        "--skip-umap", action="store_true",
        help="Omitir UMAP (más rápido, sin visualización 2D).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("[INFO] TRAKTOR ML V4 — Phase 2: Clustering")
    print("=" * 70)

    config = load_config(args.config)
    cluster_cfg = config.get("clustering", {})

    l1_min_cluster_size = args.l1_min_cluster_size or cluster_cfg.get("l1_min_cluster_size", 10)
    l1_min_samples = args.l1_min_samples or cluster_cfg.get("l1_min_samples", 3)
    l2_min_cluster_size = args.l2_min_cluster_size or cluster_cfg.get("l2_min_cluster_size", 4)
    l2_min_samples = args.l2_min_samples or cluster_cfg.get("l2_min_samples", 2)

    try:
        run_clustering(
            dataset_name=args.dataset_name,
            config=config,
            l1_min_cluster_size=l1_min_cluster_size,
            l1_min_samples=l1_min_samples,
            l2_min_cluster_size=l2_min_cluster_size,
            l2_min_samples=l2_min_samples,
            config_tag=args.config_tag,
            skip_umap=args.skip_umap,
        )
        return 0
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 1
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
