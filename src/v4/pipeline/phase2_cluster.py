"""
PURPOSE: Phase 2 — Clustering jerárquico L1/L2 sobre embeddings MERT.
         L1: HDBSCAN sobre mert_perc (groove/percusión).
         L2: HDBSCAN sobre mert_full dentro de cada cluster L1 con ≥ 8 tracks.
         Reducción PCA opcional antes de HDBSCAN (--pca-dim, default 50).
         UMAP 2D para visualización.
         Guarda clustering/results_<hash>.parquet y clustering/config_<hash>.json.
CHANGELOG:
  - 2026-09-27: --rep: clusterizar con una variante de extract_representations (L1 y L2 sobre la misma).
                --bpm-weight w: agrega el BPM estandarizado (features/bpm_key.parquet) por w como una
                dimensión más tras normalizar (cajones con tempo coherente; decisión de diseño).
                PCA con random_state=0: con N > 500 sklearn usa el solver randomized y, sin semilla,
                los grupos cambiaban entre corridas.
  - 2026-09-29: ward_two_level(): el Ward en dos niveles como función reutilizable (organize.py), con
                semillas (must-link): cada grupo se colapsa en su centroide antes de Ward en ambos
                niveles, así sus temas quedan en la misma carpeta y playlist. Sin semillas, idéntico.
                --method ward: aglomerativo Ward determinista con número de grupos fijado (--n-l1,
                y L2 con --l2-target-size temas por playlist). En CLAP, HDBSCAN deja ~88 % de ruido
                y 2-3 grupos gigantes: el espacio no tiene grupos densos, hace falta particionar.
  - 2026-02-28: Creación inicial V4 (Block 3).
  - 2026-03-01: Añadir PCA pre-HDBSCAN para reducir curse of dimensionality (1024→pca_dim).
  - 2026-03-01: Reemplazar paquete hdbscan por sklearn.cluster.HDBSCAN (evita compilación C en nodos sin Python.h).
  - 2026-03-02: Añadir _reassign_noise() — reasignación 1-NN de puntos noise post-HDBSCAN.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.config_loader import load_config
from src.v4.common.embedding_utils import load_track_embeddings
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
    from sklearn.cluster import HDBSCAN
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        n_jobs=-1,
    )
    clusterer.fit(X)
    return clusterer.labels_.astype(int)


def _ward_cluster(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """Aglomerativo Ward con n_clusters (determinista, sin ruido)."""
    from sklearn.cluster import AgglomerativeClustering
    n_clusters = max(1, min(n_clusters, len(X)))
    if n_clusters == 1:
        return np.zeros(len(X), dtype=int)
    return AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(X).astype(int)


def _collapse(X: np.ndarray, groups: Optional[List[np.ndarray]]):
    """Nodos para Ward: cada grupo (índices de filas de X) se reemplaza por su centroide.
    Devuelve (matriz de nodos, lista de índices de X por nodo)."""
    if not groups:
        return X, [np.array([i]) for i in range(len(X))]
    in_group = np.zeros(len(X), dtype=bool)
    rows, members = [], []
    for g in groups:
        g = np.asarray(g, dtype=int)
        rows.append(X[g].mean(axis=0))
        members.append(g)
        in_group[g] = True
    for i in np.flatnonzero(~in_group):
        rows.append(X[i])
        members.append(np.array([i]))
    return np.vstack(rows).astype(X.dtype), members


def _expand(node_labels: np.ndarray, members: List[np.ndarray], n: int) -> np.ndarray:
    out = np.empty(n, dtype=int)
    for lab, m in zip(node_labels, members):
        out[m] = lab
    return out


def ward_two_level(F_l1: np.ndarray, F_l2: np.ndarray, n_l1: int, l2_target_size: int, pca_dim: int,
                   groups: Optional[List[np.ndarray]] = None):
    """Ward en dos niveles (el de --method ward): L1 sobre PCA(F_l1) con n_l1 grupos; L2 dentro de
    cada L1 con round(n / l2_target_size) grupos, sobre F_l2 (con PCA propia si n >= 2*pca_dim).
    groups: listas de índices que deben quedar juntos (se colapsan en su centroide en ambos niveles).
    Devuelve (labels_l1, labels_l2, ids L1 con menos de L2_MIN_PARENT_SIZE temas)."""
    N = len(F_l1)
    groups = [np.asarray(g, dtype=int) for g in (groups or []) if len(g) > 1]
    X1 = _apply_pca(F_l1, pca_dim, label="L1/perc")
    nodes, members = _collapse(X1, groups)
    print(f"[INFO] Running L1 Ward (n_clusters={n_l1}{f', {len(groups)} semillas' if groups else ''}) ...")
    labels_l1 = _expand(_ward_cluster(nodes, n_l1), members, N)
    labels_l2 = np.full(N, -1, dtype=int)
    small = []
    for cl1 in sorted(set(labels_l1)):
        idx = np.flatnonzero(labels_l1 == cl1)
        n_in = len(idx)
        if n_in < L2_MIN_PARENT_SIZE:
            labels_l2[idx] = 0
            small.append(int(cl1))
            continue
        sub = F_l2[idx]
        if pca_dim > 0 and n_in >= 2 * pca_dim:
            sub = _apply_pca(sub, pca_dim, label=f"L2/cl{cl1}")
        pos = {g: k for k, g in enumerate(idx)}
        sub_groups = [np.array([pos[i] for i in g]) for g in groups if pos.get(int(g[0])) is not None]
        sub_nodes, sub_members = _collapse(sub, sub_groups)
        labels_l2[idx] = _expand(_ward_cluster(sub_nodes, int(round(n_in / l2_target_size))), sub_members, n_in)
        print(f"  L1={cl1}: {n_in} tracks → {len(set(labels_l2[idx]))} subclusters")
    return labels_l1, labels_l2, small


def _hdbscan_two_level(perc_norm, full_norm, pca_dim, l1_min_cluster_size, l1_min_samples,
                       l2_min_cluster_size, l2_min_samples, assign_noise):
    """HDBSCAN L1 sobre PCA(perc_norm) y L2 dentro de cada L1 sobre full_norm (camino histórico)."""
    N = len(perc_norm)
    perc_for_l1 = _apply_pca(perc_norm, pca_dim, label="L1/perc")
    print(f"[INFO] Running L1 HDBSCAN (min_cluster_size={l1_min_cluster_size}, "
          f"min_samples={l1_min_samples}) ...")
    labels_l1 = _hdbscan_cluster(perc_for_l1, l1_min_cluster_size, l1_min_samples)
    labels_l1_raw = labels_l1.copy()  # conservar para diagnóstico
    n_clusters_l1 = len(set(labels_l1[labels_l1 != -1]))
    n_noise_l1 = int(np.sum(labels_l1 == -1))
    print(f"[INFO] L1 result: {n_clusters_l1} clusters, {n_noise_l1} noise points "
          f"({n_noise_l1/N:.1%})")
    if assign_noise:
        labels_l1 = _reassign_noise(perc_for_l1, labels_l1)

    labels_l2 = np.full(N, -1, dtype=int)
    labels_l2_raw = np.full(N, -1, dtype=int)  # conservar para diagnóstico
    l2_small_cluster_ids = []
    for cl1 in sorted(set(labels_l1[labels_l1 != -1])):
        mask = labels_l1 == cl1
        n_in_cluster = int(np.sum(mask))
        if n_in_cluster < L2_MIN_PARENT_SIZE:
            # Cluster demasiado pequeño: subgrupo único trivial
            labels_l2[mask] = 0
            labels_l2_raw[mask] = 0
            l2_small_cluster_ids.append(int(cl1))
            continue
        sub_emb = full_norm[mask]
        # PCA L2: solo si hay suficientes tracks (idealmente > 2*pca_dim)
        if pca_dim > 0 and n_in_cluster >= 2 * pca_dim:
            sub_emb = _apply_pca(sub_emb, pca_dim, label=f"L2/cl{cl1}")
        sub_labels = _hdbscan_cluster(sub_emb, l2_min_cluster_size, l2_min_samples)
        labels_l2_raw[mask] = sub_labels  # conservar raw antes de reassignment
        if assign_noise:
            sub_labels = _reassign_noise(sub_emb, sub_labels)
        labels_l2[mask] = sub_labels
        n_sub = len(set(sub_labels[sub_labels != -1]))
        print(f"  L1={cl1}: {n_in_cluster} tracks → {n_sub} subclusters")
    return labels_l1, labels_l1_raw, labels_l2, labels_l2_raw, l2_small_cluster_ids, n_clusters_l1, n_noise_l1


def _apply_pca(X: np.ndarray, pca_dim: int, label: str = "") -> np.ndarray:
    """Reducir dimensionalidad con PCA. Si pca_dim<=0 o pca_dim>=X.shape[1], retorna X sin cambios."""
    if pca_dim <= 0 or pca_dim >= X.shape[1]:
        return X
    from sklearn.decomposition import PCA
    n_components = min(pca_dim, X.shape[0] - 1)
    pca = PCA(n_components=n_components, whiten=False, random_state=0)  # el solver randomized necesita semilla
    X_reduced = pca.fit_transform(X).astype(np.float32)
    var_retained = float(pca.explained_variance_ratio_.sum())
    tag = f" [{label}]" if label else ""
    print(f"[INFO] PCA{tag}: {X.shape[1]}→{n_components} dims, {var_retained:.1%} variance retained")
    return X_reduced


def _reassign_noise(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Reasignar puntos noise (label=-1) al cluster del vecino no-noise más cercano (1-NN).

    Args:
        X: embeddings en el espacio de clustering (PCA-reducido), shape (N, D)
        labels: array de labels de HDBSCAN, shape (N,), con -1 para noise

    Returns:
        new_labels: array sin -1 (todos asignados), salvo edge case todo-noise
    """
    noise_mask = labels == -1
    n_noise = int(noise_mask.sum())
    if n_noise == 0:
        return labels
    assigned_mask = ~noise_mask
    if not assigned_mask.any():
        return labels  # edge case: todo es noise, nada que hacer
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
    nn.fit(X[assigned_mask])
    _, indices = nn.kneighbors(X[noise_mask])
    assigned_labels = labels[assigned_mask]
    new_labels = labels.copy()
    new_labels[noise_mask] = assigned_labels[indices[:, 0]]
    print(f"[INFO] Noise reassignment: {n_noise} noise points → assigned to nearest cluster")
    return new_labels


def _append_bpm(X: np.ndarray, track_uids: List[str], artifacts_dir: Path, weight: float) -> np.ndarray:
    """Agrega weight * BPM estandarizado como última columna (BPM faltante = mediana)."""
    feats = pd.read_parquet(artifacts_dir / "features" / "bpm_key.parquet").drop_duplicates("track_uid")
    bpm = pd.to_numeric(feats.set_index("track_uid").reindex(track_uids)["bpm"], errors="coerce")
    bpm = bpm.fillna(bpm.median()).to_numpy(dtype=np.float64)
    std = bpm.std() or 1.0
    z = (bpm - bpm.mean()) / std
    print(f"[INFO] BPM agregado al embedding con peso {weight} (media {bpm.mean():.1f}, sd {std:.1f})")
    return np.hstack([X, (weight * z)[:, None]]).astype(np.float32)


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
    pca_dim: int = 0,
    assign_noise: bool = True,
    rep: Optional[str] = None,
    bpm_weight: float = 0.0,
    method: str = "hdbscan",
    n_l1: Optional[int] = None,
    l2_target_size: int = 30,
) -> Path:
    """Ejecutar clustering L1/L2 y guardar resultados.

    Args:
        dataset_name: nombre del dataset.
        config: configuración cargada.
        l1_min_cluster_size, l1_min_samples: parámetros HDBSCAN L1.
        l2_min_cluster_size, l2_min_samples: parámetros HDBSCAN L2.
        config_tag: etiqueta legible para el run (no afecta el hash).
        skip_umap: si True, omite UMAP (útil para tests rápidos).
        pca_dim: dims PCA antes de HDBSCAN (0 = sin PCA).
        assign_noise: si True, reasignar noise points al cluster vecino más cercano (1-NN).
        rep: variante de representations/ (p. ej. 'clap_full') para L1 y L2; None = MERT de Phase 1.
        bpm_weight: peso del BPM estandarizado agregado como dimensión extra (0 = sin BPM).
        method: 'hdbscan' (histórico) o 'ward' (número de grupos fijado, sin ruido).
        n_l1: grupos L1 con ward (default N/150). l2_target_size: temas por playlist L2 con ward.

    Returns:
        Path al parquet de resultados.
    """
    t0 = time.time()
    artifacts_dir = resolve_dataset_artifacts(dataset_name, config)
    embeddings_dir = artifacts_dir / "embeddings"
    clustering_dir = artifacts_dir / "clustering"
    clustering_dir.mkdir(parents=True, exist_ok=True)

    # --- Cargar embeddings ---
    if rep:
        # Una sola representación (extract_representations) para L1 y L2
        track_uids, mert_full = load_track_embeddings(artifacts_dir, rep=rep)
        mert_perc = mert_full
        print(f"[INFO] Representación: {rep}")
    else:
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
    if bpm_weight > 0:
        perc_norm = _append_bpm(perc_norm, track_uids, artifacts_dir, bpm_weight)
        full_norm = _append_bpm(full_norm, track_uids, artifacts_dir, bpm_weight)

    if method == "ward":
        n_l1 = n_l1 or max(2, int(round(N / 150)))
        labels_l1, labels_l2, l2_small_cluster_ids = ward_two_level(perc_norm, full_norm, n_l1, l2_target_size, pca_dim)
        labels_l1_raw, labels_l2_raw = labels_l1.copy(), labels_l2.copy()
        n_clusters_l1, n_noise_l1 = len(set(labels_l1)), 0
    else:
        (labels_l1, labels_l1_raw, labels_l2, labels_l2_raw, l2_small_cluster_ids,
         n_clusters_l1, n_noise_l1) = _hdbscan_two_level(
            perc_norm, full_norm, pca_dim, l1_min_cluster_size, l1_min_samples,
            l2_min_cluster_size, l2_min_samples, assign_noise)

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
        "pca_dim": pca_dim,
        "l1_min_cluster_size": l1_min_cluster_size,
        "l1_min_samples": l1_min_samples,
        "l2_min_cluster_size": l2_min_cluster_size,
        "l2_min_samples": l2_min_samples,
        "l2_min_parent_size": L2_MIN_PARENT_SIZE,
        "assign_noise": assign_noise,
        "config_tag": config_tag,
        "dataset_name": dataset_name,
    }
    if rep:
        cluster_cfg["rep"] = rep  # solo si se usa: no cambia los hashes históricos de MERT
    if bpm_weight > 0:
        cluster_cfg["bpm_weight"] = bpm_weight
    if method != "hdbscan":
        cluster_cfg.update({"method": method, "n_l1": n_l1, "l2_target_size": l2_target_size})
    config_hash = _hash_config(cluster_cfg)

    # --- Guardar parquet ---
    result_df = pd.DataFrame({
        "track_uid": track_uids,
        "label_l1": labels_l1.tolist(),
        "label_l1_raw": labels_l1_raw.tolist(),
        "label_l2": labels_l2.tolist(),
        "label_l2_raw": labels_l2_raw.tolist(),
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
    n_noise_l1_final = int(np.sum(labels_l1 == -1))
    n_reassigned_l1 = n_noise_l1 - n_noise_l1_final
    elapsed = round(time.time() - t0, 2)
    print("=" * 70)
    print(f"[INFO] Phase 2 clustering complete in {elapsed}s")
    print(f"[INFO] Config hash: {config_hash}")
    print(f"[INFO] L1 clusters: {n_clusters_l1}, noise (raw): {n_noise_l1/N:.1%}, "
          f"noise (final): {n_noise_l1_final/N:.1%}")
    if assign_noise and n_reassigned_l1 > 0:
        print(f"[INFO] L1 reassigned: {n_reassigned_l1} tracks moved from noise to nearest cluster")
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
    parser.add_argument(
        "--pca-dim", type=int, default=None,
        help="Dimensiones PCA antes de HDBSCAN (0 = sin PCA, default: del config).",
    )
    parser.add_argument(
        "--assign-noise", action=argparse.BooleanOptionalAction, default=None,
        help="Reasignar noise points al cluster vecino más cercano 1-NN (default: del config).",
    )
    parser.add_argument(
        "--method", choices=("hdbscan", "ward"), default="hdbscan",
        help="hdbscan (densidad, con ruido) o ward (número de grupos fijado).",
    )
    parser.add_argument("--n-l1", type=int, default=None, help="Grupos L1 con --method ward (default N/150).")
    parser.add_argument("--l2-target-size", type=int, default=30,
                        help="Temas por playlist L2 con --method ward.")
    parser.add_argument(
        "--bpm-weight", type=float, default=0.0,
        help="Peso del BPM estandarizado agregado al embedding (0 = sin BPM).",
    )
    parser.add_argument(
        "--rep", default=None,
        help="Variante de representations/ (p. ej. clap_full) en lugar de MERT de Phase 1.",
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
    pca_dim = args.pca_dim if args.pca_dim is not None else cluster_cfg.get("pca_dim", 0)
    assign_noise = (
        args.assign_noise if args.assign_noise is not None
        else cluster_cfg.get("assign_noise", True)
    )

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
            pca_dim=pca_dim,
            assign_noise=assign_noise,
            rep=args.rep,
            bpm_weight=args.bpm_weight,
            method=args.method,
            n_l1=args.n_l1,
            l2_target_size=args.l2_target_size,
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
