"""
PURPOSE: Phase 4 — Ordenamiento intra-cluster para transiciones suaves entre tracks.
         Algoritmo: greedy nearest-neighbour con score mixto (embedding + BPM + Camelot key).
         Normaliza keys de Essentia ('C minor') a Camelot ('5A') antes de calcular compatibilidad.
         Guarda clustering/ordered_<hash>.parquet con columna 'position' por L2 subcluster.
CHANGELOG:
  - 2026-03-01: Creación inicial V4.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.config_loader import load_config
from src.v4.common.path_resolver import resolve_dataset_artifacts
from src.v4.config import ORDERING_WEIGHTS


# ---------------------------------------------------------------------------
# Tabla Camelot completa (clave interna para compatibilidad harmónica)
# ---------------------------------------------------------------------------

# Mapeo: (key_name, scale) → número Camelot (1-12) y modo ('A'=minor, 'B'=major)
# Essentia devuelve key como "C", "C#", "D", ... y scale como "minor" / "major"

_KEY_SEMITONE: Dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

# Camelot wheel: número por semitono (índice 0=C)
# Minor (A) y Major (B)
_MINOR_CAMELOT = [5, 12, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10]   # C=5A, C#=12A, D=7A, ...
_MAJOR_CAMELOT = [8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6, 1]   # C=8B, C#=3B, D=10B, ...


def essentia_to_camelot(key_str: str) -> str:
    """
    Convierte el campo 'key' de bpm_key.parquet al formato Camelot.

    Soporta:
      - Formato combinado Essentia: 'C minor', 'G major', 'F# minor'
      - Ya en Camelot: '5A', '9B'
      - Formato corto: 'Cm', 'Gmaj'

    Returns '?' si no se puede parsear.
    """
    if not key_str or str(key_str).strip() in ("?", "nan", ""):
        return "?"

    s = str(key_str).strip()

    # Ya en formato Camelot (e.g. '5A', '12B')
    if len(s) <= 3 and s[:-1].isdigit() and s[-1] in ("A", "B"):
        return s

    # Formato 'C minor' / 'G major' (Essentia combinado)
    parts = s.rsplit(" ", 1)
    if len(parts) == 2:
        key_name, scale_str = parts[0].strip(), parts[1].strip().lower()
        semitone = _KEY_SEMITONE.get(key_name, -1)
        if semitone >= 0:
            if scale_str == "minor":
                return f"{_MINOR_CAMELOT[semitone]}A"
            if scale_str == "major":
                return f"{_MAJOR_CAMELOT[semitone]}B"

    # Formato corto 'Cm', 'Gm', 'Ebm', 'G', 'Eb'
    key_name = s.rstrip("mM")
    is_minor = s.endswith("m") and not s.endswith("am") or s.lower().endswith("min")
    semitone = _KEY_SEMITONE.get(key_name, -1)
    if semitone >= 0:
        if is_minor:
            return f"{_MINOR_CAMELOT[semitone]}A"
        return f"{_MAJOR_CAMELOT[semitone]}B"

    return "?"


def key_compatibility(camelot_a: str, camelot_b: str) -> float:
    """
    Compatibilidad harmónica entre dos posiciones Camelot.

    Returns:
        1.0 — misma posición (mismo número, cualquier modo) → relativa mayor/menor
        0.5 — número adyacente (±1 en el anillo de 12)
        0.0 — todo lo demás
    """
    if camelot_a == "?" or camelot_b == "?" or not camelot_a or not camelot_b:
        return 0.5  # Unknown key → penalización neutra

    try:
        num_a = int(camelot_a[:-1])
        num_b = int(camelot_b[:-1])
    except (ValueError, IndexError):
        return 0.5

    if num_a == num_b:
        return 1.0  # Mismo número (e.g. 5A↔5B = relativa mayor/menor, o 5A↔5A)

    dist = min(abs(num_a - num_b), 12 - abs(num_a - num_b))  # Distancia circular mod 12
    if dist == 1:
        return 0.5  # Adyacente en el anillo
    return 0.0


# ---------------------------------------------------------------------------
# Algoritmo de ordering
# ---------------------------------------------------------------------------

def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def order_cluster_tracks(
    track_indices: List[int],
    embeddings: np.ndarray,
    bpm: np.ndarray,
    camelot_keys: List[str],
    weights: Optional[Dict[str, float]] = None,
) -> List[int]:
    """
    Ordena tracks de un cluster usando greedy nearest-neighbour.

    Args:
        track_indices: Índices globales en embeddings/bpm/camelot_keys.
        embeddings: (N_total, D) embeddings ya L2-normalizados.
        bpm: (N_total,) BPM por track.
        camelot_keys: List[str] de longitud N_total con posiciones Camelot.
        weights: {'embedding', 'bpm', 'key'} — pesos del score mixto.

    Returns:
        Lista de índices en track_indices ordenada óptimamente.
    """
    if weights is None:
        weights = dict(ORDERING_WEIGHTS)

    w_emb = weights.get("embedding", 0.5)
    w_bpm = weights.get("bpm", 0.3)
    w_key = weights.get("key", 0.2)

    if len(track_indices) <= 1:
        return list(track_indices)

    # Extraer sub-matrices del cluster
    sub_emb = embeddings[track_indices]  # (n, D) ya normalizados
    sub_bpm = bpm[track_indices]
    sub_keys = [camelot_keys[i] for i in track_indices]

    n = len(track_indices)

    # BPM range para normalizar
    bpm_range = float(np.ptp(sub_bpm)) if np.ptp(sub_bpm) > 0 else 1.0

    visited = [False] * n
    # Empezar por el track con BPM más bajo (anclaje musicalemnte coherente)
    start = int(np.argmin(sub_bpm))
    order = [start]
    visited[start] = True

    for _ in range(n - 1):
        current = order[-1]
        best_score = -1.0
        best_next = -1

        for j in range(n):
            if visited[j]:
                continue
            # Similitud de embedding (dot product de vectores L2-normalizados = cosine sim)
            emb_score = float(np.dot(sub_emb[current], sub_emb[j]))
            emb_score = (emb_score + 1.0) / 2.0  # Remap [-1,1] → [0,1]

            # BPM score (menor diferencia = mayor score)
            bpm_diff = abs(float(sub_bpm[current]) - float(sub_bpm[j]))
            bpm_score = max(0.0, 1.0 - bpm_diff / bpm_range)

            # Key compatibility score
            key_score = key_compatibility(sub_keys[current], sub_keys[j])

            score = w_emb * emb_score + w_bpm * bpm_score + w_key * key_score
            if score > best_score:
                best_score = score
                best_next = j

        order.append(best_next)
        visited[best_next] = True

    return [track_indices[i] for i in order]


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def _find_latest_results(clustering_dir: Path) -> Optional[Path]:
    candidates = sorted(clustering_dir.glob("results_*.parquet"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def run_ordering(
    dataset_name: str,
    config: dict,
    config_hash: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Path:
    """
    Aplica ordering greedy a cada L2 subcluster.

    Returns: Ruta al ordered_<hash>.parquet generado.
    """
    artifacts_dir = resolve_dataset_artifacts(dataset_name, config)
    clustering_dir = artifacts_dir / "clustering"
    embeddings_dir = artifacts_dir / "embeddings"
    features_dir = artifacts_dir / "features"

    # Resolver config_hash
    if config_hash is None:
        results_path = _find_latest_results(clustering_dir)
        if results_path is None:
            raise FileNotFoundError(f"No results_*.parquet found in {clustering_dir}")
        config_hash = results_path.stem.replace("results_", "")
    else:
        results_path = clustering_dir / f"results_{config_hash}.parquet"
        if not results_path.exists():
            raise FileNotFoundError(f"Clustering results not found: {results_path}")

    print(f"[INFO] Loading clustering results: {results_path.name}")
    df = pd.read_parquet(results_path)

    # Cargar embeddings + UIDs (alineación posicional)
    uids_path = embeddings_dir / "track_uids.json"
    with open(uids_path) as f:
        track_uids_ordered = json.load(f)
    uid_to_idx = {uid: i for i, uid in enumerate(track_uids_ordered)}

    print("[INFO] Loading mert_full embeddings...")
    mert_full = np.load(embeddings_dir / "mert_full.npy")  # (N, D)
    mert_full = _l2_normalize_rows(mert_full)

    # Cargar BPM + key
    bpm_path = features_dir / "bpm_key.parquet"
    bpm_df = pd.read_parquet(bpm_path)
    bpm_df = bpm_df.set_index("track_uid")

    # Construir arrays alineados con track_uids_ordered
    N = len(track_uids_ordered)
    bpm_arr = np.full(N, 128.0)  # default BPM si falta
    camelot_arr: List[str] = ["?"] * N

    for i, uid in enumerate(track_uids_ordered):
        if uid in bpm_df.index:
            row = bpm_df.loc[uid]
            bpm_val = row.get("bpm", 128.0)
            bpm_arr[i] = float(bpm_val) if not pd.isna(bpm_val) else 128.0
            key_str = row.get("key", "?")
            camelot_arr[i] = essentia_to_camelot(str(key_str)) if not pd.isna(key_str) else "?"

    # Ordenar por L1 y L2
    df["_global_idx"] = df["track_uid"].map(uid_to_idx)
    df["position"] = -1  # Default: sin posición (noise)

    order_stats = []
    labels_l1 = sorted([l for l in df["label_l1"].unique() if l != -1])

    if weights is None:
        ord_cfg = config.get("ordering", {}).get("weights", {})
        weights = {
            "embedding": float(ord_cfg.get("embedding", ORDERING_WEIGHTS["embedding"])),
            "bpm": float(ord_cfg.get("bpm", ORDERING_WEIGHTS["bpm"])),
            "key": float(ord_cfg.get("key", ORDERING_WEIGHTS["key"])),
        }

    for l1 in labels_l1:
        mask_l1 = df["label_l1"] == l1
        l2_labels = sorted([l for l in df.loc[mask_l1, "label_l2"].unique() if l >= 0])

        # Tracks L2-noise dentro del L1 cluster: ordenar también como subgrupo
        if -1 in df.loc[mask_l1, "label_l2"].values:
            l2_labels_all = [-1] + l2_labels
        else:
            l2_labels_all = l2_labels

        for l2 in l2_labels_all:
            if l2 == -1:
                mask_l2 = mask_l1 & (df["label_l2"] == -1)
            else:
                mask_l2 = mask_l1 & (df["label_l2"] == l2)

            indices_in_df = df.index[mask_l2].tolist()
            global_idxs = df.loc[mask_l2, "_global_idx"].dropna().astype(int).tolist()

            if not global_idxs:
                continue

            ordered_idxs = order_cluster_tracks(
                global_idxs, mert_full, bpm_arr, camelot_arr, weights
            )
            # Asignar posición 0-based dentro del subcluster
            global_to_pos = {g: pos for pos, g in enumerate(ordered_idxs)}
            for df_idx, g_idx in zip(indices_in_df, global_idxs):
                df.at[df_idx, "position"] = global_to_pos.get(g_idx, -1)

            order_stats.append({
                "l1": l1, "l2": l2,
                "n_tracks": len(global_idxs),
                "bpm_min": float(np.min(bpm_arr[global_idxs])),
                "bpm_max": float(np.max(bpm_arr[global_idxs])),
            })

    # Guardar resultado
    df = df.drop(columns=["_global_idx"], errors="ignore")
    ordered_path = clustering_dir / f"ordered_{config_hash}.parquet"
    df.to_parquet(ordered_path, index=False)

    print("\n" + "=" * 70)
    print(f"[INFO] Ordering complete. Summary:")
    for stat in order_stats:
        print(f"  L1={stat['l1']} L2={stat['l2']}: {stat['n_tracks']} tracks, "
              f"BPM [{stat['bpm_min']:.0f}–{stat['bpm_max']:.0f}]")
    print(f"[INFO] Saved: {ordered_path}")
    print("=" * 70)

    return ordered_path


def main() -> int:
    parser = argparse.ArgumentParser(description="TRAKTOR ML V4 — Phase 4: Track Ordering")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--config-hash", default=None)
    parser.add_argument("--weights-embedding", type=float, default=None)
    parser.add_argument("--weights-bpm", type=float, default=None)
    parser.add_argument("--weights-key", type=float, default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("[INFO] TRAKTOR ML V4 — Phase 4: Track Ordering")
    print(f"[INFO] Dataset: {args.dataset_name}")
    print("=" * 70)

    config = load_config(Path(args.config) if args.config else None)

    weights = None
    if any(v is not None for v in [args.weights_embedding, args.weights_bpm, args.weights_key]):
        weights = {
            "embedding": args.weights_embedding if args.weights_embedding is not None else ORDERING_WEIGHTS["embedding"],
            "bpm": args.weights_bpm if args.weights_bpm is not None else ORDERING_WEIGHTS["bpm"],
            "key": args.weights_key if args.weights_key is not None else ORDERING_WEIGHTS["key"],
        }

    run_ordering(
        dataset_name=args.dataset_name,
        config=config,
        config_hash=args.config_hash,
        weights=weights,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
