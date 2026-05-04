"""
PURPOSE: Driver for D3.1. Reads frozen D2.2 feature artifacts and the canonical
         inventory, enumerates all unordered pairs over canonical decoded
         tracks, computes pairwise cosine features for every (MERT source,
         aggregation), raw-BPM tempo-equivalence features (8.0 BPM tolerance),
         metadata similarity over D1.3-validated sanity fields, and emits the
         committed similarity profiles. Writes the parquet table, a JSON
         manifest, and a sanitized aggregate-only summary report. CPU-only;
         no MERT, no audio, no GPU.

CHANGELOG:
  D3.1 - Initial implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import yaml

from src.dj_clustering.pair_features import (
    MERT_AGGREGATIONS,
    MERT_SOURCES,
    align_embedding_matrix,
    assemble_pair_dataframe,
    compute_bpm_pair_features,
    compute_cosine_pair_features,
    compute_metadata_pair_features,
    cosine_default_available_column,
    cosine_default_column,
    enumerate_canonical_pairs,
    load_track_ids,
)
from src.dj_clustering.similarity_profiles import (
    ALL_PROFILES,
    attach_committed_profiles,
    build_profile_registry,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DJ clustering D3.1: build pair features")
    p.add_argument(
        "--features-config",
        type=Path,
        default=Path("configs/dj_clustering/features.yaml"),
        help="Path to features.yaml (must contain a 'pairs:' block).",
    )
    p.add_argument(
        "--inventory-config",
        type=Path,
        default=Path("configs/dj_clustering/inventory.yaml"),
        help="Path to inventory.yaml (used only to resolve inventory_output).",
    )
    p.add_argument(
        "--inventory-csv",
        type=Path,
        default=None,
        help="Override inventory CSV path (else inferred from inventory.yaml).",
    )
    p.add_argument(
        "--feature-root",
        type=Path,
        default=None,
        help="Override D2.2 feature root (else from features.yaml).",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override pair output root (else from features.yaml pairs.output_root).",
    )
    return p.parse_args()


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def banner(msg: str) -> None:
    print("=" * 70)
    print(msg)
    print("=" * 70)


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def err(msg: str) -> None:
    print(f"[ERROR] {msg}")


def resolve_inventory_csv(
    features_cfg: dict, inventory_cfg: dict, override: Path | None
) -> Path:
    if override is not None:
        return override
    artifact_root = Path(inventory_cfg.get("artifact_root", "artifacts/dj_clustering"))
    inv_rel = inventory_cfg.get("inventory_output", "inventory/library_inventory.csv")
    return artifact_root / inv_rel


def select_canonical_decoded_track_ids(inv: pd.DataFrame) -> List[str]:
    if "is_canonical" not in inv.columns or "decode_status" not in inv.columns:
        raise ValueError("inventory missing is_canonical or decode_status columns")
    mask = (inv["is_canonical"].astype(bool)) & (inv["decode_status"] == "ok")
    ids = inv.loc[mask, "track_id"].astype(str).tolist()
    return sorted(set(ids))


def build_bpm_lookup(inv: pd.DataFrame, canonical_ids: List[str]) -> Dict[str, float]:
    sub = inv.loc[inv["track_id"].astype(str).isin(canonical_ids), ["track_id", "bpm"]]
    out: Dict[str, float] = {}
    for tid, bpm in zip(sub["track_id"].astype(str), sub["bpm"]):
        if pd.isna(bpm):
            continue
        out[tid] = float(bpm)
    return out


def build_metadata_table(
    inv: pd.DataFrame, canonical_ids: List[str], fields: List[str]
) -> pd.DataFrame:
    keep = ["track_id"] + [f for f in fields if f in inv.columns]
    sub = inv.loc[inv["track_id"].astype(str).isin(canonical_ids), keep].copy()
    sub["track_id"] = sub["track_id"].astype(str)
    return sub


def load_normalized_bpm(
    feature_root: Path, canonical_ids: List[str]
) -> Tuple[Dict[str, float], bool]:
    """Load D2.2 bpm_normalized.npy as a track_id -> normalized value lookup.

    Returned lookup is used ONLY to compute the optional bpm_norm_abs_diff
    column (clearly labeled). It is never substituted for raw BPM in
    tempo-equivalence calculations.
    """
    npy = feature_root / "metadata" / "bpm" / "bpm_normalized.npy"
    ids_path = feature_root / "metadata" / "bpm" / "track_ids.txt"
    if not npy.exists() or not ids_path.exists():
        return {}, False
    arr = np.load(npy)
    ids = load_track_ids(ids_path)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.shape[0] != len(ids):
        info(
            f"normalized BPM artifact rows {arr.shape[0]} != track_ids {len(ids)}; "
            f"skipping bpm_norm_abs_diff."
        )
        return {}, False
    lookup = {tid: float(v) for tid, v in zip(ids, arr.tolist())}
    return lookup, True


def compute_bpm_norm_abs_diff(
    norm_lookup: Mapping[str, float],
    canonical_ids: List[str],
    idx_a: np.ndarray,
    idx_b: np.ndarray,
) -> np.ndarray:
    arr = np.full(len(canonical_ids), np.nan, dtype=np.float64)
    for i, tid in enumerate(canonical_ids):
        v = norm_lookup.get(tid)
        if v is not None:
            arr[i] = v
    a = arr[idx_a]
    b = arr[idx_b]
    return np.abs(a - b)


def main() -> int:
    args = parse_args()

    banner("DJ Clustering D3.1 — Build pairwise feature table")

    features_cfg = load_yaml(args.features_config)
    inventory_cfg = load_yaml(args.inventory_config)

    pairs_cfg = features_cfg.get("pairs")
    if pairs_cfg is None:
        err("features.yaml missing 'pairs:' block")
        return 1

    bpm_tol = float(pairs_cfg["bpm_tolerance_bpm"])
    bpm_col = pairs_cfg.get("bpm_source_column", "bpm")
    bpm_units = pairs_cfg.get("bpm_source_units", "raw_bpm")
    if bpm_units != "raw_bpm":
        err(f"pairs.bpm_source_units must be 'raw_bpm', got '{bpm_units}'")
        return 1
    default_source = pairs_cfg["default_embedding_source"]
    default_agg = pairs_cfg["default_embedding_aggregation"]
    metadata_fields = list(pairs_cfg["metadata_sanity_fields"])

    feature_root = args.feature_root or Path(
        features_cfg.get("feature_output_root", "artifacts/dj_clustering/features")
    )
    output_root = args.output_root or Path(pairs_cfg["output_root"])
    parquet_filename = pairs_cfg["parquet_filename"]
    manifest_filename = pairs_cfg["manifest_filename"]

    inventory_csv = resolve_inventory_csv(features_cfg, inventory_cfg, args.inventory_csv)
    info(f"inventory CSV: {inventory_csv}")
    inv = pd.read_csv(inventory_csv)
    info(f"inventory rows: {len(inv)}")
    canonical_ids = select_canonical_decoded_track_ids(inv)
    n = len(canonical_ids)
    info(f"canonical decoded tracks: {n}")
    expected_pairs = n * (n - 1) // 2
    info(f"expected unordered pairs: {expected_pairs}")

    pair_ids, idx_a, idx_b = enumerate_canonical_pairs(canonical_ids)
    if len(pair_ids) != expected_pairs:
        err(f"pair enumeration count mismatch: {len(pair_ids)} vs {expected_pairs}")
        return 1
    track_ids_a = [canonical_ids[i] for i in idx_a.tolist()]
    track_ids_b = [canonical_ids[i] for i in idx_b.tolist()]

    # ---------- Cosine features for all (source, aggregation) combos ----------
    cosine_blocks: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    cosine_availability_counts: Dict[str, int] = {}
    for source in MERT_SOURCES:
        for agg in MERT_AGGREGATIONS:
            emb_path = feature_root / source / agg / "embeddings.npy"
            ids_path = feature_root / source / agg / "track_ids.txt"
            if not emb_path.exists() or not ids_path.exists():
                info(f"skipping {source}/{agg}: artifact missing")
                continue
            embeddings = np.load(emb_path)
            src_ids = load_track_ids(ids_path)
            if embeddings.shape[0] != len(src_ids):
                err(
                    f"{source}/{agg}: rows {embeddings.shape[0]} != "
                    f"track_ids {len(src_ids)}"
                )
                return 1
            aligned, avail = align_embedding_matrix(
                embeddings, src_ids, canonical_ids
            )
            block = compute_cosine_pair_features(aligned, avail, idx_a, idx_b)
            cosine_blocks[(source, agg)] = block
            n_avail = int(block["available"].sum())
            cosine_availability_counts[f"{source}_{agg}"] = n_avail
            info(f"{source}/{agg}: pairs available={n_avail}/{expected_pairs}")

    # ---------- Raw BPM features ----------
    bpm_lookup = build_bpm_lookup(inv, canonical_ids)
    info(
        f"raw BPM lookup size: {len(bpm_lookup)} (column '{bpm_col}' from inventory)"
    )
    bpm_block = compute_bpm_pair_features(
        bpm_lookup, canonical_ids, idx_a, idx_b, bpm_tol
    )
    bpm_pair_count = int(bpm_block["bpm_available"].sum())
    info(f"raw BPM pairs available: {bpm_pair_count}/{expected_pairs}")

    # Optional bpm_norm_abs_diff (alignment cross-check, not used in similarity).
    norm_lookup, norm_loaded = load_normalized_bpm(feature_root, canonical_ids)
    bpm_norm_abs_diff = None
    if norm_loaded:
        bpm_norm_abs_diff = compute_bpm_norm_abs_diff(
            norm_lookup, canonical_ids, idx_a, idx_b
        )

    # ---------- Metadata similarity ----------
    meta_table = build_metadata_table(inv, canonical_ids, metadata_fields)
    metadata_block = compute_metadata_pair_features(
        meta_table, metadata_fields, canonical_ids, idx_a, idx_b
    )
    meta_avail_count = int(metadata_block["metadata_similarity_available"].sum())
    info(f"metadata pairs with at least one shared field: {meta_avail_count}")
    field_avail_counts = {
        f: int(metadata_block[f"metadata_{f}_available"].sum())
        for f in metadata_fields
    }
    for f, c in field_avail_counts.items():
        info(f"metadata.{f} pairs available: {c}")

    # ---------- Assemble dataframe ----------
    df = assemble_pair_dataframe(
        pair_ids=pair_ids,
        track_ids_a=track_ids_a,
        track_ids_b=track_ids_b,
        cosine_blocks=cosine_blocks,
        bpm_block=bpm_block,
        metadata_block=metadata_block,
        bpm_norm_abs_diff=bpm_norm_abs_diff,
    )

    # ---------- Profiles ----------
    default_cosine_col = cosine_default_column(default_source, default_agg)
    default_avail_col = cosine_default_available_column(default_source, default_agg)
    default_present = (default_source, default_agg) in cosine_blocks
    if not default_present:
        err(
            f"default embedding {default_source}/{default_agg} unavailable; "
            "per plan decision rule, return to D2."
        )
        return 1

    # Component availability decisions (dataset-level):
    #   key has 0% coverage -> key_compat unavailable.
    available_components = {
        "cosine_embedding_01": True,
        "bpm_similarity": True,
        "key_compat": False,  # dataset-level: 0% key coverage in inventory.
        "metadata_similarity": True,
    }
    profile_registry = build_profile_registry(available_components)
    component_columns = {
        "cosine_embedding_01": default_cosine_col,
        "bpm_similarity": "bpm_similarity",
        "metadata_similarity": "metadata_similarity",
    }
    component_available_columns = {
        "cosine_embedding_01": default_avail_col,
        "bpm_similarity": "bpm_available",
        "metadata_similarity": "metadata_similarity_available",
    }
    df = attach_committed_profiles(
        df, profile_registry, component_columns, component_available_columns
    )

    # ---------- Write artifacts ----------
    output_root.mkdir(parents=True, exist_ok=True)
    parquet_path = output_root / parquet_filename
    manifest_path = output_root / manifest_filename
    info(f"writing parquet: {parquet_path}")
    df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=False)

    profile_summary = []
    for entry in profile_registry:
        rec = {"name": entry.name, "status": entry.status}
        if entry.status == "committed":
            rec["similarity_column"] = entry.similarity_column
            rec["distance_column"] = entry.distance_column
            rec["available_column"] = entry.available_column
        else:
            rec["omission_reason"] = entry.omission_reason
        profile_summary.append(rec)

    manifest = {
        "schema_version": "1.0",
        "task": "D3.1",
        "generated_at_unix": int(time.time()),
        "n_canonical_decoded": n,
        "expected_pairs": expected_pairs,
        "actual_pairs": int(len(df)),
        "pair_id_separator": "__",
        "default_embedding_source": default_source,
        "default_embedding_aggregation": default_agg,
        "bpm_tolerance_bpm": bpm_tol,
        "bpm_source_column": bpm_col,
        "bpm_source_units": bpm_units,
        "metadata_sanity_fields": metadata_fields,
        "cosine_pair_availability": cosine_availability_counts,
        "bpm_pair_availability": bpm_pair_count,
        "metadata_pair_availability": meta_avail_count,
        "metadata_field_pair_availability": field_avail_counts,
        "profiles": profile_summary,
        "bpm_norm_abs_diff_attached": bool(norm_loaded),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    info(f"writing manifest: {manifest_path}")

    # ---------- Sanity checks ----------
    if len(df) != expected_pairs:
        err(f"row count mismatch: {len(df)} vs {expected_pairs}")
        return 1
    if not df["pair_id"].is_unique:
        err("pair_id not unique")
        return 1
    if not (df["track_id_a"] < df["track_id_b"]).all():
        err("track_id_a < track_id_b violated")
        return 1

    sim_cols = []
    for c in df.columns:
        if c.endswith("__cosine_embedding_01"):
            sim_cols.append(c)
        elif c.startswith("profile_") and c.endswith("_similarity"):
            sim_cols.append(c)
        elif c in ("bpm_similarity", "metadata_similarity"):
            sim_cols.append(c)
    for c in sim_cols:
        s = df[c].dropna()
        if len(s) and not s.between(0.0, 1.0).all():
            err(f"column {c} outside [0, 1]")
            return 1

    banner("D3.1 build complete")
    info(f"rows: {len(df)}")
    info(f"columns: {len(df.columns)}")
    info(f"parquet: {parquet_path}")
    info(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
