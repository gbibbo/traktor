"""
PURPOSE: Phase 1 Merge — Consolidar shards de embeddings en archivos finales.
         Verifica cobertura completa del catálogo, genera run_manifest.json.
         Ignorar graciosamente tracks fallidos (no presentes en los shards).
CHANGELOG:
  - 2026-02-28: Creación inicial V4.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.catalog import load_catalog, update_catalog_columns
from src.v4.common.config_loader import load_config
from src.v4.common.logging_utils import compute_config_hash, get_git_commit, get_slurm_job_id
from src.v4.common.path_resolver import resolve_dataset_artifacts


def get_version(pkg: str) -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version(pkg)
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="TRAKTOR ML V4 — Phase 1 Merge Shards")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    print("=" * 70)
    print(f"[INFO] TRAKTOR ML V4 — Phase 1: Merge Shards")
    print(f"[INFO] Dataset: {args.dataset_name}")
    print("=" * 70)

    t0 = time.time()
    config = load_config(Path(args.config) if args.config else None)
    artifacts_dir = resolve_dataset_artifacts(args.dataset_name, config)
    embeddings_dir = artifacts_dir / "embeddings"
    shards_dir = embeddings_dir / "shards"
    features_dir = artifacts_dir / "features"
    shards_features_dir = features_dir / "shards"

    # Encontrar todos los shards disponibles
    uid_files = sorted(shards_dir.glob("track_uids_shard_*.json"))
    if not uid_files:
        print(f"[ERROR] No shard files found in {shards_dir}")
        return 1

    print(f"[INFO] Found {len(uid_files)} shard(s)")

    # Consolidar
    all_uids = []
    all_perc = []
    all_full = []
    all_bpm_key = []
    shards_used = []

    for uid_file in uid_files:
        shard_tag = uid_file.stem.replace("track_uids_", "")  # e.g. shard_00
        perc_file = shards_dir / f"mert_perc_{shard_tag}.npy"
        full_file = shards_dir / f"mert_full_{shard_tag}.npy"
        bpm_file = shards_features_dir / f"bpm_key_{shard_tag}.parquet"

        if not perc_file.exists() or not full_file.exists():
            print(f"[WARN] Incomplete shard {shard_tag}: missing embedding files, skipping")
            continue

        with open(uid_file) as f:
            shard_uids = json.load(f)

        shard_perc = np.load(perc_file)
        shard_full = np.load(full_file)

        if len(shard_uids) != len(shard_perc):
            print(f"[WARN] Shard {shard_tag}: uid count ({len(shard_uids)}) != embedding count ({len(shard_perc)}), skipping")
            continue

        all_uids.extend(shard_uids)
        all_perc.append(shard_perc)
        all_full.append(shard_full)
        shards_used.append(shard_tag)

        if bpm_file.exists():
            all_bpm_key.append(pd.read_parquet(bpm_file))

        print(f"  [OK] {shard_tag}: {len(shard_uids)} tracks")

    if not all_uids:
        print("[ERROR] No valid shards to merge")
        return 1

    # Verificar cobertura vs catálogo
    catalog = load_catalog(args.dataset_name, config)
    catalog_uids = set(catalog["track_uid"].tolist())
    merged_uids = set(all_uids)
    missing = catalog_uids - merged_uids
    extra = merged_uids - catalog_uids

    if missing:
        print(f"[WARN] {len(missing)} catalog tracks NOT in shards (failed during extraction): "
              f"{list(missing)[:5]}{'...' if len(missing) > 5 else ''}")
    if extra:
        print(f"[WARN] {len(extra)} shard tracks NOT in catalog (stale data?)")

    # Guardar embeddings consolidados
    mert_perc = np.vstack(all_perc)   # (N, 1024)
    mert_full = np.vstack(all_full)   # (N, 1024)

    embeddings_dir.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_dir / "mert_perc.npy", mert_perc)
    np.save(embeddings_dir / "mert_full.npy", mert_full)

    # Guardar UIDs de orden (para alinear con los .npy)
    with open(embeddings_dir / "track_uids.json", "w") as f:
        json.dump(all_uids, f)

    print(f"[INFO] Saved mert_perc.npy {mert_perc.shape}, mert_full.npy {mert_full.shape}")

    # Guardar features BPM/key
    if all_bpm_key:
        features_dir.mkdir(parents=True, exist_ok=True)
        bpm_key_df = pd.concat(all_bpm_key, ignore_index=True)
        bpm_key_path = features_dir / "bpm_key.parquet"
        bpm_key_df.to_parquet(bpm_key_path, index=False)
        print(f"[INFO] Saved bpm_key.parquet ({len(bpm_key_df)} rows)")

        # Actualizar catálogo con BPM/key
        update_cols = [c for c in bpm_key_df.columns if c != "track_uid"]
        update_catalog_columns(args.dataset_name, config, bpm_key_df[["track_uid"] + update_cols])
        print(f"[INFO] Updated catalog with BPM/key columns")

    # Generar run_manifest.json
    elapsed = time.time() - t0
    import torch
    manifest = {
        "dataset_name": args.dataset_name,
        "n_tracks_merged": len(all_uids),
        "n_tracks_catalog": len(catalog_uids),
        "n_tracks_failed": len(missing),
        "shards_used": shards_used,
        "config_hash": compute_config_hash(config),
        "git_commit": get_git_commit(REPO_ROOT),
        "slurm_job_id": get_slurm_job_id(),
        "hostname": platform.node(),
        "user": __import__("os").environ.get("USER", "unknown"),
        "elapsed_s": round(elapsed, 2),
        "model_versions": {
            "MERT": "m-a-p/MERT-v1-330M",
            "Demucs": "htdemucs",
        },
        "lib_versions": {
            "python": platform.python_version(),
            "torch": get_version("torch"),
            "transformers": get_version("transformers"),
            "numpy": get_version("numpy"),
        },
        "artifacts": {
            "mert_perc_shape": list(mert_perc.shape),
            "mert_full_shape": list(mert_full.shape),
            "mert_perc_finite": bool(np.isfinite(mert_perc).all()),
            "mert_full_finite": bool(np.isfinite(mert_full).all()),
        },
    }
    manifest_path = artifacts_dir / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 70)
    print(f"[INFO] Merge complete in {elapsed:.1f}s")
    print(f"[INFO] Merged: {len(all_uids)} tracks | Missing: {len(missing)}")
    print(f"[INFO] run_manifest.json saved to {manifest_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
