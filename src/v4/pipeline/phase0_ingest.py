"""
PURPOSE: Phase 0 — Ingesta y catálogo del dataset.
         Escanea audio_root, valida archivos, computa track_uids, merge metadata,
         genera catalog.parquet e ingest_report.json. Idempotente.
CHANGELOG:
  - 2026-02-28: Creación inicial V4.
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.audio_utils import get_audio_files, validate_audio_file
from src.v4.common.catalog import build_catalog, compute_track_uid
from src.v4.common.config_loader import load_config
from src.v4.common.logging_utils import log_event, open_phase_log
from src.v4.common.path_resolver import (
    resolve_dataset_artifacts,
    resolve_dataset_audio_root,
    resolve_dataset_metadata,
    resolve_logs_root,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="TRAKTOR ML V4 — Phase 0: Ingest & Catalog")
    parser.add_argument("--dataset-name", required=True, help="Dataset identifier")
    parser.add_argument("--audio-root", default=None, help="Override audio directory path")
    parser.add_argument("--manifest-csv", default=None, help="Optional manifest CSV path")
    parser.add_argument("--metadata-csv", default=None, help="External metadata CSV (Beatport, etc.)")
    parser.add_argument("--config", default=None, help="Path to v4.yaml config file")
    args = parser.parse_args()

    print("=" * 70)
    print(f"[INFO] TRAKTOR ML V4 — Phase 0: Ingest & Catalog")
    print(f"[INFO] Dataset: {args.dataset_name}")
    print("=" * 70)

    t0 = time.time()
    config = load_config(Path(args.config) if args.config else None)

    # Resolver rutas
    audio_root = resolve_dataset_audio_root(args.dataset_name, config, cli_override=args.audio_root)
    artifacts_dir = resolve_dataset_artifacts(args.dataset_name, config)
    logs_root = resolve_logs_root(args.dataset_name, config)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Audio root    : {audio_root}")
    print(f"[INFO] Artifacts dir : {artifacts_dir}")

    log_fh = open_phase_log(logs_root, "phase0_ingest")
    log_event(log_fh, {
        "phase": "phase0",
        "dataset_name": args.dataset_name,
        "event_type": "phase_start",
        "audio_root": str(audio_root),
    })

    # Escanear archivos
    all_files = get_audio_files(audio_root)
    print(f"[INFO] Found {len(all_files)} audio files")

    # Verificar expected_n
    expected_n = config.get("datasets", {}).get(args.dataset_name, {}).get("expected_n")
    if expected_n is not None and len(all_files) != expected_n:
        print(f"[WARN] Expected {expected_n} files, found {len(all_files)}")

    # Validar archivos
    n_found = len(all_files)
    n_failed = 0
    valid_files = []
    for f in all_files:
        if validate_audio_file(f):
            valid_files.append(f)
        else:
            n_failed += 1
            log_event(log_fh, {
                "phase": "phase0",
                "dataset_name": args.dataset_name,
                "event_type": "file_invalid",
                "filepath": str(f),
            })
            print(f"[WARN] Invalid file: {f.name}")

    print(f"[INFO] Valid: {len(valid_files)}/{n_found} files")

    # Cargar metadata externa si disponible
    metadata_df = None
    meta_path = Path(args.metadata_csv) if args.metadata_csv else resolve_dataset_metadata(args.dataset_name, config)
    if meta_path and meta_path.exists():
        import pandas as pd
        metadata_df = pd.read_csv(meta_path)
        print(f"[INFO] Loaded metadata CSV: {meta_path} ({len(metadata_df)} rows)")

    # Construir catálogo
    catalog = build_catalog(audio_root, args.dataset_name, config, metadata_df=metadata_df)

    # Ingest report
    elapsed = time.time() - t0
    report = {
        "dataset_name": args.dataset_name,
        "audio_root": str(audio_root),
        "n_files_found": n_found,
        "n_files_valid": len(valid_files),
        "n_files_failed": n_failed,
        "n_catalog_entries": len(catalog),
        "elapsed_s": round(elapsed, 2),
        "artifacts_dir": str(artifacts_dir),
    }
    report_path = artifacts_dir / "ingest_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    log_event(log_fh, {
        "phase": "phase0",
        "dataset_name": args.dataset_name,
        "event_type": "phase_end",
        "n_catalog_entries": len(catalog),
        "elapsed_s": elapsed,
    })
    log_fh.close()

    print("=" * 70)
    print(f"[INFO] Phase 0 complete in {elapsed:.1f}s")
    print(f"[INFO] Catalog: {len(catalog)} tracks → {artifacts_dir}/catalog.parquet")
    print(f"[INFO] Report  : {report_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
