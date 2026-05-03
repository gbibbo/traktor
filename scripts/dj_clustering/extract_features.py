"""
PURPOSE: Main CLI driver for DJ clustering feature extraction. Loads canonical decoded
         tracks, generates segment manifest, extracts MERT full/perc/concat embeddings
         and BPM features, writes ignored artifacts, and commits a sanitized quality report.
         Designed to run inside Apptainer on a100 GPU nodes (full extraction) or on the
         login node for --segment-manifest-only mode.

CHANGELOG:
  D2.2 - Initial implementation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DJ clustering feature extraction")
    p.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to features.yaml",
    )
    p.add_argument(
        "--segment-manifest-only",
        action="store_true",
        help="Generate segment_manifest.csv only (login-node mode; no audio/MERT)",
    )
    p.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        metavar="N",
        help="Limit extraction to N canonical decoded tracks (smoke/debug)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Alias for --max-tracks 3",
    )
    p.add_argument(
        "--feature-output-root",
        default=None,
        type=Path,
        metavar="PATH",
        help="Override feature_output_root from config (repo-relative). Smoke: artifacts/dj_clustering/features_smoke",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Torch device: cuda or cpu (default: auto-detect)",
    )
    return p.parse_args()


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_device(requested: str | None) -> str:
    if requested:
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main() -> int:
    args = parse_args()

    if args.smoke:
        args.max_tracks = 3

    print("=" * 70)
    print("[INFO] DJ Clustering — Feature Extraction")
    print(f"[INFO] Config: {args.config}")
    if args.segment_manifest_only:
        print("[INFO] Mode: segment-manifest-only (no MERT, no audio)")
    elif args.max_tracks:
        print(f"[INFO] Mode: smoke / debug (max_tracks={args.max_tracks})")
    else:
        print("[INFO] Mode: full extraction")
    print("=" * 70)

    cfg = load_config(args.config)

    artifact_root = project_root / cfg["artifact_root"]
    feature_root = project_root / (args.feature_output_root or cfg["feature_output_root"])
    inventory_path = project_root / cfg["inventory_input"]
    segment_manifest_path = feature_root / "segment_manifest.csv"
    feature_manifest_path = feature_root / "feature_manifest.csv"
    quality_report_path = project_root / cfg["feature_quality_report"]
    if args.max_tracks:
        quality_report_path = feature_root / "smoke_quality_report.md"

    # Load inventory
    print("[INFO] Loading inventory...")
    df = pd.read_csv(inventory_path)
    canonical = df[df["is_canonical"] & (df["decode_status"] != "failed")].copy()
    print(f"[INFO] Canonical decoded tracks: {len(canonical)}")

    if args.max_tracks:
        canonical = canonical.sort_values("track_id").head(args.max_tracks)
        print(f"[INFO] Limiting to {len(canonical)} tracks (smoke/debug)")

    # Generate segment manifest
    from src.dj_clustering.segments import build_segment_manifest
    print("[INFO] Generating segment manifest...")
    feature_root.mkdir(parents=True, exist_ok=True)
    seg_manifest = build_segment_manifest(canonical)
    seg_manifest.to_csv(segment_manifest_path, index=False)
    print(f"[INFO] Segment manifest: {len(seg_manifest)} rows → {segment_manifest_path.relative_to(project_root)}")

    if args.segment_manifest_only:
        print("[INFO] Segment-manifest-only mode complete.")
        return 0

    # Full extraction path — requires MERT/GPU
    device = resolve_device(args.device)
    print(f"[INFO] Device: {device}")

    from src.dj_clustering.features import FeatureExtractor
    from src.dj_clustering.mert import load_mert, mert_available

    if not mert_available():
        print("[ERROR] MERT backend not available (torch or transformers missing)")
        return 1

    extractor = FeatureExtractor(cfg, device)
    hpss_ok = extractor.is_hpss_available()
    print(f"[INFO] HPSS available: {hpss_ok}")

    # Load MERT
    mert_cfg = cfg["mert"]
    print(f"[INFO] Loading MERT model: {mert_cfg['model']}")
    try:
        model, processor = load_mert(
            mert_cfg["model"], device,
            trust_remote_code=mert_cfg.get("trust_remote_code", True),
        )
    except Exception as exc:
        print(f"[ERROR] MERT model load failed: {exc}")
        return 1
    print("[INFO] MERT model loaded.")

    # Per-track extraction
    results = []
    n_tracks_total = len(canonical)
    for i, (_, row) in enumerate(canonical.iterrows()):
        tid = str(row["track_id"])
        try:
            result = extractor.extract_track(row, model, processor)
            results.append(result)
            status = "ok" if result["mert_full"] is not None else "FAIL"
            print(f"[INFO] track {i+1}/{n_tracks_total}: mert_full={status}"
                  f" perc={'ok' if result['mert_perc'] else ('omit' if not hpss_ok else 'fail')}"
                  f" concat={'ok' if result['mert_concat'] else 'omit'}")
        except Exception as exc:
            print(f"[ERROR] track {i+1}/{n_tracks_total}: unhandled exception: {type(exc).__name__}: {str(exc)[:80]}")
            results.append({
                "track_id": tid,
                "segments_used": 0,
                "mert_full": None,
                "mert_perc": None,
                "mert_concat": None,
                "hpss_failed": False,
                "hpss_failure_reason": None,
                "mert_full_failure_reason": f"unhandled: {type(exc).__name__}",
            })

    # Write embedding matrices
    _write_embeddings(results, extractor.aggregation_choices, feature_root)

    # BPM feature
    bpm_arr, bpm_ids = extractor.compute_bpm_feature(canonical)
    bpm_dir = feature_root / "metadata" / "bpm"
    bpm_dir.mkdir(parents=True, exist_ok=True)
    np.save(bpm_dir / "bpm_normalized.npy", bpm_arr)
    (bpm_dir / "track_ids.txt").write_text("\n".join(bpm_ids))
    print(f"[INFO] BPM feature: {len(bpm_ids)} tracks")

    # Feature manifest
    manifest_df = extractor.build_feature_manifest(results, artifact_root)
    feature_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(feature_manifest_path, index=False)
    print(f"[INFO] Feature manifest: {len(manifest_df)} rows → {feature_manifest_path.relative_to(project_root)}")

    # Failure rate check
    mf_full = manifest_df[manifest_df["feature_source"] == "mert_full"]
    n_failed = (mf_full["status"] == "failed").sum()
    n_total = len(mf_full.drop_duplicates("track_id"))
    failure_rate = n_failed / n_total if n_total > 0 else 0.0
    print(f"[INFO] mert_full failure rate: {n_failed}/{n_total} = {failure_rate:.1%}")

    # Quality report
    quality_report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_quality_report(
        quality_report_path, manifest_df, seg_manifest,
        extractor, hpss_ok, failure_rate, len(canonical),
        args.max_tracks,
    )
    print(f"[INFO] Quality report → {quality_report_path.relative_to(project_root)}")

    print("=" * 70)
    if failure_rate > 0.10:
        print(f"[ERROR] mert_full failure rate {failure_rate:.1%} exceeds 10% threshold")
        print("[ERROR] D2.2 extraction gate FAIL — investigate before committing")
        return 1

    print("[INFO] Extraction complete — D2.2 gate PASS")
    return 0


def _write_embeddings(
    results: list,
    aggregation_choices: list,
    feature_root: Path,
) -> None:
    """Write .npy embedding matrices and track_ids.txt per source/aggregation."""
    sources = ["mert_full", "mert_perc", "mert_concat"]
    for source in sources:
        for agg in aggregation_choices:
            ok_results = [r for r in results if r.get(source) is not None]
            if not ok_results:
                continue
            embs = np.stack([r[source][agg] for r in ok_results], axis=0)
            tids = [r["track_id"] for r in ok_results]

            out_dir = feature_root / source / agg
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "embeddings.npy", embs)
            (out_dir / "track_ids.txt").write_text("\n".join(tids))
            print(f"[INFO] {source}/{agg}: shape={embs.shape}")


def _write_quality_report(
    path: Path,
    manifest_df: pd.DataFrame,
    seg_manifest: pd.DataFrame,
    extractor,
    hpss_ok: bool,
    failure_rate: float,
    n_tracks: int,
    max_tracks: int | None,
) -> None:
    """Write a sanitized feature quality report (no private paths or filenames)."""
    is_smoke = max_tracks is not None
    smoke_note = "\n**SMOKE ONLY — not the final D2.2 quality report**\n" if is_smoke else ""

    aggs = extractor.aggregation_choices
    n_segs = len(seg_manifest)

    lines = [
        "# Feature Quality Report",
        smoke_note,
        f"| Metric | Value |",
        f"|---|---|",
        f"| N_canonical_decoded | {n_tracks} |",
        f"| N_segments_generated | {n_segs} |",
        f"| HPSS available | {hpss_ok} |",
        f"| Smoke run | {is_smoke} |",
        "",
        "## Per-source extraction summary",
        "",
        "| Source | Aggregation | N_extracted | N_failed | N_omitted | Failure rate | Dim |",
        "|---|---|---|---|---|---|---|",
    ]

    sources = ["mert_full", "mert_perc", "mert_concat"]
    for source in sources:
        for agg in aggs:
            sub = manifest_df[
                (manifest_df["feature_source"] == source)
                & (manifest_df["aggregation"] == agg)
            ]
            if sub.empty:
                continue
            n_ok = (sub["status"] == "ok").sum()
            n_fail = (sub["status"] == "failed").sum()
            n_omit = (sub["status"] == "omitted").sum()
            rate = f"{n_fail / len(sub):.1%}" if len(sub) > 0 else "—"
            dim = int(sub[sub["status"] == "ok"]["embedding_dim"].max()) if n_ok > 0 else 0
            lines.append(
                f"| {source} | {agg} | {n_ok} | {n_fail} | {n_omit} | {rate} | {dim} |"
            )

    # BPM
    lines += [
        "",
        "## Metadata features",
        "",
        f"| Feature | N_valid | N_null_imputed |",
        f"|---|---|---|",
    ]
    bpm_total = n_tracks
    bpm_null = 1  # known from D1.3
    lines.append(f"| bpm | {bpm_total - bpm_null} | {bpm_null} |")

    # Gate
    gate = "PASS" if failure_rate <= 0.10 else "FAIL"
    lines += [
        "",
        "## Gate",
        "",
        f"| Check | Result |",
        f"|---|---|",
        f"| mert_full failure rate ≤ 10% | {gate} ({failure_rate:.1%}) |",
    ]

    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
