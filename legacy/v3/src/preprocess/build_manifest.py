#!/usr/bin/env python3
"""
PURPOSE: Build manifest.parquet for TRAKTOR ML V3 preprocessing pipeline.
         Scans audio directory, validates files, extracts metadata via soundfile,
         and generates the Source of Truth manifest for downstream embedding extraction.

CHANGELOG:
    2026-02-05: Initial implementation for V3 MERT pipeline.
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import soundfile as sf
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".aiff", ".m4a")
MIN_DURATION_DEFAULT = 30.0  # seconds


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
def generate_track_id(relative_path: Path) -> str:
    """Return a deterministic SHA1 hex digest from the POSIX relative path."""
    path_str = relative_path.as_posix()
    return hashlib.sha1(path_str.encode("utf-8")).hexdigest()


def scan_audio_files(
    audio_dir: Path,
    recursive: bool = False,
    extensions: Tuple[str, ...] = AUDIO_EXTENSIONS,
) -> List[Path]:
    """Scan *audio_dir* for files matching *extensions* (case-insensitive).

    Returns a sorted list of absolute paths.
    """
    glob_fn = audio_dir.rglob if recursive else audio_dir.glob
    files: List[Path] = []
    for ext in extensions:
        files.extend(glob_fn(f"*{ext}"))
        files.extend(glob_fn(f"*{ext.upper()}"))
    return sorted(set(files))


def extract_audio_metadata(audio_path: Path) -> Optional[Dict[str, Any]]:
    """Read lightweight metadata with *soundfile.info()*.

    Returns ``None`` if the file is corrupt or unreadable.
    """
    try:
        info = sf.info(str(audio_path))
        return {
            "duration": info.frames / info.samplerate,
            "sample_rate": info.samplerate,
        }
    except Exception:
        return None


def build_manifest(
    audio_dir: Path,
    audio_files: List[Path],
    min_duration: float = MIN_DURATION_DEFAULT,
) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    """Build the manifest DataFrame and collect validation failures.

    Returns
    -------
    manifest_df : pd.DataFrame
        Columns: track_id, file_path, relative_path, duration, sample_rate
    failures : list[dict]
        Each entry has keys ``file`` and ``reason``.
    """
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

    for audio_path in tqdm(audio_files, desc="Building manifest", unit="file"):
        rel = audio_path.relative_to(audio_dir)

        # Metadata extraction
        meta = extract_audio_metadata(audio_path)
        if meta is None:
            failures.append({"file": str(rel), "reason": "Cannot read audio file"})
            continue

        # Duration check
        if meta["duration"] < min_duration:
            failures.append({
                "file": str(rel),
                "reason": f"Duration {meta['duration']:.1f}s < minimum {min_duration}s",
            })
            continue

        rows.append({
            "track_id": generate_track_id(rel),
            "file_path": str(audio_path.resolve()),
            "relative_path": rel.as_posix(),
            "duration": meta["duration"],
            "sample_rate": meta["sample_rate"],
        })

    df = pd.DataFrame(rows)
    return df, failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build manifest.parquet for TRAKTOR ML V3 preprocessing pipeline",
    )
    parser.add_argument(
        "--audio-dir", type=Path, required=True,
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--dataset-name", type=str, required=True,
        help="Dataset name (output: artifacts/dataset/<name>/manifest.parquet)",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Scan subdirectories recursively",
    )
    parser.add_argument(
        "--min-duration", type=float, default=MIN_DURATION_DEFAULT,
        help=f"Minimum track duration in seconds (default: {MIN_DURATION_DEFAULT})",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # --- Validate inputs ---------------------------------------------------
    if not args.audio_dir.exists() or not args.audio_dir.is_dir():
        print(f"[ERROR] Not a valid directory: {args.audio_dir}")
        return 1

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "artifacts" / "dataset" / args.dataset_name
    output_path = output_dir / "manifest.parquet"

    # --- Header ------------------------------------------------------------
    print("=" * 70)
    print("TRAKTOR ML V3 — Build Manifest")
    print("=" * 70)
    print(f"  Audio directory : {args.audio_dir}")
    print(f"  Dataset name    : {args.dataset_name}")
    print(f"  Output          : {output_path}")
    print(f"  Min duration    : {args.min_duration}s")
    print(f"  Recursive       : {args.recursive}")
    print()

    # --- Scan --------------------------------------------------------------
    print("[INFO] Scanning for audio files...")
    audio_files = scan_audio_files(args.audio_dir, recursive=args.recursive)
    print(f"[INFO] Found {len(audio_files)} audio files")

    if not audio_files:
        print("[ERROR] No audio files found!")
        return 1

    # --- Build manifest ----------------------------------------------------
    t0 = time.time()
    manifest_df, failures = build_manifest(
        args.audio_dir, audio_files, args.min_duration,
    )
    elapsed = time.time() - t0

    if manifest_df.empty:
        print("[ERROR] All files failed validation. Cannot create manifest.")
        for f in failures[:10]:
            print(f"  - {f['file']}: {f['reason']}")
        return 1

    # --- Save --------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_df.to_parquet(output_path, engine="pyarrow", index=False)

    # --- Summary -----------------------------------------------------------
    print()
    print("=" * 70)
    print("Manifest Summary")
    print("=" * 70)
    print(f"  Tracks accepted : {len(manifest_df)}")
    print(f"  Tracks rejected : {len(failures)}")
    dur = manifest_df["duration"]
    print(f"  Duration range  : {dur.min():.1f}s — {dur.max():.1f}s  (median {dur.median():.1f}s)")
    sr_counts = manifest_df["sample_rate"].value_counts()
    for sr, count in sr_counts.items():
        print(f"  Sample rate     : {sr} Hz  ({count} tracks)")
    size_kb = output_path.stat().st_size / 1024
    print(f"  Output file     : {output_path}  ({size_kb:.1f} KB)")
    print(f"  Elapsed time    : {elapsed:.1f}s")

    if failures:
        print(f"\n  Rejected files (showing up to 10):")
        for f in failures[:10]:
            print(f"    - {f['file']}: {f['reason']}")

    print("\n" + "=" * 70)
    print("[SUCCESS] Manifest built successfully!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
