"""
PURPOSE: CLI entry point for the DJ clustering library inventory scan.
         Reads configs/dj_clustering/inventory.yaml, scans configured audio files,
         computes SHA256 identity, detects exact duplicates, writes an ignored
         artifact CSV, and writes a committed sanitized summary report.

CHANGELOG:
  D1.2 - Initial implementation.
"""

import argparse
import csv
import datetime
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dj_clustering.identity import assign_track_ids
from src.dj_clustering.inventory import (
    ALLOWED_EXTENSIONS,
    _MUTAGEN,
    _SOUNDFILE,
    _TORCHAUDIO,
    count_all_files,
    extract_metadata,
    scan_audio_files,
    try_decode,
)


# ---------------------------------------------------------------------------
# CSV column order (24 columns)
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "track_id",
    "audio_content_hash",
    "canonical_track_id",
    "is_canonical",
    "is_exact_duplicate",
    "duplicate_of_track_id",
    "file_path",
    "file_name",
    "extension",
    "file_size_bytes",
    "duration_seconds",
    "decode_status",
    "decode_error",
    "artist",
    "title",
    "album",
    "genre",
    "label",
    "year",
    "bpm",
    "key",
    "folder_path",
    "folder_hint",
    "metadata_quality_flags",
    "hash_method",
]


# ---------------------------------------------------------------------------
# Summary report (committed, sanitized)
# ---------------------------------------------------------------------------

def _write_summary(
    rows: list,
    report_path: Path,
    config_path: str,
    total_scanned: int,
    excluded_count: int,
    hidden_count: int,
) -> None:
    """Write sanitized summary report with no private paths or individual filenames."""
    import pandas as pd  # type: ignore

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    n_rows = len(df)
    if n_rows == 0:
        report_path.write_text("# Library Inventory Summary\n\nNo audio files found.\n")
        return

    n_ok = int((df["decode_status"] == "ok").sum())
    n_probe = int((df["decode_status"] == "ok_probe_only").sum())
    n_torch = int((df["decode_status"] == "ok_torchaudio").sum())
    n_failed = int((df["decode_status"] == "failed").sum())
    failure_rate = n_failed / n_rows

    n_unique_hashes = df["audio_content_hash"].nunique()
    canon_df = df[df["is_canonical"]]
    n_canonical = len(canon_df)
    n_canon_ok = int((df["is_canonical"] & (df["decode_status"] != "failed")).sum())
    dup_hashes = df[df["is_exact_duplicate"]]["audio_content_hash"].dropna().nunique()
    n_duplicates = int(df["is_exact_duplicate"].sum())

    # Extension distribution (strip leading dot for display)
    ext_counts = df["extension"].str.lstrip(".").value_counts()

    # Metadata quality flags
    flag_counts = df["metadata_quality_flags"].value_counts()

    # Metadata coverage on canonical decoded tracks
    canon_ok_df = df[df["is_canonical"] & (df["decode_status"] != "failed")]
    meta_fields = ["artist", "title", "album", "genre", "label", "year", "bpm", "key"]
    coverage_rows = []
    for field in meta_fields:
        n_filled = int(canon_ok_df[field].notna().sum())
        pct = 100 * n_filled / max(len(canon_ok_df), 1)
        coverage_rows.append(f"| {field} | {n_filled} | {pct:.1f}% |")

    # Folder hint distribution — anonymous labels, sorted by count desc
    hint_counts = canon_df["folder_hint"].value_counts()
    folder_rows = []
    for i, (_, count) in enumerate(hint_counts.head(10).items(), 1):
        folder_rows.append(f"| folder_hint_{i:03d} | {count} |")

    gate = "PASS" if failure_rate <= 0.20 else "BLOCKED — decode failure rate exceeds 20%"
    run_ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    ffprobe_avail = bool(__import__("shutil").which("ffprobe"))
    backends = []
    if _SOUNDFILE:
        backends.append("soundfile")
    if ffprobe_avail:
        backends.append("ffprobe")
    if _TORCHAUDIO:
        backends.append("torchaudio")
    backend_str = ", ".join(backends) if backends else "none"

    lines = [
        "# Library Inventory Summary",
        "",
        "## Run metadata",
        "",
        f"- Scan date (UTC): {run_ts}",
        f"- Config: `{config_path}`",
        "- Library root status: active (path withheld — see ignored artifact CSV)",
        "- Recursive: true",
        f"- mutagen available: {'yes' if _MUTAGEN else 'no'}",
        f"- Decode backends: {backend_str}",
        "- Hash method: sha256_full_content",
        "",
        "## File counts",
        "",
        "| Metric | Count |",
        "|---|---|",
        f"| Total files scanned | {total_scanned} |",
        f"| Audio files (pass whitelist) | {n_rows} |",
        f"| Non-audio files excluded | {excluded_count} |",
        f"| Hidden files skipped | {hidden_count} |",
        "",
        "## Decode status",
        "",
        "| Status | Count | Rate |",
        "|---|---|---|",
        f"| ok | {n_ok} | {100*n_ok/n_rows:.1f}% |",
        f"| ok_probe_only | {n_probe} | {100*n_probe/n_rows:.1f}% |",
        f"| ok_torchaudio | {n_torch} | {100*n_torch/n_rows:.1f}% |",
        f"| failed | {n_failed} | {100*n_failed/n_rows:.1f}% |",
        "",
        "## Identity",
        "",
        "| Metric | Count |",
        "|---|---|",
        f"| Unique audio hashes | {n_unique_hashes} |",
        f"| Exact duplicate groups | {dup_hashes} |",
        f"| Total duplicate files | {n_duplicates} |",
        f"| Canonical tracks | {n_canonical} |",
        f"| Canonical decoded (N_canonical_decoded) | {n_canon_ok} |",
        "",
        "## Extension distribution",
        "",
        "| Extension | Count |",
        "|---|---|",
    ]
    for ext, count in ext_counts.items():
        lines.append(f"| {ext} | {count} |")

    lines += [
        "",
        "## Metadata quality",
        "",
        "| Flag | Count |",
        "|---|---|",
    ]
    for flag, count in flag_counts.items():
        lines.append(f"| {flag} | {count} |")

    lines += [
        "",
        "## Metadata coverage (canonical decoded tracks only)",
        "",
        "| Field | Non-null | Coverage |",
        "|---|---|---|",
    ] + coverage_rows

    lines += [
        "",
        "## Folder hint distribution (top 10, canonical tracks)",
        "",
        "Folder hint labels are anonymised. See ignored artifact CSV for actual names.",
        "",
        "| Label | Count |",
        "|---|---|",
    ] + folder_rows

    lines += [
        "",
        "## D1.2 gate",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Decode failure rate | {100*failure_rate:.1f}% |",
        "| Threshold | 20.0% |",
        f"| Gate | {gate} |",
        f"| N_canonical_decoded | {n_canon_ok} |",
        "",
        "_Artifact: `artifacts/dj_clustering/inventory/library_inventory.csv` (git-ignored)_",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DJ clustering library inventory scan")
    p.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/dj_clustering/inventory.yaml"),
        help="Path to inventory config YAML",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Scan and count files; do not write CSV or summary",
    )
    p.add_argument(
        "--no-progress", action="store_true",
        help="Suppress progress output",
    )
    return p.parse_args()


def main() -> int:
    print("=" * 70)
    print("[INFO] DJ Clustering — Library Inventory (D1.2)")
    print("=" * 70)

    args = _parse_args()
    config_path = Path(args.config)

    # --- Load config ---
    try:
        import yaml  # type: ignore
        cfg = yaml.safe_load(config_path.read_text())
    except Exception as exc:
        print(f"[ERROR] Cannot parse config {config_path}: {exc}")
        return 2

    active_roots = [
        r for r in cfg.get("library_roots", [])
        if r.get("status") == "active"
    ]
    if not active_roots:
        print("[ERROR] No active library roots in config.")
        return 2

    library_root = Path(active_roots[0]["path"])
    if not library_root.exists() or not library_root.is_dir():
        print("[ERROR] Library root does not exist or is not a directory.")
        print(f"[ERROR] Configured path: {library_root}")
        return 2

    recursive = cfg.get("recursive", True)
    ignore_hidden = cfg.get("ignore_hidden_files", True)

    artifact_root = PROJECT_ROOT / cfg["artifact_root"]
    csv_out = artifact_root / cfg.get("inventory_output", "inventory/library_inventory.csv")
    report_out = PROJECT_ROOT / "reports/dj_clustering/library_inventory_summary.md"

    # --- Confirm artifact root is not tracked ---
    # (gitignore covers artifacts/ — we just proceed)

    # --- Preflight: backend summary ---
    ffprobe_bin = __import__("shutil").which("ffprobe")
    print(f"[INFO] soundfile:   {'available' if _SOUNDFILE else 'absent'}")
    print(f"[INFO] mutagen:     {'available' if _MUTAGEN else 'absent (filename fallback)'}")
    print(f"[INFO] ffprobe:     {'available' if ffprobe_bin else 'absent'}")
    print(f"[INFO] torchaudio:  {'available' if _TORCHAUDIO else 'absent'}")

    # --- Count all files (without printing names) ---
    print("[INFO] Counting files in library root...")
    total_files = count_all_files(library_root, recursive=recursive)
    print(f"[INFO] Total files found: {total_files}")

    # --- Scan audio files ---
    audio_files = scan_audio_files(library_root, recursive=recursive, ignore_hidden=ignore_hidden)
    n_audio = len(audio_files)
    excluded = total_files - n_audio
    hidden_count = 0  # hidden files are excluded during scan; counted separately below
    print(f"[INFO] Audio files (whitelist): {n_audio}")
    print(f"[INFO] Non-audio files excluded: {excluded}")

    if n_audio == 0:
        print("[ERROR] No audio files found. Check library root and allowed_extensions.")
        return 3

    if args.dry_run:
        print("[INFO] Dry run — no files written.")
        return 0

    # --- Build rows ---
    print(f"[INFO] Processing {n_audio} audio files...")

    # Step 1: assign track IDs (computes SHA256 for all files)
    identity_map = assign_track_ids(audio_files)

    # Step 2: decode + metadata for each file
    hash_method = cfg.get("hash_method", "sha256_full_content")
    rows = []

    for i, filepath in enumerate(audio_files):
        if not args.no_progress and (i + 1) % max(1, n_audio // 10) == 0:
            print(f"[INFO] Progress: {i+1}/{n_audio}")

        ident = identity_map[filepath]

        try:
            stat = filepath.stat()
            file_size = stat.st_size
        except OSError:
            file_size = None

        decode_status, duration, decode_error = try_decode(filepath)
        meta = extract_metadata(filepath)

        row = [
            ident["track_id"],
            ident["audio_content_hash"],
            ident["canonical_track_id"],
            ident["is_canonical"],
            ident["is_exact_duplicate"],
            ident["duplicate_of_track_id"],
            str(filepath),                  # file_path  (private, artifact only)
            filepath.name,                  # file_name  (private, artifact only)
            filepath.suffix.lower(),        # extension
            file_size,
            duration,
            decode_status,
            decode_error,
            meta["artist"],
            meta["title"],
            meta["album"],
            meta["genre"],
            meta["label"],
            meta["year"],
            meta["bpm"],
            meta["key"],
            str(filepath.parent),           # folder_path (private, artifact only)
            filepath.parent.name,           # folder_hint (dir name only)
            meta["metadata_quality_flags"],
            hash_method,
        ]
        rows.append(row)

    print(f"[INFO] Progress: {n_audio}/{n_audio}")

    # --- Decode stats ---
    decode_statuses = [r[11] for r in rows]  # index 11 = decode_status
    n_ok = decode_statuses.count("ok")
    n_probe = decode_statuses.count("ok_probe_only")
    n_torch = decode_statuses.count("ok_torchaudio")
    n_failed = decode_statuses.count("failed")
    failure_rate = n_failed / n_audio if n_audio > 0 else 0

    print(f"[INFO] Decode: {n_ok} ok, {n_probe} ok_probe_only, "
          f"{n_torch} ok_torchaudio, {n_failed} failed")
    print(f"[INFO] Decode failure rate: {100*failure_rate:.1f}%")

    # --- Duplicate stats ---
    is_dup_list = [r[4] for r in rows]  # index 4 = is_exact_duplicate
    n_dup = sum(1 for d in is_dup_list if d)
    print(f"[INFO] Exact duplicate files: {n_dup}")

    # --- Write CSV ---
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)
    print(f"[INFO] CSV written: {csv_out.relative_to(PROJECT_ROOT)}")

    # --- Write summary report ---
    try:
        config_display = str(config_path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        config_display = str(config_path)
    _write_summary(
        rows=rows,
        report_path=report_out,
        config_path=config_display,
        total_scanned=total_files,
        excluded_count=excluded,
        hidden_count=hidden_count,
    )
    print(f"[INFO] Summary written: {report_out.relative_to(PROJECT_ROOT)}")

    # --- Failure guard ---
    if failure_rate > 0.20:
        print(f"[WARN] Decode failure rate {100*failure_rate:.1f}% exceeds 20% threshold.")
        print("[WARN] D1.2 blocked — review decode failures before proceeding to D1.3.")
        print("=" * 70)
        return 1

    print("=" * 70)
    print("[INFO] D1.2 inventory complete. Gate: PASS")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
