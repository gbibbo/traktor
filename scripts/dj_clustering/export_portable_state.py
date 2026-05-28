#!/usr/bin/env python3
"""Export portable state for the DJ clustering cluster-exit checkpoint.

PURPOSE
    Package the critical (and optionally the optional) ignored DJ clustering
    artifacts into portable, checksummed archives so the project can be
    resumed off the Surrey cluster (local CPU host, paid GPU VM, or another
    HPC). This is the EXIT.2-EXIT.4 mechanism of the cluster-exit amendment
    documented in docs/migration/dj_clustering_cluster_exit_plan.md.

    The tool is privacy-aware: the per-file raw-audio checksum manifest (which
    contains music filenames) is written ONLY under the ignored export
    directory and must never be committed. Committed docs carry verification-
    safe aggregates only (file count, total bytes, manifest sha256).

    This tool never moves audio, never touches V4/legacy paths, never edits
    trackers, and never stages anything. It only reads critical artifacts and
    writes outputs under the ignored export directory.

CHANGELOG
    2026-05-28  EXIT.1-EXIT.4  Initial version. Inventory critical/optional
                artifacts, build .tar.zst archives, emit SHA256SUMS, build the
                ignored per-file raw-audio checksum manifest plus its safe
                aggregate, and write an ignored export manifest JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

# Repo-relative items. Directories are archived whole; files individually.
CRITICAL_ITEMS = [
    "artifacts/dj_clustering/inventory",
    "artifacts/dj_clustering/features",
    "artifacts/dj_clustering/pairs",
    "artifacts/dj_clustering/triplets",
    "artifacts/dj_clustering/tracklists_1001",
    "runs/dj_clustering/first_sweep",
]
OPTIONAL_ITEMS = [
    "artifacts/dj_clustering/features_smoke",
    "runs/dj_clustering/first_sweep_validation",
]

DEFAULT_RAW_AUDIO_ROOT = (
    "/mnt/fast/nobackup/scratch4weeks/gb0048/"
    "traktor_dj_clustering/raw_audio"
)


def sha256_file(path: Path) -> str:
    """Return the hex sha256 digest of a file, read in chunks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_items(repo_root: Path, items: list[str]) -> tuple[list[str], list[str]]:
    """Split repo-relative items into (present, missing)."""
    present, missing = [], []
    for rel in items:
        (present if (repo_root / rel).exists() else missing).append(rel)
    return present, missing


def raw_audio_aggregate(audio_root: Path) -> dict:
    """Compute verification-safe aggregate stats over raw audio.

    Returns only counts/bytes - never per-file paths - so the result is safe
    to embed in committed docs.
    """
    files = [p for p in audio_root.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    return {"raw_audio_file_count": len(files), "raw_audio_total_bytes": total}


def write_raw_audio_manifest(audio_root: Path, out_path: Path) -> str:
    """Write a per-file sha256 manifest (relative paths) and return its sha256.

    The manifest contains filenames and therefore MUST stay under the ignored
    export directory and must not be committed.
    """
    files = sorted(
        (p for p in audio_root.rglob("*") if p.is_file()),
        key=lambda p: str(p.relative_to(audio_root)),
    )
    lines = []
    for p in files:
        rel = p.relative_to(audio_root).as_posix()
        lines.append(f"{sha256_file(p)}  {rel}")
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return sha256_file(out_path)


def build_archive(repo_root: Path, members: list[str], out_path: Path) -> None:
    """Build a deterministic-ordering .tar.zst archive of repo-relative members."""
    cmd = [
        "tar", "--zstd", "--sort=name",
        "-cf", str(out_path),
        "-C", str(repo_root),
        *members,
    ]
    subprocess.run(cmd, check=True)


def build_manifest(stamp: str, archives: list[dict], raw_audio: dict) -> dict:
    """Assemble the ignored export manifest dictionary."""
    return {
        "schema": "dj_clustering_export/1",
        "stamp": stamp,
        "surrey_access_end": "2026-05-31",
        "archives": archives,
        "raw_audio": raw_audio,
        "note": (
            "Critical resume payload. Restore by extracting each archive at "
            "the repo root; verify with sha256sum -c SHA256SUMS.txt; then run "
            "the dj_clustering test suite. Raw audio travels out-of-band; the "
            "per-file manifest is ignored and not committed."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    ap.add_argument(
        "--export-dir",
        type=Path,
        default=Path("artifacts/dj_clustering/_export"),
        help="Output dir (relative to repo-root unless absolute); ignored by git.",
    )
    ap.add_argument("--stamp", default="20260528")
    ap.add_argument("--raw-audio-root", type=Path, default=Path(DEFAULT_RAW_AUDIO_ROOT))
    ap.add_argument(
        "--with-optional", action="store_true",
        help="Also build the optional archive (smoke + validation outputs).",
    )
    ap.add_argument(
        "--skip-archives", action="store_true",
        help="Inventory + checksums only; do not build tar.zst (used by tests).",
    )
    ap.add_argument(
        "--skip-raw-audio", action="store_true",
        help="Skip the raw-audio per-file manifest (e.g. audio root absent).",
    )
    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    export_dir = args.export_dir
    if not export_dir.is_absolute():
        export_dir = repo_root / export_dir
    export_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("[INFO] DJ clustering portable-state export")
    print(f"[INFO] repo_root  = {repo_root}")
    print(f"[INFO] export_dir = {export_dir}")
    print("=" * 70)

    crit_present, crit_missing = resolve_items(repo_root, CRITICAL_ITEMS)
    if crit_missing:
        for m in crit_missing:
            print(f"[ERROR] missing critical item: {m}")
        print("[ERROR] aborting: critical artifacts incomplete")
        return 1
    print(f"[INFO] critical items present: {len(crit_present)}")

    opt_present, opt_missing = resolve_items(repo_root, OPTIONAL_ITEMS)
    for m in opt_missing:
        print(f"[INFO] optional item absent (ok): {m}")

    archives: list[dict] = []

    def record_archive(name: str, members: list[str]) -> None:
        out = export_dir / name
        if not args.skip_archives:
            build_archive(repo_root, members, out)
            digest = sha256_file(out)
            size = out.stat().st_size
        else:
            digest, size = "(skipped)", 0
        archives.append({
            "name": name,
            "members": members,
            "sha256": digest,
            "size_bytes": size,
        })
        print(f"[INFO] archive {name}: sha256={digest} size={size}")

    record_archive(f"dj_clustering_critical_{args.stamp}.tar.zst", crit_present)
    if args.with_optional and opt_present:
        record_archive(f"dj_clustering_optional_{args.stamp}.tar.zst", opt_present)

    # SHA256SUMS for the archives (committed-safe values; file itself ignored).
    if not args.skip_archives:
        sums_path = export_dir / "SHA256SUMS.txt"
        sums_path.write_text(
            "".join(f"{a['sha256']}  {a['name']}\n" for a in archives)
        )
        print(f"[INFO] wrote {sums_path}")

    # Raw audio: per-file manifest (ignored) + safe aggregate.
    raw_audio = {
        "manifest_export_path": str(
            (export_dir / f"raw_audio_checksums_{args.stamp}.txt")
        ),
        "backup_destination": "user_decision_out_of_band",
    }
    if not args.skip_raw_audio and args.raw_audio_root.exists():
        agg = raw_audio_aggregate(args.raw_audio_root)
        ra_manifest = export_dir / f"raw_audio_checksums_{args.stamp}.txt"
        ra_sha = write_raw_audio_manifest(args.raw_audio_root, ra_manifest)
        raw_audio.update(agg)
        raw_audio["manifest_sha256"] = ra_sha
        raw_audio["source_path"] = str(args.raw_audio_root)
        print(
            f"[INFO] raw audio: count={agg['raw_audio_file_count']} "
            f"bytes={agg['raw_audio_total_bytes']} manifest_sha256={ra_sha}"
        )
    else:
        print("[INFO] raw audio manifest skipped")

    manifest = build_manifest(args.stamp, archives, raw_audio)
    man_path = export_dir / f"export_manifest_{args.stamp}.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[INFO] wrote {man_path}")
    print("=" * 70)
    print("[INFO] export complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
