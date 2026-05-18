"""
PURPOSE: D3.4 driver for 1001Tracklists matching input (metadata-only mode).
  Two modes:
    - create-input-template: write a blank, ignored input CSV that the user may
      optionally fill later with manually provided 1001Tracklists set metadata.
    - match: load the canonical inventory and the (optional) input CSV, match
      entries against local canonical decoded tracks, and write the committed
      aggregate-only matching report. The ignored matches CSV is written only
      when a usable source exists; the no-usable-source path writes no matches.

  No scraping, no credentials, no website-restriction bypass. 1001Tracklists
  evidence is weak evidence only and never a Regime 1 clustering feature.

CHANGELOG:
  D3.4 - Initial implementation (create-input-template + match; no-source path).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.dj_clustering.match_1001 import (
    SOURCE_STATUS_NONE,
    build_match_report,
    create_input_template,
    load_input,
    load_inventory_canonical,
    match_tracklists,
    write_match_report,
    write_matches,
)

BANNER = "=" * 70

DEFAULT_INPUT = project_root / "artifacts/dj_clustering/tracklists_1001/1001_input_template.csv"
DEFAULT_MATCHES = project_root / "artifacts/dj_clustering/tracklists_1001/1001_matches.csv"
DEFAULT_REPORT = project_root / "reports/dj_clustering/1001_matching_report.md"
DEFAULT_INVENTORY = project_root / "artifacts/dj_clustering/inventory/library_inventory.csv"


def cmd_create_input_template(args: argparse.Namespace) -> int:
    out_path = Path(args.output)

    print(BANNER)
    print("D3.4 — Create 1001Tracklists Input Template")
    print(BANNER)

    n = create_input_template(out_path)
    print(f"[INFO] Input template written: {out_path} ({n} data rows)")
    print("[INFO] Template columns: set_id, set_position, track_artist,")
    print("       track_title, version_marker, duration_seconds, source_url")
    print()
    print("NEXT STEPS (optional):")
    print("  1. To add 1001Tracklists weak evidence later, fill this CSV with")
    print("     manually provided public setlist metadata (no scraping).")
    print("  2. Re-run 'match' once the file has usable rows.")
    print("[INFO] 1001 evidence is optional; Regime 1 does not block on it.")
    print(BANNER)
    return 0


def cmd_match(args: argparse.Namespace) -> int:
    inventory_path = Path(args.inventory)
    input_path = Path(args.input)
    matches_path = Path(args.matches)
    report_path = Path(args.report)

    print(BANNER)
    print("D3.4 — Match 1001Tracklists Input")
    print(BANNER)

    print(f"[INFO] Loading inventory: {inventory_path}")
    inventory_df = load_inventory_canonical(inventory_path)
    print(f"[INFO] Canonical decoded tracks: {len(inventory_df)}")

    if input_path.exists():
        input_df = load_input(input_path)
        print(f"[INFO] Input rows received: {len(input_df)}")
    else:
        import pandas as pd

        from src.dj_clustering.match_1001 import INPUT_TEMPLATE_COLUMNS

        input_df = pd.DataFrame(columns=list(INPUT_TEMPLATE_COLUMNS))
        print("[INFO] No input file present — treating as no usable source.")

    result = match_tracklists(inventory_df, input_df)

    print(f"[INFO] Source status: {result.source_status}")
    print(f"[INFO] Usable input rows: {result.n_usable_rows}")
    print(
        f"[INFO] Accepted: {result.accepted_count} | "
        f"Rejected: {result.rejected_count} | "
        f"Ambiguous: {result.ambiguous_count}"
    )

    report_text = build_match_report(result)
    write_match_report(report_text, report_path)
    print(f"[INFO] Matching report written: {report_path}")

    if result.source_status == SOURCE_STATUS_NONE:
        print("[INFO] No usable source — matches CSV not written (no-source policy).")
    else:
        write_matches(result.matches, matches_path)
        print(f"[INFO] Matches written: {matches_path} ({len(result.matches)} rows)")

    print(BANNER)
    print("D3.4 match complete.")
    print(BANNER)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="D3.4 1001Tracklists matching input (metadata-only)"
    )
    sub = parser.add_subparsers(dest="command")

    p_tpl = sub.add_parser("create-input-template", help="Write blank input template")
    p_tpl.add_argument("--output", default=str(DEFAULT_INPUT))

    p_match = sub.add_parser("match", help="Match input against canonical tracks")
    p_match.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    p_match.add_argument("--input", default=str(DEFAULT_INPUT))
    p_match.add_argument("--matches", default=str(DEFAULT_MATCHES))
    p_match.add_argument("--report", default=str(DEFAULT_REPORT))

    args = parser.parse_args()
    if args.command == "create-input-template":
        return cmd_create_input_template(args)
    if args.command == "match":
        return cmd_match(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
