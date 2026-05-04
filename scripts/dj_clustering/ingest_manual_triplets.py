"""
PURPOSE: D3.3 driver. Manages the manual triplet answer lifecycle:
  - create-template: reads triplet_question_queue.csv and writes a blank
    answer template CSV (question_id, answer, confidence, notes).
  - ingest: validates a user-completed answers CSV, writes manual_triplets.csv
    (non-skip answered pairs), triplet_skip_log.csv, and the committed summary
    report manual_triplets_summary.md.

CHANGELOG:
  D3.3a - Initial implementation (create-template mode only; ingest scaffolded).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.dj_clustering.triplet_ingest import (
    ALLOWED_ANSWERS,
    build_manual_triplets,
    build_skip_log,
    build_summary,
    create_answer_template,
    load_answers,
    load_queue,
    validate_answers,
    write_manual_triplets,
    write_skip_log,
    write_summary_report,
)

BANNER = "=" * 70

DEFAULT_QUEUE = project_root / "artifacts/dj_clustering/triplets/triplet_question_queue.csv"
DEFAULT_TEMPLATE = project_root / "artifacts/dj_clustering/triplets/triplet_answer_template.csv"
DEFAULT_MANUAL_TRIPLETS = project_root / "artifacts/dj_clustering/triplets/manual_triplets.csv"
DEFAULT_SKIP_LOG = project_root / "artifacts/dj_clustering/triplets/triplet_skip_log.csv"
DEFAULT_SUMMARY = project_root / "reports/dj_clustering/manual_triplets_summary.md"


def cmd_create_template(args: argparse.Namespace) -> int:
    queue_path = Path(args.queue)
    out_path = Path(args.output)

    print(BANNER)
    print("D3.3a — Create Manual Triplet Answer Template")
    print(BANNER)

    print(f"[INFO] Loading queue: {queue_path}")
    queue_df = load_queue(queue_path)
    print(f"[INFO] Queue rows: {len(queue_df)}")

    n = create_answer_template(queue_df, out_path)
    print(f"[INFO] Template written: {out_path} ({n} rows)")
    print()
    print(f"[INFO] Template columns: question_id, answer, confidence, notes")
    print(f"[INFO] Allowed answer values: {sorted(ALLOWED_ANSWERS)}")
    print()
    print("NEXT STEPS:")
    print(f"  1. Open: {out_path}")
    print("  2. Fill the 'answer' column for each row: B, C, or skip")
    print("  3. Optionally fill 'confidence' (1–5) and 'notes'")
    print("  4. Run ingest mode:")
    print(f"     python {Path(__file__).name} ingest --answers {out_path}")
    print()
    print("D3.3 is BLOCKED until user-completed answers are ingested.")
    print(BANNER)
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    queue_path = Path(args.queue)
    answers_path = Path(args.answers)

    print(BANNER)
    print("D3.3 — Ingest Manual Triplet Answers")
    print(BANNER)

    print(f"[INFO] Loading queue: {queue_path}")
    queue_df = load_queue(queue_path)
    print(f"[INFO] Queue rows: {len(queue_df)}")

    print(f"[INFO] Loading answers: {answers_path}")
    answers_df = load_answers(answers_path)
    print(f"[INFO] Answer rows: {len(answers_df)}")

    print("[INFO] Validating answers...")
    result = validate_answers(queue_df, answers_df)

    if result.has_hard_errors:
        print("[ERROR] Validation failed — no output written.")
        for err in result.hard_errors:
            print(f"  [ERROR] {err}")
        return 1

    for warn in result.soft_warnings:
        print(f"  [WARN] {warn}")

    n_valid = len(result.valid_rows)
    n_skip = len(result.skip_rows)
    print(f"[INFO] Valid (B or C): {n_valid} | Skip: {n_skip}")

    manual_df = build_manual_triplets(queue_df, result.valid_rows)
    skip_df = build_skip_log(queue_df, result.skip_rows)

    out_manual = Path(args.output_manual)
    out_skip = Path(args.output_skip)
    out_summary = Path(args.output_summary)

    write_manual_triplets(manual_df, out_manual)
    print(f"[INFO] manual_triplets.csv written: {out_manual} ({len(manual_df)} rows)")

    write_skip_log(skip_df, out_skip)
    print(f"[INFO] triplet_skip_log.csv written: {out_skip} ({len(skip_df)} rows)")

    summary_text = build_summary(queue_df, manual_df, skip_df, result)
    write_summary_report(summary_text, out_summary)
    print(f"[INFO] summary report written: {out_summary}")

    if n_valid < 20:
        print("[WARN] Fewer than 20 non-skip answers — diagnostic sweeps only.")

    print(BANNER)
    print("D3.3 ingest complete.")
    print(BANNER)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="D3.3 manual triplet answer management"
    )
    sub = parser.add_subparsers(dest="command")

    p_create = sub.add_parser("create-template", help="Generate blank answer template")
    p_create.add_argument("--queue", default=str(DEFAULT_QUEUE))
    p_create.add_argument("--output", default=str(DEFAULT_TEMPLATE))

    p_ingest = sub.add_parser("ingest", help="Validate and ingest completed answers")
    p_ingest.add_argument("--answers", required=True)
    p_ingest.add_argument("--queue", default=str(DEFAULT_QUEUE))
    p_ingest.add_argument("--output-manual", default=str(DEFAULT_MANUAL_TRIPLETS))
    p_ingest.add_argument("--output-skip", default=str(DEFAULT_SKIP_LOG))
    p_ingest.add_argument("--output-summary", default=str(DEFAULT_SUMMARY))

    args = parser.parse_args()
    if args.command == "create-template":
        return cmd_create_template(args)
    if args.command == "ingest":
        return cmd_ingest(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
