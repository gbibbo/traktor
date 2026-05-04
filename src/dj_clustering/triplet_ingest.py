"""
PURPOSE: D3.3 manual triplet answer ingest library. Provides answer template
         creation, answer validation (allowed values B/C/skip, unknown/duplicate
         question_id detection, unanswered reporting), manual triplet DataFrame
         assembly (non-skip only), skip log assembly, and committed summary
         report generation (aggregate counts only, privacy-clean).

CHANGELOG:
  D3.3a - Initial implementation.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_ANSWERS: frozenset = frozenset({"B", "C", "skip"})

ANSWER_TEMPLATE_COLUMNS: Tuple[str, ...] = (
    "question_id",
    "answer",
    "confidence",
    "notes",
)

MANUAL_TRIPLET_COLUMNS: Tuple[str, ...] = (
    "question_id",
    "anchor_track_id",
    "candidate_b_track_id",
    "candidate_c_track_id",
    "selection_source",
    "answer",
    "confidence",
    "notes",
)

SKIP_LOG_COLUMNS: Tuple[str, ...] = (
    "question_id",
    "anchor_track_id",
    "candidate_b_track_id",
    "candidate_c_track_id",
    "selection_source",
    "answer",
    "confidence",
    "notes",
)

QUEUE_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "question_id",
    "anchor_track_id",
    "candidate_b_track_id",
    "candidate_c_track_id",
    "selection_source",
)

# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ValidationResult:
    valid_rows: pd.DataFrame
    skip_rows: pd.DataFrame
    hard_errors: List[str]
    soft_warnings: List[str]

    @property
    def has_hard_errors(self) -> bool:
        return bool(self.hard_errors)


# ---------------------------------------------------------------------------
# Queue loading
# ---------------------------------------------------------------------------


def load_queue(queue_csv: Path) -> pd.DataFrame:
    """Load the triplet question queue CSV and verify required columns."""
    queue_csv = Path(queue_csv)
    if not queue_csv.exists():
        raise FileNotFoundError(f"Queue CSV not found: {queue_csv}")
    df = pd.read_csv(queue_csv, dtype=str)
    missing = [c for c in QUEUE_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Queue CSV missing required columns: {missing}")
    return df


# ---------------------------------------------------------------------------
# Answer template creation
# ---------------------------------------------------------------------------


def create_answer_template(queue_df: pd.DataFrame, out_path: Path) -> int:
    """Write a blank answer template CSV from the question queue.

    Only question_id is copied; answer/confidence/notes are left blank.
    Returns the number of rows written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "question_id": row["question_id"],
            "answer": "",
            "confidence": "",
            "notes": "",
        }
        for _, row in queue_df.iterrows()
    ]
    template_df = pd.DataFrame(rows, columns=list(ANSWER_TEMPLATE_COLUMNS))
    template_df.to_csv(out_path, index=False)
    return len(rows)


# ---------------------------------------------------------------------------
# Answer loading
# ---------------------------------------------------------------------------


def load_answers(answers_csv: Path) -> pd.DataFrame:
    """Load a user-provided answers CSV."""
    answers_csv = Path(answers_csv)
    if not answers_csv.exists():
        raise FileNotFoundError(f"Answers CSV not found: {answers_csv}")
    return pd.read_csv(answers_csv, dtype=str).fillna("")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_answers(
    queue_df: pd.DataFrame,
    answers_df: pd.DataFrame,
) -> ValidationResult:
    """Validate answers against the queue.

    Hard errors (any → ingest aborts, no output written):
      - `answer` column absent from answers CSV
      - unknown question_id (not in queue)
      - duplicate question_id in answers
      - answer value not in ALLOWED_ANSWERS (for non-empty rows)

    Soft warnings (logged in summary, ingest continues):
      - questions in queue with no corresponding answer row

    Returns ValidationResult with valid_rows (non-skip), skip_rows, errors,
    and warnings.
    """
    hard_errors: List[str] = []
    soft_warnings: List[str] = []

    # Hard: answer column must exist
    if "answer" not in answers_df.columns:
        hard_errors.append("answers CSV is missing the required 'answer' column")
        return ValidationResult(
            valid_rows=pd.DataFrame(columns=list(MANUAL_TRIPLET_COLUMNS)),
            skip_rows=pd.DataFrame(columns=list(SKIP_LOG_COLUMNS)),
            hard_errors=hard_errors,
            soft_warnings=soft_warnings,
        )

    queue_qids = set(queue_df["question_id"].astype(str))
    answer_qids = list(answers_df["question_id"].astype(str))

    # Hard: unknown question_ids
    unknown = [q for q in answer_qids if q not in queue_qids]
    if unknown:
        hard_errors.append(
            f"unknown question_id values in answers ({len(unknown)}): {sorted(unknown)}"
        )

    # Hard: duplicates
    seen: set = set()
    duplicates: List[str] = []
    for q in answer_qids:
        if q in seen:
            duplicates.append(q)
        seen.add(q)
    if duplicates:
        hard_errors.append(
            f"duplicate question_id values in answers ({len(duplicates)}): {sorted(set(duplicates))}"
        )

    # Hard: bad answer values (for non-blank rows)
    answered_mask = answers_df["answer"].str.strip() != ""
    answered_df = answers_df[answered_mask]
    bad_values = answered_df[~answered_df["answer"].str.strip().isin(ALLOWED_ANSWERS)]
    if not bad_values.empty:
        bad_list = sorted(bad_values["answer"].str.strip().unique().tolist())
        hard_errors.append(
            f"invalid answer values (allowed: B, C, skip) — found: {bad_list}"
        )

    if hard_errors:
        return ValidationResult(
            valid_rows=pd.DataFrame(columns=list(MANUAL_TRIPLET_COLUMNS)),
            skip_rows=pd.DataFrame(columns=list(SKIP_LOG_COLUMNS)),
            hard_errors=hard_errors,
            soft_warnings=soft_warnings,
        )

    # Soft: unanswered questions
    answer_qid_set = set(answer_qids)
    unanswered = [q for q in queue_qids if q not in answer_qid_set]
    if unanswered:
        soft_warnings.append(
            f"{len(unanswered)} question(s) in queue have no answer row"
        )

    # Split valid non-skip vs skip rows
    answered_df = answers_df[answered_df["answer"].str.strip() != ""].copy()
    answered_df["answer"] = answered_df["answer"].str.strip()

    non_skip = answered_df[answered_df["answer"] != "skip"].copy()
    skip_rows = answered_df[answered_df["answer"] == "skip"].copy()

    return ValidationResult(
        valid_rows=non_skip,
        skip_rows=skip_rows,
        hard_errors=hard_errors,
        soft_warnings=soft_warnings,
    )


# ---------------------------------------------------------------------------
# Output DataFrame assembly
# ---------------------------------------------------------------------------


def build_manual_triplets(
    queue_df: pd.DataFrame,
    valid_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Join valid (non-skip) answers with queue to produce manual_triplets DataFrame."""
    if valid_rows.empty:
        return pd.DataFrame(columns=list(MANUAL_TRIPLET_COLUMNS))

    merged = queue_df[list(QUEUE_REQUIRED_COLUMNS)].merge(
        valid_rows[["question_id", "answer", "confidence", "notes"]],
        on="question_id",
        how="inner",
    )
    for col in MANUAL_TRIPLET_COLUMNS:
        if col not in merged.columns:
            merged[col] = ""
    return merged[list(MANUAL_TRIPLET_COLUMNS)].reset_index(drop=True)


def build_skip_log(
    queue_df: pd.DataFrame,
    skip_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Join skip answers with queue for audit log."""
    if skip_rows.empty:
        return pd.DataFrame(columns=list(SKIP_LOG_COLUMNS))

    merged = queue_df[list(QUEUE_REQUIRED_COLUMNS)].merge(
        skip_rows[["question_id", "answer", "confidence", "notes"]],
        on="question_id",
        how="inner",
    )
    for col in SKIP_LOG_COLUMNS:
        if col not in merged.columns:
            merged[col] = ""
    return merged[list(SKIP_LOG_COLUMNS)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def build_summary(
    queue_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    skip_df: pd.DataFrame,
    validation_result: ValidationResult,
) -> str:
    """Build the committed summary markdown. Contains aggregate counts only."""
    n_queue = len(queue_df)
    n_valid = len(valid_df)
    n_skip = len(skip_df)
    n_answered = n_valid + n_skip
    n_unanswered = n_queue - n_answered
    n_b = int((valid_df["answer"] == "B").sum()) if not valid_df.empty else 0
    n_c = int((valid_df["answer"] == "C").sum()) if not valid_df.empty else 0

    pct_b = f"{100 * n_b / n_answered:.1f}%" if n_answered > 0 else "—"
    pct_c = f"{100 * n_c / n_answered:.1f}%" if n_answered > 0 else "—"
    pct_skip = f"{100 * n_skip / n_answered:.1f}%" if n_answered > 0 else "—"

    diagnostic_note = (
        "**YES** — evidence available for sweep"
        if n_valid >= 20
        else "**NO** — fewer than 20 non-skip answers; diagnostic sweeps only (no winner until D5.1)"
    )

    # Source breakdown (by selection_source from queue)
    source_counts: dict = {}
    for _, row in queue_df.iterrows():
        src = str(row.get("selection_source", "unknown"))
        if src not in source_counts:
            source_counts[src] = {"answered": 0, "skip": 0, "unanswered": 0}

    answered_qids = set()
    if not valid_df.empty:
        answered_qids.update(valid_df["question_id"].tolist())
    skip_qids = set()
    if not skip_df.empty:
        skip_qids.update(skip_df["question_id"].tolist())

    for _, row in queue_df.iterrows():
        src = str(row.get("selection_source", "unknown"))
        qid = str(row["question_id"])
        if qid in answered_qids:
            source_counts[src]["answered"] += 1
        elif qid in skip_qids:
            source_counts[src]["skip"] += 1
        else:
            source_counts[src]["unanswered"] += 1

    src_table_lines = ["| selection_source | answered | skip | unanswered |", "|---|---|---|---|"]
    for src in sorted(source_counts):
        sc = source_counts[src]
        src_table_lines.append(
            f"| {src} | {sc['answered']} | {sc['skip']} | {sc['unanswered']} |"
        )

    warnings_section = ""
    if validation_result.soft_warnings:
        warnings_section = "\n## Validation Warnings\n\n"
        for w in validation_result.soft_warnings:
            warnings_section += f"- {w}\n"

    lines = [
        "# Manual Triplets Ingest Summary",
        "",
        "## Counts",
        "",
        f"- Questions in queue: {n_queue}",
        f"- Answers provided: {n_answered}",
        f"- Valid (B or C): {n_valid}",
        f"- Skip: {n_skip}",
        f"- Unanswered: {n_unanswered}",
        f"- Validation warnings: {len(validation_result.soft_warnings)}",
        "",
        "## Answer Distribution",
        "",
        f"- B: {n_b} ({pct_b})",
        f"- C: {n_c} ({pct_c})",
        f"- skip: {n_skip} ({pct_skip})",
        "",
        "## Source Breakdown",
        "",
        *src_table_lines,
        "",
        "## Decision Rule Check",
        "",
        f"- Non-skip answers ≥ 20: {diagnostic_note}",
        "",
    ]

    if warnings_section:
        lines.append(warnings_section)

    lines.append("_Last updated: D3.3_")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------


def write_manual_triplets(manual_df: pd.DataFrame, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manual_df.to_csv(out_path, index=False)


def write_skip_log(skip_df: pd.DataFrame, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skip_df.to_csv(out_path, index=False)


def write_summary_report(summary_text: str, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(summary_text, encoding="utf-8")
