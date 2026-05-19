"""
PURPOSE: Unit tests for src/dj_clustering/triplet_ingest.py — answer template
         creation, answer validation (allowed values, unknown/duplicate question_ids,
         unanswered soft warnings), manual triplet and skip log DataFrame assembly,
         summary report generation (privacy and decision rule). Pure Python; no
         audio, no GPU, no HF downloads.

CHANGELOG:
  D3.3a - Initial implementation.
  D3.3b - Add case-insensitive answer normalization tests (B/C/skip).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dj_clustering.triplet_ingest import (
    ALLOWED_ANSWERS,
    ANSWER_TEMPLATE_COLUMNS,
    MANUAL_TRIPLET_COLUMNS,
    build_manual_triplets,
    build_skip_log,
    build_summary,
    collect_answered_ids,
    create_answer_template,
    filter_unanswered,
    load_answers,
    load_queue,
    merge_answer_rows,
    validate_answers,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_queue(n: int = 5) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        rows.append(
            {
                "question_id": f"Q{i:03d}",
                "anchor_track_id": f"tid_a{i}",
                "candidate_b_track_id": f"tid_b{i}",
                "candidate_c_track_id": f"tid_c{i}",
                "selection_source": "mert_full_knn" if i % 2 == 0 else "v4_boundary",
            }
        )
    return pd.DataFrame(rows)


def _make_answers(qids, values) -> pd.DataFrame:
    rows = []
    for qid, val in zip(qids, values):
        rows.append({"question_id": qid, "answer": val, "confidence": "", "notes": ""})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Template tests
# ---------------------------------------------------------------------------


def test_create_template_columns(tmp_path):
    queue_df = _make_queue(5)
    out = tmp_path / "template.csv"
    create_answer_template(queue_df, out)
    df = pd.read_csv(out, dtype=str)
    assert list(df.columns) == list(ANSWER_TEMPLATE_COLUMNS)


def test_create_template_rows(tmp_path):
    queue_df = _make_queue(8)
    out = tmp_path / "template.csv"
    n = create_answer_template(queue_df, out)
    assert n == 8
    df = pd.read_csv(out, dtype=str).fillna("")
    assert len(df) == 8


def test_create_template_answer_blank(tmp_path):
    queue_df = _make_queue(4)
    out = tmp_path / "template.csv"
    create_answer_template(queue_df, out)
    df = pd.read_csv(out, dtype=str).fillna("")
    assert all(df["answer"] == "")
    assert all(df["confidence"] == "")
    assert all(df["notes"] == "")


def test_create_template_question_ids(tmp_path):
    queue_df = _make_queue(5)
    out = tmp_path / "template.csv"
    create_answer_template(queue_df, out)
    df = pd.read_csv(out, dtype=str)
    assert list(df["question_id"]) == list(queue_df["question_id"])


# ---------------------------------------------------------------------------
# Validation — happy path
# ---------------------------------------------------------------------------


def test_validate_ok_b_c():
    queue_df = _make_queue(4)
    answers_df = _make_answers(["Q001", "Q002", "Q003", "Q004"], ["B", "C", "B", "C"])
    result = validate_answers(queue_df, answers_df)
    assert not result.has_hard_errors
    assert len(result.valid_rows) == 4
    assert len(result.skip_rows) == 0


def test_validate_ok_with_skip():
    queue_df = _make_queue(4)
    answers_df = _make_answers(["Q001", "Q002", "Q003", "Q004"], ["B", "skip", "C", "skip"])
    result = validate_answers(queue_df, answers_df)
    assert not result.has_hard_errors
    assert len(result.valid_rows) == 2
    assert len(result.skip_rows) == 2


# ---------------------------------------------------------------------------
# Validation — hard errors
# ---------------------------------------------------------------------------


def test_validate_missing_answer_column():
    queue_df = _make_queue(3)
    answers_df = pd.DataFrame({"question_id": ["Q001"], "confidence": [""]})
    result = validate_answers(queue_df, answers_df)
    assert result.has_hard_errors
    assert any("answer" in e for e in result.hard_errors)


def test_validate_unknown_qid():
    queue_df = _make_queue(3)
    answers_df = _make_answers(["Q001", "Q999"], ["B", "C"])
    result = validate_answers(queue_df, answers_df)
    assert result.has_hard_errors
    assert any("Q999" in e for e in result.hard_errors)


def test_validate_duplicate_qid():
    queue_df = _make_queue(3)
    answers_df = _make_answers(["Q001", "Q001", "Q002"], ["B", "C", "B"])
    result = validate_answers(queue_df, answers_df)
    assert result.has_hard_errors
    assert any("duplicate" in e.lower() for e in result.hard_errors)


def test_validate_bad_answer_value():
    queue_df = _make_queue(3)
    answers_df = _make_answers(["Q001", "Q002", "Q003"], ["B", "X", "C"])
    result = validate_answers(queue_df, answers_df)
    assert result.has_hard_errors
    assert any("X" in e or "invalid" in e.lower() for e in result.hard_errors)


def test_validate_unsure_rejected():
    queue_df = _make_queue(2)
    answers_df = _make_answers(["Q001", "Q002"], ["B", "unsure"])
    result = validate_answers(queue_df, answers_df)
    assert result.has_hard_errors


def test_validate_uppercase_unsure_rejected():
    queue_df = _make_queue(2)
    answers_df = _make_answers(["Q001", "Q002"], ["B", "UNSURE"])
    result = validate_answers(queue_df, answers_df)
    assert result.has_hard_errors


# ---------------------------------------------------------------------------
# Validation — case-insensitive answer normalization (D3.3b)
# ---------------------------------------------------------------------------


def test_validate_skip_case_insensitive():
    queue_df = _make_queue(4)
    answers_df = _make_answers(
        ["Q001", "Q002", "Q003", "Q004"], ["SKIP", "Skip", "skip", "B"]
    )
    result = validate_answers(queue_df, answers_df)
    assert not result.has_hard_errors
    assert len(result.skip_rows) == 3
    assert all(result.skip_rows["answer"] == "skip")
    assert len(result.valid_rows) == 1


def test_validate_b_case_insensitive():
    queue_df = _make_queue(2)
    answers_df = _make_answers(["Q001", "Q002"], ["b", "B"])
    result = validate_answers(queue_df, answers_df)
    assert not result.has_hard_errors
    assert len(result.valid_rows) == 2
    assert all(result.valid_rows["answer"] == "B")


def test_validate_c_case_insensitive():
    queue_df = _make_queue(2)
    answers_df = _make_answers(["Q001", "Q002"], ["c", "C"])
    result = validate_answers(queue_df, answers_df)
    assert not result.has_hard_errors
    assert len(result.valid_rows) == 2
    assert all(result.valid_rows["answer"] == "C")


# ---------------------------------------------------------------------------
# Validation — soft warnings
# ---------------------------------------------------------------------------


def test_validate_unanswered_soft():
    queue_df = _make_queue(5)
    answers_df = _make_answers(["Q001", "Q002"], ["B", "C"])
    result = validate_answers(queue_df, answers_df)
    assert not result.has_hard_errors
    assert len(result.soft_warnings) > 0


def test_validate_all_answered_no_warning():
    queue_df = _make_queue(3)
    answers_df = _make_answers(["Q001", "Q002", "Q003"], ["B", "C", "skip"])
    result = validate_answers(queue_df, answers_df)
    assert not result.has_hard_errors
    assert len(result.soft_warnings) == 0


# ---------------------------------------------------------------------------
# ALLOWED_ANSWERS constant
# ---------------------------------------------------------------------------


def test_allowed_answers_set():
    assert ALLOWED_ANSWERS == frozenset({"B", "C", "skip"})
    assert "unsure" not in ALLOWED_ANSWERS


# ---------------------------------------------------------------------------
# Manual triplets assembly
# ---------------------------------------------------------------------------


def test_build_manual_triplets_columns():
    queue_df = _make_queue(4)
    answers_df = _make_answers(["Q001", "Q002", "Q003", "Q004"], ["B", "C", "B", "C"])
    result = validate_answers(queue_df, answers_df)
    manual_df = build_manual_triplets(queue_df, result.valid_rows)
    for col in MANUAL_TRIPLET_COLUMNS:
        assert col in manual_df.columns, f"missing column: {col}"


def test_build_manual_triplets_no_skip():
    queue_df = _make_queue(4)
    answers_df = _make_answers(["Q001", "Q002", "Q003", "Q004"], ["B", "skip", "C", "skip"])
    result = validate_answers(queue_df, answers_df)
    manual_df = build_manual_triplets(queue_df, result.valid_rows)
    assert "skip" not in manual_df["answer"].tolist()
    assert len(manual_df) == 2


def test_build_skip_log_contains_only_skip():
    queue_df = _make_queue(4)
    answers_df = _make_answers(["Q001", "Q002", "Q003", "Q004"], ["B", "skip", "C", "skip"])
    result = validate_answers(queue_df, answers_df)
    skip_df = build_skip_log(queue_df, result.skip_rows)
    assert all(skip_df["answer"] == "skip")
    assert len(skip_df) == 2


def test_build_manual_triplets_empty():
    queue_df = _make_queue(3)
    answers_df = _make_answers(["Q001", "Q002", "Q003"], ["skip", "skip", "skip"])
    result = validate_answers(queue_df, answers_df)
    manual_df = build_manual_triplets(queue_df, result.valid_rows)
    assert manual_df.empty


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def test_build_summary_aggregate_only():
    queue_df = _make_queue(5)
    answers_df = _make_answers(["Q001", "Q002", "Q003", "Q004", "Q005"], ["B", "C", "skip", "B", "C"])
    result = validate_answers(queue_df, answers_df)
    manual_df = build_manual_triplets(queue_df, result.valid_rows)
    skip_df = build_skip_log(queue_df, result.skip_rows)
    text = build_summary(queue_df, manual_df, skip_df, result)

    # must not contain private metadata tokens
    lower = text.lower()
    for forbidden in ["file_path", "file_name", "folder_path", "artist", "title", "album", "label"]:
        assert forbidden not in lower, f"forbidden token in summary: {forbidden}"

    # must not contain track_ids from fixture
    for i in range(1, 6):
        assert f"tid_a{i}" not in text
        assert f"tid_b{i}" not in text
        assert f"tid_c{i}" not in text

    # must contain aggregate counts
    assert "5" in text  # n_queue
    assert "Questions in queue" in text


def test_build_summary_decision_rule_below_20():
    queue_df = _make_queue(5)
    answers_df = _make_answers(["Q001", "Q002"], ["B", "C"])
    result = validate_answers(queue_df, answers_df)
    manual_df = build_manual_triplets(queue_df, result.valid_rows)
    skip_df = build_skip_log(queue_df, result.skip_rows)
    text = build_summary(queue_df, manual_df, skip_df, result)
    assert "NO" in text or "diagnostic" in text.lower()


def test_build_summary_decision_rule_at_20():
    # 20 answers, all B
    n = 20
    rows = [{"question_id": f"Q{i:03d}", "anchor_track_id": f"a{i}",
              "candidate_b_track_id": f"b{i}", "candidate_c_track_id": f"c{i}",
              "selection_source": "mert_full_knn"} for i in range(1, n + 1)]
    queue_df = pd.DataFrame(rows)
    answers_df = _make_answers([f"Q{i:03d}" for i in range(1, n + 1)], ["B"] * n)
    result = validate_answers(queue_df, answers_df)
    manual_df = build_manual_triplets(queue_df, result.valid_rows)
    skip_df = build_skip_log(queue_df, result.skip_rows)
    text = build_summary(queue_df, manual_df, skip_df, result)
    assert "YES" in text


# ---------------------------------------------------------------------------
# D4.3a active-answer filtering and merge tests
# ---------------------------------------------------------------------------


def _manual_rows(qids):
    """Build a minimal manual/skip-style DataFrame for the given question_ids."""
    return pd.DataFrame(
        {
            "question_id": list(qids),
            "anchor_track_id": [f"tid_a{q}" for q in qids],
            "candidate_b_track_id": [f"tid_b{q}" for q in qids],
            "candidate_c_track_id": [f"tid_c{q}" for q in qids],
            "selection_source": ["mert_full_knn"] * len(qids),
            "answer": ["B"] * len(qids),
            "confidence": [""] * len(qids),
            "notes": [""] * len(qids),
        }
    )


def test_collect_answered_ids_unions_csv_question_ids(tmp_path):
    manual = tmp_path / "manual_triplets.csv"
    skip = tmp_path / "triplet_skip_log.csv"
    _manual_rows(["Q001", "Q002"]).to_csv(manual, index=False)
    _manual_rows(["Q003"]).to_csv(skip, index=False)
    ids = collect_answered_ids([manual, skip])
    assert ids == {"Q001", "Q002", "Q003"}


def test_collect_answered_ids_skips_missing_files(tmp_path):
    manual = tmp_path / "manual_triplets.csv"
    _manual_rows(["Q001"]).to_csv(manual, index=False)
    ids = collect_answered_ids([manual, tmp_path / "absent.csv"])
    assert ids == {"Q001"}


def test_filter_unanswered_excludes_answered_ids():
    queue_df = _make_queue(10)  # Q001..Q010
    answered = {"Q001", "Q002", "Q003", "Q004", "Q005"}
    out = filter_unanswered(queue_df, answered)
    assert len(out) == 5
    assert set(out["question_id"]) == {"Q006", "Q007", "Q008", "Q009", "Q010"}


def test_filter_unanswered_keeps_all_when_none_answered():
    queue_df = _make_queue(6)
    out = filter_unanswered(queue_df, set())
    assert len(out) == 6


def test_active_template_covers_only_unanswered(tmp_path):
    # queue of 12; first 4 already answered -> active template has 8 rows
    queue_df = _make_queue(12)
    answered = {"Q001", "Q002", "Q003", "Q004"}
    active_queue = filter_unanswered(queue_df, answered)
    out = tmp_path / "active_template.csv"
    n = create_answer_template(active_queue, out)
    assert n == 8
    df = pd.read_csv(out, dtype=str).fillna("")
    assert list(df["question_id"]) == [f"Q{i:03d}" for i in range(5, 13)]
    assert all(df["answer"] == "")


def test_create_template_without_filter_still_full():
    # original full-template behavior is unchanged when no filter is applied
    queue_df = _make_queue(7)
    out = filter_unanswered(queue_df, set())
    assert len(out) == len(queue_df)


def test_merge_answer_rows_preserves_existing_and_appends():
    existing = _manual_rows(["Q001", "Q002"])
    new = _manual_rows(["Q041", "Q042"])
    merged = merge_answer_rows(existing, new)
    assert list(merged["question_id"]) == ["Q001", "Q002", "Q041", "Q042"]


def test_merge_answer_rows_rejects_question_id_collision():
    existing = _manual_rows(["Q001", "Q002"])
    new = _manual_rows(["Q002", "Q041"])  # Q002 collides
    with pytest.raises(ValueError):
        merge_answer_rows(existing, new)


def test_merge_answer_rows_with_empty_existing():
    existing = pd.DataFrame(columns=list(MANUAL_TRIPLET_COLUMNS))
    new = _manual_rows(["Q041", "Q042"])
    merged = merge_answer_rows(existing, new)
    assert list(merged["question_id"]) == ["Q041", "Q042"]
