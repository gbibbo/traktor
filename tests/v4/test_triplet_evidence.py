"""
PURPOSE: Tests unitarios de src/v4/evaluation/triplet_evidence.py con datos sintéticos:
         carga y deduplicación de respuestas, resolución filename -> track_uid, y baselines
         (clave sola, BPM solo) sobre tripletas construidas para tener respuesta conocida.
CHANGELOG:
  - 2026-09-11: Creación inicial.
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.evaluation.triplet_evidence import (  # noqa: E402
    baseline_accuracy, bpm_similarity, load_answers, summarize, wilson_interval,
)


def _catalog():
    return pd.DataFrame({
        "filename": ["a.mp3", "b.mp3", "c.mp3", "d.mp3"],
        "track_uid": ["ua", "ub", "uc", "ud"],
    })


def test_load_dedup_and_resolve():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        pd.DataFrame({
            "question_id": ["Q001", "Q002"], "anchor": ["a.mp3", "b.mp3"],
            "candidate_b": ["b.mp3", "c.mp3"], "candidate_c": ["c.mp3", "d.mp3"],
            "answer": ["B", "skip"], "answered_at": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z"],
        }).to_csv(d / "first.csv", index=False)
        # Segunda sesión: re-responde Q002 más tarde, gana la última
        pd.DataFrame({
            "question_id": ["Q002"], "anchor": ["b.mp3"], "candidate_b": ["c.mp3"],
            "candidate_c": ["d.mp3"], "answer": ["c"], "answered_at": ["2026-01-02T00:00:00Z"],
        }).to_csv(d / "second.csv", index=False)
        df = load_answers(d, _catalog())
    assert len(df) == 2
    assert df.set_index("question_id").loc["Q002", "answer"] == "C", "última respuesta debe ganar y normalizarse"
    assert df.set_index("question_id").loc["Q001", "b_uid"] == "ub"
    s = summarize(df)
    assert s["n_non_skip"] == 2 and s["n_skip"] == 0
    print("  OK: load_answers dedup + resolución a track_uid")


def test_baselines_known_answer():
    # ancla ua: 12A 124 BPM. ub: 12A (misma clave) 140 BPM. uc: 6A (lejana) 125 BPM.
    feats = pd.DataFrame({
        "track_uid": ["ua", "ub", "uc"],
        "key": ["C# minor", "C# minor", "G minor"],
        "bpm": [124.0, 140.0, 125.0],
    })
    df = pd.DataFrame({
        "question_id": ["Q1"], "anchor": ["a.mp3"], "candidate_b": ["b.mp3"], "candidate_c": ["c.mp3"],
        "answer": ["B"], "anchor_uid": ["ua"], "b_uid": ["ub"], "c_uid": ["uc"],
    })
    table = baseline_accuracy(df, feats).set_index("baseline")
    assert table.loc["key_only", "accuracy"] == 1.0, "la clave elige B (misma tonalidad)"
    assert table.loc["bpm_only", "accuracy"] == 0.0, "el BPM elige C (125 vs 124)"
    assert table.loc["random", "accuracy"] == 0.5
    assert abs(bpm_similarity(124.0, 248.0) - 1.0) < 1e-9, "equivalencia de tempo 2x"
    lo, hi = wilson_interval(35, 57)
    assert 0.48 < lo < 0.55 and 0.72 < hi < 0.76, f"IC Wilson inesperado {(lo, hi)}"
    print("  OK: baselines clave/BPM con respuesta conocida")


if __name__ == "__main__":
    for t in (test_load_dedup_and_resolve, test_baselines_known_answer):
        print(f"[TEST] {t.__name__}")
        t()
    print("[RESULT] PASS")
