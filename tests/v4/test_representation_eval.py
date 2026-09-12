"""
PURPOSE: Tests unitarios de representation_eval con datos sintéticos: accuracy de tripletas
         (aciertos, fallos, empates, uids ausentes), bootstrap pareado y coherencia de carpetas.
CHANGELOG:
  - 2026-09-12: Creación inicial.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.evaluation.representation_eval import (  # noqa: E402
    Representation, evaluate_triplets, folder_coherence, paired_bootstrap, triplet_outcomes,
)


def _rep(name, rows):
    uids = list(rows)
    return Representation(name, np.array([rows[u] for u in uids], dtype=float), uids)


def test_triplet_outcomes():
    rep = _rep("r", {"a": [1, 0], "b": [1, 0.1], "c": [0, 1], "d": [1, 0.1]})
    trip = pd.DataFrame({
        "anchor_uid": ["a", "a", "a", "a"],
        "b_uid": ["b", "b", "b", "zz"],
        "c_uid": ["c", "c", "d", "c"],
        "answer": ["B", "C", "B", "B"],
    })
    o = triplet_outcomes(rep, trip)
    assert o.tolist()[:3] == [1.0, 0.0, 0.5], o.tolist()
    assert np.isnan(o.iloc[3])
    print("  OK: aciertos, fallos, empates y uids ausentes")


def test_evaluate_and_paired():
    rep_good = _rep("good", {"a": [1, 0], "b": [1, 0.05], "c": [0, 1]})
    rep_bad = _rep("bpm", {"a": [1, 0], "b": [0, 1], "c": [1, 0.05]})
    trip = pd.DataFrame({
        "anchor_uid": ["a"] * 6, "b_uid": ["b"] * 6, "c_uid": ["c"] * 6,
        "answer": ["B"] * 6, "source": ["uniform"] * 3 + ["dj_branch"] * 3,
    })
    table = evaluate_triplets([rep_bad, rep_good], trip, reference="bpm")
    row = table[(table.group == "all") & (table.representation == "good")].iloc[0]
    assert row["accuracy"] == 1.0 and row["n"] == 6
    assert row["vs_bpm_diff"] == 1.0 and row["vs_bpm_p_gain"] == 1.0
    assert set(table.group) == {"all", "uniform", "dj_branch"}
    pb = paired_bootstrap(np.array([1, 1, 0, 1.0]), np.array([1, 1, 0, 1.0]))
    assert pb["diff"] == 0.0 and pb["p_gain"] == 0.0
    print("  OK: tabla por grupo y bootstrap pareado")


def test_folder_coherence():
    rng = np.random.default_rng(1)
    rows, labels = {}, {}
    for c, centre in enumerate(([10, 0, 0], [0, 10, 0], [0, 0, 10])):
        for i in range(8):
            u = f"c{c}_{i}"
            rows[u] = np.array(centre) + rng.normal(0, 0.1, 3)
            labels[u] = f"F{c}"
    res = folder_coherence(_rep("blobs", rows), labels, k=5)
    assert res["n"] == 24 and res["knn_purity"] == 1.0 and res["map"] == 1.0, res
    res2 = folder_coherence(_rep("small", {"x": [1, 0]}), {"x": "F"}, k=5)
    assert np.isnan(res2["knn_purity"])
    print("  OK: pureza kNN y MAP en blobs separados")


if __name__ == "__main__":
    for fn in (test_triplet_outcomes, test_evaluate_and_paired, test_folder_coherence):
        print(f"[TEST] {fn.__name__}")
        fn()
    print("[RESULT] PASS")
