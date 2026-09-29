"""
PURPOSE: Tests de las partes puras de src/v4/evaluation/vocal_eval.py: lectura de etiquetas .lab,
         etiqueta por segundo (más de la mitad con voz), proyección de ventanas a la grilla de 1 s,
         AUC, métricas binarias y umbral elegido por exactitud balanceada.
CHANGELOG:
  - 2026-09-29: Creación inicial.
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.evaluation.vocal_eval import (  # noqa: E402
    auc, best_threshold, binary_metrics, frame_labels, read_lab, windows_to_frames,
)


def test_read_lab_and_frame_labels(tmp_path):
    lab = tmp_path / "x.lab"
    lab.write_text("0.000 1.400 nosing\n1.400 3.600 sing\n3.600 5.000 nosing\n", encoding="utf-8")
    segs = read_lab(lab)
    assert segs[1] == (1.4, 3.6, "sing")
    # segundo 1: 0.6 s de voz (>0.5) -> 1; segundo 3: 0.6 s -> 1; segundo 4: 0 -> 0
    assert frame_labels(segs, 5).tolist() == [0, 1, 1, 1, 0]


def test_windows_to_frames():
    starts = np.array([0.0, 5.0])
    s = windows_to_frames(starts, 10.0, np.array([0.2, 0.8]), 16)
    assert s[0] == 0.2 and s[7] == 0.5 and s[14] == 0.8 and np.isnan(s[15])


def test_auc_and_metrics_and_threshold():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert auc(y, s) == 1.0 and auc(y, -s) == 0.0
    assert abs(auc(y, np.ones(6)) - 0.5) < 1e-9
    m = binary_metrics(y, np.array([0, 0, 1, 1, 1, 0]))
    assert m["precision"] == 2 / 3 and m["recall"] == 2 / 3 and abs(m["balanced_accuracy"] - 2 / 3) < 1e-9
    t = best_threshold(y, s)
    assert binary_metrics(y, (s >= t).astype(int))["balanced_accuracy"] == 1.0
