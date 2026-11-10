"""
PURPOSE: Tests de src/v4/evaluation/escucha_e12.py (E12, plan sección 12.9): elección de los temas principales fija,
         repartida en tramos de género y BPM.
CHANGELOG:
  - 2026-11-10: Creación.
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.evaluation import escucha_e12 as e12  # noqa: E402


def test_elegir_principales_fijo_y_uno_por_tramo():
    genero = ["House"] * 50 + ["Techno"] * 50
    bpm = np.r_[np.linspace(120, 128, 50), np.linspace(128, 140, 50)]
    cand = list(range(100))
    a = e12.elegir_principales(cand, genero, bpm, n=10)
    assert a == e12.elegir_principales(list(reversed(cand)), genero, bpm, n=10)
    assert len(set(a)) == 10
    assert sorted(x // 10 for x in a) == list(range(10))          # un tema de cada tramo de 10
