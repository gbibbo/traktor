"""
PURPOSE: Tests de src/v4/evaluation/pares_texto.py (G13 y P12a, plan sección 12.9): relaciones de tonalidad, distancia
         entre tónicas, cortes de BPM, pares y nulos con sus pesos, cocientes contra el azar y lo aprendido sin el
         pliegue del DJ.
CHANGELOG:
  - 2026-11-10: Creación.
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.evaluation import pares_texto as pt  # noqa: E402


def test_tonica_y_relaciones():
    assert pt.tonica("8B") == (0, "B") and pt.tonica("5A") == (0, "A") and pt.tonica("?") is None
    assert pt.relacion_clave("8A", "8A") == "misma"
    assert pt.relacion_clave("8A", "8B") == "relativa"
    assert pt.relacion_clave("8B", "5A") == "paralela" and pt.relacion_clave("5A", "8B") == "paralela"
    assert pt.relacion_clave("8A", "9A") == "vecina" and pt.relacion_clave("8A", "7A") == "vecina"
    assert pt.relacion_clave("8A", "9B") == "diagonal"
    assert pt.relacion_clave("8A", "3A") == "un semitono" and pt.relacion_clave("8A", "10A") == "dos semitonos"
    assert pt.relacion_clave("8A", "?") is None


def test_distancia_tonica_sin_modo():
    assert pt.distancia_tonica("8B", "5A") == 0          # Do mayor y Do menor
    assert pt.distancia_tonica("8B", "9B") == 5          # Do -> Sol: quinta = 5 semitonos hacia abajo
    assert pt.distancia_tonica("8B", "3B") == 1          # Do -> Re bemol


def test_bin_bpm():
    assert pt.bin_bpm(0.0) == "0-1" and pt.bin_bpm(4.0) == "3-5" and pt.bin_bpm(25) == "20+" and pt.bin_bpm(np.nan) is None


def _datos():
    n = 30
    t = pt.Temas([str(i) for i in range(n)], np.full(n, 125.0), ["8A"] * n, ["Techno"] * n, [f"a{i}" for i in range(n)], [""] * n)
    s = pt.Sets(["s1", "s2", "s3"], ["dj1", "dj2", "dj1"],
                [list(range(0, 12)), list(range(12, 20)) + [-1, 20], [21, 21, 22, 23, 24, 25, 26]])
    return t, s


def test_pares_pesos_por_set_y_nulos():
    t, s = _datos()
    P, c = pt.pares(t, s, np.random.default_rng(0))
    assert c["mismo_tema"] == 1 and c["consecutivos"] == 11 + 7 + 5
    for k in range(3):
        m = P.clase == "consecutivo"
        assert np.isclose(P.w[m].sum(), 3.0)
    for clase in ("mismo set", "otro set"):
        assert np.isclose(P.w[P.clase == clase].sum(), 3.0)
    # el nulo «otro set» nunca toma temas de un set del mismo DJ
    s1 = set(range(0, 12)) | {21, 22, 23, 24, 25, 26}
    m = (P.clase == "otro set") & (P.grupo == "dj1")
    assert not set(P.b[m].tolist()) & s1
    # el nulo «mismo set» queda a 5 filas o más
    m = (P.clase == "mismo set") & np.isin(P.a, list(range(12)))
    assert all(abs(b - a) >= 5 for a, b in zip(P.a[m], P.b[m]))


def test_razones_cuenta_contra_el_nulo():
    P = pt.Pares(np.zeros(6, int), np.zeros(6, int), np.ones(6), np.array(["x", "x", "y", "x", "y", "y"]),
                 np.array(["consecutivo", "consecutivo", "consecutivo", "otro set", "otro set", "otro set"]))
    r = pt.razones(["A", "A", "B", "A", "B", "B"], P, "otro set", np.random.default_rng(0), boot=20)
    assert np.isclose(r["A"]["veces"], (2 / 3) / (1 / 3)) and np.isclose(r["B"]["veces"], (1 / 3) / (2 / 3))


def test_aprendido_no_usa_el_pliegue_del_dj():
    # un DJ único con un par de géneros que nadie más tiene: sin su pliegue, no hay datos y el cociente vale 0
    grupos = np.array(["solo"] * 2 + [f"d{i}" for i in range(40)])
    clase = np.array(["consecutivo", "otro set"] + ["consecutivo", "otro set"] * 20)
    ga = ["Raro", "Techno"] + ["Techno"] * 40
    gb = ["Raro", "Techno"] + ["House", "Techno"] * 20
    P = pt.Pares(np.zeros(42, int), np.zeros(42, int), np.ones(42), grupos, clase)
    l = pt.aprendido(ga, gb, P, "otro set", dirigido=False)
    assert np.isclose(l[0], 0.0)
