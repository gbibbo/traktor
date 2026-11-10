"""
PURPOSE: Tests de src/v4/evaluation/juez_d5.py (E16, plan sección 12.9) y de zona_sets.fuente_d5: cruce de un set de D5
         con la biblioteca, sets repetidos con D3, fuentes juntas y la tabla de tonalidad contra el azar.
CHANGELOG:
  - 2026-11-10: Creación.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.evaluation import juez_d5 as jd  # noqa: E402
from src.v4.evaluation import zona_sets as zs  # noqa: E402


def _bib():
    nombres = [("artista uno", "tema a (original mix)"), ("artista dos", "tema b"), ("artista tres", "tema c"), ("artista cuatro", "tema d")]
    V = np.eye(4)
    por_nombre = {(zs.estricto(zs.primer_artista(a)), zs.estricto(t)): i for i, (a, t) in enumerate(nombres)}
    return zs.Biblioteca([f"u{i}" for i in range(4)], V, np.full(4, 125.0), [zs.estricto(a) for a, _ in nombres],
                         {f"u{i}": i for i in range(4)}, por_nombre)


def _set(root: Path, ident: str, titulo: str, nombres):
    d = root / "1001tracklists" / "sets"
    d.mkdir(parents=True, exist_ok=True)
    temas = [{"nombre": n, "visible": n, "junto_al_anterior": j} for n, j in nombres]
    (d / f"{ident}.json").write_text(json.dumps({"id": ident, "titulo": titulo, "temas": temas}), encoding="utf-8")


def test_fuente_d5_cruza_por_nombre_y_saca_lo_tocado_encima_y_lo_de_d3(tmp_path):
    bib = _bib()
    _set(tmp_path, "s1", "DJ Uno @ Club 2025-01-01", [("Artista Uno - Tema A (Original Mix)", False), ("Otro - Nada", False),
                                                      ("Artista Dos - Tema B", True), ("Artista Tres - Tema C", False)])
    _set(tmp_path, "s2", "DJ Dos & DJ Tres @ Club 2025-01-02", [("Artista Uno - Tema A (Original Mix)", False), ("Artista Dos - Tema B", False)])
    _set(tmp_path, "s3", "DJ Cuatro @ Club", [("Artista Cuatro - Tema D", False)])
    f = zs.fuente_d5(tmp_path, bib)
    assert f.sets == {"d5:s1": [0, 2], "d5:s2": [0, 1]}
    assert f.dj["d5:s2"] == "DJ Dos" and f.curado["sets_con_un_solo_tema"] == 1
    d3 = zs.Fuente("d3_biblioteca", "biblioteca", bib.V, bib.bpm, bib.artista, {"m1": [0, 1, 3]}, {"m1": "DJ Dos"})
    g = zs.fuente_d5(tmp_path, bib, d3)
    assert "d5:s2" not in g.sets and g.curado["sets_que_ya_estan_en_d3"] == 1


def test_juntar_fuentes():
    bib = _bib()
    a = zs.Fuente("a", "biblioteca", bib.V, bib.bpm, bib.artista, {"x": [0, 1]}, {"x": "A"})
    b = zs.Fuente("b", "biblioteca", bib.V, bib.bpm, bib.artista, {"d5:y": [2, 3]}, {"d5:y": "B"})
    j = jd.juntar_fuentes("ab", [a, b])
    assert j.sets == {"x": [0, 1], "d5:y": [2, 3]} and j.dj == {"x": "A", "d5:y": "B"}


def test_tabla_tonalidad_cociente_contra_el_azar():
    real = ["misma", "misma", "vecina", None]
    W = np.array([0.5, 0.5, 1.0, 1.0])
    K = np.array([0, 0, 1, 1])
    azar = ["misma", "vecina", "vecina", "vecina"]
    p = np.ones(4)
    t = jd.tabla_tonalidad(real, W, K, azar, p, np.random.default_rng(0), veces=50)
    assert np.isclose(t["misma"]["veces"], 0.5 / 0.25) and np.isclose(t["vecina"]["veces"], 0.5 / 0.75, atol=1e-3)
    assert t["misma"]["pares"] == 2
