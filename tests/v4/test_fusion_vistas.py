"""
PURPOSE: Tests de src/v4/evaluation/fusion_vistas.py (unidad 1 de PROMPT_FUSION_Y_CONCEPTOS.md: P11, P12, P13, E14) con
         datos sintéticos: vistas y tempo descontado, suma de z igual al producto de los vectores concatenados, acierto
         del tema siguiente, factores con las fórmulas y filtros del recomendador, pares de la ecuación y validación
         dejando cada grupo fuera.
CHANGELOG:
  - 2026-11-09: Creación.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common import harmonic  # noqa: E402
from src.v4.evaluation import fusion_vistas as fv  # noqa: E402
from src.v4.evaluation import zona_sets as zs  # noqa: E402
from tools.dj_sets.piloto import unit  # noqa: E402
from tools.traktor_live import recomendador as rc  # noqa: E402


def _crudos(n=120, seed=0):
    rng = np.random.default_rng(seed)
    bpm = rng.uniform(110, 140, n)
    estilos = np.abs(rng.normal(size=(n, 20))) + 0.0
    estilos[:, 0] += (bpm - 110) / 10          # el estilo 0 lleva el tempo
    return {"clap": rng.normal(size=(n, 16)) + 3.0, "maest": rng.normal(size=(n, 12)), "estilos": estilos}, bpm


def test_vista_centrada_aplica_la_media_de_la_biblioteca_a_otros_temas():
    crudos, bpm = _crudos()
    vistas = fv.construir_vistas(crudos, bpm)
    E = fv.espacio(vistas, crudos)
    assert np.allclose(np.linalg.norm(E["clap_c"], axis=1), 1.0)
    assert np.allclose(E["clap"], unit(crudos["clap"]))
    otro = np.random.default_rng(1).normal(size=(3, 16))
    esperado = unit(unit(otro) - unit(crudos["clap"]).mean(axis=0))
    assert np.allclose(vistas["clap_c"](otro), esperado)


def test_estilos_sin_tempo_no_predicen_el_bpm_linealmente():
    crudos, bpm = _crudos(n=400)
    vistas = fv.construir_vistas(crudos, bpm)
    E = fv.espacio(vistas, crudos)
    Q = vistas["estilos_st"].Q
    assert Q.shape == (20, 2) and np.allclose(Q.T @ Q, np.eye(2))
    # lo que queda es ortogonal a las direcciones quitadas (antes de normalizar)
    X = unit(crudos["estilos"]) - vistas["estilos_st"].mu
    resto = X - (X @ Q) @ Q.T
    assert np.allclose(resto @ Q, 0.0, atol=1e-10)
    # y la correlación con el tempo baja
    c_antes = abs(np.corrcoef(E["estilos_c"][:, 0], bpm)[0, 1])
    rho_antes = fv.spearman_tempo(E["estilos_c"], bpm, n=20000)
    rho_despues = fv.spearman_tempo(E["estilos_st"], bpm, n=20000)
    assert c_antes > 0.5
    assert abs(rho_despues) < abs(rho_antes)


def test_escala_z_es_la_media_y_el_desvio_fuera_de_la_diagonal():
    rng = np.random.default_rng(2)
    V = unit(rng.normal(size=(30, 5)))
    m, s = fv.escala_z({"v": V})["v"]
    S = V @ V.T
    off = ~np.eye(30, dtype=bool)
    assert np.isclose(m, S[off].mean()) and np.isclose(s, S[off].std())


def test_concatenar_ordena_igual_que_la_suma_de_z():
    crudos, bpm = _crudos()
    E = fv.espacio(fv.construir_vistas(crudos, bpm), crudos)
    esc = fv.escala_z(E)
    pesos = {"clap_c": 1.0, "maest_c": 0.5, "estilos": 2.0}
    C = fv.concatenar(E, pesos, esc)
    i, j = np.arange(0, 60), np.arange(60, 120)
    z = fv.suma_z(E, i, j, pesos, esc)
    const = sum(w * esc[k][0] / esc[k][1] for k, w in pesos.items())
    assert np.allclose(np.sum(C[i] * C[j], axis=1), z + const)


def test_percentil_filas_saca_la_diagonal():
    S = np.array([[9.0, 1.0, 2.0], [1.0, 9.0, 3.0], [2.0, 3.0, 9.0]])
    P = fv.percentil_filas(S)
    assert P[0, 0] == 0.0 and P[0, 2] == 1.0 and P[0, 1] == 0.5


def test_acierto_siguiente_saca_lo_tocado_y_filtra_por_bpm():
    M = np.array([[0, 5, 4, 3, 2],
                  [5, 0, 1, 9, 8],
                  [4, 1, 0, 2, 3],
                  [3, 9, 2, 0, 1],
                  [2, 8, 3, 1, 0]], dtype=float)
    sets = {"s": [0, 1, 4]}
    h = fv.acierto_siguiente(sets, M, top=1)
    # 0 -> 1: 1 es el más parecido a 0 (acierto); 1 -> 4: 3 gana a 4 (sin 0 y 1, ya tocados)
    assert h.tolist() == [[0, 1, 4], [0, 0, 3]]
    bpm = np.array([120.0, 121.0, 150.0, 150.0, 122.0])
    h = fv.acierto_siguiente(sets, M, bpm, tol=4.0, top=1)
    assert h.tolist() == [[0, 1, 4 - 2], [0, 1, 1]]      # sin los de 150 BPM, 4 queda solo


def test_comparar_cuenta_discordantes():
    base = np.array([[0, 1, 9], [0, 0, 9], [1, 0, 9], [1, 0, 9]])
    h = np.array([[0, 1, 9], [0, 1, 9], [1, 1, 9], [1, 0, 9]])
    r = fv.comparar(h, base, n=50)
    assert (r["solo_variante"], r["solo_base"]) == (2, 0)
    assert r["acierto"] == 0.75 and r["diferencia"] == 0.5
    with pytest.raises(ValueError):
        fv.comparar(h[:3], base)


def _tablas(E):
    return fv.Tablas(fv.escala_z(E), fv.tabla_claves(), np.array([[1.0, 0.2], [0.2, 1.0]]), [("Techno", None), ("House", None)])


def _lado(n=8, seed=0):
    rng = np.random.default_rng(seed)
    E = {"clap": unit(rng.normal(size=(n, 6))), "clap_c": unit(rng.normal(size=(n, 6))),
         "maest_c": unit(rng.normal(size=(n, 6))), "estilos": unit(rng.normal(size=(n, 6)))}
    bpm = np.array([124.0, 125.0, 150.0, np.nan, 124.0, 126.0, 123.0, 120.0])[:n]
    clave = np.array([fv.indice_clave(k) for k in ["8A", "8A", "8A", "8A", "3B", "?", "9A", "8B"]])[:n]
    genero = np.array([0, 0, 1, -1, 1, 0, 0, 1])[:n]
    return fv.Lado(E, bpm, clave, genero, [f"a{i}" for i in range(n)])


def test_tabla_claves_es_best_key_shift():
    T = fv.tabla_claves()
    assert T.shape == (24, 24)
    assert T[fv.indice_clave("8A"), fv.indice_clave("8A")] == harmonic.best_key_shift("8A", "8A")[0] == 1.0
    assert T[fv.indice_clave("8A"), fv.indice_clave("3B")] == harmonic.best_key_shift("8A", "3B")[0]
    assert fv.indice_clave("?") == -1 and fv.indice_clave(None) == -1


def test_factores_usan_las_formulas_y_los_filtros_del_recomendador():
    lado = _lado()
    t = _tablas(lado.E)
    fx, ok = fv.factores(lado, 0, lado, t)
    # bpm: 1 − |Δ|/20, nan sin BPM
    assert np.isclose(fx["bpm"][1], 1 - 1 / 20) and np.isnan(fx["bpm"][3])
    # clave: la tabla; nan para el candidato sin tonalidad
    assert fx["clave"][1] == 1.0 and np.isnan(fx["clave"][5])
    # género: la tabla; nan sin género
    assert fx["genero"][1] == 1.0 and fx["genero"][2] == 0.2 and np.isnan(fx["genero"][3])
    # vistas en z y CLAP crudo como coseno
    assert np.isclose(fx["clap"][4], lado.E["clap"][0] @ lado.E["clap"][4])
    m, s = t.escala["clap_c"]
    assert np.isclose(fx["clap_c"][4], (lado.E["clap_c"][0] @ lado.E["clap_c"][4] - m) / s)
    # filtros: 150 BPM fuera, sin BPM fuera, sin tonalidad fuera, 3B incompatible con 8A fuera
    esperado = [harmonic.best_key_shift("8A", k)[0] >= rc.KEY_MIN for k in ["8A", "8A", "8A", "8A", "3B", "?", "9A", "8B"]]
    esperado = np.array(esperado) & np.array([True, True, False, False, True, False, True, True])
    assert ok.tolist() == esperado.tolist()


def test_factores_sin_bpm_ni_tonalidad_del_principal_no_filtran():
    lado = _lado()
    t = _tablas(lado.E)
    fx, ok = fv.factores(lado, 3, lado, t)          # el tema 3 no tiene BPM
    assert np.isnan(fx["bpm"]).all()
    assert ok[2]                                       # 150 BPM pasa: el principal no tiene BPM
    lado.clave[3] = -1
    fx, ok = fv.factores(lado, 3, lado, t)
    assert np.isnan(fx["clave"]).all() and ok.all()


def test_pares_ecuacion_respetan_filtros_artista_y_pesos():
    lado = _lado()
    t = _tablas(lado.E)
    f = zs.Fuente("x", "biblioteca", lado.E["clap"], lado.bpm, lado.artista, {"s1": [0, 1, 2], "s2": [4, 6]}, {"s1": "DJ Uno", "s2": "DJ Dos"})
    P = fv.pares_ecuacion("x", f, lado, lado, t, np.random.default_rng(0), mismo_espacio=True, k_neg=5)
    # s1: (0,1) pasa; (0,2) y (1,2) no (150 BPM); s2: (4,6) no pasa (3B y 9A no son compatibles)
    assert P.resumen["pares_que_pasan_los_filtros"] == 1 and P.resumen["pares_fuera_por_filtros"] == 3
    assert P.y.sum() == 1 and len(P.y) == 6
    neg = P.y == 0
    assert np.isclose(P.w[P.y == 1].sum(), 1.0) and np.isclose(P.w[neg].sum(), 1.0)
    # los negativos pasan los filtros respecto del principal y no son del set
    fx, ok = fv.factores(lado, 0, lado, t)
    assert set(np.round(P.X["bpm"][neg], 6)) <= set(np.round(fx["bpm"][ok & ~np.isin(np.arange(8), [0, 1, 2])], 6))
    assert set(P.grupo) == {"dj uno"}
    assert ((P.consenso >= 0) & (P.consenso <= 3)).all()


def test_validar_por_grupo_no_ve_el_grupo_que_predice():
    rng = np.random.default_rng(0)
    n = 400
    y = np.r_[np.ones(n // 2), np.zeros(n // 2)].astype(int)
    X = {c: rng.normal(size=n) + (y if c == "clap_c" else 0) for c in fv.FACTORES}
    X["clap"] = X["clap_c"]
    grupo = np.asarray(["a", "b", "c", "d"] * (n // 4))
    P = fv.Pares(X, y, np.ones(n), grupo, np.full(n, "x"), np.zeros(n, dtype=int), {})
    p = fv.validar_por_grupo(P, fv.FACTORES)
    assert np.isfinite(p).all()
    assert fv.auc_pares(P, p) > 0.7
    # un grupo que solo tiene una clase en el entrenamiento no se predice
    P2 = fv.Pares(X, y, np.ones(n), np.where(y == 1, "solo_pos", "solo_neg"), np.full(n, "x"), np.zeros(n, dtype=int), {})
    assert np.isnan(fv.validar_por_grupo(P2, fv.FACTORES)).all()


def test_consenso_por_nivel_reparte_el_peso():
    P = fv.Pares({}, np.array([1, 1, 0, 0]), np.array([0.5, 0.5, 0.5, 0.5]), np.array(["a"] * 4), np.array(["x"] * 4),
                 np.array([3, 0, 0, 0]), {})
    r = fv.consenso_por_nivel(P)
    assert r["3"]["mismo_set"] == 0.5 and r["3"]["azar"] == 0.0 and r["3"]["fraccion_mismo_set"] == 1.0
    assert r["0"]["fraccion_mismo_set"] == round(0.5 / 1.5, 3)


def test_auc_entre_sets_compara_dentro_de_la_misma_fuente():
    # dos sets, cada uno agrupado en su propia dirección: los pares del mismo set están más cerca que los cruzados
    V = unit(np.array([[1.0, 0.1], [1.0, 0.2], [1.0, 0.0], [0.1, 1.0], [0.2, 1.0], [0.0, 1.0]]))
    f = zs.Fuente("x", "biblioteca", V, np.full(6, 124.0), ["a", "b", "c", "d", "e", "f"], {"s1": [0, 1, 2], "s2": [3, 4, 5]}, {})
    assert fv.auc_entre_sets(f, V) == 1.0
    assert fv.auc_entre_sets(f, V, consecutivos=True) == 1.0
    # el mismo artista no cuenta en ningún lado
    f2 = zs.Fuente("x", "biblioteca", V, np.full(6, 124.0), ["a"] * 6, {"s1": [0, 1, 2], "s2": [3, 4, 5]}, {})
    assert fv.auc_entre_sets(f2, V) is None
