"""
PURPOSE: Tests de las funciones puras de src/v4/evaluation/zona_sets.py (E9 y E10 del plan de playlists en vivo) y
         del tramo que mide tools/dj_sets/tempo_sets.py.
CHANGELOG:
  - 2026-11-08: Creación.
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.evaluation import zona_sets as zs  # noqa: E402
from tools.dj_sets.piloto import unit  # noqa: E402
from tools.dj_sets.tempo_sets import excerpt  # noqa: E402


def test_tema_de_d3_quita_tiempo_y_sello_y_conserva_la_version():
    assert zs.tema_de_d3("[05] Code 61 - Drop The Deal [Limited Club Edition - LCE 003]") == ("Code 61", "Drop The Deal")
    assert zs.tema_de_d3("[24] Off - Electrica Salsa (12\" Version) [ZYX - 5548]") == ("Off", "Electrica Salsa (12\" Version)")
    assert zs.tema_de_d3("[29] sin separador") is None


def test_estricto_distingue_versiones_y_quita_acentos():
    assert zs.estricto("Discótico Pléxico (Maceo Plex Remix)") == "discotico plexico maceo plex remix"
    assert zs.estricto("Domino") != zs.estricto("Domino (Dub)")
    assert zs.primer_artista("Mark Knight, Funkagenda") == "Mark Knight"
    assert zs.primer_artista("Catz 'n Dogz & Joseph Ashworth") == "Catz 'n Dogz"


def test_anio_de():
    assert zs.anio_de("2015-06-07 - Laurent Garnier") == 2015
    assert zs.anio_de("sin fecha") is None


def test_cuantil_pesado_respeta_los_pesos():
    x = np.array([0.0, 1.0])
    assert zs.cuantil_pesado(x, np.array([1.0, 1.0]), 50) == 0.5
    assert zs.cuantil_pesado(x, np.array([9.0, 1.0]), 50) < 0.5
    assert zs.cuantil_pesado(np.array([3.0, 1.0, 2.0]), np.ones(3), 50) == 2.0


def test_pares_del_set_cada_set_pesa_uno_y_saca_el_mismo_artista():
    sets = {"grande": [0, 1, 2, 3], "chico": [4, 5], "un_artista": [6, 7]}
    artista = ["a", "b", "c", "d", "e", "f", "g", "g"]
    I, J, K, W, nombres = zs.pares_del_set(sets, artista)
    assert nombres == ["grande", "chico"]
    assert len(I) == 7
    assert np.isclose(W[K == 0].sum(), 1.0) and np.isclose(W[K == 1].sum(), 1.0)
    # un artista desconocido no cuenta como el mismo artista
    assert len(zs.pares_del_set({"s": [0, 1]}, ["", ""])[0]) == 1


def test_azar_mismo_bpm_respeta_tolerancia_grupo_y_reparte_el_peso():
    rng = np.random.default_rng(0)
    bpm_pool = np.array([120.0, 121.0, 122.0, 140.0, 141.0, np.nan])
    grupo = np.array(["x", "x", "y", "y", "z", "z"])
    a, b, p = zs.azar_mismo_bpm(np.array([120.0, np.nan]), np.array([121.0, 130.0]), np.array([1.0, 1.0]), bpm_pool, rng,
                                grupo_pool=grupo, tol=2.0, por_par=50)
    assert len(a) > 0
    assert np.all(np.abs(bpm_pool[a] - 120.0) <= 2.0) and np.all(np.abs(bpm_pool[b] - 121.0) <= 2.0)
    assert np.all(a != b) and np.all(grupo[a] != grupo[b])
    assert np.isclose(p.sum(), 1.0)                 # el par con BPM desconocido no aporta


def test_auc_mas_cerca_es_uno_si_los_reales_estan_siempre_mas_cerca():
    assert zs.auc_mas_cerca(np.array([0.1, 0.2]), np.ones(2), np.array([0.3, 0.4]), np.ones(2)) == 1.0
    assert zs.auc_mas_cerca(np.zeros(0), np.zeros(0), np.array([0.3]), np.ones(1)) is None


def test_cobertura_y_fraccion_dentro():
    assert zs.cobertura(np.array([0.1, 0.2, 0.3]), np.array([1.0, 1.0, 2.0]), 0.2) == 0.5
    D = np.array([[0.1, 0.1, 0.3, 0.3], [0.1, 0.1, 0.1, np.nan], [0.5, 0.5, 0.5, 0.5]])
    f = zs.fraccion_dentro(D, 0.2)
    assert f["media"] == round((0.5 + 1.0 + 0.0) / 3, 4)
    assert f["listas_enteras_dentro"] == round(1 / 3, 4)
    assert f["listas_con_menos_de_la_mitad"] == round(1 / 3, 4)


def test_fraccion_por_grupo_ordena_y_saca_grupos_chicos():
    D = np.array([[0.1, 0.1], [0.1, 0.1], [0.3, 0.3], [0.3, 0.1], [0.1, 0.1]])
    r = zs.fraccion_por_grupo(D, ["a", "a", "b", "b", "c"], 0.2, min_n=2)
    assert list(r) == ["b", "a"] and r["a"]["listas_enteras_dentro"] == 1.0 and r["b"]["media"] == 0.25


def test_pureza_por_bpm_no_prefiere_el_tempo_estimado():
    """Con el BPM redondeado, 125,9 (estimado) y 126 (tag) empatan: el orden no favorece a la misma fuente."""
    rng = np.random.default_rng(5)
    V = unit(rng.normal(size=(60, 16)))
    bpm_f = np.array([125.9] * 10)
    f = zs.Fuente("x", "biblioteca", V[:10], bpm_f, [str(i) for i in range(10)], {"s": list(range(10))})
    bpm_pool = np.r_[bpm_f, np.full(50, 126.0)]
    r = zs.pureza(f, V, bpm_pool, rng, top=5)
    # sin redondear, los 5 primeros serían siempre del set (pureza 1); empatados, sale cerca de lo esperado al azar
    assert r["bpm"]["media"] < 0.5


def fuente_de_prueba(rng):
    """Dos sets de 12 temas muy juntos entre sí y lejos del otro; 40 temas sueltos en otras direcciones."""
    centros = unit(rng.normal(size=(2, 32)))
    V = [unit(c + 0.02 * rng.normal(size=(12, 32))) for c in centros]
    resto = unit(rng.normal(size=(40, 32)))
    M = np.vstack(V + [resto])
    bpm = np.r_[np.full(24, 124.0), np.full(40, 124.0)]
    sets = {"A": list(range(12)), "B": list(range(12, 24))}
    return zs.Fuente("prueba", "biblioteca", M[:24], bpm[:24], [str(i) for i in range(24)], sets), M, bpm


def test_pureza_cuenta_solo_temas_posteriores_y_saca_lo_tocado():
    rng = np.random.default_rng(1)
    f, M, bpm = fuente_de_prueba(rng)
    r = zs.pureza(f, M, bpm, rng, top=5)
    assert r["sets"] == 2 and r["consultas"] == 22
    # con 11, 10, ... temas posteriores los 5 primeros son del set hasta que quedan menos de 5
    assert r["clap"]["media"] > 0.7
    assert r["clap"]["media"] > r["azar"]["media"] > 0.0
    # la recuperación llega a 1 cuando quedan pocos temas por sonar
    assert r["clap_rec"]["media"] > r["azar_rec"]["media"]


def test_pureza_con_pool_extra_informa_cuanto_sale_de_afuera():
    rng = np.random.default_rng(2)
    f, M, bpm = fuente_de_prueba(rng)
    extra = np.r_[np.zeros(24, bool), np.ones(40, bool)]
    r_excl = zs.pureza(f, M, bpm, rng, excluir=extra, top=5)
    r = zs.pureza(f, M, bpm, rng, top=5)
    assert r["de_la_biblioteca_en_los_30"] < 0.5
    assert "de_la_biblioteca_en_los_30" in r_excl and r_excl["de_la_biblioteca_en_los_30"] == 0.0


def test_e9_fuente_mide_lo_que_junta_el_set_frente_al_azar():
    rng = np.random.default_rng(3)
    f, M, bpm = fuente_de_prueba(rng)
    out, d, w = zs.e9_fuente(f, M, bpm, rng)
    assert out["pares"] == 2 * 66 and len(d) == out["pares"] and np.isclose(w.sum(), 2.0)
    assert out["mismo_set"]["p50"] < out["azar_mismo_bpm"]["p50"]
    assert out["auc_frente_a_azar_mismo_bpm"] > 0.5
    assert out["ic95_por_sets"]["p50"][0] <= out["mismo_set"]["p50"] <= out["ic95_por_sets"]["p50"][1]


def test_e9_bloques_solo_cuenta_pares_separados():
    rng = np.random.default_rng(4)
    c = unit(rng.normal(size=(2, 16)))
    B1 = unit(c[0] + 0.05 * rng.normal(size=(60, 16)))
    B2 = unit(c[1] + 0.05 * rng.normal(size=(60, 16)))
    t = np.arange(60) * 30.0
    r = zs.e9_bloques([("a", "g1", B1, t), ("b", "g2", B2, t)], rng, max_pares=200)
    assert r["discos"] == 2
    assert r["mismo_disco"]["p50"] < r["otro_disco_otra_compilacion"]["p50"]
    # un disco demasiado corto no da pares separados, pero no rompe
    assert zs.e9_bloques([("a", "g1", B1[:3], t[:3]), ("b", "g2", B2[:3], t[:3])], rng, max_pares=10)["mismo_disco"] is None


def test_excerpt_centra_y_evita_los_cambios():
    assert excerpt(0.0, 300.0) == (120.0, 60.0)
    assert excerpt(1000.0, 1100.0) == (1025.0, 50.0)
    assert excerpt(5.0, 5.0)[1] == 0.0
