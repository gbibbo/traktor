"""
PURPOSE: Tests de src/v4/evaluation/codec_clap.py (E17 y V8, plan sección 12.9): muestra fija, alineación del retardo
         del códec, ida y vuelta por MP3, regla de E17, base de direcciones, exceso de vecinos y cambios en la lista.
CHANGELOG:
  - 2026-11-10: Creación.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.evaluation import codec_clap as cc  # noqa: E402


def test_muestra_e17_fija_alta_fuera_de_e15_y_salta_los_que_no_abren():
    uids = [f"u{i:03d}" for i in range(80)]
    kbps = np.asarray([320.0 if i % 2 else 192.0 for i in range(80)])
    excluir = {"u001", "u003"}
    a = cc.muestra_e17(uids, kbps, excluir, lambda u: u != "u005", n=10)
    assert a == cc.muestra_e17(list(reversed(uids)), kbps[::-1], excluir, lambda u: u != "u005", n=10)
    assert len(a) == 10 and not set(a) & (excluir | {"u005"})
    assert all(int(u[1:]) % 2 for u in a)


def test_retardo_encuentra_el_atraso():
    sr = 8000
    rng = np.random.default_rng(0)
    x = rng.normal(size=10 * sr).astype(np.float32)
    y = np.r_[np.zeros(1105, dtype=np.float32), x][: len(x)]
    assert cc.retardo(x, y, sr) == 1105


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="sin ffmpeg")
def test_mp3_ida_y_vuelta_alinea_y_conserva_el_largo():
    sr = 44100
    t = np.arange(10 * sr) / sr
    rng = np.random.default_rng(1)
    mono = (0.3 * np.sin(2 * np.pi * 440 * t) * (1 + np.sin(2 * np.pi * 0.7 * t)) + 0.05 * rng.normal(size=len(t))).astype(np.float32)
    x = np.column_stack([mono, 0.8 * mono])
    y = cc.mp3_ida_y_vuelta(x, sr, 192)
    assert y.shape == x.shape
    a, b = x.mean(axis=1)[sr:-sr], y.mean(axis=1)[sr:-sr]
    assert np.corrcoef(a, b)[0, 1] > 0.95


def test_veredicto_e17():
    assert cc.veredicto_e17({"mp3_96": 2.0, "menos_6db": 0.1, "mp3_128": 1, "mp3_192": 0.5}, 2.5) == "compresión"
    assert cc.veredicto_e17({"mp3_96": 0.1, "menos_6db": 2.0}, 2.5) == "nivel"
    assert cc.veredicto_e17({"mp3_96": 0.3, "menos_6db": 0.2}, 2.5) == "ninguna de las dos"
    assert cc.veredicto_e17({"mp3_96": 1.5, "menos_6db": 1.0}, 2.5) == "las dos"


def test_base_y_reserva_son_ortonormales_y_contienen_u1():
    rng = np.random.default_rng(2)
    u1 = rng.normal(size=8)
    u1 /= np.linalg.norm(u1)
    D = rng.normal(size=(30, 8)) + 3 * u1
    extra = cc.direcciones_reserva(D, u1, k=3)
    assert np.allclose(extra @ u1, 0, atol=1e-9)
    Q = cc.base(u1, extra, 3)
    assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.norm(Q.T @ u1), 1.0)


def test_exceso_vecinos_cuenta_contra_el_genero():
    # 6 temas en dos grupos de vecinos: los bajos (0, 1, 2) son vecinos entre sí; géneros mezclados
    V = np.array([[1, 0], [0.99, 0.14], [0.98, 0.2], [0, 1], [0.14, 0.99], [0.2, 0.98]], dtype=float)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    bajo = np.array([True, True, True, False, False, False])
    genero = np.array([0, 1, 0, 1, 0, 1])
    r = cc.exceso_vecinos(V, bajo, genero, k=2)
    assert r["anclas"] == 3 and r["observado"] == 1.0
    # tasa de bajos del género de cada vecino sin él: género 0 = {0, 2, 4}, género 1 = {1, 3, 5}
    assert 0 < r["esperado_por_genero"] < 1 and r["exceso"] > 0


def test_cambios_top_solo_mira_los_que_pasan_los_filtros():
    s_a = np.array([5.0, 4, 3, 2, 1, 0])
    s_b = np.array([0.0, 4, 3, 2, 1, 5])
    ok = np.array([True, True, True, True, True, False])
    assert cc.cambios_top(s_a, s_b, ok, top=2) == 1      # {0, 1} frente a {1, 2}
    assert cc.cambios_top(s_a, s_a, ok, top=30) == 0
