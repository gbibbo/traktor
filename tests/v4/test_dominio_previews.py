"""
PURPOSE: Tests de src/v4/evaluation/dominio_previews.py (E15, plan sección 12.9): ventanas del preview, submuestra fija,
         muestra congelada, puesto del tema verdadero, lectura en streaming de un .tar sin red y puntajes por pares.
CHANGELOG:
  - 2026-11-10: Creación.
"""
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.evaluation import dominio_previews as dp  # noqa: E402
from src.v4.evaluation import fusion_vistas as fv  # noqa: E402
from tools.dj_sets.piloto import unit  # noqa: E402


def _tema(bid, same_cut=True):
    return dp.Tema("u" + bid, bid, "lote", "tanda_00000", f"{bid}.mp3", {"sample_start_ms": 0, "track_length_ms": 1}, same_cut, True)


def test_segmentos_en_tramo_toma_los_que_tienen_15_s_dentro():
    # tema de 400 s, preview desde el 40 % (160 s): el tramo 35-65 % es 140-260 s
    assert dp.segmentos_en_tramo(160_000, 400_000) == [0, 1, 2]          # 160-190, 190-220, 220-250; 250-280 tiene 10 s
    # tema de 300 s: tramo 105-195 s, preview desde 120 s: 120-150, 150-180 dentro; 180-210 tiene 15 s
    assert dp.segmentos_en_tramo(120_000, 300_000) == [0, 1, 2]
    # sin ninguno con 15 s: el que más tiene
    assert dp.segmentos_en_tramo(250_000, 400_000) == [0]


def test_cortar_rellena_con_ceros():
    x = np.ones(100, dtype=np.float32)
    s = dp.cortar(x, sr=1, n_seg=4, seg_s=30)
    assert [len(a) for a in s] == [30] * 4
    assert s[3][:10].sum() == 10 and s[3][10:].sum() == 0


def test_submuestra_control_es_fija_y_solo_same_cut():
    temas = [_tema(str(i), same_cut=i % 3 != 0) for i in range(1, 60)]
    a = dp.submuestra_control(temas, n=10, semilla=0)
    assert a == dp.submuestra_control(list(reversed(temas)), n=10, semilla=0)
    assert len(a) == 10 and all(int(i) % 3 != 0 for i in a)


def test_congelar_conserva_la_primera_muestra(tmp_path):
    temas = [_tema(str(i)) for i in range(1, 50)]
    m1, c1 = dp.congelar(tmp_path, temas)
    m2, c2 = dp.congelar(tmp_path, temas + [_tema("999")])        # la bajada sumó uno: no entra
    assert [t.bp_id for t in m2] == [t.bp_id for t in m1] and c2 == c1
    assert json.loads((tmp_path / "e15" / "muestra.json").read_text())["control"] == c1


def test_puesto_cuenta_los_mas_parecidos_que_el_verdadero():
    L = unit(np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]))
    P = unit(np.array([[0.9, 0.11], [0.0, 1.0]]))
    assert dp.puesto(P, L, np.array([0, 2])).tolist() == [1, 0]     # el 1 queda más cerca que el 0 del primer preview
    r = dp.resumen_puesto(np.array([0, 0, 3, 7]))
    assert r["primero"] == 0.5 and r["entre_5"] == 0.75 and r["peor"] == 7


def test_miembros_lee_el_tar_en_streaming_y_saca_solo_lo_pedido(tmp_path):
    tar = tmp_path / "t.tar"
    with tarfile.open(tar, "w") as t:
        for nombre, datos in (("manifest.json", b"{}"), ("1.mp3", b"uno"), ("2.mp3", b"dos"), ("3.mp3", b"tres")):
            info = tarfile.TarInfo(nombre)
            info.size = len(datos)
            t.addfile(info, io.BytesIO(datos))
    dest = tmp_path / "out"
    dest.mkdir()
    cmd = [sys.executable, "-c", f"import shutil,sys; shutil.copyfileobj(open(r'{tar}','rb'), sys.stdout.buffer)"]
    hechos = dp.miembros(cmd, ["2.mp3", "9.mp3"], dest)
    assert hechos == ["2.mp3"]
    assert (dest / "2.mp3").read_bytes() == b"dos" and not (dest / "1.mp3").exists()


def test_puntajes_pares_suma_vistas_y_factores():
    rng = np.random.default_rng(0)
    E = {v: unit(rng.normal(size=(4, 5))) for v in ("clap",) + fv.VISTAS_P11}
    t = fv.Tablas(fv.escala_z({v: unit(rng.normal(size=(20, 5))) for v in E}), fv.tabla_claves(),
                  np.array([[1.0, 0.2], [0.2, 1.0]]), [("Techno", None), ("House", None)])
    lado = fv.Lado(E, np.array([124.0, 126.0, np.nan, 124.0]), np.array([fv.indice_clave(k) for k in ("8A", "9A", "?", "8A")]),
                   np.array([0, 1, 0, -1]), ["", "", "", ""])
    est = {"bpm": (0.8, 0.2), "clave": (0.7, 0.1), "genero": (0.5, 0.25)}
    p = dp.puntajes_pares(lado, t, est)
    assert len(p["p11"]) == 6
    assert np.allclose(p["p11"], sum(p[v] for v in fv.VISTAS_P11))
    # par (0, 1): bpm 1 − 2/20, clave 8A-9A, género 0,2; todos con valor conocido
    z = (0.9 - 0.8) / 0.2 + (t.clave[fv.indice_clave("8A"), fv.indice_clave("9A")] - 0.7) / 0.1 + (0.2 - 0.5) / 0.25
    assert np.isclose(p["todos_los_factores"][0], p["p11"][0] + z)
    # par (0, 2): el 2 no tiene BPM ni tonalidad: esos factores no cuentan
    assert np.isclose(p["todos_los_factores"][1], p["p11"][1] + (1.0 - 0.5) / 0.25)
