"""
PURPOSE: Tests de src/v4/evaluation/tablero_a1.py (A1 de la línea A) sobre un DJSets en miniatura: la cuarentena de
         los sets con DJs en dos partes, los positivos (consecutivos con filas «w/» y sin identificar), los grupos, las
         condiciones de los negativos por nivel, los pesos por set, la lectura de las tablas y el AUC pesado.
CHANGELOG:
  - 2026-11-13: Creación.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.evaluation import tablero_a1 as ta  # noqa: E402

PARTE = {"ana": "train", "beto": "test", "caro": "test", "dani": "test", "eva": "train"}


def _csv(path: Path, filas):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


def _dataset(raiz: Path) -> Path:
    rng = np.random.default_rng(1)
    tracks = []
    for i in range(60):
        tracks.append({"track_id": f"t{i:02d}", "recording_id": f"bp_{i}", "has_audio": "1", "audio_source": "beatport",
                       "bp_bpm": str(int(rng.choice([124, 125, 126, 128]))), "bp_genre": str(rng.choice(["Techno", "House"])),
                       "bp_key": "A Minor"})
    tracks[59].update(bp_bpm="", bp_genre="", audio_source="youtube", recording_id="yt_x")      # sin metadatos
    tracks[58]["recording_id"] = "bp_44"                       # misma grabación que t44: t44 queda visto por su grabación
    sets = {"s1": ("Ana", [f"t{i:02d}" for i in range(0, 10)]),
            "s2": ("Beto", ["t10", "t11", "W", "t12", "ID", "t13", "t14", "t15", "t16", "t17"]),
            "s3": ("Caro", ["t18", "t19", "t00", "t20", "t21", "t22", "t23", "t24"]),
            "s4": ("Dani", [f"t{i:02d}" for i in range(25, 45)]),
            "s6": ("Dani", [f"t{i:02d}" for i in range(45, 58)] + ["t59"]),
            "s5": ("Ana & Beto", ["t01", "t26"]),
            "s7": ("Eva", ["t58", "t40"])}
    st, split = [], []
    for s, (dj, ids) in sets.items():
        for pos, t in enumerate(ids, 1):
            st.append({"set_id": s, "position": str(pos), "with_previous": "1" if t == "W" else "0",
                       "identified": "0" if t == "ID" else "1", "track_id": "" if t in ("W", "ID") else t})
        clave = ta.nombres(dj)[0]
        split.append({"set_id": s, "dj": dj, "split_key": clave, "split": PARTE[clave]})
    _csv(raiz / "metadata" / "tracks.csv", tracks)
    _csv(raiz / "metadata" / "set_tracks.csv", st)
    _csv(raiz / "metadata" / "split_by_dj.csv", split)
    return raiz


@pytest.fixture(scope="module")
def armado(tmp_path_factory):
    raiz = _dataset(tmp_path_factory.mktemp("ds"))
    salida = tmp_path_factory.mktemp("tablero")
    r = ta.armar(raiz, salida, parte_de=lambda n: PARTE[n], k=3, verificar=False)
    return raiz, salida, r


def test_nombres_y_cuarentena():
    assert ta.nombres("Pete Tong & Sasha - Radio 1's Essential Mix") == ["pete tong", "sasha"]
    assert ta.nombres("Adam Beyer b2b Cirez D (Live)") == ["adam beyer", "cirez d"]
    filas = [{"set_id": "a", "dj": "Ana", "split": "train"}, {"set_id": "b", "dj": "Ana & Beto", "split": "train"}]
    assert ta.particion(filas, lambda n: PARTE[n]) == {"a": "train", "b": None}


def test_resumen_y_controles(armado):
    _, salida, r = armado
    assert r["c1"]["pasa"] and r["c2"]["pasa"]
    assert r["cuarentena"] == {"train": 1}
    assert {p: r["partes"][p]["sets"] for p in ("train", "test")} == {"train": 2, "test": 4}
    assert (salida / "particion.csv").exists() and (salida / "pares_test.csv.gz").exists()


def test_positivos_y_grupos(armado):
    _, salida, _ = armado
    tab = ta.leer_tablero(salida / "pares_test.csv.gz")
    pos = tab["y"] == 1
    p1_s2 = {(a, x) for a, x, f, s in zip(tab["a"][pos], tab["x"][pos], tab["familia"][pos], tab["set_id"][pos])
             if f == "P1" and s == "s2"}
    # «w/» se salta sin cortar (t11 -> t12); la fila sin identificar corta (t12 / t13 no son consecutivos)
    assert ("t11", "t12") in p1_s2 and ("t12", "t13") not in p1_s2 and ("t13", "t14") in p1_s2
    grupos = {(a, x): g for a, x, g in zip(tab["a"][pos], tab["x"][pos], tab["grupo"][pos])}
    assert grupos[("t19", "t00")] == "vistos"                        # t00 está en un set de train
    assert grupos[("t43", "t44")] == "vistos"                        # t44 comparte grabación con t58, de train
    assert grupos[("t57", "t59")] == "sin_metadatos"
    assert all(tab["nivel"][pos] == "pos")
    # s5 está en cuarentena: su par no aparece en ninguna parte
    assert not ((tab["a"] == "t01") & (tab["x"] == "t26")).any()


def test_negativos_cumplen_cada_nivel(armado):
    raiz, salida, _ = armado
    T = ta.temas(ta.leer_csv(raiz / "metadata" / "tracks.csv"))
    tab = ta.leer_tablero(salida / "pares_test.csv.gz")
    pos_de = {p: (a, x) for p, a, x, y in zip(tab["pid"], tab["a"], tab["x"], tab["y"]) if y == 1}
    sets_de = defaultdict(set)
    for f in ta.leer_csv(raiz / "metadata" / "set_tracks.csv"):
        if f["track_id"]:
            sets_de[f["track_id"]].add(f["set_id"])
    dj_set = {f["set_id"]: f["split_key"] for f in ta.leer_csv(raiz / "metadata" / "split_by_dj.csv")}
    neg = np.flatnonzero(tab["y"] == 0)
    assert len(neg)
    for i in neg:
        a, b = pos_de[tab["pid"][i]]
        c = tab["x"][i]
        assert not (sets_de[a] & sets_de[c])                         # nunca en un mismo set que el ancla
        assert tab["dj"][i] not in {dj_set[s] for s in sets_de[c]}    # otro DJ
        assert T[c].rec not in (T[a].rec, T[b].rec)
        if tab["nivel"][i] in ("N2", "N3"):
            assert abs(abs(T[c].bpm - T[a].bpm) - abs(T[b].bpm - T[a].bpm)) <= ta.TOL_BPM
        if tab["nivel"][i] == "N3":
            assert T[c].genero == T[b].genero


def test_pesos_suman_uno_por_set(armado):
    _, salida, _ = armado
    tab = ta.leer_tablero(salida / "pares_test.csv.gz")
    for n in ta.NIVELES:
        suma = defaultdict(lambda: [0.0, 0.0])
        for s, f, g, y, w, nivel in zip(tab["set_id"], tab["familia"], tab["grupo"], tab["y"], tab[f"peso_{n}"], tab["nivel"]):
            if y == 1 or nivel == n:
                suma[(s, f, g)][1 - y] += w
        for (pos, neg) in suma.values():
            if pos > 0:
                assert pos == pytest.approx(1.0) and neg == pytest.approx(1.0)


def test_auc_pesado_igual_a_sklearn():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    s = np.round(rng.normal(size=500), 1)                            # con empates
    y = (rng.random(500) < 0.3).astype(int)
    w = rng.random(500)
    assert ta.auc_pesado(s, y, w) == pytest.approx(roc_auc_score(y, s, sample_weight=w))


def test_medir_celda_intervalos():
    rng = np.random.default_rng(0)
    n = 400
    tab = {"familia": np.full(n, "P1"), "grupo": np.full(n, "principal"), "y": (np.arange(n) % 4 == 0).astype(int),
           "nivel": np.where(np.arange(n) % 4 == 0, "pos", "N1"), "dj": np.asarray([f"d{i % 10}" for i in range(n)]),
           "set_id": np.asarray([f"s{i % 20}" for i in range(n)])}
    for m in ta.NIVELES:
        tab[f"peso_{m}"] = np.ones(n) if m == "N1" else np.zeros(n)
    S = {"bueno": tab["y"] + rng.normal(scale=0.5, size=n), "azar": rng.random(n)}
    c = ta.medir_celda(S, tab, "P1", "principal", "N1", ["azar"], boot=200)
    assert c["positivos"] == 100 and c["djs"] == 10
    b = c["puntajes"]["bueno"]
    assert b["ic95"][0] <= b["auc"] <= b["ic95"][1] and b["auc"] > 0.8
    assert b["dif_con_azar"][0] > 0
