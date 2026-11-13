"""
PURPOSE: A1 de la línea A (docs/plans/linea_a_protocolo_a0_a1.md, fijado en el commit a6fc0f3): el tablero fijo de
         DJSets v1 sobre el que se mide todo modelo posterior, y sus líneas de base.
           armar  la partición de split_by_dj.csv con la cuarentena (un set cuyos nombres de DJ caen, por el hash del
                  armador, en partes distintas no entra en ninguna parte); los positivos de cada parte (P1 consecutivos,
                  P2 a 2 o 3 filas, P3 a 4 o más, con tope de TOPE_P3 por set), en grupos (principal: los dos temas
                  no vistos en train y con BPM y género de Beatport; vistos; sin_metadatos); 20 negativos por positivo
                  y nivel (N1 al azar, N2 mismo BPM respecto del ancla, N3 además el género de b) del pool de la parte;
                  pesos (cada set suma 1). Escribe pares_<parte>.csv.gz, particion.csv y resumen.json (C1, C2, conteos)
                  en una carpeta local y, con --subir, en erclm:traktor_ml/linea_a/a1_tablero/.
           medir  las líneas de base sobre las tablas armadas (sin volver a sortear): bpm, tonalidad, genero, azar y,
                  si están las vistas de A0 (djsets_v1_vistas/vistas), clap, clap_c, v8c, maest_c, estilos y los pesos
                  iguales; AUC pesado por familia, grupo y nivel; intervalos remuestreando DJs de la parte (1000 veces,
                  semilla 0, los mismos para todos); controles C3 y C4.
           python src/v4/evaluation/tablero_a1.py armar --dataset <copia local de djsets_v1> --salida <dir> [--subir]
           python src/v4/evaluation/tablero_a1.py medir --tablero <dir> --dataset <dir> [--vistas <dir> --u1 <e17/direcciones.npz>]
                                                        [--partes test]
         Nada de esto toca el recomendador ni usa los sets o la biblioteca de Gabriel.
CHANGELOG:
  - 2026-11-13: Creación (A1).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)

PARTES = ("train", "validation", "test")
FAMILIAS = ("P1", "P2", "P3")
NIVELES = ("N1", "N2", "N3")
GRUPOS = ("principal", "vistos", "sin_metadatos")
K_NEG = 20
K_NEG_TRAIN = 5          # train solo ajusta (A2, medias y desvíos): 5 negativos por nivel alcanzan y la tabla entra en memoria
TOL_BPM = 1.0           # N2: | |BPM(c) − BPM(a)| − |BPM(b) − BPM(a)| | ≤ 1
TOPE_P3 = 50             # positivos P3 por set (sorteados con la semilla), para que las tablas no crezcan con n²
SEMILLA = 0
BOOT = 1000
REMOTO = "linea_a/a1_tablero"
COLUMNAS = ("pid", "parte", "familia", "grupo", "nivel", "y", "set_id", "dj", "a", "x", "n_N1", "n_N2", "n_N3", "peso_N1",
            "peso_N2", "peso_N3")


# ---------------------------------------------------------------------------
# Partición
# ---------------------------------------------------------------------------

def nombres(dj: str) -> List[str]:
    """Los DJs de un nombre de set: lo de antes de « - », « @ » o «(», cortado en &, b2b, vs, x y comas, en minúsculas."""
    base = re.split(r"\s+[-@]\s+|\s*\(", dj or "")[0]
    return [p.strip().lower() for p in re.split(r"\s+(?:&|b2b|B2B|vs\.?|x)\s+|,\s*", base) if p.strip()]


def particion(filas_split: Sequence[Dict[str, str]], parte_de) -> Dict[str, Optional[str]]:
    """set_id -> parte de split_by_dj.csv, o None si el set queda en cuarentena (sus nombres caen en partes distintas
    por `parte_de`, el hash del armador)."""
    out = {}
    for f in filas_split:
        partes = {parte_de(n) for n in nombres(f["dj"])} | {f["split"]}
        out[f["set_id"]] = f["split"] if len(partes) == 1 else None
    return out


def control_c1(filas_split: Sequence[Dict[str, str]], parte: Dict[str, Optional[str]]) -> Dict[str, object]:
    """C1: ningún nombre de DJ en dos partes, fuera de la cuarentena."""
    donde: Dict[str, Set[str]] = defaultdict(set)
    for f in filas_split:
        if parte[f["set_id"]]:
            for n in nombres(f["dj"]):
                donde[n].add(parte[f["set_id"]])
    malos = sorted(n for n, p in donde.items() if len(p) > 1)
    return {"pasa": not malos, "nombres_en_dos_partes": malos[:20]}


# ---------------------------------------------------------------------------
# Temas y positivos
# ---------------------------------------------------------------------------

@dataclass
class Tema:
    track_id: str
    rec: str
    audio: bool
    bpm: float
    genero: str
    clave: str
    fuente: str

    @property
    def meta(self) -> bool:
        return self.audio and np.isfinite(self.bpm) and bool(self.genero)


def temas(tracks: Sequence[Dict[str, str]]) -> Dict[str, Tema]:
    out = {}
    for t in tracks:
        try:
            bpm = float(t.get("bp_bpm") or "nan")
        except ValueError:
            bpm = float("nan")
        out[t["track_id"]] = Tema(t["track_id"], t.get("recording_id", ""), t.get("has_audio") == "1", bpm,
                                  t.get("bp_genre", ""), t.get("bp_key", ""), t.get("audio_source", ""))
    return out


def filas_por_set(set_tracks: Sequence[Dict[str, str]]) -> Dict[str, List[Tuple[int, str]]]:
    """set_id -> [(índice, track_id)]: el índice cuenta las filas que no son «w/» (como ds.pairs), las filas sin
    tema ocupan su índice y no aparecen."""
    out: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    idx: Dict[str, int] = defaultdict(int)
    for f in set_tracks:
        if f["with_previous"] == "1":
            continue
        s = f["set_id"]
        i = idx[s]
        idx[s] += 1
        if f["track_id"]:
            out[s].append((i, f["track_id"]))
    return out


def familia(d: int) -> str:
    return "P1" if d == 1 else ("P2" if d <= 3 else "P3")


def pares_de_set(filas: Sequence[Tuple[int, str]], T: Dict[str, Tema]) -> Dict[Tuple[str, str], Tuple[int, str, str]]:
    """{(a, b) sin orden: (distancia mínima, ancla, otro)} para los temas con audio de un set; sin pares de la misma
    grabación. El ancla es el que sonó antes en la aparición más cercana."""
    con = [(i, t) for i, t in filas if t in T and T[t].audio]
    out: Dict[Tuple[str, str], Tuple[int, str, str]] = {}
    for x in range(len(con)):
        for y in range(x + 1, len(con)):
            (i, a), (j, b) = con[x], con[y]
            if a == b or T[a].rec == T[b].rec:
                continue
            k = (min(a, b), max(a, b))
            d = j - i
            if k not in out or d < out[k][0]:
                out[k] = (d, a, b)
    return out


# ---------------------------------------------------------------------------
# Negativos
# ---------------------------------------------------------------------------

@dataclass
class Pool:
    """Candidatos a negativo de una parte (vectores sobre el pool, para filtrar rápido)."""
    ids: List[str]
    bpm: np.ndarray
    genero: np.ndarray            # código del género, -1 sin género
    rec: np.ndarray               # código de la grabación
    djs: Dict[str, np.ndarray] = field(default_factory=dict)   # split_key -> máscara de temas que ese DJ tocó en la parte
    pos: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def armar(cls, ids: Sequence[str], T: Dict[str, Tema], dj_de_tema: Dict[str, Set[str]], codigos_genero: Dict[str, int],
              codigos_rec: Dict[str, int]) -> "Pool":
        ids = list(ids)
        p = cls(ids, np.asarray([T[t].bpm for t in ids], float),
                np.asarray([codigos_genero.get(T[t].genero, -1) for t in ids], int),
                np.asarray([codigos_rec[T[t].rec] for t in ids], int))
        p.pos = {t: i for i, t in enumerate(ids)}
        idx_dj: Dict[str, List[int]] = defaultdict(list)
        for i, t in enumerate(ids):
            for k in dj_de_tema.get(t, ()):
                idx_dj[k].append(i)
        for k, ii in idx_dj.items():
            m = np.zeros(len(ids), bool)
            m[ii] = True
            p.djs[k] = m
        return p


def candidatos(p: Pool, a: str, b: str, dj: str, T: Dict[str, Tema], co: Dict[str, Set[str]], codigos_genero: Dict[str, int],
               codigos_rec: Dict[str, int], meta: bool) -> Dict[str, np.ndarray]:
    """Índices del pool que pueden ser negativo de (a, b) en cada nivel (solo N1 si meta es False)."""
    ok = np.ones(len(p.ids), bool)
    if dj in p.djs:
        ok &= ~p.djs[dj]
    for t in co.get(a, ()):
        i = p.pos.get(t)
        if i is not None:
            ok[i] = False
    ok &= (p.rec != codigos_rec[T[a].rec]) & (p.rec != codigos_rec[T[b].rec])
    out = {"N1": np.flatnonzero(ok)}
    if meta:
        d = abs(T[b].bpm - T[a].bpm)
        n2 = ok & (np.abs(np.abs(p.bpm - T[a].bpm) - d) <= TOL_BPM)
        out["N2"] = np.flatnonzero(n2)
        out["N3"] = np.flatnonzero(n2 & (p.genero == codigos_genero.get(T[b].genero, -2)))
    return out


# ---------------------------------------------------------------------------
# Armar
# ---------------------------------------------------------------------------

def leer_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def md5_gz(path: Path) -> str:
    """md5 del contenido descomprimido: identifica el tablero aunque cambie la compresión."""
    h = hashlib.md5()
    with gzip.open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verificar_tablas(dataset: Path) -> Dict[str, str]:
    """md5 de las tablas de metadata frente a checksums.md5 del dataset (la copia local tiene que ser la del Drive)."""
    esperado = {}
    for linea in (dataset / "checksums.md5").read_text(encoding="utf-8").splitlines():
        h, _, rel = linea.partition("  ")
        esperado[rel.strip()] = h
    out = {}
    for p in sorted((dataset / "metadata").glob("*.csv")):
        rel = f"metadata/{p.name}"
        h = md5(p)
        if esperado.get(rel) != h:
            raise SystemExit(f"{rel}: md5 {h} distinto del de checksums.md5 ({esperado.get(rel)})")
        out[rel] = h
    return out


def armar(dataset: Path, salida: Path, parte_de=None, semilla: int = SEMILLA, k: int = K_NEG,
          tope_p3: int = TOPE_P3, verificar: bool = True) -> Dict[str, object]:
    t0 = time.perf_counter()
    if parte_de is None:
        from datasets.djsets.build_djsets import split_de as parte_de
    md5s = verificar_tablas(dataset) if verificar else {}
    meta = dataset / "metadata"
    tracks, set_tracks = leer_csv(meta / "tracks.csv"), leer_csv(meta / "set_tracks.csv")
    filas_split = leer_csv(meta / "split_by_dj.csv")
    T = temas(tracks)
    parte = particion(filas_split, parte_de)
    clave = {f["set_id"]: f["split_key"] for f in filas_split}
    por_set = filas_por_set(set_tracks)
    # Coaparición en todo el dataset (cualquier parte, también la cuarentena)
    co: Dict[str, Set[str]] = defaultdict(set)
    for s, filas in por_set.items():
        ids = {t for _, t in filas}
        for t in ids:
            co[t] |= ids
    # Temas de cada parte y quién los tocó
    en_parte: Dict[str, Set[str]] = defaultdict(set)
    dj_de: Dict[str, Dict[str, Set[str]]] = {p: defaultdict(set) for p in PARTES}
    for s, filas in por_set.items():
        p = parte.get(s)
        if p:
            for _, t in filas:
                en_parte[p].add(t)
                dj_de[p][t].add(clave[s])
    vistos_t = en_parte["train"]
    vistos_r = {T[t].rec for t in vistos_t if t in T and T[t].rec}

    def visto(t: str) -> bool:
        return t in vistos_t or (bool(T[t].rec) and T[t].rec in vistos_r)

    codigos_genero = {g: i for i, g in enumerate(sorted({x.genero for x in T.values() if x.genero}))}
    codigos_rec = {r: i for i, r in enumerate(sorted({x.rec for x in T.values()}))}
    rng = np.random.default_rng(semilla)
    salida.mkdir(parents=True, exist_ok=True)
    resumen: Dict[str, object] = {"dataset": str(dataset), "md5_tablas": md5s, "semilla": semilla, "k": k,
                                  "tol_bpm": TOL_BPM, "tope_p3": tope_p3, "partes": {}}
    for p in PARTES:
        sets_p = sorted(s for s in por_set if parte.get(s) == p)
        es_train = p == "train"
        pool_meta = sorted(t for t in en_parte[p] if t in T and T[t].meta and (es_train or not visto(t)))
        pool_aud = sorted(t for t in en_parte[p] if t in T and T[t].audio and (es_train or not visto(t)))
        PM = Pool.armar(pool_meta, T, dj_de[p], codigos_genero, codigos_rec)
        PA = Pool.armar(pool_aud, T, dj_de[p], codigos_genero, codigos_rec)
        k_p = K_NEG_TRAIN if es_train else k
        pos: List[Dict[str, object]] = []
        negs: List[Tuple[Pool, Dict[str, np.ndarray]]] = []
        cuenta = Counter()
        pid = 0
        for s in sets_p:
            pares = pares_de_set(por_set[s], T)
            por_fam: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
            for _, (d, a, b) in sorted(pares.items()):
                por_fam[familia(d)].append((a, b))
            if len(por_fam["P3"]) > tope_p3:
                elegidos = rng.choice(len(por_fam["P3"]), tope_p3, replace=False)
                por_fam["P3"] = [por_fam["P3"][i] for i in sorted(elegidos)]
            for fam in FAMILIAS:
                for a, b in por_fam[fam]:
                    con_meta = T[a].meta and T[b].meta
                    no_vistos = es_train or (not visto(a) and not visto(b))
                    grupo = "principal" if con_meta and no_vistos else ("vistos" if con_meta else
                                                                        ("sin_metadatos" if no_vistos else None))
                    if grupo is None:
                        cuenta[f"{fam}_vistos_sin_metadatos_fuera"] += 1
                        continue
                    pool = PM if con_meta else PA
                    cand = candidatos(pool, a, b, clave[s], T, co, codigos_genero, codigos_rec, con_meta)
                    elegidos = {n: (np.sort(rng.choice(c, min(k_p, len(c)), replace=False)) if len(c) else np.zeros(0, int))
                                for n, c in cand.items()}
                    pid += 1
                    pos.append({"pid": pid, "parte": p, "familia": fam, "grupo": grupo, "nivel": "pos", "y": 1,
                                "set_id": s, "dj": clave[s], "a": a, "x": b,
                                **{f"n_{n}": len(elegidos.get(n, ())) for n in NIVELES}})
                    negs.append((pool, elegidos))
                    cuenta[f"{fam}_{grupo}"] += 1
        pesar(pos)
        ruta = salida / f"pares_{p}.csv.gz"
        n_filas = escribir(ruta, pos, negs)
        resumen["partes"][p] = {
            "sets": len(sets_p), "djs": len({clave[s] for s in sets_p}), "pool_con_metadatos": len(pool_meta),
            "pool_con_audio": len(pool_aud), "k": k_p, "positivos": dict(sorted(cuenta.items())), "filas": n_filas,
            "md5": md5(ruta), "md5_contenido": md5_gz(ruta),
            "principal_P1": {"positivos": sum(1 for f in pos if f["familia"] == "P1" and f["grupo"] == "principal"),
                             "djs": len({f["dj"] for f in pos if f["familia"] == "P1" and f["grupo"] == "principal"}),
                             "sets": len({f["set_id"] for f in pos if f["familia"] == "P1" and f["grupo"] == "principal"})},
            "positivos_sin_negativos": {n: sum(1 for f in pos if f["grupo"] != "sin_metadatos" and f[f"n_{n}"] == 0)
                                        for n in NIVELES},
            "positivos_con_menos_de_k": {n: sum(1 for f in pos if f["grupo"] != "sin_metadatos" and 0 < f[f"n_{n}"] < k_p)
                                         for n in NIVELES}}
        print(f"{p}: {len(sets_p)} sets, {len(pos)} positivos, {n_filas} filas ({time.perf_counter() - t0:.0f} s)",
              flush=True)
    # P1 por índice frente a ds.pairs (misma definición)
    ds_pairs = set()
    con_audio = {t for t, x in T.items() if x.audio}
    for s, filas in por_set.items():
        for (i, a), (j, b) in zip(filas, filas[1:]):
            if j == i + 1 and a in con_audio and b in con_audio:
                ds_pairs.add((s, a, b))
    resumen["c1"] = control_c1(filas_split, parte)
    pool_test = {t for t in en_parte["test"] if t in T and T[t].audio and not visto(t)}
    resumen["c2"] = {"pasa": not (pool_test & vistos_t) and not ({T[t].rec for t in pool_test} & vistos_r)}
    resumen["cuarentena"] = dict(Counter(f["split"] for f in filas_split if parte[f["set_id"]] is None))
    resumen["pares_consecutivos_con_audio"] = len(ds_pairs)
    resumen["commit"] = git_head()
    resumen["segundos"] = round(time.perf_counter() - t0, 1)
    with open(salida / "particion.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set_id", "dj", "split_key", "split", "parte_tablero"])
        for x in filas_split:
            w.writerow([x["set_id"], x["dj"], x["split_key"], x["split"], parte[x["set_id"]] or "cuarentena"])
    resumen["md5_particion"] = md5(salida / "particion.csv")
    (salida / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return resumen


def pesar(pos: List[Dict[str, object]]) -> None:
    """Pesos de los positivos: cada set suma 1 por familia, grupo y nivel, y un positivo cuenta en un nivel solo si
    tiene negativos ahí. Cada negativo pesa lo de su positivo dividido por cuántos negativos tiene en ese nivel (así
    los negativos de un set también suman 1); lo aplica escribir()."""
    for n in NIVELES:
        cuantos = Counter((f["set_id"], f["familia"], f["grupo"]) for f in pos if f[f"n_{n}"] > 0)
        for f in pos:
            c = cuantos.get((f["set_id"], f["familia"], f["grupo"]), 0)
            f[f"peso_{n}"] = (1.0 / c) if f[f"n_{n}"] > 0 and c else 0.0


def _fmt(v) -> str:
    return f"{v:.8g}" if isinstance(v, float) else str(v)


def escribir(ruta: Path, pos: Sequence[Dict[str, object]], negs: Sequence[Tuple[Pool, Dict[str, np.ndarray]]]) -> int:
    """Cada positivo seguido de sus negativos, por nivel. Devuelve las filas escritas. El gzip va sin fecha ni nombre
    en la cabecera (mtime=0), así el mismo tablero da el mismo md5 cada vez que se arma."""
    n = 0
    with open(ruta, "wb") as crudo, gzip.GzipFile(filename="", fileobj=crudo, mode="wb", mtime=0) as gz, \
            io.TextIOWrapper(gz, encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNAS)
        for x, (pool, elegidos) in zip(pos, negs):
            w.writerow([_fmt(x.get(c, "")) for c in COLUMNAS])
            n += 1
            for nivel, ii in elegidos.items():
                pn = x[f"peso_{nivel}"] / len(ii) if len(ii) else 0.0
                pesos = {m: (pn if m == nivel else 0.0) for m in NIVELES}
                for i in ii:
                    w.writerow([x["pid"], x["parte"], x["familia"], x["grupo"], nivel, 0, x["set_id"], x["dj"], x["a"],
                                pool.ids[i], "", "", "", _fmt(pesos["N1"]), _fmt(pesos["N2"]), _fmt(pesos["N3"])])
                    n += 1
    return n


def leer_tablero(ruta: Path) -> Dict[str, np.ndarray]:
    import pandas as pd
    df = pd.read_csv(ruta, dtype={c: str for c in ("parte", "familia", "grupo", "nivel", "set_id", "dj", "a", "x")},
                     keep_default_na=False)
    out = {c: df[c].to_numpy(dtype=object).astype(str) for c in ("parte", "familia", "grupo", "nivel", "set_id", "dj", "a", "x")}
    out["pid"] = df["pid"].to_numpy(dtype=np.int64)
    out["y"] = df["y"].to_numpy(dtype=np.int64)
    for n in NIVELES:
        out[f"peso_{n}"] = df[f"peso_{n}"].to_numpy(dtype=np.float64)
    return out


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------------------
# Medir
# ---------------------------------------------------------------------------

def unit(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)


def vector_tema(V: np.ndarray, n_validos: np.ndarray) -> np.ndarray:
    """(n, 4, d) -> (n, d): media de los segmentos válidos, normalizada."""
    m = (np.arange(V.shape[1])[None, :] < np.maximum(n_validos, 1)[:, None]).astype(np.float32)
    return unit((V * m[:, :, None]).sum(axis=1) / m.sum(axis=1, keepdims=True))


@dataclass
class Vistas:
    fila: Dict[str, int]                    # track_id -> fila
    E: Dict[str, np.ndarray]                # vista -> (n, d) normalizada

    @classmethod
    def cargar(cls, carpeta: Path, train_ids: Iterable[str], u1: np.ndarray) -> "Vistas":
        """Las vistas de A0 por tema; centros con la media de los temas de train; v8c sin u1 (E17)."""
        ind = leer_csv(carpeta / "indice.csv")
        fila = {r["track_id"]: i for i, r in enumerate(ind) if r["n_seg_validos"] != "0"}
        nv = np.asarray([int(r["n_seg_validos"]) for r in ind])
        crudo = {k: vector_tema(np.load(carpeta / f"{k}.npy"), nv) for k in ("clap", "l7", "styles")}
        tr = sorted({fila[t] for t in train_ids if t in fila})
        mu = {k: v[tr].mean(axis=0) for k, v in crudo.items()}
        clap_c = unit(crudo["clap"] - mu["clap"])
        v8 = clap_c - np.outer(clap_c @ u1, u1)
        return cls(fila, {"clap": crudo["clap"], "clap_c": clap_c, "v8c": unit(v8),
                          "maest_c": unit(crudo["l7"] - mu["l7"]), "estilos": crudo["styles"]})


def _auc_grupos(inv: np.ndarray, G: int, y: np.ndarray, w: np.ndarray) -> Optional[float]:
    """AUC pesado con los puntajes ya agrupados en G valores ordenados (inv: el grupo de cada fila); empates a la mitad."""
    wp = np.bincount(inv, weights=w * (y == 1), minlength=G)
    wn = np.bincount(inv, weights=w * (y == 0), minlength=G)
    P, N = wp.sum(), wn.sum()
    if P <= 0 or N <= 0:
        return None
    antes = np.cumsum(wn) - wn
    return float((wp * (antes + 0.5 * wn)).sum() / (P * N))


def auc_pesado(s: np.ndarray, y: np.ndarray, w: np.ndarray) -> Optional[float]:
    """P(positivo > negativo) pesada, empates a la mitad (= roc_auc_score con sample_weight)."""
    _, inv = np.unique(s, return_inverse=True)
    return _auc_grupos(inv.ravel(), int(inv.max()) + 1 if len(inv) else 0, y, w)


def puntajes(tab: Dict[str, np.ndarray], T: Dict[str, Tema], vistas: Optional[Vistas], semilla: int = SEMILLA
             ) -> Dict[str, np.ndarray]:
    """Factores de cada fila (a, x). nan = no se puede calcular (falta el dato)."""
    from src.v4.common import harmonic
    from tools.traktor_live import recomendador as rc
    tax = rc.load_taxonomy()[0]
    a, x = tab["a"], tab["x"]
    ba = np.asarray([T[t].bpm for t in a])
    bx = np.asarray([T[t].bpm for t in x])
    out = {"bpm": np.maximum(0.0, 1.0 - np.abs(ba - bx) / rc.BPM_RANGE)}
    cam = {t: harmonic.to_camelot(T[t].clave) for t in set(a) | set(x)}
    cache: Dict[Tuple[str, str], float] = {}

    def clave(p, q):
        if cam[p] == "?" or cam[q] == "?":
            return np.nan
        kk = (cam[p], cam[q])
        if kk not in cache:
            cache[kk] = harmonic.best_key_shift(*kk)[0]
        return cache[kk]

    out["tonalidad"] = np.asarray([clave(p, q) for p, q in zip(a, x)])
    gcache: Dict[Tuple[str, str], float] = {}

    def gen(p, q):
        gp, gq = T[p].genero, T[q].genero
        if not gp or not gq:
            return np.nan
        if (gp, gq) not in gcache:
            v = tax.compare((gp, None), (gq, None)) if tax is not None else float(gp == gq)
            gcache[(gp, gq)] = np.nan if v is None else float(v)
        return gcache[(gp, gq)]

    out["genero"] = np.asarray([gen(p, q) for p, q in zip(a, x)])
    out["azar"] = np.random.default_rng(semilla).random(len(a))
    if vistas is not None:
        ia = np.asarray([vistas.fila.get(t, -1) for t in a])
        ix = np.asarray([vistas.fila.get(t, -1) for t in x])
        ok = (ia >= 0) & (ix >= 0)
        for v, E in vistas.E.items():
            c = np.full(len(a), np.nan)
            c[ok] = np.einsum("ij,ij->i", E[ia[ok]], E[ix[ok]])
            out[v] = c
    return out


def z_stats(P: Dict[str, np.ndarray], tab: Dict[str, np.ndarray], cols: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    """Media y desvío pesados de cada factor sobre los negativos N1 (de train)."""
    m = (tab["y"] == 0) & (tab["nivel"] == "N1")
    w = tab["peso_N1"]
    out = {}
    for c in cols:
        x = P[c][m]
        ok = np.isfinite(x)
        mu = float(np.average(x[ok], weights=w[m][ok]))
        sd = float(np.sqrt(np.average((x[ok] - mu) ** 2, weights=w[m][ok])))
        out[c] = (mu, sd if sd > 1e-12 else 1.0)
    return out


def suma_z(P: Dict[str, np.ndarray], cols: Sequence[str], st: Dict[str, Tuple[float, float]]) -> np.ndarray:
    Z = np.column_stack([(P[c] - st[c][0]) / st[c][1] for c in cols])
    Z[~np.isfinite(Z)] = 0.0
    return Z.sum(axis=1)


PESOS_IGUALES = {"pesos iguales (v8c)": ("v8c", "maest_c", "estilos", "bpm", "tonalidad", "genero"),
                 "pesos iguales (clap_c)": ("clap_c", "maest_c", "estilos", "bpm", "tonalidad", "genero")}


def mascara(tab: Dict[str, np.ndarray], familia_: str, grupo: str, nivel: str) -> Tuple[np.ndarray, np.ndarray]:
    """(filas, pesos) de una celda: los positivos con negativos en el nivel y los negativos de ese nivel."""
    base = (tab["familia"] == familia_) & (tab["grupo"] == grupo)
    m = base & (((tab["y"] == 1) & (tab[f"peso_{nivel}"] > 0)) | ((tab["y"] == 0) & (tab["nivel"] == nivel)))
    return m, tab[f"peso_{nivel}"]


def medir_celda(S: Dict[str, np.ndarray], tab: Dict[str, np.ndarray], fam: str, grupo: str, nivel: str,
                refs: Sequence[str], boot: int = BOOT, semilla: int = SEMILLA) -> Dict[str, object]:
    m, w = mascara(tab, fam, grupo, nivel)
    idx = np.flatnonzero(m)
    if not len(idx) or not (tab["y"][idx] == 1).any():
        return {}
    y, ww, djs = tab["y"][idx], w[idx], tab["dj"][idx]
    ug, dj_i = np.unique(djs, return_inverse=True)
    dj_i = dj_i.ravel()
    rng = np.random.default_rng(semilla)
    # Remuestrear DJs con reemplazo = multiplicar el peso de cada fila por cuántas veces salió su DJ (los mismos
    # remuestreos para todos los puntajes de la celda).
    veces = np.stack([np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)) for _ in range(boot)])
    out: Dict[str, object] = {"positivos": int((y == 1).sum()), "negativos": int((y == 0).sum()), "djs": len(ug),
                              "sets": int(len(np.unique(tab["set_id"][idx]))), "puntajes": {}}
    auc_boot: Dict[str, np.ndarray] = {}
    for nombre, s in S.items():
        s = s[idx]
        ok = np.isfinite(s)
        if not ok.all():
            s = np.where(ok, s, np.nanmin(s[ok]) - 1 if ok.any() else 0.0)   # falta el dato: queda último
        _, inv = np.unique(s, return_inverse=True)
        inv = inv.ravel()
        G = int(inv.max()) + 1
        a = _auc_grupos(inv, G, y, ww)
        b = np.asarray([_auc_grupos(inv, G, y, ww * v[dj_i]) or np.nan for v in veces])
        auc_boot[nombre] = b
        out["puntajes"][nombre] = {"auc": None if a is None else round(a, 4),
                                   "ic95": [round(float(np.nanpercentile(b, 2.5)), 4), round(float(np.nanpercentile(b, 97.5)), 4)],
                                   "sin_dato": int((~ok).sum())}
    for r in refs:
        if r not in auc_boot:
            continue
        for nombre in S:
            if nombre == r:
                continue
            d = auc_boot[nombre] - auc_boot[r]
            out["puntajes"][nombre][f"dif_con_{r}"] = [round(float(np.nanpercentile(d, 2.5)), 4),
                                                        round(float(np.nanpercentile(d, 97.5)), 4)]
    return out


def medir(tablero: Path, dataset: Path, vistas_dir: Optional[Path], partes: Sequence[str], u1_path: Optional[Path],
          boot: int = BOOT) -> Dict[str, object]:
    T = temas(leer_csv(dataset / "metadata" / "tracks.csv"))
    tab_train = leer_tablero(tablero / "pares_train.csv.gz")
    vistas = None
    if vistas_dir is not None:
        u1 = np.load(u1_path)["clap_u1"]
        train_ids = set(tab_train["a"]) | set(tab_train["x"])
        vistas = Vistas.cargar(vistas_dir, train_ids, u1 / np.linalg.norm(u1))
    P_train = puntajes(tab_train, T, vistas)
    cols = sorted({c for cs in PESOS_IGUALES.values() for c in cs if c in P_train})
    st = z_stats(P_train, tab_train, cols)
    res: Dict[str, object] = {"tablero": str(tablero), "vistas": str(vistas_dir) if vistas_dir else None,
                              "z_de_train": {c: [round(v[0], 5), round(v[1], 5)] for c, v in st.items()},
                              "commit": git_head(), "partes": {}}
    for p in partes:
        tab = tab_train if p == "train" else leer_tablero(tablero / f"pares_{p}.csv.gz")
        P = P_train if p == "train" else puntajes(tab, T, vistas)
        S = dict(P)
        if vistas is not None:
            for nombre, cs in PESOS_IGUALES.items():
                S[nombre] = suma_z(P, cs, st)
        refs = [r for r in ("pesos iguales (v8c)", "bpm") if r in S]
        celdas = {}
        for fam in FAMILIAS:
            for g in GRUPOS:
                for n in (NIVELES if g != "sin_metadatos" else ("N1",)):
                    c = medir_celda(S, tab, fam, g, n, refs, boot)
                    if c:
                        celdas[f"{fam}/{g}/{n}"] = c
                        print(f"  {p} {fam}/{g}/{n}: " + "  ".join(f"{k} {v['auc']}" for k, v in c["puntajes"].items()),
                              flush=True)
        res["partes"][p] = celdas
    prin = res["partes"].get("test", {})
    def auc(celda, k):
        return prin.get(celda, {}).get("puntajes", {}).get(k, {}).get("auc")
    res["c3"] = {"bpm_N2": auc("P1/principal/N2", "bpm"), "bpm_N3": auc("P1/principal/N3", "bpm"),
                 "genero_N3": auc("P1/principal/N3", "genero")}
    res["c3"]["pasa"] = (all(v is not None and 0.47 <= v <= 0.53 for v in (res["c3"]["bpm_N2"], res["c3"]["bpm_N3"]))
                         and res["c3"]["genero_N3"] is not None and abs(res["c3"]["genero_N3"] - 0.5) <= 0.005)
    ic = prin.get("P1/principal/N3", {}).get("puntajes", {}).get("azar", {}).get("ic95")
    res["c4"] = {"azar_ic95": ic, "pasa": bool(ic) and ic[0] <= 0.5 <= ic[1]}
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="A1: tablero fijo de DJSets v1")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("armar")
    a.add_argument("--dataset", required=True)
    a.add_argument("--salida", required=True)
    a.add_argument("--subir", action="store_true", help=f"Sube la carpeta a erclm:traktor_ml/{REMOTO}/")
    m = sub.add_parser("medir")
    m.add_argument("--tablero", required=True)
    m.add_argument("--dataset", required=True)
    m.add_argument("--vistas", default=None)
    m.add_argument("--u1", default=None, help="direcciones.npz de E17 (<dj_sets>/e17/); obligatoria con --vistas")
    m.add_argument("--partes", default="test")
    m.add_argument("--boot", type=int, default=BOOT)
    m.add_argument("--json", default=None)
    args = ap.parse_args()
    if args.cmd == "medir" and args.vistas and not args.u1:
        ap.error("medir con --vistas necesita --u1 (la dirección del códec de E17)")
    if args.cmd == "armar":
        r = armar(Path(args.dataset), Path(args.salida))
        print(json.dumps({k: v for k, v in r.items() if k != "md5_tablas"}, ensure_ascii=False, indent=1))
        if args.subir:
            from src.v4.common.remote_storage import open_remote
            remote = open_remote()
            for f in sorted(Path(args.salida).iterdir()):
                remote.put(f, f"{REMOTO}/{f.name}")
            print(f"subido a {REMOTO}/")
        return 0
    r = medir(Path(args.tablero), Path(args.dataset), Path(args.vistas) if args.vistas else None,
              [x for x in args.partes.split(",") if x], Path(args.u1) if args.vistas else None, args.boot)
    texto = json.dumps(r, ensure_ascii=False, indent=1)
    if args.json:
        Path(args.json).write_text(texto + "\n", encoding="utf-8")
    print(json.dumps({k: r[k] for k in ("c3", "c4")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
