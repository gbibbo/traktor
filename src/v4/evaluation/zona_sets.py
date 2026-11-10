"""
PURPOSE: Medidas E9 y E10 del plan de playlists en vivo (docs/plans/recomendacion_playlists_plan.md, sección 11).
         Qué distancia CLAP separa a dos temas que un DJ profesional tocó en el mismo set (radio de zona) y
         cuánto de la lista de Parecidos cae dentro de ese radio. Mide precisión de la lista, no el acierto en
         30 del tema siguiente (banco_sets.py). Solo CLAP crudo: acá no se aprende nada.
         Fuentes, cada una curada; lo que queda afuera se cuenta en «curado»:
           espacio de la biblioteca (originales, donde se usa el radio):
             d3_biblioteca  temas de la biblioteca que aparecen en un mismo set de D3 (MixesDB): misma versión,
                            artistas distintos, sets de 2010 en adelante;
             bedrock        los originales de los discos de Digweed del piloto (un solo DJ);
             d1             los sets de Gabriel: referencia con ruido (regla E6), nunca para fijar nada;
           audio mezclado (otra escala de distancias: la mezcla junta los temas):
             d7_por_tema, d7_continua  compilaciones validadas (tools/dj_sets/validar_d7.py);
             bloques de discos enteros y D8, solo pares separados por 10 minutos o más.
         E9: distancia entre temas de un mismo set frente a pares al azar del mismo rango de BPM. Cada set pesa
         igual, con intervalo por remuestreo de sets. Cobertura cruzada: el radio de una fuente aplicado a los
         pares de otra. Control de dominio: los mismos pares de Bedrock en el original y en la mezcla.
         E10: (a) fracción de los 30 de Parecidos dentro de cada radio, con el recomendador en vivo y sus filtros,
         frente a una lista al azar y a una por BPM; (b) pureza: desde un tema de un set, cuántos de los 30 son de
         temas posteriores de ese set, con lo ya tocado fuera del pool y el azar exacto al lado. El pool no mezcla
         dominios salvo en una fila de control, que informa cuánto de los 30 sale de cada uno.
           python src/v4/evaluation/zona_sets.py --root "D:/TRAKTOR ML datos/dj_sets" --low-priority
         El tempo de D7 y de Bedrock sale de tools/dj_sets/tempo_sets.py (sets/tempo_temas.csv).
CHANGELOG:
  - 2026-11-08: Creación (unidad A1 de la sección 11 del plan).
  - 2026-11-09: Fuente.fila_archivo: carpeta/archivo de cada original de Bedrock, para sumarle otras vistas
                (MAEST de tools/dj_sets/maest_originales.py; fusion_vistas.py).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from src.v4.evaluation.banco_sets import load_sets  # noqa: E402
from tools.dj_sets.piloto import bounds, match_tracks, unit  # noqa: E402
from tools.dj_sets.tempo_sets import BEDROCK as CARPETAS_BEDROCK  # noqa: E402
from tools.dj_sets.trayectorias_d7 import accepted, identity, select_blocks, track_vector  # noqa: E402
from tools.traktor_live import recomendador as rc  # noqa: E402

DEFAULT_ROOT = REPO_ROOT / "artifacts" / "v4" / "dj_sets"
DEFAULT_D3 = DEFAULT_ROOT / "djmix" / "djmix-dataset.json"
RELEASES_BEDROCK = dict(zip(CARPETAS_BEDROCK, ("4cb11ccd-699c-4711-b438-ba9b222878a7", "1893e875-c31c-4497-8c7e-646d21a66d78",
                                                "6810b907-0d77-422c-9e20-a03f176a2ca0")))
TOP = 30
PERCENTILES = (10, 25, 50)
BPM_TOL = 4.0                  # «mismo rango de BPM»: a ±4 BPM de cada tema del par real
AZAR_POR_PAR = 20
MIN_TEMAS_D7 = 5
MIN_TEMAS_D1 = 8
D3_DESDE = 2010
SEP_BLOQUES_S = 600.0
MAX_PARES_BLOQUES = 2000
COS_DUPLICADO = 0.97
BOOTSTRAP = 1000
_TIEMPO = re.compile(r"^\s*\[[^\]]*\]\s*")
_SELLO = re.compile(r"\s*\[[^\]]*\]\s*$")


def estricto(s: Optional[str]) -> str:
    """Texto comparable que conserva la versión (lo que va entre paréntesis): minúsculas, sin acentos ni signos."""
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def primer_artista(a: Optional[str]) -> str:
    return re.split(r",|&| x | vs\.? ", a or "")[0].strip()


def tema_de_d3(title: Optional[str]) -> Optional[Tuple[str, str]]:
    """(artista, título) de una línea de tracklist de MixesDB: «[05] Code 61 - Drop The Deal [Sello - Cat]»."""
    s = _SELLO.sub("", _TIEMPO.sub("", title or ""))
    if " - " not in s:
        return None
    a, t = s.split(" - ", 1)
    return (a.strip(), t.strip()) if a.strip() and t.strip() else None


def anio_de(texto: Optional[str]) -> Optional[int]:
    try:
        y = int(str(texto or "")[:4])
    except ValueError:
        return None
    return y if 1900 < y < 2100 else None


@dataclass
class Biblioteca:
    uids: List[str]
    V: np.ndarray
    bpm: np.ndarray
    artista: List[str]
    pos: Dict[str, int]
    por_nombre: Dict[Tuple[str, str], int]


@dataclass
class Fuente:
    """Temas (filas válidas de V, unitarias) y sets (filas en el orden de la noche). Las etiquetas por fila son
    opcionales y solo las usa D7."""
    nombre: str
    dominio: str                                  # «biblioteca» o «mezcla»
    V: np.ndarray
    bpm: np.ndarray
    artista: List[str]
    sets: Dict[str, List[int]]
    dj: Dict[str, str] = field(default_factory=dict)
    curado: Dict[str, object] = field(default_factory=dict)
    fila_grupo: Optional[np.ndarray] = None       # compilación
    fila_dj: Optional[np.ndarray] = None
    fila_anio: Optional[np.ndarray] = None
    fila_archivo: Optional[List[str]] = None      # «carpeta/archivo» de cada fila (Bedrock)


def cargar_biblioteca(items: Sequence[rc.Item]) -> Biblioteca:
    items = [it for it in items if it.vec is not None]
    V = unit(np.asarray([it.vec for it in items], dtype=np.float64))
    bpm = np.asarray([it.bpm if it.bpm else np.nan for it in items], dtype=np.float64)
    artista = [estricto(primer_artista(it.artist)) for it in items]
    por_nombre: Dict[Tuple[str, str], int] = {}
    for i, it in enumerate(items):
        if artista[i] and estricto(it.title):
            por_nombre.setdefault((artista[i], estricto(it.title)), i)
    return Biblioteca([it.uid for it in items], V, bpm, artista, {it.uid: i for i, it in enumerate(items)}, por_nombre)


def cargar_tempo(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as fh:
        return {r["clave"]: float(r["bpm"]) for r in csv.DictReader(fh) if r["bpm"] != ""}


def fuente_d1(path: Path, bib: Biblioteca) -> Fuente:
    sets, dj = {}, {}
    fuera, repetidos, descartados = 0, 0, 0
    for sid, rows in load_sets(path).items():
        filas: List[int] = []
        for r in rows:
            i = bib.pos.get(r["track_uid"])
            if i is None:
                fuera += 1
            elif i in filas:
                repetidos += 1
            else:
                filas.append(i)
        if len(filas) < MIN_TEMAS_D1:
            descartados += 1
            continue
        sets[sid], dj[sid] = filas, "Gabriel"
    return Fuente("d1", "biblioteca", bib.V, bib.bpm, bib.artista, sets, dj,
                  {"sets": len(sets), "sets_descartados_por_pocos_temas": descartados, "filas_fuera_del_indice": fuera,
                   "temas_repetidos_en_un_set": repetidos})


def fuente_d3(path: Path, bib: Biblioteca) -> Fuente:
    """Sets de D3 con dos temas o más de la biblioteca, comparando artista y título con la versión incluida."""
    mixes = json.loads(path.read_text(encoding="utf-8"))
    sets, dj = {}, {}
    viejos, uno = 0, 0
    for m in mixes:
        y = anio_de(m.get("title"))
        filas: List[int] = []
        for tr in m.get("tracklist") or []:
            t = tema_de_d3(tr.get("title"))
            if t is None:
                continue
            i = bib.por_nombre.get((estricto(primer_artista(t[0])), estricto(t[1])))
            if i is not None and i not in filas:
                filas.append(i)
        if len(filas) < 2:
            uno += int(len(filas) == 1)
            continue
        if y is None or y < D3_DESDE:
            viejos += 1
            continue
        tags = m.get("tags") or [{}]
        sets[m["id"]], dj[m["id"]] = filas, str(tags[0].get("key", "?")).replace("Category:", "")
    return Fuente("d3_biblioteca", "biblioteca", bib.V, bib.bpm, bib.artista, sets, dj,
                  {"sets_d3": len(mixes), "sets_con_2_o_mas_temas": len(sets), "sets_con_un_solo_tema": uno,
                   f"sets_anteriores_a_{D3_DESDE}_o_sin_fecha": viejos})


def discos_bedrock(root: Path, carpeta: str) -> Tuple[Dict, List[Dict], Dict[int, Dict], Dict]:
    man = json.loads((root / "audio" / carpeta / "manifest.json").read_text(encoding="utf-8"))
    mb = json.loads((root / "musicbrainz" / f"{RELEASES_BEDROCK[carpeta]}.json").read_text(encoding="utf-8"))
    entradas = man["entradas"]
    originales = [e for e in entradas if not re.search(r"continuous mix", e.get("titulo") or "", re.I)]
    mezclas = {int(m.group(1)): e for e in entradas if (m := re.search(r"continuous mix (\d+)", e.get("titulo") or "", re.I))}
    return man, originales, mezclas, mb


def fuente_bedrock(root: Path, tempo: Dict[str, float], bib: Biblioteca) -> Tuple[Fuente, np.ndarray]:
    """Originales de los discos de Bedrock con el embedding como en la biblioteca («dj»: 3 tramos del 35-65 %), en el
    orden del set. Devuelve además la máscara de los temas de la biblioteca que son el mismo tema (por contenido)."""
    V, bpm, artista, sets, dj, archivos = [], [], [], {}, {}, []
    sin_dj, sin_pareja = 0, 0
    for carpeta in CARPETAS_BEDROCK:
        man, originales, _, mb = discos_bedrock(root, carpeta)
        vistos, filas = set(), []
        for disc in mb["media"]:
            for t, j in zip(disc["pistas"], match_tracks(disc["pistas"], originales)):
                if j is None:
                    sin_pareja += 1
                    continue
                if j in vistos or not originales[j].get("archivo"):
                    continue
                e = originales[j]
                p = root / "clap" / carpeta / (Path(e["archivo"]).stem + ".npz")
                z = np.load(p) if p.exists() else None
                if z is None or "dj" not in z:
                    sin_dj += 1
                    continue
                vistos.add(j)
                filas.append(len(V))
                V.append(unit(z["dj"]))
                bpm.append(tempo.get(f"bedrock:{carpeta}:{e['archivo']}", np.nan))
                artista.append(estricto(primer_artista(t.get("artista"))))
                archivos.append(f"{carpeta}/{e['archivo']}")
        sets[carpeta], dj[carpeta] = filas, "John Digweed"
    M = np.stack(V)
    dup = (bib.V @ M.T).max(axis=1) >= COS_DUPLICADO
    return (Fuente("bedrock", "biblioteca", M, np.asarray(bpm, dtype=np.float64), artista, sets, dj,
                   {"originales": len(V), "pistas_sin_original": sin_pareja, "originales_sin_embedding": sin_dj,
                    "temas_de_la_biblioteca_que_son_estos_originales": int(dup.sum())}, fila_archivo=archivos), dup)


def fuente_d7(root: Path, tempo: Dict[str, float], forma: Optional[str] = None) -> Fuente:
    """Temas de D7 validados (límites usables) con vector de borde 60; los discos de menos de 5 temas quedan afuera.
    forma: «por_tema», «continua» o None para las dos."""
    with (root / "sets" / "d7_temas.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    M = np.load(root / "sets" / "d7_clap.npz")["borde_60"].astype(np.float64)
    por_set: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        if not np.isnan(M[i, 0]) and (forma is None or r["forma"] == forma):
            por_set.setdefault(r["set_id"], []).append(i)
    V, bpm, artista, grp, djs, anios, sets, dj = [], [], [], [], [], [], {}, {}
    repetidos, chicos = 0, 0
    for sid, filas in por_set.items():
        filas.sort(key=lambda f: int(rows[f]["posicion"]))
        vistos, dedup = set(), []
        for f in filas:
            ident = identity(rows[f])
            if ident in vistos:
                repetidos += 1
            else:
                vistos.add(ident)
                dedup.append(f)
        if len(dedup) < MIN_TEMAS_D7:
            chicos += 1
            continue
        nuevos = []
        for f in dedup:
            r = rows[f]
            nuevos.append(len(V))
            V.append(M[f])
            bpm.append(tempo.get(f"d7:{f}", np.nan))
            artista.append(estricto(primer_artista(r["artista"])))
            grp.append(r["grupo"])
            djs.append(r["dj"])
            anios.append(anio_de(r["fecha"]) or 0)
        sets[sid], dj[sid] = nuevos, rows[dedup[0]]["dj"]
    nombre = f"d7_{forma}" if forma else "d7"
    return Fuente(nombre, "mezcla", unit(np.stack(V)), np.asarray(bpm, dtype=np.float64), artista, sets, dj,
                  {"temas": len(V), "discos": len(sets), "discos_de_menos_de_5_temas": chicos, "temas_repetidos_en_un_disco": repetidos,
                   "djs": sorted(set(djs))},
                  np.asarray(grp), np.asarray(djs), np.asarray(anios))


def pares_del_set(sets: Dict[str, List[int]], artista: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """(i, j, k del set, peso) de cada par de temas de un set, de artistas distintos. Cada set pesa 1 en total."""
    I, J, K, W, nombres = [], [], [], [], []
    for sid, filas in sets.items():
        par = [(a, b) for p, a in enumerate(filas) for b in filas[p + 1:] if not artista[a] or artista[a] != artista[b]]
        if not par:
            continue
        k = len(nombres)
        nombres.append(sid)
        for a, b in par:
            I.append(a), J.append(b), K.append(k), W.append(1.0 / len(par))
    return np.asarray(I, dtype=int), np.asarray(J, dtype=int), np.asarray(K, dtype=int), np.asarray(W, dtype=np.float64), nombres


def cuantil_pesado(x: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(x, kind="stable")
    x, w = np.asarray(x, dtype=np.float64)[o], np.asarray(w, dtype=np.float64)[o]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q / 100.0, c, x))


def distancia(V: np.ndarray, I: np.ndarray, J: np.ndarray, U: Optional[np.ndarray] = None) -> np.ndarray:
    return 1.0 - np.sum(V[I] * (V if U is None else U)[J], axis=1)


def ic_por_sets(d: np.ndarray, w: np.ndarray, k: np.ndarray, qs: Sequence[float], rng: np.random.Generator,
                n: int = BOOTSTRAP) -> Dict[float, List[float]]:
    """Intervalo del 95 % de cada percentil remuestreando sets enteros."""
    n_sets = int(k.max()) + 1
    por_set = [np.where(k == s)[0] for s in range(n_sets)]
    out: Dict[float, List[float]] = {q: [] for q in qs}
    for _ in range(n):
        idx = np.concatenate([por_set[s] for s in rng.integers(0, n_sets, n_sets)])
        for q in qs:
            out[q].append(cuantil_pesado(d[idx], w[idx], q))
    return {q: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for q, v in out.items()}


def azar_mismo_bpm(bpm_a: np.ndarray, bpm_b: np.ndarray, w: np.ndarray, bpm_pool: np.ndarray, rng: np.random.Generator,
                   grupo_pool: Optional[np.ndarray] = None, tol: float = BPM_TOL, por_par: int = AZAR_POR_PAR
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Para cada par real (con sus dos BPM), `por_par` pares al azar del pool con cada tema a ±tol de su BPM; si hay
    grupos, de grupos distintos. Devuelve (a, b, peso): el peso del par real se reparte entre sus pares al azar."""
    finitos = np.where(np.isfinite(bpm_pool))[0]
    orden = finitos[np.argsort(bpm_pool[finitos], kind="stable")]
    valores = bpm_pool[orden]
    A, B, P = [], [], []
    for x, y, peso in zip(bpm_a, bpm_b, w):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        lo_a, hi_a = np.searchsorted(valores, x - tol), np.searchsorted(valores, x + tol, side="right")
        lo_b, hi_b = np.searchsorted(valores, y - tol), np.searchsorted(valores, y + tol, side="right")
        if hi_a <= lo_a or hi_b <= lo_b:
            continue
        a = orden[rng.integers(lo_a, hi_a, por_par)]
        b = orden[rng.integers(lo_b, hi_b, por_par)]
        ok = a != b
        if grupo_pool is not None:
            ok &= grupo_pool[a] != grupo_pool[b]
        if ok.any():
            A.append(a[ok]), B.append(b[ok]), P.append(np.full(int(ok.sum()), peso / int(ok.sum())))
    if not A:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    return np.concatenate(A), np.concatenate(B), np.concatenate(P)


def auc_mas_cerca(d_real: np.ndarray, w_real: np.ndarray, d_azar: np.ndarray, w_azar: np.ndarray) -> Optional[float]:
    """Probabilidad (pesada) de que un par real esté más cerca que uno al azar."""
    if not len(d_real) or not len(d_azar):
        return None
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(d_real)), np.zeros(len(d_azar))]
    return round(float(roc_auc_score(y, -np.r_[d_real, d_azar], sample_weight=np.r_[w_real, w_azar])), 4)


def resumen(d: np.ndarray, w: np.ndarray) -> Dict[str, float]:
    return {f"p{q}": round(cuantil_pesado(d, w, q), 4) for q in (10, 25, 50, 75)}


def e9_fuente(f: Fuente, V_pool: np.ndarray, bpm_pool: np.ndarray, rng: np.random.Generator,
              grupo_pool: Optional[np.ndarray] = None) -> Tuple[Dict, np.ndarray, np.ndarray]:
    I, J, K, W, nombres = pares_del_set(f.sets, f.artista)
    d = distancia(f.V, I, J)
    a, b, p = azar_mismo_bpm(f.bpm[I], f.bpm[J], W, bpm_pool, rng, grupo_pool)
    d_azar = distancia(V_pool, a, b) if len(a) else np.zeros(0)
    n = len(V_pool)
    a0, b0 = rng.integers(0, n, 20000), rng.integers(0, n, 20000)
    ok = a0 != b0
    if grupo_pool is not None:
        ok &= grupo_pool[a0] != grupo_pool[b0]
    d0 = distancia(V_pool, a0[ok], b0[ok])
    ic = ic_por_sets(d, W, K, PERCENTILES, rng)
    total = sum(len(v) * (len(v) - 1) // 2 for v in f.sets.values())
    out = {"dominio": f.dominio, "sets_con_pares": len(nombres), "pares": int(len(d)), "pares_del_mismo_artista_fuera": total - int(len(d)),
           "djs": len(set(f.dj.get(s, "?") for s in nombres)),
           "mismo_set": resumen(d, W), "ic95_por_sets": {f"p{q}": ic[q] for q in PERCENTILES},
           "azar_mismo_bpm": resumen(d_azar, p) if len(d_azar) else None, "pares_con_bpm": int(np.isfinite(f.bpm[I] + f.bpm[J]).sum()),
           "azar_sin_bpm": resumen(d0, np.ones(len(d0))),
           "auc_frente_a_azar_mismo_bpm": auc_mas_cerca(d, W, d_azar, p)}
    return out, d, W


def cobertura(d: np.ndarray, w: np.ndarray, radio: float) -> float:
    return round(float(w[d <= radio].sum() / w.sum()), 4)


def mezcla_frente_a_originales(root: Path) -> Dict:
    """Los mismos pares de Bedrock (mismo disco y álbum) medidos en el original y en la mezcla: cuánto junta la mezcla."""
    salida = {}
    d_o_tot, d_m_tot = [], []
    for carpeta in CARPETAS_BEDROCK:
        man, originales, mezclas, mb = discos_bedrock(root, carpeta)
        O, X, artista = [], [], []
        for disc in mb["media"]:
            e = mezclas.get(disc["disco"])
            p = root / "clap" / carpeta / (Path(e["archivo"]).stem + ".npz") if e and e.get("archivo") else None
            if p is None or not p.exists():
                continue
            z = np.load(p)
            starts, dur, blocks = z["starts"], float(z["dur"]), z["blocks"]
            for t, (a, b), j in zip(disc["pistas"], bounds([t["ms"] for t in disc["pistas"]]), match_tracks(disc["pistas"], originales)):
                if j is None or not originales[j].get("archivo"):
                    continue
                q = root / "clap" / carpeta / (Path(originales[j]["archivo"]).stem + ".npz")
                zo = np.load(q) if q.exists() else None
                if zo is None or "dj" not in zo:
                    continue
                idx, _ = select_blocks(starts, dur, a, b, 60.0)
                v = track_vector(blocks, idx)
                if v is not None:
                    O.append(unit(zo["dj"])), X.append(v), artista.append(estricto(primer_artista(t.get("artista"))))
        O, X = np.stack(O), np.stack(X)
        I, J, _, W, _ = pares_del_set({"a": list(range(len(O)))}, artista)
        d_o, d_m = distancia(O, I, J), distancia(X, I, J)
        d_o_tot.append(d_o), d_m_tot.append(d_m)
        salida[man.get("titulo")] = {"temas": len(O), "pares": len(d_o), "original_mediana": round(float(np.median(d_o)), 4),
                                     "mezcla_mediana": round(float(np.median(d_m)), 4),
                                     "diferencia_mediana": round(float(np.median(d_o - d_m)), 4),
                                     "fraccion_mas_cerca_en_la_mezcla": round(float(np.mean(d_m < d_o)), 3)}
    d_o, d_m = np.concatenate(d_o_tot), np.concatenate(d_m_tot)
    salida["total"] = {"pares": len(d_o), "original_mediana": round(float(np.median(d_o)), 4), "mezcla_mediana": round(float(np.median(d_m)), 4),
                       "diferencia_mediana": round(float(np.median(d_o - d_m)), 4), "fraccion_mas_cerca_en_la_mezcla": round(float(np.mean(d_m < d_o)), 3),
                       "correlacion": round(float(np.corrcoef(d_o, d_m)[0, 1]), 3)}
    return salida


def bloques_discos(root: Path) -> Dict[str, List[Tuple[str, str, np.ndarray, np.ndarray]]]:
    """(set, grupo, bloques unitarios, inicios): discos enteros de D7 validados por título y los sets crudos de D8."""
    validation = json.loads((root / "sets" / "d7_validacion.json").read_text(encoding="utf-8"))
    out: Dict[str, List[Tuple[str, str, np.ndarray, np.ndarray]]] = {"d7_discos_enteros": [], "d8": []}
    for man_path in sorted((root / "audio").glob("d7_*/manifest.json")):
        man = json.loads(man_path.read_text(encoding="utf-8"))
        if man["forma"] != "continua":
            continue
        folder = man_path.parent.name
        for e in man["entradas"]:
            p = root / "clap" / folder / (Path(e["archivo"]).stem + ".npz") if e.get("archivo") else None
            if p is not None and p.exists() and accepted(validation, folder, e["archivo"], "titulo"):
                z = np.load(p)
                out["d7_discos_enteros"].append((f"{folder}/{e['archivo']}", man["grupo"], unit(z["blocks"]), np.asarray(z["starts"], float)))
    for d in sorted((root / "clap").glob("set_*")):
        for p in d.glob("*.npz"):
            z = np.load(p)
            out["d8"].append((d.name, d.name, unit(z["blocks"]), np.asarray(z["starts"], float)))
    return out


def e9_bloques(discos: List[Tuple[str, str, np.ndarray, np.ndarray]], rng: np.random.Generator, sep: float = SEP_BLOQUES_S,
               max_pares: int = MAX_PARES_BLOQUES) -> Optional[Dict]:
    """Dentro de un disco, pares de bloques separados por `sep` segundos o más (no la continuidad de la mezcla), frente
    a pares de bloques de discos de otra compilación. Sin BPM: los bloques no lo tienen."""
    if len(discos) < 2:
        return None
    d_m, w_m, d_o, w_o = [], [], [], []
    for k, (sid, grupo, B, starts) in enumerate(discos):
        n = len(B)
        a, b = rng.integers(0, n, max_pares * 6), rng.integers(0, n, max_pares * 6)
        ok = np.abs(starts[a] - starts[b]) >= sep
        a, b = a[ok][:max_pares], b[ok][:max_pares]
        if len(a):
            d_m.append(1.0 - np.sum(B[a] * B[b], axis=1)), w_m.append(np.full(len(a), 1.0 / len(a)))
        otros = [x for x in discos if x[1] != grupo]
        if otros:
            elegidos = rng.integers(0, len(otros), max_pares)
            d = np.asarray([1.0 - float(B[rng.integers(0, n)] @ otros[e][2][rng.integers(0, len(otros[e][2]))]) for e in elegidos])
            d_o.append(d), w_o.append(np.full(len(d), 1.0 / len(d)))
    return {"discos": len(discos), "separacion_minima_s": sep,
            "mismo_disco": resumen(np.concatenate(d_m), np.concatenate(w_m)) if d_m else None,
            "otro_disco_otra_compilacion": resumen(np.concatenate(d_o), np.concatenate(w_o)) if d_o else None,
            "auc": auc_mas_cerca(np.concatenate(d_m), np.concatenate(w_m), np.concatenate(d_o), np.concatenate(w_o)) if d_m and d_o else None}


def listas_en_vivo(rec: rc.Recommender, rng: np.random.Generator, top: int = TOP, limite: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Para cada tema sugerible como principal, la distancia CLAP de los `top` de cada lista, con los filtros en vivo:
    CLAP (Parecidos), los más cercanos en BPM y al azar. NaN donde hay menos de `top` candidatos."""
    clap, bpm, azar, cuantos, uids, generos = [], [], [], [], [], []
    principales = [it for it in rec.candidates if it.vec is not None]
    if limite:
        principales = [principales[i] for i in rng.permutation(len(principales))[:limite]]
    for it in principales:
        info = rec.compute({"path": str(it.path or it.original or it.rel), "bpm": it.bpm, "tempo": 1.0, "key": None}, [])[1]
        if info["modo"] != "clap":
            continue
        dist = 1.0 - np.asarray(info["parecido"], dtype=np.float64)
        n = len(dist)
        cuantos.append(n)
        uids.append(it.uid)
        generos.append(it.genre or "")
        comp = info["componentes"].get("bpm")
        por_bpm = np.argsort(-(np.asarray([x if x is not None else -1.0 for x in comp]) + rng.random(n) * 1e-6), kind="stable") if comp else rng.permutation(n)

        def fila(orden: np.ndarray) -> np.ndarray:
            x = np.full(top, np.nan)
            x[:min(top, n)] = dist[orden[:top]]
            return x
        clap.append(fila(np.arange(n))), bpm.append(fila(por_bpm)), azar.append(fila(rng.permutation(n)))
    return {"clap": np.stack(clap), "bpm": np.stack(bpm), "azar": np.stack(azar), "candidatos": np.asarray(cuantos),
            "uids": uids, "generos": generos}


def fraccion_dentro(D: np.ndarray, radio: float) -> Dict[str, float]:
    """D (principales, top): fracción de cada lista dentro del radio, resumida entre principales."""
    con_dato = np.isfinite(D)
    f = np.sum((D <= radio) & con_dato, axis=1) / np.maximum(con_dato.sum(axis=1), 1)
    return {"media": round(float(f.mean()), 4), "p10": round(float(np.percentile(f, 10)), 4), "mediana": round(float(np.median(f)), 4),
            "listas_enteras_dentro": round(float(np.mean(f >= 1.0)), 4), "listas_con_menos_de_la_mitad": round(float(np.mean(f < 0.5)), 4)}


def fraccion_por_grupo(D: np.ndarray, etiquetas: Sequence[str], radio: float, min_n: int = 20) -> Dict[str, Dict[str, float]]:
    """fraccion_dentro por grupo de temas principales (género), de menor a mayor; solo grupos de min_n o más."""
    out: Dict[str, Dict[str, float]] = {}
    et = np.asarray(etiquetas)
    for g in sorted(set(etiquetas)):
        m = et == g
        if g and m.sum() >= min_n:
            r = fraccion_dentro(D[m], radio)
            out[g] = {"n": int(m.sum()), "media": r["media"], "listas_enteras_dentro": r["listas_enteras_dentro"]}
    return dict(sorted(out.items(), key=lambda kv: kv[1]["media"]))


def pureza(f: Fuente, V_pool: np.ndarray, bpm_pool: np.ndarray, rng: np.random.Generator, excluir: Optional[np.ndarray] = None,
           top: int = TOP, solo_sets: Optional[Sequence[str]] = None) -> Dict:
    """Desde cada tema de un set (menos el último), los `top` más parecidos del pool sin lo ya tocado: qué parte son
    temas posteriores de ese set (pureza) y qué parte de esos temas aparece (recuperación), con CLAP, por BPM y al azar
    exacto. El pool tiene primero las filas de la fuente; lo que sigue son otros temas. Cada set pesa igual."""
    n_f, n_pool = len(f.V), len(V_pool)
    por_set: Dict[str, Dict[str, List[float]]] = {}
    compil, dj_epoca, de_bib = [], [], []
    for sid, filas in f.sets.items():
        if solo_sets is not None and sid not in solo_sets:
            continue
        for p, q in enumerate(filas[:-1]):
            hits = np.zeros(n_pool, bool)
            hits[filas[p + 1:]] = True
            fuera = np.zeros(n_pool, bool)
            fuera[filas[: p + 1]] = True
            if excluir is not None:
                fuera |= excluir
            disp = np.where(~fuera)[0]
            if len(disp) < top or not hits[disp].any():
                continue
            s = V_pool[disp] @ f.V[q]
            top_clap = disp[np.argpartition(-s, top - 1)[:top]]
            n_hits = int(hits[disp].sum())
            r = por_set.setdefault(sid, {k: [] for k in ("clap", "bpm", "azar", "clap_rec", "bpm_rec", "azar_rec")})
            r["clap"].append(float(hits[top_clap].mean()))
            r["clap_rec"].append(float(hits[top_clap].sum() / n_hits))
            r["azar"].append(n_hits / len(disp))
            r["azar_rec"].append(min(1.0, top / len(disp)))
            if np.isfinite(f.bpm[q]):
                # BPM redondeado: el tempo estimado cae en valores finos (125,9) que los tags de la biblioteca no
                # tienen, y sin redondear el orden por BPM prefiere a los temas estimados con el mismo método
                gap = np.abs(np.round(bpm_pool[disp]) - np.round(f.bpm[q]))
                gap[np.isnan(gap)] = np.inf
                top_bpm = disp[np.argpartition(gap + rng.random(len(gap)) * 1e-3, top - 1)[:top]]
                r["bpm"].append(float(hits[top_bpm].mean()))
                r["bpm_rec"].append(float(hits[top_bpm].sum() / n_hits))
            if f.fila_grupo is not None:
                en_f = top_clap[top_clap < n_f]
                no_hit = en_f[~hits[en_f]]
                compil.append(float(np.sum(f.fila_grupo[no_hit] == f.fila_grupo[q]) / top))
                cerca = (f.fila_dj[no_hit] == f.fila_dj[q]) & (np.abs(f.fila_anio[no_hit] - f.fila_anio[q]) <= 2) & (f.fila_grupo[no_hit] != f.fila_grupo[q])
                dj_epoca.append(float(np.sum(cerca) / top))
            if n_pool > n_f:
                de_bib.append(float(np.mean(top_clap >= n_f)))
    if not por_set:
        return {"sets": 0}
    nombres = list(por_set)

    def por_sets(k: str) -> np.ndarray:
        return np.asarray([np.mean(por_set[s][k]) if por_set[s][k] else np.nan for s in nombres])

    out: Dict[str, object] = {"sets": len(nombres), "consultas": sum(len(v["clap"]) for v in por_set.values())}
    for k in ("clap", "bpm", "azar", "clap_rec", "bpm_rec", "azar_rec"):
        x = por_sets(k)
        x = x[np.isfinite(x)]
        if len(x):
            b = [float(np.mean(x[rng.integers(0, len(x), len(x))])) for _ in range(BOOTSTRAP)]
            out[k] = {"media": round(float(x.mean()), 4), "ic95_por_sets": [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]}
    if compil:
        out["de_la_misma_compilacion_en_los_30_sin_ese_set"] = round(float(np.mean(compil)), 4)
        out["del_mismo_dj_y_epoca_en_los_30"] = round(float(np.mean(dj_epoca)), 4)
    if de_bib:
        out["de_la_biblioteca_en_los_30"] = round(float(np.mean(de_bib)), 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="E9 y E10: radio de zona y precisión de la lista de Parecidos")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--indice", default=str(rc.DEFAULT_INDEX))
    ap.add_argument("--d3", default=str(DEFAULT_D3))
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--principales", type=int, default=None, help="Solo N temas principales en E10a (prueba chica)")
    ap.add_argument("--salida", default=None)
    ap.add_argument("--low-priority", action="store_true")
    args = ap.parse_args()
    if args.low_priority:
        from src.v4.pipeline.extract_representations import lower_priority
        lower_priority()
    t0 = time.perf_counter()
    root = Path(args.root)
    rng = np.random.default_rng(args.semilla)
    _, items = rc.load_index(Path(args.indice))
    bib = cargar_biblioteca(items)
    tempo = cargar_tempo(root / "sets" / "tempo_temas.csv")
    print(f"biblioteca {bib.V.shape}, tempo de {len(tempo)} temas", flush=True)
    d1 = fuente_d1(root / "sets" / "d1.csv", bib)
    d3 = fuente_d3(Path(args.d3), bib)
    bedrock, dup = fuente_bedrock(root, tempo, bib)
    d7_pt, d7_co, d7_all = (fuente_d7(root, tempo, f) for f in ("por_tema", "continua", None))
    res: Dict[str, object] = {"semilla": args.semilla, "biblioteca": {"temas": len(bib.uids)},
                              "curado": {f.nombre: f.curado for f in (d3, bedrock, d1, d7_pt, d7_co)}}
    res["e9"], crudos = {}, {}
    for f in (d3, bedrock, d1):
        res["e9"][f.nombre], *crudos[f.nombre] = e9_fuente(f, bib.V, bib.bpm, rng)
    for f in (d7_pt, d7_co):
        res["e9"][f.nombre], *crudos[f.nombre] = e9_fuente(f, f.V, f.bpm, rng, f.fila_grupo)
    res["e9_mezcla_frente_a_originales"] = mezcla_frente_a_originales(root)
    bloques = bloques_discos(root)
    res["e9_bloques"] = {k: e9_bloques(v, rng) for k, v in bloques.items()}
    print(f"E9 listo ({time.perf_counter() - t0:.0f} s)", flush=True)

    radios = {n: {f"p{q}": round(cuantil_pesado(crudos[n][0], crudos[n][1], q), 4) for q in PERCENTILES} for n in ("d3_biblioteca", "bedrock", "d1")}
    res["radios"] = radios
    res["cobertura_cruzada"] = {f"radio_de_{a}": {f"p{q}_en_{b}": cobertura(crudos[b][0], crudos[b][1], radios[a][f"p{q}"])
                                                   for b in radios if b != a for q in PERCENTILES} for a in radios}

    from tools.traktor_live.sugerencias_explorer import entries_by_path
    from tools.traktor_live.sugerencias_itunes import find_collection
    rec = rc.Recommender(items, entries_by_path(find_collection()), n=10 ** 6, taxonomy=rc.load_taxonomy()[0])
    listas = listas_en_vivo(rec, rng, limite=args.principales)
    print(f"Listas en vivo: {len(listas['clap'])} temas principales ({time.perf_counter() - t0:.0f} s)", flush=True)
    res["e10_listas"] = {"principales": int(len(listas["clap"])), "candidatos_mediana": float(np.median(listas["candidatos"])),
                         "con_menos_de_30_candidatos": int(np.sum(listas["candidatos"] < TOP)),
                         "distancia_mediana_por_puesto": {f"puesto_{p}": {m: round(float(np.nanmedian(listas[m][:, p - 1])), 4) for m in ("clap", "bpm", "azar")} for p in (1, 10, 30)}}
    res["e10_fraccion"] = {n: {p: {"radio": r, **{m: fraccion_dentro(listas[m], r) for m in ("clap", "bpm", "azar")}} for p, r in rs.items()} for n, rs in radios.items()}
    tocados = {bib.uids[i] for filas in d1.sets.values() for i in filas}
    m_toc = np.asarray([u in tocados for u in listas["uids"]])
    res["e10_fraccion_temas_que_toco_gabriel"] = {"principales": int(m_toc.sum()), **{p: fraccion_dentro(listas["clap"][m_toc], r) for p, r in radios["d3_biblioteca"].items()}}
    res["e10_por_genero"] = {p: fraccion_por_grupo(listas["clap"], listas["generos"], r) for p, r in radios["d3_biblioteca"].items() if p != "p50"}

    con_bib = np.r_[np.zeros(len(bedrock.V), bool), dup]
    res["e10_pureza"] = {
        "d3_biblioteca": pureza(d3, bib.V, bib.bpm, rng),
        "bedrock_con_biblioteca": pureza(bedrock, np.vstack([bedrock.V, bib.V]), np.r_[bedrock.bpm, bib.bpm], rng, excluir=con_bib),
        "d1": pureza(d1, bib.V, bib.bpm, rng),
        "d7_por_tema_solo_d7": pureza(d7_all, d7_all.V, d7_all.bpm, rng, solo_sets=list(d7_pt.sets)),
        "d7_continua_solo_d7": pureza(d7_all, d7_all.V, d7_all.bpm, rng, solo_sets=list(d7_co.sets)),
        "control_d7_con_biblioteca": pureza(d7_all, np.vstack([d7_all.V, bib.V]), np.r_[d7_all.bpm, bib.bpm], rng),
    }
    res["segundos"] = round(time.perf_counter() - t0, 1)
    out = Path(args.salida) if args.salida else root / "sets" / "zona_sets.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("curado", "radios", "e9", "e10_listas")}, ensure_ascii=False)[:3000])
    print(f"[OK] -> {out} ({res['segundos']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
