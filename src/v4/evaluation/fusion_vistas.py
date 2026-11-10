"""
PURPOSE: Unidad 1 de docs/plans/PROMPT_FUSION_Y_CONCEPTOS.md (plan de playlists en vivo, sección 12): la fusión de
         vistas de Parecidos (P11), el consenso entre vistas (P13) y la ecuación de factores aprendida de sets
         profesionales (P12), con el tempo controlado al medir los modelos (E14). Reutiliza las fuentes, los pares y el
         azar del mismo BPM de zona_sets.py. Mide: no toca el índice ni el recomendador.
         Vistas, en el espacio de la biblioteca y alineadas por track_uid con indice_recomendacion.json:
           clap        CLAP crudo (lo que usa hoy el recomendador);
           clap_c      CLAP centrado (menos la media de la biblioteca, normalizado);
           maest_c     MAEST capa 7 centrado;
           estilos     los 400 estilos Discogs de MAEST (coseno);
           estilos_c   centrados; estilos_st: centrados y sin las 2 direcciones que predicen el tempo (ridge de los
                       estilos al BPM y al BPM al cuadrado, en la biblioteca): la variante con el tempo descontado (E14).
         Fusión (P11): cada coseno en desvíos de su distribución entre pares de la biblioteca (z), sumados con pesos. Es
         el producto punto de los vectores de cada vista escalados por sqrt(peso / desvío) y concatenados, más una
         constante: ordena igual, y así se mide con las funciones de zona_sets y así entraría al índice versión 4.
         Tabla de vistas, cada fila con el tempo controlado (E14): acierto en 30 del tema siguiente en D1 (pool completo,
         a ±8 y a ±4 BPM; discordantes con CLAP crudo y p binomial; diferencia con intervalo por sets); AUC «mismo set»
         frente a azar del mismo BPM (±4) en D3 × biblioteca (diferencia con CLAP crudo e intervalo por sets), en los
         originales de Bedrock y en D1; pureza en Bedrock y D1; Spearman de cada vista con la diferencia de BPM.
         P12: regresión logística sobre los factores con las fórmulas del recomendador (vistas en z, bpm, clave, genero),
         pares de un mismo set profesional (D3 × biblioteca, Bedrock) que pasan los filtros en vivo frente a pares al
         azar dentro de esos filtros (BPM ±20; tonalidad a ±2 semitonos con corte 0,60), cada DJ fuera del
         entrenamiento por turno. Pesos, AUC fuera de muestra, calibración y comparación con pesos iguales y con CLAP
         crudo. D1 se informa aparte y no ajusta nada (AUC y acierto en 30 con los filtros en vivo).
         P13: cuántas vistas ponen al candidato entre sus 30 primeros de la biblioteca, y qué parte de los pares de cada
         nivel de consenso es de un mismo set.
           python src/v4/evaluation/fusion_vistas.py --root "D:/TRAKTOR ML datos/dj_sets" --low-priority
         MAEST de los originales de Bedrock: tools/dj_sets/maest_originales.py (<root>/maest/<carpeta>/*.npz).
CHANGELOG:
  - 2026-11-09: Creación (unidad 1 de PROMPT_FUSION_Y_CONCEPTOS.md).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from src.v4.common import harmonic  # noqa: E402
from src.v4.evaluation import zona_sets as zs  # noqa: E402
from tools.dj_sets.piloto import unit  # noqa: E402
from tools.traktor_live import recomendador as rc  # noqa: E402

REPS_DIR = REPO_ROOT / "artifacts" / "v4" / "datasets" / "musica" / "representations"
REPS = {"clap": "clap_full", "maest": "maesthf_full_l7", "estilos": "maesthf_full_styles"}
TOP = 30
BOOT = 2000                      # remuestreos de sets para el acierto en D1
BOOT_AUC = 1000                  # remuestreos de sets para la diferencia de AUC en D3
BOOT_DJ = 500                    # remuestreos de DJs para la ecuación
K_NEG = 20                       # pares al azar por par real en la ecuación
RIDGE_TEMPO = 1.0
PARES_TEMPO = 200_000
LR_C = 1.0                       # regularización fija de antemano; pesos de muestra con media 1
VISTAS_P11 = ("clap_c", "maest_c", "estilos")
FACTORES = VISTAS_P11 + ("bpm", "clave", "genero")
CAMELOT = [f"{n}{m}" for m in "AB" for n in range(1, 13)]
METODOS: Dict[str, Dict[str, float]] = {
    "clap crudo": {"clap": 1.0},
    "clap centrado": {"clap_c": 1.0},
    "maest centrado": {"maest_c": 1.0},
    "estilos": {"estilos": 1.0},
    "estilos centrados": {"estilos_c": 1.0},
    "estilos sin tempo": {"estilos_st": 1.0},
    "z: clap_c + maest_c": {"clap_c": 1.0, "maest_c": 1.0},
    "z: clap_c + estilos": {"clap_c": 1.0, "estilos": 1.0},
    "z: las tres": {"clap_c": 1.0, "maest_c": 1.0, "estilos": 1.0},
    "z: las tres, estilos centrados": {"clap_c": 1.0, "maest_c": 1.0, "estilos_c": 1.0},
    "z: las tres, estilos sin tempo": {"clap_c": 1.0, "maest_c": 1.0, "estilos_st": 1.0},
}
CONSENSO = {"consenso: mínimo de percentiles, clap_c y maest_c": ("clap_c", "maest_c"),
            "consenso: mínimo de percentiles, las tres": ("clap_c", "maest_c", "estilos")}


# ---------------------------------------------------------------------------
# Vistas
# ---------------------------------------------------------------------------

@dataclass
class Vista:
    """Transformación de una representación cruda a la vista: normalizar, restar la media de la biblioteca y quitar
    una base ortonormal Q (columnas), y normalizar de nuevo. Se ajusta en la biblioteca y se aplica igual a otros temas."""
    rep: str
    mu: Optional[np.ndarray] = None
    Q: Optional[np.ndarray] = None

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = unit(X)
        if self.mu is not None:
            X = X - self.mu
        if self.Q is not None:
            X = X - (X @ self.Q) @ self.Q.T
        return unit(X)


def direcciones_de_tempo(X: np.ndarray, bpm: np.ndarray, alfa: float = RIDGE_TEMPO) -> np.ndarray:
    """Base ortonormal (d, 2) de las direcciones de X que predicen el tempo: ridge de X centrado a [b, b² − media]
    (b = BPM estandarizado), columnas ortonormalizadas. Solo los temas con BPM."""
    ok = np.isfinite(bpm)
    b = (bpm[ok] - bpm[ok].mean()) / bpm[ok].std()
    phi = np.c_[b, b ** 2 - np.mean(b ** 2)]
    S = X[ok] - X[ok].mean(axis=0)
    W = np.linalg.solve(S.T @ S + alfa * np.eye(S.shape[1]), S.T @ phi)
    Q, _ = np.linalg.qr(W)
    return Q


def construir_vistas(crudos: Dict[str, np.ndarray], bpm: np.ndarray) -> Dict[str, Vista]:
    """Las vistas, ajustadas en la biblioteca (crudos alineados con bpm)."""
    mu = {k: unit(v).mean(axis=0) for k, v in crudos.items()}
    q = direcciones_de_tempo(unit(crudos["estilos"]) - mu["estilos"], bpm)
    return {"clap": Vista("clap"), "clap_c": Vista("clap", mu["clap"]), "maest_c": Vista("maest", mu["maest"]),
            "estilos": Vista("estilos"), "estilos_c": Vista("estilos", mu["estilos"]),
            "estilos_st": Vista("estilos", mu["estilos"], q)}


def espacio(vistas: Dict[str, Vista], crudos: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: v(crudos[v.rep]) for k, v in vistas.items() if v.rep in crudos}


def escala_z(E: Dict[str, np.ndarray]) -> Dict[str, Tuple[float, float]]:
    """(media, desvío) del coseno entre pares distintos de la biblioteca, por vista (desvío poblacional)."""
    out = {}
    for k, V in E.items():
        S = V @ V.T
        n = len(V)
        d = np.diag(S)
        m = (S.sum() - d.sum()) / (n * (n - 1))
        m2 = ((S ** 2).sum() - (d ** 2).sum()) / (n * (n - 1))
        out[k] = (float(m), float(np.sqrt(max(m2 - m * m, 1e-18))))
    return out


def concatenar(E: Dict[str, np.ndarray], pesos: Dict[str, float], escala: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """Vectores de cada vista por sqrt(peso / desvío), concatenados: el producto punto de dos temas es la suma pesada
    de los z de sus cosenos más la constante Σ peso·media/desvío, así que ordena igual."""
    return np.hstack([np.sqrt(w / escala[k][1]) * E[k] for k, w in pesos.items() if w > 0])


def suma_z(E: Dict[str, np.ndarray], i: np.ndarray, j: np.ndarray, pesos: Dict[str, float],
           escala: Dict[str, Tuple[float, float]], E_j: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    """Suma pesada de los z de los cosenos de los pares (i de E, j de E_j o de E)."""
    E_j = E if E_j is None else E_j
    return sum(w * (dot_pares(E[k], i, E_j[k], j) - escala[k][0]) / escala[k][1] for k, w in pesos.items() if w > 0)


def percentil_filas(S: np.ndarray) -> np.ndarray:
    """Percentil de cada columna dentro de su fila, sin la diagonal (1 = el más parecido)."""
    S = np.array(S, dtype=np.float64, copy=True)
    np.fill_diagonal(S, -np.inf)
    r = np.argsort(np.argsort(S, axis=1), axis=1).astype(np.float64)
    return r / (S.shape[1] - 1)


# ---------------------------------------------------------------------------
# Medidas de la tabla de vistas
# ---------------------------------------------------------------------------

def acierto_siguiente(sets: Dict[str, List[int]], M: np.ndarray, bpm: Optional[np.ndarray] = None,
                      tol: Optional[float] = None, top: int = TOP) -> np.ndarray:
    """(set, acierto, candidatos) de cada transición: el tema siguiente entre los `top` de la fila del actual, sin lo ya
    tocado. Con tol, solo transiciones con los dos BPM a ±tol y el pool limitado a ±tol del actual."""
    out = []
    for k, filas in enumerate(sets.values()):
        for p in range(len(filas) - 1):
            q, nx = filas[p], filas[p + 1]
            s = np.array(M[q], dtype=np.float64, copy=True)
            s[filas[: p + 1]] = -np.inf
            if tol is not None:
                if not (np.isfinite(bpm[q]) and np.isfinite(bpm[nx]) and abs(bpm[nx] - bpm[q]) <= tol):
                    continue
                s[~(np.abs(bpm - bpm[q]) <= tol)] = -np.inf
            out.append((k, int((s > s[nx]).sum() < top), int(np.isfinite(s).sum())))
    return np.asarray(out, dtype=int).reshape(-1, 3)


def comparar(h: np.ndarray, base: np.ndarray, semilla: int = 0, n: int = BOOT) -> Dict[str, object]:
    """Acierto de h frente a base en las mismas transiciones: discordantes (solo h / solo base), p binomial exacta y la
    diferencia con intervalo del 95 % remuestreando sets enteros (los mismos remuestreos para cualquier h)."""
    from scipy.stats import binomtest
    if len(h) != len(base) or not np.array_equal(h[:, 0], base[:, 0]):
        raise ValueError("h y base tienen que ser las mismas transiciones")
    a = int(((h[:, 1] == 1) & (base[:, 1] == 0)).sum())
    b = int(((h[:, 1] == 0) & (base[:, 1] == 1)).sum())
    rng = np.random.default_rng(semilla)
    n_sets = int(base[:, 0].max()) + 1 if len(base) else 0
    por_set = [np.where(base[:, 0] == s)[0] for s in range(n_sets)]
    difs = []
    for _ in range(n):
        idx = np.concatenate([por_set[s] for s in rng.integers(0, n_sets, n_sets)])
        difs.append(h[idx, 1].mean() - base[idx, 1].mean())
    return {"acierto": round(float(h[:, 1].mean()), 4), "solo_variante": a, "solo_base": b,
            "p": round(float(binomtest(a, a + b, 0.5).pvalue), 4) if a + b else 1.0,
            "diferencia": round(float(h[:, 1].mean() - base[:, 1].mean()), 4),
            "ic95_por_sets": [round(float(np.percentile(difs, 2.5)), 4), round(float(np.percentile(difs, 97.5)), 4)]}


def dot_pares(A: np.ndarray, i: np.ndarray, B: np.ndarray, j: np.ndarray, bloque: int = 20000) -> np.ndarray:
    """Producto punto fila a fila de A[i] y B[j], por bloques (cientos de miles de pares sin copiar las filas enteras)."""
    out = np.empty(len(i), dtype=np.float64)
    for s in range(0, len(i), bloque):
        out[s:s + bloque] = np.einsum("ij,ij->i", A[i[s:s + bloque]], B[j[s:s + bloque]])
    return out


def auc_pesado(pos: np.ndarray, w_pos: np.ndarray, neg: np.ndarray, w_neg: np.ndarray) -> Optional[float]:
    """Probabilidad (pesada) de que un positivo puntúe más que un negativo."""
    if not len(pos) or not len(neg):
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg], sample_weight=np.r_[w_pos, w_neg]))


@dataclass
class NuloBpm:
    """Pares reales de una fuente (I, J del set k con peso W) y su azar del mismo BPM en la biblioteca (a, b, p)."""
    I: np.ndarray
    J: np.ndarray
    K: np.ndarray
    W: np.ndarray
    a: np.ndarray
    b: np.ndarray
    p: np.ndarray


def nulo_bpm(f: zs.Fuente, bpm_pool: np.ndarray, semilla: int) -> NuloBpm:
    I, J, K, W, _ = zs.pares_del_set(f.sets, f.artista)
    a, b, p = zs.azar_mismo_bpm(f.bpm[I], f.bpm[J], W, bpm_pool, np.random.default_rng(semilla))
    return NuloBpm(I, J, K, W, a, b, p)


def auc_con_nulo(n: NuloBpm, real: np.ndarray, azar: np.ndarray) -> Optional[float]:
    return auc_pesado(real, n.W, azar, n.p)


def auc_dif_por_sets(n: NuloBpm, real: np.ndarray, azar: np.ndarray, real_base: np.ndarray, azar_base: np.ndarray,
                     semilla: int = 1, veces: int = BOOT_AUC) -> List[float]:
    """Intervalo del 95 % de AUC(variante) − AUC(base) remuestreando sets de pares reales; el azar queda fijo."""
    rng = np.random.default_rng(semilla)
    n_sets = int(n.K.max()) + 1
    por_set = [np.where(n.K == s)[0] for s in range(n_sets)]
    difs = []
    for _ in range(veces):
        ii = np.concatenate([por_set[s] for s in rng.integers(0, n_sets, n_sets)])
        difs.append(auc_pesado(real[ii], n.W[ii], azar, n.p) - auc_pesado(real_base[ii], n.W[ii], azar_base, n.p))
    return [round(float(np.percentile(difs, 2.5)), 4), round(float(np.percentile(difs, 97.5)), 4)]


def spearman_tempo(C: np.ndarray, bpm: np.ndarray, semilla: int = 0, n: int = PARES_TEMPO) -> float:
    """Spearman entre el parecido y la diferencia absoluta de BPM en n pares al azar (negativo = la vista lleva tempo)."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(semilla)
    a, b = rng.integers(0, len(C), n), rng.integers(0, len(C), n)
    ok = (a != b) & np.isfinite(bpm[a]) & np.isfinite(bpm[b])
    return float(spearmanr(dot_pares(C, a[ok], C, b[ok]), np.abs(bpm[a[ok]] - bpm[b[ok]])).statistic)


def auc_entre_sets(f: zs.Fuente, C: np.ndarray, consecutivos: bool = False) -> Optional[float]:
    """Control sin cambio de dominio: pares de un mismo set (o solo los consecutivos) frente a pares de sets distintos de
    la misma fuente, todos con el mismo origen de audio. Artistas distintos; cada set pesa igual en los dos lados."""
    sets = list(f.sets.values())
    pos, wp, neg, wn = [], [], [], []
    for filas in sets:
        par = [(a, b) for p, a in enumerate(filas) for b in (filas[p + 1:p + 2] if consecutivos else filas[p + 1:])
               if not f.artista[a] or f.artista[a] != f.artista[b]]
        if par:
            i, j = np.asarray(par).T
            pos.append(dot_pares(C, i, C, j)), wp.append(np.full(len(i), 1.0 / len(i)))
    for x in range(len(sets)):
        for y in range(x + 1, len(sets)):
            i, j = np.repeat(sets[x], len(sets[y])), np.tile(sets[y], len(sets[x]))
            ok = np.asarray([not f.artista[a] or f.artista[a] != f.artista[b] for a, b in zip(i, j)])
            if ok.any():
                neg.append(dot_pares(C, i[ok], C, j[ok])), wn.append(np.full(int(ok.sum()), 1.0 / int(ok.sum())))
    if not pos or not neg:
        return None
    return auc_pesado(np.concatenate(pos), np.concatenate(wp), np.concatenate(neg), np.concatenate(wn))


def subfuente(f: zs.Fuente, keep: np.ndarray) -> zs.Fuente:
    """La fuente con solo las filas de keep (los sets conservan el orden)."""
    idx = np.where(keep)[0]
    nuevo = {int(o): i for i, o in enumerate(idx)}
    sets = {s: [nuevo[i] for i in filas if i in nuevo] for s, filas in f.sets.items()}
    return dataclasses.replace(f, V=f.V[idx], bpm=f.bpm[idx], artista=[f.artista[i] for i in idx], sets=sets,
                               fila_archivo=None if f.fila_archivo is None else [f.fila_archivo[i] for i in idx])


def maest_de_bedrock(root: Path, bed: zs.Fuente) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(l7, estilos, hay) de cada fila de Bedrock desde <root>/maest/<carpeta>/<nombre>.npz."""
    l7 = np.full((len(bed.V), 768), np.nan)
    st = np.full((len(bed.V), 400), np.nan)
    for i, clave in enumerate(bed.fila_archivo or []):
        carpeta, archivo = clave.split("/", 1)
        p = root / "maest" / carpeta / (Path(archivo).stem + ".npz")
        if p.exists():
            z = np.load(p)
            l7[i], st[i] = z["l7"], z["styles"]
    hay = np.isfinite(l7).all(axis=1) & np.isfinite(st).all(axis=1)
    return l7, st, hay


# ---------------------------------------------------------------------------
# Factores de la ecuación (las fórmulas del recomendador, vectorizadas)
# ---------------------------------------------------------------------------

@dataclass
class Lado:
    """Temas de un espacio con lo que necesitan los factores: vistas (unitarias), BPM en vivo, tonalidad (índice en
    CAMELOT, −1 sin tonalidad), género (índice en Tablas.generos, −1 sin género) y artista normalizado."""
    E: Dict[str, np.ndarray]
    bpm: np.ndarray
    clave: np.ndarray
    genero: np.ndarray
    artista: List[str]


@dataclass
class Tablas:
    escala: Dict[str, Tuple[float, float]]
    clave: np.ndarray            # (24, 24) best_key_shift con la penalización por defecto
    genero: np.ndarray           # (G, G) Taxonomy.compare; nan = no cuenta
    generos: List[Tuple[str, Optional[str]]]


def indice_clave(k: Optional[str]) -> int:
    return CAMELOT.index(k) if k in CAMELOT else -1


def tabla_claves() -> np.ndarray:
    return np.asarray([[harmonic.best_key_shift(a, b)[0] for b in CAMELOT] for a in CAMELOT], dtype=np.float64)


def tabla_generos(generos: Sequence[Tuple[str, Optional[str]]], taxonomy) -> np.ndarray:
    T = np.full((len(generos), len(generos)), np.nan)
    for i, a in enumerate(generos):
        for j, b in enumerate(generos):
            if taxonomy is not None:
                v = taxonomy.compare(a, b)
            else:
                v = 1.0 if a[0] == b[0] else 0.0
            T[i, j] = np.nan if v is None else float(v)
    return T


def factores(a: Lado, i: int, pool: Lado, t: Tablas) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Factores del tema i de `a` frente a cada tema de `pool` y la máscara de los filtros en vivo (BPM a ±20 si el
    principal tiene BPM; tonalidad conocida y compatible, best_key_shift ≥ 0,60, si el principal tiene tonalidad), con
    las fórmulas de recomendador.Recommender.compute. Vistas en z; «clap» es el coseno crudo; nan = no cuenta."""
    out: Dict[str, np.ndarray] = {}
    for v, V in a.E.items():
        if v in pool.E:
            cos = pool.E[v] @ V[i]
            out[v] = cos if v == "clap" else (cos - t.escala[v][0]) / t.escala[v][1]
    n = len(pool.bpm)
    ok = np.ones(n, dtype=bool)
    b0 = a.bpm[i]
    if np.isfinite(b0):
        d = np.abs(pool.bpm - b0)
        out["bpm"] = np.where(np.isfinite(d), np.maximum(0.0, 1.0 - d / rc.BPM_RANGE), np.nan)
        ok &= np.isfinite(d) & (d <= rc.BPM_RANGE)
    else:
        out["bpm"] = np.full(n, np.nan)
    k0 = int(a.clave[i])
    if k0 >= 0:
        sc = np.where(pool.clave >= 0, t.clave[k0, np.maximum(pool.clave, 0)], np.nan)
        out["clave"] = sc
        ok &= (pool.clave >= 0) & (np.nan_to_num(sc, nan=-1.0) >= rc.KEY_MIN)
    else:
        out["clave"] = np.full(n, np.nan)
    g0 = int(a.genero[i])
    out["genero"] = (np.where(pool.genero >= 0, t.genero[g0, np.maximum(pool.genero, 0)], np.nan) if g0 >= 0
                     else np.full(n, np.nan))
    return out, ok


@dataclass
class Pares:
    """Pares (principal, candidato) para la ecuación: y = 1 si son del mismo set, 0 si son azar dentro de los filtros;
    w = peso (cada set suma 1 en positivos y 1 en negativos); X = factores; consenso = vistas que ponen al candidato
    entre sus TOP primeros de la biblioteca."""
    X: Dict[str, np.ndarray]
    y: np.ndarray
    w: np.ndarray
    grupo: np.ndarray
    fuente: np.ndarray
    consenso: np.ndarray
    resumen: Dict[str, object]

    @staticmethod
    def juntar(partes: Sequence["Pares"]) -> "Pares":
        claves = sorted(set.intersection(*[set(p.X) for p in partes]))
        return Pares({k: np.concatenate([p.X[k] for p in partes]) for k in claves},
                     *(np.concatenate([getattr(p, c) for p in partes]) for c in ("y", "w", "grupo", "fuente", "consenso")),
                     {"partes": [p.resumen for p in partes]})


def pares_ecuacion(nombre: str, f: zs.Fuente, ancla: Lado, pool: Lado, t: Tablas, rng: np.random.Generator,
                   mismo_espacio: bool, excluir_pool: Optional[np.ndarray] = None, k_neg: int = K_NEG) -> Pares:
    """Pares de un mismo set de `f` (filas de `ancla`; el principal es el que sonó antes, artistas distintos) que pasan
    los filtros en vivo, y por cada uno k_neg candidatos al azar del pool que también los pasan (sin el principal, sin
    temas de ese set, sin el mismo artista y sin excluir_pool). Cada set pesa 1 en positivos y 1 en negativos."""
    I, J, K, W, nombres = zs.pares_del_set(f.sets, f.artista)
    excl = np.zeros(len(pool.bpm), dtype=bool) if excluir_pool is None else excluir_pool.copy()
    vistas = [v for v in VISTAS_P11 if v in ancla.E and v in pool.E]
    filas: Dict[str, list] = {k: [] for k in ("i", "set", "y", "w", "cons")}
    X: Dict[str, list] = {}
    fuera = 0
    for i in np.unique(I):
        fp, okp = factores(ancla, int(i), ancla, t) if not mismo_espacio else (None, None)
        fn, okn = factores(ancla, int(i), pool, t)
        if mismo_espacio:
            fp, okp = fn, okn
        mio = excl.copy()
        if mismo_espacio:
            mio[int(i)] = True
        orden = {v: np.sort(fn[v][~mio]) for v in vistas}

        def consenso(valores: Dict[str, np.ndarray]) -> np.ndarray:
            """Vistas en las que el candidato quedaría entre los TOP primeros del pool (sin el principal)."""
            return np.sum([(len(orden[v]) - np.searchsorted(orden[v], valores[v], side="right")) < TOP for v in vistas],
                          axis=0).astype(int)

        for k in np.where(I == i)[0]:
            j = int(J[k])
            if not okp[j]:
                fuera += 1
                continue
            disp = okn & ~mio
            if mismo_espacio:
                disp[f.sets[nombres[K[k]]]] = False
            if ancla.artista[int(i)]:
                disp &= np.asarray(pool.artista) != ancla.artista[int(i)]
            cand = np.where(disp)[0]
            if not len(cand):
                fuera += 1
                continue
            neg = rng.choice(cand, size=k_neg, replace=True)
            for c in fn:
                X.setdefault(c, []).append(np.r_[fp[c][j], fn[c][neg]])
            cons = consenso({v: fn[v][neg] for v in vistas})
            cons_pos = int(consenso({v: np.asarray([fp[v][j]]) for v in vistas})[0])
            filas["i"].append(np.full(k_neg + 1, i))
            filas["set"].append(np.full(k_neg + 1, K[k]))
            filas["y"].append(np.r_[1, np.zeros(k_neg, dtype=int)])
            filas["w"].append(np.r_[W[k], np.full(k_neg, W[k] / k_neg)])
            filas["cons"].append(np.r_[cons_pos, cons])
    if not filas["y"]:
        raise ValueError(f"{nombre}: ningún par pasa los filtros")
    y = np.concatenate(filas["y"])
    w = np.concatenate(filas["w"])
    s = np.concatenate(filas["set"])
    for k in np.unique(s):                       # cada set vuelve a sumar 1 en positivos y 1 en negativos
        for lab in (0, 1):
            m = (s == k) & (y == lab)
            w[m] /= w[m].sum()
    grupo = np.asarray([zs.estricto(f.dj.get(nombres[k], "?")) or "?" for k in s])
    return Pares({c: np.concatenate(v) for c, v in X.items()}, y, w, grupo, np.full(len(y), nombre),
                 np.concatenate(filas["cons"]),
                 {"fuente": nombre, "pares_del_set": int(len(I)), "pares_que_pasan_los_filtros": int(y.sum()),
                  "pares_fuera_por_filtros": fuera, "sets": int(len(np.unique(s))), "djs": int(len(np.unique(grupo))),
                  "negativos_por_par": k_neg})


# ---------------------------------------------------------------------------
# La ecuación (P12)
# ---------------------------------------------------------------------------

def estandarizar(X: Dict[str, np.ndarray], cols: Sequence[str], ref: np.ndarray, w: np.ndarray,
                 stats: Optional[Dict[str, Tuple[float, float]]] = None) -> Tuple[np.ndarray, Dict[str, Tuple[float, float]]]:
    """Columnas en desvíos de los pares de referencia (los negativos, pesados); nan → 0: el factor no cuenta."""
    if stats is None:
        stats = {}
        for c in cols:
            x, ww = X[c][ref], w[ref]
            ok = np.isfinite(x)
            m = float(np.average(x[ok], weights=ww[ok])) if ok.any() else 0.0
            sd = float(np.sqrt(np.average((x[ok] - m) ** 2, weights=ww[ok]))) if ok.any() else 1.0
            stats[c] = (m, sd if sd > 1e-12 else 1.0)
    Z = np.column_stack([(X[c] - stats[c][0]) / stats[c][1] for c in cols])
    Z[~np.isfinite(Z)] = 0.0
    return Z, stats


def ajustar(Z: np.ndarray, y: np.ndarray, w: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=LR_C, max_iter=2000).fit(Z, y, sample_weight=w * len(w) / w.sum())


def validar_por_grupo(P: Pares, cols: Sequence[str], idx: Optional[np.ndarray] = None) -> np.ndarray:
    """Probabilidad fuera de muestra de cada par: el modelo de cada grupo (DJ) se entrena sin ese grupo."""
    idx = np.arange(len(P.y)) if idx is None else idx
    out = np.full(len(P.y), np.nan)
    X = {c: P.X[c] for c in cols}
    for g in np.unique(P.grupo[idx]):
        te = idx[P.grupo[idx] == g]
        tr = idx[P.grupo[idx] != g]
        if len(np.unique(P.y[tr])) < 2:
            continue
        Ztr, st = estandarizar({c: X[c][tr] for c in cols}, cols, P.y[tr] == 0, P.w[tr])
        Zte, _ = estandarizar({c: X[c][te] for c in cols}, cols, P.y[te] == 0, P.w[te], st)
        out[te] = ajustar(Ztr, P.y[tr], P.w[tr]).predict_proba(Zte)[:, 1]
    return out


def auc_pares(P: Pares, s: np.ndarray, m: Optional[np.ndarray] = None) -> Optional[float]:
    m = np.ones(len(P.y), dtype=bool) if m is None else m
    m = m & np.isfinite(s)
    pos, neg = m & (P.y == 1), m & (P.y == 0)
    return auc_pesado(s[pos], P.w[pos], s[neg], P.w[neg])


def auc_dif_por_grupos(P: Pares, s: np.ndarray, base: np.ndarray, semilla: int = 0, veces: int = BOOT_DJ) -> List[float]:
    """Intervalo del 95 % de AUC(s) − AUC(base) remuestreando grupos (DJs) enteros."""
    rng = np.random.default_rng(semilla)
    gs = np.unique(P.grupo)
    por = {g: np.where(P.grupo == g)[0] for g in gs}
    difs = []
    for _ in range(veces):
        ii = np.concatenate([por[g] for g in rng.choice(gs, len(gs), replace=True)])
        sub = Pares({}, P.y[ii], P.w[ii], P.grupo[ii], P.fuente[ii], P.consenso[ii], {})
        a, b = auc_pares(sub, s[ii]), auc_pares(sub, base[ii])
        if a is not None and b is not None:
            difs.append(a - b)
    return [round(float(np.percentile(difs, 2.5)), 4), round(float(np.percentile(difs, 97.5)), 4)] if difs else []


def calibracion(p: np.ndarray, y: np.ndarray, w: np.ndarray, bins: int = 10) -> Dict[str, object]:
    """Probabilidad media frente a fracción pesada de pares del mismo set, por deciles de la probabilidad, y Brier."""
    ok = np.isfinite(p)
    p, y, w = p[ok], y[ok], w[ok]
    cortes = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    b = np.clip(np.searchsorted(cortes, p, side="right") - 1, 0, len(cortes) - 2)
    filas = [{"p_media": round(float(np.average(p[b == k], weights=w[b == k])), 3),
              "fraccion_mismo_set": round(float(np.average(y[b == k], weights=w[b == k])), 3),
              "peso": round(float(w[b == k].sum() / w.sum()), 3)} for k in range(len(cortes) - 1) if (b == k).any()]
    return {"brier": round(float(np.average((p - y) ** 2, weights=w)), 4), "deciles": filas}


def puntajes_base(P: Pares, stats_todo: Dict[str, Tuple[float, float]]) -> Dict[str, np.ndarray]:
    """Puntajes que no se entrenan: CLAP crudo, las vistas con pesos iguales (P11) y todos los factores con pesos
    iguales en desvíos de los negativos."""
    Z, _ = estandarizar(P.X, FACTORES, P.y == 0, P.w, stats_todo)
    return {"clap crudo": P.X["clap"], "vistas, pesos iguales (P11)": sum(P.X[v] for v in VISTAS_P11),
            "todos los factores, pesos iguales": Z.sum(axis=1)}


def consenso_por_nivel(P: Pares, m: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
    """Por nivel de consenso: qué fracción del peso de los pares del mismo set y del azar cae ahí, y la fracción del
    mismo set entre los dos (con pesos parejos, 0,5 = no distingue)."""
    m = np.ones(len(P.y), dtype=bool) if m is None else m
    out = {}
    wp, wn = P.w[m & (P.y == 1)].sum(), P.w[m & (P.y == 0)].sum()
    for c in range(len(VISTAS_P11) + 1):
        a = P.w[m & (P.y == 1) & (P.consenso == c)].sum()
        b = P.w[m & (P.y == 0) & (P.consenso == c)].sum()
        out[str(c)] = {"mismo_set": round(float(a / wp), 4), "azar": round(float(b / wn), 4),
                       "fraccion_mismo_set": round(float(a / wp / (a / wp + b / wn)), 3) if a + b > 0 else None}
    return out


def acierto_en_vivo(f: zs.Fuente, lado: Lado, t: Tablas, puntuar, top: int = TOP) -> Dict[str, np.ndarray]:
    """Acierto en `top` del tema siguiente con los filtros en vivo: candidatos = biblioteca sin lo tocado que pasa los
    filtros respecto del actual. puntuar(factores) -> {método: puntaje}. Las transiciones cuyo tema siguiente no pasa
    los filtros quedan fuera y se cuentan («recuperación de los filtros»)."""
    filas_out: Dict[str, list] = {}
    sets_out, fuera, total = [], 0, 0
    for k, filas in enumerate(f.sets.values()):
        for p in range(len(filas) - 1):
            q, nx = filas[p], filas[p + 1]
            total += 1
            fx, ok = factores(lado, q, lado, t)
            ok = ok.copy()
            ok[filas[: p + 1]] = False
            if not ok[nx]:
                fuera += 1
                continue
            sets_out.append(k)
            for nombre, s in puntuar(fx).items():
                s = np.where(ok, s, -np.inf)
                filas_out.setdefault(nombre, []).append((int((s > s[nx]).sum() < top), int(ok.sum())))
    out = {n: np.column_stack([np.asarray(sets_out), np.asarray(v)]) for n, v in filas_out.items()}
    out["_recuperacion"] = np.asarray([total - fuera, total])
    return out


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def cargar_rep(nombre: str, uids: Sequence[str], reps_dir: Path = REPS_DIR) -> np.ndarray:
    E = np.load(reps_dir / nombre / "embeddings.npy").astype(np.float64)
    U = json.loads((reps_dir / nombre / "track_uids.json").read_text(encoding="utf-8"))
    pos = {u: i for i, u in enumerate(U)}
    falta = [u for u in uids if u not in pos]
    if falta:
        raise ValueError(f"{nombre}: faltan {len(falta)} temas del índice (p. ej. {falta[:3]})")
    X = E[[pos[u] for u in uids]]
    if not np.isfinite(X).all():
        raise ValueError(f"{nombre}: valores no finitos")
    return X


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "?"


def r4(x) -> Optional[float]:
    return None if x is None else round(float(x), 4)


def main() -> int:
    ap = argparse.ArgumentParser(description="Unidad 1: fusión de vistas (P11), consenso (P13) y ecuación de factores (P12), con E14")
    ap.add_argument("--root", default=str(zs.DEFAULT_ROOT))
    ap.add_argument("--indice", default=str(rc.DEFAULT_INDEX))
    ap.add_argument("--d3", default=str(zs.DEFAULT_D3))
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--salida", default=None)
    ap.add_argument("--low-priority", action="store_true")
    args = ap.parse_args()
    if args.low_priority:
        from src.v4.pipeline.extract_representations import lower_priority
        lower_priority()
    t0 = time.perf_counter()
    root = Path(args.root)

    # --- biblioteca y vistas (BPM de los tags, como zona_sets y la auditoría)
    data, items = rc.load_index(Path(args.indice))
    bib = zs.cargar_biblioteca(items)
    crudos = {k: cargar_rep(v, bib.uids) for k, v in REPS.items()}
    vistas = construir_vistas(crudos, bib.bpm)
    E = espacio(vistas, crudos)
    if not np.allclose(E["clap"], bib.V, atol=1e-5):
        raise ValueError("el CLAP del índice no coincide con clap_full")
    escala = escala_z(E)
    C = {m: concatenar(E, p, escala) for m, p in METODOS.items()}
    print(f"biblioteca {len(bib.uids)} temas; vistas {', '.join(f'{k} {v.shape[1]}' for k, v in E.items())} "
          f"({time.perf_counter() - t0:.0f} s)", flush=True)

    tempo = zs.cargar_tempo(root / "sets" / "tempo_temas.csv")
    d1 = zs.fuente_d1(root / "sets" / "d1.csv", bib)
    d3 = zs.fuente_d3(Path(args.d3), bib)
    bed_todo, dup = zs.fuente_bedrock(root, tempo, bib)
    l7, st, hay = maest_de_bedrock(root, bed_todo)
    bed = subfuente(bed_todo, hay)
    E_bed = espacio(vistas, {"clap": bed.V, "maest": l7[hay], "estilos": st[hay]}) if hay.any() else {}
    con_bib = np.r_[np.zeros(len(bed.V), dtype=bool), dup]
    # Control de dominio: los temas de la biblioteca que son estos mismos originales, en cada vista
    dominio = {}
    if hay.any():
        Cb = (bib.V @ bed.V.T)
        pares_dup = [(i, int(np.argmax(Cb[i]))) for i in np.where(dup)[0] if Cb[i].max() >= zs.COS_DUPLICADO]
        dominio = {v: [round(float(E[v][i] @ E_bed[v][j]), 4) for i, j in pares_dup] for v in ("clap", "maest_c", "estilos")}

    res: Dict[str, object] = {
        "creado": dt.datetime.now().isoformat(timespec="seconds"), "commit": git_head(), "semilla": args.semilla,
        "indice": {"ruta": str(args.indice), "creado": data.get("creado"), "version": data.get("version"), "temas": len(bib.uids)},
        "representaciones": {k: json.loads((REPS_DIR / v / "manifest.json").read_text(encoding="utf-8")).get("created") for k, v in REPS.items()},
        "escala_z": {k: [round(m, 5), round(s, 5)] for k, (m, s) in escala.items()},
        "fuentes": {"d1": d1.curado, "d3_biblioteca": d3.curado,
                    "bedrock": {**bed_todo.curado, "originales_con_maest": int(hay.sum()), "originales_sin_maest": int((~hay).sum()),
                                "sets": {s: len(v) for s, v in bed.sets.items()}}},
        "bedrock_dominio_mismo_tema": dominio,
    }

    # --- tabla de vistas (E14)
    base_pool = {tol: acierto_siguiente(d1.sets, C["clap crudo"] @ C["clap crudo"].T, bib.bpm, tol) for tol in (None, 8.0, 4.0)}
    nulo_d3 = nulo_bpm(d3, bib.bpm, args.semilla)
    nulo_d1 = nulo_bpm(d1, bib.bpm, args.semilla)
    nulo_bed = nulo_bpm(bed, bib.bpm, args.semilla) if hay.any() else None
    filas_vistas: Dict[str, Dict[str, object]] = {}
    real_base = azar_base = None
    for m, pesos in METODOS.items():
        Cm = C[m]
        M = Cm @ Cm.T
        fila: Dict[str, object] = {"pesos": pesos}
        for tol in (None, 8.0, 4.0):
            h = acierto_siguiente(d1.sets, M, bib.bpm, tol)
            clave = "d1_pool_completo" if tol is None else f"d1_pool_{int(tol)}_bpm"
            fila[clave] = {**comparar(h, base_pool[tol], args.semilla), "transiciones": int(len(h)),
                           "candidatos_mediana": int(np.median(h[:, 2]))}
        real, azar = dot_pares(Cm, nulo_d3.I, Cm, nulo_d3.J), dot_pares(Cm, nulo_d3.a, Cm, nulo_d3.b)
        if m == "clap crudo":
            real_base, azar_base = real, azar
        fila["auc_d3_biblioteca"] = r4(auc_con_nulo(nulo_d3, real, azar))
        fila["auc_d3_dif_ic95_por_sets"] = auc_dif_por_sets(nulo_d3, real, azar, real_base, azar_base) if m != "clap crudo" else [0.0, 0.0]
        fila["auc_d1"] = r4(auc_con_nulo(nulo_d1, dot_pares(Cm, nulo_d1.I, Cm, nulo_d1.J), dot_pares(Cm, nulo_d1.a, Cm, nulo_d1.b)))
        if nulo_bed is not None:
            Cb = concatenar(E_bed, pesos, escala)
            fila["auc_bedrock"] = r4(auc_con_nulo(nulo_bed, dot_pares(Cb, nulo_bed.I, Cb, nulo_bed.J),
                                                  dot_pares(Cm, nulo_bed.a, Cm, nulo_bed.b)))
            fb = dataclasses.replace(bed, V=Cb)
            pz = zs.pureza(fb, np.vstack([Cb, Cm]), np.r_[bed.bpm, bib.bpm], np.random.default_rng(args.semilla), excluir=con_bib)
            fila["pureza_bedrock"] = pz.get("clap")
            fila["auc_bedrock_mismo_disco_frente_a_otro_disco"] = r4(auc_entre_sets(bed, Cb))
            fila["auc_bedrock_consecutivos_frente_a_otro_disco"] = r4(auc_entre_sets(bed, Cb, consecutivos=True))
        pz1 = zs.pureza(dataclasses.replace(d1, V=Cm), Cm, bib.bpm, np.random.default_rng(args.semilla))
        fila["pureza_d1"] = pz1.get("clap")
        fila["spearman_dif_bpm"] = round(spearman_tempo(Cm, bib.bpm, args.semilla), 3)
        filas_vistas[m] = fila
        print(f"  {m:34s} D1 {100 * fila['d1_pool_completo']['acierto']:5.2f} %  ±8 {100 * fila['d1_pool_8_bpm']['acierto']:5.2f} %  "
              f"AUC D3 {fila['auc_d3_biblioteca']}  Bedrock {fila.get('auc_bedrock')}  D1 {fila['auc_d1']}  "
              f"tempo {fila['spearman_dif_bpm']:+.3f}", flush=True)
    for m, vs in CONSENSO.items():
        M = np.minimum.reduce([percentil_filas(E[v] @ E[v].T) for v in vs])
        fila = {"vistas": list(vs)}
        for tol in (None, 8.0):
            h = acierto_siguiente(d1.sets, M, bib.bpm, tol)
            fila["d1_pool_completo" if tol is None else "d1_pool_8_bpm"] = {**comparar(h, base_pool[tol], args.semilla), "transiciones": int(len(h))}
        fila["auc_d3_biblioteca"] = r4(auc_con_nulo(nulo_d3, M[nulo_d3.I, nulo_d3.J], M[nulo_d3.a, nulo_d3.b]))
        fila["auc_d1"] = r4(auc_con_nulo(nulo_d1, M[nulo_d1.I, nulo_d1.J], M[nulo_d1.a, nulo_d1.b]))
        filas_vistas[m] = fila
    res["vistas"] = filas_vistas
    # complementariedad: qué transiciones acierta cada vista y cuánto se parecen sus listas
    hits = {v: acierto_siguiente(d1.sets, E[v] @ E[v].T)[:, 1] for v in VISTAS_P11}
    tops = {v: np.argsort(-np.where(np.eye(len(E[v]), dtype=bool), -np.inf, E[v] @ E[v].T), axis=1)[:, :TOP] for v in VISTAS_P11}
    res["complementariedad"] = {
        "aciertos": {v: int(h.sum()) for v, h in hits.items()},
        "union_clap_c_maest_c": int((hits["clap_c"] | hits["maest_c"]).sum()),
        "union_las_tres": int((hits["clap_c"] | hits["maest_c"] | hits["estilos"]).sum()),
        "temas_en_comun_entre_listas_de_30": {f"{a}-{b}": round(float(np.mean([len(set(tops[a][i]) & set(tops[b][i])) for i in range(len(bib.uids))])), 2)
                                              for a, b in (("clap_c", "maest_c"), ("clap_c", "estilos"), ("maest_c", "estilos"))}}
    print(f"tabla de vistas lista ({time.perf_counter() - t0:.0f} s)", flush=True)

    # --- la ecuación (P12) y el consenso (P13): BPM y tonalidad en vivo (Traktor manda sobre los tags)
    from tools.traktor_live.sugerencias_explorer import entries_by_path
    from tools.traktor_live.sugerencias_itunes import find_collection
    taxonomy = rc.load_taxonomy()[0]
    rc.Recommender(items, entries_by_path(find_collection()), n=TOP, taxonomy=taxonomy)   # pone tonalidad y BPM de Traktor
    por_uid = {it.uid: it for it in items}
    generos = sorted({(por_uid[u].genre, por_uid[u].subgenre) for u in bib.uids if por_uid[u].genre}, key=str)
    gid = {g: i for i, g in enumerate(generos)}
    tablas = Tablas(escala, tabla_claves(), tabla_generos(generos, taxonomy), generos)
    lado_bib = Lado({v: E[v] for v in ("clap",) + VISTAS_P11},
                    np.asarray([por_uid[u].bpm if por_uid[u].bpm else np.nan for u in bib.uids], dtype=np.float64),
                    np.asarray([indice_clave(por_uid[u].key) for u in bib.uids]),
                    np.asarray([gid.get((por_uid[u].genre, por_uid[u].subgenre), -1) for u in bib.uids]), bib.artista)
    rng = np.random.default_rng(args.semilla)
    partes = [pares_ecuacion("d3_biblioteca", d3, lado_bib, lado_bib, tablas, rng, mismo_espacio=True)]
    if hay.any():
        lado_bed = Lado({v: E_bed[v] for v in ("clap",) + VISTAS_P11}, bed.bpm, np.full(len(bed.V), -1),
                        np.full(len(bed.V), -1), bed.artista)
        partes.append(pares_ecuacion("bedrock", bed, lado_bed, lado_bib, tablas, rng, mismo_espacio=False, excluir_pool=dup))
    P = Pares.juntar(partes)
    P_d1 = pares_ecuacion("d1", d1, lado_bib, lado_bib, tablas, rng, mismo_espacio=True)
    print(f"pares de la ecuación: {[p.resumen for p in partes]}; D1 {P_d1.resumen} ({time.perf_counter() - t0:.0f} s)", flush=True)

    cols_vistas = list(VISTAS_P11)
    oof = {"ecuación aprendida (P12)": validar_por_grupo(P, FACTORES),
           "ecuación aprendida, solo vistas": validar_por_grupo(P, cols_vistas)}
    _, stats_todo = estandarizar(P.X, FACTORES, P.y == 0, P.w)
    puntajes = {**puntajes_base(P, stats_todo), **oof}
    ecu: Dict[str, object] = {"pares": P.resumen, "pares_d1": P_d1.resumen, "factores": list(FACTORES), "C": LR_C}
    base = puntajes["clap crudo"]
    ecu["auc_fuera_de_muestra"] = {
        n: {"todas": r4(auc_pares(P, s)), **{f: r4(auc_pares(P, s, P.fuente == f)) for f in np.unique(P.fuente)},
            "dif_con_clap_crudo_ic95_por_djs": auc_dif_por_grupos(P, s, base, args.semilla) if n != "clap crudo" else [0.0, 0.0]}
        for n, s in puntajes.items()}
    ecu["calibracion_fuera_de_muestra"] = calibracion(oof["ecuación aprendida (P12)"], P.y, P.w)
    Z, st_all = estandarizar(P.X, FACTORES, P.y == 0, P.w)
    modelo = ajustar(Z, P.y, P.w)
    coef = modelo.coef_[0]
    boot = []
    rng_b = np.random.default_rng(args.semilla)
    gs = np.unique(P.grupo)
    por = {g: np.where(P.grupo == g)[0] for g in gs}
    for _ in range(BOOT_DJ // 2):
        ii = np.concatenate([por[g] for g in rng_b.choice(gs, len(gs), replace=True)])
        if len(np.unique(P.y[ii])) == 2:
            Zi, _ = estandarizar({c: P.X[c][ii] for c in FACTORES}, FACTORES, P.y[ii] == 0, P.w[ii])
            boot.append(ajustar(Zi, P.y[ii], P.w[ii]).coef_[0])
    boot = np.asarray(boot)
    ecu["pesos"] = {c: {"coef_en_desvios": round(float(coef[k]), 3),
                        "ic95_por_djs": [round(float(np.percentile(boot[:, k], 2.5)), 3), round(float(np.percentile(boot[:, k], 97.5)), 3)],
                        "media_y_desvio_de_referencia": [round(st_all[c][0], 4), round(st_all[c][1], 4)]}
                    for k, c in enumerate(FACTORES)}
    ecu["intercepto"] = round(float(modelo.intercept_[0]), 3)
    ecu["consenso_profesionales"] = consenso_por_nivel(P)
    ecu["consenso_d1"] = consenso_por_nivel(P_d1)

    # D1, aparte: nada se ajusta con estos sets
    Zd1, _ = estandarizar(P_d1.X, FACTORES, P_d1.y == 0, P_d1.w, st_all)
    s_d1 = {**puntajes_base(P_d1, stats_todo), "ecuación aprendida (P12)": modelo.decision_function(Zd1)}
    ecu["d1_auc"] = {n: r4(auc_pares(P_d1, s)) for n, s in s_d1.items()}

    def puntuar(fx: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        Zx, _ = estandarizar(fx, FACTORES, np.ones(len(fx["clap"]), dtype=bool), np.ones(len(fx["clap"])), st_all)
        Zt, _ = estandarizar(fx, FACTORES, np.ones(len(fx["clap"]), dtype=bool), np.ones(len(fx["clap"])), stats_todo)
        return {"clap crudo": fx["clap"], "vistas, pesos iguales (P11)": sum(fx[v] for v in VISTAS_P11),
                "todos los factores, pesos iguales": Zt.sum(axis=1), "ecuación aprendida (P12)": modelo.decision_function(Zx)}
    viv = acierto_en_vivo(d1, lado_bib, tablas, puntuar)
    rec_ok, rec_tot = (int(x) for x in viv.pop("_recuperacion"))
    ecu["d1_acierto_en_vivo"] = {"recuperacion_de_los_filtros": [rec_ok, rec_tot],
                                 **{n: {**comparar(h, viv["clap crudo"], args.semilla), "candidatos_mediana": int(np.median(h[:, 2]))}
                                    for n, h in viv.items()}}
    res["ecuacion"] = ecu
    res["segundos"] = round(time.perf_counter() - t0, 1)
    out = Path(args.salida) if args.salida else root / "sets" / "fusion_vistas.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: res["ecuacion"][k] for k in ("auc_fuera_de_muestra", "pesos", "d1_auc", "d1_acierto_en_vivo")}, ensure_ascii=False)[:4000])
    print(f"[OK] -> {out} ({res['segundos']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
