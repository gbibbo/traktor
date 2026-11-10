"""
PURPOSE: G13 y P12a del plan de playlists en vivo (sección 12.9), solo texto: los pares de temas consecutivos de los sets
         de D5 (1001Tracklists) con los metadatos de Beatport de los previews de D6 (`detalles` de los manifiestos:
         género, BPM y tonalidad). No le pide nada a Beatport ni a 1001Tracklists; lee lo que ya bajó la otra sesión.
           G13   ¿la taxonomía de géneros ordena las transiciones de los DJs? Cuántas veces más que al azar aparece cada
                 valor del puntaje de la taxonomía; frente a «mismo género o no» y a lo aprendido de los sets (G7,
                 coaparición sin dirección; T1, transición dirigida), con los DJs de prueba fuera; los pares de géneros
                 donde la taxonomía y los DJs más discrepan.
           P12a  cuántas veces más que al azar aparece cada diferencia de BPM, cada relación de tonalidad (P7) y cada
                 distancia entre tónicas; con la tonalidad de Beatport no se aprende un peso para la regla actual
                 (Gabriel, DECISIONS 2026-11-10, «después de E15», punto 2). La ecuación de solo texto (BPM y género, con
                 los filtros en vivo) se ajusta con los DJs de prueba fuera.
           python src/v4/evaluation/pares_texto.py [--root "D:/TRAKTOR ML datos/dj_sets"]  -> <root>/sets/pares_texto.json
         Decisiones (agente, fijadas antes de medir):
           Pares: filas seguidas de un set sin las tocadas encima de otra (sección 6), los dos temas con `detalles` y
             distintos (otro id de Beatport). Dos nulos, informados por separado: «mismo set» = el principal con otro
             tema del mismo set a 5 filas o más (mide la transición); «otro set» = el principal con un tema de un set de
             otro DJ (mide el estilo de la noche). 5 por cada par consecutivo. Cada set pesa 1 en cada clase.
           DJ: el primero del título (`partes_titulo`, `djs_de`); un B2B va con su primer DJ. Pliegues: 10 grupos de
             DJs al azar con semilla 0.
           Intervalos: remuestreo de DJs, 500 veces.
           Ecuación de solo texto: positivos = consecutivos que pasan los filtros en vivo con BPM y tonalidad de Beatport
             (BPM a ±20; tonalidad compatible, best_key_shift ≥ 0,60, si el principal la tiene); 20 negativos por
             positivo de los temas con `detalles` de otros sets, de otro artista, que pasan los filtros. Factores bpm =
             1 − |ΔBPM|/20 y genero = la taxonomía; regresión logística C = 1 en desvíos de los negativos.
CHANGELOG:
  - 2026-11-10: Creación (G13 y P12a).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from src.v4.common import harmonic  # noqa: E402
from src.v4.evaluation import fusion_vistas as fv  # noqa: E402
from src.v4.evaluation import zona_sets as zs  # noqa: E402
from tools.dj_sets.tracklists_1001 import djs_de, norm, partes_titulo  # noqa: E402
from tools.traktor_live import recomendador as rc  # noqa: E402

SEMILLA = 0
K_NULO = 5
DIST_MIN = 5
PLIEGUES = 10
BOOT = 500
BOOT_PESOS = 250
BINS_BPM = (0, 1, 2, 3, 5, 8, 12, 20)           # |ΔBPM| en [0,1), [1,2), [2,3), [3,5), [5,8), [8,12), [12,20), 20 o más
CATEGORIAS_CLAVE = ("misma", "relativa", "paralela", "vecina", "diagonal", "un semitono", "dos semitonos", "otra")


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

@dataclass
class Temas:
    """Los temas con `detalles` (uno por id de Beatport)."""
    bp: List[str]
    bpm: np.ndarray
    camelot: List[str]
    genero: List[Optional[str]]
    artista: List[str]
    genero_1001: List[str]


@dataclass
class Sets:
    nombres: List[str]
    dj: List[str]
    filas: List[List[int]]            # índice en Temas de cada fila usable, en orden (−1 = sin detalles)


def cargar(root: Path) -> Tuple[Temas, Sets, Dict[str, object]]:
    det: Dict[str, dict] = {}
    for f in sorted((root / "beatport_previews").glob("*/tanda_*.json")):
        for t in json.loads(f.read_text(encoding="utf-8")).get("temas", []):
            if t.get("estado") == "ok" and t.get("clave_1001") and t.get("detalles"):
                det.setdefault(t["clave_1001"], t)
    pos: Dict[str, int] = {}
    bp, bpm, cam, gen, art, g1001 = [], [], [], [], [], []
    nombres, djs, filas = [], [], []
    n_sets = n_filas = n_juntas = 0
    for f in sorted((root / "1001tracklists" / "sets").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        n_sets += 1
        fila: List[int] = []
        for t in d.get("temas", []):
            n_filas += 1
            if t.get("junto_al_anterior"):
                n_juntas += 1
                continue
            m = det.get(t.get("id_texto") or "")
            if m is None:
                fila.append(-1)
                continue
            b = str(m["beatport_id"])
            if b not in pos:
                x = m["detalles"]
                pos[b] = len(bp)
                bp.append(b)
                bpm.append(float(x["bpm"]) if x.get("bpm") else np.nan)
                cam.append(harmonic.to_camelot(x.get("key") or "") or "?")
                gen.append(x.get("genre") or None)
                art.append(norm((x.get("artists") or [""])[0]))
                g1001.append(t.get("genero") or "")
            fila.append(pos[b])
        dj = djs_de(partes_titulo(d.get("titulo", ""))[0])
        nombres.append(d.get("id") or f.stem)
        djs.append(norm(dj[0]) if dj else "?")
        filas.append(fila)
    temas = Temas(bp, np.asarray(bpm), cam, gen, art, g1001)
    info = {"sets": n_sets, "filas": n_filas, "filas_tocadas_encima": n_juntas, "previews_con_detalles": len(det),
            "temas_de_los_sets_con_detalles": len(bp), "djs": len(set(djs))}
    return temas, Sets(nombres, djs, filas), info


# ---------------------------------------------------------------------------
# Pares
# ---------------------------------------------------------------------------

@dataclass
class Pares:
    a: np.ndarray
    b: np.ndarray
    w: np.ndarray
    grupo: np.ndarray                  # DJ
    clase: np.ndarray                  # «consecutivo», «mismo set», «otro set»


def pares(t: Temas, s: Sets, rng: np.random.Generator, k: int = K_NULO, dist_min: int = DIST_MIN) -> Tuple[Pares, Dict[str, int]]:
    """Consecutivos y sus dos nulos. Cada set suma 1 de peso en cada clase."""
    por_dj: Dict[str, List[int]] = defaultdict(list)
    for i, dj in enumerate(s.dj):
        por_dj[dj].append(i)
    temas_set = [np.asarray([x for x in f if x >= 0]) for f in s.filas]
    filas: Dict[str, list] = {c: [] for c in ("a", "b", "w", "grupo", "clase")}
    cuentas = {"consecutivos": 0, "mismo_tema": 0, "repetidos_en_otros_sets": 0}
    vistos = set()
    con_temas = np.asarray([i for i in range(len(s.filas)) if len(temas_set[i])])
    dj_arr = np.asarray(s.dj, dtype=object)
    for k_set, fila in enumerate(s.filas):
        cons = []
        for p in range(len(fila) - 1):
            x, y = fila[p], fila[p + 1]
            if x >= 0 and y >= 0:
                if x == y:
                    cuentas["mismo_tema"] += 1
                else:
                    cons.append((p, x, y))
        if not cons:
            continue
        lejos = [[q for q in range(len(fila)) if fila[q] >= 0 and abs(q - p) >= dist_min and fila[q] != x] for p, x, _ in cons]
        otros = con_temas[dj_arr[con_temas] != s.dj[k_set]]
        n, n_lj = len(cons), sum(1 for lj in lejos if lj)
        for (p, x, y), lj in zip(cons, lejos):
            cuentas["consecutivos"] += 1
            if (x, y) in vistos:
                cuentas["repetidos_en_otros_sets"] += 1
            vistos.add((x, y))
            filas["a"].append(x); filas["b"].append(y); filas["w"].append(1.0 / n)
            filas["grupo"].append(s.dj[k_set]); filas["clase"].append("consecutivo")
            if lj:
                for q in rng.choice(lj, size=k, replace=True):
                    filas["a"].append(x); filas["b"].append(fila[q]); filas["w"].append(1.0 / (n_lj * k))
                    filas["grupo"].append(s.dj[k_set]); filas["clase"].append("mismo set")
            for o in rng.choice(otros, size=k, replace=True):
                filas["a"].append(x); filas["b"].append(int(rng.choice(temas_set[o]))); filas["w"].append(1.0 / (n * k))
                filas["grupo"].append(s.dj[k_set]); filas["clase"].append("otro set")
    P = Pares(*(np.asarray(filas[c]) for c in ("a", "b", "w", "grupo", "clase")))
    return P, cuentas


# ---------------------------------------------------------------------------
# Medidas
# ---------------------------------------------------------------------------

def tonica(camelot: str) -> Optional[Tuple[int, str]]:
    """(semitono de la tónica desde Do, modo «A»/«B») de una tonalidad Camelot; None si no se conoce."""
    try:
        n, m = int(camelot[:-1]), camelot[-1]
    except (ValueError, IndexError, TypeError):
        return None
    if m not in "AB" or not 1 <= n <= 12:
        return None
    return (rc.MAJOR_CAMELOT if m == "B" else rc.MINOR_CAMELOT).index(n), m


def relacion_clave(a: str, b: str) -> Optional[str]:
    """La relación de P7 entre dos tonalidades Camelot, sin transponer."""
    ta, tb = tonica(a), tonica(b)
    if ta is None or tb is None:
        return None
    paso = harmonic.wheel_step(a, b)
    if ta[1] != tb[1] and ta[0] == tb[0]:
        return "paralela"
    return {"+0": "misma", "+0x": "relativa", "+1": "vecina", "-1": "vecina", "+1x": "diagonal", "-1x": "diagonal",
            "+5": "un semitono", "-5": "un semitono", "+2": "dos semitonos", "-2": "dos semitonos"}.get(paso, "otra")


def distancia_tonica(a: str, b: str) -> Optional[int]:
    """Semitonos entre las tónicas, sin mirar el modo, de 0 a 6."""
    ta, tb = tonica(a), tonica(b)
    if ta is None or tb is None:
        return None
    d = (tb[0] - ta[0]) % 12
    return min(d, 12 - d)


def bin_bpm(d: float) -> Optional[str]:
    if not np.isfinite(d):
        return None
    for lo, hi in zip(BINS_BPM, BINS_BPM[1:]):
        if lo <= d < hi:
            return f"{lo}-{hi}"
    return f"{BINS_BPM[-1]}+"


def razones(cat: Sequence[Optional[object]], P: Pares, nulo: str, rng: np.random.Generator, boot: int = BOOT
            ) -> Dict[str, Dict[str, object]]:
    """Por categoría: fracción (pesada) entre los consecutivos y en el nulo, su cociente y el intervalo remuestreando
    los DJs que tienen consecutivos (sumas por DJ y categoría, una vez)."""
    cat = [None if c is None else str(c) for c in cat]
    usa = (P.clase == "consecutivo") | (P.clase == nulo)
    cats = sorted({c for c, u in zip(cat, usa) if u and c is not None}, key=lambda x: (len(x), x))
    codigo = {c: i for i, c in enumerate(cats)}
    cod = np.asarray([-1 if c is None else codigo.get(c, -1) for c in cat])
    mp, mn = (cod >= 0) & (P.clase == "consecutivo"), (cod >= 0) & (P.clase == nulo)
    gs, gi = np.unique(P.grupo, return_inverse=True)
    Sp = np.zeros((len(gs), len(cats)))
    Sn = np.zeros((len(gs), len(cats)))
    np.add.at(Sp, (gi[mp], cod[mp]), P.w[mp])
    np.add.at(Sn, (gi[mn], cod[mn]), P.w[mn])
    cuenta = np.bincount(cod[mp], minlength=len(cats))
    djs = np.unique(gi[mp])

    def frac(m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p, n = m @ Sp, m @ Sn
        return p / max(p.sum(), 1e-300), n / max(n.sum(), 1e-300)

    fp, fn = frac(np.isin(np.arange(len(gs)), djs).astype(float))
    bs = []
    for _ in range(boot):
        a, b = frac(np.bincount(rng.choice(djs, len(djs), replace=True), minlength=len(gs)).astype(float))
        with np.errstate(divide="ignore", invalid="ignore"):
            bs.append(np.where(b > 0, a / b, np.nan))
    bs = np.asarray(bs)
    out = {}
    for k, c in enumerate(cats):
        v = bs[:, k][np.isfinite(bs[:, k])]
        out[c] = {"consecutivos": round(float(fp[k]), 4), "nulo": round(float(fn[k]), 4),
                  "veces": round(float(fp[k] / fn[k]), 3) if fn[k] > 0 else None,
                  "ic95": [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] if len(v) else None,
                  "pares_consecutivos": int(cuenta[k])}
    return out


def auc_clases(s: np.ndarray, P: Pares, nulo: str) -> Optional[float]:
    ok = np.isfinite(s)
    mp, mn = ok & (P.clase == "consecutivo"), ok & (P.clase == nulo)
    return fv.auc_pesado(s[mp], P.w[mp], s[mn], P.w[mn])


def auc_dif_dj(s: np.ndarray, base: np.ndarray, P: Pares, nulo: str, rng: np.random.Generator, boot: int = BOOT) -> List[float]:
    """Intervalo del 95 % de AUC(s) − AUC(base) remuestreando DJs."""
    gs = np.unique(P.grupo)
    por = {g: np.where(P.grupo == g)[0] for g in gs}
    difs = []
    for _ in range(boot):
        ii = np.concatenate([por[g] for g in rng.choice(gs, len(gs), replace=True)])
        sub = Pares(P.a[ii], P.b[ii], P.w[ii], P.grupo[ii], P.clase[ii])
        x, y = auc_clases(s[ii], sub, nulo), auc_clases(base[ii], sub, nulo)
        if x is not None and y is not None:
            difs.append(x - y)
    return [round(float(np.percentile(difs, 2.5)), 4), round(float(np.percentile(difs, 97.5)), 4)] if difs else []


def pliegues(grupos: np.ndarray, n: int = PLIEGUES, semilla: int = SEMILLA) -> Dict[str, int]:
    gs = sorted(set(grupos.tolist()))
    perm = np.random.default_rng(semilla).permutation(len(gs))
    return {gs[i]: k % n for k, i in enumerate(perm)}


def aprendido(ga: Sequence[Optional[str]], gb: Sequence[Optional[str]], P: Pares, nulo: str, dirigido: bool,
              n: int = PLIEGUES) -> np.ndarray:
    """log del cociente (suavizado) entre consecutivos y el nulo de cada par de géneros, aprendido sin el pliegue del DJ
    del par (G7 sin dirección, T1 dirigido). nan si a alguno le falta el género."""
    pl = pliegues(P.grupo, n)
    f = np.asarray([pl[g] for g in P.grupo])
    clave = [None if a is None or b is None else ((a, b) if dirigido else tuple(sorted((a, b)))) for a, b in zip(ga, gb)]
    out = np.full(len(P.a), np.nan)
    for k in range(n):
        tr = f != k
        cp, cn = defaultdict(float), defaultdict(float)
        wp = P.w[tr & (P.clase == "consecutivo")].sum()
        wn = P.w[tr & (P.clase == nulo)].sum()
        for i in np.where(tr & ((P.clase == "consecutivo") | (P.clase == nulo)))[0]:
            if clave[i] is not None:
                (cp if P.clase[i] == "consecutivo" else cn)[clave[i]] += P.w[i]
        alfa = 1.0 / max(len(P.a), 1)
        for i in np.where(~tr)[0]:
            c = clave[i]
            if c is not None:
                out[i] = np.log((cp[c] / wp + alfa) / (cn[c] / wn + alfa))
    return out


def discrepancias(tax: np.ndarray, ga, gb, P: Pares, nulo: str, minimo: int = 30, n: int = 15) -> List[Dict[str, object]]:
    """Pares de géneros (sin dirección) con al menos `minimo` consecutivos: cociente de los DJs frente al valor de la
    taxonomía; los n donde más discrepan (rango de uno frente al rango del otro)."""
    cp, cn, cnt, val = defaultdict(float), defaultdict(float), defaultdict(int), {}
    wp, wn = P.w[P.clase == "consecutivo"].sum(), P.w[P.clase == nulo].sum()
    for i in range(len(P.a)):
        if ga[i] is None or gb[i] is None or not np.isfinite(tax[i]):
            continue
        c = tuple(sorted((ga[i], gb[i])))
        val[c] = float(tax[i])
        if P.clase[i] == "consecutivo":
            cp[c] += P.w[i]
            cnt[c] += 1
        elif P.clase[i] == nulo:
            cn[c] += P.w[i]
    filas = [{"generos": list(c), "taxonomia": round(val[c], 3), "veces": round((cp[c] / wp) / (cn[c] / wn), 3),
              "consecutivos": cnt[c]} for c in cnt if cnt[c] >= minimo and cn[c] > 0]
    if not filas:
        return []
    from scipy.stats import rankdata
    rt = rankdata([f["taxonomia"] for f in filas]) / len(filas)
    rv = rankdata([f["veces"] for f in filas]) / len(filas)
    for f, a, b in zip(filas, rt, rv):
        f["discrepancia"] = round(float(b - a), 3)
    return sorted(filas, key=lambda f: -abs(f["discrepancia"]))[:n]


# ---------------------------------------------------------------------------
# Ecuación de solo texto
# ---------------------------------------------------------------------------

def ecuacion_texto(t: Temas, s: Sets, tax_de: Callable[[Optional[str], Optional[str]], float], rng: np.random.Generator,
                   k_neg: int = fv.K_NEG) -> Dict[str, object]:
    tabla = fv.tabla_claves()
    ci = np.asarray([fv.indice_clave(c) for c in t.camelot])
    art = np.asarray(t.artista)
    en_set = [set(x for x in f if x >= 0) for f in s.filas]
    X: Dict[str, list] = {"bpm": [], "genero": []}
    y, w, grupo = [], [], []
    fuera = total = 0
    for k_set, fila in enumerate(s.filas):
        cons = [(fila[p], fila[p + 1]) for p in range(len(fila) - 1) if fila[p] >= 0 and fila[p + 1] >= 0 and fila[p] != fila[p + 1]]
        filas_ok = []
        for a, b in cons:
            total += 1
            ok = np.ones(len(t.bp), dtype=bool)
            if np.isfinite(t.bpm[a]):
                d = np.abs(t.bpm - t.bpm[a])
                ok &= np.isfinite(d) & (d <= rc.BPM_RANGE)
            if ci[a] >= 0:
                ok &= (ci >= 0) & (tabla[ci[a], np.maximum(ci, 0)] >= rc.KEY_MIN)
            if not ok[b]:
                fuera += 1
                continue
            disp = ok.copy()
            disp[list(en_set[k_set])] = False
            if art[a]:
                disp &= art != art[a]
            cand = np.where(disp)[0]
            if not len(cand):
                fuera += 1
                continue
            filas_ok.append((a, b, rng.choice(cand, size=k_neg, replace=True)))
        for a, b, neg in filas_ok:
            todos = np.r_[b, neg]
            X["bpm"].append(np.where(np.isfinite(t.bpm[todos]), np.maximum(0.0, 1.0 - np.abs(t.bpm[todos] - t.bpm[a]) / rc.BPM_RANGE), np.nan)
                            if np.isfinite(t.bpm[a]) else np.full(len(todos), np.nan))
            X["genero"].append(np.asarray([tax_de(t.genero[a], t.genero[j]) for j in todos], dtype=float))
            y.append(np.r_[1, np.zeros(k_neg)])
            w.append(np.r_[1.0 / len(filas_ok), np.full(k_neg, 1.0 / (len(filas_ok) * k_neg))])
            grupo.append(np.full(k_neg + 1, s.dj[k_set], dtype=object))
    P = fv.Pares({c: np.concatenate(v) for c, v in X.items()}, np.concatenate(y).astype(int), np.concatenate(w),
                 np.concatenate(grupo).astype(str), np.full(sum(len(v) for v in y), "d5"), np.zeros(sum(len(v) for v in y), int), {})
    cols = ("bpm", "genero")
    pl = pliegues(P.grupo)
    f = np.asarray([pl[g] for g in P.grupo])
    oof = np.full(len(P.y), np.nan)
    for k in range(PLIEGUES):
        tr, te = np.where(f != k)[0], np.where(f == k)[0]
        Ztr, st = fv.estandarizar({c: P.X[c][tr] for c in cols}, cols, P.y[tr] == 0, P.w[tr])
        Zte, _ = fv.estandarizar({c: P.X[c][te] for c in cols}, cols, P.y[te] == 0, P.w[te], st)
        oof[te] = fv.ajustar(Ztr, P.y[tr], P.w[tr]).predict_proba(Zte)[:, 1]
    Z, st = fv.estandarizar(P.X, cols, P.y == 0, P.w)
    puntajes = {"bpm": Z[:, 0], "genero": Z[:, 1], "pesos iguales": Z.sum(axis=1), "aprendida, fuera de muestra": oof}
    modelo = fv.ajustar(Z, P.y, P.w)
    gs = np.unique(P.grupo)
    por = {g: np.where(P.grupo == g)[0] for g in gs}
    boot = []
    for _ in range(BOOT_PESOS):
        ii = np.concatenate([por[g] for g in rng.choice(gs, len(gs), replace=True)])
        Zi, _ = fv.estandarizar({c: P.X[c][ii] for c in cols}, cols, P.y[ii] == 0, P.w[ii])
        boot.append(fv.ajustar(Zi, P.y[ii], P.w[ii]).coef_[0])
    boot = np.asarray(boot)
    return {"consecutivos": total, "fuera_por_filtros_o_sin_candidatos": fuera, "positivos": int(P.y.sum()),
            "negativos_por_positivo": k_neg, "djs": int(len(gs)),
            "auc": {n: fv.r4(fv.auc_pares(P, s_)) for n, s_ in puntajes.items()},
            "auc_dif_aprendida_menos_iguales_ic95_por_djs": fv.auc_dif_por_grupos(P, oof, puntajes["pesos iguales"], SEMILLA),
            "pesos": {c: {"coef_en_desvios": round(float(modelo.coef_[0][k]), 3),
                          "ic95_por_djs": [round(float(np.percentile(boot[:, k], 2.5)), 3), round(float(np.percentile(boot[:, k], 97.5)), 3)],
                          "media_y_desvio_de_los_negativos": [round(st[c][0], 4), round(st[c][1], 4)]} for k, c in enumerate(cols)},
            "cociente_de_pesos_genero_sobre_bpm": round(float(modelo.coef_[0][1] / modelo.coef_[0][0]), 3)}


# ---------------------------------------------------------------------------

def medir(root: Path, semilla: int = SEMILLA) -> Dict[str, object]:
    t0 = time.perf_counter()
    temas, sets, info = cargar(root)
    rng = np.random.default_rng(semilla)
    P, cuentas = pares(temas, sets, rng)
    taxonomy = rc.load_taxonomy()[0]
    cache: Dict[Tuple[Optional[str], Optional[str]], float] = {}

    def tax_de(a: Optional[str], b: Optional[str]) -> float:
        if (a, b) not in cache:
            v = taxonomy.compare((a, None), (b, None)) if a and b else None
            cache[(a, b)] = np.nan if v is None else float(v)
        return cache[(a, b)]

    ga = [temas.genero[i] for i in P.a]
    gb = [temas.genero[i] for i in P.b]
    tax = np.asarray([tax_de(a, b) for a, b in zip(ga, gb)])
    mismo = np.asarray([np.nan if a is None or b is None else float(a == b) for a, b in zip(ga, gb)])
    res: Dict[str, object] = {"creado": dt.datetime.now().isoformat(timespec="seconds"), "commit": fv.git_head(), "semilla": semilla,
                              "datos": info, "pares": {**cuentas, **{c: int(np.sum(P.clase == c)) for c in np.unique(P.clase)},
                                                       "djs_con_consecutivos": int(len(np.unique(P.grupo[P.clase == "consecutivo"])))}}
    # género de Beatport frente al de 1001Tracklists, en los temas
    raiz = lambda g: (g or "").split(" (")[0].strip().lower()  # noqa: E731
    con = [(g, h) for g, h in zip(temas.genero, temas.genero_1001) if g and h]
    res["genero_beatport_frente_a_1001"] = {"temas_con_los_dos": len(con), "igual": round(float(np.mean([g.lower() == h.lower() for g, h in con])), 3),
                                            "igual_sin_el_parentesis": round(float(np.mean([raiz(g) == raiz(h) for g, h in con])), 3)}
    # G13
    g13: Dict[str, object] = {"valores_distintos_de_la_taxonomia": int(len(np.unique(tax[np.isfinite(tax)])))}
    for nulo in ("mismo set", "otro set"):
        r = razones([None if not np.isfinite(v) else f"{v:.3f}" for v in tax], P, nulo, rng)
        vals = sorted(r, key=float)
        from scipy.stats import spearmanr
        sp = spearmanr([float(v) for v in vals], [np.log(r[v]["veces"]) for v in vals]).statistic if len(vals) > 2 else None
        l_g7 = aprendido(ga, gb, P, nulo, dirigido=False)
        l_t1 = aprendido(ga, gb, P, nulo, dirigido=True)
        comun = np.isfinite(tax) & np.isfinite(mismo) & np.isfinite(l_g7) & np.isfinite(l_t1)
        aucs = {n: fv.r4(auc_clases(np.where(comun, s_, np.nan), P, nulo)) for n, s_ in
                (("taxonomia", tax), ("mismo genero o no", mismo), ("G7 coaparicion aprendida", l_g7), ("T1 transicion aprendida", l_t1))}
        g13[nulo] = {"por_valor": r, "spearman_valor_log_veces": None if sp is None else round(float(sp), 3), "auc": aucs,
                     "dif_taxonomia_menos_mismo_genero_ic95": auc_dif_dj(np.where(comun, tax, np.nan), np.where(comun, mismo, np.nan), P, nulo, rng),
                     "dif_g7_menos_taxonomia_ic95": auc_dif_dj(np.where(comun, l_g7, np.nan), np.where(comun, tax, np.nan), P, nulo, rng),
                     "dif_t1_menos_taxonomia_ic95": auc_dif_dj(np.where(comun, l_t1, np.nan), np.where(comun, tax, np.nan), P, nulo, rng),
                     "discrepancias": discrepancias(tax, ga, gb, P, nulo)}
        print(f"  G13 {nulo}: AUC {aucs} ({time.perf_counter() - t0:.0f} s)", flush=True)
    res["g13"] = g13
    # P12a: curvas
    db = np.abs(temas.bpm[P.a] - temas.bpm[P.b])
    rel = [relacion_clave(temas.camelot[a], temas.camelot[b]) for a, b in zip(P.a, P.b)]
    ton = [distancia_tonica(temas.camelot[a], temas.camelot[b]) for a, b in zip(P.a, P.b)]
    res["p12a"] = {nulo: {"bpm": razones([bin_bpm(d) for d in db], P, nulo, rng), "relacion_de_tonalidad": razones(rel, P, nulo, rng),
                          "distancia_entre_tonicas": razones(ton, P, nulo, rng),
                          "modo": razones([None if tonica(temas.camelot[a]) is None or tonica(temas.camelot[b]) is None else
                                           f"{tonica(temas.camelot[a])[1]}{tonica(temas.camelot[b])[1]}" for a, b in zip(P.a, P.b)], P, nulo, rng)}
                   for nulo in ("mismo set", "otro set")}
    res["p12a"]["ecuacion_solo_texto"] = ecuacion_texto(temas, sets, tax_de, rng)
    res["segundos"] = round(time.perf_counter() - t0, 1)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="G13 y P12a: pares consecutivos de D5 con los metadatos de Beatport (solo texto)")
    ap.add_argument("--root", default=str(zs.DEFAULT_ROOT))
    ap.add_argument("--semilla", type=int, default=SEMILLA)
    args = ap.parse_args()
    res = medir(Path(args.root), args.semilla)
    out = Path(args.root) / "sets" / "pares_texto.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"datos": res["datos"], "pares": res["pares"], "g13_auc": {k: v["auc"] for k, v in res["g13"].items() if isinstance(v, dict)},
                      "ecuacion": res["p12a"]["ecuacion_solo_texto"]["auc"]}, ensure_ascii=False))
    print(f"[OK] -> {out} ({res['segundos']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
