"""
PURPOSE: Tabla de momentos de la noche (L10 del plan de recomendación v2, docs/plans/plan_recomendacion_v2.md
         § 3.4) medida en DJSets v1 (tracklists de 1001Tracklists con BPM y género de Beatport), no escrita a ojo.
         Dos fuentes de «momento», independientes:
           1. Títulos: palabras clave del título del set (Opening Set, Warm Up, Peak Time, Closing Set, After
              Party, Sunrise Set, Dawn Patrol). Las que nombran la temporada de un club («Closing Party»,
              «Opening Night») y las marcas («Afterlife», «Sunrise Festival») se cuentan aparte y no son momento.
           2. Posición dentro de los sets largos (`use` con «trayectoria», 3 h o más): fracción del set por la
              hora de entrada (cue_s, interpolada entre filas) cuando el set la tiene, si no por fila. Curvas
              por quintos y deciles del BPM de Beatport (absoluto y relativo a la mediana del propio set) y de
              los géneros de Beatport.
         Aparte, las mismas curvas por familia de estilo del set (techno; house y tech house; melódica y
         progresiva), porque el estilo mueve el BPM mucho más que el momento.
         La tabla sale de la «noche entera»: sets largos titulados All Night Long / Open To Close / N Hour Set,
         o de 6 h o más. Apertura = primer quinto, subida = 20-60 %, pico = 60-80 %, cierre = último quinto.
         «after» sale de los títulos. Banda de BPM = percentiles 25 y 75 de las filas del momento; la mediana y
         la banda llevan un intervalo bootstrap por DJ (split_key de split_by_dj.csv, el primer DJ del nombre).
         Un momento con menos de 30 sets o de 15 DJs queda con "bpm": null y "generos": {}.
         Filas que entran: las que no suenan encima de otra («w/»); BPM y género solo de las identificadas con
         metadatos de Beatport. Lee las tablas con dtype=str y no las modifica. El JSON se escribe solo con --json.
           python tools/dj_sets/momentos_noche.py --json config/telefono/momentos.json
CHANGELOG:
  - 2026-11-13: Creación (L10; informe docs/reports/momentos_djsets_2026-11-13.md).
  - 2026-11-13: El JSON lleva también bpm_delta (Δ dentro del set, por momento) y generos_noche (la proporción de cada
                género en toda la noche), que usa el ejecutor del teléfono para ordenar por momento.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)

DEFAULT_METADATA = Path("D:/TRAKTOR ML datos/linea_a/djsets_v1/metadata")
MIN_SETS, MIN_DJS = 30, 15
TOP_GENEROS = 8
GENEROS_CURVA = 10  # géneros que se siguen a lo largo del set
NOCHE_MIN = 360  # minutos: un set largo de 6 h o más se toma como noche entera aunque el título no lo diga
CUE_MIN_SHARE = 0.5  # con menos filas con hora de entrada, la posición sale del orden de las filas

# Momento por posición en la noche entera, como fracción del set: [desde, hasta), el último incluye el 1.
# Apertura y cierre son el primer y el último quinto (fijado antes de medir); el corte del 60 % entre subida y
# pico se eligió mirando las curvas (es exploratorio; el informe da la variante pico = 40-80 %).
MOMENTOS_POSICION: Dict[str, Tuple[float, float]] = {
    "apertura": (0.0, 0.2),
    "subida": (0.2, 0.6),
    "pico": (0.6, 0.8),
    "cierre": (0.8, 1.0),
}
MOMENTOS = ["apertura", "subida", "pico", "cierre", "after"]
QUINTOS = np.linspace(0.0, 1.0, 6)
DECILES = np.linspace(0.0, 1.0, 11)

# Palabras clave del título: (etiqueta, nombre, regex sin distinguir mayúsculas). «temporada», «marca» y
# «atardecer» no son momento: se informan para ver qué se descartó.
KEYWORDS: List[Tuple[str, str, str]] = [
    ("apertura", "Opening Set / Opener", r"\bopening set\b|\bopener\b"),
    ("apertura", "Warm Up", r"\bwarm[ -]?up\b"),
    ("pico", "Peak Time / Prime Time", r"\bpeak[ -]?time\b|\bprime[ -]?time\b"),
    ("cierre", "Closing Set / Closer Set", r"\bclosing set\b|\bcloser set\b"),
    ("after", "After Party / After Hours / After by Day", r"\bafter[ -]?(?:party|hours?|by day)\b|\bafterhours?\b"),
    ("after", "Sunrise Set / Sunrise Edition", r"\bsunrise (?:set|edition)\b"),
    ("after", "Dawn Patrol", r"\bdawn patrol\b"),
    ("temporada", "Opening/Closing Party, Night, Fiesta, Weekend; Reopening",
     r"\b(?:re)?opening (?:party|night|fiesta|weekend)\b|\bclosing (?:party|fiesta|weekend|night)\b|\breopening\b"),
    ("marca", "Afterlife, Afterwatt, Sunrise Festival", r"\bafterlife\b|\bafterwatt\b|\bsunrise festival\b"),
    ("atardecer", "Sunset", r"\bsunsets?\b"),
    ("noche_entera", "All Night Long / Allnighter / Open To Close / N Hour Set",
     r"\ball ?night\b|\ballnighter\b|\bopen to close\b|\b\d+ ?(?:hour|hr)s? (?:dj )?set\b|\(\d+ hour"),
]
ETIQUETAS_MOMENTO = ["apertura", "pico", "cierre", "after"]

# Familia de estilo de un set, para ver el momento dentro de cada estilo (géneros de Beatport).
FAMILIAS: Dict[str, List[str]] = {
    "techno": ["Techno (Peak Time / Driving)", "Techno (Raw / Deep / Hypnotic)", "Hard Techno"],
    "house y tech house": ["Tech House", "House", "Minimal / Deep Tech", "Deep House", "Afro House", "Jackin House",
                           "Funky House", "Bass House"],
    "melódica y progresiva": ["Melodic House & Techno", "Progressive House", "Organic House", "Indie Dance",
                              "Electronica", "Downtempo", "Indie Tech", "Trance (Main Floor)",
                              "Trance (Raw / Deep / Hypnotic)"],
}


# ---------------------------------------------------------------- datos

def load(metadata: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """sets (con split_key), filas del tracklist y temas; todo leído como texto y convertido acá."""
    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(metadata / name, dtype=str, keep_default_na=False)

    sets, rows, tracks, split = read("sets.csv"), read("set_tracks.csv"), read("tracks.csv"), read("split_by_dj.csv")
    sets = sets.merge(split[["set_id", "split_key"]], on="set_id", how="left", validate="one_to_one")
    if (sets.split_key.fillna("") == "").any():
        raise RuntimeError("Hay sets sin split_key en split_by_dj.csv")
    sets["min"] = pd.to_numeric(sets.minutes, errors="raise")
    sets["swc"] = pd.to_numeric(sets.share_with_cue, errors="coerce").fillna(0.0)
    sets["trayectoria"] = sets.use.str.contains("trayectoria")
    rows["pos"] = rows.position.astype(int)
    rows["cue"] = pd.to_numeric(rows.cue_s, errors="coerce")
    tracks["bpm"] = pd.to_numeric(tracks.bp_bpm, errors="coerce")
    return sets, rows, tracks


def title_labels(sets: pd.DataFrame) -> pd.DataFrame:
    """Una columna booleana por palabra clave y una por etiqueta («L:<etiqueta>»)."""
    out = pd.DataFrame(index=sets.index)
    for _, name, rx in KEYWORDS:
        out[name] = sets.title.str.contains(rx, case=False, regex=True)
    for label in dict.fromkeys(k[0] for k in KEYWORDS):
        out["L:" + label] = out[[n for lab, n, _ in KEYWORDS if lab == label]].any(axis=1)
    return out


def positions(sets: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Filas que no suenan encima de otra, con su fracción del set `f` en [0, 1] y cómo se midió (`como`).

    Por tiempo si el set tiene hora de entrada en al menos CUE_MIN_SHARE de las filas, las horas no retroceden
    y la última no pasa del 120 % de la duración: f = hora / duración, con las filas sin hora interpoladas entre
    sus vecinas (0 s antes de la primera fila, la duración después de la última). Si no, por fila:
    f = (orden - 0,5) / filas. `f_fila` guarda siempre la segunda, para comparar."""
    info = sets.set_index("set_id")
    main = rows[rows.with_previous != "1"].sort_values(["set_id", "pos"])
    parts = []
    for sid, d in main.groupby("set_id", sort=False):
        n = len(d)
        d = d.copy()
        slot = np.arange(1, n + 1, dtype=float)
        d["f_fila"] = (slot - 0.5) / n
        total = float(info.at[sid, "min"]) * 60.0
        known = d.cue.dropna().to_numpy()
        by_time = (info.at[sid, "swc"] >= CUE_MIN_SHARE and len(known) >= 2 and bool((np.diff(known) >= 0).all())
                   and known.max() <= 1.2 * total)
        if by_time:
            x = np.r_[0.0, slot, n + 1.0]
            y = np.r_[0.0, d.cue.to_numpy(float), total]
            ok = ~np.isnan(y)
            d["f"] = np.clip(np.interp(x[1:-1], x[ok], y[ok]) / total, 0.0, 1.0)
            d["como"] = "tiempo"
        else:
            d["f"] = d.f_fila
            d["como"] = "fila"
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def table(sets: pd.DataFrame, rows: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    """Filas propias (sin «w/») con posición, BPM y género de Beatport, y los datos del set."""
    pos = positions(sets, rows)
    pos = pos.merge(tracks[["track_id", "bpm", "bp_genre"]], on="track_id", how="left")
    pos["bp_genre"] = pos.bp_genre.fillna("")
    pos.loc[pos.track_id == "", "bpm"] = np.nan
    pos.loc[pos.track_id == "", "bp_genre"] = ""
    cols = ["set_id", "dj", "split_key", "min", "b2b", "trayectoria", "title"]
    return pos.merge(sets[cols], on="set_id", how="left", validate="many_to_one")


def set_families(T: pd.DataFrame) -> pd.Series:
    """Familia de estilo de cada set: la de FAMILIAS con más filas con género de Beatport; «otros» si gana el resto."""
    g2f = {g: f for f, gs in FAMILIAS.items() for g in gs}
    G = T[T.bp_genre != ""].assign(fam=lambda d: d.bp_genre.map(g2f).fillna("otros"))
    return G.groupby(["set_id", "fam"]).size().unstack(fill_value=0).idxmax(axis=1)


def in_slice(B: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    return B[(B.f >= lo) & ((B.f < hi) if hi < 1.0 else (B.f <= hi))]


def bins_of(f: pd.Series, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, f.to_numpy(float), side="right") - 1, 0, len(edges) - 2)


# ---------------------------------------------------------------- estadística

def weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    order = np.argsort(x)
    x, w = x[order], w[order]
    c = np.cumsum(w) - 0.5 * w
    return float(np.interp(q * w.sum(), c, x))


def boot_percentiles(B: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """Intervalos del 95 % de p25, p50 y p75 del BPM, remuestreando DJs enteros: matriz (3, 2)."""
    groups = [g.to_numpy(float) for _, g in B.groupby("split_key").bpm]
    k = len(groups)
    res = np.empty((n_boot, 3))
    for i in range(n_boot):
        pick = rng.integers(0, k, k)
        res[i] = np.percentile(np.concatenate([groups[j] for j in pick]), [25, 50, 75])
    return np.percentile(res, [2.5, 97.5], axis=0).T


def genre_shares(B: pd.DataFrame) -> pd.Series:
    return B.loc[B.bp_genre != "", "bp_genre"].value_counts(normalize=True)


def resumen(B: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> Dict:
    """Conteos, BPM (banda p25-p75, mediana, intervalos por DJ, mediana con cada DJ pesando igual) y géneros."""
    Bb = B[B.bpm.notna()]
    out: Dict = {"n_sets": int(B.set_id.nunique()), "n_djs": int(B.split_key.nunique()),
                 "n_filas": int(len(B)), "n_temas": int(len(Bb)), "temas_distintos": int(Bb.track_id.nunique()),
                 "n_con_genero": int((B.bp_genre != "").sum())}
    if len(Bb) == 0:
        return out
    x = Bb.bpm.to_numpy(float)
    p25, p50, p75 = np.percentile(x, [25, 50, 75])
    w = 1.0 / Bb.groupby("split_key").bpm.transform("size").to_numpy(float)
    ci = boot_percentiles(Bb, rng, n_boot)
    top_dj = Bb.split_key.value_counts(normalize=True)
    out.update({
        "p25": float(p25), "p50": float(p50), "p75": float(p75), "media": float(x.mean()),
        "ic_p25": ci[0].tolist(), "ic_p50": ci[1].tolist(), "ic_p75": ci[2].tolist(),
        "p50_por_dj": weighted_quantile(x, w, 0.5),
        "banda_por_dj": [weighted_quantile(x, w, 0.25), weighted_quantile(x, w, 0.75)],
        "en_banda": float(((x >= p25) & (x <= p75)).mean()),
        "fuera_95_150": int(((x < 95) | (x > 150)).sum()),
        "dj_mayor": [str(top_dj.index[0]), float(top_dj.iloc[0])],
        "generos": genre_shares(B),
    })
    return out


def within_set(B: pd.DataFrame, edges: np.ndarray, rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    """Δ set: en cada set y tramo, la mediana de (BPM menos la mediana de BPM de todo el set); después la media
    entre sets (cada set pesa igual). Medianas y no medias para que un tema listado a medio tiempo (65 en lugar
    de 130) no arrastre el tramo. Intervalo del 95 % remuestreando DJs enteros."""
    nb = len(edges) - 1
    Bb = B[B.bpm.notna()].copy()
    Bb["d"] = Bb.bpm - Bb.groupby("set_id").bpm.transform("median")
    Bb["bin"] = bins_of(Bb.f, edges)
    m = Bb.groupby(["set_id", "bin"]).d.median().unstack().reindex(columns=range(nb))
    dj = Bb.groupby("set_id").split_key.first().reindex(m.index)
    M = m.to_numpy(float)
    idx = [np.flatnonzero((dj == k).to_numpy()) for k in dj.unique()]
    k = len(idx)
    res = np.empty((n_boot, nb))
    for i in range(n_boot):
        sel = np.concatenate([idx[j] for j in rng.integers(0, k, k)])
        res[i] = np.nanmean(M[sel], axis=0)
    lo, hi = np.percentile(res, [2.5, 97.5], axis=0)
    return pd.DataFrame({"delta": np.nanmean(M, axis=0), "lo": lo, "hi": hi}, index=range(nb))


# ---------------------------------------------------------------- impresión

def print_resumen(name: str, r: Dict, base: Optional[pd.Series] = None, top: int = TOP_GENEROS) -> None:
    print(f"\n  [{name}] {r['n_sets']} sets, {r['n_djs']} DJs, {r['n_temas']} filas con BPM "
          f"({r['temas_distintos']} temas distintos), {r['n_con_genero']} con género")
    if "p50" not in r:
        return
    print(f"    BPM p25-p50-p75 {r['p25']:.0f}-{r['p50']:.1f}-{r['p75']:.0f} (IC por DJ: p25 {r['ic_p25'][0]:.0f}-"
          f"{r['ic_p25'][1]:.0f}, p50 {r['ic_p50'][0]:.1f}-{r['ic_p50'][1]:.1f}, p75 {r['ic_p75'][0]:.0f}-"
          f"{r['ic_p75'][1]:.0f}); media {r['media']:.1f}; cada DJ igual: p50 {r['p50_por_dj']:.1f}, banda "
          f"{r['banda_por_dj'][0]:.1f}-{r['banda_por_dj'][1]:.1f}")
    print(f"    {r['en_banda']:.0%} de las filas dentro de la banda; {r['fuera_95_150']} fuera de 95-150 BPM; "
          f"el DJ con más filas: {r['dj_mayor'][0]} ({r['dj_mayor'][1]:.0%})")
    cells = []
    for genre, share in r["generos"].head(top).items():
        lift = f" x{share / base[genre]:.2f}" if base is not None and base.get(genre, 0) > 0 else ""
        cells.append(f"{genre} {share:.1%}{lift}")
    print("    géneros: " + "; ".join(cells))


def tramo(edges: np.ndarray, b: int) -> str:
    return f"{round(100 * edges[b]):>3}-{round(100 * edges[b + 1]):<3} %"


def print_curve(B: pd.DataFrame, edges: np.ndarray, rng: np.random.Generator, n_boot: int,
                genres: Sequence[str]) -> None:
    nb = len(edges) - 1
    curve = within_set(B, edges, rng, n_boot)
    Bb = B.assign(bin=bins_of(B.f, edges))
    print(f"    {'tramo':>9} {'filas':>6} {'sets':>5} {'p25':>5} {'p50':>5} {'p75':>5} {'media':>6}   Δ set (IC por DJ)"
          f"   identif.  con BPM")
    for b in range(nb):
        x = Bb[Bb.bin == b]
        y = x[x.bpm.notna()]
        c = curve.loc[b]
        ident = (x.track_id != "").mean()
        con_bpm = x[x.track_id != ""].bpm.notna().mean()
        print(f"    {tramo(edges, b)} {len(y):>6} {y.set_id.nunique():>5} {y.bpm.quantile(.25):>5.0f} "
              f"{y.bpm.median():>5.1f} {y.bpm.quantile(.75):>5.0f} {y.bpm.mean():>6.1f}   "
              f"{c.delta:>+5.2f} ({c.lo:+.2f} a {c.hi:+.2f})   {ident:>6.0%} {con_bpm:>8.0%}")
    print("    géneros por tramo (fracción de las filas con género de Beatport):")
    print("    " + f"{'tramo':>9} " + " ".join(f"{g[:14]:>14}" for g in genres))
    for b in range(nb):
        g = Bb[(Bb.bin == b) & (Bb.bp_genre != "")].bp_genre.value_counts(normalize=True)
        print("    " + f"{tramo(edges, b)} " + " ".join(f"{g.get(x, 0.0):>14.1%}" for x in genres))


def quick_band(B: pd.DataFrame) -> str:
    x = B.bpm.dropna()
    if len(x) == 0:
        return "sin BPM"
    return f"{x.quantile(.25):.0f}-{x.median():.1f}-{x.quantile(.75):.0f}"


# ---------------------------------------------------------------- tabla

def moment_entry(r: Dict, de_donde: str) -> Dict:
    enough = r["n_sets"] >= MIN_SETS and r["n_djs"] >= MIN_DJS and "p50" in r
    if not enough:
        return {"bpm": None, "bpm_mediana": None, "generos": {}, "n_sets": r["n_sets"], "n_djs": r["n_djs"],
                "n_temas": r["n_temas"], "de_donde": de_donde}
    g = r["generos"].head(TOP_GENEROS)
    return {"bpm": [round(r["p25"], 1), round(r["p75"], 1)], "bpm_mediana": round(r["p50"], 1),
            "generos": {k: round(float(v), 3) for k, v in g.items()}, "n_sets": r["n_sets"], "n_djs": r["n_djs"],
            "n_temas": r["n_temas"], "de_donde": de_donde}


def head_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "desconocido"


def main() -> int:
    ap = argparse.ArgumentParser(description="Tabla de momentos de la noche medida en DJSets v1")
    ap.add_argument("--metadata", default=str(DEFAULT_METADATA), help="carpeta metadata/ de djsets_v1")
    ap.add_argument("--json", default=None, help="escribe la tabla propuesta en esta ruta (relativa al repo o absoluta)")
    ap.add_argument("--boot", type=int, default=1000, help="remuestreos bootstrap por DJ")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--titulos", action="store_true", help="lista cada set que coincide con una palabra clave")
    args = ap.parse_args()
    metadata = Path(args.metadata)
    rng = np.random.default_rng(args.semilla)

    sets, rows, tracks = load(metadata)
    T = table(sets, rows, tracks)
    ident = T[T.track_id != ""]
    print(f"[datos] {metadata}")
    print(f"  {len(sets)} sets, {sets.split_key.nunique()} DJs por primer nombre ({sets.dj.nunique()} nombres de DJ), "
          f"{len(rows)} filas: {(rows.with_previous == '1').sum()} «w/» fuera, quedan {len(T)}; {len(ident)} "
          f"identificadas, {ident.bpm.notna().sum()} con BPM de Beatport, {(ident.bp_genre != '').sum()} con género")
    tr = sets[sets.trayectoria]
    print(f"  sets largos (use con trayectoria): {len(tr)} sets, {tr.split_key.nunique()} DJs; posición por tiempo en "
          f"{T[T.trayectoria & (T.como == 'tiempo')].set_id.nunique()}, por fila en "
          f"{T[T.trayectoria & (T.como == 'fila')].set_id.nunique()}")

    # ---- fuente 1: títulos
    L = title_labels(sets)
    print("\n== 1. Títulos: palabras clave (sets que coinciden: todos / largos)")
    for label, name, rx in KEYWORDS:
        m = L[name]
        print(f"  {label:<12} {name:<58} {int(m.sum()):>4} / {int((m & sets.trayectoria).sum()):>3}   /{rx}/")
    print("  «temporada» y «marca» no son momento; «atardecer» no es ninguno de los cinco.")
    for label in ETIQUETAS_MOMENTO + ["atardecer", "noche_entera"]:
        m = L["L:" + label]
        print(f"  {label:<12} {int(m.sum()):>4} sets, {sets[m].split_key.nunique():>3} DJs")
    if args.titulos:
        for label in ETIQUETAS_MOMENTO + ["temporada", "marca", "atardecer", "noche_entera"]:
            print(f"\n  -- {label}")
            for _, s in sets[L["L:" + label]].sort_values("title").iterrows():
                print(f"    {s.set_id:<9} {s['min']:>4.0f} min {s.use:<17} {s.kind:<8} {s.title}")

    base_all = genre_shares(T)
    print("\n  Todas las filas de los sets de cada etiqueta, frente a todos los sets (xN: N veces la proporción "
          "del género en todos los sets):")
    print_resumen("todos los sets", resumen(T, rng, args.boot))
    titulo: Dict[str, Dict] = {}
    for label in ETIQUETAS_MOMENTO + ["atardecer"]:
        B = T[T.set_id.isin(set(sets[L["L:" + label]].set_id))]
        titulo[label] = resumen(B, rng, args.boot)
        print_resumen(f"título {label}", titulo[label], base_all)
        if titulo[label]["n_sets"] > 1:
            top = titulo[label]["dj_mayor"][0]
            rest = B[B.split_key != top]
            print(f"    sin {top}: {rest.set_id.nunique()} sets, {rest.split_key.nunique()} DJs, p25-p50-p75 "
                  f"{quick_band(rest)}; géneros: "
                  + "; ".join(f"{g} {v:.0%}" for g, v in genre_shares(rest).head(4).items()))
    for name, rx in [("after: solo after party / after hours", KEYWORDS[4][2]),
                     ("after: solo sunrise / dawn patrol", KEYWORDS[5][2] + "|" + KEYWORDS[6][2])]:
        ids = set(sets[sets.title.str.contains(rx, case=False, regex=True)].set_id)
        print_resumen(name, resumen(T[T.set_id.isin(ids)], rng, args.boot), base_all)

    # ---- fuente 2: posición
    long_ = T[T.trayectoria]
    ne_ids = set(sets[sets.trayectoria & (L["L:noche_entera"] | (sets["min"] >= NOCHE_MIN))].set_id)
    anl_ids = set(sets[sets.trayectoria & L["L:noche_entera"]].set_id)
    main_pop = "noche entera (título o 6 h o más)"
    pops = {
        main_pop: long_[long_.set_id.isin(ne_ids)],
        "todos los sets largos (3 h o más)": long_,
        "solo títulos All Night Long y similares": long_[long_.set_id.isin(anl_ids)],
        "sets largos que no son noche entera": long_[~long_.set_id.isin(ne_ids)],
    }
    top_genres = list(genre_shares(long_).head(GENEROS_CURVA).index)
    print("\n== 2. Posición dentro de los sets largos")
    print("  Δ set = mediana, en el set y el tramo, de BPM menos la mediana del set; media entre sets.")
    print("  identif. = filas identificadas; con BPM = de las identificadas, las que tienen BPM de Beatport.")
    for name, B in pops.items():
        sets_dj = B.drop_duplicates("set_id").split_key.value_counts()
        print(f"\n  -- {name}: {B.set_id.nunique()} sets, {B.split_key.nunique()} DJs; b2b {int((B.drop_duplicates('set_id').b2b == '1').sum())}; "
              f"DJs con más sets: " + ", ".join(f"{k} {v}" for k, v in sets_dj.head(5).items()))
        print_curve(B, QUINTOS, rng, args.boot, top_genres)
    for name in [main_pop, "todos los sets largos (3 h o más)"]:
        print(f"\n  -- deciles, {name}")
        print_curve(pops[name], DECILES, rng, args.boot, top_genres)

    for name in [main_pop, "todos los sets largos (3 h o más)"]:
        B = pops[name]
        x = B[B.bpm.notna()].assign(q=lambda d: bins_of(d.f, QUINTOS))
        m = x.groupby(["set_id", "q"]).bpm.median().unstack()
        lento = m.idxmin(axis=1).value_counts().reindex(range(5), fill_value=0)
        rapido = m.idxmax(axis=1).value_counts().reindex(range(5), fill_value=0)
        print(f"\n  {name}, por quinto (1.º a 5.º):")
        print(f"    sets cuyo quinto más lento es ese (mediana de BPM; empate = el primero): {lento.tolist()}")
        print(f"    sets cuyo quinto más rápido es ese: {rapido.tolist()}")
        print("    filas con BPM de 130 o más: " + " ".join(f"{v:.1%}" for v in x.groupby("q").bpm.apply(lambda v: (v >= 130).mean())))
        print("    filas con BPM de menos de 110: " + " ".join(f"{v:.1%}" for v in x.groupby("q").bpm.apply(lambda v: (v < 110).mean())))

    med = long_[long_.bpm.notna()].groupby("set_id").bpm.median()
    print("\n  Escala: mediana de BPM de cada set largo, entre sets: "
          f"p10 {med.quantile(.1):.1f}, p25 {med.quantile(.25):.1f}, p50 {med.median():.1f}, "
          f"p75 {med.quantile(.75):.1f}, p90 {med.quantile(.9):.1f}; desvío {med.std():.2f}")
    st_ = long_[(long_.como == "tiempo")]
    gap = (st_.f - st_.f_fila).abs()
    print(f"  Control: en los sets largos medidos por tiempo, |f por tiempo - f por fila| mediana {gap.median():.3f}, "
          f"p90 {gap.quantile(.9):.3f}")

    # ---- por familia de estilo del set
    fam = set_families(T)
    print("\n== 2b. Por familia de estilo del set (la familia con más filas del set; xN frente a la misma familia "
          "entera)")
    for name in [main_pop, "todos los sets largos (3 h o más)"]:
        B = pops[name].assign(fam=lambda d: d.set_id.map(fam))
        comp = B.drop_duplicates("set_id").fam.value_counts()
        print(f"\n  -- {name}: " + ", ".join(f"{k} {v} sets" for k, v in comp.items()))
        edges_m = np.array([v[0] for v in MOMENTOS_POSICION.values()] + [1.0])
        for f in FAMILIAS:
            Bf = B[B.fam == f]
            if Bf.empty:
                continue
            delta = within_set(Bf, edges_m, rng, args.boot)
            base = genre_shares(Bf)
            print(f"    {f}: {Bf.set_id.nunique()} sets, {Bf.split_key.nunique()} DJs")
            for i, (mom, (lo, hi)) in enumerate(MOMENTOS_POSICION.items()):
                S = in_slice(Bf, lo, hi)
                d = delta.loc[i]
                gs = genre_shares(S).head(3)
                print(f"      {mom:<9} p25-p50-p75 {quick_band(S):>15}  Δ set {d.delta:+.1f} ({d.lo:+.1f} a {d.hi:+.1f})  "
                      + "; ".join(f"{k} {v:.0%} x{v / base[k]:.2f}" for k, v in gs.items()))

    # ---- momentos
    print("\n== 3. Momentos (posición: noche entera; after: títulos). xN frente a la misma población entera")
    edges_m = np.array([v[0] for v in MOMENTOS_POSICION.values()] + [1.0])
    resultados: Dict[str, Dict[str, Dict]] = {}
    deltas: Dict[str, pd.DataFrame] = {}
    bases: Dict[str, pd.Series] = {}
    for name, B in pops.items():
        base = genre_shares(B)
        delta = within_set(B, edges_m, rng, args.boot)
        deltas[name], bases[name] = delta, base
        resultados[name] = {}
        print(f"\n  -- {name}")
        for i, (mom, (lo, hi)) in enumerate(MOMENTOS_POSICION.items()):
            r = resumen(in_slice(B, lo, hi), rng, args.boot)
            resultados[name][mom] = r
            print_resumen(f"{mom} {lo:.0%}-{hi:.0%}", r, base)
            d = delta.loc[i]
            print(f"    Δ set {d.delta:+.2f} BPM (IC por DJ {d.lo:+.2f} a {d.hi:+.2f})")

    print("\n  Sensibilidad, noche entera: p25-p50-p75 por momento")
    B0 = pops[main_pop]
    variants = {
        "tabla": B0,
        "sin b2b": B0[B0.b2b != "1"],
        "solo posición por tiempo": B0[B0.como == "tiempo"],
        "posición por fila en todos": B0.assign(f=B0.f_fila),
        "sin BPM fuera de 95-150": B0[B0.bpm.isna() | B0.bpm.between(95, 150)],
    }
    print(f"    {'variante':<28} " + " ".join(f"{m:>16}" for m in MOMENTOS_POSICION) + "   sets")
    for vname, V in variants.items():
        cells = [quick_band(in_slice(V, lo, hi)) for lo, hi in MOMENTOS_POSICION.values()]
        print(f"    {vname:<28} " + " ".join(f"{c:>16}" for c in cells) + f"   {V.set_id.nunique()}")
    print(f"    apertura solo 0-10 %: {quick_band(in_slice(B0, 0.0, 0.1))}; pico 40-80 %: "
          f"{quick_band(in_slice(B0, 0.4, 0.8))}; cierre solo 90-100 %: {quick_band(in_slice(B0, 0.9, 1.0))}")

    print("\n  De cuántos DJs sale cada género de la tabla (noche entera): filas, DJs, el DJ con más filas")
    for mom, (lo, hi) in MOMENTOS_POSICION.items():
        S = in_slice(B0, lo, hi)
        cells = []
        for g in resultados[main_pop][mom]["generos"].head(TOP_GENEROS).index:
            vc = S[S.bp_genre == g].split_key.value_counts()
            cells.append(f"{g} {vc.sum()}/{vc.size}/{vc.index[0]} {vc.iloc[0] / vc.sum():.0%}")
        print(f"    {mom}: " + "; ".join(cells))

    tabla: Dict[str, Dict] = {mom: moment_entry(resultados[main_pop][mom], "posicion") for mom in MOMENTOS_POSICION}
    tabla["after"] = moment_entry(titulo["after"], "titulos")
    for i, mom in enumerate(MOMENTOS_POSICION):                 # corrimiento dentro del set, para el ejecutor
        if tabla[mom]["bpm"] is not None:
            tabla[mom]["bpm_delta"] = round(float(deltas[main_pop].loc[i].delta), 1)

    print("\n== 4. Las dos fuentes, donde se pueden comparar")
    rp = resultados[main_pop]["cierre"]
    for label in ["cierre", "after"]:
        rt = titulo[label]
        if "p50" in rt:
            print(f"  título {label}: {rt['n_sets']} sets, {rt['n_djs']} DJs, p50 {rt['p50']:.1f} "
                  f"(IC {rt['ic_p50'][0]:.1f}-{rt['ic_p50'][1]:.1f}), banda {rt['p25']:.0f}-{rt['p75']:.0f}; "
                  f"cierre por posición: p50 {rp['p50']:.1f} (IC {rp['ic_p50'][0]:.1f}-{rp['ic_p50'][1]:.1f}), "
                  f"banda {rp['p25']:.0f}-{rp['p75']:.0f}")
    for label in ["apertura", "pico"]:
        print(f"  título {label}: {titulo[label]['n_sets']} sets; no hay con qué comparar")

    print("\n== Tabla propuesta")
    for mom in MOMENTOS:
        e = tabla[mom]
        gen = ", ".join(f"{k} {v:.0%}" for k, v in list(e["generos"].items())[:3]) or "-"
        print(f"  {mom:<9} bpm {e['bpm']} mediana {e['bpm_mediana']}; {e['n_sets']} sets, {e['n_djs']} DJs, "
              f"{e['n_temas']} filas con BPM; {e['de_donde']}; {gen}")

    if args.json:
        out = Path(args.json)
        if not out.is_absolute():
            out = REPO_ROOT / out
        doc = {
            "version": 1,
            "estado": "propuesta",
            "fuente": ("DJSets v1 (erclm:traktor_ml/djsets_v1, tablas del 2026-11-13). apertura, subida, pico y "
                       f"cierre: posición (0-20, 20-60, 60-80 y 80-100 %) en {len(ne_ids)} sets de noche entera; "
                       "after: títulos. bpm = percentiles 25 y 75 de las filas; generos = fracción de las filas con "
                       "género de Beatport. tools/dj_sets/momentos_noche.py; docs/reports/momentos_djsets_2026-11-13.md"),
            "commit": head_commit(),
            "generos_noche": {k: round(float(v), 3) for k, v in bases[main_pop].head(20).items()},
            "momentos": tabla,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
        print(f"\n[OK] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
