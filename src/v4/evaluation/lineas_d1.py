"""
PURPOSE: Dos líneas del plan de playlists en vivo medidas con el banco E1 sobre los sets de Gabriel (D1,
         tools/dj_sets/sets_traktor.py), después de la revalidación de D7:
           1. Tonalidad aprendida (plan, P7). En lugar de los valores de la regla armónica, el componente
              clave vale, por cada paso en la rueda Camelot (harmonic.wheel_step: «+0», «+0x», «+1», «-5»…),
              cuántas veces más que al azar Gabriel fue a un tema con ese paso: los pasos observados en sus
              transiciones frente a los que había en el pool de cada consulta (los candidatos con BPM ±20),
              con suavizado (+1). Los temas sin tonalidad son un paso más («?»). Se evalúa dejando un set
              fuera: la tabla sale de los otros sets. Con el filtro de tonalidad apagado y CLAP en percentil,
              contra la regla con el mismo peso y contra CLAP solo. Descomposición: los pasos solos (los
              temas sin tonalidad valen 0, también en la regla) y los temas sin tonalidad solos (todos los
              pasos iguales), porque Gabriel casi no va a temas sin tonalidad y eso puede ser otra cosa
              (temas recién agregados, por ejemplo).
           3. Comienzos y finales (--extremos <clap_extremos.npz>): el componente «extremos» del recomendador,
              el final del tema actual frente al comienzo del candidato (60 s o 30 s), solo y junto a CLAP,
              con el sentido inverso (comienzo del actual frente al final del candidato) como control: si
              lo que importa es la transición, el sentido correcto tiene que ganar.
           2. Zona de la noche. La consulta CLAP es el tema actual más un peso por el centro de lo ya tocado
              en el set (todo, o los últimos 5): recomendador, opción «clap_historia».
         Cada perfil con 5 semillas del desempate (media) y la prueba de McNemar con la primera (las mismas
         transiciones). Por la regla E6 del plan, lo medido sobre D1 orienta y no fija pesos.
           python src/v4/evaluation/lineas_d1.py --sets "D:/TRAKTOR ML datos/dj_sets/sets/d1.csv"
CHANGELOG:
  - 2026-11-07: Creación (Gabriel, 2026-11-07: «tienes el visto bueno para avanzar con todos los cálculos
                que creas prometedores»).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from src.v4.common import harmonic  # noqa: E402
from src.v4.evaluation.banco_sets import NO_FILTER, evaluate, load_sets  # noqa: E402
from tools.traktor_live import recomendador as rc  # noqa: E402

KEY_WEIGHTS = (0.25, 0.5, 1.0)
NIGHT = ((0.5, 0), (1.0, 0), (0.5, 5), (1.0, 5))      # (peso del centro, últimos n; 0 = todo el set)
PCT = {**NO_FILTER, "clap_escala": "percentil"}


def binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    pmf = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    return round(min(1.0, sum(q for q in pmf if q <= pmf[k] + 1e-15)), 4)


def mcnemar(a: List[int], b: List[int]) -> Dict:
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    return {"solo_el_primero": only_a, "solo_el_segundo": only_b, "p": binom_two_sided(only_a, only_a + only_b)}


def step_of(a: rc.Item, b: rc.Item) -> str:
    return "?" if b.key == "?" else (harmonic.wheel_step(a.key, b.key) or "?")


class PoolSteps:
    """Distribución de los pasos en el pool de cada tema principal (candidatos con BPM ±20, sin él)."""

    def __init__(self, rec: rc.Recommender):
        self.rec = rec
        self.cache: Dict[str, Dict[str, float]] = {}

    def __call__(self, a: rc.Item) -> Dict[str, float]:
        if a.uid not in self.cache:
            c = collections.Counter()
            for it in self.rec.candidates:
                if it.uid == a.uid or (a.bpm and (it.bpm is None or abs(it.bpm - a.bpm) > rc.BPM_RANGE)):
                    continue
                c[step_of(a, it)] += 1
            n = sum(c.values()) or 1
            self.cache[a.uid] = {k: v / n for k, v in c.items()}
        return self.cache[a.uid]


def learn_table(pool: PoolSteps, items: Dict[str, rc.Item], sets: Dict[str, List[Dict]],
                alpha: float = 1.0) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Valor por paso: (observadas + alpha) / (esperadas + alpha), dividido por el mayor; «otro» = el menor."""
    suggestible = {it.uid for it in pool.rec.candidates}
    obs, exp = collections.Counter(), collections.Counter()
    for rows in sets.values():
        for i in range(1, len(rows)):
            a, b = items.get(rows[i - 1]["track_uid"]), items.get(rows[i]["track_uid"])
            if a is None or b is None or a.key == "?" or b.uid not in suggestible:
                continue
            for k, p in pool(a).items():
                exp[k] += p
            obs[step_of(a, b)] += 1
    cats = set(obs) | set(exp)
    lift = {k: (obs[k] + alpha) / (exp[k] + alpha) for k in cats}
    top = max(lift.values())
    table = {k: round(v / top, 4) for k, v in lift.items()}
    table["otro"] = min(table.values())
    stats = {k: {"observadas": obs[k], "esperadas": round(exp[k], 1), "veces_el_azar": round(lift[k], 3)}
             for k in sorted(cats, key=lambda k: -lift[k])}
    return table, stats


def per_set(rec, items, sets, weights, options, seed, table_for=None) -> List[int]:
    """Aciertos en los 30 primeros por transición, set por set (el mismo orden para todos los perfiles)."""
    hits: List[int] = []
    for sid in sets:
        opts = dict(options)
        if table_for is not None:
            opts["clave_tabla"] = table_for(sid)
        hits += evaluate(rec, items, {sid: sets[sid]}, weights, seed=seed, options=opts)["aciertos_por_transicion"]
    return hits


def attach_ends(items: Dict[str, rc.Item], path: Path, seconds: int) -> int:
    """Pone en cada tema el comienzo y el final normalizados: 60 s (los dos bloques) o 30 s (el primero del
    comienzo y el último del final)."""
    import numpy as np
    with np.load(path) as z:
        uids, ini, fin = [str(u) for u in z["uids"]], z["comienzo"], z["final"]
    n = 0
    for k, u in enumerate(uids):
        it = items.get(u)
        if it is None:
            continue
        a = ini[k].mean(axis=0) if seconds == 60 else ini[k][0]
        b = fin[k].mean(axis=0) if seconds == 60 else fin[k][1]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        it.inicio = (a / np.linalg.norm(a)).tolist()
        it.fin = (b / np.linalg.norm(b)).tolist()
        n += 1
    return n


def run_ends(rec, items, sets, path: Path, args) -> int:
    res = {"sets": len(sets), "perfiles": {}, "apareadas": {}}
    first: Dict[str, List[int]] = {}
    base_opts = PCT
    for seconds in (60, 30):
        res[f"temas_con_extremos_{seconds}"] = attach_ends(items, path, seconds)
        profiles = {"clap sin filtro": ({"clap": 1.0}, NO_FILTER)}
        for sense in ("fin_inicio", "inicio_fin"):
            opts = {**base_opts, "extremos_sentido": sense}
            profiles[f"extremos {seconds} {sense}"] = ({"clap": 0.0, "extremos": 1.0}, opts)
            profiles[f"clap% + extremos {seconds} {sense}"] = ({"clap": 1.0, "extremos": 1.0}, opts)
            profiles[f"clap% + extremos {seconds} {sense} 0.5"] = ({"clap": 1.0, "extremos": 0.5}, opts)
        for name, (weights, options) in profiles.items():
            if name in first:
                continue
            runs = [per_set(rec, items, sets, weights, options, s) for s in range(args.semillas)]
            first[name] = runs[0]
            rates = [sum(r) / len(r) for r in runs]
            res["perfiles"][name] = {"transiciones": len(runs[0]), "acierto_30": round(st.mean(rates), 4),
                                     "min": round(min(rates), 4), "max": round(max(rates), 4)}
            print(f"{name}: {res['perfiles'][name]}", flush=True)
        for kind in ("extremos {s} {d}", "clap% + extremos {s} {d}", "clap% + extremos {s} {d} 0.5"):
            a, b = kind.format(s=seconds, d="fin_inicio"), kind.format(s=seconds, d="inicio_fin")
            res["apareadas"][f"{a} frente a {b}"] = mcnemar(first[a], first[b])
            res["apareadas"][f"{a} frente a clap sin filtro"] = mcnemar(first[a], first["clap sin filtro"])
    for k, v in res["apareadas"].items():
        print(f"  {k}: {v}", flush=True)
    out = Path(args.salida) if args.salida else Path(args.sets).with_name("d1_extremos.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Tonalidad aprendida y zona de la noche en E1 sobre D1")
    ap.add_argument("--sets", required=True)
    ap.add_argument("--indice", default=str(rc.DEFAULT_INDEX))
    ap.add_argument("--semillas", type=int, default=5)
    ap.add_argument("--salida", default=None)
    ap.add_argument("--extremos", default=None, help="clap_extremos.npz: corre solo los perfiles de comienzos y finales")
    args = ap.parse_args()
    from tools.traktor_live.sugerencias_explorer import entries_by_path
    from tools.traktor_live.sugerencias_itunes import find_collection
    _, items_l = rc.load_index(Path(args.indice))
    rec = rc.Recommender(items_l, entries_by_path(find_collection()), n=10 ** 6, taxonomy=rc.load_taxonomy()[0])
    items = {it.uid: it for it in items_l}
    sets = load_sets(Path(args.sets))
    if args.extremos:
        return run_ends(rec, items, sets, Path(args.extremos), args)
    pool = PoolSteps(rec)
    folds = {sid: learn_table(pool, items, {k: v for k, v in sets.items() if k != sid})[0] for sid in sets}
    full_table, full_stats = learn_table(pool, items, sets)
    profiles: Dict[str, Tuple[Dict[str, float], Dict, Optional[object]]] = {
        "clap sin filtro": ({"clap": 1.0}, NO_FILTER, None)}
    zero = {**PCT, "clave_sin_tonalidad": 0.0}

    def steps_only(sid: str) -> Dict[str, float]:          # sin el valor aprendido para «?»: vale 0
        return {k: v for k, v in folds[sid].items() if k != "?"}

    def keyless_only(sid: str) -> Dict[str, float]:        # todos los pasos iguales; solo «?» aprendido
        return {"otro": 1.0, "?": folds[sid]["?"]}
    for w in KEY_WEIGHTS:
        profiles[f"clap% + regla {w}"] = ({"clap": 1.0, "clave": w}, PCT, None)
        profiles[f"clap% + aprendida {w}"] = ({"clap": 1.0, "clave": w}, PCT, folds.get)
        # descomposición: los pasos en la rueda frente a los temas sin tonalidad
        profiles[f"clap% + regla {w}, sin tonalidad 0"] = ({"clap": 1.0, "clave": w}, zero, None)
        profiles[f"clap% + aprendida {w}, solo pasos, sin tonalidad 0"] = ({"clap": 1.0, "clave": w}, zero, steps_only)
        profiles[f"clap% + solo sin tonalidad aprendida {w}"] = ({"clap": 1.0, "clave": w}, PCT, keyless_only)
    for lam, n in NIGHT:
        profiles[f"noche {lam} ({'todo' if not n else f'últimos {n}'})"] = (
            {"clap": 1.0}, {**NO_FILTER, "clap_historia": lam, "clap_historia_n": n}, None)
    res = {"sets": len(sets), "tabla_completa": full_table, "pasos": full_stats, "perfiles": {}, "apareadas": {}}
    first: Dict[str, List[int]] = {}
    for name, (weights, options, table_for) in profiles.items():
        runs = [per_set(rec, items, sets, weights, options, s, table_for) for s in range(args.semillas)]
        first[name] = runs[0]
        rates = [sum(r) / len(r) for r in runs]
        res["perfiles"][name] = {"transiciones": len(runs[0]), "acierto_30": round(st.mean(rates), 4),
                                 "min": round(min(rates), 4), "max": round(max(rates), 4)}
        print(f"{name}: {res['perfiles'][name]}", flush=True)
    base = first["clap sin filtro"]
    for name, hits in first.items():
        if name == "clap sin filtro":
            continue
        res["apareadas"][f"{name} frente a clap sin filtro"] = mcnemar(hits, base)
    for w in KEY_WEIGHTS:
        res["apareadas"][f"aprendida {w} frente a regla {w}"] = mcnemar(first[f"clap% + aprendida {w}"],
                                                                         first[f"clap% + regla {w}"])
        res["apareadas"][f"solo pasos {w} frente a regla {w} (las dos con sin tonalidad 0)"] = mcnemar(
            first[f"clap% + aprendida {w}, solo pasos, sin tonalidad 0"], first[f"clap% + regla {w}, sin tonalidad 0"])
    for k, v in res["apareadas"].items():
        print(f"  {k}: {v}", flush=True)
    print("pasos (tabla con todos los sets):", flush=True)
    for k, v in list(full_stats.items())[:12]:
        print(f"  {k:5} {v}", flush=True)
    out = Path(args.salida) if args.salida else Path(args.sets).with_name("d1_lineas.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
