"""
PURPOSE: Banco de evaluación E1 del plan de playlists en vivo (docs/plans/recomendacion_playlists_plan.md,
         sección 3.6, con las medidas de la auditoría 2026-11-07). Repite sets reales: en cada transición
         (tema a -> tema b) el tema principal es a, lo tocado son los temas anteriores del set, y los
         candidatos salen del recomendador en vivo (tools/traktor_live/recomendador.py: la colección de
         Traktor, BPM ±20, tonalidad compatible, sin lo tocado). Mide dónde queda b:
           acierto_30       b entre los 30 primeros (lo que Gabriel ve);
           rr_medio         rango recíproco medio (0 si b no está en la lista);
           percentil        puesto de b dividido por el número de candidatos (mediana; 0 es el primero);
           recuperacion     fracción de las b del pool que pasan los filtros (la medida de P3);
           fuera_del_pool   b no es sugerible (no está en la colección o no tiene tonalidad).
         Perfiles: «azar» (orden aleatorio, semilla fija), y pesos del recomendador ({"clap": 1},
         {"bpm": 1}, ...). Lee las filas del almacén de sets (tools/dj_sets/sets_traktor.py: d1.csv).
         Por la regla E6 del plan, lo medido sobre D1 (sets de Gabriel) orienta y no fija pesos.
         Desempate al azar (auditoría de E1, 2026-11-07): bpm, clave y genero toman pocos valores y el
         recomendador ordena de forma estable, así que dentro de un empate mandaba el orden del índice.
         Antes de cada consulta se mezcla el orden de los candidatos con una semilla, y se repite con
         varias semillas (media, mínimo y máximo). Con la primera semilla se informa además un intervalo
         del 95 % por remuestreo de sets (las transiciones de un set no son independientes) y el acierto
         de cada transición, para comparaciones apareadas.
CHANGELOG:
  - 2026-11-07: Creación (etapa 1 del plan: E1 sobre D1 con líneas de base).
  - 2026-11-07: Desempate al azar con semillas, intervalo por remuestreo de sets y aciertos por
                transición (propuesta 1 de docs/reports/auditoria_e1_sets_gabriel_2026-11-07.md).
  - 2026-11-07: Perfiles con opciones del recomendador (filtro de tonalidad, penalización por semitono,
                escala de CLAP), para probar la tonalidad como componente (Gabriel, 2026-11-07).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from tools.traktor_live import recomendador as rc  # noqa: E402

PROFILES: Dict[str, Optional[Dict[str, float]]] = {
    "azar": None,
    "bpm": {"clap": 0.0, "bpm": 1.0},
    "clave": {"clap": 0.0, "clave": 1.0},
    "clap": {"clap": 1.0},
    "genero": {"clap": 0.0, "genero": 1.0},
    "clap+bpm": {"clap": 1.0, "bpm": 1.0},
    "clap sin filtro": {"clap": 1.0},
    "clap% + clave 0.25 sin filtro": {"clap": 1.0, "clave": 0.25},
    "clap% + clave 0.5 sin filtro": {"clap": 1.0, "clave": 0.5},
    "clap% + clave 1 sin filtro": {"clap": 1.0, "clave": 1.0},
    "clap% + clave 0.5 con filtro": {"clap": 1.0, "clave": 0.5},
}
NO_FILTER = {"filtro_clave": False, "clave_penalizacion": 0.05}
OPTIONS: Dict[str, Dict[str, object]] = {
    "clap sin filtro": NO_FILTER,
    "clap% + clave 0.25 sin filtro": {**NO_FILTER, "clap_escala": "percentil"},
    "clap% + clave 0.5 sin filtro": {**NO_FILTER, "clap_escala": "percentil"},
    "clap% + clave 1 sin filtro": {**NO_FILTER, "clap_escala": "percentil"},
    "clap% + clave 0.5 con filtro": {"clave_penalizacion": 0.05, "clap_escala": "percentil"},
}


def load_sets(path: Path) -> Dict[str, List[Dict]]:
    sets: Dict[str, List[Dict]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            sets.setdefault(r["set_id"], []).append(r)
    for v in sets.values():
        v.sort(key=lambda r: int(r["posicion"]))
    return sets


def bootstrap_sets(per_set: Dict[str, List[int]], n: int = 1000, seed: int = 0) -> List[float]:
    """Intervalo del 95 % de la fracción de aciertos remuestreando sets enteros (con reposición)."""
    keys = [k for k, v in per_set.items() if v]
    if not keys:
        return [0.0, 0.0]
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        pick = [rng.choice(keys) for _ in keys]
        vals.append(sum(sum(per_set[k]) for k in pick) / sum(len(per_set[k]) for k in pick))
    vals.sort()
    return [round(vals[int(0.025 * n)], 3), round(vals[int(0.975 * n) - 1], 3)]


def evaluate(rec: rc.Recommender, items: Dict[str, rc.Item], sets: Dict[str, List[Dict]],
             weights: Optional[Dict[str, float]], seed: int = 0, top: int = 30,
             options: Optional[Dict[str, object]] = None) -> Dict:
    rng = random.Random(seed)
    if weights is not None:
        rec.weights = rc.weights_from(weights)
    if hasattr(rec, "options"):
        rec.options = rc.options_from(options)
    pool = {it.uid for it in rec.candidates}
    original_order = list(rec.candidates)
    hits, rr, pct, passed, in_pool, out_pool, n = 0, [], [], 0, 0, 0, 0
    per_set: Dict[str, List[int]] = {}
    per_transition: List[int] = []
    for sid, rows in sets.items():
        for i in range(1, len(rows)):
            a, b = rows[i - 1]["track_uid"], rows[i]["track_uid"]
            if not a or not b or a not in items:
                continue
            n += 1
            if b not in pool:
                out_pool += 1
                continue
            in_pool += 1
            main = items[a]
            principal = {"path": str(main.path or main.original or main.rel), "bpm": main.bpm, "tempo": 1.0,
                         "key": None}
            played = [str(items[r["track_uid"]].path or items[r["track_uid"]].rel)
                      for r in rows[:i - 1] if r["track_uid"] in items]
            rng.shuffle(rec.candidates)                   # desempate al azar (el orden es estable)
            _, info = rec.compute(principal, played)
            uids = list(info["uids"])
            if weights is None:
                rng.shuffle(uids)
            hit = 0
            if b in uids:
                passed += 1
                k = uids.index(b) + 1
                hit = int(k <= top)
                rr.append(1.0 / k)
                pct.append((k - 1) / max(1, len(uids) - 1))
            else:
                rr.append(0.0)
            hits += hit
            per_set.setdefault(sid, []).append(hit)
            per_transition.append(hit)
    rec.candidates[:] = original_order
    return {"transiciones": n, "en_el_pool": in_pool, "fuera_del_pool": out_pool,
            "recuperacion": round(passed / in_pool, 3) if in_pool else None,
            "acierto_30": round(hits / in_pool, 3) if in_pool else None,
            "acierto_30_de_las_que_pasan": round(hits / passed, 3) if passed else None,
            "rr_medio": round(st.mean(rr), 4) if rr else None,
            "percentil_mediano": round(st.median(pct), 3) if pct else None,
            "intervalo_acierto_30_por_sets": bootstrap_sets(per_set, seed=seed),
            "aciertos_por_transicion": per_transition}


def over_seeds(rec: rc.Recommender, items: Dict[str, rc.Item], sets: Dict[str, List[Dict]],
               weights: Optional[Dict[str, float]], seeds: List[int],
               options: Optional[Dict[str, object]] = None) -> Dict:
    """evaluate con cada semilla: la primera completa y, de las medidas, media, mínimo y máximo."""
    runs = [evaluate(rec, items, sets, weights, seed=s, options=options) for s in seeds]
    out = dict(runs[0])
    for m in ("acierto_30", "rr_medio", "percentil_mediano"):
        vals = [r[m] for r in runs if r[m] is not None]
        if vals:
            out[m + "_semillas"] = {"media": round(st.mean(vals), 4), "min": min(vals), "max": max(vals)}
    out["semillas"] = list(seeds)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Banco E1: repite sets reales con el recomendador en vivo")
    ap.add_argument("--sets", required=True, help="CSV del almacén de sets (p. ej. <root>/sets/d1.csv)")
    ap.add_argument("--indice", default=str(rc.DEFAULT_INDEX))
    ap.add_argument("--perfil", action="append", default=None, help=f"Perfiles (por defecto todos: {', '.join(PROFILES)})")
    ap.add_argument("--salida", default=None, help="JSON de resultados (por defecto junto al CSV)")
    ap.add_argument("--semillas", type=int, default=5, help="Semillas del desempate al azar")
    args = ap.parse_args()
    from tools.traktor_live.sugerencias_explorer import entries_by_path
    from tools.traktor_live.sugerencias_itunes import find_collection
    _, items = rc.load_index(Path(args.indice))
    rec = rc.Recommender(items, entries_by_path(find_collection()), n=10 ** 6, taxonomy=rc.load_taxonomy()[0])
    by_uid = {it.uid: it for it in items}
    sets = load_sets(Path(args.sets))
    res = {"sets": len(sets), "candidatos_sugeribles": len(rec.candidates), "perfiles": {}}
    for name in args.perfil or list(PROFILES):
        res["perfiles"][name] = over_seeds(rec, by_uid, sets, PROFILES[name], list(range(args.semillas)),
                                           OPTIONS.get(name))
        res["perfiles"][name]["opciones"] = rc.options_from(OPTIONS.get(name))
        shown = {k: v for k, v in res["perfiles"][name].items() if k != "aciertos_por_transicion"}
        print(name, json.dumps(shown, ensure_ascii=False), flush=True)
    out = Path(args.salida) if args.salida else Path(args.sets).with_name(Path(args.sets).stem + "_e1.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
