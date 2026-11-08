"""
PURPOSE: Trayectorias tema a tema en las compilaciones mezcladas (D7) y banco E1 sobre sets de otros DJs
         (plan de playlists en vivo, etapa 2, punto 5: T11 y E1 con D7; adenda 11-07).
           1. Almacén de sets D7: un tema por fila (compilación, disco, posición, inicio en la mezcla,
              artista, título, grabación de MusicBrainz, track_uid de la biblioteca si coincide por artista
              y título) con un vector CLAP: la media de los bloques de 30 s que caen dentro del tema lejos
              de los cambios (borde de 30, 60 o 90 s a cada lado). En las «continua» los límites salen de
              las duraciones de MusicBrainz; en las «por_tema» cada archivo es un tema. Cada disco es un
              set (son mezclas separadas).
           2. T11: distancia coseno entre temas vecinos frente al tema de dos puestos más allá y frente a
              pares al azar del mismo disco, de la misma compilación, de otras compilaciones y de la
              biblioteca, por borde. Si la ventaja de los vecinos se achica al agrandar el borde, parte de
              ella es audio de la transición filtrado en el vector. Paso y distancia al primer tema según
              la posición dentro del disco (quintos). Entre discos: distancia entre centroides de la misma
              compilación, de otra del mismo DJ y de otro DJ, por diferencia de años (DJ frente a época).
           3. Puente con la biblioteca: para los temas que coinciden por artista y título, el puesto del
              original de la biblioteca entre los 1760 por CLAP.
           4. E1 en D7: en cada transición a -> b, dónde queda b entre (i) los temas que quedan del disco
              (orden dentro del cajón, sin la señal de la compilación) y (ii) todos los temas de D7 menos lo
              tocado (con la señal de la compilación: se informa qué parte de los 30 primeros viene de la
              misma compilación). Los temas repetidos entre compilaciones cuentan una vez (identidad:
              artista principal y título sin paréntesis). Perfiles: azar, clap (el tema actual),
              historia (los últimos 3, con pesos 1, 0,5 y 0,25) y momento (el actual más el último paso).
         Escribe <root>/sets/d7_temas.csv, d7_clap.npz (vectores por borde, alineados con «fila») y
         d7_trayectorias.json.
           python tools/dj_sets/trayectorias_d7.py --root "D:/TRAKTOR ML datos/dj_sets"
CHANGELOG:
  - 2026-11-07: Creación.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from src.v4.evaluation.banco_sets import bootstrap_sets  # noqa: E402
from tools.dj_sets.piloto import bounds, unit  # noqa: E402
from tools.dj_sets.sets_traktor import norm_text  # noqa: E402
from tools.traktor_live import recomendador as rc  # noqa: E402

DEFAULT_ROOT = REPO_ROOT / "artifacts" / "v4" / "dj_sets"
BLOCK_S = 30.0
EDGES = (30, 60, 90)
PROFILES = ("azar", "clap", "historia", "momento")


def select_blocks(starts: np.ndarray, dur: float, a: float, b: float, edge: float) -> Tuple[np.ndarray, bool]:
    """Bloques enteros dentro de [a + edge, b - edge]; si no hay ninguno, los que tienen el centro dentro
    del tema (y se marca que el borde se redujo)."""
    ends = np.minimum(starts + BLOCK_S, dur)
    core = (starts >= a + edge) & (ends <= b - edge)
    if core.any():
        return np.where(core)[0], False
    mid = (starts + ends) / 2
    return np.where((mid >= a) & (mid < b))[0], True


def track_vector(blocks: np.ndarray, idx: np.ndarray) -> Optional[np.ndarray]:
    if len(idx) == 0:
        return None
    return unit(unit(blocks[idx]).mean(axis=0))


def identity(row: Dict) -> str:
    """El mismo tema en dos compilaciones cuenta una vez: artista principal y título sin paréntesis (las
    versiones «(Mixed)» de MusicBrainz suelen ser grabaciones distintas); la grabación si faltan."""
    a = norm_text((row.get("artista") or "").split(",")[0].split("&")[0])
    t = norm_text(row.get("titulo") or "")
    if a and t:
        return "at:" + a + "|" + t
    return "mb:" + (row.get("recording") or str(row.get("fila")))


def library_matcher(items: Sequence[rc.Item]):
    by_name: Dict[Tuple[str, str], str] = {}
    for it in items:
        a, t = norm_text(it.artist.split(",")[0].split("&")[0]), norm_text(it.title)
        if a and t:
            by_name.setdefault((a, t), it.uid)

    def match(artist: str, title: str) -> str:
        return by_name.get((norm_text(artist.split(",")[0].split("&")[0]), norm_text(title)), "")
    return match


def build_store(root: Path, match) -> Tuple[List[Dict], Dict[int, np.ndarray]]:
    """Filas del almacén D7 y sus vectores por borde ({borde: matriz con NaN donde no hay vector})."""
    rows: List[Dict] = []
    vecs: Dict[int, List[Optional[np.ndarray]]] = {e: [] for e in EDGES}
    for man_path in sorted((root / "audio").glob("d7_*/manifest.json")):
        man = json.loads(man_path.read_text(encoding="utf-8"))
        folder = man_path.parent.name
        mb = json.loads((root / "musicbrainz" / f"{man['release']}.json").read_text(encoding="utf-8"))
        files = [e for e in man["entradas"] if e.get("archivo")]
        pos_global = 0
        for disc in mb["media"]:
            n, tracks = disc["disco"], disc["pistas"]
            if man["forma"] == "continua":
                e = next((x for x in files if x["disco"] == n), None)
                p = root / "clap" / folder / (Path(e["archivo"]).stem + ".npz") if e else None
                z = np.load(p) if p is not None and p.exists() else None
                spans = bounds([t["ms"] for t in tracks])
                per_track = [(z, a, b) for a, b in spans] if z is not None else [None] * len(tracks)
            else:
                per_track = []
                for t in tracks:
                    e = next((x for x in files if x["disco"] == n and x["pista"] == t["posicion"]), None)
                    p = root / "clap" / folder / (Path(e["archivo"]).stem + ".npz") if e else None
                    if p is not None and p.exists():
                        z = np.load(p)
                        per_track.append((z, 0.0, float(z["dur"])))
                    else:
                        per_track.append(None)
            start = 0.0
            for i, (t, src) in enumerate(zip(tracks, per_track)):
                dur = (t["ms"] or 0) / 1000.0
                row = {"fila": len(rows), "set_id": f"D7_{man['grupo'][:8]}_d{n}", "grupo": man["grupo"],
                       "release": man["release"], "dj": man["artista"], "compilacion": mb.get("titulo") or man["titulo"],
                       "fecha": mb.get("fecha") or "", "disco": n, "pistas_disco": len(tracks), "posicion": i + 1,
                       "posicion_global": pos_global + i + 1, "inicio_s": round(start, 1), "dur_s": round(dur, 1),
                       "artista": t.get("artista") or "", "titulo": t.get("titulo") or "",
                       "recording": t.get("recording") or "", "forma": man["forma"],
                       "track_uid": match(t.get("artista") or "", t.get("titulo") or "")}
                for edge in EDGES:
                    v, nb, reduced = None, 0, False
                    if src is not None:
                        z, a, b = src
                        idx, reduced = select_blocks(z["starts"], float(z["dur"]), a, b, edge)
                        v, nb = track_vector(z["blocks"], idx), len(idx)
                    vecs[edge].append(v)
                    row[f"bloques_{edge}"] = nb
                    row[f"borde_reducido_{edge}"] = int(reduced)
                rows.append(row)
                start += dur
            pos_global += len(tracks)
    mats = {e: np.stack([v if v is not None else np.full(512, np.nan) for v in vs]) for e, vs in vecs.items()}
    return rows, mats


def sets_of(rows: Sequence[Dict]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for r in rows:
        out.setdefault(r["set_id"], []).append(r["fila"])
    for k in out:
        out[k].sort(key=lambda f: rows[f]["posicion"])
    return out


def summary(x: Sequence[float]) -> Dict[str, float]:
    x = list(x)
    if not x:
        return {"n": 0}
    return {"n": len(x), "media": round(float(np.mean(x)), 4), "mediana": round(float(np.median(x)), 4),
            "p25": round(float(np.percentile(x, 25)), 4), "p75": round(float(np.percentile(x, 75)), 4)}


def trajectory(rows: Sequence[Dict], V: np.ndarray, lib: np.ndarray, rng: np.random.Generator) -> Dict:
    ok = ~np.isnan(V[:, 0])
    sets = sets_of(rows)
    step, skip, same_disc, step_by_q, first_by_q = [], [], [], {}, {}
    for sid, f in sets.items():
        good = [x for x in f if ok[x]]
        for x, y in zip(f, f[1:]):
            if ok[x] and ok[y]:
                d = 1 - float(V[x] @ V[y])
                step.append(d)
                q = min(4, int(5 * (rows[y]["posicion"] - 1) / max(1, rows[y]["pistas_disco"])))
                step_by_q.setdefault(q, []).append(d)
        for x, y in zip(f, f[2:]):
            if ok[x] and ok[y]:
                skip.append(1 - float(V[x] @ V[y]))
        for _ in range(3 * len(good)):
            if len(good) > 2:
                a, b = rng.choice(good, 2, replace=False)
                if abs(rows[a]["posicion"] - rows[b]["posicion"]) > 2:
                    same_disc.append(1 - float(V[a] @ V[b]))
        if good and ok[f[0]]:
            for y in good[1:]:
                q = min(4, int(5 * (rows[y]["posicion"] - 1) / max(1, rows[y]["pistas_disco"])))
                first_by_q.setdefault(q, []).append(1 - float(V[f[0]] @ V[y]))
    idx = np.where(ok)[0]
    grp = np.asarray([rows[i]["grupo"] for i in idx])
    dj = np.asarray([rows[i]["dj"] for i in idx])
    same_comp, other_comp_same_dj, other_dj = [], [], []
    for _ in range(20000):
        a, b = rng.choice(len(idx), 2, replace=False)
        d = 1 - float(V[idx[a]] @ V[idx[b]])
        if grp[a] == grp[b]:
            if rows[idx[a]]["disco"] != rows[idx[b]]["disco"]:
                same_comp.append(d)
        elif dj[a] == dj[b]:
            other_comp_same_dj.append(d)
        else:
            other_dj.append(d)
    L = unit(lib)
    rnd_lib = [1 - float(L[a] @ L[b]) for a, b in rng.integers(0, len(L), size=(5000, 2)) if a != b]
    return {"temas_con_vector": int(ok.sum()), "vecinos": summary(step), "a_dos_puestos": summary(skip),
            "azar_mismo_disco_no_vecinos": summary(same_disc), "azar_misma_compilacion_otro_disco": summary(same_comp),
            "azar_otra_compilacion_mismo_dj": summary(other_comp_same_dj), "azar_otro_dj": summary(other_dj),
            "azar_biblioteca": summary(rnd_lib),
            "vecinos_por_quinto_del_disco": {str(q): summary(v) for q, v in sorted(step_by_q.items())},
            "distancia_al_primero_por_quinto": {str(q): summary(v) for q, v in sorted(first_by_q.items())}}


def year(text: str) -> Optional[int]:
    try:
        return int(str(text)[:4])
    except ValueError:
        return None


def between_sets(rows: Sequence[Dict], V: np.ndarray) -> Dict:
    """Distancia entre los centroides de dos discos: misma compilación; otra del mismo DJ u otro DJ,
    según la diferencia de años. Separa la señal del DJ de la de la época (o la cadena de mezcla)."""
    ok = ~np.isnan(V[:, 0])
    cents, meta = [], []
    for sid, f in sets_of(rows).items():
        good = [x for x in f if ok[x]]
        if len(good) >= 3:
            cents.append(unit(V[good].mean(axis=0)))
            r = rows[f[0]]
            meta.append((r["grupo"], r["dj"], year(r["fecha"])))
    C = np.stack(cents)
    S = 1 - C @ C.T
    buckets: Dict[str, List[float]] = {}
    for i in range(len(C)):
        for j in range(i + 1, len(C)):
            (g1, d1, y1), (g2, d2, y2) = meta[i], meta[j]
            if g1 == g2:
                key = "misma_compilacion"
            else:
                gap = "?" if y1 is None or y2 is None else ("0-1" if abs(y1 - y2) <= 1 else "2-5" if abs(y1 - y2) <= 5 else "6+")
                key = f"{'mismo_dj' if d1 == d2 else 'otro_dj'}_anios_{gap}"
            buckets.setdefault(key, []).append(float(S[i, j]))
    return {"discos": len(C), **{k: summary(v) for k, v in sorted(buckets.items())}}


def bridge(rows: Sequence[Dict], V: np.ndarray, items: Sequence[rc.Item]) -> Dict:
    uids = [it.uid for it in items]
    L = unit(np.asarray([it.vec for it in items], dtype=np.float64))
    pos = {u: i for i, u in enumerate(uids)}
    ranks = []
    for r in rows:
        f = r["fila"]
        if r["track_uid"] and r["track_uid"] in pos and not np.isnan(V[f, 0]):
            s = L @ V[f]
            ranks.append(int((s > s[pos[r["track_uid"]]]).sum()) + 1)
    return {"temas_en_comun": len(ranks), "candidatos": len(uids),
            "top1": round(float(np.mean(np.asarray(ranks) <= 1)), 3) if ranks else None,
            "top10": round(float(np.mean(np.asarray(ranks) <= 10)), 3) if ranks else None,
            "mediana_del_puesto": float(np.median(ranks)) if ranks else None}


def query(V: np.ndarray, played: Sequence[int], profile: str) -> np.ndarray:
    a = V[played[-1]]
    if profile == "historia":
        w = [1.0, 0.5, 0.25][: len(played)]
        return unit(sum(wi * V[f] for wi, f in zip(w, reversed(played))))
    if profile == "momento" and len(played) > 1:
        return unit(a + (a - V[played[-2]]))
    return a


def e1(rows: Sequence[Dict], V: np.ndarray, pool: str, profile: str, seed: int = 0) -> Dict:
    """Banco E1 en D7. pool «disco»: lo que queda del disco; «d7»: todo D7 menos lo tocado."""
    rng = np.random.default_rng(seed)
    ok = ~np.isnan(V[:, 0])
    ids = [identity(r) for r in rows]
    grp = np.asarray([r["grupo"] for r in rows])
    allowed = np.where(ok)[0]
    ranks, pcts, same_comp_share, base_share = [], [], [], []
    repeated = 0
    per_set: Dict[str, List[int]] = {}
    per_set_1: Dict[str, List[int]] = {}
    for sid, f in sets_of(rows).items():
        for i in range(1, len(f)):
            a, b = f[i - 1], f[i]
            if not (ok[a] and ok[b]):
                continue
            played = [x for x in f[:i] if ok[x]]
            seen = {ids[x] for x in f[:i]}
            if ids[b] in seen:                   # el tema ya sonó en el disco (otra versión con el mismo nombre)
                repeated += 1
                continue
            if pool == "disco":
                cand = [x for x in f[i:] if ok[x] and ids[x] not in seen]
            else:
                cand = [x for x in allowed if ids[x] not in seen]
            if len(cand) < 2:
                continue
            q = query(V, played, profile)
            s = rng.random(len(cand)) if profile == "azar" else V[cand] @ q
            best: Dict[str, Tuple[float, int]] = {}
            for x, sc in zip(cand, s):
                if ids[x] not in best or sc > best[ids[x]][0]:
                    best[ids[x]] = (float(sc), x)
            target = best[ids[b]][0]
            k = 1 + sum(1 for sc, _ in best.values() if sc > target)
            ranks.append(k)
            pcts.append((k - 1) / max(1, len(best) - 1))
            per_set.setdefault(sid, []).append(int(k <= 30))
            per_set_1.setdefault(sid, []).append(int(k <= 1))
            if pool == "d7":
                order = sorted(best.values(), reverse=True)[:30]
                same_comp_share.append(np.mean([grp[x] == grp[b] for _, x in order]))
                base_share.append(np.mean([grp[x] == grp[b] for _, x in best.values()]))
    r = np.asarray(ranks)
    out = {"transiciones": len(r), "sets": len(per_set), "repetidas_saltadas": repeated,
           "acierto_1": round(float(np.mean(r <= 1)), 3), "acierto_10": round(float(np.mean(r <= 10)), 3),
           "acierto_30": round(float(np.mean(r <= 30)), 3), "rr_medio": round(float(np.mean(1.0 / r)), 4),
           "percentil_mediano": round(st.median(pcts), 3),
           "intervalo_acierto_30_por_sets": bootstrap_sets(per_set, seed=seed),
           "intervalo_acierto_1_por_sets": bootstrap_sets(per_set_1, seed=seed)}
    if same_comp_share:
        out["de_la_misma_compilacion_en_los_30"] = round(float(np.mean(same_comp_share)), 3)
        out["de_la_misma_compilacion_entre_los_candidatos"] = round(float(np.mean(base_share)), 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Trayectorias y E1 sobre las compilaciones D7")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--indice", default=str(rc.DEFAULT_INDEX))
    ap.add_argument("--semillas", type=int, default=5, help="Semillas para el perfil azar")
    args = ap.parse_args()
    root = Path(args.root)
    _, items = rc.load_index(Path(args.indice))
    items = [it for it in items if it.vec]
    rows, mats = build_store(root, library_matcher(items))
    dest = root / "sets"
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "d7_temas.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(dest / "d7_clap.npz", **{f"borde_{e}": m.astype(np.float32) for e, m in mats.items()})
    lib = np.asarray([it.vec for it in items], dtype=np.float64)
    res = {"temas": len(rows), "sets": len(sets_of(rows)), "compilaciones": len({r["grupo"] for r in rows}),
           "temas_con_track_uid": sum(1 for r in rows if r["track_uid"]), "por_borde": {}}
    for edge, V in mats.items():
        V = V.astype(np.float64)
        rng = np.random.default_rng(0)
        block = {"trayectoria": trajectory(rows, V, lib, rng), "entre_discos": between_sets(rows, V),
                 "puente_biblioteca": bridge(rows, V, items),
                 "borde_reducido": sum(r[f"borde_reducido_{edge}"] for r in rows), "e1": {}}
        for pool in ("disco", "d7"):
            for prof in PROFILES:
                if prof == "azar":
                    runs = [e1(rows, V, pool, prof, seed=s) for s in range(args.semillas)]
                    keys = [k for k, v in runs[0].items() if isinstance(v, float)]
                    block["e1"][f"{pool}/{prof}"] = {**runs[0], **{k: round(float(np.mean([x[k] for x in runs])), 4) for k in keys}}
                else:
                    block["e1"][f"{pool}/{prof}"] = e1(rows, V, pool, prof)
        res["por_borde"][str(edge)] = block
        t = block["trayectoria"]
        print(f"borde {edge} s: vecinos {t['vecinos'].get('mediana')} | a dos {t['a_dos_puestos'].get('mediana')} | "
              f"azar disco {t['azar_mismo_disco_no_vecinos'].get('mediana')} | otro dj {t['azar_otro_dj'].get('mediana')}",
              flush=True)
        for k, v in block["e1"].items():
            print(f"  {k}: acierto_1 {v['acierto_1']} acierto_30 {v['acierto_30']} rr {v['rr_medio']} "
                  f"pct {v['percentil_mediano']}", flush=True)
    (dest / "d7_trayectorias.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {res['temas']} temas en {res['sets']} sets ({res['compilaciones']} compilaciones), "
          f"{res['temas_con_track_uid']} en la biblioteca -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
