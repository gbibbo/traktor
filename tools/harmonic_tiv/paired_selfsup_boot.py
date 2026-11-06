"""
PURPOSE: Comparación pareada entre variantes del TIV sobre un conjunto congelado de temas dev: S1-hard
         (mismas consultas y mismos 50 negativos de la misma tonalidad MIK para todas las variantes;
         bootstrap por tema) y Spearman contra la teoría de tonalidades en los temas de consenso
         (bootstrap remuestreando temas en los dos roles, con multiplicidades).
         Uso: python tools/harmonic_tiv/paired_selfsup_boot.py <repo> <eval_dir> <cache_dir> <ids_congelados.txt>
CHANGELOG:
  - 2026-11-06: Creación (rama harmonic-tiv, informe de auditoría).
  - 2026-11-06: El bootstrap de teoría conserva las multiplicidades del remuestreo (antes dict.fromkeys las
                descartaba: era una submuestra de ~63 % de temas distintos, no un bootstrap). Señalado por el
                auditor. B = 1000.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1])
from src.v4.harmonic_tiv import evaluate as ev  # noqa: E402
from src.v4.harmonic_tiv import features as ft  # noqa: E402
from src.v4.harmonic_tiv import tiv as tv  # noqa: E402

eval_dir, cache = Path(sys.argv[2]), Path(sys.argv[3])
frozen = set(Path(sys.argv[4]).read_text(encoding="utf-8").split())
keys = pd.read_csv(eval_dir / "keys_musica.csv")
dev = [u for u in keys[keys.split == "dev"].track_uid if u in frozen]
tag = keys.set_index("track_uid")["k_tag"].to_dict()
cons = set(keys[(keys.k_tag != "?") & (keys.k_tag == keys.k_beatport)].track_uid)
VARIANTS = {"FEM#10": {}, "sin blanqueo": {"spec": "raw_lin"}, "solo bajo": {"profile": "bass"},
            "con bajo": {"profile": "full"}, "30% central": {"section": "center30"}}
rng = np.random.default_rng(0)
feats = {name: {u: ft.fragment_tivs(ft.load_cache(cache / f"{u}.npz"), ft.THESIS_FEM10.with_(**ch)) for u in dev}
         for name, ch in VARIANTS.items()}
whole = {name: {u: ft.track_tiv(ft.load_cache(cache / f"{u}.npz"), ft.THESIS_FEM10.with_(**ch)) for u in dev}
         for name, ch in VARIANTS.items()}
# fixed queries and same-key negatives, defined from FEM#10 finiteness (fragments' loudness mask is variant-independent)
base = feats["FEM#10"]
pairs, pool = [], []
for u, f in base.items():
    ok = [i for i in range(len(f) - 1) if np.all(np.isfinite(f[i])) and np.all(np.isfinite(f[i + 1]))]
    for i in rng.permutation(ok)[:4]:
        pairs.append((u, int(i)))
    pool += [(u, i) for i in range(len(f)) if np.all(np.isfinite(f[i]))]
owner = np.array([u for u, _ in pool])
okey = np.array([tag.get(u, "?") for u in owner])
queries = []
for u, i in pairs:
    k = tag.get(u, "?")
    cand = np.flatnonzero((owner != u) & (okey == k)) if k != "?" else np.array([], int)
    if len(cand) >= 50:
        queries.append((u, i, rng.choice(cand, 50, replace=False)))
print("dev tracks", len(dev), "queries with 50 same-key negatives", len(queries))


def rr_per_track(name):
    f = feats[name]
    out = {}
    for u, i, neg in queries:
        q, p = f[u][i], f[u][i + 1]
        negs = np.array([f[owner[j]][pool[j][1]] for j in neg])
        sp = 1 - tv.hc_tivlib(q, p)
        sn = 1 - tv.hc_tivlib(q[None, :], negs)
        rank = 1 + (sn > sp).sum() + 0.5 * (sn == sp).sum()
        out.setdefault(u, []).append(1 / rank)
    return {u: np.mean(v) for u, v in out.items()}


def theory_matrices(name, ids):
    """Score sin transponer y distancia de quintas entre los temas de consenso (en el orden de ids)."""
    T = np.stack([whole[name][u] for u in ids])
    S = 1 - tv.hc_matrix(T, T, shifts=[0])[:, :, 0]
    D = np.array([[ev.fifths_distance(tag[a], tag[b]) for b in ids] for a in ids], float)
    return S, D


def spearman_resampled(S, D, idx):
    """Spearman de score contra -distancia sobre los pares (p, q) del remuestreo con multiplicidades:
    un tema repetido cuenta tantas veces como salió; se excluyen solo los pares de un tema consigo mismo
    (mismo tema original), no los pares entre sus copias y otros temas. Corrige la versión anterior, que
    con dict.fromkeys descartaba las repeticiones (revisión del auditor, 2026-11-06)."""
    idx = np.asarray(idx)
    mask = idx[:, None] != idx[None, :]
    s = S[np.ix_(idx, idx)][mask]
    d = D[np.ix_(idx, idx)][mask]
    return ev.spearman(s, -d)


rr = {n: rr_per_track(n) for n in VARIANTS}
tracks = sorted(rr["FEM#10"])
cons_dev = [u for u in dev if u in cons]
print("consensus dev tracks (for theory):", len(cons_dev))
mats = {n: theory_matrices(n, cons_dev) for n in VARIANTS}
B = 1000
boots = [rng.choice(len(tracks), len(tracks)) for _ in range(B)]
boots_c = [rng.choice(len(cons_dev), len(cons_dev)) for _ in range(B)]
full = np.arange(len(cons_dev))
for n in VARIANTS:
    if n == "FEM#10":
        print(f"FEM#10: S1-hard MRR {np.mean([rr[n][u] for u in tracks]):.3f}  theory Spearman "
              f"{spearman_resampled(*mats[n], full):.3f}")
        continue
    d = np.array([rr[n][u] - rr["FEM#10"][u] for u in tracks])
    db = [d[b].mean() for b in boots]
    point = spearman_resampled(*mats[n], full) - spearman_resampled(*mats["FEM#10"], full)
    dt = np.array([spearman_resampled(*mats[n], b) - spearman_resampled(*mats["FEM#10"], b) for b in boots_c])
    print(f"{n:12s} dS1-hard {d.mean():+.3f} [{np.percentile(db, 2.5):+.3f},{np.percentile(db, 97.5):+.3f}]"
          f"   dTheory {point:+.3f} [{np.percentile(dt, 2.5):+.3f},{np.percentile(dt, 97.5):+.3f}]")
