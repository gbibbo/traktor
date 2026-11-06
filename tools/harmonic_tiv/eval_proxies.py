"""
PURPOSE: Corre las métricas proxy (src/v4/harmonic_tiv/evaluate.py) sobre una grilla de variantes que
         cambian de a una variable respecto de una configuración base, y las compara con la regla
         Camelot de src/v4/common/harmonic.py.
         Conjuntos:
           thesis48 : tonalidad verificada; pares dentro de cada género (como la tesis).
           musica   : tonalidad de consenso (tag MIK == Beatport) en el split pedido; transiciones reales
                      de los History en las sesiones del split; hubness sobre todo el split.
         Uso:
           python tools/harmonic_tiv/eval_proxies.py --set thesis48 --eval-dir <dir> --cache <dir>
           python tools/harmonic_tiv/eval_proxies.py --set musica --split dev --eval-dir <dir> --cache <dir>
               --bpm <bpm_key.parquet>
CHANGELOG:
  - 2026-11-06: Creación (rama harmonic-tiv).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.v4.common.harmonic import best_key_shift  # noqa: E402
from src.v4.harmonic_tiv import evaluate as ev  # noqa: E402
from src.v4.harmonic_tiv import features as ft  # noqa: E402
from src.v4.harmonic_tiv import tiv as tv  # noqa: E402

BASE = ft.THESIS_FEM10
ONE_AT_A_TIME = [
    ("base = FEM#10 (tesis)", {}),
    ("section=center30 (código publicado)", {"section": "center30"}),
    ("section=center50", {"section": "center50"}),
    ("section=core", {"section": "core"}),
    ("preproc=orig (sin HPSS)", {"preproc": "orig"}),
    ("spec=raw_lin (sin blanqueo)", {"spec": "raw_lin"}),
    ("spec=white_nnls (NNLS de verdad)", {"spec": "white_nnls"}),
    ("profile=full (con bajo)", {"profile": "full"}),
    ("profile=mid (sin bajo ni agudos)", {"profile": "mid"}),
    ("profile=bass (solo bajo)", {"profile": "bass"}),
    ("weighting=norm", {"weighting": "norm"}),
    ("weighting=salience", {"weighting": "salience"}),
    ("metric=distance", {"metric": "distance"}),
    ("metric=mix_dissonance", {"metric": "mix_dissonance"}),
    ("metric=coherence_loss", {"metric": "coherence_loss"}),
]


def split_change(change: dict):
    """Separa la métrica (no es parte de TivConfig) del resto del cambio."""
    change = dict(change)
    metric = change.pop("metric", "tivlib")
    return BASE.with_(**change), metric


def tiv_matrix(cache_dir: Path, ids, cfg: ft.TivConfig):
    tivs = ft.collection_tivs(cache_dir, ids, cfg)
    keep = [i for i in ids if i in tivs]
    return keep, np.stack([tivs[i] for i in keep])


def scores_from_tivs(T: np.ndarray, metric: str = "tivlib"):
    fn = tv.METRICS[metric]
    hc0 = tv.hc_matrix(T, T, shifts=[0], metric=fn)[:, :, 0]
    full = tv.hc_matrix(T, T, shifts=range(-5, 7), metric=fn)           # -5..6
    best = np.argmin(full, axis=2) - 5
    return 1.0 - hc0, best, hc0


def key_rule_matrix(keys, max_shift=0):
    n = len(keys)
    m = np.full((n, n), 0.5)
    for i in range(n):
        for j in range(n):
            m[i, j] = best_key_shift(keys[i], keys[j], max_shift=max_shift)[0]
    return m


def estimated_keys(cache_dir: Path, ids, cfg: ft.TivConfig, profile="temperley"):
    out = []
    for i in ids:
        ch = ft.aggregate_chroma(ft.load_cache(Path(cache_dir) / f"{i}.npz"), cfg)
        out.append(ft.key_from_chroma(ch, profile))
    return out


def run_thesis48(args) -> dict:
    man = pd.read_csv(args.eval_dir / "thesis48_manifest.csv")
    results = []
    for label, change in ONE_AT_A_TIME:
        cfg, metric = split_change(change)
        ids, T = tiv_matrix(args.cache, list(man["id"]), cfg)
        sub = man.set_index("id").loc[ids]
        keys, genres = list(sub["camelot"]), list(sub["genre"])
        score, best, hc0 = scores_from_tivs(T, metric)
        row = {"variant": label, "config": cfg.name() + "/" + metric, "n": len(ids)}
        for g in ("progressive_house", "techno"):
            idx = [k for k, gg in enumerate(genres) if gg == g]
            kk = [keys[k] for k in idx]
            pm = ev.key_pair_metrics(score[np.ix_(idx, idx)], kk)
            perf = ev.perf_thesis(100 * (1 - hc0[np.ix_(idx, idx)]), kk)
            sa = ev.shift_accuracy(best[np.ix_(idx, idx)], kk)
            est = estimated_keys(args.cache, [ids[k] for k in idx], cfg)
            row[g] = {"Perf": round(perf["Perf"], 2), "spearman": round(pm["spearman_fifths"], 3),
                      "auc_near_far": round(pm["auc_near_far"], 3), "mrr_same": round(pm["mrr_same"], 3),
                      "shift_exact": round(sa["exact"], 3), "shift_within_fifth": round(sa["within_fifth"], 3),
                      "key_acc_temperley": round(float(np.mean([a == b for a, b in zip(est, kk)])), 3),
                      "tonic_or_fifth": round(float(np.mean([ev.fifths_distance(a, b) <= 1 for a, b in zip(est, kk)])), 3)}
            if label.startswith("base"):
                # referencia: regla Camelot con la tonalidad estimada del mismo croma (sin transponer)
                kr = key_rule_matrix(est)
                pm_k = ev.key_pair_metrics(kr, kk)
                row[g]["camelot_on_estimated_keys"] = {"spearman": round(pm_k["spearman_fifths"], 3),
                                                       "auc_near_far": round(pm_k["auc_near_far"], 3),
                                                       "mrr_same": round(pm_k["mrr_same"], 3)}
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return {"set": "thesis48", "results": results}


def run_musica(args) -> dict:
    keys = pd.read_csv(args.eval_dir / "keys_musica.csv")
    keys = keys[keys["split"] == args.split]
    tag_bp = keys[(keys.k_tag != "?") & (keys.k_tag == keys.k_beatport)]
    bpm = pd.read_parquet(args.bpm)[["track_uid", "bpm"]].set_index("track_uid")["bpm"]
    trans = pd.read_csv(args.eval_dir / "transitions_history.csv").dropna(subset=["a_uid", "b_uid"])
    trans = trans[trans["split"] == args.split]
    results = []
    all_ids = list(keys["track_uid"])
    tag_of = keys.set_index("track_uid")["k_tag"]
    for label, change in ONE_AT_A_TIME:
        t0 = time.time()
        cfg, metric = split_change(change)
        tivs = ft.collection_tivs(args.cache, all_ids, cfg)
        ids = [i for i in all_ids if i in tivs]
        T = np.stack([tivs[i] for i in ids])
        pos = {i: k for k, i in enumerate(ids)}
        score, best, _ = scores_from_tivs(T, metric)
        cons = [i for i in tag_bp["track_uid"] if i in pos]
        ci = [pos[i] for i in cons]
        ck = list(tag_bp.set_index("track_uid").loc[cons, "k_tag"])
        pm = ev.key_pair_metrics(score[np.ix_(ci, ci)], ck)
        sa = ev.shift_accuracy(best[np.ix_(ci, ci)], ck)
        hub = ev.hubness(score, k=10)
        # transiciones reales: a y b en este split con TIV; pool = temas del split con BPM compatible
        all_trans = [(a, b) for a, b in zip(trans.a_uid, trans.b_uid) if a in pos and b in pos and a != b]
        bpm_of = {i: float(bpm.get(i, np.nan)) for i in ids}
        pools = {}

        def pool_fn(a):
            if a not in pools:
                pools[a] = [x for x in ids if ev.bpm_compatible(bpm_of[a], bpm_of[x])]
            return pools[a]
        tp = ev.transition_percentile(all_trans, lambda a, b: score[pos[a], pos[b]], pool_fn)
        row = {"variant": label, "config": cfg.name() + "/" + metric, "n_tracks": len(ids), "consensus_n": len(ci),
               "spearman": round(pm["spearman_fifths"], 3), "auc_near_far": round(pm["auc_near_far"], 3),
               "auc_rule": round(pm["auc_rule"], 3), "mrr_same": round(pm["mrr_same"], 3),
               "shift_exact": round(sa["exact"], 3), "hub_skew": round(hub["skew"], 2),
               "hub_never": round(hub["never"], 3), "hub_max": hub["max_occ"], "history": tp,
               "seconds": round(time.time() - t0, 1)}
        if label.startswith("base"):
            kr = key_rule_matrix([tag_of[i] for i in ids])
            row["camelot_tag_history"] = ev.transition_percentile(all_trans, lambda a, b: kr[pos[a], pos[b]], pool_fn)
            row["camelot_tag_hubness"] = ev.hubness(kr + 1e-6 * np.random.default_rng(0).random(kr.shape), k=10)
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return {"set": "musica", "split": args.split, "results": results}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["thesis48", "musica"], required=True)
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--bpm", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    res = run_thesis48(args) if args.set == "thesis48" else run_musica(args)
    out = args.out or args.eval_dir / f"proxies_{args.set}{'_' + args.split if args.set == 'musica' else ''}.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
