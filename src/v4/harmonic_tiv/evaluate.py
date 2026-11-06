"""
PURPOSE: Métricas proxy para ajustar y descartar variantes de compatibilidad armónica (protocolo en
         docs/plans/armonia_tiv_protocolo.md). Ninguna decide la integración: decide la escucha.
           - perf_thesis: la métrica Perf de la tesis (3.10), para reproducir sus números.
           - key_pair_metrics: con tonalidades confiables, ¿el score ordena los pares como la teoría
             (distancia en el círculo de quintas)? Spearman, AUC compatible/incompatible y MRR del
             compañero de misma tonalidad.
           - shift_accuracy: ¿la transposición sugerida lleva el candidato a la tonalidad del objetivo?
           - hubness: cuánto se repiten los mismos temas en los top-k de todos (Radovanović 2010).
           - transition_percentile: percentil del tema que Gabriel tocó de verdad entre los candidatos
             con BPM compatible (0,5 = el score no distingue lo que tocó de lo que no).
         Convención: `score` más alto = más compatible (para TIV.lib usar 1 - hc).
CHANGELOG:
  - 2026-11-06: Creación (rama harmonic-tiv).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from src.v4.common.harmonic import best_key_shift
from src.v4.harmonic_tiv import tiv as tv


def camelot_parts(k: str):
    return int(k[:-1]), k[-1]


def fifths_distance(ka: str, kb: str) -> Optional[int]:
    """Diferencia de armadura (0..6) entre dos tonalidades Camelot, como k_tc de la tesis: número de
    pasos en la rueda, contando la relativa (mismo número) como 0."""
    if "?" in (ka, kb):
        return None
    na, _ = camelot_parts(ka)
    nb, _ = camelot_parts(kb)
    d = (nb - na) % 12
    return min(d, 12 - d)


def key_shift_target(ka: str, kb: str) -> Optional[int]:
    """Semitonos (-5..6) que llevan kb al mismo número Camelot que ka (misma armadura). +1 semitono =
    +7 en la rueda, así que s = 7·(na - nb) mod 12."""
    if "?" in (ka, kb):
        return None
    na, _ = camelot_parts(ka)
    nb, _ = camelot_parts(kb)
    s = (7 * (na - nb)) % 12
    return s - 12 if s > 6 else s


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score de un par positivo > uno negativo), empates a medias (Mann-Whitney)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # rangos promedio para empates
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    r = spearmanr(x, y).correlation
    return float(r)


def key_pair_metrics(score: np.ndarray, keys: Sequence[str], groups: Optional[Sequence[str]] = None) -> Dict:
    """`score` (N, N) sin transponer, `keys` confiables. Pares i != j (y del mismo grupo si se da).
    - spearman de score contra -distancia de quintas
    - auc_near_far: distancia 0-1 (positivo) contra 3-6 (negativo)
    - auc_rule: regla de Gabriel sin transponer >= 0.6 contra < 0.6
    - mrr_same: rango recíproco medio del primer candidato con la misma tonalidad exacta"""
    n = len(keys)
    groups = list(groups) if groups is not None else ["all"] * n
    s_list, d_list, rule_list = [], [], []
    rr = []
    for i in range(n):
        cand = [j for j in range(n) if j != i and groups[j] == groups[i] and keys[j] != "?"]
        if keys[i] == "?" or not cand:
            continue
        for j in cand:
            s_list.append(score[i, j])
            d_list.append(fifths_distance(keys[i], keys[j]))
            rule_list.append(best_key_shift(keys[i], keys[j], max_shift=0)[0])
        same = [j for j in cand if keys[j] == keys[i]]
        if same:
            order = sorted(cand, key=lambda j: -score[i, j])
            rr.append(1.0 / (1 + min(order.index(j) for j in same)))
    s, d, rule = np.array(s_list), np.array(d_list, float), np.array(rule_list)
    return {"n_pairs": int(len(s)),
            "spearman_fifths": spearman(s, -d),
            "auc_near_far": auc(s[d <= 1], s[d >= 3]),
            "auc_rule": auc(s[rule >= 0.6], s[rule < 0.6]),
            "mrr_same": float(np.mean(rr)) if rr else float("nan"),
            "n_targets_with_same": len(rr)}


def perf_thesis(hc_percent: np.ndarray, keys: Sequence[str]) -> Dict:
    """Perf = (100 - meanHC) / meanSTD (tesis, ecuaciones 3.3 a 3.10). hc_percent (N, N) en 0..100."""
    n = len(keys)
    mean_by_p = {p: [] for p in range(7)}
    std_by_p = {p: [] for p in range(7)}
    for t in range(n):
        for p in range(7):
            vals = [hc_percent[t, c] for c in range(n) if c != t and fifths_distance(keys[t], keys[c]) == p]
            if vals:
                mean_by_p[p].append(np.mean(vals))
                std_by_p[p].append(np.std(vals))
    mean_hc = np.mean([np.mean(v) for v in mean_by_p.values() if v])
    mean_std = np.mean([np.mean(v) for v in std_by_p.values() if v])
    return {"meanHC": float(mean_hc), "meanSTD": float(mean_std), "Perf": float((100 - mean_hc) / mean_std),
            "curve": {p: float(np.mean(v)) for p, v in mean_by_p.items() if v}}


def shift_accuracy(best_shifts: np.ndarray, keys: Sequence[str], groups: Optional[Sequence[str]] = None) -> Dict:
    """best_shifts (N, N): semitonos sugeridos para el candidato j frente al objetivo i. Acierto si lleva
    a la misma armadura (±6 cuentan como iguales); 'within1' tolera una quinta (±1 en la rueda)."""
    n = len(keys)
    groups = list(groups) if groups is not None else ["all"] * n
    hit, near = [], []
    for i in range(n):
        for j in range(n):
            if i == j or groups[i] != groups[j] or "?" in (keys[i], keys[j]):
                continue
            target = key_shift_target(keys[i], keys[j])
            s = int(best_shifts[i, j])
            ok = (s - target) % 12 == 0
            hit.append(ok)
            # una quinta = ±7 semitonos ≡ ±5
            near.append(ok or (s - target) % 12 in (5, 7))
    return {"n": len(hit), "exact": float(np.mean(hit)), "within_fifth": float(np.mean(near))}


def hubness(score: np.ndarray, k: int = 10, mask: Optional[np.ndarray] = None) -> Dict:
    """k-occurrence: cuántas veces cada tema aparece en el top-k de los demás. Asimetría alta y una
    fracción grande de temas que nunca aparecen = unos pocos 'donantes universales' copan las listas."""
    from scipy.stats import skew
    s = np.array(score, dtype=float)
    np.fill_diagonal(s, -np.inf)
    if mask is not None:
        s = np.where(mask, s, -np.inf)
    top = np.argsort(-s, axis=1)[:, :k]
    occ = np.bincount(top.ravel(), minlength=len(s))
    return {"k": k, "skew": float(skew(occ)), "max_occ": int(occ.max()), "never": float(np.mean(occ == 0)),
            "top1pct_share": float(np.sort(occ)[::-1][: max(1, len(occ) // 100)].sum() / max(occ.sum(), 1))}


def bpm_compatible(a: float, b: float, tol: float = 0.06) -> bool:
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b <= 0:
        return False
    r = b / a
    return any(abs(r * m - 1) <= tol for m in (1.0, 2.0, 0.5))


def transition_percentile(pairs: List[tuple], score_fn: Callable[[str, str], float], pool_fn: Callable[[str], List[str]],
                          rng: Optional[np.random.Generator] = None, n_boot: int = 2000) -> Dict:
    """Para cada transición real (a, b): percentil de score(a, b) entre score(a, x) de los candidatos x
    de pool_fn(a) (empates a medias). Devuelve la media y su IC bootstrap 95 %."""
    rng = rng or np.random.default_rng(0)
    pct = []
    for a, b in pairs:
        pool = [x for x in pool_fn(a) if x not in (a, b)]
        if len(pool) < 20:
            continue
        sb = score_fn(a, b)
        sp = np.array([score_fn(a, x) for x in pool])
        pct.append(float((sp < sb).mean() + 0.5 * (sp == sb).mean()))
    return _summ(pct, rng, n_boot)


def _summ(pct, rng, n_boot):
    pct = np.array(pct)
    if not len(pct):
        return {"n": 0}
    boots = [rng.choice(pct, len(pct)).mean() for _ in range(n_boot)]
    return {"n": int(len(pct)), "mean_pct": float(pct.mean()), "ci95": [float(np.percentile(boots, 2.5)),
                                                                         float(np.percentile(boots, 97.5))]}


def _rank_metrics(pos_scores: np.ndarray, neg_scores: np.ndarray) -> Dict:
    """pos_scores (Q,), neg_scores (Q, N): rango del positivo (empates a medias), MRR, top-1 y AUC."""
    better = (neg_scores > pos_scores[:, None]).sum(axis=1)
    ties = (neg_scores == pos_scores[:, None]).sum(axis=1)
    rank = 1 + better + 0.5 * ties
    n = neg_scores.shape[1]
    return {"n_queries": int(len(pos_scores)), "mrr": float(np.mean(1.0 / rank)),
            "top1": float(np.mean(rank <= 1.0)), "auc": float(np.mean(1 - (better + 0.5 * ties) / n)),
            "chance_mrr": float(np.mean(1.0 / np.arange(1, n + 2)))}


def selfsup_continuation(frags: Dict[str, np.ndarray], metric: Callable, n_neg: int = 200, pairs_per_track: int = 4,
                         seed: int = 0, key_of: Optional[Dict[str, str]] = None) -> Dict:
    """Objetivo autosupervisado (propuesta de Gabriel, 2026-11-06): para cada tema, pares de fragmentos
    consecutivos (f_i, f_i+1); el positivo f_i+1 compite con `n_neg` fragmentos de otros temas. Score =
    1 - metric(f_i, c) sin transponer. Además: % de pares cuya mejor transposición es 0 (en ±6 y en ±2).
    Con `key_of` (tonalidad por tema) calcula lo mismo para la regla Camelot como referencia."""
    rng = np.random.default_rng(seed)
    pairs, pool, pool_owner = [], [], []
    for uid, f in frags.items():
        ok = [i for i in range(len(f) - 1) if np.all(np.isfinite(f[i])) and np.all(np.isfinite(f[i + 1]))]
        for i in rng.permutation(ok)[:pairs_per_track]:
            pairs.append((uid, int(i)))
        for i in range(len(f)):
            if np.all(np.isfinite(f[i])):
                pool.append(f[i])
                pool_owner.append(uid)
    pool = np.array(pool)
    pool_owner = np.array(pool_owner)
    q = np.array([frags[u][i] for u, i in pairs])
    pos = np.array([frags[u][i + 1] for u, i in pairs])
    negs_idx = np.empty((len(pairs), n_neg), dtype=int)
    for k, (u, _) in enumerate(pairs):
        cand = np.flatnonzero(pool_owner != u)
        negs_idx[k] = rng.choice(cand, size=n_neg, replace=False)
    pos_s = 1.0 - metric(q, pos)
    neg_s = 1.0 - metric(q[:, None, :], pool[negs_idx])
    out = {"tiv": _rank_metrics(pos_s, neg_s)}
    shifts = np.arange(-5, 7)
    by_shift = metric(q[:, None, :], tv.transpose(pos, shifts))          # (Q, 12)
    out["tiv"]["shift0_full"] = float(np.mean(shifts[np.argmin(by_shift, axis=1)] == 0))
    near = np.abs(shifts) <= 2
    out["tiv"]["shift0_pm2"] = float(np.mean(shifts[near][np.argmin(by_shift[:, near], axis=1)] == 0))
    if key_of is not None:
        kq = [key_of.get(u, "?") for u, _ in pairs]
        pos_k = np.array([best_key_shift(k, k, max_shift=0)[0] for k in kq])
        neg_k = np.array([[best_key_shift(kq[r], key_of.get(pool_owner[j], "?"), max_shift=0)[0]
                           for j in negs_idx[r]] for r in range(len(pairs))])
        out["camelot"] = _rank_metrics(pos_k, neg_k)
    return out


def selfsup_intro_outro(first: Dict[str, np.ndarray], last: Dict[str, np.ndarray], metric: Callable,
                        n_neg: int = 200, seed: int = 0, key_of: Optional[Dict[str, str]] = None) -> Dict:
    """Los primeros 2 min de un tema contra sus últimos 2 min, entre los últimos 2 min de `n_neg` temas más
    (sin etiquetas: un tema mezcla bien consigo mismo)."""
    rng = np.random.default_rng(seed)
    uids = [u for u in first if u in last and np.all(np.isfinite(first[u])) and np.all(np.isfinite(last[u]))]
    L = np.array([last[u] for u in uids])
    q = np.array([first[u] for u in uids])
    negs = np.array([rng.choice(np.delete(np.arange(len(uids)), i), size=min(n_neg, len(uids) - 1), replace=False)
                     for i in range(len(uids))])
    out = {"tiv": _rank_metrics(1.0 - metric(q, L), 1.0 - metric(q[:, None, :], L[negs]))}
    if key_of is not None:
        ks = [key_of.get(u, "?") for u in uids]
        pos_k = np.array([best_key_shift(k, k, max_shift=0)[0] for k in ks])
        neg_k = np.array([[best_key_shift(ks[i], ks[j], max_shift=0)[0] for j in negs[i]] for i in range(len(uids))])
        out["camelot"] = _rank_metrics(pos_k, neg_k)
    return out


def within_class_percentile(pairs: List[tuple], score_fn: Callable[[str, str], float],
                            class_fn: Callable[[str, str], object], pool_fn: Callable[[str], List[str]],
                            min_pool: int = 5, rng: Optional[np.random.Generator] = None, n_boot: int = 2000) -> Dict:
    """Como transition_percentile, pero el pool de cada transición real (a, b) son solo los candidatos x
    con class_fn(a, x) == class_fn(a, b) (p. ej. el mismo score Camelot con la tonalidad que Gabriel ve).
    Ahí la regla de referencia empata por construcción: un percentil > 0,5 dice que `score_fn` aporta
    información sobre lo que Gabriel tocó más allá de esa clase."""
    rng = rng or np.random.default_rng(0)
    pct = []
    for a, b in pairs:
        cb = class_fn(a, b)
        pool = [x for x in pool_fn(a) if x not in (a, b) and class_fn(a, x) == cb]
        if len(pool) < min_pool:
            continue
        sb = score_fn(a, b)
        sp = np.array([score_fn(a, x) for x in pool])
        pct.append(float((sp < sb).mean() + 0.5 * (sp == sb).mean()))
    return _summ(pct, rng, n_boot)
