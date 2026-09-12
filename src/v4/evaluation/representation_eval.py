"""
PURPOSE: Instrumento de evaluación de representaciones (embeddings) contra la evidencia humana fija:
         (1) accuracy de tripletas del DJ (coseno con el ancla; empates 0.5) por fuente de respuestas y
             combinada, con IC bootstrap y bootstrap PAREADO contra el baseline de BPM sobre las mismas
             tripletas; (2) coherencia de vecindario contra las carpetas de legacy v1 nombradas a mano
             (pureza kNN y precisión media de recuperación con "misma carpeta" como relevante).
         Baselines incluidos: BPM (equivalencia 0.5x/1x/2x), clave (regla armónica), clase mayoritaria.
         Cualquier .npy alineado por track_uids.json se puede evaluar; nada aquí ajusta parámetros.
CHANGELOG:
  - 2026-09-12: Creación inicial (fase 1 de docs/plans/representation_model_plan.md).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.config_loader import load_config  # noqa: E402
from src.v4.common.harmonic import key_compatibility, to_camelot  # noqa: E402
from src.v4.common.path_resolver import resolve_dataset_artifacts  # noqa: E402
from src.v4.evaluation.legacy_crosscheck import V1_CLUSTERS, parse_v1_clusters  # noqa: E402
from src.v4.evaluation.triplet_evidence import ANSWERS_DIR, bpm_similarity, load_answers  # noqa: E402

SOURCE_LABELS = {
    "tripletas_respuestas_2026-09-11.csv": "uniform",
    "tripletas_rama_dj_2026-05-28.csv": "dj_branch",
}
N_BOOT = 2000
SEED = 0
KNN_K = 5


# ---------------------------------------------------------------------------
# Representaciones y funciones de similitud
# ---------------------------------------------------------------------------

class Representation:
    """Matriz (N, D) alineada con una lista de track_uid. Similitud = coseno."""

    def __init__(self, name: str, matrix: np.ndarray, uids: List[str]):
        if matrix.shape[0] != len(uids):
            raise ValueError(f"{name}: {matrix.shape[0]} filas vs {len(uids)} uids")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name}: valores no finitos")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.name = name
        self.matrix = (matrix / norms).astype(np.float64)
        self.index = {u: i for i, u in enumerate(uids)}

    def has(self, uid: str) -> bool:
        return uid in self.index

    def sim(self, uid_a: str, uid_b: str) -> float:
        return float(self.matrix[self.index[uid_a]] @ self.matrix[self.index[uid_b]])

    def sim_matrix(self, uids: List[str]) -> np.ndarray:
        rows = self.matrix[[self.index[u] for u in uids]]
        return rows @ rows.T


def load_representation(name: str, path: Path) -> Representation:
    """Acepta un directorio con embeddings.npy + track_uids.json, o un .npy con track_uids.json al lado."""
    path = Path(path)
    if path.is_dir():
        npy = path / "embeddings.npy"
        uids_path = path / "track_uids.json"
    else:
        npy = path
        uids_path = path.parent / "track_uids.json"
    matrix = np.load(npy)
    with open(uids_path, encoding="utf-8") as fh:
        uids = json.load(fh)
    return Representation(name, matrix, uids)


class FeatureBaseline:
    """Baseline simbólico (BPM o clave) con la misma interfaz que Representation."""

    def __init__(self, name: str, features: pd.DataFrame, kind: str):
        self.name = name
        self.kind = kind
        self.feats = features.set_index("track_uid")

    def has(self, uid: str) -> bool:
        return uid in self.feats.index

    def sim(self, uid_a: str, uid_b: str) -> float:
        a, b = self.feats.loc[uid_a], self.feats.loc[uid_b]
        if self.kind == "bpm":
            return bpm_similarity(float(a["bpm"]), float(b["bpm"]))
        return key_compatibility(to_camelot(a["key"]), to_camelot(b["key"]))

    def sim_matrix(self, uids: List[str]) -> np.ndarray:
        n = len(uids)
        m = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                m[i, j] = m[j, i] = self.sim(uids[i], uids[j])
        return m


# ---------------------------------------------------------------------------
# Tripletas
# ---------------------------------------------------------------------------

def triplet_outcomes(rep, triplets: pd.DataFrame) -> pd.Series:
    """Por tripleta: 1 acierto, 0 fallo, 0.5 empate, NaN si falta algún uid en la representación."""
    out = []
    for r in triplets.itertuples():
        if not (rep.has(r.anchor_uid) and rep.has(r.b_uid) and rep.has(r.c_uid)):
            out.append(np.nan)
            continue
        s_b, s_c = rep.sim(r.anchor_uid, r.b_uid), rep.sim(r.anchor_uid, r.c_uid)
        if abs(s_b - s_c) < 1e-9:
            out.append(0.5)
        else:
            out.append(float(("B" if s_b > s_c else "C") == r.answer))
    return pd.Series(out, index=triplets.index, dtype=float)


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    means = [rng.choice(values, len(values)).mean() for _ in range(n_boot)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> Dict[str, float]:
    """Diferencia a - b sobre las mismas tripletas. p_gain = fracción de remuestreos con a > b."""
    rng = np.random.default_rng(seed)
    n = len(a)
    if n == 0:
        return {"diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_gain": float("nan"), "n": 0}
    d = a - b
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return {"diff": float(d.mean()), "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)), "p_gain": float((boots > 0).mean()), "n": int(n)}


def evaluate_triplets(reps: list, triplets: pd.DataFrame, reference: str = "bpm") -> pd.DataFrame:
    outcomes = {rep.name: triplet_outcomes(rep, triplets) for rep in reps}
    groups = {"all": np.ones(len(triplets), dtype=bool)}
    for src in sorted(triplets["source"].unique()):
        groups[src] = (triplets["source"] == src).to_numpy()
    if "selection_source" in triplets.columns:
        for sel in sorted(triplets["selection_source"].dropna().unique()):
            groups[f"dj_branch/{sel}"] = (triplets["selection_source"] == sel).to_numpy()
    rows = []
    for gname, mask in groups.items():
        sub = triplets[mask]
        counts = sub["answer"].value_counts()
        majority = counts.max() / len(sub) if len(sub) else float("nan")
        for rep in reps:
            o = outcomes[rep.name][mask]
            valid = o.dropna().to_numpy()
            row = {"group": gname, "representation": rep.name, "n": int(len(valid)),
                   "accuracy": float(valid.mean()) if len(valid) else float("nan"),
                   "ties": int((valid == 0.5).sum()), "majority_class": float(majority)}
            row["ci_low"], row["ci_high"] = bootstrap_ci(valid)
            if reference in outcomes and rep.name != reference:
                both = o.notna() & outcomes[reference][mask].notna()
                pb = paired_bootstrap(o[both].to_numpy(), outcomes[reference][mask][both].to_numpy())
                row.update({f"vs_{reference}_diff": pb["diff"], f"vs_{reference}_ci_low": pb["ci_low"],
                            f"vs_{reference}_ci_high": pb["ci_high"], f"vs_{reference}_p_gain": pb["p_gain"]})
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Coherencia contra carpetas de legacy v1
# ---------------------------------------------------------------------------

def folder_coherence(rep, labels: Dict[str, str], k: int = KNN_K) -> Dict[str, float]:
    """Pureza kNN (leave-one-out) y precisión media de recuperación con misma carpeta como relevante."""
    uids = [u for u in labels if rep.has(u)]
    if len(uids) < k + 2:
        return {"n": len(uids), "knn_purity": float("nan"), "map": float("nan")}
    lab = np.array([labels[u] for u in uids])
    s = rep.sim_matrix(uids)
    purity, aps = [], []
    for i in range(len(uids)):
        order = np.argsort(-s[i], kind="stable")
        order = order[order != i]  # leave-one-out: el propio tema no cuenta
        rel = (lab[order] == lab[i]).astype(float)
        purity.append(rel[:k].mean())
        n_rel = rel.sum()
        if n_rel == 0:
            continue
        hits = np.cumsum(rel)
        ranks = np.arange(1, len(rel) + 1)
        aps.append(float((hits / ranks * rel).sum() / n_rel))
    return {"n": len(uids), "knn_purity": float(np.mean(purity)), "map": float(np.mean(aps)) if aps else float("nan")}


def load_v1_labels(catalog: pd.DataFrame) -> Dict[str, str]:
    """Carpetas de v1 (sin NOISE) mapeadas a track_uid por nombre de archivo."""
    by_name = parse_v1_clusters()
    name_to_uid = dict(zip(catalog["filename"], catalog["track_uid"]))
    return {name_to_uid[f]: c for f, c in by_name.items() if c != "NOISE" and f in name_to_uid}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_triplets(answers_dir: Path, catalog: pd.DataFrame) -> pd.DataFrame:
    df = load_answers(answers_dir, catalog)
    df = df[df["answer"] != "skip"].copy()
    df["source"] = df["source_file"].map(lambda f: SOURCE_LABELS.get(f, Path(f).stem))
    if "selection_source" in df.columns:
        df["selection_source"] = df["selection_source"].where(df["source"] == "dj_branch")
    return df.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evalúa representaciones contra tripletas del DJ y carpetas de v1.")
    parser.add_argument("--dataset-name", default="test_20")
    parser.add_argument("--config", default=None)
    parser.add_argument("--answers-dir", default=str(ANSWERS_DIR))
    parser.add_argument("--rep", action="append", default=[],
                        help="NOMBRE=RUTA (directorio con embeddings.npy+track_uids.json, o .npy con track_uids.json al lado). Repetible.")
    parser.add_argument("--rep-root", default=None,
                        help="Directorio con subcarpetas <nombre>/embeddings.npy; por defecto artifacts/<dataset>/representations")
    parser.add_argument("--out", default=None, help="JSON de salida (por defecto artifacts/<dataset>/evaluation/representations_<fecha>.json)")
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else None)
    artifacts = resolve_dataset_artifacts(args.dataset_name, config)
    catalog = pd.read_parquet(artifacts / "catalog.parquet")
    triplets = load_triplets(Path(args.answers_dir), catalog)
    print(f"[INFO] Tripletas no saltadas: {len(triplets)} "
          f"({triplets['source'].value_counts().to_dict()})")

    reps: list = []
    feats_path = artifacts / "features" / "bpm_key.parquet"
    if feats_path.exists():
        feats = pd.read_parquet(feats_path)
        reps.append(FeatureBaseline("bpm", feats, "bpm"))
        reps.append(FeatureBaseline("key", feats, "key"))
        print(f"[INFO] Baselines BPM/clave con {len(feats)} temas")
    else:
        print(f"[WARN] {feats_path} no existe: sin baselines BPM/clave")

    rep_root = Path(args.rep_root) if args.rep_root else artifacts / "representations"
    if rep_root.exists():
        for d in sorted(rep_root.iterdir()):
            if (d / "embeddings.npy").exists():
                reps.append(load_representation(d.name, d))
    for spec in args.rep:
        name, _, path = spec.partition("=")
        reps.append(load_representation(name, Path(path)))
    if not any(isinstance(r, Representation) for r in reps):
        print("[WARN] Ninguna representación cargada; solo baselines")

    table = evaluate_triplets(reps, triplets)
    v1 = load_v1_labels(catalog)
    coherence = {rep.name: folder_coherence(rep, v1) for rep in reps}

    pd.set_option("display.width", 200)
    print("\n[RESULT] Accuracy de tripletas (empates = 0.5; bootstrap pareado contra BPM)")
    cols = ["group", "representation", "n", "accuracy", "ci_low", "ci_high", "ties", "majority_class",
            "vs_bpm_diff", "vs_bpm_ci_low", "vs_bpm_ci_high", "vs_bpm_p_gain"]
    cols = [c for c in cols if c in table.columns]
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n[RESULT] Coherencia contra {len(v1)} temas en carpetas de v1 (k={KNN_K})")
    print(pd.DataFrame(coherence).T.to_string(float_format=lambda v: f"{v:.3f}"))

    out = Path(args.out) if args.out else artifacts / "evaluation" / f"representations_{dt.date.today().isoformat()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": args.dataset_name, "date": dt.datetime.now(dt.timezone.utc).isoformat(),
               "n_triplets": int(len(triplets)), "triplets": table.to_dict(orient="records"),
               "folder_coherence": coherence, "v1_labels_file": str(V1_CLUSTERS.relative_to(REPO_ROOT))}
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[INFO] Escrito {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
