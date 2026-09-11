"""
PURPOSE: Cargar y evaluar la evidencia humana de tripletas del Anotador DJ
         (tools/dj_feedback/answers/*.csv). Semántica de la respuesta (Gabriel, 2026-09-11):
         "cuál de los dos candidatos tocaría a continuación del ancla sin salto de estilo",
         es decir mezclabilidad, no pertenencia a carpeta. Provee:
           - load_answers(): une todos los CSV, resuelve filename -> track_uid vía catálogo,
             conserva la última respuesta por question_id.
           - baseline_accuracy(): acierto de predictores simples (clave sola, BPM solo,
             clave+BPM, azar) sobre las tripletas no saltadas, con intervalo binomial.
         CLI: python src/v4/evaluation/triplet_evidence.py --dataset-name test_20
CHANGELOG:
  - 2026-09-11: Creación inicial.
"""
import argparse
import math
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.config_loader import load_config  # noqa: E402
from src.v4.common.harmonic import key_compatibility, to_camelot  # noqa: E402
from src.v4.common.path_resolver import resolve_dataset_artifacts  # noqa: E402

ANSWERS_DIR = REPO_ROOT / "tools" / "dj_feedback" / "answers"
VALID_ANSWERS = ("B", "C", "skip")


def load_answers(answers_dir: Path = ANSWERS_DIR,
                 catalog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Une todos los CSV de respuestas. Última respuesta por question_id gana.

    Columnas de salida: question_id, anchor, candidate_b, candidate_c, answer, answered_at,
    source_file y, si hay catálogo, anchor_uid, b_uid, c_uid.
    """
    files = sorted(answers_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No answer CSV files in {answers_dir}")
    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="utf-8-sig")
        required = {"question_id", "anchor", "candidate_b", "candidate_c", "answer"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{f.name}: missing columns {sorted(missing)}")
        df["source_file"] = f.name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["answer"] = df["answer"].astype(str).str.strip().str.upper().replace({"SKIP": "skip"})
    bad = df[~df["answer"].isin(VALID_ANSWERS)]
    if not bad.empty:
        raise ValueError(f"Invalid answer values: {bad['answer'].unique().tolist()}")
    if "answered_at" in df.columns:
        df = df.sort_values("answered_at", kind="stable")
    df = df.drop_duplicates("question_id", keep="last").reset_index(drop=True)

    if catalog is not None:
        name_to_uid = dict(zip(catalog["filename"], catalog["track_uid"]))
        for col, out in (("anchor", "anchor_uid"), ("candidate_b", "b_uid"), ("candidate_c", "c_uid")):
            df[out] = df[col].map(name_to_uid)
        unresolved = df[df[["anchor_uid", "b_uid", "c_uid"]].isna().any(axis=1)]
        if not unresolved.empty:
            print(f"[WARN] {len(unresolved)} triplets reference files not in catalog; dropped")
            df = df.drop(unresolved.index).reset_index(drop=True)
    return df


def summarize(df: pd.DataFrame) -> Dict[str, int]:
    counts = df["answer"].value_counts().to_dict()
    return {
        "n_questions": int(len(df)),
        "n_B": int(counts.get("B", 0)),
        "n_C": int(counts.get("C", 0)),
        "n_skip": int(counts.get("skip", 0)),
        "n_non_skip": int(len(df) - counts.get("skip", 0)),
        "n_unique_anchors": int(df["anchor"].nunique()),
    }


# ---------------------------------------------------------------------------
# Baselines: predicen B o C a partir de BPM y clave
# ---------------------------------------------------------------------------

def bpm_similarity(bpm_a: float, bpm_b: float) -> float:
    """Similitud de tempo en [0,1] con equivalencia 0.5x / 1x / 2x. 1 - min diff / 16 BPM."""
    if not (np.isfinite(bpm_a) and np.isfinite(bpm_b)) or bpm_a <= 0 or bpm_b <= 0:
        return 0.5
    diff = min(abs(bpm_a - bpm_b * f) for f in (0.5, 1.0, 2.0))
    return max(0.0, 1.0 - diff / 16.0)


def _score_fn(features: pd.DataFrame, w_key: float, w_bpm: float) -> Callable[[str, str], float]:
    feats = features.set_index("track_uid")

    def score(uid_anchor: str, uid_cand: str) -> float:
        a, c = feats.loc[uid_anchor], feats.loc[uid_cand]
        s_key = key_compatibility(to_camelot(a["key"]), to_camelot(c["key"]))
        s_bpm = bpm_similarity(float(a["bpm"]), float(c["bpm"]))
        return w_key * s_key + w_bpm * s_bpm
    return score


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def baseline_accuracy(df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Acierto de predictores simples sobre tripletas no saltadas.

    Un empate exacto del predictor cuenta como 0.5 (equivale a elegir al azar).
    """
    d = df[df["answer"] != "skip"]
    d = d[d["anchor_uid"].isin(features["track_uid"]) & d["b_uid"].isin(features["track_uid"])
          & d["c_uid"].isin(features["track_uid"])]
    n = len(d)
    baselines = {
        "key_only": _score_fn(features, 1.0, 0.0),
        "bpm_only": _score_fn(features, 0.0, 1.0),
        "key_plus_bpm": _score_fn(features, 0.5, 0.5),
    }
    rows = []
    for name, fn in baselines.items():
        correct = 0.0
        ties = 0
        for _, r in d.iterrows():
            sb, sc = fn(r["anchor_uid"], r["b_uid"]), fn(r["anchor_uid"], r["c_uid"])
            if abs(sb - sc) < 1e-9:
                correct += 0.5
                ties += 1
            elif (sb > sc) == (r["answer"] == "B"):
                correct += 1
        lo, hi = wilson_interval(int(round(correct)), n)
        rows.append({"baseline": name, "n": n, "accuracy": correct / n if n else float("nan"),
                     "ci95_low": lo, "ci95_high": hi, "ties": ties})
    rows.append({"baseline": "random", "n": n, "accuracy": 0.5, "ci95_low": float("nan"),
                 "ci95_high": float("nan"), "ties": 0})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evidencia de tripletas del DJ: resumen y baselines.")
    parser.add_argument("--dataset-name", default="test_20")
    parser.add_argument("--config", default=None)
    parser.add_argument("--answers-dir", default=str(ANSWERS_DIR))
    parser.add_argument("--out", default=None, help="CSV de salida con las tripletas resueltas a track_uid.")
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else None)
    artifacts = resolve_dataset_artifacts(args.dataset_name, config)
    catalog = pd.read_parquet(artifacts / "catalog.parquet")

    df = load_answers(Path(args.answers_dir), catalog)
    print("[INFO] Evidence summary:", summarize(df))

    out = Path(args.out) if args.out else artifacts / "evidence" / "manual_triplets.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[INFO] Wrote {out}")

    feats_path = artifacts / "features" / "bpm_key.parquet"
    if feats_path.exists():
        features = pd.read_parquet(feats_path)
        table = baseline_accuracy(df, features)
        print("\n[INFO] Baseline accuracy on non-skip triplets (key from Essentia):")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    else:
        print(f"[WARN] {feats_path} not found: run phase1_extract.py --essentia-only first for baselines.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
