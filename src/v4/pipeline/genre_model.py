"""
PURPOSE: Género en la taxonomía de Beatport para los temas que Beatport no tiene (niveles C y D de
         beatport_lookup), aprendido de los temas que sí tiene. No escribe tags.
         Etiquetas: features/beatport.parquet (por defecto solo matches confirmados: A, o B con sello,
         fecha o duración iguales a las del archivo). Representaciones congeladas alineadas por
         track_uids.json: maesthf_full_l7 + clap_full (clasificador) y maesthf_full_styles (gate).
         Dos capas:
           1. gate MAEST: si el estilo Discogs no electrónico más fuerte supera al electrónico más
              fuerte (cociente > --gate-ratio), el género sale de DISCOGS_TO_BEATPORT (rap, pop,
              rock, latin: géneros que la colección casi no tiene para entrenar);
           2. si no, regresión logística (StandardScaler + LogisticRegression balanceada, C fijo, sin
              ajustar contra la evaluación) con las clases de Beatport con >= --min-class ejemplos.
         --eval: validación cruzada de 5 partes estratificada y agrupada por artista (un artista no
           cae en entrenamiento y prueba a la vez): accuracy, top-3, macro-F1, precisión según la
           confianza, reporte por clase y comportamiento del gate -> features/genre_model_eval.json.
         --predict: entrena con todo y escribe features/genre_pred.parquet/.csv con el género final
           (Beatport si hay match A/B, si no el modelo), el top-3 con probabilidades y la fuente.
CHANGELOG:
  - 2026-09-29: Creación inicial.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.beatport import norm, split_local_artists  # noqa: E402
from src.v4.common.catalog import load_catalog  # noqa: E402
from src.v4.common.config_loader import load_config  # noqa: E402
from src.v4.common.embedding_utils import load_track_embeddings  # noqa: E402
from src.v4.common.path_resolver import resolve_dataset_artifacts, resolve_hf_cache  # noqa: E402

CLF_REPS = ("maesthf_full_l7", "clap_full")
STYLES_REP = "maesthf_full_styles"
LR_C = 0.05
SEED = 0
CONF_THRESHOLDS = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7)
# Género padre de Discogs (prefijo del estilo de MAEST) -> género de Beatport (nombres vistos en las
# búsquedas del 2026-09-29: 47 géneros). None = Beatport no tiene equivalente: se informa el padre de
# Discogs marcado como tal.
DISCOGS_TO_BEATPORT: Dict[str, Optional[str]] = {
    "Hip Hop": "Hip-Hop",
    "Latin": "Latin",
    "Pop": "Pop",
    "Rock": "Rock",
    "Funk / Soul": "R&B",
    "Reggae": "Caribbean",
    "Jazz": None,
    "Blues": None,
    "Folk, World, & Country": None,
    "Classical": None,
    "Stage & Screen": None,
    "Brass & Military": None,
    "Children's": None,
    "Non-Music": None,
}


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

def style_labels(hf_cache: Optional[Path]) -> List[str]:
    from transformers import AutoConfig
    from src.v4.pipeline.extract_representations import MAESTHF_NAME
    kwargs = {"cache_dir": str(hf_cache)} if hf_cache else {}
    cfg = AutoConfig.from_pretrained(MAESTHF_NAME, **kwargs)
    return [cfg.id2label[i] for i in range(len(cfg.id2label))]


def load_features(artifacts: Path) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """X = l7 ‖ clap y styles = sigmoides MAEST (400), con las filas de cada representación
    reordenadas por su propio track_uids.json (re-ensamblar una variante puede cambiar el orden)."""
    loaded = [load_track_embeddings(artifacts, rep=rep) for rep in CLF_REPS + (STYLES_REP,)]
    uids = loaded[0][0]
    mats = []
    for rep, (u, m) in zip(CLF_REPS + (STYLES_REP,), loaded):
        if set(u) != set(uids):
            raise ValueError(f"{rep}: otro conjunto de track_uids que {CLF_REPS[0]}")
        pos = {uid: i for i, uid in enumerate(u)}
        mats.append(m[[pos[uid] for uid in uids]].astype(np.float32))
    if not np.isfinite(np.hstack(mats)).all():
        raise ValueError("representaciones con valores no finitos")
    return uids, np.hstack(mats[:-1]), mats[-1]


def gate_scores(styles: np.ndarray, labels: List[str]) -> Tuple[np.ndarray, List[Optional[str]]]:
    """Cociente (estilo no electrónico más fuerte / electrónico más fuerte) y el padre Discogs ganador."""
    parent = np.array([lab.split("---")[0] for lab in labels])
    elec = parent == "Electronic"
    non = np.where(~elec)[0]
    ratio = styles[:, ~elec].max(1) / np.maximum(styles[:, elec].max(1), 1e-6)
    top_parent = [str(parent[non[i]]) for i in styles[:, ~elec].argmax(1)]
    return ratio, top_parent


def gate_genre(discogs_parent: str) -> str:
    mapped = DISCOGS_TO_BEATPORT.get(discogs_parent)
    return mapped if mapped else f"{discogs_parent} (Discogs)"


def primary_artist(s: str) -> str:
    parts = split_local_artists(s.replace(";", ","))
    return norm(parts[0]) if parts else ""


def labeled_frame(bp: pd.DataFrame, uids: List[str], confirmed_only: bool) -> pd.DataFrame:
    ok = bp["level"].isin(["A", "B"]) & bp["bp_genre"].notna()
    if confirmed_only:
        ok &= bp["confirmed"]
    lab = bp[ok].drop_duplicates("track_uid").set_index("track_uid")
    row = {u: i for i, u in enumerate(uids)}
    lab = lab[lab.index.isin(row)]
    lab = lab.assign(row=[row[u] for u in lab.index],
                     group=[primary_artist(a or c) for a, c in zip(lab["bp_artists"].fillna(""), lab["cur_artist"].fillna(""))])
    return lab


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

def make_clf():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=LR_C, max_iter=4000, class_weight="balanced"))


def cross_validate(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import StratifiedGroupKFold
    classes = np.array(sorted(set(y)))
    P = np.zeros((len(y), len(classes)), dtype=np.float64)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, groups):
        clf = make_clf().fit(X[tr], y[tr])
        P[np.ix_(te, np.searchsorted(classes, clf.classes_))] = clf.predict_proba(X[te])
    return classes, P


def metrics(y: np.ndarray, classes: np.ndarray, P: np.ndarray) -> dict:
    from sklearn.metrics import classification_report, f1_score
    pred = classes[P.argmax(1)]
    conf = P.max(1)
    top3 = np.array([y[i] in classes[np.argsort(-P[i])[:3]] for i in range(len(y))])
    vc = pd.Series(y).value_counts()
    out = {
        "n": int(len(y)), "n_classes": int(len(classes)),
        "majority_baseline": round(float(vc.iloc[0] / len(y)), 3), "majority_class": str(vc.index[0]),
        "accuracy": round(float((pred == y).mean()), 3), "top3": round(float(top3.mean()), 3),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 3),
        "by_confidence": [{"min_conf": t, "coverage": round(float((conf >= t).mean()), 3),
                           "accuracy": round(float((pred[conf >= t] == y[conf >= t]).mean()), 3) if (conf >= t).any() else None}
                          for t in CONF_THRESHOLDS],
        "per_class": {k: {m: round(float(v[m]), 2) for m in ("precision", "recall", "f1-score")} | {"n": int(v["support"])}
                      for k, v in classification_report(y, pred, output_dict=True, zero_division=0).items()
                      if k in set(classes)},
    }
    confusion = pd.crosstab(pd.Series(y, name="beatport"), pd.Series(pred, name="pred"))
    pairs = [(a, b, int(confusion.loc[a, b])) for a in confusion.index for b in confusion.columns if a != b]
    out["top_confusions"] = [f"{a} -> {b}: {n}" for a, b, n in sorted(pairs, key=lambda x: -x[2])[:12]]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Modelo de género (taxonomía Beatport) para temas sin match")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--beatport", default="beatport", help="Nombre base de la tabla en features/")
    parser.add_argument("--all-matches", action="store_true", help="Entrenar también con B sin confirmar")
    parser.add_argument("--min-class", type=int, default=15)
    parser.add_argument("--gate-ratio", type=float, default=1.0)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()
    if not (args.eval or args.predict):
        parser.error("usar --eval y/o --predict")

    config = load_config(Path(args.config) if args.config else None)
    artifacts = resolve_dataset_artifacts(args.dataset_name, config)
    out_dir = artifacts / "features"
    bp = pd.read_parquet(out_dir / f"{args.beatport}.parquet")
    uids, X, styles = load_features(artifacts)
    ratio, discogs_parent = gate_scores(styles, style_labels(resolve_hf_cache(config)))
    lab = labeled_frame(bp, uids, confirmed_only=not args.all_matches)
    counts = lab["bp_genre"].value_counts()
    kept = counts[counts >= args.min_class].index
    train = lab[lab["bp_genre"].isin(kept)]
    rows, y, groups = train["row"].to_numpy(), train["bp_genre"].to_numpy(), train["group"].to_numpy()
    print(f"[genre] {args.dataset_name}: {len(uids)} temas con representaciones; etiquetados {len(lab)} "
          f"({'A+B' if args.all_matches else 'confirmados'}); entrenamiento {len(train)} en {len(kept)} clases "
          f"(>= {args.min_class}); fuera {len(lab) - len(train)} en {len(counts) - len(kept)} clases chicas")

    if args.eval:
        classes, P = cross_validate(X[rows], y, groups)
        report = {"dataset": args.dataset_name, "labels": "A+B" if args.all_matches else "confirmed",
                  "features": "+".join(CLF_REPS), "C": LR_C, "min_class": args.min_class,
                  "n_artists": int(len(set(groups))), "classifier": metrics(y, classes, P),
                  "classes_excluded": {k: int(v) for k, v in counts[counts < args.min_class].items()}}
        fired = ratio[lab["row"].to_numpy()] > args.gate_ratio
        g = pd.DataFrame({"beatport": lab["bp_genre"].to_numpy(), "fired": fired,
                          "gate_genre": [gate_genre(discogs_parent[r]) for r in lab["row"]]})
        report["gate"] = {
            "ratio": args.gate_ratio, "fired": int(fired.sum()), "of": int(len(g)),
            "fired_by_beatport_genre": g[g["fired"]].groupby("beatport").size().sort_values(ascending=False).to_dict(),
            "fired_cases": g[g["fired"]].groupby(["beatport", "gate_genre"]).size()
                           .sort_values(ascending=False).head(30).reset_index()
                           .apply(lambda r: f"{r['beatport']} -> {r['gate_genre']}: {r[0]}", axis=1).tolist(),
        }
        (out_dir / "genre_model_eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=1))

    if args.predict:
        clf = make_clf().fit(X[rows], y)
        P = clf.predict_proba(X)
        order = np.argsort(-P, axis=1)[:, :3]
        by_uid = bp.drop_duplicates("track_uid").set_index("track_uid")
        catalog = load_catalog(args.dataset_name, config).set_index("track_uid")
        recs = []
        for i, u in enumerate(uids):
            r = by_uid.loc[u] if u in by_uid.index else None
            level = r["level"] if r is not None else None
            fired = bool(ratio[i] > args.gate_ratio)
            model_genre = gate_genre(discogs_parent[i]) if fired else str(clf.classes_[order[i, 0]])
            has_bp = r is not None and level in ("A", "B") and pd.notna(r["bp_genre"])
            recs.append({
                "track_uid": u, "rel_path": catalog.loc[u, "rel_path"] if u in catalog.index else None,
                "level": level, "confirmed": bool(r["confirmed"]) if r is not None else False,
                "cur_genre": r["cur_genre"] if r is not None else None,
                "beatport_genre": r["bp_genre"] if has_bp else None,
                "hint_genres": r["hint_genres"] if r is not None else None,
                "model_genre": model_genre, "model_source": "gate_discogs" if fired else "classifier",
                "model_conf": None if fired else round(float(P[i, order[i, 0]]), 3),
                "gate_ratio": round(float(ratio[i]), 3), "discogs_parent": discogs_parent[i],
                **{f"top{k + 1}": f"{clf.classes_[order[i, k]]} ({P[i, order[i, k]]:.2f})" for k in range(3)},
                "final_genre": r["bp_genre"] if has_bp else model_genre,
                "final_source": f"beatport_{level}" if has_bp else ("gate_discogs" if fired else "classifier"),
                "model_agrees_with_beatport": (model_genre == r["bp_genre"]) if has_bp else None,
            })
        pred = pd.DataFrame(recs)
        pred.to_parquet(out_dir / "genre_pred.parquet", index=False)
        pred.to_csv(out_dir / "genre_pred.csv", index=False, encoding="utf-8-sig")
        print(pred["final_source"].value_counts().to_string())
        print(pred.loc[pred["final_source"] != "beatport_A", "final_genre"].value_counts().head(20).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
