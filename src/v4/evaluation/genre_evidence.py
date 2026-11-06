"""
PURPOSE: Evidencia medible por género de Beatport, para construir la taxonomía y la similitud entre
         géneros que usará la recomendación en vivo (DECISIONS 2026-11-06, punto 9). Solo cuentan los
         temas con género de Beatport (niveles A y B de features/beatport.parquet): las etiquetas del
         clasificador harían circular la medición. Por género:
           - cuántos temas, BPM típico (mediana y cuartiles, features/bpm_key.parquet);
           - los estilos de Discogs que más predice MAEST (maesthf_full_styles, 400 sigmoides);
           - los subgéneros, si ya se bajaron (features/beatport_genres.parquet).
         Entre géneros con al menos --min-n temas:
           - clap_centroides: coseno entre los centroides CLAP (clap_full, embeddings normalizados);
           - clap_vecinos: de los k vecinos CLAP más cercanos de los temas de A (entre los etiquetados,
             sin el propio), la fracción que es de B (fila A; no es simétrica);
           - clap_vecinos_lift: esa fracción dividida por la parte de B entre los etiquetados (1 = lo
             esperable por azar), promediada en los dos sentidos;
           - versiones: cuántos temas tienen versiones que Beatport puso en A y en B (same_version_genres).
           - borrador_similitud: promedio de los percentiles de las tres señales (centroides, lift y
             versiones / temas del género más chico) fuera de la diagonal; diagonal 1. Es un borrador
             para revisar, no la similitud final.
         Salida: features/genre_evidence/ (generos.csv, clap_centroides.csv, clap_vecinos.csv,
         clap_vecinos_lift.csv, versiones.csv, borrador_similitud.csv, resumen.json). No decide nada:
         es la entrada de la taxonomía.
CHANGELOG:
  - 2026-11-06: Creación.
"""
import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.config_loader import load_config  # noqa: E402
from src.v4.common.embedding_utils import load_track_embeddings  # noqa: E402
from src.v4.common.path_resolver import resolve_dataset_artifacts, resolve_hf_cache  # noqa: E402


def unit(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype="float64")
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def centroid_similarity(emb: np.ndarray, labels: np.ndarray, genres: list) -> pd.DataFrame:
    cents = unit(np.stack([emb[labels == g].mean(axis=0) for g in genres]))
    return pd.DataFrame(cents @ cents.T, index=genres, columns=genres)


def neighbor_mix(emb: np.ndarray, labels: np.ndarray, genres: list, k: int) -> pd.DataFrame:
    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)
    nn = np.argsort(-sim, axis=1)[:, :k]
    out = pd.DataFrame(0.0, index=genres, columns=genres)
    for g in genres:
        rows = np.where(labels == g)[0]
        c = Counter(labels[nn[rows]].ravel())
        total = len(rows) * k
        for h in genres:
            out.loc[g, h] = c.get(h, 0) / total
    return out


def version_pairs(series: pd.Series) -> Counter:
    pairs: Counter = Counter()
    for s in series.dropna():
        gs = sorted({x.strip() for x in str(s).split(";") if x.strip()})
        for a, b in combinations(gs, 2):
            pairs[(a, b)] += 1
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description="Evidencia por género de Beatport (BPM, estilos, CLAP, versiones)")
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--min-n", type=int, default=10, help="Temas mínimos por género para las matrices")
    ap.add_argument("--k", type=int, default=10, help="Vecinos CLAP por tema")
    args = ap.parse_args()

    config = load_config(Path(args.config) if args.config else None)
    artifacts = resolve_dataset_artifacts(args.dataset_name, config)
    feats = artifacts / "features"
    out_dir = feats / "genre_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)

    bp = pd.read_parquet(feats / "beatport.parquet")
    bp = bp[bp["bp_genre"].notna()].set_index("track_uid")
    uids, clap = load_track_embeddings(artifacts, rep="clap_full")
    row = {u: i for i, u in enumerate(uids)}
    lab = bp[bp.index.isin(row)]
    idx = np.array([row[u] for u in lab.index])
    emb = unit(clap[idx])
    labels = lab["bp_genre"].to_numpy()

    counts = lab["bp_genre"].value_counts()
    big = [g for g, n in counts.items() if n >= args.min_n]

    bk = pd.read_parquet(feats / "bpm_key.parquet").set_index("track_uid")
    s_uids, styles = load_track_embeddings(artifacts, rep="maesthf_full_styles")
    s_row = {u: i for i, u in enumerate(s_uids)}
    try:
        from src.v4.pipeline.genre_model import style_labels
        names = style_labels(resolve_hf_cache(config))
    except Exception as exc:                          # sin transformers: los nombres no son críticos
        print(f"[AVISO] sin nombres de estilos de Discogs ({type(exc).__name__}); se usan índices")
        names = [f"estilo_{i}" for i in range(styles.shape[1])]
    sub = None
    if (feats / "beatport_genres.parquet").exists():
        sub = pd.read_parquet(feats / "beatport_genres.parquet").set_index("track_uid")

    rows = []
    for g, n in counts.items():
        us = lab.index[lab["bp_genre"] == g]
        bpm = bk.reindex(us)["bpm"].dropna()
        st = np.stack([styles[s_row[u]] for u in us if u in s_row]) if any(u in s_row for u in us) else None
        top = []
        if st is not None:
            mean = st.mean(axis=0)
            top = [f"{names[i]} {mean[i]:.2f}" for i in np.argsort(-mean)[:5]]
        subs = ""
        if sub is not None:
            vc = sub.reindex(us)["sub_genre"].dropna().value_counts()
            subs = "; ".join(f"{k} {v}" for k, v in vc.items())
        rows.append({"genero": g, "temas": int(n), "confirmados": int(lab.loc[us, "confirmed"].sum()),
                     "bpm_mediana": round(float(bpm.median()), 1) if len(bpm) else None,
                     "bpm_p25": round(float(bpm.quantile(0.25)), 1) if len(bpm) else None,
                     "bpm_p75": round(float(bpm.quantile(0.75)), 1) if len(bpm) else None,
                     "estilos_discogs": " | ".join(top), "subgeneros": subs})
    pd.DataFrame(rows).to_csv(out_dir / "generos.csv", index=False, encoding="utf-8-sig")

    mask = np.isin(labels, big)
    cs = centroid_similarity(emb[mask], labels[mask], big)
    nm = neighbor_mix(emb[mask], labels[mask], big, args.k)
    cs.round(4).to_csv(out_dir / "clap_centroides.csv", encoding="utf-8-sig")
    nm.round(4).to_csv(out_dir / "clap_vecinos.csv", encoding="utf-8-sig")
    share = pd.Series({g: float((labels[mask] == g).mean()) for g in big})
    lift = nm.div(share, axis=1)
    lift = (lift + lift.T) / 2
    lift.round(3).to_csv(out_dir / "clap_vecinos_lift.csv", encoding="utf-8-sig")
    vp = version_pairs(bp["same_version_genres"])
    ver = pd.DataFrame(0.0, index=big, columns=big)
    for (a, b), n in vp.items():
        if a in ver.index and b in ver.index:
            ver.loc[a, b] = ver.loc[b, a] = n / min(counts[a], counts[b])
    off = ~np.eye(len(big), dtype=bool)

    def pct(m: pd.DataFrame) -> pd.DataFrame:
        r = m.where(off).stack().rank(pct=True).unstack()
        return r.reindex(index=big, columns=big)

    draft = (pct(cs) + pct(lift) + pct(ver)) / 3
    draft = draft.where(off, 1.0)
    draft.round(3).to_csv(out_dir / "borrador_similitud.csv", encoding="utf-8-sig")
    pd.DataFrame([{"genero_a": a, "genero_b": b, "temas": n} for (a, b), n in vp.most_common()]) \
        .to_csv(out_dir / "versiones.csv", index=False, encoding="utf-8-sig")
    summary = {"dataset": args.dataset_name, "temas_con_genero_beatport": int(len(lab)),
               "generos": int(len(counts)), "generos_en_matrices": big, "min_n": args.min_n, "k": args.k,
               "representaciones": ["clap_full", "maesthf_full_styles"],
               "subgeneros": bool(sub is not None)}
    (out_dir / "resumen.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {out_dir}: {len(lab)} temas con género de Beatport, {len(counts)} géneros, "
          f"{len(big)} con >= {args.min_n} temas en las matrices")
    return 0


if __name__ == "__main__":
    sys.exit(main())
