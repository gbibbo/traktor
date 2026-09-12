"""
PURPOSE: Cruce empírico entre las respuestas del anotador DJ (tripletas "¿B o C tras el ancla?")
         y los artefactos de clustering que sí están versionados en el repo: los clusters de
         legacy v1 (EffNet+MAEST, UMAP 2D + HDBSCAN, nombrados a mano), los de legacy v2
         (EffNet sobre stem de batería), las predicciones de genre_discogs400 y las playlists
         exportadas de V4 (MERT). Sirve para comparar representaciones sin re-extraer audio.
         Dos predictores por artefacto: distancia euclídea en las coordenadas UMAP 2D guardadas
         (cuando existen) y "misma carpeta que el ancla" (empates cuentan 0.5).
CHANGELOG:
  - 2026-09-12: Creación inicial para la revisión científica (docs/reports/scientific_review_2026-09-12.md).
"""
from __future__ import annotations

import csv
import glob
import os
import re
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

ANSWERS_DIR = REPO_ROOT / "tools" / "dj_feedback" / "answers"
V1_CLUSTERS = REPO_ROOT / "legacy" / "v2" / "results" / "full_collection" / "clusters_by_group.txt"
V1_RESULTS = REPO_ROOT / "legacy" / "v2" / "results" / "full_collection" / "results.csv"
V2_L1 = REPO_ROOT / "legacy" / "v2" / "results" / "v2_hierarchy" / "level1_clusters.csv"
V2_FINAL = REPO_ROOT / "legacy" / "v2" / "results" / "v2_hierarchy" / "final_organization.csv"
V2_GENRE = REPO_ROOT / "legacy" / "v2" / "results" / "v2_hierarchy" / "genre_predictions.csv"
V4_PLAYLISTS = REPO_ROOT / "playlists" / "V4_5"


def load_answers() -> list[dict]:
    rows: dict[str, dict] = {}
    for path in sorted(ANSWERS_DIR.glob("*.csv")):
        with open(path, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                r["answer"] = r["answer"].strip().upper()
                prev = rows.get(r["question_id"])
                if prev is None or r["answered_at"] >= prev["answered_at"]:
                    rows[r["question_id"]] = r
    return [r for r in rows.values() if r["answer"] in ("B", "C")]


def parse_v1_clusters() -> dict[str, str]:
    labels: dict[str, str] = {}
    current: Optional[str] = None
    for line in V1_CLUSTERS.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"=== (CLUSTER \d+|NOISE)", line.strip())
        if m:
            current = m.group(1)
            continue
        if line.startswith("  ") and current:
            labels[line.strip()] = current
    return labels


def parse_v4_playlists() -> dict[str, tuple[str, str]]:
    labels: dict[str, tuple[str, str]] = {}
    for p in glob.glob(str(V4_PLAYLISTS / "**" / "*.m3u"), recursive=True):
        l1, l2 = Path(p).parent.name, Path(p).stem
        for line in open(p, encoding="utf-8", errors="replace"):
            line = line.strip()
            if line and not line.startswith("#"):
                labels[os.path.basename(line)] = (l1, l2)
    return labels


def same_folder_accuracy(rows: list[dict], label_of: Callable[[str], Optional[str]],
                         noise: set, tag: str) -> None:
    n = decisive = hit = unmatched = 0
    for r in rows:
        a, b, c = label_of(r["anchor"]), label_of(r["candidate_b"]), label_of(r["candidate_c"])
        if a is None or b is None or c is None:
            unmatched += 1
            continue
        same_b = a == b and a not in noise
        same_c = a == c and a not in noise
        n += 1
        if same_b != same_c:
            decisive += 1
            hit += int((same_b and r["answer"] == "B") or (same_c and r["answer"] == "C"))
    overall = (hit + 0.5 * (n - decisive)) / n if n else float("nan")
    dec_acc = hit / decisive if decisive else float("nan")
    print(f"{tag:58s} n={n:3d} unmatched={unmatched} decisive={decisive:2d} "
          f"decisive_acc={dec_acc:.3f} overall_ties_as_half={overall:.3f}")


def coord_accuracy(rows: list[dict], df: pd.DataFrame, tag: str, seed: int = 0) -> None:
    xy = {r.track: np.array([r.umap_x, r.umap_y]) for r in df.itertuples()}
    outcomes = []
    for r in rows:
        a, b, c = xy.get(r["anchor"]), xy.get(r["candidate_b"]), xy.get(r["candidate_c"])
        if a is None or b is None or c is None:
            continue
        pred = "B" if np.linalg.norm(a - b) < np.linalg.norm(a - c) else "C"
        outcomes.append(int(pred == r["answer"]))
    o = np.array(outcomes)
    rng = np.random.default_rng(seed)
    bs = [rng.choice(o, len(o)).mean() for _ in range(2000)]
    print(f"{tag:58s} n={len(o):3d} acc={o.mean():.3f} "
          f"bootstrap95=({np.percentile(bs, 2.5):.2f},{np.percentile(bs, 97.5):.2f})")


def main() -> None:
    rows = load_answers()
    print(f"tripletas no saltadas: {len(rows)}")
    v1 = parse_v1_clusters()
    same_folder_accuracy(rows, v1.get, {"NOISE"}, "v1 clusters nombrados a mano (effnet+maest, UMAP+HDBSCAN)")
    v2 = pd.read_csv(V2_FINAL).set_index("track")
    same_folder_accuracy(rows, lambda f: v2["cluster_l1"].get(f) if f in v2.index else None,
                         {"Noise"}, "v2 L1 (effnet sobre stem drums)")
    same_folder_accuracy(rows, lambda f: v2["cluster_l2"].get(f) if f in v2.index else None,
                         {"Noise"}, "v2 L2 (effnet full mix)")
    v4 = parse_v4_playlists()
    same_folder_accuracy(rows, lambda f: v4.get(f, (None, None))[0], {"All_Noise"},
                         "V4_5 L1 (MERT perc, PCA50+HDBSCAN+1NN)")
    same_folder_accuracy(rows, lambda f: v4.get(f, (None, None))[1], {"All_Noise"},
                         "V4_5 L2 (MERT full)")
    coord_accuracy(rows, pd.read_csv(V1_RESULTS), "v1 distancia UMAP-2D (effnet+maest concat)")
    coord_accuracy(rows, pd.read_csv(V2_L1), "v2 distancia UMAP-2D (effnet drums)")
    g = pd.read_csv(V2_GENRE)
    gv = {r.track: {r.genre_1: r.conf_1, r.genre_2: r.conf_2, r.genre_3: r.conf_3} for r in g.itertuples()}
    n = hit = ties = 0
    for r in rows:
        a, b, c = gv.get(r["anchor"]), gv.get(r["candidate_b"]), gv.get(r["candidate_c"])
        if a is None or b is None or c is None:
            continue
        sb = sum(a.get(k, 0) * b.get(k, 0) for k in a)
        sc = sum(a.get(k, 0) * c.get(k, 0) for k in a)
        n += 1
        if sb == sc:
            ties += 1
            hit += 0.5
        else:
            hit += int(("B" if sb > sc else "C") == r["answer"])
    print(f"{'genre_discogs400 top-3 similitud de vectores':58s} n={n:3d} acc={hit / n:.3f} ties={ties}")


if __name__ == "__main__":
    main()
