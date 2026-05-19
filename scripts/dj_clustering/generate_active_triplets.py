"""
PURPOSE: Driver for D4.2 — generate active triplet questions from Regime 1
         first-sweep disagreement. Selects the top-3 executed sweep configs
         (documented deterministic tie-break), rebuilds each config's
         transformed embedding space, enumerates a nearest-neighbor candidate
         triplet pool, scores each candidate by config disagreement
         (disagreement_margin, with rank-difference variance as tie-break),
         fills any shortfall with cluster-boundary candidates, de-duplicates
         against the already-asked D3.2/D3.3 queue, extends the ignored
         triplet question queue and M3U, and writes an aggregate-only summary.

         Active questions are candidates for future human comparison only.
         This script does not invent or ingest answers (that is D4.3).

         CPU-only; no MERT inference, no audio, no GPU.

CHANGELOG:
  D4.2a - Initial implementation (scaffold; generator not executed in D4.2a).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import pandas as pd

from src.dj_clustering.clustering import (
    cluster_hdbscan,
    load_embedding_source,
    normalize_embeddings,
    reduce_dimensions,
)
from src.dj_clustering.triplets import (
    ACTIVE_BOUNDARY_FILL_SOURCE,
    ACTIVE_DISAGREEMENT_SOURCE,
    assemble_question_df,
    build_active_triplet_list,
    build_knn_index,
    config_rank_difference,
    deduplicate_triplets,
    exclude_existing_triplets,
    existing_triplet_keys,
    rank_active_candidates,
    sample_boundary_triplets,
    sample_nn_triplets,
)

BANNER = "=" * 70

# Seed for active candidate sampling (consistent with the D3.2 triplet lineage).
ACTIVE_SEED = 42
# Seed for rebuilding config embedding spaces (matches the D4.1 sweep seed).
SPACE_SEED = 13
# Default HDBSCAN baseline parameters for the preliminary boundary cluster map.
BOUNDARY_HDBSCAN_MIN_CLUSTER_SIZE = 6
BOUNDARY_HDBSCAN_MIN_SAMPLES = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DJ clustering D4.2: generate active triplet questions"
    )
    p.add_argument(
        "--leaderboard",
        type=Path,
        default=project_root / "runs/dj_clustering/first_sweep/first_sweep_leaderboard.csv",
    )
    p.add_argument(
        "--features-dir",
        type=Path,
        default=project_root / "artifacts/dj_clustering/features",
    )
    p.add_argument(
        "--pair-features",
        type=Path,
        default=project_root / "artifacts/dj_clustering/pairs/pair_features.parquet",
    )
    p.add_argument(
        "--inventory",
        type=Path,
        default=project_root / "artifacts/dj_clustering/inventory/library_inventory.csv",
    )
    p.add_argument(
        "--existing-queue",
        type=Path,
        default=project_root / "artifacts/dj_clustering/triplets/triplet_question_queue.csv",
    )
    p.add_argument(
        "--manual",
        type=Path,
        default=project_root / "artifacts/dj_clustering/triplets/manual_triplets.csv",
    )
    p.add_argument(
        "--skip-log",
        type=Path,
        default=project_root / "artifacts/dj_clustering/triplets/triplet_skip_log.csv",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=project_root / "artifacts/dj_clustering/triplets/triplet_question_queue.csv",
    )
    p.add_argument(
        "--output-m3u",
        type=Path,
        default=project_root / "artifacts/dj_clustering/triplets/triplet_question_queue.m3u",
    )
    p.add_argument(
        "--output-report",
        type=Path,
        default=project_root / "reports/dj_clustering/active_triplet_queue_summary.md",
    )
    p.add_argument("--n-questions", type=int, default=85)
    p.add_argument("--candidate-pool-size", type=int, default=1500)
    p.add_argument("--knn-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=ACTIVE_SEED)
    p.add_argument("--space-seed", type=int, default=SPACE_SEED)
    return p.parse_args()


def load_canonical_inventory(path: Path) -> pd.DataFrame:
    """Load the inventory restricted to canonical decoded tracks."""
    raw = pd.read_csv(path)
    df = raw[
        raw["is_canonical"].astype(bool) & (raw["decode_status"] == "ok")
    ].copy()
    df["track_id"] = df["track_id"].astype(str)
    return df


def build_config_spaces(
    top_configs: List[Dict], features_dir: Path, space_seed: int
) -> List[Dict]:
    """Rebuild each top-3 config's transformed embedding space.

    Returns one dict per config with keys: config_id, matrix, track_index.
    """
    spaces: List[Dict] = []
    for cfg in top_configs:
        matrix, track_ids = load_embedding_source(
            features_dir, cfg["embedding_source"], cfg["mert_aggregation"]
        )
        normalized = normalize_embeddings(matrix, cfg["normalization"])
        transformed = reduce_dimensions(
            normalized, cfg["dim_reduction"], seed=space_seed
        )
        spaces.append(
            {
                "config_id": cfg["config_id"],
                "matrix": transformed,
                "track_index": {tid: i for i, tid in enumerate(track_ids)},
            }
        )
    return spaces


def score_candidates(candidates: List[Dict], spaces: List[Dict]) -> List[Dict]:
    """Attach per-config rank differences to each candidate triplet."""
    for cand in candidates:
        cand["rank_diffs"] = [
            config_rank_difference(
                space["matrix"],
                space["track_index"],
                cand["anchor_track_id"],
                cand["candidate_b_track_id"],
                cand["candidate_c_track_id"],
            )
            for space in spaces
        ]
    return candidates


def build_boundary_cluster_map(
    features_dir: Path, space_seed: int
) -> Dict[str, str]:
    """Preliminary MERT_full_HDBSCAN_default cluster map for boundary fill.

    Uses the default HDBSCAN baseline representation (mert_full /
    last_layer_mean / l2 / umap_15) per the D3.2 fallback policy.
    """
    matrix, track_ids = load_embedding_source(
        features_dir, "mert_full", "last_layer_mean"
    )
    transformed = reduce_dimensions(
        normalize_embeddings(matrix, "l2"), "umap_15", seed=space_seed
    )
    labels = cluster_hdbscan(
        transformed,
        BOUNDARY_HDBSCAN_MIN_CLUSTER_SIZE,
        BOUNDARY_HDBSCAN_MIN_SAMPLES,
    )
    return {tid: str(int(lbl)) for tid, lbl in zip(track_ids, labels)}


def next_question_index(queue_df: pd.DataFrame) -> int:
    """Return the question index to start active numbering from."""
    if queue_df.empty:
        return 1
    nums = []
    for qid in queue_df["question_id"].astype(str):
        digits = "".join(ch for ch in qid if ch.isdigit())
        if digits:
            nums.append(int(digits))
    return (max(nums) + 1) if nums else 1


def write_m3u(question_df: pd.DataFrame, output_path: Path) -> None:
    """Write an M3U review file for the given questions (absolute paths)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#EXTM3U"]
    for _, row in question_df.iterrows():
        qid = row["question_id"]
        for role, path_col in [
            ("anchor", "anchor_file_path"),
            ("B", "candidate_b_file_path"),
            ("C", "candidate_c_file_path"),
        ]:
            lines.append(f"#EXTINF:-1,{qid} {role}")
            lines.append(str(row[path_col]))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_report(
    output_path: Path,
    *,
    total_queue: int,
    n_existing: int,
    n_active: int,
    n_requested: int,
    n_disagreement: int,
    n_boundary_fill: int,
    n_top_configs: int,
    candidate_pool: int,
    disagreement_pool: int,
    n_dedup_existing: int,
    n_dedup_other: int,
    limitation: str,
    seed: int,
    space_seed: int,
) -> None:
    """Write the aggregate-only committed D4.2 summary report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Active Triplet Question Queue Summary (D4.2)",
        "",
        "Aggregate-only report for the sweep-driven active triplet questions.",
        "No track identifiers, question identifiers, paths, or library metadata",
        "values are included.",
        "",
        "## Selection method",
        "",
        "Active questions are selected from disagreement among the top-3",
        "executed Regime 1 first-sweep configs. Top-3 selection uses a",
        "documented deterministic tie-break: sort executed (non-diagnostic)",
        "leaderboard rows by triplet_accuracy descending, then by config_id",
        "ascending, then take the first three. For a candidate triplet (A,B,C)",
        "each config chooses B if d(A,B) < d(A,C); a disagreement exists when",
        "at least one config chooses B and at least one chooses C. Candidates",
        "are ranked by disagreement_margin = min(count_B, count_C) / 3, with",
        "the variance of the B-vs-C rank difference as tie-break. Any shortfall",
        "is filled with cluster-boundary candidates.",
        "",
        "## Results",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Top configs compared | {n_top_configs} |",
        f"| Candidate pool size | {candidate_pool} |",
        f"| Disagreement candidates found | {disagreement_pool} |",
        f"| Active questions requested | {n_requested} |",
        f"| Active questions generated | {n_active} |",
        f"| From disagreement | {n_disagreement} |",
        f"| From boundary fill | {n_boundary_fill} |",
        f"| Removed: duplicates of existing queue | {n_dedup_existing} |",
        f"| Removed: within-set / hash collisions | {n_dedup_other} |",
        f"| Existing queue questions | {n_existing} |",
        f"| Total queue questions after update | {total_queue} |",
        f"| Active sampling seed | {seed} |",
        f"| Config-space rebuild seed | {space_seed} |",
        "",
        "## Status",
        "",
        limitation,
        "",
        "Active questions are candidates for future human comparison only.",
        "No answers are produced in D4.2; answer ingestion is Task D4.3.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    print(BANNER)
    print("DJ clustering D4.2 - generate active triplet questions")
    print(BANNER)

    for path in (
        args.leaderboard,
        args.pair_features,
        args.inventory,
        args.existing_queue,
    ):
        if not Path(path).exists():
            print(f"[ERROR] required input missing: {path}")
            return 1

    inventory_df = load_canonical_inventory(args.inventory)
    print(f"[INFO] canonical decoded tracks: {len(inventory_df)}")

    leaderboard = pd.read_csv(args.leaderboard)
    from src.dj_clustering.triplets import select_top3_configs

    top_configs = select_top3_configs(leaderboard, n=3)
    if len(top_configs) < 2:
        print("[ERROR] fewer than 2 comparable configs; cannot compute disagreement")
        return 1
    print(f"[INFO] top configs compared: {len(top_configs)}")

    spaces = build_config_spaces(top_configs, args.features_dir, args.space_seed)

    pair_df = pd.read_parquet(args.pair_features)
    track_ids = list(inventory_df["track_id"])
    knn_index = build_knn_index(pair_df, track_ids, k=args.knn_k)

    # Candidate pool: oversampled nearest-neighbor triplets.
    candidates = sample_nn_triplets(
        knn_index, inventory_df, n=args.candidate_pool_size, seed=args.seed
    )
    candidates = score_candidates(candidates, spaces)
    ranked = rank_active_candidates(candidates)
    print(f"[INFO] candidate pool: {len(candidates)}; disagreements: {len(ranked)}")

    # Boundary-fill candidates from preliminary HDBSCAN clusters.
    cluster_map = build_boundary_cluster_map(args.features_dir, args.space_seed)
    boundary = sample_boundary_triplets(
        knn_index,
        cluster_map,
        inventory_df,
        n=args.n_questions * 4,
        seed=args.seed,
        selection_source=ACTIVE_BOUNDARY_FILL_SOURCE,
    )

    # De-duplicate against the already-asked queue and within each pool.
    existing_queue = pd.read_csv(args.existing_queue)
    existing_keys = existing_triplet_keys(existing_queue)
    for extra_path in (args.manual, args.skip_log):
        if Path(extra_path).exists():
            existing_keys |= existing_triplet_keys(pd.read_csv(extra_path))

    ranked_before = len(ranked)
    ranked = exclude_existing_triplets(ranked, existing_keys)
    ranked = deduplicate_triplets(ranked, inventory_df)
    boundary = exclude_existing_triplets(boundary, existing_keys)
    boundary = deduplicate_triplets(boundary, inventory_df)
    n_dedup_existing = ranked_before - len(ranked)

    active = build_active_triplet_list(ranked, boundary, args.n_questions)
    active = deduplicate_triplets(active, inventory_df)
    n_disagreement = sum(
        1 for t in active if t["selection_source"] == ACTIVE_DISAGREEMENT_SOURCE
    )
    n_boundary_fill = len(active) - n_disagreement

    start_index = next_question_index(existing_queue)
    active_df = assemble_question_df(active, inventory_df, start_index=start_index)
    combined = pd.concat([existing_queue, active_df], ignore_index=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_csv, index=False)
    write_m3u(active_df, args.output_m3u)

    if len(active) < args.n_questions:
        limitation = (
            f"LIMITATION: generated {len(active)} of {args.n_questions} "
            "requested active questions (candidate supply exhausted)."
        )
    else:
        limitation = (
            f"Generated the full requested {args.n_questions} active questions."
        )

    write_summary_report(
        args.output_report,
        total_queue=len(combined),
        n_existing=len(existing_queue),
        n_active=len(active),
        n_requested=args.n_questions,
        n_disagreement=n_disagreement,
        n_boundary_fill=n_boundary_fill,
        n_top_configs=len(top_configs),
        candidate_pool=len(candidates),
        disagreement_pool=ranked_before,
        n_dedup_existing=n_dedup_existing,
        n_dedup_other=max(0, len(ranked) + len(boundary) - len(active)),
        limitation=limitation,
        seed=args.seed,
        space_seed=args.space_seed,
    )

    print(f"[INFO] active questions generated: {len(active)} "
          f"(disagreement={n_disagreement}, boundary_fill={n_boundary_fill})")
    print(f"[INFO] total queue questions: {len(combined)}")
    print(f"[INFO] {limitation}")
    print(BANNER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
