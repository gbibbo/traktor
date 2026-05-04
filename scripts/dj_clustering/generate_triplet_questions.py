"""
PURPOSE: D3.2 driver. Builds an initial 40-question manual comparison queue
         for the DJ clustering Regime 1 evaluation. Reads the frozen D2.2
         feature embeddings (via D3.1 pair_features.parquet) and the D1.2
         library inventory, generates 20 nearest-neighbor triplet questions
         and 20 cluster-boundary triplet questions, deduplicates, and writes:
           - ignored artifact: triplet_question_queue.csv (14 columns, private)
           - ignored artifact: triplet_question_queue.m3u (absolute paths)
           - committed report: triplet_queue_summary.md (aggregate only)
           - committed report: v4_playlist_mapping_report.md (aggregate only)
         CPU-only; no MERT, no audio loading, no GPU.

CHANGELOG:
  D3.2 - Initial implementation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import pandas as pd

from src.dj_clustering.triplets import (
    DEFAULT_KNN_K,
    DEFAULT_SEED,
    V4_COVERAGE_THRESHOLD,
    assemble_question_df,
    build_knn_index,
    deduplicate_triplets,
    genre_cluster_map,
    parse_v4_cluster_map,
    sample_boundary_triplets,
    sample_nn_triplets,
)

BANNER = "=" * 70


def _write_m3u(question_df: pd.DataFrame, output_path: Path) -> None:
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


def _write_summary_report(
    output_path: Path,
    n_total: int,
    n_nn: int,
    n_boundary: int,
    boundary_source: str,
    v4_dir_name: str,
    v4_l1_groups: int,
    v4_coverage: float,
    v4_coverage_pass: bool,
    fallback_used: bool,
    n_removed_dups: int,
    seed: int,
    knn_k: int,
) -> None:
    coverage_gate = "PASS" if v4_coverage_pass else "FAIL"
    fallback_str = "genre_boundary" if fallback_used else "None"
    lines = [
        "# Triplet Question Queue Summary",
        "",
        "## Generation parameters",
        "",
        "| Parameter | Value |",
        "| :--- | :--- |",
        f"| Total questions | {n_total} |",
        f"| Seed | {seed} |",
        f"| Default embedding source | mert_full / last_layer_mean |",
        f"| KNN k | {knn_k} |",
        "",
        "## Question source breakdown",
        "",
        "| Source | Count |",
        "| :--- | :--- |",
        f"| mert_full_knn (nearest-neighbor) | {n_nn} |",
        f"| {boundary_source} | {n_boundary} |",
        "",
        "## V4 boundary source",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| V4 directory used | {v4_dir_name} |",
        f"| L1 cluster groups | {v4_l1_groups} |",
        f"| V4 mapping coverage | {v4_coverage:.1%} |",
        f"| Coverage gate (≥70%) | {coverage_gate} |",
        f"| Fallback used | {fallback_str} |",
        "",
        "## De-duplication result",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| Candidate duplicates removed | {n_removed_dups} |",
        f"| Final unique questions | {n_total} |",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_v4_mapping_report(
    output_path: Path,
    v4_dir_name: str,
    n_m3u_files: int,
    n_l1_groups: int,
    total_entries: int,
    mapped_entries: int,
    coverage: float,
    primary_method: str,
    extinf_used: bool,
    coverage_pass: bool,
    boundary_source_used: str,
) -> None:
    coverage_gate = "YES" if coverage_pass else "NO"
    fallback_desc = "EXTINF metadata fallback" if extinf_used else "none"
    lines = [
        "# V4 Playlist Mapping Report",
        "",
        "## Source",
        "",
        "| Field | Value |",
        "| :--- | :--- |",
        f"| V4 directory | {v4_dir_name} |",
        f"| M3U files parsed | {n_m3u_files} |",
        f"| L1 groups | {n_l1_groups} |",
        f"| Path format | Windows absolute paths (mapped by basename) |",
        "",
        "## Mapping result",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| Total playlist entries | {total_entries} |",
        f"| Entries mapped to canonical track_id | {mapped_entries} |",
        f"| Coverage | {coverage:.1%} |",
        f"| Primary mapping method | {primary_method} |",
        f"| Secondary mapping method | {fallback_desc} |",
        "",
        "## Decision",
        "",
        "| Decision | Value |",
        "| :--- | :--- |",
        f"| Coverage ≥70% | {coverage_gate} |",
        f"| Boundary source used for D3.2 | {boundary_source_used} |",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _count_v4_entries(v4_dir: Path) -> tuple[int, int, int, int]:
    """Return (n_m3u, n_l1_groups, total_entries, mapped_cannot_determine).

    We cannot know mapped_entries here without re-running the parser; the
    caller passes the cluster_map size for that. Returns total_entries from
    a lightweight scan.
    """
    m3u_files = sorted(v4_dir.rglob("*.m3u"))
    n_l1_groups = len({f.parent.name.split("_")[1] for f in m3u_files if len(f.parent.name.split("_")) >= 2})
    total_entries = 0
    for m3u_path in m3u_files:
        try:
            content = m3u_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                total_entries += 1
    return len(m3u_files), n_l1_groups, total_entries


def main() -> int:
    parser = argparse.ArgumentParser(
        description="D3.2: generate initial manual comparison question queue"
    )
    parser.add_argument(
        "--inventory",
        default="artifacts/dj_clustering/inventory/library_inventory.csv",
    )
    parser.add_argument(
        "--pair-features",
        default="artifacts/dj_clustering/pairs/pair_features.parquet",
    )
    parser.add_argument(
        "--v4-playlist-dir",
        default="playlists/V4_5",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=DEFAULT_KNN_K,
    )
    parser.add_argument(
        "--output-csv",
        default="artifacts/dj_clustering/triplets/triplet_question_queue.csv",
    )
    parser.add_argument(
        "--output-m3u",
        default="artifacts/dj_clustering/triplets/triplet_question_queue.m3u",
    )
    parser.add_argument(
        "--output-report",
        default="reports/dj_clustering/triplet_queue_summary.md",
    )
    parser.add_argument(
        "--v4-mapping-report",
        default="reports/dj_clustering/v4_playlist_mapping_report.md",
    )
    args = parser.parse_args()

    print(BANNER)
    print("[INFO] D3.2: Generate initial manual comparison question queue")
    print(BANNER)

    # ------------------------------------------------------------------
    # 1. Load inventory
    # ------------------------------------------------------------------
    inventory_path = Path(args.inventory)
    if not inventory_path.exists():
        print(f"[ERROR] inventory not found: {inventory_path}")
        return 1
    print(f"[INFO] Loading inventory from {inventory_path}")
    inventory_raw = pd.read_csv(inventory_path)
    inventory_df = inventory_raw[
        inventory_raw["is_canonical"].astype(bool)
        & (inventory_raw["decode_status"] == "ok")
    ].copy()
    inventory_df["track_id"] = inventory_df["track_id"].astype(str)
    n_canonical = len(inventory_df)
    print(f"[INFO] Canonical decoded tracks: {n_canonical}")

    # ------------------------------------------------------------------
    # 2. Load pair features
    # ------------------------------------------------------------------
    pair_path = Path(args.pair_features)
    if not pair_path.exists():
        print(f"[ERROR] pair_features not found: {pair_path}")
        return 1
    print(f"[INFO] Loading pair features")
    pair_df = pd.read_parquet(pair_path)
    print(f"[INFO] Pair rows: {len(pair_df)}, columns: {len(pair_df.columns)}")

    # ------------------------------------------------------------------
    # 3. Build KNN index
    # ------------------------------------------------------------------
    print(f"[INFO] Building KNN index (k={args.knn_k})")
    track_ids = list(inventory_df["track_id"])
    knn_index = build_knn_index(pair_df, track_ids, k=args.knn_k)
    n_with_neighbors = sum(1 for v in knn_index.values() if len(v) >= 2)
    print(f"[INFO] Tracks with ≥2 KNN neighbors: {n_with_neighbors}/{n_canonical}")

    # ------------------------------------------------------------------
    # 4. V4 cluster map
    # ------------------------------------------------------------------
    v4_dir = Path(args.v4_playlist_dir)
    v4_dir_name = v4_dir.name
    fallback_used = False
    extinf_used = False
    n_m3u_files = 0
    n_l1_groups = 0
    total_entries = 0

    if v4_dir.exists():
        n_m3u_files, n_l1_groups, total_entries = _count_v4_entries(v4_dir)
        print(
            f"[INFO] V4 playlist dir: {v4_dir_name} "
            f"({n_m3u_files} m3u files, {n_l1_groups} L1 groups, "
            f"{total_entries} entries)"
        )
        cluster_map, v4_coverage, v4_method = parse_v4_cluster_map(
            v4_dir, inventory_df
        )
        extinf_used = "extinf_fallback" in v4_method
        mapped_entries = len(cluster_map)
        v4_coverage_pass = v4_coverage >= V4_COVERAGE_THRESHOLD
        print(
            f"[INFO] V4 mapping: {mapped_entries}/{total_entries} "
            f"({v4_coverage:.1%} coverage, gate {'PASS' if v4_coverage_pass else 'FAIL'})"
        )
        if not v4_coverage_pass:
            print(
                f"[WARN] V4 coverage {v4_coverage:.1%} < {V4_COVERAGE_THRESHOLD:.0%}; "
                f"switching to genre boundary fallback"
            )
            cluster_map = genre_cluster_map(inventory_df)
            fallback_used = True
            boundary_source = "genre_boundary"
        else:
            boundary_source = "v4_boundary"
            n_l1_groups = len(set(cluster_map.values()))
    else:
        print(f"[WARN] V4 playlist dir not found: {v4_dir}; using genre boundary fallback")
        cluster_map = genre_cluster_map(inventory_df)
        fallback_used = True
        boundary_source = "genre_boundary"
        v4_coverage = 0.0
        v4_coverage_pass = False
        mapped_entries = 0

    n_clusters = len(set(cluster_map.values()))
    print(f"[INFO] Cluster groups for boundary sampling: {n_clusters}")
    if n_clusters < 3 and fallback_used:
        print("[ERROR] Fewer than 3 distinct genre groups; cannot generate boundary triplets")
        return 1

    # ------------------------------------------------------------------
    # 5. Sample NN triplets (20)
    # ------------------------------------------------------------------
    print(f"[INFO] Sampling 20 nearest-neighbor questions (seed={args.seed})")
    nn_triplets = sample_nn_triplets(knn_index, inventory_df, n=20, seed=args.seed)
    print(f"[INFO] NN triplets sampled: {len(nn_triplets)}")

    # ------------------------------------------------------------------
    # 6. Sample boundary triplets (20)
    # ------------------------------------------------------------------
    print(f"[INFO] Sampling 20 boundary questions (source={boundary_source})")
    bnd_triplets = sample_boundary_triplets(
        knn_index,
        cluster_map,
        inventory_df,
        n=20,
        seed=args.seed,
        selection_source=boundary_source,
    )
    print(f"[INFO] Boundary triplets sampled: {len(bnd_triplets)}")

    # ------------------------------------------------------------------
    # 7. Combine, deduplicate, assign question_ids
    # ------------------------------------------------------------------
    all_triplets = nn_triplets + bnd_triplets
    before_dedup = len(all_triplets)
    all_triplets = deduplicate_triplets(all_triplets, inventory_df)
    n_removed = before_dedup - len(all_triplets)
    n_total = len(all_triplets)
    n_nn_final = sum(1 for t in all_triplets if t["selection_source"] == "mert_full_knn")
    n_bnd_final = n_total - n_nn_final
    print(f"[INFO] After dedup: {n_total} questions ({n_removed} removed)")

    if n_total < 40:
        print(
            f"[WARN] Only {n_total} questions generated (target: 40). "
            "Stopping before commit — report count and reason to user."
        )
        # Still write outputs for inspection before deciding whether to commit.

    # ------------------------------------------------------------------
    # 8. Assemble DataFrame and write outputs
    # ------------------------------------------------------------------
    question_df = assemble_question_df(all_triplets, inventory_df)

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    question_df.to_csv(out_csv, index=False)
    print(f"[INFO] Wrote {len(question_df)} questions to {out_csv} (ignored artifact)")

    out_m3u = Path(args.output_m3u)
    _write_m3u(question_df, out_m3u)
    print(f"[INFO] Wrote M3U to {out_m3u} (ignored artifact)")

    # ------------------------------------------------------------------
    # 9. Write committed reports (aggregate only)
    # ------------------------------------------------------------------
    out_report = Path(args.output_report)
    _write_summary_report(
        out_report,
        n_total=n_total,
        n_nn=n_nn_final,
        n_boundary=n_bnd_final,
        boundary_source=boundary_source,
        v4_dir_name=v4_dir_name,
        v4_l1_groups=n_l1_groups,
        v4_coverage=v4_coverage,
        v4_coverage_pass=v4_coverage_pass,
        fallback_used=fallback_used,
        n_removed_dups=n_removed,
        seed=args.seed,
        knn_k=args.knn_k,
    )
    print(f"[INFO] Wrote summary report to {out_report}")

    out_v4_report = Path(args.v4_mapping_report)
    _write_v4_mapping_report(
        out_v4_report,
        v4_dir_name=v4_dir_name,
        n_m3u_files=n_m3u_files,
        n_l1_groups=n_l1_groups,
        total_entries=total_entries,
        mapped_entries=mapped_entries,
        coverage=v4_coverage,
        primary_method="normalized filename",
        extinf_used=extinf_used,
        coverage_pass=v4_coverage_pass,
        boundary_source_used=boundary_source,
    )
    print(f"[INFO] Wrote V4 mapping report to {out_v4_report}")

    # ------------------------------------------------------------------
    # 10. Final status
    # ------------------------------------------------------------------
    print(BANNER)
    if n_total < 40:
        print(
            f"[WARN] INCOMPLETE: generated {n_total}/40 questions. "
            "Review logs above and do not advance tracker."
        )
        return 1

    print(f"[INFO] D3.2 DONE: {n_total} questions written.")
    print(f"[INFO]   NN questions       : {n_nn_final}")
    print(f"[INFO]   Boundary questions : {n_bnd_final}")
    print(f"[INFO]   V4 coverage        : {v4_coverage:.1%}")
    print(f"[INFO]   Boundary source    : {boundary_source}")
    print(BANNER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
