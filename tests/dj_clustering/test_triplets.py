"""
PURPOSE: Unit tests for src/dj_clustering/triplets.py — KNN index construction,
         V4 playlist cluster-map parsing (Windows paths, EXTINF fallback,
         coverage gate), genre fallback map, nearest-neighbor and boundary
         triplet sampling (count, determinism, no-self-reference, cross-cluster
         constraint, deduplication), question DataFrame schema, and
         question_id format.  Pure Python; no audio, no GPU, no HF downloads.

CHANGELOG:
  D3.2 - Initial implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dj_clustering.triplets import (
    DEFAULT_KNN_K,
    QUESTION_COLUMNS,
    assemble_question_df,
    build_knn_index,
    deduplicate_triplets,
    genre_cluster_map,
    parse_v4_cluster_map,
    sample_boundary_triplets,
    sample_nn_triplets,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_inventory(n: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    genres = ["techno", "house", "techno", "house", "techno", "house"]
    rows = []
    for i in range(n):
        rows.append(
            {
                "track_id": f"tid{i:03d}",
                "audio_content_hash": f"hash{i:03d}",
                "file_name": f"track_{i:03d}.mp3",
                "file_path": f"/mnt/audio/track_{i:03d}.mp3",
                "artist": f"Artist {i}",
                "title": f"Title {i}",
                "genre": genres[i % len(genres)],
                "is_canonical": True,
                "decode_status": "ok",
            }
        )
    return pd.DataFrame(rows)


def _make_pair_df(
    track_ids: list[str],
    seed: int = 0,
    all_available: bool = True,
) -> pd.DataFrame:
    """Return upper-triangle pair DataFrame with random cosine similarities."""
    rng = np.random.default_rng(seed)
    sorted_ids = sorted(track_ids)
    rows = []
    for i, a in enumerate(sorted_ids):
        for b in sorted_ids[i + 1 :]:
            rows.append(
                {
                    "track_id_a": a,
                    "track_id_b": b,
                    "mert_full_last_layer_mean__cosine_similarity": float(
                        rng.uniform(0.1, 0.99)
                    ),
                    "mert_full_last_layer_mean__available": all_available,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# KNN index tests
# ---------------------------------------------------------------------------


def test_build_knn_index_basic():
    inv = _make_inventory(5)
    track_ids = list(inv["track_id"])
    pair_df = _make_pair_df(track_ids)
    knn = build_knn_index(pair_df, track_ids)
    for tid in track_ids:
        assert tid in knn
        neighbors = knn[tid]
        assert len(neighbors) == 4  # 5 tracks → 4 neighbors each
        sims = [s for s, _ in neighbors]
        # Sorted descending.
        assert sims == sorted(sims, reverse=True)


def test_build_knn_index_only_available():
    inv = _make_inventory(4)
    track_ids = list(inv["track_id"])
    pair_df = _make_pair_df(track_ids, all_available=True)
    # Mark one pair as unavailable.
    pair_df.loc[0, "mert_full_last_layer_mean__available"] = False
    knn = build_knn_index(pair_df, track_ids)
    a = pair_df.loc[0, "track_id_a"]
    b = pair_df.loc[0, "track_id_b"]
    # The unavailable pair should not appear in either track's neighbor list.
    neighbor_ids_a = {ntid for _, ntid in knn[a]}
    neighbor_ids_b = {ntid for _, ntid in knn[b]}
    assert b not in neighbor_ids_a
    assert a not in neighbor_ids_b


def test_build_knn_index_determinism():
    inv = _make_inventory(5)
    track_ids = list(inv["track_id"])
    pair_df = _make_pair_df(track_ids)
    knn1 = build_knn_index(pair_df, track_ids)
    knn2 = build_knn_index(pair_df.sample(frac=1, random_state=99), track_ids)
    for tid in track_ids:
        assert knn1[tid] == knn2[tid]


def test_build_knn_index_k_cap():
    inv = _make_inventory(10)
    track_ids = list(inv["track_id"])
    pair_df = _make_pair_df(track_ids)
    knn = build_knn_index(pair_df, track_ids, k=3)
    for tid in track_ids:
        assert len(knn[tid]) <= 3


# ---------------------------------------------------------------------------
# V4 cluster map parsing tests
# ---------------------------------------------------------------------------


def test_parse_v4_cluster_map_windows_paths(tmp_path):
    v4_dir = tmp_path / "V4_5"
    l1_a = v4_dir / "L1_A_Group A"
    l1_a.mkdir(parents=True)

    inv = _make_inventory(2)
    # m3u uses Windows-style paths.
    m3u = l1_a / "L2_A1.m3u"
    m3u.write_text(
        "#EXTM3U\n"
        "#EXTINF:-1,Display Name 0\n"
        "C:\\Música\\2024\\track_000.mp3\n"
        "#EXTINF:-1,Display Name 1\n"
        "C:\\Música\\2024\\subfolder\\track_001.mp3\n"
    )

    cluster_map, coverage, method = parse_v4_cluster_map(v4_dir, inv)
    assert "tid000" in cluster_map
    assert cluster_map["tid000"] == "L1_A"
    assert "tid001" in cluster_map
    assert cluster_map["tid001"] == "L1_A"
    assert coverage == 1.0
    assert "normalized_filename" in method


def test_parse_v4_cluster_map_multiple_l1_groups(tmp_path):
    v4_dir = tmp_path / "V4_5"
    for g in ["A", "B"]:
        group_dir = v4_dir / f"L1_{g}_Group {g}"
        group_dir.mkdir(parents=True)

    inv = _make_inventory(4)
    (v4_dir / "L1_A_Group A" / "L2_A1.m3u").write_text(
        "#EXTM3U\n"
        "C:\\Music\\track_000.mp3\n"
        "C:\\Music\\track_001.mp3\n"
    )
    (v4_dir / "L1_B_Group B" / "L2_B1.m3u").write_text(
        "#EXTM3U\n"
        "C:\\Music\\track_002.mp3\n"
        "C:\\Music\\track_003.mp3\n"
    )

    cluster_map, coverage, _ = parse_v4_cluster_map(v4_dir, inv)
    assert cluster_map["tid000"] == "L1_A"
    assert cluster_map["tid001"] == "L1_A"
    assert cluster_map["tid002"] == "L1_B"
    assert cluster_map["tid003"] == "L1_B"
    assert coverage == 1.0


def test_parse_v4_cluster_map_coverage_gate(tmp_path):
    """50% coverage is returned as a float; caller decides fallback."""
    v4_dir = tmp_path / "V4_5"
    l1_a = v4_dir / "L1_A_Group A"
    l1_a.mkdir(parents=True)

    inv = _make_inventory(1)  # only track_000 is in inventory
    m3u = l1_a / "L2_A1.m3u"
    m3u.write_text(
        "#EXTM3U\n"
        "C:\\Music\\track_000.mp3\n"    # maps to tid000
        "C:\\Music\\nonexistent.mp3\n"  # unmapped
    )

    _, coverage, _ = parse_v4_cluster_map(v4_dir, inv)
    assert abs(coverage - 0.5) < 1e-9


def test_parse_v4_cluster_map_extinf_fallback(tmp_path):
    """EXTINF display-name fallback matches when filename differs."""
    v4_dir = tmp_path / "V4_5"
    l1_b = v4_dir / "L1_B_Group B"
    l1_b.mkdir(parents=True)

    # Inventory has track with file_name 'track_000.mp3'
    inv = _make_inventory(1)
    # m3u references a completely different filename, but EXTINF stem = 'track_000'
    m3u = l1_b / "L2_B1.m3u"
    m3u.write_text(
        "#EXTM3U\n"
        "#EXTINF:-1,track_000\n"
        "C:\\Music\\some_other_name_entirely.mp3\n"
    )

    cluster_map, coverage, method = parse_v4_cluster_map(v4_dir, inv)
    assert "tid000" in cluster_map
    assert cluster_map["tid000"] == "L1_B"
    assert coverage == 1.0
    assert "extinf_fallback" in method


def test_parse_v4_cluster_map_empty_dir(tmp_path):
    v4_dir = tmp_path / "V4_5_empty"
    v4_dir.mkdir()
    inv = _make_inventory(2)
    cluster_map, coverage, method = parse_v4_cluster_map(v4_dir, inv)
    assert cluster_map == {}
    assert coverage == 0.0
    assert method == "no_m3u_files"


# ---------------------------------------------------------------------------
# Genre cluster map
# ---------------------------------------------------------------------------


def test_genre_cluster_map_groups():
    inv = _make_inventory(6)
    cmap = genre_cluster_map(inv)
    assert len(cmap) == 6
    # tid000=techno, tid001=house, tid002=techno, ...
    assert cmap["tid000"] == "techno"
    assert cmap["tid001"] == "house"
    assert len(set(cmap.values())) >= 2


def test_genre_cluster_map_unknown_fallback():
    inv = _make_inventory(2)
    inv.loc[0, "genre"] = None
    inv.loc[1, "genre"] = ""
    cmap = genre_cluster_map(inv)
    assert cmap["tid000"] == "unknown"
    assert cmap["tid001"] == "unknown"


# ---------------------------------------------------------------------------
# NN triplet sampling
# ---------------------------------------------------------------------------


def test_sample_nn_triplets_count():
    inv = _make_inventory(10)
    track_ids = list(inv["track_id"])
    pair_df = _make_pair_df(track_ids)
    knn = build_knn_index(pair_df, track_ids)
    triplets = sample_nn_triplets(knn, inv, n=5, seed=42)
    assert len(triplets) == 5


def test_sample_nn_triplets_no_self_reference():
    inv = _make_inventory(8)
    track_ids = list(inv["track_id"])
    pair_df = _make_pair_df(track_ids)
    knn = build_knn_index(pair_df, track_ids)
    triplets = sample_nn_triplets(knn, inv, n=10, seed=42)
    for t in triplets:
        a = t["anchor_track_id"]
        b = t["candidate_b_track_id"]
        c = t["candidate_c_track_id"]
        assert a != b
        assert a != c
        assert b != c


def test_sample_nn_triplets_determinism():
    inv = _make_inventory(8)
    track_ids = list(inv["track_id"])
    pair_df = _make_pair_df(track_ids)
    knn = build_knn_index(pair_df, track_ids)
    t1 = sample_nn_triplets(knn, inv, n=5, seed=42)
    t2 = sample_nn_triplets(knn, inv, n=5, seed=42)
    assert t1 == t2


def test_sample_nn_triplets_different_seeds_differ():
    inv = _make_inventory(10)
    track_ids = list(inv["track_id"])
    pair_df = _make_pair_df(track_ids)
    knn = build_knn_index(pair_df, track_ids)
    t1 = sample_nn_triplets(knn, inv, n=5, seed=42)
    t2 = sample_nn_triplets(knn, inv, n=5, seed=99)
    anchors1 = [t["anchor_track_id"] for t in t1]
    anchors2 = [t["anchor_track_id"] for t in t2]
    assert anchors1 != anchors2


def test_sample_nn_triplets_selection_source():
    inv = _make_inventory(6)
    track_ids = list(inv["track_id"])
    pair_df = _make_pair_df(track_ids)
    knn = build_knn_index(pair_df, track_ids)
    triplets = sample_nn_triplets(knn, inv, n=3, seed=42)
    for t in triplets:
        assert t["selection_source"] == "mert_full_knn"


def test_sample_nn_triplets_no_duplicates():
    inv = _make_inventory(8)
    track_ids = list(inv["track_id"])
    pair_df = _make_pair_df(track_ids)
    knn = build_knn_index(pair_df, track_ids)
    triplets = sample_nn_triplets(knn, inv, n=10, seed=42)
    keys = [(t["anchor_track_id"], frozenset([t["candidate_b_track_id"], t["candidate_c_track_id"]])) for t in triplets]
    assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# Boundary triplet sampling
# ---------------------------------------------------------------------------


def _make_cluster_map(inventory_df: pd.DataFrame, assignment: dict[str, str]) -> dict[str, str]:
    """Map track_ids to cluster labels by index."""
    ids = list(inventory_df["track_id"])
    return {ids[i]: v for i, v in assignment.items()}


def test_sample_boundary_triplets_cross_cluster():
    inv = _make_inventory(8)
    track_ids = list(inv["track_id"])
    pair_df = _make_pair_df(track_ids, seed=7)
    knn = build_knn_index(pair_df, track_ids)
    # Assign alternating clusters
    cmap = {tid: ("A" if i % 2 == 0 else "B") for i, tid in enumerate(sorted(track_ids))}
    triplets = sample_boundary_triplets(knn, cmap, inv, n=4, seed=42)
    for t in triplets:
        a = t["anchor_track_id"]
        b = t["candidate_b_track_id"]
        c = t["candidate_c_track_id"]
        assert cmap.get(b) == cmap.get(a), "B should be same cluster as anchor"
        assert cmap.get(c) != cmap.get(a), "C should be different cluster from anchor"


def test_sample_boundary_triplets_anchor_eligibility():
    """Only tracks with same-cluster AND cross-cluster neighbors are eligible."""
    inv = _make_inventory(6)
    track_ids = list(inv["track_id"])
    pair_df = _make_pair_df(track_ids, seed=3)
    knn = build_knn_index(pair_df, track_ids)
    # Put all tracks in same cluster → no eligible anchors
    cmap = {tid: "A" for tid in track_ids}
    triplets = sample_boundary_triplets(knn, cmap, inv, n=5, seed=42)
    assert len(triplets) == 0


def test_sample_boundary_triplets_determinism():
    inv = _make_inventory(8)
    track_ids = list(inv["track_id"])
    pair_df = _make_pair_df(track_ids)
    knn = build_knn_index(pair_df, track_ids)
    cmap = {tid: ("A" if i % 2 == 0 else "B") for i, tid in enumerate(sorted(track_ids))}
    t1 = sample_boundary_triplets(knn, cmap, inv, n=4, seed=42)
    t2 = sample_boundary_triplets(knn, cmap, inv, n=4, seed=42)
    assert t1 == t2


def test_sample_boundary_triplets_selection_source():
    inv = _make_inventory(6)
    track_ids = list(inv["track_id"])
    pair_df = _make_pair_df(track_ids, seed=5)
    knn = build_knn_index(pair_df, track_ids)
    cmap = {tid: ("X" if i % 2 == 0 else "Y") for i, tid in enumerate(sorted(track_ids))}
    triplets = sample_boundary_triplets(knn, cmap, inv, n=2, seed=42, selection_source="genre_boundary")
    for t in triplets:
        assert t["selection_source"] == "genre_boundary"


# ---------------------------------------------------------------------------
# De-duplication
# ---------------------------------------------------------------------------


def test_deduplicate_triplets_removes_unordered_dups():
    inv = _make_inventory(4)
    dups = [
        {"anchor_track_id": "tid000", "candidate_b_track_id": "tid001", "candidate_c_track_id": "tid002", "selection_source": "mert_full_knn"},
        {"anchor_track_id": "tid000", "candidate_b_track_id": "tid002", "candidate_c_track_id": "tid001", "selection_source": "mert_full_knn"},  # same unordered pair
    ]
    result = deduplicate_triplets(dups, inv)
    assert len(result) == 1


def test_deduplicate_triplets_hash_collision():
    inv = _make_inventory(3)
    # Give tid000 and tid001 the same hash → collision
    inv.loc[inv["track_id"] == "tid001", "audio_content_hash"] = "hash000"
    triplets = [
        {"anchor_track_id": "tid000", "candidate_b_track_id": "tid001", "candidate_c_track_id": "tid002", "selection_source": "mert_full_knn"},
    ]
    result = deduplicate_triplets(triplets, inv)
    assert len(result) == 0


def test_deduplicate_triplets_preserves_distinct():
    inv = _make_inventory(5)
    triplets = [
        {"anchor_track_id": "tid000", "candidate_b_track_id": "tid001", "candidate_c_track_id": "tid002", "selection_source": "s"},
        {"anchor_track_id": "tid001", "candidate_b_track_id": "tid002", "candidate_c_track_id": "tid003", "selection_source": "s"},
    ]
    result = deduplicate_triplets(triplets, inv)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Question DataFrame assembly
# ---------------------------------------------------------------------------


def test_assemble_question_df_schema():
    inv = _make_inventory(4)
    triplets = [
        {"anchor_track_id": "tid000", "candidate_b_track_id": "tid001", "candidate_c_track_id": "tid002", "selection_source": "mert_full_knn"},
        {"anchor_track_id": "tid001", "candidate_b_track_id": "tid002", "candidate_c_track_id": "tid003", "selection_source": "v4_boundary"},
    ]
    df = assemble_question_df(triplets, inv)
    for col in QUESTION_COLUMNS:
        assert col in df.columns, f"missing column: {col}"
    assert len(df) == 2


def test_question_id_format():
    inv = _make_inventory(5)
    triplets = [
        {"anchor_track_id": "tid000", "candidate_b_track_id": "tid001", "candidate_c_track_id": "tid002", "selection_source": "s"},
        {"anchor_track_id": "tid001", "candidate_b_track_id": "tid002", "candidate_c_track_id": "tid003", "selection_source": "s"},
        {"anchor_track_id": "tid002", "candidate_b_track_id": "tid003", "candidate_c_track_id": "tid004", "selection_source": "s"},
    ]
    df = assemble_question_df(triplets, inv)
    assert list(df["question_id"]) == ["Q001", "Q002", "Q003"]


def test_assemble_question_df_empty():
    inv = _make_inventory(2)
    df = assemble_question_df([], inv)
    assert len(df) == 0
    for col in QUESTION_COLUMNS:
        assert col in df.columns
