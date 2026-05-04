"""
PURPOSE: Unit tests for src/dj_clustering/pair_features.py — pair enumeration
         determinism, pair_id uniqueness, cosine math (similarity / distance /
         cosine_embedding_01), raw-BPM tempo equivalence on the approved
         (120,60) (128,64) (128,130) (120,80) cases, BPM missingness, refusal
         to use normalized BPM for tempo equivalence, metadata missingness,
         failed-MERT preservation with availability flags, and schema layout.
         Pure-Python; no audio, no GPU, no HF downloads.

CHANGELOG:
  D3.1 - Initial implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dj_clustering.pair_features import (
    MERT_AGGREGATIONS,
    MERT_SOURCES,
    align_embedding_matrix,
    assemble_pair_dataframe,
    compute_bpm_pair_features,
    compute_cosine_pair_features,
    compute_metadata_pair_features,
    enumerate_canonical_pairs,
)


# ---------------------------------------------------------------------------
# Enumeration / determinism
# ---------------------------------------------------------------------------

def test_enumerate_pair_count_and_ordering():
    ids = ["t05", "t01", "t03", "t04", "t02"]
    pair_ids, idx_a, idx_b = enumerate_canonical_pairs(ids)
    assert len(pair_ids) == 10
    sorted_ids = sorted(set(ids))
    # Every pair_id must use a < b ordering on sorted IDs.
    for pid, ia, ib in zip(pair_ids, idx_a.tolist(), idx_b.tolist()):
        a, b = pid.split("__")
        assert a == sorted_ids[ia]
        assert b == sorted_ids[ib]
        assert a < b


def test_enumerate_is_deterministic_on_unsorted_input():
    a = ["zzz", "aaa", "mmm", "bbb"]
    b = list(reversed(a))
    pair_ids_a, _, _ = enumerate_canonical_pairs(a)
    pair_ids_b, _, _ = enumerate_canonical_pairs(b)
    assert pair_ids_a == pair_ids_b


def test_enumerate_pair_id_uniqueness():
    ids = [f"t{i:02d}" for i in range(50)]
    pair_ids, _, _ = enumerate_canonical_pairs(ids)
    assert len(pair_ids) == 50 * 49 // 2
    assert len(set(pair_ids)) == len(pair_ids)


def test_enumerate_rejects_duplicates():
    with pytest.raises(ValueError):
        enumerate_canonical_pairs(["a", "b", "a"])


# ---------------------------------------------------------------------------
# Cosine math
# ---------------------------------------------------------------------------

def _unit(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d).astype(np.float32)
    return v / np.linalg.norm(v)


def test_cosine_identical_and_antipodal():
    ids = ["a", "b", "c"]
    v = _unit(8, seed=1)
    embeddings = np.stack([v, v, -v]).astype(np.float32)
    aligned, avail = align_embedding_matrix(embeddings, ids, ids)
    pair_ids, idx_a, idx_b = enumerate_canonical_pairs(ids)
    block = compute_cosine_pair_features(aligned, avail, idx_a, idx_b)
    sim = block["cosine_similarity"]
    cos01 = block["cosine_embedding_01"]
    dist = block["cosine_distance"]
    # pair (a,b) identical
    pair_index = pair_ids.index("a__b")
    assert np.isclose(sim[pair_index], 1.0, atol=1e-6)
    assert np.isclose(cos01[pair_index], 1.0, atol=1e-6)
    assert np.isclose(dist[pair_index], 0.0, atol=1e-6)
    # pair (a,c) antipodal
    pair_index = pair_ids.index("a__c")
    assert np.isclose(sim[pair_index], -1.0, atol=1e-6)
    assert np.isclose(cos01[pair_index], 0.0, atol=1e-6)
    assert np.isclose(dist[pair_index], 2.0, atol=1e-6)


def test_cosine_embedding_01_in_unit_interval():
    ids = [f"t{i:02d}" for i in range(8)]
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((len(ids), 16)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    raw = raw / norms
    aligned, avail = align_embedding_matrix(raw, ids, ids)
    _, idx_a, idx_b = enumerate_canonical_pairs(ids)
    block = compute_cosine_pair_features(aligned, avail, idx_a, idx_b)
    assert np.all((block["cosine_embedding_01"] >= 0.0) & (block["cosine_embedding_01"] <= 1.0))


def test_cosine_failed_mert_track_preserved_with_availability():
    """A track absent from the MERT source must yield available=False on its
    pairs but stay in the universe."""
    ids = ["a", "b", "c", "d"]
    src_ids = ["a", "c", "d"]  # 'b' missing for MERT
    embeddings = np.stack([_unit(8, s) for s in [11, 12, 13]]).astype(np.float32)
    aligned, avail = align_embedding_matrix(embeddings, src_ids, ids)
    assert avail.tolist() == [True, False, True, True]
    pair_ids, idx_a, idx_b = enumerate_canonical_pairs(ids)
    block = compute_cosine_pair_features(aligned, avail, idx_a, idx_b)
    expected_unavailable = {pid for pid in pair_ids if "b" in pid.split("__")}
    for pid, ok in zip(pair_ids, block["available"].tolist()):
        if pid in expected_unavailable:
            assert ok is False
            i = pair_ids.index(pid)
            assert np.isnan(block["cosine_similarity"][i])
            assert np.isnan(block["cosine_embedding_01"][i])
            assert np.isnan(block["cosine_distance"][i])
        else:
            assert ok is True
    assert int(block["available"].sum()) == 3  # C(3,2) over the 3 available tracks
    assert len(pair_ids) == 6  # universe still has 4 tracks


# ---------------------------------------------------------------------------
# BPM tempo equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "a,b,expected_eff",
    [
        (120.0, 60.0, 0.0),    # exact 2x equivalence
        (128.0, 64.0, 0.0),    # exact 2x equivalence
        (128.0, 130.0, 2.0),   # close in 1x branch
        # (120, 80): the approved formula yields min(40, 40, 80, 160, 20) = 20,
        # because 0.5 * 120 = 60 is only 20 away from 80. The plan summary
        # called out 40 as a "no equivalence" expectation, but the formula
        # actually delivers 20 here. The test asserts the formula's true output.
        (120.0, 80.0, 20.0),
    ],
)
def test_bpm_tempo_equivalence_raw_bpm(a, b, expected_eff):
    ids = ["x", "y"]
    bpm_lookup = {"x": a, "y": b}
    _, idx_a, idx_b = enumerate_canonical_pairs(ids)
    block = compute_bpm_pair_features(bpm_lookup, ids, idx_a, idx_b, tolerance_bpm=8.0)
    assert block["bpm_available"].tolist() == [True]
    assert np.isclose(block["bpm_eff_abs_diff"][0], expected_eff)
    sim = block["bpm_similarity"][0]
    assert 0.0 <= sim <= 1.0


def test_bpm_similarity_clamped_in_unit_interval():
    ids = ["x", "y"]
    bpm_lookup = {"x": 120.0, "y": 200.0}  # very different
    _, idx_a, idx_b = enumerate_canonical_pairs(ids)
    block = compute_bpm_pair_features(bpm_lookup, ids, idx_a, idx_b, tolerance_bpm=8.0)
    assert block["bpm_similarity"][0] == 0.0


def test_bpm_missing_yields_unavailable_and_nan():
    ids = ["x", "y", "z"]
    # 'z' has no BPM
    bpm_lookup = {"x": 120.0, "y": 124.0}
    pair_ids, idx_a, idx_b = enumerate_canonical_pairs(ids)
    block = compute_bpm_pair_features(bpm_lookup, ids, idx_a, idx_b, tolerance_bpm=8.0)
    for pid, ok in zip(pair_ids, block["bpm_available"].tolist()):
        if "z" in pid.split("__"):
            assert ok is False
            i = pair_ids.index(pid)
            assert np.isnan(block["bpm_similarity"][i])
            assert np.isnan(block["bpm_eff_abs_diff"][i])
            assert np.isnan(block["bpm_abs_diff"][i])
            assert np.isnan(block["bpm_raw_a"][i])
            assert np.isnan(block["bpm_raw_b"][i])
        else:
            assert ok is True


def test_normalized_bpm_must_not_be_used_for_tempo_equivalence():
    """If we substituted normalized BPM (different scale, e.g. z-scores), the
    expected tempo-equivalence outcomes would not hold. This test confirms the
    raw-BPM contract by checking that swapping to a normalized scale yields a
    different bpm_eff_abs_diff for the (120, 60) case."""
    ids = ["x", "y"]
    raw_lookup = {"x": 120.0, "y": 60.0}
    norm_lookup = {"x": 1.0, "y": -1.0}  # arbitrary normalized scale
    _, idx_a, idx_b = enumerate_canonical_pairs(ids)
    raw_block = compute_bpm_pair_features(raw_lookup, ids, idx_a, idx_b, tolerance_bpm=8.0)
    norm_block = compute_bpm_pair_features(norm_lookup, ids, idx_a, idx_b, tolerance_bpm=8.0)
    assert np.isclose(raw_block["bpm_eff_abs_diff"][0], 0.0)
    # If normalized BPM were (incorrectly) used, eff diff would be 0.5 (= |1 - 2*(-1)| etc.
    # -> min(|1-(-1)|, |1-(-2)|, |1-(-0.5)|, |2-(-1)|, |0.5-(-1)|) = 1.5).
    assert not np.isclose(norm_block["bpm_eff_abs_diff"][0], 0.0)


def test_bpm_tolerance_must_be_positive():
    with pytest.raises(ValueError):
        compute_bpm_pair_features(
            {"a": 120.0, "b": 124.0},
            ["a", "b"],
            np.array([0]),
            np.array([1]),
            tolerance_bpm=0.0,
        )


# ---------------------------------------------------------------------------
# Metadata missingness
# ---------------------------------------------------------------------------

def test_metadata_similarity_partial_availability():
    ids = ["a", "b", "c"]
    inv = pd.DataFrame(
        {
            "track_id": ids,
            "genre": ["techno", "techno", None],
            "artist": ["x", "y", "x"],
            "label": [None, None, None],
        }
    )
    pair_ids, idx_a, idx_b = enumerate_canonical_pairs(ids)
    fields = ["genre", "artist", "label"]
    block = compute_metadata_pair_features(inv, fields, ids, idx_a, idx_b)
    # (a, b): genre matches (both techno), artist differs, label missing both -> 1/2
    i = pair_ids.index("a__b")
    assert np.isclose(block["metadata_similarity"][i], 0.5)
    assert block["metadata_similarity_available"][i]
    # (a, c): genre missing on c, artist matches, label missing both -> 1/1 = 1.0
    i = pair_ids.index("a__c")
    assert np.isclose(block["metadata_similarity"][i], 1.0)
    # (b, c): genre missing on c, artist differs (y vs x), label missing -> 0/1 = 0.0
    i = pair_ids.index("b__c")
    assert np.isclose(block["metadata_similarity"][i], 0.0)
    assert block["metadata_similarity_available"][i]


def test_metadata_no_fields_available_yields_zero_and_unavailable():
    ids = ["a", "b"]
    inv = pd.DataFrame(
        {"track_id": ids, "genre": [None, None], "artist": [None, None], "label": [None, None]}
    )
    _, idx_a, idx_b = enumerate_canonical_pairs(ids)
    block = compute_metadata_pair_features(inv, ["genre", "artist", "label"], ids, idx_a, idx_b)
    assert block["metadata_similarity"][0] == 0.0
    assert bool(block["metadata_similarity_available"][0]) is False


# ---------------------------------------------------------------------------
# DataFrame schema
# ---------------------------------------------------------------------------

def test_assemble_pair_dataframe_schema():
    ids = ["a", "b", "c"]
    pair_ids, idx_a, idx_b = enumerate_canonical_pairs(ids)
    track_ids_a = [ids[i] for i in idx_a.tolist()]
    track_ids_b = [ids[i] for i in idx_b.tolist()]

    # Build minimal cosine block: only mert_full / last_layer_mean.
    v = _unit(4, 7)
    emb = np.stack([v, v, -v]).astype(np.float32)
    aligned, avail = align_embedding_matrix(emb, ids, ids)
    cosine_block = compute_cosine_pair_features(aligned, avail, idx_a, idx_b)
    cosine_blocks = {("mert_full", "last_layer_mean"): cosine_block}

    bpm_lookup = {"a": 120.0, "b": 121.0, "c": 122.0}
    bpm_block = compute_bpm_pair_features(bpm_lookup, ids, idx_a, idx_b, tolerance_bpm=8.0)
    inv = pd.DataFrame(
        {"track_id": ids, "genre": ["t", "t", "h"], "artist": ["x", "y", "z"], "label": [None] * 3}
    )
    metadata_block = compute_metadata_pair_features(
        inv, ["genre", "artist", "label"], ids, idx_a, idx_b
    )

    df = assemble_pair_dataframe(
        pair_ids=pair_ids,
        track_ids_a=track_ids_a,
        track_ids_b=track_ids_b,
        cosine_blocks=cosine_blocks,
        bpm_block=bpm_block,
        metadata_block=metadata_block,
    )
    expected = {
        "pair_id",
        "pair_index",
        "track_id_a",
        "track_id_b",
        "mert_full_last_layer_mean__cosine_similarity",
        "mert_full_last_layer_mean__cosine_distance",
        "mert_full_last_layer_mean__cosine_embedding_01",
        "mert_full_last_layer_mean__available",
        "bpm_raw_a",
        "bpm_raw_b",
        "bpm_abs_diff",
        "bpm_eff_abs_diff",
        "bpm_similarity",
        "bpm_available",
        "metadata_similarity",
        "metadata_similarity_available",
        "metadata_genre_available",
        "metadata_artist_available",
        "metadata_label_available",
    }
    assert expected.issubset(set(df.columns))
    assert df["pair_id"].is_unique
    assert (df["track_id_a"] < df["track_id_b"]).all()
    assert df["pair_index"].tolist() == list(range(len(df)))
