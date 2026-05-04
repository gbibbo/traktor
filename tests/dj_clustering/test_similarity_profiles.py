"""
PURPOSE: Unit tests for src/dj_clustering/similarity_profiles.py — registry
         decisions for committed vs omitted profiles, profile math (weighted
         sum + distance duality), and per-pair availability propagation.

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

from src.dj_clustering.similarity_profiles import (
    ALL_PROFILES,
    attach_committed_profiles,
    build_profile_registry,
    compute_profile_columns,
)


def test_registry_omits_key_profiles_when_key_unavailable():
    avail = {
        "cosine_embedding_01": True,
        "bpm_similarity": True,
        "key_compat": False,
        "metadata_similarity": True,
    }
    registry = build_profile_registry(avail)
    by_name = {e.name: e for e in registry}
    assert by_name["audio_only"].status == "committed"
    assert by_name["audio_plus_metadata_light"].status == "committed"
    assert by_name["audio_plus_bpm_key_light"].status == "omitted"
    assert by_name["audio_plus_bpm_key_metadata_light"].status == "omitted"
    assert "key_compat_unavailable" in by_name["audio_plus_bpm_key_light"].omission_reason


def test_registry_commits_all_profiles_when_components_available():
    avail = {
        "cosine_embedding_01": True,
        "bpm_similarity": True,
        "key_compat": True,
        "metadata_similarity": True,
    }
    registry = build_profile_registry(avail)
    for entry in registry:
        assert entry.status == "committed"
        assert entry.similarity_column == f"profile_{entry.name}_similarity"


def test_audio_only_profile_equals_default_cosine_embedding_01():
    df = pd.DataFrame(
        {
            "default_cos01": [0.0, 0.5, 1.0],
            "default_avail": [True, True, True],
        }
    )
    cols = compute_profile_columns(
        df,
        ALL_PROFILES["audio_only"],
        component_columns={"cosine_embedding_01": "default_cos01"},
        component_available_columns={"cosine_embedding_01": "default_avail"},
    )
    sim = cols["profile_audio_only_similarity"]
    dist = cols["profile_audio_only_distance"]
    assert np.allclose(sim, np.array([0.0, 0.5, 1.0], dtype=np.float32))
    assert np.allclose(dist, 1.0 - sim)


def test_audio_plus_metadata_light_weighted_combination():
    df = pd.DataFrame(
        {
            "default_cos01": [1.0, 0.5, 0.0, 1.0],
            "default_avail": [True, True, True, False],
            "meta_sim": [1.0, 1.0, 0.0, 1.0],
            "meta_avail": [True, False, True, True],
        }
    )
    cols = compute_profile_columns(
        df,
        ALL_PROFILES["audio_plus_metadata_light"],
        component_columns={
            "cosine_embedding_01": "default_cos01",
            "metadata_similarity": "meta_sim",
        },
        component_available_columns={
            "cosine_embedding_01": "default_avail",
            "metadata_similarity": "meta_avail",
        },
    )
    sim = cols["profile_audio_plus_metadata_light_similarity"]
    avail = cols["profile_audio_plus_metadata_light_available"]
    # row 0: 0.85*1 + 0.15*1 = 1.0
    assert np.isclose(sim[0], 1.0)
    assert avail[0]
    # row 1: meta unavailable -> profile unavailable, NaN
    assert not avail[1]
    assert np.isnan(sim[1])
    # row 2: 0.85*0 + 0.15*0 = 0.0
    assert np.isclose(sim[2], 0.0)
    assert avail[2]
    # row 3: cosine unavailable -> NaN
    assert not avail[3]
    assert np.isnan(sim[3])


def test_distance_equals_one_minus_similarity():
    df = pd.DataFrame(
        {
            "default_cos01": [0.1, 0.9],
            "default_avail": [True, True],
            "meta_sim": [0.4, 0.7],
            "meta_avail": [True, True],
        }
    )
    cols = compute_profile_columns(
        df,
        ALL_PROFILES["audio_plus_metadata_light"],
        component_columns={
            "cosine_embedding_01": "default_cos01",
            "metadata_similarity": "meta_sim",
        },
        component_available_columns={
            "cosine_embedding_01": "default_avail",
            "metadata_similarity": "meta_avail",
        },
    )
    sim = cols["profile_audio_plus_metadata_light_similarity"]
    dist = cols["profile_audio_plus_metadata_light_distance"]
    assert np.allclose(dist, 1.0 - sim, atol=1e-6)


def test_attach_committed_profiles_emits_only_committed_columns():
    avail_components = {
        "cosine_embedding_01": True,
        "bpm_similarity": True,
        "key_compat": False,
        "metadata_similarity": True,
    }
    registry = build_profile_registry(avail_components)
    df = pd.DataFrame(
        {
            "default_cos01": [0.5, 0.7],
            "default_avail": [True, True],
            "bpm_sim": [0.8, 0.9],
            "bpm_avail": [True, True],
            "meta_sim": [0.6, 1.0],
            "meta_avail": [True, True],
        }
    )
    df_out = attach_committed_profiles(
        df,
        registry,
        component_columns={
            "cosine_embedding_01": "default_cos01",
            "bpm_similarity": "bpm_sim",
            "metadata_similarity": "meta_sim",
        },
        component_available_columns={
            "cosine_embedding_01": "default_avail",
            "bpm_similarity": "bpm_avail",
            "metadata_similarity": "meta_avail",
        },
    )
    cols = set(df_out.columns)
    # committed
    assert "profile_audio_only_similarity" in cols
    assert "profile_audio_plus_metadata_light_similarity" in cols
    # omitted (must NOT appear)
    assert "profile_audio_plus_bpm_key_light_similarity" not in cols
    assert "profile_audio_plus_bpm_key_metadata_light_similarity" not in cols


def test_profile_weights_sum_to_one():
    for name, spec in ALL_PROFILES.items():
        s = sum(spec.weights.values())
        assert abs(s - 1.0) < 1e-9, f"profile {name} weights sum to {s}"
