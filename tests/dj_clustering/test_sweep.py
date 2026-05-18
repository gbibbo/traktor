"""
PURPOSE: Unit tests for src/dj_clustering/sweep.py — committed Regime 1 grid
         expansion size and per-clusterer breakdown, invalid-combination
         pruning via is_valid_config, deterministic seed-13 capped sampling,
         fixed-baseline injection from baseline_definition, diagnostic-only
         V2/V4 reference rows, and run-plan assembly. Pure-Python; no audio,
         no GPU, no embedding artifacts.

CHANGELOG:
  D4.1a - Initial implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dj_clustering.sweep import (
    NOISE_NOT_APPLICABLE,
    assemble_run_plan,
    build_baselines,
    build_diagnostic_rows,
    expand_grid,
    is_valid_config,
    load_sweep_config,
    sample_configs,
)

CONFIG_PATH = PROJECT_ROOT / "configs/dj_clustering/sweep_regime1.yaml"

# 3 sources x 3 aggregations x 2 normalizations x 4 dim_reductions = 72 base
# combos; hdbscan branch = 4 x 2 x 3 = 24, kmeans = 4, agglomerative = 4.
EXPECTED_GRID_TOTAL = 72 * (24 + 4 + 4)


@pytest.fixture(scope="module")
def config() -> dict:
    return load_sweep_config(CONFIG_PATH)


def test_committed_config_loads(config):
    assert config["regime"] == 1
    assert config["sampling"]["max_configs"] == 200
    assert config["sampling"]["seed"] == 13


def test_grid_expansion_total(config):
    grid = expand_grid(config)
    assert len(grid) == EXPECTED_GRID_TOTAL == 2304


def test_grid_expansion_per_clusterer(config):
    grid = expand_grid(config)
    counts = {"hdbscan": 0, "kmeans": 0, "agglomerative": 0}
    for cfg in grid:
        counts[cfg["clusterer"]] += 1
    assert counts == {"hdbscan": 72 * 24, "kmeans": 72 * 4, "agglomerative": 72 * 4}


def test_all_expanded_configs_are_valid(config):
    grid = expand_grid(config)
    assert all(is_valid_config(cfg) for cfg in grid)
    assert len({cfg["config_id"] for cfg in grid}) == len(grid)


def test_invalid_combinations_are_pruned():
    # kmeans must not carry HDBSCAN-specific parameters.
    bad_kmeans = {
        "clusterer": "kmeans",
        "kmeans_k": 12,
        "noise_policy": NOISE_NOT_APPLICABLE,
        "hdbscan_min_cluster_size": 6,
        "agglomerative_k": None,
    }
    assert not is_valid_config(bad_kmeans)

    # hdbscan must have a real noise policy.
    bad_hdbscan = {
        "clusterer": "hdbscan",
        "hdbscan_min_cluster_size": 6,
        "hdbscan_min_samples": 1,
        "noise_policy": NOISE_NOT_APPLICABLE,
        "kmeans_k": None,
        "agglomerative_k": None,
    }
    assert not is_valid_config(bad_hdbscan)

    # agglomerative must use average linkage, never ward.
    bad_agglo = {
        "clusterer": "agglomerative",
        "agglomerative_k": 12,
        "agglomerative_linkage": "ward",
        "noise_policy": NOISE_NOT_APPLICABLE,
        "hdbscan_min_cluster_size": None,
        "kmeans_k": None,
    }
    assert not is_valid_config(bad_agglo)


def test_sampling_is_deterministic_with_seed_13(config):
    grid = expand_grid(config)
    first = sample_configs(grid, 200, 13)
    second = sample_configs(grid, 200, 13)
    assert len(first) == 200
    assert [c["config_id"] for c in first] == [c["config_id"] for c in second]
    grid_ids = {c["config_id"] for c in grid}
    assert all(c["config_id"] in grid_ids for c in first)


def test_sampling_returns_input_when_under_cap():
    small = [{"config_id": f"c{i}"} for i in range(5)]
    assert sample_configs(small, 200, 13) == small


def test_fixed_baselines(config):
    baselines = build_baselines(config)
    assert len(baselines) == 3
    sources = {b["embedding_source"] for b in baselines}
    assert sources == {"mert_perc", "mert_full", "mert_concat"}
    for b in baselines:
        assert b["kind"] == "baseline"
        assert b["clusterer"] == "hdbscan"
        assert b["mert_aggregation"] == "last_layer_mean"
        assert b["normalization"] == "l2"
        assert b["dim_reduction"] == "umap_15"
        assert b["hdbscan_min_cluster_size"] == 6
        assert b["hdbscan_min_samples"] == 1
        assert b["noise_policy"] == "no_reassignment"


def test_diagnostic_only_reference_rows(config):
    rows = build_diagnostic_rows(config)
    assert {r["reference_name"] for r in rows} == {"V2", "V4"}
    for r in rows:
        assert r["kind"] == "diagnostic"
        assert r["executable"] is False


def test_assemble_run_plan(config):
    plan = assemble_run_plan(config)
    assert plan["grid_total"] == 2304
    assert plan["n_sampled"] == 200
    assert plan["n_baselines"] == 3
    assert plan["n_diagnostic"] == 2
    assert plan["n_executable"] == 203
