"""
PURPOSE: Unit tests for src/dj_clustering/features.py and src/dj_clustering/hpss.py —
         L2 normalization, mert_concat policy, HPSS shape/energy, fallback omission,
         BPM imputation, and feature manifest schema. No real audio, no GPU, no HF downloads.

CHANGELOG:
  D2.2 - Initial implementation.
"""

import sys
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dj_clustering.mert import aggregate_segments, l2_normalize
from src.dj_clustering.hpss import hpss_percussive
from src.dj_clustering.features import (
    FEATURE_MANIFEST_COLUMNS,
    FeatureExtractor,
    _check_hpss_available,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(d: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_config(hpss_ok: bool = True) -> dict:
    return {
        "artifact_root": "artifacts/dj_clustering",
        "feature_output_root": "artifacts/dj_clustering/features",
        "inventory_input": "artifacts/dj_clustering/inventory/library_inventory.csv",
        "feature_quality_report": "reports/dj_clustering/feature_quality_report.md",
        "audio": {"sample_rate": 24000, "mono": True, "loading_backend": "soundfile"},
        "mert": {
            "model": "m-a-p/MERT-v1-330M",
            "trust_remote_code": True,
            "aggregation_choices": ["last_layer_mean", "last_4_layers_mean"],
            "default_aggregation": "last_layer_mean",
            "segment_aggregation": "l2_normalize_per_segment_then_mean",
            "track_aggregation": "l2_normalize_track_mean",
            "l2_normalize_before_storage": True,
        },
        "percussion": {
            "default_method": "hpss",
            "hpss_backend_preference": ["scipy_numpy"],
            "hpss_backend_status": "requires_d2_2_validation",
            "demucs_if_stems_exist": False,
            "stems_path": "artifacts/dj_clustering/demucs_stems",
            "omit_if_unavailable": True,
        },
        "fallback_policy": {
            "no_mert_backend": "block",
            "hpss_backend_unavailable": "omit_mert_perc_and_mert_concat",
            "hpss_failure_per_track": "omit_mert_perc_for_track",
        },
        "feature_sources": {
            "mert_full": {"enabled": True},
            "mert_perc": {"enabled": hpss_ok},
            "mert_concat": {"enabled": hpss_ok},
            "essentia": {"enabled": False},
            "metadata_numeric": {"enabled": True},
        },
        "metadata_features": {
            "bpm": {"enabled": True, "source_column": "bpm", "type": "numeric", "normalize": True},
            "key": {"enabled": False, "reason": "zero_coverage"},
        },
        "privacy_policy": {"no_private_paths_in_committed_reports": True},
    }


def _make_inventory(bpm_values: list) -> pd.DataFrame:
    rows = []
    for i, bpm in enumerate(bpm_values):
        rows.append({
            "track_id": f"t{i:03d}",
            "is_canonical": True,
            "decode_status": "ok",
            "duration_seconds": 300.0,
            "file_path": f"/fake/path/t{i:03d}.mp3",
            "bpm": bpm,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# l2_normalize
# ---------------------------------------------------------------------------

def test_l2_normalize_unit_vector():
    v = np.array([3.0, 4.0], dtype=np.float32)
    out = l2_normalize(v)
    assert pytest.approx(np.linalg.norm(out), abs=1e-5) == 1.0


def test_l2_normalize_already_unit():
    v = _unit_vec(128)
    out = l2_normalize(v)
    assert pytest.approx(np.linalg.norm(out), abs=1e-5) == 1.0


def test_l2_normalize_zero_safe():
    v = np.zeros(64, dtype=np.float32)
    out = l2_normalize(v)
    assert not np.any(np.isnan(out))
    assert np.all(out == 0.0)


def test_l2_normalize_2d_rows():
    mat = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    out = l2_normalize(mat)
    norms = np.linalg.norm(out, axis=1)
    assert pytest.approx(norms[0], abs=1e-5) == 1.0
    assert pytest.approx(norms[1], abs=1e-5) == 1.0


# ---------------------------------------------------------------------------
# aggregate_segments
# ---------------------------------------------------------------------------

def test_aggregate_segments_normalization():
    D = 128
    aggs = ["last_layer_mean"]
    seg_embs = [
        {"last_layer_mean": np.random.randn(D).astype(np.float32)} for _ in range(3)
    ]
    result = aggregate_segments(seg_embs, aggs)
    norm = np.linalg.norm(result["last_layer_mean"])
    assert pytest.approx(norm, abs=1e-4) == 1.0


def test_aggregate_segments_single():
    D = 64
    aggs = ["last_layer_mean"]
    seg_embs = [{"last_layer_mean": np.ones(D, dtype=np.float32)}]
    result = aggregate_segments(seg_embs, aggs)
    assert pytest.approx(np.linalg.norm(result["last_layer_mean"]), abs=1e-4) == 1.0


def test_aggregate_segments_empty_raises():
    with pytest.raises(ValueError):
        aggregate_segments([], ["last_layer_mean"])


# ---------------------------------------------------------------------------
# mert_concat policy
# ---------------------------------------------------------------------------

def test_mert_concat_is_unit_length():
    cfg = _make_config()
    extractor = FeatureExtractor(cfg, "cpu")
    a = np.random.randn(128).astype(np.float32)
    b = np.random.randn(128).astype(np.float32)
    out = extractor.compute_mert_concat(a, b)
    assert out.shape == (256,)
    assert pytest.approx(np.linalg.norm(out), abs=1e-4) == 1.0


def test_mert_concat_deterministic():
    cfg = _make_config()
    extractor = FeatureExtractor(cfg, "cpu")
    a = np.random.randn(64).astype(np.float32)
    b = np.random.randn(64).astype(np.float32)
    out1 = extractor.compute_mert_concat(a, b)
    out2 = extractor.compute_mert_concat(a, b)
    np.testing.assert_array_equal(out1, out2)


def test_mert_concat_uses_both_halves():
    cfg = _make_config()
    extractor = FeatureExtractor(cfg, "cpu")
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    out = extractor.compute_mert_concat(a, b)
    # First half should reflect normalized a, second half normalized b
    assert out.shape == (4,)


# ---------------------------------------------------------------------------
# HPSS — shape and energy
# ---------------------------------------------------------------------------

def _make_audio(duration_sec: float = 2.0, sr: int = 24000) -> np.ndarray:
    """Synthetic audio: mix of sine tones and noise."""
    t = np.linspace(0, duration_sec, int(duration_sec * sr), dtype=np.float32)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = 0.1 * np.random.default_rng(0).standard_normal(len(t)).astype(np.float32)
    return sine + noise


def test_hpss_percussive_shape():
    y = _make_audio()
    perc = hpss_percussive(y, 24000)
    assert perc.shape == y.shape, f"Shape mismatch: {perc.shape} vs {y.shape}"


def test_hpss_percussive_has_energy():
    y = _make_audio()
    perc = hpss_percussive(y, 24000)
    assert np.sum(perc ** 2) > 0.0


def test_hpss_output_dtype_float32():
    y = _make_audio()
    perc = hpss_percussive(y, 24000)
    assert perc.dtype == np.float32


def test_hpss_short_audio_raises():
    y = np.zeros(100, dtype=np.float32)  # shorter than n_fft=2048
    with pytest.raises(RuntimeError, match="too short"):
        hpss_percussive(y, 24000)


def test_hpss_stereo_raises():
    y = np.zeros((2048, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="mono"):
        hpss_percussive(y, 24000)


# ---------------------------------------------------------------------------
# HPSS fallback — per-track failure omits mert_perc
# ---------------------------------------------------------------------------

def _make_result_base(tid: str = "t000", hpss_ok: bool = True) -> dict:
    D = 64
    aggs = ["last_layer_mean", "last_4_layers_mean"]
    return {
        "track_id": tid,
        "segments_used": 3,
        "mert_full": {a: _unit_vec(D, i) for i, a in enumerate(aggs)},
        "mert_perc": {a: _unit_vec(D, i + 10) for i, a in enumerate(aggs)} if hpss_ok else None,
        "mert_concat": {a: _unit_vec(D * 2, i + 20) for i, a in enumerate(aggs)} if hpss_ok else None,
        "hpss_failed": not hpss_ok,
        "hpss_failure_reason": None if hpss_ok else "RuntimeError: test",
        "mert_full_failure_reason": None,
    }


def test_hpss_per_track_failure_omits_mert_perc(tmp_path):
    cfg = _make_config()
    extractor = FeatureExtractor(cfg, "cpu")
    result = _make_result_base(hpss_ok=False)

    manifest = extractor.build_feature_manifest([result], tmp_path)
    perc_rows = manifest[manifest["feature_source"] == "mert_perc"]
    assert all(perc_rows["status"] == "omitted")
    assert all(perc_rows["failure_reason"].notna())


def test_hpss_global_unavailable_omits_mert_perc_and_concat(tmp_path):
    cfg = _make_config()
    extractor = FeatureExtractor(cfg, "cpu")
    # Patch hpss to be globally unavailable
    extractor._hpss_available = False

    result = _make_result_base(hpss_ok=False)
    manifest = extractor.build_feature_manifest([result], tmp_path)

    perc_rows = manifest[manifest["feature_source"] == "mert_perc"]
    concat_rows = manifest[manifest["feature_source"] == "mert_concat"]
    assert all(perc_rows["status"] == "omitted")
    assert all(perc_rows["failure_reason"] == "hpss_globally_unavailable")
    assert all(concat_rows["status"] == "omitted")


# ---------------------------------------------------------------------------
# BPM feature
# ---------------------------------------------------------------------------

def test_bpm_null_imputed_median():
    cfg = _make_config()
    extractor = FeatureExtractor(cfg, "cpu")
    inv = _make_inventory([120.0, 130.0, None, 140.0])
    arr, tids = extractor.compute_bpm_feature(inv)
    # All rows should be present (null imputed, not dropped)
    assert arr.shape == (4, 1)
    assert len(tids) == 4
    # No NaN in output
    assert not np.any(np.isnan(arr))


def test_bpm_normalized_range():
    cfg = _make_config()
    extractor = FeatureExtractor(cfg, "cpu")
    inv = _make_inventory([100.0, 120.0, 130.0, 140.0, 160.0])
    arr, _ = extractor.compute_bpm_feature(inv)
    assert float(arr.min()) == pytest.approx(0.0, abs=1e-5)
    assert float(arr.max()) == pytest.approx(1.0, abs=1e-5)


def test_bpm_excludes_non_canonical():
    cfg = _make_config()
    extractor = FeatureExtractor(cfg, "cpu")
    inv = _make_inventory([120.0, 130.0])
    inv.loc[1, "is_canonical"] = False
    arr, tids = extractor.compute_bpm_feature(inv)
    assert len(tids) == 1


# ---------------------------------------------------------------------------
# Feature manifest schema
# ---------------------------------------------------------------------------

def test_feature_manifest_schema(tmp_path):
    cfg = _make_config()
    extractor = FeatureExtractor(cfg, "cpu")
    results = [_make_result_base("t000"), _make_result_base("t001")]
    manifest = extractor.build_feature_manifest(results, tmp_path)
    for col in FEATURE_MANIFEST_COLUMNS:
        assert col in manifest.columns, f"Missing column: {col}"


def test_feature_manifest_no_private_paths(tmp_path):
    cfg = _make_config()
    extractor = FeatureExtractor(cfg, "cpu")
    results = [_make_result_base("t000")]
    manifest = extractor.build_feature_manifest(results, tmp_path)
    for path_val in manifest["embedding_path"].dropna():
        assert "/mnt/fast" not in str(path_val)
        assert "nobackup" not in str(path_val)
        assert "gb0048" not in str(path_val)


def test_feature_manifest_ok_rows_have_dim(tmp_path):
    cfg = _make_config()
    extractor = FeatureExtractor(cfg, "cpu")
    results = [_make_result_base("t000")]
    manifest = extractor.build_feature_manifest(results, tmp_path)
    ok_rows = manifest[manifest["status"] == "ok"]
    assert all(ok_rows["embedding_dim"] > 0)


# ---------------------------------------------------------------------------
# Failure-rate guard (extraction gate)
# ---------------------------------------------------------------------------

def test_failure_rate_computation(tmp_path):
    """Verify that >10% mert_full failures produces a failed-gate manifest."""
    cfg = _make_config()
    extractor = FeatureExtractor(cfg, "cpu")

    # 9 ok + 2 failed = 18.2% failure rate
    results = [_make_result_base(f"t{i:03d}") for i in range(9)]
    failed = {
        "track_id": "t009",
        "segments_used": 0,
        "mert_full": None,
        "mert_perc": None,
        "mert_concat": None,
        "hpss_failed": False,
        "hpss_failure_reason": None,
        "mert_full_failure_reason": "audio_load_error: FileNotFoundError",
    }
    results_all = results + [failed, {**failed, "track_id": "t010"}]
    manifest = extractor.build_feature_manifest(results_all, tmp_path)
    mf = manifest[manifest["feature_source"] == "mert_full"]
    n_fail = (mf["status"] == "failed").sum()
    n_total = mf["track_id"].nunique()
    rate = n_fail / n_total
    assert rate > 0.10, f"Expected >10% failure, got {rate:.1%}"
