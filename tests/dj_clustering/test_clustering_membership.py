"""
PURPOSE: Unit tests for src/dj_clustering/clustering.py — L2 / no-op
         normalization, dimensionality reduction shape behaviour, the KMeans
         and agglomerative clusterer wrappers on tiny well-separated fixtures,
         the ward-linkage rejection, the optional HDBSCAN wrapper (skipped when
         the dependency is absent), and HDBSCAN noise-policy reassignment.
         Pure synthetic fixtures; no audio, no GPU, no private data.

CHANGELOG:
  D4.1a - Initial implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dj_clustering.clustering import (
    HDBSCAN_AVAILABLE,
    apply_noise_policy,
    cluster_agglomerative,
    cluster_hdbscan,
    cluster_kmeans,
    normalize_embeddings,
    reduce_dimensions,
)


def _two_blobs():
    """Two tight, well-separated blobs of 5 points each in 4-D."""
    rng = np.random.RandomState(13)
    blob_a = np.array([0.0, 0.0, 0.0, 0.0]) + rng.normal(0, 0.01, (5, 4))
    blob_b = np.array([10.0, 10.0, 10.0, 10.0]) + rng.normal(0, 0.01, (5, 4))
    return np.vstack([blob_a, blob_b])


def test_normalize_l2_yields_unit_rows():
    matrix = np.array([[3.0, 4.0], [1.0, 0.0]])
    out = normalize_embeddings(matrix, "l2")
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0)


def test_normalize_none_is_unchanged():
    matrix = np.array([[3.0, 4.0], [1.0, 0.0]])
    out = normalize_embeddings(matrix, "none")
    assert np.array_equal(out, matrix)
    assert out is not matrix  # returns a copy


def test_normalize_l2_handles_zero_row():
    matrix = np.array([[0.0, 0.0], [3.0, 4.0]])
    out = normalize_embeddings(matrix, "l2")
    assert np.allclose(out[0], [0.0, 0.0])
    assert np.isclose(np.linalg.norm(out[1]), 1.0)


def test_reduce_dimensions_none_is_identity():
    matrix = _two_blobs()
    out = reduce_dimensions(matrix, "none")
    assert np.array_equal(out, matrix)


def test_reduce_dimensions_pca_clamps_components():
    matrix = _two_blobs()  # 10 samples x 4 features
    out = reduce_dimensions(matrix, "pca_50", seed=13)
    # requested 50 components clamped to min(50, 10, 4) = 4.
    assert out.shape == (10, 4)


def test_kmeans_wrapper_separates_blobs():
    matrix = _two_blobs()
    labels = cluster_kmeans(matrix, k=2, seed=13)
    assert len(set(labels)) == 2
    assert set(labels[:5]) != set(labels[5:])


def test_agglomerative_wrapper_separates_blobs():
    matrix = _two_blobs()
    labels = cluster_agglomerative(matrix, k=2, linkage="average")
    assert len(set(labels)) == 2
    assert set(labels[:5]) != set(labels[5:])


def test_agglomerative_rejects_ward_linkage():
    matrix = _two_blobs()
    with pytest.raises(ValueError):
        cluster_agglomerative(matrix, k=2, linkage="ward")


@pytest.mark.skipif(not HDBSCAN_AVAILABLE, reason="hdbscan not installed")
def test_hdbscan_wrapper_runs_on_blobs():
    matrix = _two_blobs()
    labels = cluster_hdbscan(matrix, min_cluster_size=3, min_samples=1)
    assert labels.shape == (10,)
    # at least one real cluster should be discovered.
    assert any(lbl >= 0 for lbl in labels)


def test_noise_policy_no_reassignment_keeps_noise():
    matrix = np.array([[1.0, 0.0], [1.0, 0.01], [1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 0, -1, -1])
    metric, export = apply_noise_policy(labels, matrix, "no_reassignment")
    assert list(metric) == [0, 0, -1, -1]
    assert list(export) == [0, 0, -1, -1]


def test_noise_policy_confidence_limited_reassigns_only_close_noise():
    # Row 2 (noise) shares direction with cluster 0 -> cosine distance ~0.
    # Row 3 (noise) is orthogonal -> cosine distance 1.0 > 0.30.
    matrix = np.array([[1.0, 0.0], [1.0, 0.01], [1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 0, -1, -1])
    metric, export = apply_noise_policy(labels, matrix, "confidence_limited_1nn")
    assert metric[2] == 0  # close noise reassigned
    assert metric[3] == -1  # far noise kept
    assert list(export) == list(metric)


def test_noise_policy_forced_1nn_export_only():
    matrix = np.array([[1.0, 0.0], [1.0, 0.01], [1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 0, -1, -1])
    metric, export = apply_noise_policy(labels, matrix, "forced_1nn_export_only")
    # metric labels keep native noise for evaluation.
    assert list(metric) == [0, 0, -1, -1]
    # export labels force every noise row into a cluster.
    assert -1 not in set(export)
