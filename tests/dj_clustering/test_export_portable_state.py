"""Unit tests for the cluster-exit portable-state exporter.

PURPOSE
    Exercise the pure, privacy-relevant logic of
    scripts/dj_clustering/export_portable_state.py against a temporary fixture
    tree: sha256 hashing, item presence/missing resolution, raw-audio
    aggregate computation (count/bytes only), the per-file raw-audio manifest
    (filenames stay out of the aggregate), and manifest assembly. No real
    project artifacts are read.

CHANGELOG
    2026-05-28  EXIT.1  Initial version.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "dj_clustering" / "export_portable_state.py"
)
_spec = importlib.util.spec_from_file_location("export_portable_state", _SCRIPT)
eps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eps)


def test_sha256_file_matches_hashlib(tmp_path: Path):
    p = tmp_path / "blob.bin"
    data = b"dj-clustering-exit\x00\x01"
    p.write_bytes(data)
    assert eps.sha256_file(p) == hashlib.sha256(data).hexdigest()


def test_resolve_items_present_and_missing(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    present, missing = eps.resolve_items(tmp_path, ["a", "b", "c"])
    assert present == ["a", "b"]
    assert missing == ["c"]


def test_raw_audio_aggregate_counts_and_bytes_only(tmp_path: Path):
    root = tmp_path / "raw"
    (root / "sub").mkdir(parents=True)
    (root / "one.bin").write_bytes(b"x" * 10)
    (root / "sub" / "two.bin").write_bytes(b"y" * 25)
    agg = eps.raw_audio_aggregate(root)
    assert agg == {"raw_audio_file_count": 2, "raw_audio_total_bytes": 35}
    # Aggregate must not leak any filename.
    assert "one.bin" not in str(agg)
    assert "two.bin" not in str(agg)


def test_raw_audio_manifest_sha256_stable(tmp_path: Path):
    root = tmp_path / "raw"
    root.mkdir()
    (root / "track_b.bin").write_bytes(b"bbb")
    (root / "track_a.bin").write_bytes(b"aaa")
    out = tmp_path / "manifest.txt"
    sha1 = eps.write_raw_audio_manifest(root, out)
    # Deterministic: rewriting yields the same digest.
    sha2 = eps.write_raw_audio_manifest(root, out)
    assert sha1 == sha2 == eps.sha256_file(out)
    # Manifest lists relative names sorted; contains per-file digests.
    text = out.read_text()
    assert "track_a.bin" in text and "track_b.bin" in text
    assert text.index("track_a.bin") < text.index("track_b.bin")


def test_build_manifest_shape():
    archives = [{"name": "crit.tar.zst", "members": ["x"], "sha256": "ab", "size_bytes": 1}]
    raw = {"raw_audio_file_count": 492, "raw_audio_total_bytes": 4415100271}
    m = eps.build_manifest("20260528", archives, raw)
    assert m["stamp"] == "20260528"
    assert m["surrey_access_end"] == "2026-05-31"
    assert m["archives"] == archives
    assert m["raw_audio"]["raw_audio_file_count"] == 492


def test_critical_items_cover_resume_payload():
    # Guards against accidental removal of a critical resume directory.
    assert "artifacts/dj_clustering/features" in eps.CRITICAL_ITEMS
    assert "artifacts/dj_clustering/pairs" in eps.CRITICAL_ITEMS
    assert "artifacts/dj_clustering/triplets" in eps.CRITICAL_ITEMS
    assert "artifacts/dj_clustering/inventory" in eps.CRITICAL_ITEMS
    assert "runs/dj_clustering/first_sweep" in eps.CRITICAL_ITEMS
