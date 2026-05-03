"""
PURPOSE: Unit tests for src/dj_clustering/identity.py — SHA256 hashing, track_id
         generation, duplicate detection, and canonical selection logic.

CHANGELOG:
  D1.2 - Initial implementation.
"""

import hashlib
import random
import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dj_clustering.identity import (
    assign_track_ids,
    compute_audio_content_hash,
    select_canonical,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    return path


def _known_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


# ---------------------------------------------------------------------------
# compute_audio_content_hash
# ---------------------------------------------------------------------------

def test_hash_deterministic(tmp_path):
    f = _write(tmp_path / "a.mp3", b"hello world audio")
    h1 = compute_audio_content_hash(f)
    h2 = compute_audio_content_hash(f)
    assert h1 == h2
    assert h1 is not None


def test_hash_returns_64_chars(tmp_path):
    f = _write(tmp_path / "a.mp3", b"test data")
    h = compute_audio_content_hash(f)
    assert h is not None
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_hash_correct_value(tmp_path):
    content = b"deterministic content"
    f = _write(tmp_path / "a.mp3", content)
    expected = _known_sha256(content)
    assert compute_audio_content_hash(f) == expected


def test_hash_different_content(tmp_path):
    f1 = _write(tmp_path / "a.mp3", b"content A")
    f2 = _write(tmp_path / "b.mp3", b"content B")
    assert compute_audio_content_hash(f1) != compute_audio_content_hash(f2)


def test_hash_unreadable_returns_none(tmp_path):
    f = _write(tmp_path / "a.mp3", b"data")
    f.chmod(0o000)
    try:
        result = compute_audio_content_hash(f)
        assert result is None
    finally:
        f.chmod(0o644)


# ---------------------------------------------------------------------------
# assign_track_ids — unique files
# ---------------------------------------------------------------------------

def test_unique_files_no_suffix(tmp_path):
    files = [
        _write(tmp_path / "a.mp3", b"aaa"),
        _write(tmp_path / "b.mp3", b"bbb"),
        _write(tmp_path / "c.mp3", b"ccc"),
    ]
    result = assign_track_ids(files)
    for f in files:
        r = result[f]
        assert "_d" not in r["track_id"], f"Unique file should have no _d suffix: {r['track_id']}"
        assert r["is_canonical"] is True
        assert r["is_exact_duplicate"] is False
        assert r["duplicate_of_track_id"] is None


def test_all_track_ids_unique_in_mixed_set(tmp_path):
    content_a = b"unique_a"
    content_b = b"shared_content"
    files = [
        _write(tmp_path / "a.mp3", content_a),
        _write(tmp_path / "b.mp3", content_b),
        _write(tmp_path / "c.mp3", content_b),  # duplicate of b
        _write(tmp_path / "d.mp3", b"unique_d"),
    ]
    result = assign_track_ids(files)
    ids = [r["track_id"] for r in result.values()]
    assert len(ids) == len(set(ids)), f"Duplicate track_ids: {ids}"


# ---------------------------------------------------------------------------
# assign_track_ids — exact duplicates
# ---------------------------------------------------------------------------

def test_two_duplicates_suffix_assignment(tmp_path):
    content = b"same audio bytes"
    f_lex_first = _write(tmp_path / "alpha.mp3", content)
    f_lex_second = _write(tmp_path / "beta.mp3", content)
    result = assign_track_ids([f_lex_first, f_lex_second])

    ra = result[f_lex_first]
    rb = result[f_lex_second]

    # f_lex_first sorts before f_lex_second alphabetically
    assert not ra["track_id"].endswith("_d01"), "canonical should have no _d suffix"
    assert rb["track_id"].endswith("_d01")
    assert ra["is_canonical"] is True
    assert rb["is_canonical"] is False
    assert rb["is_exact_duplicate"] is True
    assert ra["is_exact_duplicate"] is False


def test_three_duplicates_suffix_assignment(tmp_path):
    content = b"triplicate"
    files = [
        _write(tmp_path / "a.mp3", content),
        _write(tmp_path / "b.mp3", content),
        _write(tmp_path / "c.mp3", content),
    ]
    result = assign_track_ids(files)
    ids = sorted(r["track_id"] for r in result.values())
    base = _known_sha256(content)[:16]
    assert base in ids
    assert f"{base}_d01" in ids
    assert f"{base}_d02" in ids


def test_canonical_has_no_suffix(tmp_path):
    content = b"shared bytes"
    files = sorted([
        _write(tmp_path / "z.mp3", content),
        _write(tmp_path / "a.mp3", content),
    ])
    result = assign_track_ids(files)
    canonical = next(r for r in result.values() if r["is_canonical"])
    assert "_d" not in canonical["track_id"]


def test_duplicate_has_suffix(tmp_path):
    content = b"shared bytes"
    files = [
        _write(tmp_path / "a.mp3", content),
        _write(tmp_path / "b.mp3", content),
    ]
    result = assign_track_ids(files)
    dup = next(r for r in result.values() if r["is_exact_duplicate"])
    assert dup["track_id"].endswith("_d01")


# ---------------------------------------------------------------------------
# canonical_track_id and duplicate_of_track_id semantics
# ---------------------------------------------------------------------------

def test_canonical_track_id_field_all_rows(tmp_path):
    content = b"same"
    files = [
        _write(tmp_path / "a.mp3", content),
        _write(tmp_path / "b.mp3", content),
        _write(tmp_path / "c.mp3", content),
    ]
    result = assign_track_ids(files)
    canonical_tid = next(r["track_id"] for r in result.values() if r["is_canonical"])
    for r in result.values():
        assert r["canonical_track_id"] == canonical_tid


def test_canonical_track_id_equals_own_for_unique(tmp_path):
    f = _write(tmp_path / "u.mp3", b"unique")
    result = assign_track_ids([f])
    r = result[f]
    assert r["canonical_track_id"] == r["track_id"]


def test_duplicate_of_none_for_canonical(tmp_path):
    content = b"pair"
    files = [
        _write(tmp_path / "a.mp3", content),
        _write(tmp_path / "b.mp3", content),
    ]
    result = assign_track_ids(files)
    canonical = next(r for r in result.values() if r["is_canonical"])
    assert canonical["duplicate_of_track_id"] is None


def test_duplicate_of_set_for_noncanoical(tmp_path):
    content = b"pair"
    files = [
        _write(tmp_path / "a.mp3", content),
        _write(tmp_path / "b.mp3", content),
    ]
    result = assign_track_ids(files)
    canonical_tid = next(r["track_id"] for r in result.values() if r["is_canonical"])
    dup = next(r for r in result.values() if r["is_exact_duplicate"])
    assert dup["duplicate_of_track_id"] == canonical_tid


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_suffix_ordering_deterministic(tmp_path):
    content = b"determinism test"
    files = [
        _write(tmp_path / "gamma.mp3", content),
        _write(tmp_path / "alpha.mp3", content),
        _write(tmp_path / "beta.mp3", content),
    ]
    result1 = assign_track_ids(list(files))
    shuffled = list(files)
    random.shuffle(shuffled)
    result2 = assign_track_ids(shuffled)

    for f in files:
        assert result1[f]["track_id"] == result2[f]["track_id"], (
            f"track_id not deterministic for {f.name}"
        )


def test_canonical_is_lex_first_path(tmp_path):
    content = b"lex order test"
    f_a = _write(tmp_path / "aardvark.mp3", content)
    f_z = _write(tmp_path / "zebra.mp3", content)
    result = assign_track_ids([f_z, f_a])  # pass in reverse order
    assert result[f_a]["is_canonical"] is True
    assert result[f_z]["is_canonical"] is False


# ---------------------------------------------------------------------------
# select_canonical
# ---------------------------------------------------------------------------

def test_select_canonical_returns_lex_first(tmp_path):
    paths = [
        tmp_path / "z.mp3",
        tmp_path / "a.mp3",
        tmp_path / "m.mp3",
    ]
    assert select_canonical(paths) == tmp_path / "a.mp3"
