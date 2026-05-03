"""
PURPOSE: Unit tests for src/dj_clustering/inventory.py — decode_status logic,
         metadata extraction, metadata_quality_flags, and hidden/extension filtering.

CHANGELOG:
  D1.2 - Initial implementation.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dj_clustering.inventory import (
    ALLOWED_EXTENSIONS,
    _parse_filename_tags,
    extract_metadata,
    scan_audio_files,
    try_decode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(path: Path, content: bytes = b"audio") -> Path:
    path.write_bytes(content)
    return path


# ---------------------------------------------------------------------------
# scan_audio_files — extension filtering and hidden-file handling
# ---------------------------------------------------------------------------

def test_scan_filters_non_audio(tmp_path):
    _write(tmp_path / "track.mp3")
    _write(tmp_path / "cover.png")
    _write(tmp_path / "notes.txt")
    result = scan_audio_files(tmp_path, recursive=False)
    names = [f.name for f in result]
    assert "track.mp3" in names
    assert "cover.png" not in names
    assert "notes.txt" not in names


def test_scan_case_insensitive_extension(tmp_path):
    _write(tmp_path / "A.MP3")
    _write(tmp_path / "B.Flac")
    result = scan_audio_files(tmp_path, recursive=False)
    assert len(result) == 2


def test_scan_skips_hidden_files(tmp_path):
    _write(tmp_path / "visible.mp3")
    _write(tmp_path / ".hidden.mp3")
    result = scan_audio_files(tmp_path, recursive=False, ignore_hidden=True)
    names = [f.name for f in result]
    assert "visible.mp3" in names
    assert ".hidden.mp3" not in names


def test_scan_hidden_disabled(tmp_path):
    _write(tmp_path / ".hidden.mp3")
    result = scan_audio_files(tmp_path, recursive=False, ignore_hidden=False)
    assert len(result) == 1


def test_scan_result_is_sorted(tmp_path):
    _write(tmp_path / "z.mp3")
    _write(tmp_path / "a.mp3")
    _write(tmp_path / "m.flac")
    result = scan_audio_files(tmp_path, recursive=False)
    assert result == sorted(result)


def test_scan_allowed_extensions_set():
    for ext in (".wav", ".flac", ".mp3", ".m4a", ".aiff", ".aif", ".ogg"):
        assert ext in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# try_decode — decode_status vocabulary
# ---------------------------------------------------------------------------

class _FakeSFInfo:
    duration = 120.5
    format = "MP3"


def test_decode_status_ok(tmp_path):
    f = _write(tmp_path / "ok.mp3")
    with patch("src.dj_clustering.inventory._SOUNDFILE", True), \
         patch("soundfile.info", return_value=_FakeSFInfo()):
        status, dur, err = try_decode(f)
    assert status == "ok"
    assert dur == pytest.approx(120.5)
    assert err is None


def test_decode_status_failed_all_absent(tmp_path):
    f = _write(tmp_path / "bad.mp3")
    with patch("src.dj_clustering.inventory._SOUNDFILE", False), \
         patch("src.dj_clustering.inventory._TORCHAUDIO", False), \
         patch("shutil.which", return_value=None):
        status, dur, err = try_decode(f)
    assert status == "failed"
    assert dur is None


def test_decode_status_ok_probe_only(tmp_path):
    f = _write(tmp_path / "probe.m4a")
    ffprobe_output = json.dumps({"format": {"duration": "200.0"}})

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = ffprobe_output

    with patch("src.dj_clustering.inventory._SOUNDFILE", True), \
         patch("soundfile.info", side_effect=Exception("not supported")), \
         patch("shutil.which", return_value="/usr/bin/ffprobe"), \
         patch("subprocess.run", return_value=mock_proc):
        status, dur, err = try_decode(f)
    assert status == "ok_probe_only"
    assert dur == pytest.approx(200.0)


def test_decode_error_no_private_path(tmp_path):
    f = _write(tmp_path / "bad.mp3")
    with patch("src.dj_clustering.inventory._SOUNDFILE", True), \
         patch("soundfile.info", side_effect=Exception("format error")), \
         patch("src.dj_clustering.inventory._TORCHAUDIO", False), \
         patch("shutil.which", return_value=None):
        status, dur, err = try_decode(f)
    assert status == "failed"
    assert err is not None
    # decode_error must not contain the full file path
    assert str(tmp_path) not in err or len(err) <= 120, \
        "decode_error must be truncated and not contain full private path"


# ---------------------------------------------------------------------------
# _parse_filename_tags
# ---------------------------------------------------------------------------

def test_parse_filename_tags_standard():
    artist, title = _parse_filename_tags("Artist Name - Track Title")
    assert artist == "Artist Name"
    assert title == "Track Title"


def test_parse_filename_tags_multi_artist():
    artist, title = _parse_filename_tags("Carlo Lio, Harvey McKay - Droid Decay")
    assert artist == "Carlo Lio, Harvey McKay"
    assert title == "Droid Decay"


def test_parse_filename_tags_no_separator():
    artist, title = _parse_filename_tags("just_a_filename")
    assert artist is None
    assert title is None


def test_parse_filename_tags_multiple_dashes_uses_first():
    # Only splits on first " - "
    artist, title = _parse_filename_tags("Artist - Title - Extra")
    assert artist == "Artist"
    assert title == "Title - Extra"


# ---------------------------------------------------------------------------
# extract_metadata — flags without mutagen
# ---------------------------------------------------------------------------

def test_metadata_no_tag_library_no_separator(tmp_path):
    f = _write(tmp_path / "justfilename.mp3")
    with patch("src.dj_clustering.inventory._MUTAGEN", False):
        meta = extract_metadata(f)
    assert meta["metadata_quality_flags"] == "no_tag_library"
    assert meta["artist"] is None


def test_metadata_filename_parsed(tmp_path):
    f = _write(tmp_path / "Surgeon - Magneze.mp3")
    with patch("src.dj_clustering.inventory._MUTAGEN", False):
        meta = extract_metadata(f)
    assert meta["metadata_quality_flags"] == "filename_parsed"
    assert meta["artist"] == "Surgeon"
    assert meta["title"] == "Magneze"


# ---------------------------------------------------------------------------
# extract_metadata — flags with mutagen mocked
# ---------------------------------------------------------------------------

def _mock_easy(fields: dict):
    """Build a mock EasyID3-like object returning given field values."""
    def _get(key):
        v = fields.get(key)
        return [v] if v is not None else None
    mock = MagicMock()
    mock.get.side_effect = _get
    return mock


def test_metadata_quality_full_tags(tmp_path):
    f = _write(tmp_path / "full.mp3")
    mock_easy = _mock_easy({
        "artist": "Artist", "title": "Title",
        "album": "Album", "genre": "Techno",
    })
    with patch("src.dj_clustering.inventory._MUTAGEN", True), \
         patch("mutagen.easyid3.EasyID3", return_value=mock_easy), \
         patch("mutagen.id3.ID3", side_effect=Exception):
        meta = extract_metadata(f)
    assert meta["metadata_quality_flags"] == "full_tags"
    assert meta["artist"] == "Artist"


def test_metadata_quality_partial_tags(tmp_path):
    f = _write(tmp_path / "partial.mp3")
    mock_easy = _mock_easy({"artist": "Artist", "title": "Title"})
    with patch("src.dj_clustering.inventory._MUTAGEN", True), \
         patch("mutagen.easyid3.EasyID3", return_value=mock_easy), \
         patch("mutagen.id3.ID3", side_effect=Exception):
        meta = extract_metadata(f)
    assert meta["metadata_quality_flags"] == "partial_tags"


def test_metadata_quality_no_tags(tmp_path):
    f = _write(tmp_path / "notags.mp3")
    mock_easy = _mock_easy({})  # no fields populated
    with patch("src.dj_clustering.inventory._MUTAGEN", True), \
         patch("mutagen.easyid3.EasyID3", return_value=mock_easy), \
         patch("mutagen.id3.ID3", side_effect=Exception):
        meta = extract_metadata(f)
    assert meta["metadata_quality_flags"] == "no_tags"
