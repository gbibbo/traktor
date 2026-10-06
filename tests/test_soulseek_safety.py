"""PURPOSE: Verify Soulseek download gates offline, including fail-closed antivirus.

CHANGELOG: 2026-10-05: Cover proportional sizes, exact transfers and disguised files.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "download_JIJIJI/spotify_soulseek_download.py"
SPEC = importlib.util.spec_from_file_location("soulseek_safety_downloader", MODULE_PATH)
assert SPEC and SPEC.loader
D = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = D
SPEC.loader.exec_module(D)


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory(prefix="soulseek_safety_") as directory:
        yield Path(directory)


def candidate(**overrides):
    return {
        "username": "peer", "filename": "Music/Artist - Track.mp3",
        "format": "mp3", "length_s": 300, "bitrate_kbps": 320,
        "size_bytes": 12_000_000, **overrides,
    }


def raw_candidate(**overrides):
    c = candidate(**overrides)
    return {"User": {"Username": c["username"]}, "File": {
        "Filename": c["filename"], "Length": c["length_s"],
        "Bitrate": c["bitrate_kbps"], "Size": c["size_bytes"],
        "SampleRate": c.get("sample_rate"), "BitDepth": c.get("bit_depth"),
    }}


def track():
    return D.track_from_report({
        "spotify_id": "track-1", "uri": "spotify:track:track-1", "url": "",
        "title": "Track", "artists": ["Artist"], "album": "Album",
        "album_artists": ["Artist"], "duration_ms": 300_000,
        "album_id": None, "release_date": None, "track_number": None,
        "total_tracks": None, "disc_number": None, "isrc": None, "cover_url": None,
    })


@pytest.mark.parametrize("size", [10_200_000, 12_000_000, 15_897_152])
def test_mp3_margin_and_tags(size):
    assert D._audio_size_error(candidate(size_bytes=size)) is None


@pytest.mark.parametrize("size", [1, 10_199_999, 15_897_153, 600_000_000, None, 0, -1, float("nan"), True])
def test_rejects_implausible_or_invalid_sizes(size):
    assert D._audio_size_error(candidate(size_bytes=size))


@pytest.mark.parametrize("length", [None, 0, -1, "bad", float("inf"), True])
def test_rejects_invalid_duration(length):
    assert D._audio_size_error(candidate(length_s=length))


@pytest.mark.parametrize("name", [
    "song.mp3.exe", "song.exe.mp3", "song.zip.mp3", "song.mp3:payload",
    "../song.mp3", "Music/../song.mp3", "song\u202egpj.mp3", "song.mp3 ",
    "song.m4a", "song.lnk", "C:/song.mp3", "song\x00.mp3",
])
def test_rejects_dangerous_names(name):
    assert not D._safe_audio_filename(name)
    assert not D._classify_availability_raw([raw_candidate(filename=name)], track()).downloadable


def test_lossless_bounds_account_for_pcm_and_compression():
    base = candidate(format="flac", sample_rate=44100, bit_depth=16)
    assert D._audio_size_error({**base, "size_bytes": 25_000_000}) is None
    assert D._audio_size_error({**base, "size_bytes": 70_000_000})
    assert D._audio_size_error({**base, "format": "wav", "size_bytes": 52_920_044}) is None
    assert D._audio_size_error({**base, "format": "wav", "size_bytes": 26_460_044}) is None
    assert D._audio_size_error({**base, "format": "wav", "size_bytes": 1_000_000})
    assert D._audio_size_error({**base, "sample_rate": None})


def test_preflight_reports_reason_and_keeps_only_plausible_files():
    check = D._classify_availability_raw([
        raw_candidate(size_bytes=1), raw_candidate(),
    ], track())
    assert check.result_count == 1
    assert check.candidates[0]["size_bytes"] == 12_000_000
    rejected = D._classify_availability_raw([raw_candidate(size_bytes=1)], track())
    assert "proporcional" in rejected.candidates[0]["rejection_reason"]


def test_unknown_bitrate_requires_size_gate_but_remains_available():
    check = D._classify_availability_raw([raw_candidate(bitrate_kbps=None)], track())
    assert check.status == "available_quality_unknown"
    assert not D._classify_availability_raw([raw_candidate(bitrate_kbps="bad")], track()).downloadable


def test_huge_duration_cannot_justify_huge_size():
    assert not D._classify_availability_raw([
        raw_candidate(length_s=10_000, size_bytes=400_000_000),
    ], track()).downloadable
    assert D._classify_availability_raw([
        raw_candidate(length_s=450, size_bytes=18_000_000),
    ], track()).downloadable


def test_malformed_peer_metadata_is_rejected():
    assert not D._classify_availability_raw([{"User": [], "File": "bad"}], track()).downloadable


def test_disguised_executable_is_blocked_before_parser_or_scan(tmp_path):
    path = tmp_path / "song.mp3"
    path.write_bytes(b"MZ" + b"\0" * 20)
    with mock.patch.object(D, "inspect") as parser, mock.patch.object(D, "_scan_antivirus") as scanner:
        with pytest.raises(RuntimeError, match="contenido"):
            D._validate_download(path)
    parser.assert_not_called()
    scanner.assert_not_called()


@pytest.mark.parametrize("code", [2, 1, -1])
def test_antivirus_detection_or_error_blocks_file(tmp_path, code):
    path = tmp_path / "song.mp3"
    path.write_bytes(b"ID3")
    with mock.patch.object(D, "_antivirus_command", return_value=["scanner"]), mock.patch.object(
        D.subprocess, "run", return_value=mock.Mock(returncode=code)
    ):
        with pytest.raises(RuntimeError, match="bloqueado"):
            D._scan_antivirus(path)


def test_antivirus_timeout_blocks_file(tmp_path):
    with mock.patch.object(D, "_antivirus_command", return_value=["scanner"]), mock.patch.object(
        D.subprocess, "run", side_effect=subprocess.TimeoutExpired("scanner", 120)
    ):
        with pytest.raises(RuntimeError, match="bloqueado"):
            D._scan_antivirus(tmp_path / "song.mp3")


def test_clamav_limits_cannot_silently_skip_large_files(tmp_path):
    with mock.patch.object(D, "os", SimpleNamespace(name="posix")), mock.patch.object(
        D.shutil, "which", return_value="/usr/bin/clamscan"
    ):
        cmd = D._antivirus_command(tmp_path / "song.flac")
    assert "--alert-exceeds-max=yes" in cmd
    assert "--max-filesize=512M" in cmd
    assert "--max-scansize=1024M" in cmd


def test_wav_real_content_validates_and_scanner_precedes_parser(tmp_path):
    path = tmp_path / "song.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(44100)
        audio.writeframes(b"\0" * (44100 * 4))
    with mock.patch.object(D, "_scan_antivirus") as scanner:
        local = D._validate_download(path)
    scanner.assert_called_once_with(path)
    assert local.length_s == 1
    with mock.patch.object(D, "_scan_antivirus", side_effect=RuntimeError("blocked")), mock.patch.object(D, "inspect") as parser:
        with pytest.raises(RuntimeError):
            D._validate_download(path)
        parser.assert_not_called()


def test_received_size_must_match_advertised(tmp_path):
    path = tmp_path / "song.mp3"
    path.write_bytes(b"ID3" + b"\0" * 20)
    with mock.patch.object(D, "_scan_antivirus") as scanner:
        with pytest.raises(RuntimeError, match="anunciado"):
            D._validate_download(path, candidate())
        scanner.assert_not_called()


def test_config_inherits_only_login(tmp_path):
    config = tmp_path / ".config/sockseek/sockseek.conf"
    config.parent.mkdir(parents=True)
    config.write_text("username = test\npassword = fixture\non-complete = dangerous\nremote = unsafe\n[profile]\nalbum = true\n", encoding="utf-8")
    dest = tmp_path / "safe.conf"
    with mock.patch.object(D.Path, "home", return_value=tmp_path):
        D._write_sockseek_safe_config(dest, "sockseek")
    assert dest.read_text(encoding="utf-8") == "username = test\npassword = fixture\n"


def test_exact_download_never_searches_again_and_cleans_rejected_bytes(tmp_path):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        incoming = Path(next(s.split("=", 1)[1] for s in cmd if s.startswith("--output-dir=")))
        (incoming / "Artist - Track.mp3").write_bytes(b"MZbad")
        return mock.Mock(returncode=0)

    with mock.patch.object(D, "find_sockseek", return_value="sockseek"), mock.patch.object(
        D, "_check_antivirus_ready"
    ), mock.patch.object(D, "_write_sockseek_safe_config"), mock.patch.object(D.subprocess, "run", side_effect=run):
        files = D.download_with_sockseek(track(), tmp_path, availability=D.AvailabilityCheck("available", 1, [candidate()]))
    assert not files
    assert calls[0][1] == "slsk://peer/Music/Artist%20-%20Track.mp3"
    assert "--format=flac,wav,mp3" in calls[0]
    assert "--name-format={slsk-filename}" in calls[0]
    assert not list(tmp_path.iterdir())


def test_no_scanner_prevents_any_transfer(tmp_path):
    with mock.patch.object(D, "find_sockseek", return_value="sockseek"), mock.patch.object(
        D, "_antivirus_command", side_effect=RuntimeError("no scanner")
    ), mock.patch.object(D.subprocess, "run") as run:
        with pytest.raises(RuntimeError):
            D.download_with_sockseek(track(), tmp_path)
        run.assert_not_called()


def test_successful_exact_transfer_preserves_filename_after_validation(tmp_path):
    c = candidate(filename="Music/Artist - Track.wav", format="wav", length_s=1,
                  size_bytes=176444, sample_rate=44100, bit_depth=16)

    def run(cmd, **kwargs):
        incoming = Path(next(s.split("=", 1)[1] for s in cmd if s.startswith("--output-dir=")))
        with wave.open(str(incoming / "Artist - Track.wav"), "wb") as audio:
            audio.setnchannels(2)
            audio.setsampwidth(2)
            audio.setframerate(44100)
            audio.writeframes(b"\0" * (44100 * 4))
        return mock.Mock(returncode=0)

    with mock.patch.object(D, "find_sockseek", return_value="sockseek"), mock.patch.object(
        D, "_check_antivirus_ready"
    ), mock.patch.object(D, "_write_sockseek_safe_config"), mock.patch.object(
        D, "_scan_antivirus"
    ) as scanner, mock.patch.object(D.subprocess, "run", side_effect=run):
        files = D.download_with_sockseek(track(), tmp_path, availability=D.AvailabilityCheck("available", 1, [c]))
    assert len(files) == 1 and files[0].name == "Artist - Track.wav"
    assert files[0].is_file()
    scanner.assert_called_once()
    assert not list(tmp_path.glob("incoming_*"))


def test_final_antivirus_failure_does_not_promote_output(tmp_path):
    src = tmp_path / "Artist - Track.mp3"
    src.write_bytes(b"ID3fixture")
    local = D.LocalFile(src, 300, 320, "mp3")
    out = tmp_path / "library"
    with mock.patch.object(D, "_validate_download", return_value=local), mock.patch.object(
        D, "download_cover", return_value=None
    ), mock.patch.object(D, "MP3", return_value=mock.Mock(info=mock.Mock(bitrate=320000))), mock.patch.object(
        D, "write_tags"
    ), mock.patch.object(D, "_scan_antivirus", side_effect=RuntimeError("blocked")):
        with pytest.raises(RuntimeError, match="blocked"):
            D.finalize(src, out, track(), local, False, require_cover=False)
    assert not list(out.iterdir())


def test_saved_preflight_changes_on_existing_20_track_fixture():
    path = MODULE_PATH.parent / "soulseek_prueba_descarga/soulseek_availability.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    candidates = [c for row in report["tracks"] for c in row["top_candidates"]]
    assert candidates
    assert any(D._audio_size_error(c) for c in candidates)
