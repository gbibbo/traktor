"""PURPOSE: Verify Soulseek download gates offline, including fail-closed antivirus.

CHANGELOG: 2026-10-05: Cover size gates, live limits, bounded conversion and images.
"""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import struct
import wave
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from PIL import Image

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

    def run(cmd, incoming, expected_bytes, config):
        calls.append(cmd)
        incoming = Path(next(s.split("=", 1)[1] for s in cmd if s.startswith("--output-dir=")))
        (incoming / "Artist - Track.mp3").write_bytes(b"MZbad")
        return 0

    with mock.patch.object(D, "find_sockseek", return_value="sockseek"), mock.patch.object(
        D, "_check_antivirus_ready"
    ), mock.patch.object(D, "_write_sockseek_safe_config"), mock.patch.object(D, "_run_sockseek_transfer", side_effect=run):
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

    def run(cmd, incoming, expected_bytes, config):
        incoming = Path(next(s.split("=", 1)[1] for s in cmd if s.startswith("--output-dir=")))
        with wave.open(str(incoming / "Artist - Track.wav"), "wb") as audio:
            audio.setnchannels(2)
            audio.setsampwidth(2)
            audio.setframerate(44100)
            audio.writeframes(b"\0" * (44100 * 4))
        return 0

    with mock.patch.object(D, "find_sockseek", return_value="sockseek"), mock.patch.object(
        D, "_check_antivirus_ready"
    ), mock.patch.object(D, "_write_sockseek_safe_config"), mock.patch.object(
        D, "_scan_antivirus"
    ) as scanner, mock.patch.object(D, "_run_sockseek_transfer", side_effect=run):
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
    ), mock.patch.object(D, "MP3", return_value=mock.Mock(info=mock.Mock(bitrate=320000, length=300))), mock.patch.object(
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


def image_bytes(fmt="PNG"):
    output = io.BytesIO()
    Image.new("RGB", (8, 8), "blue").save(output, format=fmt)
    return output.getvalue()


def test_cover_is_rebuilt_without_trailing_payload():
    payload = b"trailing-fixture-payload"
    with mock.patch.object(D, "_scan_antivirus") as scan:
        rebuilt, mime = D._validated_cover(image_bytes("JPEG") + payload, "image/jpeg")
    assert mime == "image/jpeg" and payload not in rebuilt
    assert scan.call_count == 2
    with Image.open(io.BytesIO(rebuilt)) as image:
        image.load()
        assert image.size == (8, 8)


@pytest.mark.parametrize("data,mime", [
    (b"MZ-not-an-image", "image/jpeg"),
    (image_bytes("PNG"), "image/jpeg"),
    (image_bytes("PNG")[:-10], "image/png"),
    (image_bytes("JPEG")[:40], "image/jpeg"),
])
def test_rejects_fake_mismatched_and_truncated_cover(data, mime):
    with mock.patch.object(D, "_scan_antivirus"):
        with pytest.raises(RuntimeError, match="carátula inválido"):
            D._validated_cover(data, mime)


def test_cover_pixel_limit_checked_before_decoding():
    data = bytearray(image_bytes())
    data[16:24] = struct.pack(">II", 5000, 5000)
    data[29:33] = struct.pack(">I", zlib.crc32(data[12:29]))
    with mock.patch.object(D, "_scan_antivirus"):
        with pytest.raises(RuntimeError):
            D._validated_cover(bytes(data), "image/png")


def test_animated_cover_rejected():
    output = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(
        output, format="PNG", save_all=True,
        append_images=[Image.new("RGB", (2, 2), "blue")], duration=100,
    )
    with mock.patch.object(D, "_scan_antivirus"):
        with pytest.raises(RuntimeError):
            D._validated_cover(output.getvalue(), "image/png")


def test_cover_scan_failure_prevents_image_parser():
    with mock.patch.object(D, "_scan_antivirus", side_effect=RuntimeError("blocked")), mock.patch.object(
        Image, "open"
    ) as parser:
        with pytest.raises(RuntimeError):
            D._validated_cover(image_bytes(), "image/png")
        parser.assert_not_called()


def test_cover_download_rejects_local_file_url():
    with mock.patch.object(D.urllib.request, "urlopen") as request:
        with pytest.raises(RuntimeError, match="HTTPS"):
            D.download_cover("file:///fixture.jpg", retries=1)
        request.assert_not_called()


@pytest.mark.parametrize("sleep_seconds", [0, 10])
def test_monitor_rejects_excess_bytes_even_when_child_finishes(tmp_path, sleep_seconds):
    processes = []
    real_popen = D.subprocess.Popen

    def start(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    cmd = [sys.executable, "-c",
           "import sys,time; from pathlib import Path; Path(sys.argv[1]).write_bytes(b'x'*2048); time.sleep(float(sys.argv[2]))",
           str(tmp_path / "song.mp3.incomplete"), str(sleep_seconds)]
    with mock.patch.object(D.subprocess, "Popen", side_effect=start):
        with pytest.raises(RuntimeError, match="bytes anunciados"):
            D._run_sockseek_transfer(cmd, tmp_path, 1024, tmp_path / "sockseek.conf")
    assert processes and processes[0].poll() is not None


def test_monitor_times_out_and_reaps_child(tmp_path):
    processes = []
    real_popen = D.subprocess.Popen

    def start(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    with mock.patch.object(D, "TRANSFER_TIMEOUT_SECONDS", 0.2), mock.patch.object(
        D.subprocess, "Popen", side_effect=start
    ):
        with pytest.raises(RuntimeError, match="timeout"):
            D._run_sockseek_transfer([sys.executable, "-c", "import time; time.sleep(10)"], tmp_path, 1024, tmp_path / "sockseek.conf")
    assert processes[0].poll() is not None


def test_monitor_accepts_exact_size_and_excludes_config(tmp_path):
    config = tmp_path / "sockseek.conf"
    config.write_bytes(b"x" * 3000)
    cmd = [sys.executable, "-c",
           "import sys; from pathlib import Path; Path(sys.argv[1]).write_bytes(b'x'*1024)",
           str(tmp_path / "song.mp3")]
    assert D._run_sockseek_transfer(cmd, tmp_path, 1024, config) == 0


def test_no_free_space_prevents_starting_transfer(tmp_path):
    with mock.patch.object(D.shutil, "disk_usage", return_value=SimpleNamespace(free=0)), mock.patch.object(
        D.subprocess, "Popen"
    ) as start:
        with pytest.raises(RuntimeError, match="espacio libre"):
            D._run_sockseek_transfer(["fixture"], tmp_path, 1024, tmp_path / "sockseek.conf")
        start.assert_not_called()


def test_over_limit_transfer_cleans_rejected_payload(tmp_path):
    def overflow(cmd, incoming, expected_bytes, config):
        (incoming / "song.mp3.incomplete").write_bytes(b"fixture")
        raise RuntimeError("bytes anunciados")

    with mock.patch.object(D, "find_sockseek", return_value="sockseek"), mock.patch.object(
        D, "_check_antivirus_ready"
    ), mock.patch.object(D, "_write_sockseek_safe_config"), mock.patch.object(
        D, "_run_sockseek_transfer", side_effect=overflow
    ), mock.patch.object(D, "_validate_download") as validate:
        assert D.download_with_sockseek(track(), tmp_path, availability=D.AvailabilityCheck("available", 1, [candidate()])) == []
    validate.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_ffmpeg_timeout_blocks_promotion_and_has_resource_limits(tmp_path):
    src = tmp_path / "Artist - Track.wav"
    with wave.open(str(src), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(44100)
        audio.writeframes(b"\0" * (44100 * 4))
    local = D.inspect(src)
    out = tmp_path / "library"
    with mock.patch.object(D, "_scan_antivirus"), mock.patch.object(D, "download_cover", return_value=None), mock.patch.object(
        D.subprocess, "run", side_effect=subprocess.TimeoutExpired("ffmpeg", 180)
    ) as run:
        with pytest.raises(RuntimeError, match="tiempo permitido"):
            D.finalize(src, out, track(), local, False, require_cover=False)
    cmd = run.call_args.args[0]
    assert run.call_args.kwargs["timeout"] == D.FFMPEG_TIMEOUT_SECONDS
    assert cmd[cmd.index("-protocol_whitelist") + 1] == "file"
    assert cmd[cmd.index("-f") + 1] == "wav"
    assert "-fs" in cmd and "-max_alloc" in cmd and "-nostdin" in cmd
    assert not list(out.iterdir())


def test_truncated_conversion_is_not_promoted(tmp_path):
    src = tmp_path / "Artist - Track.wav"
    with wave.open(str(src), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(44100)
        audio.writeframes(b"\0" * (44100 * 4 * 5))
    local = D.inspect(src)
    out = tmp_path / "library"
    with mock.patch.object(D, "_scan_antivirus"), mock.patch.object(D, "download_cover", return_value=None), mock.patch.object(
        D.subprocess, "run"
    ), mock.patch.object(D, "MP3", return_value=SimpleNamespace(info=SimpleNamespace(bitrate=320000, length=1))), mock.patch.object(
        D, "write_tags"
    ) as tags:
        with pytest.raises(RuntimeError, match="truncamiento"):
            D.finalize(src, out, track(), local, False, require_cover=False)
    tags.assert_not_called()
    assert not list(out.iterdir())
