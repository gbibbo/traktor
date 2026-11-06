"""PURPOSE: Verify Soulseek download gates offline, including fail-closed antivirus.

CHANGELOG: 2026-10-05: Cover size gates, live limits, bounded conversion and images.
           2026-11-02: A failed transfer prints Sockseek's last lines, never credentials.
           2026-11-02: Preflight progress from --progress-json, kept out of the results.
           2026-11-02: Preflight title judged like the verification (no --strict-title); reuse of
                       finished MP3s from earlier runs.
           2026-11-06: Track by track: search and download interleaved, recording queue, search limiter
                       across processes, failed searches held until another search works, resume.
           2026-11-06: Several tracks at once through one daemon session, stalled transfers dropped,
                       free upload slots first, a second search for tracks whose files never arrived.
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

    def run(cmd, incoming, expected_bytes, config, failure=None):
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

    def run(cmd, incoming, expected_bytes, config, failure=None):
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


FAKE_SOCKSEEK = """
import json, time
def state(title, outcome):
    print(json.dumps({"type": "track_state", "data": {"artist": "A", "title": title,
          "lifecycleState": "Terminal", "terminalOutcome": outcome}}), flush=True)
print(json.dumps({"type": "search_start", "data": {"artist": "A", "title": "1"}}), flush=True)
state("1", "Succeeded"); time.sleep(0.05); state("2", "Succeeded"); time.sleep(0.05); state("3", "Failed")
print("[]"); print("[]"); print("[]")
raise SystemExit(1)
"""


def test_preflight_reports_progress_and_ignores_progress_events(tmp_path):
    seen, commands = [], []
    real_popen = D.subprocess.Popen

    def fake_popen(cmd, **kwargs):
        commands.append(cmd)
        return real_popen([sys.executable, "-c", FAKE_SOCKSEEK], **kwargs)

    with mock.patch.object(D, "find_sockseek", return_value="sockseek"), \
            mock.patch.object(D, "_write_sockseek_safe_config"), \
            mock.patch.object(D.subprocess, "Popen", side_effect=fake_popen):
        checks = D.check_soulseek_availability_batch(
            [track(), track(), track()], timeout=30, on_progress=lambda *p: seen.append(p),
            log_file=tmp_path / "sockseek_preflight.log")
    assert seen == [(1, 1), (2, 2), (3, 2)]
    assert [c.status for c in checks] == ["missing"] * 3  # three empty results, not the events
    assert "--progress-json" in commands[0] and f"--log-file={tmp_path / 'sockseek_preflight.log'}" in commands[0]
    # El título lo juzga la clasificación (no el --strict-title literal) y la duración llega hasta 1,9×.
    assert "--strict-title" not in commands[0] and "--pref-strict-title" in commands[0]
    assert "--length-tol=270" in commands[0] and "--pref-length-tol=60" in commands[0]


def spotify_track(title, artists=("Artist",), seconds=300):
    return D.track_from_report({
        "spotify_id": "track-1", "uri": "spotify:track:track-1", "url": "", "title": title, "artists": list(artists),
        "album": "Album", "album_artists": list(artists), "duration_ms": seconds * 1000, "album_id": None,
        "release_date": None, "track_number": None, "total_tracks": None, "disc_number": None, "isrc": None,
        "cover_url": None,
    })


def preflight_status(track_, name, seconds):
    return D._classify_availability_raw(
        [raw_candidate(filename=f"Music\\{name}", length_s=seconds, size_bytes=seconds * 40_000)], track_).status


@pytest.mark.parametrize("name, seconds, available", [
    ("Mita Gami - Sun (Rafael Remix).mp3", 208, True),               # «(Versión)» frente a « - Versión» de Spotify
    ("01-mita_gami-sun_(rafael_remix).mp3", 208, True),              # nombre scene
    ("Sun - Rafael Remix - Mita Gami 124.mp3", 208, True),           # pool de DJ: artista y BPM al final
    ("Mita Gami - Sun (Rafael Extended Remix).mp3", 368, True),      # la Extended se prefiere
    ("Mita Gami - Sun (Other Guy Remix).mp3", 208, False),           # otro remix
    ("Mita Gami - Sun.mp3", 208, False),                             # la original no es el remix
])
def test_preflight_title_is_judged_like_the_verification(name, seconds, available):
    sun = spotify_track("Sun - Rafael Remix", ("Mita Gami", "Rafael"), 208)
    assert preflight_status(sun, name, seconds).startswith("available") is available


def test_preflight_title_reads_album_names_tags_and_scene_groups():
    aquarela = spotify_track("Aquarela", ("Toquinho",), 240)
    assert preflight_status(aquarela, "Toquinho - Aquarela - 06 - Palavras Cruzadas.mp3", 240) == "missing"  # otra del disco
    assert preflight_status(aquarela, "Toquinho - Aquarela - 10 - Aquarela.mp3", 240) == "available"
    house = spotify_track("The Sound Of House", ("Jude & Frank",), 300)
    assert preflight_status(house, "Jude & Frank - The Sound Of House (Original Mix) [www.dj-promo.org].mp3", 300) == "available"
    assert preflight_status(spotify_track("Peter Gunn", ("Esh",), 300), "07-esh_-_peter_gunn-zzzz.mp3", 300) == "available"
    assert preflight_status(spotify_track("365", ("Charli xcx",), 200), "Charli xcx - BRAT - 15 - 365.mp3", 200) == "available"


def mp3_320(path, spotify_id, cover=True):
    from mutagen.id3 import APIC, ID3, TXXX
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((b"\xff\xfb\xe0\x64" + b"\x00" * 1040) * 40)  # MPEG-1 Layer III, 320 kbps
    tags = ID3()
    tags.add(TXXX(encoding=3, desc="Spotify Track ID", text=[spotify_id]))
    if cover:
        tags.add(APIC(encoding=3, mime="image/jpeg", type=3, desc="", data=b"\xff\xd8\xff\xe0cover"))
    tags.save(path, v2_version=3)
    return path


def test_finished_downloads_are_reused_instead_of_downloaded_again(tmp_path):
    old = mp3_320(tmp_path / "old" / "soulseek" / "Artist - Track.mp3", "track-1")
    bare = mp3_320(tmp_path / "old" / "soulseek" / "Other.mp3", "track-2", cover=False)
    assert D.reusable_downloads([tmp_path / "old" / "soulseek", tmp_path / "missing"]) == {"track-1": old, "track-2": bare}
    out = tmp_path / "new" / "soulseek"
    dest = D.reuse_download(old, out, "track-1", require_cover=True)
    assert dest == out / "Artist - Track.mp3" and dest.exists() and not old.exists()  # movido, no copiado
    assert D.reuse_download(dest, out, "track-1", require_cover=True) == dest          # ya está en esta descarga
    with pytest.raises(RuntimeError, match="cover"):
        D.reuse_download(bare, out, "track-2", require_cover=True)
    with pytest.raises(RuntimeError, match="otro Spotify Track ID"):
        D.reuse_download(dest, out, "track-9", require_cover=True)


def test_search_outcome_reads_only_finished_searches():
    assert D._search_outcome(b'{"type":"track_state","data":{"lifecycleState":"Terminal","terminalOutcome":"Succeeded"}}\n')
    assert D._search_outcome(b'{"type":"track_state","data":{"lifecycleState":"Terminal","terminalOutcome":"Failed"}}') is False
    assert D._search_outcome(b'{"type":"track_state","data":{"lifecycleState":"Searching"}}') is None
    assert D._search_outcome(b'{"type":"search_start","data":{}}') is None
    assert D._search_outcome(b'[{"User":{}}]') is None


def test_failed_transfer_prints_sockseek_tail_without_credentials(tmp_path, capsys):
    cmd = [sys.executable, "-c",
           "print('Login: ok'); print('Password: hunter2'); print('\\x1b[31mPeer rejected the transfer\\x1b[0m');"
           " raise SystemExit(3)"]
    assert D._run_sockseek_transfer(cmd, tmp_path, 1024, tmp_path / "sockseek.conf") == 3
    out = capsys.readouterr().out
    assert "código 3" in out and "sockseek> Peer rejected the transfer" in out and "sockseek> Login: ok" in out
    assert "hunter2" not in out and "\x1b" not in out


def test_no_free_space_prevents_starting_transfer(tmp_path):
    with mock.patch.object(D.shutil, "disk_usage", return_value=SimpleNamespace(free=0)), mock.patch.object(
        D.subprocess, "Popen"
    ) as start:
        with pytest.raises(RuntimeError, match="espacio libre"):
            D._run_sockseek_transfer(["fixture"], tmp_path, 1024, tmp_path / "sockseek.conf")
        start.assert_not_called()


def test_over_limit_transfer_cleans_rejected_payload(tmp_path):
    def overflow(cmd, incoming, expected_bytes, config, failure=None):
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


# ---------------------------------------------------------------------------
# Elegir y verificar candidatos (2026-10-30, run «Jazz-Bossa mío»)
# ---------------------------------------------------------------------------

def spotify_track(title="Track", artists=("Artist",), seconds=300, album="Album", isrc=None, position_id="t1"):
    return D.track_from_report({
        "spotify_id": position_id, "uri": f"spotify:track:{position_id}", "url": "",
        "title": title, "artists": list(artists), "album": album,
        "album_artists": list(artists), "duration_ms": seconds * 1000,
        "album_id": None, "release_date": None, "track_number": None,
        "total_tracks": None, "disc_number": None, "isrc": isrc, "cover_url": None,
    })


def local_file(name, artists, seconds, isrc=None, title=None, fmt="flac"):
    return D.LocalFile(Path(name), seconds, 900, fmt, tag_title=title, tag_artists=list(artists), isrc=isrc)


def test_combined_artist_tags_are_split():
    assert {"toquinho", "vinicius", "toquinho e vinicius"} <= set(D.artist_names("Toquinho e Vinícius"))
    assert {"stan getz", "joao gilberto"} <= set(D.artist_names("Stan Getz & Joao Gilberto"))
    assert D.artist_names("Daft Punk") == ["daft punk"]


def test_duet_written_in_one_tag_is_accepted():
    spotify = spotify_track("Águas De Março", ["Elis Regina", "Antônio Carlos Jobim"], 214)
    local = local_file("01 Aguas de Março.flac", ["António Carlos Jobim & Elis Regina"], 211, title="Águas de Março")
    accepted, _, _, reasons = D.score_candidate(spotify, local)
    assert accepted, reasons


def test_exact_isrc_accepts_differently_written_artist_but_not_another_isrc():
    spotify = spotify_track("Samba Pra Vinícius", ["Toquinho", "Chico Buarque"], 154, isrc="BRMCA7400010")
    same = local_file("07 Samba pra Vinicius.mp3", ["Vários Artistas"], 155, isrc="BRMCA7400010",
                      title="Samba pra Vinicius", fmt="flac")
    accepted, _, _, reasons = D.score_candidate(spotify, same)
    assert accepted and any("ISRC exacto" in r for r in reasons)
    other = local_file("07 Samba pra Vinicius.mp3", ["Vários Artistas"], 155, isrc="BRXXX0000001",
                       title="Samba pra Vinicius")
    accepted, _, _, reasons = D.score_candidate(spotify, other)
    assert not accepted and reasons[0].startswith("artista mismatch")


def test_another_performer_is_still_rejected():
    spotify = spotify_track("A Felicidade", ["Antônio Carlos Jobim"], 125)
    local = local_file("A felicidade.flac", ["Ella Fitzgerald"], 126, title="A felicidade")
    accepted, _, _, reasons = D.score_candidate(spotify, local)
    assert not accepted and reasons[0].startswith("artista mismatch")


def test_candidates_with_spotify_duration_and_album_come_first():
    spotify = spotify_track("Chega De Saudade", ["Antônio Carlos Jobim"], 233, album="Inédito")
    cands = [
        candidate(filename="Jobim/10-Antonio Carlos Jobim - Chega De Saudade.flac", length_s=259),
        candidate(filename="Jobim/Live/Chega de Saudade.flac", length_s=236),
        candidate(filename="Tom Jobim - Inédito - 02 - Chega de Saudade.flac", length_s=233),
        candidate(filename="Jobim/Chega De Saudade.flac", length_s=258),
    ]
    order = [c["length_s"] for c in D.rank_candidates(spotify, cands)]
    assert order == [233, 236, 258, 259]


def test_explicit_extended_keeps_priority_for_dj_music():
    spotify = spotify_track("Track", ["Artist"], 200)
    cands = [candidate(filename="Artist - Track (Radio Edit).mp3", length_s=200),
             candidate(filename="Artist - Track (Extended Mix).mp3", length_s=380)]
    assert D.rank_candidates(spotify, cands)[0]["length_s"] == 380
    assert D.rank_candidates(spotify, cands, prefer_extended=False)[0]["length_s"] == 200


def _fake_transfer(names):
    def run(cmd, incoming, expected_bytes, config, failure=None):
        incoming = Path(next(s.split("=", 1)[1] for s in cmd if s.startswith("--output-dir=")))
        remote = next(s for s in cmd if s.startswith("slsk://"))
        names.append(remote)
        (incoming / Path(remote).name).write_bytes(b"audio")
        return 0
    return run


def test_rejected_download_tries_the_next_candidate(tmp_path):
    names = []
    cands = [candidate(filename="Music/first.mp3"), candidate(filename="Music/second.mp3")]
    with mock.patch.object(D, "find_sockseek", return_value="sockseek"), mock.patch.object(
        D, "_check_antivirus_ready"
    ), mock.patch.object(D, "_write_sockseek_safe_config"), mock.patch.object(
        D, "_validate_download"
    ), mock.patch.object(D, "_run_sockseek_transfer", side_effect=_fake_transfer(names)):
        files = D.download_with_sockseek(track(), tmp_path, availability=D.AvailabilityCheck("available", 2, cands),
                                         verify=lambda path: path.name == "second.mp3")
    assert [Path(n).name for n in names] == ["first.mp3", "second.mp3"]
    assert len(files) == 1 and files[0].name == "second.mp3" and files[0].is_file()
    assert [p.name for p in tmp_path.rglob("*.mp3")] == ["second.mp3"]  # el rechazado se borró


def test_transfers_per_track_are_bounded(tmp_path):
    names = []
    cands = [candidate(filename=f"Music/{i}.mp3") for i in range(6)]
    with mock.patch.object(D, "find_sockseek", return_value="sockseek"), mock.patch.object(
        D, "_check_antivirus_ready"
    ), mock.patch.object(D, "_write_sockseek_safe_config"), mock.patch.object(
        D, "_validate_download"
    ), mock.patch.object(D, "_run_sockseek_transfer", side_effect=_fake_transfer(names)):
        files = D.download_with_sockseek(track(), tmp_path, availability=D.AvailabilityCheck("available", 6, cands),
                                         verify=lambda path: False)
    assert files == [] and len(names) == D.MAX_CANDIDATE_TRANSFERS
    assert not list(tmp_path.rglob("*.mp3"))


def test_inferred_extended_needs_the_full_artist_not_a_part():
    spotify = spotify_track("Tomara", ["Vinícius de Moraes", "Marilia Medalha", "Toquinho"], 132)
    other_version = local_file("06 Tomara.flac", ["Vinicius de Moraes, Toquinho & Maria Creuza"], 186.7, title="Tomara")
    accepted, _, _, reasons = D.score_candidate(spotify, other_version)
    assert not accepted and reasons[0].startswith("duración mismatch")
    single = local_file("Tomara.flac", ["Toquinho"], 160, title="Tomara")
    accepted, _, extended, _ = D.score_candidate(spotify, single)
    assert accepted and extended  # la regla de siempre, con el artista completo


def test_title_in_the_file_name_beats_title_in_the_folder():
    spotify = spotify_track("Wave", ["Antônio Carlos Jobim", "Frank Sinatra"], 200)
    cands = [candidate(filename="Jobim/Wave (1967)/04 - Batidinha.flac", length_s=197),
             candidate(filename="Sinatra & Jobim/07 - Wave.flac", length_s=201)]
    assert D.rank_candidates(spotify, cands)[0]["filename"].endswith("07 - Wave.flac")


# ---------------------------------------------------------------------------
# Tema a tema (2026-11-06)
# ---------------------------------------------------------------------------

def playlist(n: int) -> list:
    tracks = [spotify_track(f"Tema {i}", ("Artista",), 300) for i in range(1, n + 1)]
    for i, t in enumerate(tracks, 1):
        t.spotify_id = f"id{i}"
    return tracks


def run_downloader(tmp_path, monkeypatch, tracks, results, *, downloads_fail=(), extra=(), searches=None, barrier=None):
    """main() con Spotify, sockseek y la descarga simulados. `results`: id -> estados que devuelve cada
    búsqueda de ese tema (el último se repite). Devuelve (código, eventos, cola, filas de disponibilidad)."""
    events = [] if searches is None else searches

    class FakeSpotify:
        def __init__(self, client_id):
            pass

        def playlist_tracks(self, playlist):
            return "Lista", tracks

    calls: dict[str, int] = {}

    def batch(pending, **kwargs):
        assert len(pending) == 1  # una búsqueda por tema
        tid = pending[0].spotify_id
        options = results[tid]
        status = options[min(calls.get(tid, 0), len(options) - 1)]
        calls[tid] = calls.get(tid, 0) + 1
        events.append(("search", tid))
        cands = [candidate()] if status == "available" else []
        return [D.AvailabilityCheck(status=status, result_count=len(cands), candidates=cands,
                                    error="timeout" if status == "error" else None)]

    def download(track_, staging, length_tol=60, *, availability=None, verify=None, prefer_extended=True, **_):
        if barrier is not None:
            barrier.wait()
        events.append(("download", track_.spotify_id))
        if track_.spotify_id in downloads_fail:
            return []
        path = staging / f"{track_.spotify_id}.flac"
        path.write_bytes(b"x")
        assert verify(path)
        return [path]

    def finalize(src, out_dir, track_, source, extended, **_):
        out_dir.mkdir(parents=True, exist_ok=True)
        return mp3_320(out_dir / f"Artista - {track_.title}.mp3", track_.spotify_id), False

    monkeypatch.setattr(D, "Spotify", FakeSpotify)
    monkeypatch.setattr(D, "check_soulseek_availability_batch", batch)
    monkeypatch.setattr(D, "download_with_sockseek", download)
    monkeypatch.setattr(D, "finalize", finalize)
    monkeypatch.setattr(D, "inspect", lambda path: D.LocalFile(path=path, length_s=300, bitrate_kbps=None, format="flac"))
    monkeypatch.setattr(D, "score_candidate", lambda *a, **k: (True, 1.0, False, ["ok"]))
    monkeypatch.setattr(D, "find_sockseek", lambda: "sockseek")
    monkeypatch.setattr(D.shutil, "which", lambda name: name)
    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    queue = tmp_path / "run" / "recording_queue.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "dl", "https://open.spotify.com/playlist/PL1", "--client-id", "x",
        "--output", str(tmp_path / "run" / "soulseek"), "--recording-queue", str(queue),
        "--availability-report", str(tmp_path / "run" / "soulseek_availability.json"),
        "--report", str(tmp_path / "run" / "spotify_reconcile_report.json"),
        "--search-log", str(tmp_path / "searches.json"), "--staging", str(tmp_path / "staging"),
        *(extra if "--parallel" in extra else ("--no-daemon", *extra))])
    (tmp_path / "staging").mkdir(exist_ok=True)
    code = D.main()
    rows = json.loads((tmp_path / "run" / "soulseek_availability.json").read_text(encoding="utf-8"))["tracks"]
    return code, events, D.read_queue(queue), rows


def test_each_track_is_searched_and_downloaded_before_the_next(tmp_path, monkeypatch):
    code, events, queue, rows = run_downloader(
        tmp_path, monkeypatch, playlist(4),
        {"id1": ["available"], "id2": ["missing"], "id3": ["available"], "id4": ["available"]},
        downloads_fail={"id3"})
    # Como a mano: buscar un tema y bajarlo enseguida, antes de buscar el siguiente. Del 3 no llegó ningún
    # archivo: se busca de nuevo al final y, como tampoco llega, va a grabarse.
    assert events == [("search", "id1"), ("download", "id1"), ("search", "id2"), ("search", "id3"),
                      ("download", "id3"), ("search", "id4"), ("download", "id4"), ("search", "id3"),
                      ("download", "id3")]
    assert [(t["spotify"]["spotify_id"], t["origin"]) for t in queue.tracks] == [("id2", "missing"), ("id3", "failed")]
    assert queue.tracks[0]["spotify"]["title"] == "Tema 2" and queue.tracks[0]["duration_seconds"] == 300
    assert queue.end["status"] == "finished"
    assert [r["status"] for r in rows] == ["available", "missing", "available", "available"]
    assert code == 3  # un tema que estaba no se pudo bajar (va a la cola para grabarlo)
    assert sorted(p.name for p in (tmp_path / "run" / "soulseek").glob("*.mp3")) == [
        "Artista - Tema 1.mp3", "Artista - Tema 4.mp3"]


def test_a_failed_search_waits_for_a_search_that_works(tmp_path, monkeypatch):
    code, events, queue, _ = run_downloader(
        tmp_path, monkeypatch, playlist(3), {"id1": ["error"], "id2": ["missing"], "id3": ["error", "available"]})
    assert events[:2] == [("search", "id1"), ("search", "id1")]   # reintento
    # El 1 va a grabarse recién cuando la búsqueda del 2 funcionó (antes, el problema podía ser la búsqueda).
    assert queue.ids() == ["id1", "id2"] and queue.tracks[0]["origin"] == "search_error"
    assert ("download", "id3") in events and code == 0


def test_three_failed_searches_in_a_row_stop_without_recording_them(tmp_path, monkeypatch):
    code, events, queue, rows = run_downloader(
        tmp_path, monkeypatch, playlist(5), {f"id{i}": ["error"] for i in range(1, 6)})
    assert code == D.EXIT_SEARCH_FAILED
    assert queue.tracks == [] and queue.end["status"] == "search_failed" and "timeout" in queue.end["error"]
    assert ("search", "id4") not in events                          # se cortó en el tercero
    assert [r["status"] for r in rows] == ["error", "error", "error", "pending", "pending"]


def test_failed_searches_at_the_end_are_recorded_only_if_a_search_still_works(tmp_path, monkeypatch):
    code, events, queue, _ = run_downloader(
        tmp_path, monkeypatch, playlist(3), {"id1": ["available"], "id2": ["missing"], "id3": ["error"]})
    assert events[-1] == ("search", "id2")  # prueba con uno que se encontró
    assert queue.ids() == ["id2", "id3"] and queue.end["status"] == "finished" and code == 0


def test_failed_searches_at_the_end_with_soulseek_down_are_not_recorded(tmp_path, monkeypatch):
    code, _, queue, _ = run_downloader(
        tmp_path, monkeypatch, playlist(3), {"id1": ["available"], "id2": ["missing", "error"], "id3": ["error"]})
    assert code == D.EXIT_SEARCH_FAILED
    assert queue.ids() == ["id2"] and queue.end["status"] == "search_failed"


def test_a_short_playlist_whose_searches_all_fail_is_not_recorded(tmp_path, monkeypatch):
    code, _, queue, _ = run_downloader(tmp_path, monkeypatch, playlist(2), {"id1": ["error"], "id2": ["error"]})
    assert code == D.EXIT_SEARCH_FAILED and queue.tracks == []


def test_a_resumed_download_keeps_what_was_searched_downloaded_and_queued(tmp_path, monkeypatch):
    tracks = playlist(4)
    run_downloader(tmp_path, monkeypatch, tracks[:2] + tracks[3:],
                   {"id1": ["available"], "id2": ["missing"], "id4": ["available"]})
    # La descarga se cortó (la cola no se cerró) y se retoma con la playlist entera: el 3 no se buscó.
    queue = tmp_path / "run" / "recording_queue.jsonl"
    queue.write_text("".join(line + "\n" for line in queue.read_text(encoding="utf-8").splitlines()
                             if '"type": "end"' not in line), encoding="utf-8")
    # El 4 se bajó, pero el reporte no alcanzó a registrarlo antes del corte: el MP3 está en la carpeta.
    report = tmp_path / "run" / "spotify_reconcile_report.json"
    data = json.loads(report.read_text(encoding="utf-8"))
    data["tracks"] = [row for row in data["tracks"] if row["spotify"]["spotify_id"] != "id4"]
    report.write_text(json.dumps(data), encoding="utf-8")
    code, events, state, rows = run_downloader(tmp_path, monkeypatch, tracks, {"id3": ["missing"]}, searches=[])
    assert events == [("search", "id3")]                    # solo lo que no se había hecho
    assert state.ids() == ["id2", "id3"] and state.end["status"] == "finished"
    assert [r["status"] for r in rows] == ["available", "missing", "missing", "available"]
    assert code == 0


def test_tracks_already_in_the_music_folder_or_downloaded_before_are_not_searched(tmp_path, monkeypatch, capsys):
    folder = tmp_path / "lib" / "Lista"
    mp3_320(folder / "Artista - Tema 1.mp3", "id1")                                # ya guardado
    mp3_320(tmp_path / "old" / "soulseek" / "Artista - Tema 2.mp3", "id2")         # bajado por una descarga cortada
    code, events, queue, rows = run_downloader(
        tmp_path, monkeypatch, playlist(3), {"id3": ["missing"]},
        extra=("--have-dir", str(folder), "--reuse-dir", str(tmp_path / "old" / "soulseek")))
    assert events == [("search", "id3")]
    assert [r["status"] for r in rows] == ["available", "available", "missing"]
    assert (tmp_path / "run" / "soulseek" / "Artista - Tema 2.mp3").is_file()  # traído a esta descarga
    assert queue.ids() == ["id3"] and code == 0
    assert "ya está en tu carpeta" in capsys.readouterr().out


def test_a_repeated_track_is_resolved_once(tmp_path, monkeypatch):
    tracks = playlist(2)
    tracks.append(tracks[1])
    _, events, queue, rows = run_downloader(tmp_path, monkeypatch, tracks, {"id1": ["available"], "id2": ["missing"]})
    assert events.count(("search", "id2")) == 1 and queue.ids() == ["id2"]
    assert [r["status"] for r in rows] == ["available", "missing", "missing"]


def test_search_limit_holds_across_processes_and_runs(tmp_path):
    now = [1000.0]
    slept: list[float] = []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    log = tmp_path / "searches.json"
    first = D.SearchLimiter(log, per_window=3, window_s=100, clock=lambda: now[0], sleep=sleep)
    for _ in range(3):
        first.acquire()
        now[0] += 1
    assert slept == []
    second = D.SearchLimiter(log, per_window=3, window_s=100, clock=lambda: now[0], sleep=sleep)  # otro proceso
    waits = []
    second.acquire(on_wait=waits.append)
    assert waits and round(waits[0]) == 97 and round(sum(slept)) == 97  # hasta que la primera sale de la ventana
    # Una búsqueda cuenta desde que terminó: sockseek se conecta antes de buscar (hasta 20 s).
    now[0] = 5000.0
    third = D.SearchLimiter(log, per_window=3, window_s=100, clock=lambda: now[0], sleep=sleep)
    stamp = third.acquire()
    now[0] += 20
    third.finish(stamp)
    assert max(json.loads(log.read_text(encoding="utf-8"))["searches"]) == 5020.0
    assert 5000.0 not in json.loads(log.read_text(encoding="utf-8"))["searches"]
    log.write_text(json.dumps({"searches": [now[0] + 5000] * 5}), encoding="utf-8")  # reloj atrasado
    assert D.SearchLimiter(log, per_window=3, window_s=100, clock=lambda: now[0]).wait_s() == 0


def test_queue_reader_skips_a_line_still_being_written_and_duplicates(tmp_path):
    path = tmp_path / "q.jsonl"
    entry = {"playlist_position": 1, "spotify": {"spotify_id": "id1"}}
    D.append_track(path, entry, origin="missing", reason="No está en Soulseek")
    D.append_track(path, entry, origin="failed", reason="otra vez")
    torn = '{"type": "track", "reason": "Peña", "spotify": {"spotify_id": "id2"'.encode()
    cut = torn.index("ñ".encode()) + 1
    with path.open("ab") as handle:  # a medio escribir, cortado en medio de la «ñ»
        handle.write(torn[:cut])
    assert D.read_queue(path).ids() == ["id1"]
    with path.open("ab") as handle:
        handle.write(torn[cut:])
    state = D.read_queue(path)
    assert state.ids() == ["id1"] and state.tracks[0]["origin"] == "missing" and not state.closed
    with path.open("a", encoding="utf-8") as handle:
        handle.write("}}\n")
    D.close_queue(path, "finished")
    state = D.read_queue(path)
    assert state.ids() == ["id1", "id2"] and state.closed
    assert D.read_queue(tmp_path / "no_existe.jsonl").tracks == []


# ---------------------------------------------------------------------------
# Varios temas a la vez con una sola sesión (2026-11-06)
# ---------------------------------------------------------------------------

class FakeDaemon:
    started: list = []

    def __init__(self, binary, *, state_path, log_path, mock_dir=None):
        self.url = "http://127.0.0.1:1"

    def start(self, timeout=30.0):
        FakeDaemon.started.append(self)
        return self.url

    def stop(self):
        FakeDaemon.started.remove(self)

    def cancel_matching(self, query_text):
        return 0


def test_two_tracks_are_searched_and_downloaded_at_the_same_time(tmp_path, monkeypatch):
    import threading
    together = threading.Barrier(2, timeout=10)  # solo pasa si los dos temas se están bajando a la vez
    monkeypatch.setattr(D, "SockseekDaemon", FakeDaemon)
    code, events, queue, rows = run_downloader(
        tmp_path, monkeypatch, playlist(2), {"id1": ["available"], "id2": ["available"]},
        extra=("--parallel", "2"), barrier=together)
    assert code == 0 and [r["status"] for r in rows] == ["available", "available"]
    assert sorted(events) == [("download", "id1"), ("download", "id2"), ("search", "id1"), ("search", "id2")]
    assert sorted(p.name for p in (tmp_path / "run" / "soulseek").glob("*.mp3")) == [
        "Artista - Tema 1.mp3", "Artista - Tema 2.mp3"]
    assert FakeDaemon.started == []  # la sesión se cerró al terminar


def test_without_the_shared_session_it_goes_one_track_at_a_time(tmp_path, monkeypatch, capsys):
    class Broken(FakeDaemon):
        def start(self, timeout=30.0):
            raise RuntimeError("puerto ocupado")
    monkeypatch.setattr(D, "SockseekDaemon", Broken)
    code, events, _, _ = run_downloader(
        tmp_path, monkeypatch, playlist(3), {"id1": ["missing"], "id2": ["missing"], "id3": ["missing"]},
        extra=("--parallel", "3"))
    assert code == 0 and events == [("search", "id1"), ("search", "id2"), ("search", "id3")]
    assert "1 tema(s) a la vez" in capsys.readouterr().out


def test_a_track_whose_files_never_arrived_is_searched_again_at_the_end(tmp_path, monkeypatch):
    code, events, queue, _ = run_downloader(tmp_path, monkeypatch, playlist(2),
                                            {"id1": ["available"], "id2": ["missing"]}, downloads_fail={"id1"})
    assert events == [("search", "id1"), ("download", "id1"), ("search", "id2"), ("search", "id1"), ("download", "id1")]
    assert queue.ids() == ["id2", "id1"] and queue.tracks[1]["origin"] == "failed"


def test_a_transfer_that_receives_nothing_is_dropped(tmp_path):
    with mock.patch.object(D, "TRANSFER_STALL_SECONDS", 0.3):
        with pytest.raises(RuntimeError, match="no avanzó"):
            D._run_sockseek_transfer([sys.executable, "-c", "import time; time.sleep(10)"], tmp_path, 1024,
                                     tmp_path / "sockseek.conf")


def test_peers_with_a_free_upload_slot_and_speed_come_first():
    spotify = spotify_track("Track", ["Artist"], 300)
    cands = [candidate(filename="Music/Artist - Track.mp3", username="busy", has_free_upload_slot=False,
                       upload_speed_mib_s=50),
             candidate(filename="Music/Artist - Track.mp3", username="slow", has_free_upload_slot=True,
                       upload_speed_mib_s=0.1),
             candidate(filename="Music/Artist - Track.mp3", username="fast", has_free_upload_slot=True,
                       upload_speed_mib_s=5)]
    assert [c["username"] for c in D.rank_candidates(spotify, cands)] == ["fast", "slow", "busy"]


def test_failed_transfers_do_not_use_up_the_verification_attempts(tmp_path):
    cands = [candidate(filename=f"Music/{i}.mp3", username=f"u{i}") for i in range(10)]
    avoid: set = set()
    outcome: dict = {}
    with mock.patch.object(D, "find_sockseek", return_value="sockseek"), mock.patch.object(
        D, "_check_antivirus_ready"
    ), mock.patch.object(D, "_write_sockseek_safe_config"), mock.patch.object(
        D, "_run_sockseek_transfer", return_value=1
    ) as transfer:
        files = D.download_with_sockseek(track(), tmp_path, availability=D.AvailabilityCheck("available", 10, cands),
                                         verify=lambda path: True, avoid=avoid, outcome=outcome)
    assert files == [] and transfer.call_count == D.MAX_TRANSFER_ATTEMPTS > D.MAX_CANDIDATE_TRANSFERS
    assert outcome == {"attempts": D.MAX_TRANSFER_ATTEMPTS, "arrived": 0} and len(avoid) == D.MAX_TRANSFER_ATTEMPTS


def test_why_a_peer_did_not_send_the_file_is_readable():
    assert D.transfer_failure_reason(" Transfer rejected: Banned") == ("nos tiene bloqueados", True)
    assert D.transfer_failure_reason("Failed to read 12972451 bytes from 104.255.1.223:2299: Remote connection closed") \
        == ("cortó la conexión", False)
    assert D.transfer_failure_reason("User DjMj appears to be offline")[1] is True
    assert D.transfer_failure_reason("Transfer rejected: Too many files")[0] == "tiene la cola llena"


def test_a_peer_that_blocked_us_is_not_asked_again(tmp_path, capsys):
    cands = [candidate(filename="Music/a.mp3", username="koalabeer"),
             candidate(filename="Music/b.mp3", username="koalabeer"),
             candidate(filename="Music/c.mp3", username="otro")]
    asked = []

    def transfer(cmd, incoming, expected, config, failure=None):
        asked.append(next(part for part in cmd if part.startswith("slsk://")))
        failure.update(reason="nos tiene bloqueados", whole_user=True)
        return 1
    avoid: set = set()
    with mock.patch.object(D, "find_sockseek", return_value="sockseek"), mock.patch.object(
        D, "_check_antivirus_ready"
    ), mock.patch.object(D, "_write_sockseek_safe_config"), mock.patch.object(D, "_run_sockseek_transfer", side_effect=transfer):
        D.download_with_sockseek(track(), tmp_path, availability=D.AvailabilityCheck("available", 3, cands),
                                 verify=lambda path: True, avoid=avoid)
    assert [a.split("/")[2] for a in asked] == ["koalabeer", "otro"]  # el segundo archivo de koalabeer no se pide
    assert ("koalabeer", "*") in avoid
    assert "koalabeer no lo mandó: nos tiene bloqueados" in capsys.readouterr().out


def test_sockseek_lines_of_a_failed_transfer_go_to_the_run_log(tmp_path, capsys, monkeypatch):
    log = tmp_path / "sockseek_transferencias.log"
    monkeypatch.setattr(D, "_TRANSFER_LOG", log)
    cmd = [sys.executable, "-c", "print('[015] SongJob: failed'); print('Error: Transfer rejected: Banned'); raise SystemExit(1)"]
    failure: dict = {}
    assert D._run_sockseek_transfer(cmd, tmp_path, 1024, tmp_path / "sockseek.conf", failure) == 1
    assert failure == {"reason": "nos tiene bloqueados", "whole_user": True}
    assert "sockseek>" not in capsys.readouterr().out and "Transfer rejected: Banned" in log.read_text(encoding="utf-8")


def test_a_stale_daemon_is_closed_only_if_it_is_that_daemon(tmp_path, monkeypatch):
    state = tmp_path / D.DAEMON_STATE_NAME
    killed = []
    monkeypatch.setattr(D.subprocess, "run", lambda cmd, **k: killed.append(cmd))
    monkeypatch.setattr(D.os, "kill", lambda pid, sig: killed.append(pid), raising=False)
    D.write_report(state, {"pid": 4242, "port": 5555})
    monkeypatch.setattr(D, "_process_command_line", lambda pid: "C:/x/notepad.exe")
    assert D.stop_daemon_from_state(state) is False and killed == [] and not state.exists()
    D.write_report(state, {"pid": 4242, "port": 5555})
    monkeypatch.setattr(D, "_process_command_line", lambda pid: "C:/tools/sockseek.exe daemon --server-port 5555 --config=x")
    assert D.stop_daemon_from_state(state) is True and len(killed) == 1


def test_the_daemon_cancels_only_live_jobs_of_that_file(monkeypatch):
    daemon = D.SockseekDaemon("sockseek", state_path=Path("x.json"), log_path=Path("x.log"))
    daemon.url = "http://127.0.0.1:1"
    calls = []

    def request(method, path, timeout=5):
        calls.append((method, path))
        if path == "/api/jobs":
            return [{"jobId": "a", "kind": "song", "lifecycleState": "Running", "queryText": "Artist - Track"},
                    {"jobId": "b", "kind": "song", "lifecycleState": "Terminal", "queryText": "Artist - Track"},
                    {"jobId": "c", "kind": "song", "lifecycleState": "Running", "queryText": "Otro"}]
        return None
    monkeypatch.setattr(daemon, "_request", request)
    assert daemon.cancel_matching("Artist - Track") == 1
    assert ("POST", "/api/jobs/a/cancel") in calls and ("POST", "/api/jobs/c/cancel") not in calls


def test_through_the_daemon_sockseek_runs_without_config_or_its_own_session(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "_SOCKSEEK_REMOTE", "http://127.0.0.1:5392")
    assert D._sockseek_session_args(tmp_path, "sockseek") == ["--no-config", "--remote=http://127.0.0.1:5392"]
    monkeypatch.setattr(D, "_SOCKSEEK_REMOTE", None)
    monkeypatch.setattr(D, "_SOCKSEEK_MOCK_DIR", tmp_path)
    assert D._sockseek_session_args(tmp_path, "sockseek")[1] == f"--mock-files-dir={tmp_path}"
