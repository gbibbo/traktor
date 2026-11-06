"""PURPOSE: Test Spotify/Soulseek synchronization, metadata and consolidation.

CHANGELOG:
- 2026-11-06: Track by track: recording sessions start with the first queued track while the downloader
              goes on, an empty queue waits for the next track, postponing or Spotify stopping keep the
              download going, sessions resume from their checkpoint, a broken search saves a partial
              playlist; the batch-preflight handshakes and old-layout resume tests are gone.
- 2026-11-02: A failed read of the preflight status does not shrink the wait for the manifest.
              Postponed recording: from the start, during the download, and turned off again.
- 2026-10-30: Cover .env output roots and the absolute player token cache.
- 2026-10-05: Isolate placement tests from the mandatory audio/AV gate.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import unittest
import wave
from array import array
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from mutagen.id3 import ID3, TXXX


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "download_JIJIJI"
    / "spotify_soulseek_orchestrator.py"
)
SPEC = importlib.util.spec_from_file_location("spotify_soulseek_orchestrator", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ORCHESTRATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORCHESTRATOR)

RECORDER_PATH = (
    Path(__file__).resolve().parents[1]
    / "download_JIJIJI"
    / "playlist_loopback_recorder.py"
)
RECORDER_SPEC = importlib.util.spec_from_file_location(
    "playlist_loopback_recorder", RECORDER_PATH
)
assert RECORDER_SPEC is not None and RECORDER_SPEC.loader is not None
RECORDER = importlib.util.module_from_spec(RECORDER_SPEC)
sys.modules[RECORDER_SPEC.name] = RECORDER
RECORDER_SPEC.loader.exec_module(RECORDER)
DOWNLOADER = sys.modules["spotify_soulseek_download"]

PLAYER_PATH = (
    Path(__file__).resolve().parents[1]
    / "download_JIJIJI"
    / "spotify_playlist.py"
)
PLAYER_SPEC = importlib.util.spec_from_file_location("spotify_playlist", PLAYER_PATH)
assert PLAYER_SPEC is not None and PLAYER_SPEC.loader is not None
PLAYER = importlib.util.module_from_spec(PLAYER_SPEC)
PLAYER_SPEC.loader.exec_module(PLAYER)
QUEUE = sys.modules["recording_queue"]


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class PlaylistIdTests(unittest.TestCase):
    def test_extracts_id_from_supported_forms(self) -> None:
        playlist_id = "37i9dQZF1DX6J5NfMJS675"
        self.assertEqual(
            ORCHESTRATOR.playlist_id_from_url(
                f"https://open.spotify.com/playlist/{playlist_id}?si=abc"
            ),
            playlist_id,
        )
        self.assertEqual(
            ORCHESTRATOR.playlist_id_from_url(f"spotify:playlist:{playlist_id}"),
            playlist_id,
        )
        self.assertEqual(ORCHESTRATOR.playlist_id_from_url(playlist_id), playlist_id)

    def test_normalizes_markdown_link_to_canonical_url(self) -> None:
        playlist_id = "31xPzrgElxRxrMErJ1WBhD"
        url = f"https://open.spotify.com/playlist/{playlist_id}?si=abc"
        self.assertEqual(
            ORCHESTRATOR.canonical_playlist_url(f"[{url}]({url})"),
            f"https://open.spotify.com/playlist/{playlist_id}",
        )

    def test_rejects_invalid_url(self) -> None:
        with self.assertRaises(ORCHESTRATOR.WorkflowError):
            ORCHESTRATOR.playlist_id_from_url("https://example.com/not-spotify")


class PlaylistBatchTests(unittest.TestCase):
    def test_downloader_creates_ordered_batches_of_ten(self) -> None:
        client = object.__new__(DOWNLOADER.Spotify)
        tracks = [mock.Mock(uri=f"spotify:track:{index}") for index in range(23)]

        def create(_source: str, selected: list, *, name: str | None = None) -> dict:
            return {
                "id": f"playlist-{len(selected)}-{selected[0].uri}",
                "name": name,
                "url": f"https://open.spotify.com/playlist/{selected[0].uri[-1]}",
            }

        client.create_missing_playlist = mock.Mock(side_effect=create)

        batches = client.create_missing_playlists("Source", tracks, batch_size=10)

        self.assertEqual([batch["track_count"] for batch in batches], [10, 10, 3])
        self.assertEqual(
            [(batch["start_index"], batch["end_index"]) for batch in batches],
            [(0, 10), (10, 20), (20, 23)],
        )
        self.assertTrue(batches[0]["name"].endswith("[01/03]"))
        self.assertTrue(batches[2]["name"].endswith("[03/03]"))

    def test_recorder_target_is_limited_to_current_batch(self) -> None:
        self.assertEqual(RECORDER.recording_target(20, 100, 10), 30)
        self.assertEqual(RECORDER.recording_target(95, 100, 10), 100)
        self.assertEqual(RECORDER.recording_target(20, 100, None), 100)


class RecordedMetadataTests(unittest.TestCase):
    @staticmethod
    def spotify_report() -> dict[str, object]:
        return {
            "spotify_id": "spotify-track-1",
            "uri": "spotify:track:spotify-track-1",
            "url": "https://open.spotify.com/track/spotify-track-1",
            "title": 'Title: One?*',
            "artists": ["Primary Artist", "Guest Artist"],
            "album": "Official Album",
            "album_id": "album-1",
            "album_artists": ["Album Artist"],
            "release_date": "2026-09-27",
            "track_number": 3,
            "total_tracks": 10,
            "disc_number": 1,
            "duration_ms": 123456,
            "isrc": "UYABC2600001",
            "cover_url": "https://example.test/cover.jpg",
            "explicit": True,
            "label": "Example Label",
            "copyrights": ["(C) Example", "(P) Example"],
            "playlist_id": "playlist-1",
            "playlist_name": "Source Playlist",
            "added_at": "2026-09-27T00:00:00Z",
            "duration_seconds": 123.456,
        }

    def test_recorded_filename_uses_soulseek_criterion(self) -> None:
        track = {"spotify": self.spotify_report()}

        filename = RECORDER.make_output_filename(track, fallback_index=7)

        self.assertEqual(filename, "Primary Artist - Title One.mp3")

    def test_filename_collision_uses_spotify_id_suffix(self) -> None:
        first_report = self.spotify_report()
        second_report = {**first_report, "spotify_id": "spotify-track-2"}
        filenames = DOWNLOADER.spotify_output_filenames(
            [
                DOWNLOADER.track_from_report(first_report),
                DOWNLOADER.track_from_report(second_report),
            ]
        )

        self.assertEqual(filenames[0], "Primary Artist - Title One.mp3")
        self.assertEqual(
            filenames[1],
            "Primary Artist - Title One [spotify-track-2].mp3",
        )

    def test_recorded_mp3_gets_canonical_tags_and_cover(self) -> None:
        report = self.spotify_report()
        track = DOWNLOADER.track_from_report(report)
        cover = (b"fake-jpeg-data", "image/jpeg")

        with tempfile.TemporaryDirectory() as temp_dir:
            mp3_path = Path(temp_dir) / "recorded.mp3"
            mp3_path.write_bytes(b"")
            source = DOWNLOADER.LocalFile(
                path=mp3_path,
                length_s=123.4,
                bitrate_kbps=320.0,
                format="mp3",
            )
            with (
                mock.patch.object(DOWNLOADER, "inspect", return_value=source),
                mock.patch.object(DOWNLOADER, "download_cover", return_value=cover),
            ):
                DOWNLOADER.apply_spotify_metadata(mp3_path, track)

            tags = ID3(mp3_path)
            self.assertEqual(str(tags["TIT2"]), report["title"])
            self.assertEqual(str(tags["TALB"]), report["album"])
            self.assertEqual(str(tags["TRCK"]), "3/10")
            self.assertEqual(str(tags["TPOS"]), "1")
            self.assertEqual(str(tags["TSRC"]), report["isrc"])
            self.assertTrue(tags.getall("APIC"))
            custom = {
                frame.desc: str(frame)
                for frame in tags.getall("TXXX")
            }
            self.assertEqual(custom["Spotify Track ID"], report["spotify_id"])
            self.assertEqual(custom["Version"], "Spotify-length")
            self.assertEqual(
                set(custom),
                {
                    "Spotify Track ID",
                    "Spotify URI",
                    "Spotify URL",
                    "Spotify Album ID",
                    "Version",
                },
            )
            self.assertNotIn("TCOP", tags)
            self.assertNotIn("TPUB", tags)


class ConsolidationTests(unittest.TestCase):
    @staticmethod
    def tagged_file(path: Path, spotify_id: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tags = ID3()
        tags.add(
            TXXX(
                encoding=3,
                desc="Spotify Track ID",
                text=[spotify_id],
            )
        )
        tags.save(path, v2_version=3)

    @mock.patch.object(ORCHESTRATOR, "_validate_final_mp3", side_effect=ORCHESTRATOR.embedded_spotify_id)
    def test_moves_both_branches_and_resolves_cross_branch_collision(self, validator) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "runs" / "run-1"
            soulseek_dir = run_dir / "soulseek"
            recorded_dir = run_dir / "recorded_missing"
            first = soulseek_dir / "Artist - Title.mp3"
            second = recorded_dir / "Artist - Title.mp3"
            self.tagged_file(first, "track-one")
            self.tagged_file(second, "track-two")

            playlist_dir, moved, report = ORCHESTRATOR.consolidate_audio(
                run_dir,
                soulseek_dir,
                recorded_dir,
                root / "Playlists_DOWNLOAD",
                'My: Playlist?*',
                expected_spotify_ids={"track-one", "track-two"},
            )

            self.assertEqual(playlist_dir.name, "My Playlist")
            self.assertTrue((playlist_dir / "Artist - Title.mp3").is_file())
            self.assertTrue(
                (playlist_dir / "Artist - Title [track-two].mp3").is_file()
            )
            self.assertFalse(first.exists())
            self.assertFalse(second.exists())
            self.assertEqual(report["audio_count"], 2)
            self.assertEqual(report["moved"], 2)
            self.assertEqual(len(moved), 2)

    @mock.patch.object(ORCHESTRATOR, "_validate_final_mp3", side_effect=ORCHESTRATOR.embedded_spotify_id)
    def test_existing_same_spotify_id_is_deduplicated(self, validator) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "runs" / "run-1"
            source = run_dir / "soulseek" / "Artist - Title.mp3"
            existing = root / "Playlists_DOWNLOAD" / "Playlist" / source.name
            self.tagged_file(source, "same-track")
            self.tagged_file(existing, "same-track")

            _, _, report = ORCHESTRATOR.consolidate_audio(
                run_dir,
                run_dir / "soulseek",
                run_dir / "recorded_missing",
                root / "Playlists_DOWNLOAD",
                "Playlist",
                expected_spotify_ids={"same-track"},
            )

            self.assertFalse(source.exists())
            self.assertTrue(existing.exists())
            self.assertEqual(report["deduplicated"], 1)

    def test_completed_run_keeps_readable_root_and_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            recorded = run_dir / "recorded_missing"
            recorded.mkdir(parents=True)
            (run_dir / "orchestration_report.json").write_text("{}", encoding="utf-8")
            (run_dir / "soulseek_missing.json").write_text("{}", encoding="utf-8")
            (run_dir / "consolidation_report.json").write_text("{}", encoding="utf-8")
            (run_dir / "player_ready.json").write_text("{}", encoding="utf-8")
            (run_dir / "start.signal").write_text("start", encoding="utf-8")
            (recorded / "recording_progress.json").write_text(
                json.dumps({"tracks": []}),
                encoding="utf-8",
            )
            (recorded / "partial.wav").write_bytes(b"partial")

            diagnostics = ORCHESTRATOR.simplify_completed_run(run_dir, {})

            self.assertTrue((run_dir / "orchestration_report.json").is_file())
            self.assertTrue((run_dir / "soulseek_missing.json").is_file())
            self.assertTrue((run_dir / "consolidation_report.json").is_file())
            self.assertTrue((diagnostics / "recording_progress.json").is_file())
            self.assertFalse((run_dir / "player_ready.json").exists())
            self.assertFalse((run_dir / "start.signal").exists())
            self.assertFalse((recorded / "partial.wav").exists())


class ConsolidationSafetyTests(unittest.TestCase):
    def test_id3_only_file_is_blocked_without_moving_or_parsing_before_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "run/soulseek/fake.mp3"
            ConsolidationTests.tagged_file(source, "fixture-track")
            with mock.patch.object(DOWNLOADER, "_scan_antivirus") as scan:
                with self.assertRaises(ORCHESTRATOR.WorkflowError):
                    ORCHESTRATOR.consolidate_audio(root / "run", source.parent, root / "run/recorded", root / "library", "Fixture", {"fixture-track"})
            scan.assert_called_once_with(source)
            self.assertTrue(source.exists())
            self.assertFalse((root / "library").exists())

    def test_all_sources_validate_before_any_move(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "run/soulseek/a.mp3"
            second = root / "run/soulseek/b.mp3"
            ConsolidationTests.tagged_file(first, "first")
            ConsolidationTests.tagged_file(second, "second")
            with mock.patch.object(ORCHESTRATOR, "_validate_final_mp3", side_effect=["first", ORCHESTRATOR.WorkflowError("blocked")]):
                with self.assertRaises(ORCHESTRATOR.WorkflowError):
                    ORCHESTRATOR.consolidate_audio(root / "run", first.parent, root / "run/recorded", root / "library", "Fixture", {"first", "second"})
            self.assertTrue(first.exists() and second.exists())
            self.assertFalse((root / "library").exists())

    def test_invalid_existing_duplicate_does_not_delete_valid_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "run/soulseek/song.mp3"
            existing = root / "library/Fixture/song.mp3"
            ConsolidationTests.tagged_file(source, "fixture-track")
            ConsolidationTests.tagged_file(existing, "fixture-track")
            real_validate = ORCHESTRATOR._validate_final_mp3

            def validate(path):
                return "fixture-track" if path == source else real_validate(path)

            with mock.patch.object(ORCHESTRATOR, "_validate_final_mp3", side_effect=validate), mock.patch.object(DOWNLOADER, "_scan_antivirus"):
                with self.assertRaises(ORCHESTRATOR.WorkflowError):
                    ORCHESTRATOR.consolidate_audio(root / "run", source.parent, root / "run/recorded", root / "library", "Fixture", {"fixture-track"})
            self.assertTrue(source.exists() and existing.exists())

    def test_detection_during_consolidation_blocks_identity_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fixture.mp3"
            ConsolidationTests.tagged_file(source, "fixture-track")
            with mock.patch.object(DOWNLOADER, "_scan_antivirus", side_effect=RuntimeError("blocked")), mock.patch.object(ORCHESTRATOR, "embedded_spotify_id") as parser:
                with self.assertRaises(ORCHESTRATOR.WorkflowError):
                    ORCHESTRATOR._validate_final_mp3(source)
                parser.assert_not_called()


class ResumeTests(unittest.TestCase):
    class FakeSpotify:
        def __init__(self, state: dict) -> None:
            self.state = state

        def current_playback(self) -> dict:
            return self.state

    def test_recorder_checkpoint_validates_completed_files(self) -> None:
        tracks = [
            {
                "playlist_position": 4,
                "duration_seconds": 90.0,
                "spotify": {"spotify_id": "track-a"},
            },
            {
                "playlist_position": 7,
                "duration_seconds": 120.0,
                "spotify": {"spotify_id": "track-b"},
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            json_file = root / "missing.json"
            json_file.write_text("{}", encoding="utf-8")
            output = root / "first.mp3"
            output.write_bytes(b"completed")
            progress_path = root / "recording_progress.json"
            progress = RECORDER.new_progress(json_file, tracks)
            progress["completed_count"] = 1
            progress["completed_track_ids"] = [RECORDER.track_identity(tracks[0])]
            progress["tracks"] = [{"output": str(output)}]
            RECORDER.write_json_atomic(progress_path, progress)

            resumed = RECORDER.load_resume_progress(progress_path, json_file, tracks)

            self.assertEqual(resumed["completed_count"], 1)
            self.assertEqual(resumed["attempt"], 2)
            self.assertEqual(resumed["status"], "recording")

    def test_completed_track_is_checkpointed_immediately(self) -> None:
        track = {
            "playlist_position": 4,
            "duration_seconds": 90.0,
            "spotify": {"spotify_id": "track-a", "title": "Example"},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            progress_path = root / "recording_progress.json"
            progress = RECORDER.new_progress(root / "missing.json", [track])
            boundary = RECORDER.BoundaryResult(
                track_index=0,
                playlist_position=4,
                expected_sec=90.0,
                detected_sec=90.1,
                correction_sec=0.1,
                method="test",
            )
            with (
                mock.patch.object(RECORDER, "encode_segment") as encode,
                mock.patch.object(RECORDER, "write_metadata") as metadata,
            ):
                RECORDER.encode_and_checkpoint(
                    ffmpeg="ffmpeg",
                    temp_wav=root / "capture.wav",
                    output_path=root / "track.mp3",
                    start_sec=0.1,
                    end_sec=90.1,
                    track=track,
                    manifest={"tracks": [track]},
                    global_index=0,
                    total_tracks=1,
                    boundary=boundary,
                    progress=progress,
                    progress_path=progress_path,
                )

            encode.assert_called_once()
            metadata.assert_called_once()
            stored = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(stored["completed_count"], 1)
            self.assertEqual(stored["completed_track_ids"], ["track-a"])

    def test_ffmpeg_can_encode_a_flushed_prefix_while_capture_remains_open(self) -> None:
        try:
            ffmpeg = ORCHESTRATOR.find_required_program("ffmpeg")
        except ORCHESTRATOR.WorkflowError:
            self.skipTest("ffmpeg is not installed")

        rate = 8000
        samples = array(
            "h",
            (
                int(8000 * math.sin(2.0 * math.pi * 440.0 * frame / rate))
                for frame in range(rate)
                for _channel in range(2)
            ),
        ).tobytes()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            wav_path = root / "capture.wav"
            mp3_path = root / "completed.mp3"
            wav_file = wave.open(str(wav_path), "wb")
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(rate)
            wav_file.writeframesraw(samples)
            RECORDER.flush_wave_for_reader(wav_file)

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    RECORDER.encode_segment,
                    str(ffmpeg),
                    wav_path,
                    mp3_path,
                    0.0,
                    0.8,
                )
                wav_file.writeframesraw(samples)
                future.result(timeout=10)
            wav_file.close()

            self.assertTrue(mp3_path.is_file())
            self.assertGreater(mp3_path.stat().st_size, 1000)

    def test_spotify_watchdog_marks_mid_track_pause_as_stalled(self) -> None:
        spotify = self.FakeSpotify(
            {
                "is_playing": False,
                "progress_ms": 30_000,
                "item": {
                    "id": "track-a",
                    "name": "Example",
                    "duration_ms": 90_000,
                },
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "status.json"
            code = PLAYER.monitor_playback_health(spotify, status_path, 0.001)
            status = json.loads(status_path.read_text(encoding="utf-8"))

        self.assertEqual(code, 75)
        self.assertEqual(status["status"], "stalled")

    def test_spotify_watchdog_allows_natural_track_end(self) -> None:
        spotify = self.FakeSpotify(
            {
                "is_playing": False,
                "progress_ms": 89_500,
                "item": {
                    "id": "track-a",
                    "name": "Example",
                    "duration_ms": 90_000,
                },
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "status.json"
            code = PLAYER.monitor_playback_health(spotify, status_path, 0.001)
            status = json.loads(status_path.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(status["status"], "natural_end")


class FallbackRecordingTests(unittest.TestCase):
    """Temas que Soulseek no pudo dar: se graban, se reanudan y, si aun así faltan, se guarda lo que hay."""

    @staticmethod
    def manifest(positions: list[int]) -> dict:
        tracks = [{"playlist_position": p, "spotify": {"spotify_id": f"id{p}"}} for p in positions]
        playlists = [{"url": "https://open.spotify.com/playlist/x", "start_index": 0,
                      "end_index": len(tracks), "track_count": len(tracks)}] if tracks else None
        return {"missing_count": len(tracks), "tracks": tracks, "generated_missing_playlists": playlists}

    @staticmethod
    def write(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def make_run(self, root: Path, stamp: str = "20261030-100000") -> Path:
        run_dir = root / f"{stamp}_PL1"
        run_dir.mkdir(parents=True)
        self.write(run_dir / "orchestration_report.json", {"status": "failed"})
        return run_dir

    @mock.patch.object(ORCHESTRATOR, "_validate_final_mp3", side_effect=ORCHESTRATOR.embedded_spotify_id)
    def test_recorded_failed_tracks_are_consolidated(self, validator) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            ConsolidationTests.tagged_file(run_dir / "soulseek" / "A - One.mp3", "id1")
            ConsolidationTests.tagged_file(run_dir / "recorded_failed" / "A - Two.mp3", "id2")
            args = (run_dir, run_dir / "soulseek", run_dir / "recorded_missing", root / "lib", "Lista")
            with self.assertRaises(ORCHESTRATOR.IncompleteAudio):
                ORCHESTRATOR.consolidate_audio(*args, expected_spotify_ids={"id1", "id2", "id3"},
                                               extra_recorded_dirs=(run_dir / "recorded_failed",))
            self.assertTrue((run_dir / "soulseek" / "A - One.mp3").exists())  # nada se movió
            _, _, report = ORCHESTRATOR.consolidate_audio(*args, expected_spotify_ids={"id1", "id2"},
                                                          extra_recorded_dirs=(run_dir / "recorded_failed",))
            self.assertEqual(report["audio_count"], 2)
            self.assertEqual({r["origin"] for r in report["tracks"]}, {"soulseek", "loopback"})

    @mock.patch.object(ORCHESTRATOR, "_validate_final_mp3", side_effect=ORCHESTRATOR.embedded_spotify_id)
    def test_incomplete_run_saves_what_it_has(self, validator) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            self.write(run_dir / "soulseek_availability.json", {"playlist": {"name": "Lista"}, "tracks": [
                {"spotify": {"spotify_id": "id1"}}, {"spotify": {"spotify_id": "id2"}}]})
            ConsolidationTests.tagged_file(run_dir / "soulseek" / "A - One.mp3", "id1")
            summary_path = run_dir / "orchestration_report.json"
            summary = {"status": "running"}
            with contextlib.redirect_stdout(io.StringIO()):
                code = ORCHESTRATOR.finish_partial(
                    run_dir=run_dir,
                    audio_dirs=(run_dir / "soulseek", run_dir / "recorded_missing", run_dir / "recorded_failed"),
                    playlists_root=root / "lib", playlist_id="PL1", summary=summary,
                    summary_path=summary_path, reason="faltó uno")
            self.assertEqual(code, ORCHESTRATOR.EXIT_PARTIAL)
            self.assertTrue((root / "lib" / "Lista" / "A - One.mp3").is_file())
            saved = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual((saved["status"], saved["missing_spotify_ids"]), ("partial", ["id2"]))
            self.assertTrue(json.loads((run_dir / "consolidation_report.json").read_text(encoding="utf-8"))["partial"])

    def test_incomplete_run_without_audio_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir()
            summary_path = run_dir / "orchestration_report.json"
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ORCHESTRATOR.WorkflowError):
                ORCHESTRATOR.finish_partial(
                    run_dir=run_dir,
                    audio_dirs=(run_dir / "soulseek", run_dir / "recorded_missing", run_dir / "recorded_failed"),
                    playlists_root=Path(temp_dir) / "lib", playlist_id="PL1", summary={},
                    summary_path=summary_path, reason="nada")
            self.assertEqual(json.loads(summary_path.read_text(encoding="utf-8"))["status"], "failed")


class FakeProcess:
    """Proceso simulado: `step()` decide cuándo termina y qué archivos deja."""

    def __init__(self, step):
        self.step = step
        self.returncode = None

    def poll(self):
        if self.returncode is None:
            self.returncode = self.step()
        return self.returncode

    def terminate(self):
        if self.returncode is None:
            self.returncode = 0

    kill = terminate

    def wait(self, timeout=None):
        return self.returncode


class TrackByTrackFlowTests(unittest.TestCase):
    """run_workflow de punta a punta (2026-11-06): el descargador simulado va agregando temas a la cola de
    grabación mientras busca; el grabador y el reproductor simulados graban cada tanda."""

    PID = "PL1"

    @staticmethod
    def track(n: int) -> dict:
        return {"spotify_id": f"id{n}", "title": f"Tema {n}", "artists": ["Artista"]}

    def entry(self, n: int) -> dict:
        return {"playlist_position": n, "duration_ms": 200_000, "duration_seconds": 200.0, "spotify": self.track(n)}

    def setUp(self) -> None:
        self.events: list[tuple] = []
        self.player_failures = 0  # reproductores que terminan sin empezar a sonar (Spotify que no aparece)
        self.player_timeouts = 0  # reproductores que no avisan a tiempo (pudieron empezar a sonar)
        self.output_checks: list = []  # resultados de comprobar la salida de audio (vacío: libre)

    def popen(self, script, code, recorder_ops):
        calls = []

        def fake(cmd, env=None):
            name = Path(cmd[2]).name
            arg = lambda flag: Path(cmd[cmd.index(flag) + 1])  # noqa: E731
            calls.append((name, cmd))
            if name == "spotify_soulseek_download.py":
                return self.downloader(cmd, arg, list(script), code)
            if name == "playlist_loopback_recorder.py":
                return self.recorder(cmd, arg, recorder_ops)
            self.events.append(("player", cmd[3], int(cmd[cmd.index("--offset-position") + 1])))
            if self.player_failures:
                self.player_failures -= 1
                return FakeProcess(lambda: 1)
            if self.player_timeouts:
                self.player_timeouts -= 1
                write_json(arg("--ready-file"), {"status": "starting", "device_id": "dev1"})
                return FakeProcess(lambda: None)
            write_json(arg("--ready-file"), {"status": "ready", "device_id": "dev1"})  # reproductor
            return FakeProcess(lambda: None)
        return fake, calls

    def downloader(self, cmd, arg, steps, code):
        queue, out = arg("--recording-queue"), arg("--output")
        have = arg("--have-dir") if "--have-dir" in cmd else None
        polls = [0]

        def step():
            polls[0] += 1
            assert polls[0] < 500, "el orquestador no terminó"
            if not steps:
                return code
            for op in steps.pop(0):
                kind = op[0]
                if kind == "availability":
                    write_json(arg("--availability-report"), {"playlist": {"name": "Lista"}, "tracks": [
                        {"position": n, "spotify": self.track(n), "status": "available"} for n in op[1]]})
                elif kind == "download":
                    if not (have and (have / f"Artista - Tema {op[1]}.mp3").exists()):  # como el real
                        ConsolidationTests.tagged_file(out / f"Artista - Tema {op[1]}.mp3", f"id{op[1]}")
                elif kind == "queue":
                    QUEUE.append_track(queue, self.entry(op[1]), origin=op[2] if len(op) > 2 else "missing", reason="x")
                elif kind == "close":
                    QUEUE.close_queue(queue, op[1], op[2] if len(op) > 2 else None)
                self.events.append(("downloader", kind, op[1] if len(op) > 1 else None))
            return None
        return FakeProcess(step)

    def recorder(self, cmd, arg, recorder_ops):
        out, watchdog = arg("--output-dir"), arg("--watchdog-file")
        tracks = json.loads(Path(cmd[3]).read_text(encoding="utf-8"))["tracks"]
        progress_path = out / "recording_progress.json"
        progress = (json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists()
                    else {"completed_count": 0, "completed_track_ids": [], "tracks": []})
        progress["total_tracks"] = len(tracks)
        self.events.append(("recorder", out.name, progress["completed_count"]))
        ops = list(recorder_ops.pop(out.name, None) or ["record"] * len(tracks))
        if ops and ops[0] == "busy":  # la salida se ocupó entre la comprobación y el grabador
            progress.update(status="device_unavailable", last_error="No se puede grabar la salida «DJM» (ocupada).")
            write_json(progress_path, progress)
            return FakeProcess(lambda: ORCHESTRATOR.DEVICE_UNAVAILABLE_EXIT_CODE)
        write_json(arg("--ready-file"), {"status": "ready"})

        def record_one():
            t = tracks[progress["completed_count"]]
            mp3 = out / f"Artista - {t['spotify']['title']}.mp3"
            ConsolidationTests.tagged_file(mp3, t["spotify"]["spotify_id"])
            progress["completed_count"] += 1
            progress["completed_track_ids"].append(t["spotify"]["spotify_id"])
            progress["tracks"].append({"playlist_position": t["playlist_position"], "output": str(mp3)})

        def step():
            if not (out.parent.parent / "player_ready.json").exists():  # out = <run>/recordings/NN
                return None  # como el real: sin Spotify sonando no hay nada que grabar
            if watchdog.exists() and "stalled" in watchdog.read_text(encoding="utf-8"):
                progress.update(status="stalled", last_error="Spotify dejó de producir audio.")
                write_json(progress_path, progress)
                return 75
            op = ops.pop(0) if ops else "record"
            if op == "record":
                record_one()
            elif op == "stall":
                progress.update(status="stalled", last_error="Spotify dejó de producir audio.")
                write_json(progress_path, progress)
                return 75
            elif op == "defer":  # la app pospone la grabación mientras graba
                (out.parent.parent / ORCHESTRATOR.DEFER_SIGNAL).write_text("1", encoding="utf-8")
            write_json(progress_path, progress)
            return 0 if progress["completed_count"] >= len(tracks) else None
        return FakeProcess(step)

    def run_flow(self, root: Path, script=(), code=0, *, defer=False, recorder_ops=None, playlist_error=None,
                 recorder_timeout=30.0, recording_output=""):
        fake, calls = self.popen(script, code, dict(recorder_ops or {}))
        pauses: list[str] = []
        playlists: list[list[str]] = []
        deleted: list[str] = []

        def check_output(env, output=None):
            self.events.append(("check",) if output is None else ("check", output))
            if self.output_checks:
                return self.output_checks.pop(0)
            return True, "CABLE Input"

        def create_playlist(source, tracks, name):
            if playlist_error:
                raise RuntimeError(playlist_error)
            playlists.append([t["spotify"]["spotify_id"] for t in tracks])
            return {"id": f"s{len(playlists)}", "url": f"https://open.spotify.com/playlist/s{len(playlists)}",
                    "name": name}

        args = argparse.Namespace(playlist_url=f"https://open.spotify.com/playlist/{self.PID}",
                                  output_root=root / "runs", library_root=root / "lib",
                                  recorder_timeout=recorder_timeout, fresh=False, defer_recording=defer,
                                  parallel=2, recording_output=recording_output)
        with mock.patch.object(ORCHESTRATOR.subprocess, "Popen", side_effect=fake), \
                mock.patch.object(ORCHESTRATOR, "validate_environment"), \
                mock.patch.object(ORCHESTRATOR, "find_required_program", return_value=root / "tool.exe"), \
                mock.patch.object(ORCHESTRATOR, "_validate_final_mp3", side_effect=ORCHESTRATOR.embedded_spotify_id), \
                mock.patch.object(ORCHESTRATOR, "create_spotify_playlist", side_effect=create_playlist), \
                mock.patch.object(ORCHESTRATOR, "delete_spotify_playlist",
                                  side_effect=lambda playlist: deleted.append(playlist["id"])), \
                mock.patch.object(ORCHESTRATOR, "check_recording_device", side_effect=check_output), \
                mock.patch.object(ORCHESTRATOR, "DEVICE_RETRY_S", 0.0), \
                mock.patch.object(ORCHESTRATOR, "pause_spotify",
                                  side_effect=lambda env, device: pauses.append(device) or self.events.append(("pause", device))), \
                mock.patch.object(ORCHESTRATOR.time, "sleep"), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = ORCHESTRATOR.run_workflow(args)
        run_dir = sorted((root / "runs").iterdir())[-1]
        report = json.loads((run_dir / "orchestration_report.json").read_text(encoding="utf-8"))
        return code, calls, run_dir, report, {"pauses": pauses, "playlists": playlists, "deleted": deleted}

    def library(self, root: Path) -> list[str]:
        return sorted(p.name for p in (root / "lib" / "Lista").glob("*.mp3"))

    @staticmethod
    def scripts(calls, name: str) -> int:
        return sum(script == name for script, _ in calls)

    ALL3 = ["Artista - Tema 1.mp3", "Artista - Tema 2.mp3", "Artista - Tema 3.mp3"]

    def test_recording_starts_with_the_first_queued_track_and_waits_for_the_next(self) -> None:
        script = [
            [("availability", [1, 2, 3]), ("download", 1), ("queue", 2)],
            [],                      # la tanda 01 ya terminó y la cola está vacía, pero la búsqueda sigue
            [("queue", 3, "failed")],  # entra un tema nuevo después: otra tanda
            [("close", "finished")],
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            code, calls, run_dir, report, spotify = self.run_flow(root, script)
            self.assertEqual((code, report["status"]), (0, "completed"))
            self.assertEqual(self.library(root), self.ALL3)
            self.assertEqual(spotify["playlists"], [["id2"], ["id3"]])  # una playlist por tanda
            self.assertEqual(spotify["deleted"], ["s1", "s2"])  # cada una se borra al terminar su tanda
            order = [e[:2] for e in self.events if e[0] in {"recorder", "downloader"}]
            # Se grabó el primero antes de que la búsqueda siguiera, y el segundo apenas entró.
            self.assertLess(order.index(("recorder", "01")), order.index(("downloader", "queue")) + 3)
            self.assertLess(order.index(("recorder", "01")), order.index(("downloader", "close")))
            self.assertLess(order.index(("recorder", "02")), order.index(("downloader", "close")))
            self.assertIn("dev1", spotify["pauses"])  # sin nada para grabar, Spotify no sigue sonando
            m3u = (root / "lib" / "Lista" / "Lista.m3u8").read_text(encoding="utf-8").splitlines()
            self.assertEqual([Path(line).name for line in m3u[2::2]], self.ALL3)
            self.assertTrue((run_dir / "diagnostics" / "recordings" / "01" / "session.json").is_file())
            downloader = next(cmd for name, cmd in calls if name == "spotify_soulseek_download.py")
            self.assertIn("--recording-queue", downloader)
            self.assertEqual(downloader[downloader.index("--parallel") + 1], "2")

    def test_postponed_from_the_start_saves_soulseek_and_records_later(self) -> None:
        script = [[("availability", [1, 2, 3]), ("download", 1), ("queue", 2)], [("queue", 3), ("close", "finished")]]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            code, calls, run_dir, report, _ = self.run_flow(root, script, defer=True)
            self.assertEqual((code, report["status"], report["pending_recordings"]),
                             (ORCHESTRATOR.EXIT_RESUMABLE, "recording_deferred", 2))
            self.assertEqual(self.scripts(calls, "playlist_loopback_recorder.py"), 0)
            self.assertEqual(self.library(root), ["Artista - Tema 1.mp3"])  # lo de Soulseek, ya en la carpeta
            self.assertEqual(ORCHESTRATOR.resume_point(root / "runs", self.PID), run_dir)

            code, calls, _, report, spotify = self.run_flow(root)  # «Grabar ahora»
            self.assertEqual((code, report["status"]), (0, "completed"))
            self.assertEqual(self.scripts(calls, "spotify_soulseek_download.py"), 0)  # ni busca ni baja
            self.assertEqual(spotify["playlists"], [["id2", "id3"]])
            self.assertEqual(self.library(root), self.ALL3)

    def test_postponing_during_a_session_stops_it_and_keeps_what_was_recorded(self) -> None:
        script = [[("availability", [1, 2, 3]), ("download", 1), ("queue", 2), ("queue", 3)], [], [("close", "finished")]]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            code, _, run_dir, report, spotify = self.run_flow(root, script, recorder_ops={"01": ["record", "defer", "record"]})
            self.assertEqual((code, report["status"], report["pending_recordings"]),
                             (ORCHESTRATOR.EXIT_RESUMABLE, "recording_deferred", 1))
            self.assertIn("dev1", spotify["pauses"])
            self.assertEqual(spotify["deleted"], [])  # la tanda quedó a medias: su playlist hace falta para seguir
            status = json.loads((run_dir / "spotify_playback_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["reason"], "deferred")
            self.assertTrue((run_dir / "recordings" / "01" / "Artista - Tema 2.mp3").is_file())  # se conserva

            (run_dir / ORCHESTRATOR.DEFER_SIGNAL).unlink()
            self.events.clear()
            code, _, _, report, spotify = self.run_flow(root)
            self.assertEqual((code, report["status"]), (0, "completed"))
            self.assertIn(("player", "https://open.spotify.com/playlist/s1", 1), self.events)  # sigue en el 2.º
            self.assertEqual(spotify["deleted"], ["s1"])
            self.assertEqual(self.library(root), self.ALL3)

    def test_spotify_stopping_stops_the_recording_but_not_the_download(self) -> None:
        script = [
            [("availability", [1, 2, 3, 4]), ("download", 1), ("queue", 2)],
            [("queue", 3)],
            [("download", 4), ("close", "finished")],
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            code, calls, _, report, _ = self.run_flow(root, script, recorder_ops={"01": ["stall"]})
            self.assertEqual((code, report["status"], report["pending_recordings"]),
                             (ORCHESTRATOR.EXIT_RESUMABLE, "resumable", 2))
            self.assertIn("Spotify dejó de producir audio", report["reason"])
            self.assertEqual(self.scripts(calls, "playlist_loopback_recorder.py"), 1)  # no se insiste
            self.assertEqual(self.library(root), ["Artista - Tema 1.mp3", "Artista - Tema 4.mp3"])

            code, calls, _, report, _ = self.run_flow(root)  # «Continuar»
            self.assertEqual((code, report["status"]), (0, "completed"))
            self.assertEqual(self.scripts(calls, "spotify_soulseek_download.py"), 0)
            self.assertEqual(self.library(root), self.ALL3 + ["Artista - Tema 4.mp3"])

    def test_a_session_playlist_that_cannot_be_created_leaves_the_recording_pending(self) -> None:
        script = [[("availability", [1, 2]), ("download", 1), ("queue", 2), ("close", "finished")]]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            code, calls, _, report, _ = self.run_flow(root, script, playlist_error="Spotify 429")
            self.assertEqual((code, report["status"]), (ORCHESTRATOR.EXIT_RESUMABLE, "resumable"))
            self.assertIn("Spotify 429", report["reason"])
            self.assertEqual(self.scripts(calls, "playlist_loopback_recorder.py"), 0)
            self.assertEqual(self.library(root), ["Artista - Tema 1.mp3"])

    def test_spotify_slow_to_appear_gets_a_second_try(self) -> None:
        script = [[("availability", [1, 2]), ("download", 1), ("queue", 2), ("close", "finished")]]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.player_failures = 1
            code, _, _, report, _ = self.run_flow(root, script)
            self.assertEqual((code, report["status"]), (0, "completed"))
            self.assertEqual(sum(e[0] == "player" for e in self.events), 2)

    def test_a_session_whose_playlist_does_not_play_gets_a_new_one_next_time(self) -> None:
        script = [[("availability", [1, 2]), ("download", 1), ("queue", 2), ("close", "finished")]]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.player_failures = 2
            code, _, run_dir, report, _ = self.run_flow(root, script)
            self.assertEqual((code, report["status"]), (ORCHESTRATOR.EXIT_RESUMABLE, "resumable"))
            self.assertIn("reproductor de Spotify", report["reason"])
            session = json.loads((run_dir / "recordings" / "01" / "session.json").read_text(encoding="utf-8"))
            self.assertIsNone(session["playlist"])
            code, _, _, report, spotify = self.run_flow(root)  # «Continuar»: playlist nueva para la tanda
            self.assertEqual((code, report["status"], spotify["playlists"]), (0, "completed", [["id2"]]))

    def test_a_playlist_that_cannot_be_removed_stays_noted_and_is_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir) / "recordings" / "01"
            write_json(session / "session.json", {"tracks": [], "playlist": {"id": "s9", "name": "Grabar - Lista"}})
            with (
                mock.patch.object(ORCHESTRATOR, "delete_spotify_playlist", side_effect=RuntimeError("403")),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                ORCHESTRATOR.forget_session_playlist(session)  # no corta nada
            self.assertIn("403", json.loads((session / "session.json").read_text(encoding="utf-8"))["playlist_delete_error"])
            with (
                mock.patch.object(ORCHESTRATOR, "delete_spotify_playlist") as delete,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                ORCHESTRATOR.forget_session_playlist(session)
                ORCHESTRATOR.forget_session_playlist(session)  # ya borrada: no se pide de nuevo
            delete.assert_called_once_with({"id": "s9", "name": "Grabar - Lista"})
            stored = json.loads((session / "session.json").read_text(encoding="utf-8"))
            self.assertTrue(stored["playlist_deleted"] and "playlist_delete_error" not in stored)

    def test_a_busy_output_waits_while_the_download_goes_on_and_records_when_free(self) -> None:
        # 2026-11-06: Traktor tenía el DJM (la salida predeterminada) y el grabador cayó con «Invalid device».
        busy = (False, "No se puede grabar la salida de audio «Line (DJM-750MK2)»: otro programa la usa.")
        script = [[("availability", [1, 2, 3]), ("download", 1), ("queue", 2)], [], [("download", 3)],
                  [("close", "finished")]]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.output_checks = [busy, busy, busy]
            code, _, _, report, spotify = self.run_flow(root, script)
            self.assertEqual((code, report["status"]), (0, "completed"))
            checks = [i for i, e in enumerate(self.events) if e == ("check",)]
            recorder = next(i for i, e in enumerate(self.events) if e[0] == "recorder")
            self.assertGreaterEqual(len([i for i in checks if i < recorder]), 4)  # esperó y volvió a probar
            self.assertEqual(spotify["playlists"], [["id2"]])  # ninguna playlist mientras no se podía grabar
            self.assertEqual(self.library(root), self.ALL3)

    def test_an_output_busy_until_the_end_leaves_the_recording_pending_with_the_reason(self) -> None:
        busy = (False, "No se puede grabar la salida de audio «Line (DJM-750MK2)»: otro programa la usa.")
        script = [[("availability", [1, 2]), ("download", 1), ("queue", 2), ("close", "finished")]]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.output_checks = [busy] * 50
            code, calls, _, report, spotify = self.run_flow(root, script)
            self.assertEqual((code, report["status"]), (ORCHESTRATOR.EXIT_RESUMABLE, "resumable"))
            self.assertIn("DJM-750MK2", report["reason"])
            self.assertEqual((spotify["playlists"], self.scripts(calls, "playlist_loopback_recorder.py")), ([], 0))
            self.assertEqual(self.library(root), ["Artista - Tema 1.mp3"])  # lo de Soulseek, ya guardado

    def test_an_output_taken_during_a_session_is_waited_for_not_given_up(self) -> None:
        script = [[("availability", [1, 2]), ("download", 1), ("queue", 2)], [], [("close", "finished")]]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            code, _, _, report, _ = self.run_flow(root, script, recorder_ops={"01": ["busy"]})
            self.assertEqual((code, report["status"]), (0, "completed"))
            self.assertEqual([e for e in self.events if e[0] == "recorder"], [("recorder", "01", 0), ("recorder", "01", 0)])
            self.assertEqual(self.library(root), ["Artista - Tema 1.mp3", "Artista - Tema 2.mp3"])

    def test_the_chosen_output_is_what_is_checked_and_recorded(self) -> None:
        script = [[("availability", [1, 2]), ("download", 1), ("queue", 2), ("close", "finished")]]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            code, calls, _, report, _ = self.run_flow(root, script, recording_output="Voicemeeter Input")
            self.assertEqual((code, report["status"]), (0, "completed"))
            self.assertIn(("check", "Voicemeeter Input"), self.events)
            recorder = next(cmd for name, cmd in calls if name == "playlist_loopback_recorder.py")
            self.assertEqual(recorder[recorder.index("--output-device") + 1], "Voicemeeter Input")

    def test_spotify_is_paused_before_the_next_session_starts_recording(self) -> None:
        # 12 en la cola: tanda de 10 y tanda de 2, una detrás de la otra. La reproducción automática que
        # sigue a la primera no puede ser lo que el grabador de la segunda toma como su comienzo.
        script = [[("availability", list(range(1, 13)))] + [("queue", n) for n in range(1, 13)]
                  + [("close", "finished")]]
        with tempfile.TemporaryDirectory() as temp_dir:
            code, _, _, report, spotify = self.run_flow(Path(temp_dir), script)
            self.assertEqual((code, report["status"]), (0, "completed"))
            self.assertEqual([len(p) for p in spotify["playlists"]], [10, 2])
            order = [e for e in self.events if e[0] in {"recorder", "pause"}]
            second = order.index(("recorder", "02", 0))
            self.assertIn(("pause", "dev1"), order[:second])

    def test_a_player_that_times_out_is_not_retried_and_spotify_is_paused(self) -> None:
        script = [[("availability", [1, 2]), ("download", 1), ("queue", 2), ("close", "finished")]]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.player_timeouts = 1
            code, _, _, report, spotify = self.run_flow(root, script, recorder_timeout=0.01)
            self.assertEqual((code, report["status"]), (ORCHESTRATOR.EXIT_RESUMABLE, "resumable"))
            # Pudo empezar a sonar y el grabador habría tomado ese comienzo: otro intento correría los cortes.
            self.assertEqual(sum(e[0] == "player" for e in self.events), 1)
            self.assertIn("dev1", spotify["pauses"])

    def test_a_run_reached_by_another_path_keeps_its_recordings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = self.interrupted_session(root, output_root=Path("D:/otra_carpeta/runs/x/recordings/01"))
            code, _, _, report, spotify = self.run_flow(root)
            self.assertEqual((code, report["status"]), (0, "completed"))
            self.assertEqual(spotify["playlists"], [])  # se retomó la misma tanda, con su playlist
            self.assertIn(("player", "https://open.spotify.com/playlist/old", 1), self.events)
            self.assertEqual(self.library(root), ["Artista - Tema 1.mp3", "Artista - Tema 2.mp3"])
            self.assertFalse((run_dir / "recordings").exists())

    def test_an_empty_session_folder_left_by_a_cut_does_not_block_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = self.interrupted_session(root, session=False)
            (run_dir / "recordings" / "01").mkdir(parents=True)
            code, _, _, report, spotify = self.run_flow(root)
            self.assertEqual((code, report["status"]), (0, "completed"))
            self.assertEqual(spotify["playlists"], [["id1", "id2"]])
            self.assertIn(("recorder", "02", 0), self.events)

    def test_a_failure_while_archiving_does_not_undo_the_saved_playlist(self) -> None:
        script = [[("availability", [1]), ("download", 1), ("close", "finished")]]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with mock.patch.object(ORCHESTRATOR, "simplify_completed_run", side_effect=PermissionError("en uso")):
                code, _, run_dir, report, _ = self.run_flow(root, script)
            self.assertEqual((code, report["status"]), (0, "completed"))
            self.assertIsNone(ORCHESTRATOR.resume_point(root / "runs", self.PID))  # no se retoma a medio archivar
            self.assertEqual(self.library(root), ["Artista - Tema 1.mp3"])

    def interrupted_session(self, root: Path, *, output_root: Path | None = None, session: bool = True) -> Path:
        """Run tema a tema cortado con la cola cerrada (1 y 2 para grabar) y, si `session`, la tanda 01 con el
        1 ya grabado; `output_root`: la carpeta de la tanda según el checkpoint (otra si el run se movió)."""
        run_dir = root / "runs" / f"20261106-100000_{self.PID}"
        write_json(run_dir / "orchestration_report.json", {"layout": ORCHESTRATOR.LAYOUT, "status": "interrupted"})
        write_json(run_dir / "soulseek_availability.json", {"playlist": {"name": "Lista"}, "tracks": [
            {"position": n, "spotify": self.track(n), "status": "missing"} for n in (1, 2)]})
        for n in (1, 2):
            QUEUE.append_track(run_dir / QUEUE.QUEUE_NAME, self.entry(n), origin="missing", reason="x")
        QUEUE.close_queue(run_dir / QUEUE.QUEUE_NAME, "finished")
        if session:
            folder = run_dir / "recordings" / "01"
            write_json(folder / "session.json", {"tracks": [self.entry(1), self.entry(2)],
                                                 "playlist": {"url": "https://open.spotify.com/playlist/old"}})
            ConsolidationTests.tagged_file(folder / "Artista - Tema 1.mp3", "id1")
            write_json(folder / "recording_progress.json", {
                "completed_count": 1, "completed_track_ids": ["id1"], "total_tracks": 2,
                "tracks": [{"playlist_position": 1, "output": str((output_root or folder) / "Artista - Tema 1.mp3")}]})
        return run_dir

    def test_a_broken_search_records_what_was_queued_and_saves_a_partial_playlist(self) -> None:
        script = [[("availability", [1, 2, 3]), ("queue", 1)], [("close", "search_failed", "timeout")]]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            code, _, _, report, _ = self.run_flow(root, script, code=4)
            self.assertEqual((code, report["status"]), (ORCHESTRATOR.EXIT_PARTIAL, "partial"))
            self.assertIn("dejó de funcionar (timeout)", report["reason"])
            self.assertEqual(report["missing_spotify_ids"], ["id2", "id3"])
            self.assertEqual(self.library(root), ["Artista - Tema 1.mp3"])  # el que estaba en la cola se grabó

    def test_a_search_that_never_worked_records_nothing(self) -> None:
        script = [[("availability", [1, 2]), ("close", "search_failed", "timeout")]]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(ORCHESTRATOR.WorkflowError, "dejó de funcionar"):
                self.run_flow(root, script, code=4)
            self.assertFalse(any(e[0] == "recorder" for e in self.events))

    def test_a_downloader_that_dies_saves_what_there_is(self) -> None:
        script = [[("availability", [1, 2]), ("download", 1)]]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            code, _, _, report, _ = self.run_flow(root, script, code=1)
            self.assertEqual((code, report["status"]), (ORCHESTRATOR.EXIT_PARTIAL, "partial"))
            self.assertIn("se cortó antes de terminar (código 1)", report["reason"])
            self.assertEqual(self.library(root), ["Artista - Tema 1.mp3"])

    def test_tracks_already_in_the_music_folder_count_as_saved(self) -> None:
        script = [[("availability", [1, 2, 3]), ("download", 1), ("download", 2), ("queue", 3), ("close", "finished")]]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            earlier = root / "runs" / f"20200101-000000_{self.PID}"
            folder = root / "lib" / "Lista"
            ConsolidationTests.tagged_file(folder / "Artista - Tema 1.mp3", "id1")
            write_json(earlier / "orchestration_report.json", {"status": "completed", "final_output": str(folder)})
            code, calls, _, report, _ = self.run_flow(root, script)
            downloader = next(cmd for name, cmd in calls if name == "spotify_soulseek_download.py")
            self.assertEqual(Path(downloader[downloader.index("--have-dir") + 1]), folder.resolve())
            self.assertEqual((code, report["status"], report["saved_spotify_ids"]), (0, "completed", ["id1"]))
            self.assertEqual(self.library(root), self.ALL3)

    def test_earlier_unsaved_downloads_and_recordings_are_offered_for_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            earlier = root / "runs" / f"20200101-000000_{self.PID}"
            ConsolidationTests.tagged_file(earlier / "soulseek" / "Artista - Tema 1.mp3", "id1")
            ConsolidationTests.tagged_file(earlier / "recorded_missing" / "Artista - Tema 2.mp3", "id2")
            write_json(earlier / "orchestration_report.json", {"status": "resumable"})  # de antes del cambio
            self.assertIsNone(ORCHESTRATOR.resume_point(root / "runs", self.PID))      # no se retoma: se reutiliza
            _, calls, _, _, _ = self.run_flow(root, [[("availability", [1]), ("download", 1), ("close", "finished")]])
            downloader = next(cmd for name, cmd in calls if name == "spotify_soulseek_download.py")
            reused = [Path(downloader[i + 1]).name for i, part in enumerate(downloader) if part == "--reuse-dir"]
            self.assertEqual(reused, ["soulseek", "recorded_missing"])

    def test_finished_runs_are_not_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runs = Path(temp_dir)
            run_dir = runs / f"20261106-100000_{self.PID}"
            for status, resumed in (("interrupted", True), ("resumable", True), ("recording_deferred", True),
                                    ("running", True), ("completed", False), ("partial", False), ("failed", False)):
                write_json(run_dir / "orchestration_report.json", {"layout": ORCHESTRATOR.LAYOUT, "status": status})
                self.assertEqual(ORCHESTRATOR.resume_point(runs, self.PID) == run_dir, resumed, status)

    def test_a_session_whose_recordings_were_moved_goes_back_to_the_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "runs" / f"20261106-100000_{self.PID}"
            write_json(run_dir / "orchestration_report.json", {"layout": ORCHESTRATOR.LAYOUT, "status": "interrupted"})
            write_json(run_dir / "soulseek_availability.json", {"playlist": {"name": "Lista"}, "tracks": [
                {"position": n, "spotify": self.track(n), "status": "missing"} for n in (1, 2)]})
            for n in (1, 2):
                QUEUE.append_track(run_dir / QUEUE.QUEUE_NAME, self.entry(n), origin="missing", reason="x")
            QUEUE.close_queue(run_dir / QUEUE.QUEUE_NAME, "finished")
            session = run_dir / "recordings" / "01"
            write_json(session / "session.json", {"tracks": [self.entry(1), self.entry(2)],
                                                  "playlist": {"url": "https://open.spotify.com/playlist/old"}})
            # «Guardar los temas listos» de la app ya llevó el 1 a la carpeta.
            ConsolidationTests.tagged_file(root / "lib" / "Lista" / "Artista - Tema 1.mp3", "id1")
            write_json(session / "recording_progress.json", {
                "completed_count": 1, "completed_track_ids": ["id1"], "total_tracks": 2,
                "tracks": [{"playlist_position": 1, "output": str(session / "Artista - Tema 1.mp3")}]})
            code, _, _, report, spotify = self.run_flow(root)
            self.assertEqual((code, report["status"]), (0, "completed"))
            self.assertEqual(spotify["playlists"], [["id2"]])  # una tanda nueva solo con lo que faltaba
            self.assertEqual(self.library(root), ["Artista - Tema 1.mp3", "Artista - Tema 2.mp3"])


class OutputCheckTests(unittest.TestCase):
    class SilentStream:
        """Un loopback por el que no suena nada: no entrega datos (una lectura bloqueante esperaría siempre)."""

        def get_read_available(self):
            return 0

        def read(self, frames, exception_on_overflow=False):
            raise AssertionError("no se lee sin datos: se bloquearía")

        def stop_stream(self):
            pass

        def close(self):
            pass

    class FakeAudio:
        LOOPBACKS = [
            {"name": "Line (DJM-750MK2) [Loopback]", "index": 23, "maxInputChannels": 2, "defaultSampleRate": 48000.0},
            {"name": "Voicemeeter Input (VB-Audio Voicemeeter VAIO) [Loopback]", "index": 31, "maxInputChannels": 2,
             "defaultSampleRate": 48000.0},
        ]

        def __init__(self, fail_open=False, stream=None):
            self.fail_open = fail_open
            self.stream = stream
            self.terminated = False
            self.opened: list[int] = []

        def get_default_wasapi_loopback(self):
            return self.LOOPBACKS[0]

        def get_loopback_device_info_generator(self):
            yield from self.LOOPBACKS

        def get_sample_size(self, fmt):
            return 2

        def open(self, **kwargs):
            if self.fail_open:
                raise OSError(-9996, "Invalid device")
            self.opened.append(kwargs["input_device_index"])
            return self.stream or mock.Mock()

        def terminate(self):
            self.terminated = True

    def test_an_output_is_chosen_by_its_name_or_part_of_it(self) -> None:
        audio = self.FakeAudio()
        self.assertEqual(RECORDER.find_output_loopback(audio, "Voicemeeter Input")["index"], 31)
        self.assertEqual(RECORDER.find_output_loopback(audio, "voicemeeter input (vb-audio voicemeeter vaio)")["index"], 31)
        self.assertEqual(RECORDER.find_output_loopback(audio, None)["index"], 23)  # la predeterminada
        with self.assertRaises(LookupError):
            RECORDER.find_output_loopback(audio, "CABLE Input")
        with mock.patch.object(RECORDER.pyaudio, "PyAudio", return_value=self.FakeAudio()):
            missing = RECORDER.check_device("CABLE Input")
            found = RECORDER.check_device("Voicemeeter Input")
        self.assertFalse(missing["ok"])
        self.assertIn("No encuentro la salida de audio «CABLE Input»", missing["error"])
        self.assertEqual(found, {"ok": True, "device": "Voicemeeter Input (VB-Audio Voicemeeter VAIO)"})

    def test_a_silent_output_ends_the_recording_with_the_reason_instead_of_waiting_forever(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "session.json"
            manifest.write_text(json.dumps({"tracks": [{"playlist_position": 1, "duration_seconds": 120.0,
                                                        "spotify": RecordedMetadataTests.spotify_report()}]}),
                                encoding="utf-8")
            audio = self.FakeAudio(stream=self.SilentStream())
            argv = ["recorder", str(manifest), "--output-dir", str(root / "out"), "--output-device", "Voicemeeter",
                    "--start-timeout", "0.3"]
            with mock.patch.object(RECORDER.pyaudio, "PyAudio", return_value=audio), \
                    mock.patch.object(RECORDER, "check_ffmpeg", return_value="ffmpeg"), \
                    mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = RECORDER.main()
            self.assertEqual(code, RECORDER.STALL_EXIT_CODE)
            self.assertEqual(audio.opened, [31])  # grabó la salida elegida, no la predeterminada
            progress = json.loads((root / "out" / "recording_progress.json").read_text(encoding="utf-8"))
            self.assertIn("No llegó sonido a «Voicemeeter Input (VB-Audio Voicemeeter VAIO)»", progress["last_error"])

    def test_an_output_held_by_another_program_is_explained(self) -> None:
        audio = self.FakeAudio(fail_open=True)
        with mock.patch.object(RECORDER.pyaudio, "PyAudio", return_value=audio):
            result = RECORDER.check_device()
        self.assertFalse(result["ok"])
        self.assertIn("«Line (DJM-750MK2)»", result["error"])
        self.assertIn("Traktor", result["error"])
        self.assertTrue(audio.terminated)
        with mock.patch.object(RECORDER.pyaudio, "PyAudio", return_value=self.FakeAudio(fail_open=False)):
            self.assertEqual(RECORDER.check_device(), {"ok": True, "device": "Line (DJM-750MK2)"})

    def test_check_device_answers_in_one_line_and_its_exit_code(self) -> None:
        out = io.StringIO()
        with mock.patch.object(RECORDER.pyaudio, "PyAudio", return_value=self.FakeAudio(fail_open=True)), \
                mock.patch.object(sys, "argv", ["recorder", "--check-device"]), contextlib.redirect_stdout(out):
            code = RECORDER.main()
        self.assertEqual(code, RECORDER.DEVICE_UNAVAILABLE_EXIT_CODE)
        self.assertEqual(code, ORCHESTRATOR.DEVICE_UNAVAILABLE_EXIT_CODE)
        self.assertFalse(json.loads(out.getvalue().strip())["ok"])


class EnvironmentSetupTests(unittest.TestCase):
    def test_env_file_sets_default_output_roots(self) -> None:
        def fake_load_environment() -> None:
            os.environ["SPOTIFY_DOWNLOAD_ROOT"] = "D:/descargas/runs"
            os.environ["SPOTIFY_PLAYLISTS_ROOT"] = "D:/descargas/playlists"

        received: list[object] = []
        with mock.patch.dict(os.environ), mock.patch.object(
            ORCHESTRATOR, "load_environment", fake_load_environment
        ), mock.patch.object(
            ORCHESTRATOR, "run_workflow", lambda args: received.append(args) or 0
        ), mock.patch.object(
            sys, "argv", ["orchestrator", "https://open.spotify.com/playlist/abc"]
        ):
            self.assertEqual(ORCHESTRATOR.main(), 0)

        self.assertEqual(received[0].output_root, Path("D:/descargas/runs"))
        self.assertEqual(received[0].library_root, Path("D:/descargas/playlists"))

    def test_player_token_cache_ignores_working_directory(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        self.assertEqual(PLAYER.TOKEN_CACHE, repo_root / ".spotify_token_cache")

    def test_pause_only_touches_this_computer(self) -> None:
        spotify = mock.Mock()
        spotify.current_playback.return_value = {"is_playing": True, "device": {"id": "phone"}}
        self.assertFalse(PLAYER.pause_device(spotify, "dev1"))
        spotify.current_playback.return_value = {"is_playing": True, "device": {"id": "dev1"}}
        self.assertTrue(PLAYER.pause_device(spotify, "dev1"))
        spotify.pause_playback.assert_called_once_with(device_id="dev1")


if __name__ == "__main__":
    unittest.main()
