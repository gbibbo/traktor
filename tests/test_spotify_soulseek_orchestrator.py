"""PURPOSE: Test Spotify/Soulseek synchronization, metadata and consolidation.

CHANGELOG:
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


class ManifestHandshakeTests(unittest.TestCase):
    def write_payload(self, payload: dict) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "missing.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_zero_missing_is_final_without_generated_playlist(self) -> None:
        path = self.write_payload(
            {
                "missing_count": 0,
                "tracks": [],
                "generated_missing_playlist": None,
            }
        )
        self.assertIsNotNone(ORCHESTRATOR.final_missing_manifest(path))

    def test_first_write_without_playlist_url_is_not_final(self) -> None:
        path = self.write_payload(
            {
                "missing_count": 1,
                "tracks": [{"playlist_position": 1}],
                "generated_missing_playlist": None,
            }
        )
        self.assertIsNone(ORCHESTRATOR.final_missing_manifest(path))

    def test_rewrite_with_playlist_url_is_final(self) -> None:
        path = self.write_payload(
            {
                "missing_count": 1,
                "tracks": [{"playlist_position": 1}],
                "generated_missing_playlist": {
                    "url": "https://open.spotify.com/playlist/generated"
                },
            }
        )
        self.assertIsNotNone(ORCHESTRATOR.final_missing_manifest(path))

    def test_rewrite_with_multiple_playlist_urls_is_final(self) -> None:
        path = self.write_payload(
            {
                "missing_count": 11,
                "tracks": [{} for _ in range(11)],
                "generated_missing_playlist": None,
                "generated_missing_playlists": [
                    {
                        "url": "https://open.spotify.com/playlist/first",
                        "start_index": 0,
                        "end_index": 10,
                        "track_count": 10,
                    },
                    {
                        "url": "https://open.spotify.com/playlist/second",
                        "start_index": 10,
                        "end_index": 11,
                        "track_count": 1,
                    },
                ],
            }
        )
        manifest = ORCHESTRATOR.final_missing_manifest(path)

        self.assertIsNotNone(manifest)
        assert manifest is not None
        batches = ORCHESTRATOR.missing_playlist_batches(manifest)
        self.assertEqual([batch["track_count"] for batch in batches], [10, 1])
        self.assertEqual(ORCHESTRATOR.batch_for_completed(batches, 10), batches[1])

    def test_failed_status_read_does_not_shrink_the_wait(self) -> None:
        # The status file is rewritten with the search progress: a read that fails (base deadline,
        # already past) must not end a wait the preflight had extended.
        path = self.write_payload({"missing_count": 1, "tracks": [{}], "generated_missing_playlist": None})
        final = {"missing_count": 0, "tracks": [], "generated_missing_playlist": None}
        reads = iter([None, None, final])
        process = mock.Mock(poll=mock.Mock(return_value=None))
        deadlines = iter([ORCHESTRATOR.time.monotonic() + 60, 0.0, 0.0, 0.0])
        with mock.patch.object(ORCHESTRATOR, "manifest_deadline", side_effect=lambda *a: next(deadlines)), \
                mock.patch.object(ORCHESTRATOR, "final_missing_manifest", side_effect=lambda *a: next(reads)), \
                mock.patch.object(ORCHESTRATOR, "MANIFEST_POLL_SECONDS", 0):
            payload = ORCHESTRATOR.wait_for_missing_manifest(process, path, 0, status_path=path)
        self.assertEqual(payload, final)


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

    def test_finds_incomplete_run_and_preserves_completed_prefix(self) -> None:
        playlist_id = "31xPzrgElxRxrMErJ1WBhD"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / f"20260927-120000_{playlist_id}"
            recorded_dir = run_dir / "recorded_missing"
            recorded_dir.mkdir(parents=True)
            tracks = [
                {
                    "playlist_position": index + 1,
                    "duration_seconds": 60.0,
                    "spotify": {"spotify_id": f"track-{index}"},
                }
                for index in range(2)
            ]
            (run_dir / "soulseek_missing.json").write_text(
                json.dumps(
                    {
                        "missing_count": 2,
                        "tracks": tracks,
                        "generated_missing_playlist": {
                            "url": "https://open.spotify.com/playlist/generated"
                        },
                    }
                ),
                encoding="utf-8",
            )
            (recorded_dir / "recording_progress.json").write_text(
                json.dumps(
                    {
                        "total_tracks": 2,
                        "completed_count": 1,
                        "status": "stalled",
                    }
                ),
                encoding="utf-8",
            )

            result = ORCHESTRATOR.resumable_run(root, playlist_id)

            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result[0], run_dir)
            self.assertEqual(result[2]["completed_count"], 1)

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

    def test_pending_missing_recording_is_resumed_first(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = self.make_run(Path(temp_dir))
            self.write(run_dir / "soulseek_missing.json", self.manifest([3, 4]))
            self.write(run_dir / "recorded_missing" / "recording_progress.json",
                       {"completed_count": 1, "total_tracks": 2, "status": "recording"})
            self.write(run_dir / ORCHESTRATOR.FAILED_JSON_NAME, self.manifest([1]))
            point = ORCHESTRATOR.resume_point(Path(temp_dir), "PL1")
            self.assertEqual(point[0], "missing")
            self.assertEqual(point[1], run_dir)

    def test_failed_tracks_resume_after_the_download_finished(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = self.make_run(Path(temp_dir))
            self.write(run_dir / "soulseek_missing.json", self.manifest([3]))
            self.write(run_dir / "recorded_missing" / "recording_progress.json",
                       {"completed_count": 1, "total_tracks": 1, "status": "completed"})
            self.write(run_dir / ORCHESTRATOR.FAILED_JSON_NAME, self.manifest([1, 2]))
            point = ORCHESTRATOR.resume_point(Path(temp_dir), "PL1")
            self.assertEqual((point[0], point[1], point[3]), ("failed", run_dir, None))

            self.write(run_dir / ORCHESTRATOR.RECORDED_FAILED_DIR / "recording_progress.json",
                       {"completed_count": 1, "total_tracks": 2, "status": "recording"})
            self.assertEqual(ORCHESTRATOR.resume_point(Path(temp_dir), "PL1")[3]["completed_count"], 1)

            self.write(run_dir / ORCHESTRATOR.RECORDED_FAILED_DIR / "recording_progress.json",
                       {"completed_count": 2, "total_tracks": 2, "status": "completed"})
            self.assertIsNone(ORCHESTRATOR.resume_point(Path(temp_dir), "PL1"))

    def test_finished_or_still_downloading_runs_are_not_resumed_as_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = self.make_run(Path(temp_dir))
            self.write(run_dir / "soulseek_missing.json", self.manifest([]))
            self.assertIsNone(ORCHESTRATOR.resume_point(Path(temp_dir), "PL1"))  # sin lista de no bajados
            self.write(run_dir / ORCHESTRATOR.FAILED_JSON_NAME, self.manifest([1]))
            self.assertEqual(ORCHESTRATOR.resume_point(Path(temp_dir), "PL1")[0], "failed")
            self.write(run_dir / "orchestration_report.json", {"status": "partial"})
            self.assertIsNone(ORCHESTRATOR.resume_point(Path(temp_dir), "PL1"))

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


class WorkflowSimulationTests(unittest.TestCase):
    """run_workflow de punta a punta con descargador, grabador y reproductor simulados."""

    PID = "PL1"

    @staticmethod
    def track(n: int) -> dict:
        return {"spotify_id": f"id{n}", "title": f"Tema {n}", "artists": ["Artista"]}

    def manifest(self, positions, playlists=True, error=None):
        tracks = [{"playlist_position": p, "spotify": self.track(p)} for p in positions]
        generated = None
        if tracks and playlists:
            generated = [{"url": "https://open.spotify.com/playlist/b1", "start_index": 0,
                          "end_index": len(tracks), "track_count": len(tracks)}]
        if tracks and error:
            generated = [{"url": "", "error": error}]
        return {"missing_count": len(tracks), "tracks": tracks, "generated_missing_playlists": generated}

    def popen(self, failed_playlists_error=None, during_download=None):
        calls = []

        def fake(cmd, env=None):
            script = Path(cmd[2]).name
            arg = lambda flag: Path(cmd[cmd.index(flag) + 1])  # noqa: E731
            calls.append((script, cmd))
            if script == "spotify_soulseek_download.py":
                write = FallbackRecordingTests.write
                write(arg("--availability-report"), {"playlist": {"name": "Lista"}, "tracks": [
                    {"position": p, "spotify": self.track(p), "status": s}
                    for p, s in ((1, "available"), (2, "available"), (3, "missing"))]})
                write(arg("--missing-json"), self.manifest([3]))
                gate = arg("--download-gate")
                have = arg("--have-dir") if "--have-dir" in cmd else None

                def step():
                    if not gate.exists():
                        return None
                    if during_download is not None:  # p. ej. la app pospone la grabación
                        during_download(arg("--output").parent)
                    if not (have and (have / "Artista - Tema 1.mp3").exists()):  # como el real: no baja lo que ya está
                        ConsolidationTests.tagged_file(arg("--output") / "Artista - Tema 1.mp3", "id1")
                    write(arg("--failed-json"), self.manifest([2], error=failed_playlists_error))
                    return 3  # el tema 2 no se pudo bajar
                return FakeProcess(step)
            if script == "playlist_loopback_recorder.py":
                manifest = json.loads(Path(cmd[3]).read_text(encoding="utf-8"))
                out = arg("--output-dir")
                FallbackRecordingTests.write(arg("--ready-file"), {"status": "ready"})
                polls = []

                def step():
                    polls.append(1)
                    if len(polls) < 2:
                        return None  # el descargador termina mientras se graba
                    for t in manifest["tracks"]:
                        ConsolidationTests.tagged_file(out / f"Artista - {t['spotify']['title']}.mp3",
                                                       t["spotify"]["spotify_id"])
                    n = len(manifest["tracks"])
                    FallbackRecordingTests.write(out / "recording_progress.json", {
                        "completed_count": n, "total_tracks": n, "status": "completed",
                        "source_json": str(Path(cmd[3])),
                        "tracks": [{"playlist_position": t["playlist_position"]} for t in manifest["tracks"]]})
                    return 0
                return FakeProcess(step)
            FallbackRecordingTests.write(arg("--ready-file"), {"status": "ready"})  # reproductor
            return FakeProcess(lambda: None)
        return fake, calls

    def run_flow(self, root: Path, defer_recording=False, **kwargs):
        fake, calls = self.popen(**kwargs)
        args = argparse.Namespace(playlist_url=f"https://open.spotify.com/playlist/{self.PID}",
                                  output_root=root / "runs", library_root=root / "lib",
                                  manifest_timeout=30.0, recorder_timeout=30.0, fresh=False,
                                  defer_recording=defer_recording)
        with mock.patch.object(ORCHESTRATOR.subprocess, "Popen", side_effect=fake), \
                mock.patch.object(ORCHESTRATOR, "validate_environment"), \
                mock.patch.object(ORCHESTRATOR, "find_required_program", return_value=root / "tool.exe"), \
                mock.patch.object(ORCHESTRATOR, "_validate_final_mp3", side_effect=ORCHESTRATOR.embedded_spotify_id), \
                mock.patch.object(ORCHESTRATOR.time, "sleep"), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = ORCHESTRATOR.run_workflow(args)
        run_dir = next((root / "runs").iterdir())
        report = json.loads((run_dir / "orchestration_report.json").read_text(encoding="utf-8"))
        return code, calls, run_dir, report

    def test_undownloaded_track_is_recorded_and_the_playlist_completes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            code, calls, run_dir, report = self.run_flow(root)
            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["soulseek_returncode"], 3)
            self.assertEqual(sorted(p.name for p in (root / "lib" / "Lista").iterdir()),
                             ["Artista - Tema 1.mp3", "Artista - Tema 2.mp3", "Artista - Tema 3.mp3"])
            recorders = [cmd for script, cmd in calls if script == "playlist_loopback_recorder.py"]
            self.assertEqual([Path(c[c.index("--output-dir") + 1]).name for c in recorders],
                             ["recorded_missing", ORCHESTRATOR.RECORDED_FAILED_DIR])
            downloader = next(cmd for script, cmd in calls if script == "spotify_soulseek_download.py")
            self.assertIn("--failed-json", downloader)
            status = run_dir / "diagnostics" / "spotify_playback_status.json"
            self.assertFalse(status.exists() and "stalled" in status.read_text(encoding="utf-8"))

    def test_without_playlists_to_record_the_run_saves_what_it_has(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            code, calls, run_dir, report = self.run_flow(root, failed_playlists_error="Spotify 429")
            self.assertEqual(code, ORCHESTRATOR.EXIT_PARTIAL)
            self.assertEqual(report["status"], "partial")
            self.assertEqual(report["missing_spotify_ids"], ["id2"])
            self.assertIn("Spotify 429", report["reason"])
            self.assertEqual(sorted(p.name for p in (root / "lib" / "Lista").iterdir()),
                             ["Artista - Tema 1.mp3", "Artista - Tema 3.mp3"])
            self.assertEqual(sum(script == "playlist_loopback_recorder.py" for script, _ in calls), 1)

    ALL = ["Artista - Tema 1.mp3", "Artista - Tema 2.mp3", "Artista - Tema 3.mp3"]

    def library(self, root: Path) -> list[str]:
        return sorted(p.name for p in (root / "lib" / "Lista").iterdir())

    @staticmethod
    def recorders(calls) -> list[str]:
        return [Path(c[c.index("--output-dir") + 1]).name for s, c in calls if s == "playlist_loopback_recorder.py"]

    def test_postponed_recording_saves_soulseek_now_and_records_later(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            code, calls, run_dir, report = self.run_flow(root, defer_recording=True)
            self.assertEqual(code, ORCHESTRATOR.EXIT_RESUMABLE)
            self.assertEqual(report["status"], "recording_deferred")
            self.assertEqual((report["saved_spotify_ids"], report["pending_recordings"]), (["id1"], 2))
            self.assertEqual(self.recorders(calls), [])  # no se grabó nada
            self.assertEqual(self.library(root), ["Artista - Tema 1.mp3"])  # lo de Soulseek, ya en la carpeta
            self.assertEqual(ORCHESTRATOR.resume_point(root / "runs", self.PID)[:2], ("deferred", run_dir))

            code, calls, _, report = self.run_flow(root)  # «Grabar ahora»
            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "completed")
            self.assertFalse(any(s == "spotify_soulseek_download.py" for s, _ in calls))  # ni busca ni baja
            self.assertEqual(self.recorders(calls), ["recorded_missing", ORCHESTRATOR.RECORDED_FAILED_DIR])
            self.assertEqual(self.library(root), self.ALL)
            consolidation = json.loads((run_dir / "consolidation_report.json").read_text(encoding="utf-8"))
            self.assertEqual(sorted(r["spotify_id"] for r in consolidation["tracks"]), ["id1", "id2", "id3"])

    def test_postponing_during_the_download_keeps_what_was_recorded(self) -> None:
        postpone = lambda run_dir: (run_dir / ORCHESTRATOR.DEFER_SIGNAL).write_text("1", encoding="utf-8")  # noqa: E731
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            code, calls, run_dir, report = self.run_flow(root, during_download=postpone)
            self.assertEqual((code, report["status"], report["pending_recordings"]),
                             (ORCHESTRATOR.EXIT_RESUMABLE, "recording_deferred", 1))
            self.assertEqual(self.recorders(calls), ["recorded_missing"])  # los faltantes ya se grabaron
            code, calls, _, report = self.run_flow(root)
            self.assertEqual((code, report["status"]), (0, "completed"))
            self.assertEqual(self.recorders(calls), [ORCHESTRATOR.RECORDED_FAILED_DIR])
            self.assertEqual(self.library(root), self.ALL)

    def test_tracks_already_in_the_music_folder_count_as_saved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            earlier = root / "runs" / f"20200101-000000_{self.PID}"
            folder = root / "lib" / "Lista"
            ConsolidationTests.tagged_file(folder / "Artista - Tema 1.mp3", "id1")
            FallbackRecordingTests.write(earlier / "orchestration_report.json",
                                         {"status": "completed", "final_output": str(folder)})
            code, calls, _, _ = self.run_flow(root)
            downloader = next(cmd for script, cmd in calls if script == "spotify_soulseek_download.py")
            self.assertEqual(Path(downloader[downloader.index("--have-dir") + 1]), folder.resolve())
            self.assertEqual(code, 0)  # el tema 1 no se bajó de nuevo y la carpeta quedó completa
            self.assertEqual(self.library(root), self.ALL)
            new_run = next(d for d in (root / "runs").iterdir() if d != earlier)
            report = json.loads((new_run / "orchestration_report.json").read_text(encoding="utf-8"))
            self.assertEqual((report["status"], report["saved_spotify_ids"]), ("completed", ["id1"]))

    def test_a_music_folder_outside_the_destination_is_not_used(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            other = root / "otra_unidad" / "Lista"
            other.mkdir(parents=True)
            FallbackRecordingTests.write(root / "runs" / f"20200101-000000_{self.PID}" / "orchestration_report.json",
                                         {"status": "completed", "final_output": str(other)})
            self.assertIsNone(ORCHESTRATOR.playlist_music_folder(root / "runs", self.PID, root / "lib"))
            self.assertEqual(ORCHESTRATOR.playlist_music_folder(root / "runs", self.PID, root), other.resolve())

    def test_earlier_unsaved_downloads_are_offered_for_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            earlier = root / "runs" / f"20200101-000000_{self.PID}"
            ConsolidationTests.tagged_file(earlier / "soulseek" / "Artista - Tema 1.mp3", "id1")
            FallbackRecordingTests.write(earlier / "orchestration_report.json", {"status": "interrupted"})
            _, calls, _, _ = self.run_flow(root)
            downloader = next(cmd for script, cmd in calls if script == "spotify_soulseek_download.py")
            self.assertEqual(downloader[downloader.index("--reuse-dir") + 1], str(earlier / "soulseek"))

    def test_turning_postponing_off_during_the_download_records_everything(self) -> None:
        resume = lambda run_dir: (run_dir / ORCHESTRATOR.DEFER_SIGNAL).unlink()  # noqa: E731
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            code, calls, _, report = self.run_flow(root, defer_recording=True, during_download=resume)
            self.assertEqual((code, report["status"]), (0, "completed"))
            self.assertEqual(self.recorders(calls), ["recorded_missing", ORCHESTRATOR.RECORDED_FAILED_DIR])
            self.assertEqual(self.library(root), self.ALL)


class SearchFailureTests(unittest.TestCase):
    """Run del 2026-10-30 con 152 temas: el preflight agotó su tiempo y todo fue a grabarse."""

    def test_deadline_extends_to_what_the_preflight_announced(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            status = Path(temp_dir) / "preflight_status.json"
            base = ORCHESTRATOR.time.monotonic() + 10
            self.assertEqual(ORCHESTRATOR.manifest_deadline(base, status), base)
            status.write_text(json.dumps({"tracks": 152, "timeout_s": 2092,
                                          "started_at": ORCHESTRATOR.time.time() - 100}), encoding="utf-8")
            remaining = ORCHESTRATOR.manifest_deadline(base, status) - ORCHESTRATOR.time.monotonic()
            self.assertAlmostEqual(remaining, 2092 - 100 + 300, delta=5)

    def test_failed_search_is_detected_and_never_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "20261030-183518_PL1"
            write = FallbackRecordingTests.write
            write(run_dir / "orchestration_report.json", {"status": "resumable"})
            write(run_dir / "soulseek_availability.json", {"tracks": [
                {"status": "error", "error": "timeout del preflight Soulseek batch (600s)"}] * 3})
            write(run_dir / "soulseek_missing.json", FallbackRecordingTests.manifest([]))
            write(run_dir / ORCHESTRATOR.FAILED_JSON_NAME, FallbackRecordingTests.manifest([1, 2, 3]))
            self.assertIn("timeout", ORCHESTRATOR.preflight_failed_entirely(run_dir))
            self.assertIsNone(ORCHESTRATOR.resume_point(Path(temp_dir), "PL1"))
            write(run_dir / "soulseek_availability.json", {"tracks": [
                {"status": "error", "error": "x"}, {"status": "available"}]})
            self.assertIsNone(ORCHESTRATOR.preflight_failed_entirely(run_dir))
            self.assertEqual(ORCHESTRATOR.resume_point(Path(temp_dir), "PL1")[0], "failed")

    def test_workflow_stops_instead_of_recording_the_whole_playlist(self) -> None:
        sim = WorkflowSimulationTests()
        fake, calls = sim.popen()

        def failing(cmd, env=None):
            process = fake(cmd, env)
            if Path(cmd[2]).name == "spotify_soulseek_download.py":
                FallbackRecordingTests.write(Path(cmd[cmd.index("--availability-report") + 1]), {
                    "playlist": {"name": "Lista"},
                    "tracks": [{"position": p, "spotify": sim.track(p), "status": "error",
                                "error": "timeout del preflight Soulseek batch (600s)"} for p in (1, 2, 3)]})
                FallbackRecordingTests.write(Path(cmd[cmd.index("--missing-json") + 1]), sim.manifest([]))
            return process
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = argparse.Namespace(playlist_url="https://open.spotify.com/playlist/PL1", output_root=root / "runs",
                                      library_root=root / "lib", manifest_timeout=30.0, recorder_timeout=30.0,
                                      fresh=False, defer_recording=False)
            with mock.patch.object(ORCHESTRATOR.subprocess, "Popen", side_effect=failing),                     mock.patch.object(ORCHESTRATOR, "validate_environment"),                     mock.patch.object(ORCHESTRATOR, "find_required_program", return_value=root / "tool.exe"),                     mock.patch.object(ORCHESTRATOR.time, "sleep"),                     contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()),                     self.assertRaisesRegex(ORCHESTRATOR.WorkflowError, "búsqueda en Soulseek falló"):
                ORCHESTRATOR.run_workflow(args)
            self.assertFalse(any(script == "playlist_loopback_recorder.py" for script, _ in calls))
            downloader = next(cmd for script, cmd in calls if script == "spotify_soulseek_download.py")
            self.assertIn("--preflight-status", downloader)


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


if __name__ == "__main__":
    unittest.main()
