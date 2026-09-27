"""Tests for the Spotify/Soulseek workflow synchronization helpers."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
import wave
from array import array
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock


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

if __name__ == "__main__":
    unittest.main()
