"""Tests for the Spotify/Soulseek workflow synchronization helpers."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "download_JIJIJI"
    / "spotify_soulseek_orchestrator.py"
)
SPEC = importlib.util.spec_from_file_location("spotify_soulseek_orchestrator", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ORCHESTRATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORCHESTRATOR)


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


if __name__ == "__main__":
    unittest.main()
