"""
PURPOSE: Open a Spotify playlist in the Windows desktop client and start playback.

CHANGELOG:
- 2026-09-27: Add ``--start-only`` for non-blocking orchestration.
- 2026-09-27: Add a zero-based playlist offset for resumable recording.
- 2026-09-27: Add a playback watchdog handshake for resumable capture.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import spotipy
from spotipy.oauth2 import SpotifyOAuth

from dotenv import load_dotenv
load_dotenv()


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


SCOPES = " ".join([
    "user-read-playback-state",
    "user-read-currently-playing",
    "user-modify-playback-state",
])


def playlist_url_to_uri(url: str) -> str:
    match = re.search(r"open\.spotify\.com/playlist/([A-Za-z0-9]+)", url)

    if not match:
        raise ValueError("Invalid Spotify playlist URL")

    playlist_id = match.group(1)
    return f"spotify:playlist:{playlist_id}"


def open_spotify(uri: str):
    if os.name != "nt":
        raise RuntimeError("This example is written for Windows.")

    os.startfile(uri)


def wait_for_desktop_device(sp, timeout=20):
    deadline = time.time() + timeout

    while time.time() < deadline:
        devices = sp.devices()["devices"]

        computers = [
            d for d in devices
            if d["type"].lower() == "computer"
            and not d.get("is_restricted", False)
        ]

        if computers:
            active = [d for d in computers if d["is_active"]]

            if active:
                return active[0]

            return computers[0]

        time.sleep(0.5)

    raise RuntimeError("Spotify Desktop did not appear as an available device.")


def monitor_tracks(sp):
    last_track_id = None

    print("\nMonitoring playback. Ctrl+C to stop.\n")

    while True:
        state = sp.current_playback()

        if not state or not state.get("item"):
            time.sleep(0.2)
            continue

        track = state["item"]
        track_id = track["id"]

        if track_id != last_track_id:
            wall_time_ns = time.time_ns()

            artists = ", ".join(
                artist["name"]
                for artist in track["artists"]
            )

            print("=" * 70)
            print(f"TRACK START DETECTED")
            print(f"Wall clock ns : {wall_time_ns}")
            print(f"Spotify time  : {state.get('timestamp')}")
            print(f"Track ID      : {track_id}")
            print(f"Track URI     : {track['uri']}")
            print(f"Title         : {track['name']}")
            print(f"Artist        : {artists}")
            print(f"Duration      : {track['duration_ms']} ms")
            print(f"Progress      : {state.get('progress_ms')} ms")
            print("=" * 70)

            last_track_id = track_id

        time.sleep(0.2)


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def monitor_playback_health(sp, status_file: Path, grace_seconds: float) -> int:
    """Return 75 if Spotify remains paused away from a natural track ending."""
    stopped_since: float | None = None
    while True:
        state = sp.current_playback()
        now = time.monotonic()
        if state and state.get("is_playing"):
            stopped_since = None
            time.sleep(0.4)
            continue

        if stopped_since is None:
            stopped_since = now
        if now - stopped_since < grace_seconds:
            time.sleep(0.2)
            continue

        item = (state or {}).get("item") or {}
        duration_ms = int(item.get("duration_ms") or 0)
        progress_ms = int((state or {}).get("progress_ms") or 0)
        remaining_ms = duration_ms - progress_ms if duration_ms else None
        if remaining_ms is not None and remaining_ms <= 1000:
            write_status(
                status_file,
                {
                    "status": "natural_end",
                    "track_id": item.get("id"),
                    "progress_ms": progress_ms,
                    "duration_ms": duration_ms,
                },
            )
            return 0

        write_status(
            status_file,
            {
                "status": "stalled",
                "track_id": item.get("id"),
                "track_name": item.get("name"),
                "progress_ms": progress_ms,
                "duration_ms": duration_ms,
            },
        )
        print(
            "Spotify playback stopped before the current track completed.",
            file=sys.stderr,
            flush=True,
        )
        return 75


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "playlist_url",
        help="Spotify playlist URL"
    )
    parser.add_argument(
        "--start-only",
        action="store_true",
        help="Start playback and exit instead of monitoring track changes",
    )
    parser.add_argument(
        "--offset-position",
        type=int,
        default=0,
        help="Zero-based playlist position at which playback should start",
    )
    parser.add_argument(
        "--ready-file",
        type=Path,
        default=None,
        help="Write a readiness signal immediately after playback starts",
    )
    parser.add_argument(
        "--watchdog-file",
        type=Path,
        default=None,
        help="Write playback health status and keep monitoring until completion",
    )
    parser.add_argument(
        "--watchdog-grace",
        type=float,
        default=1.5,
        help="Seconds Spotify may remain paused before it is considered stalled",
    )

    args = parser.parse_args()
    if args.offset_position < 0:
        parser.error("--offset-position must be >= 0")
    if args.watchdog_grace <= 0:
        parser.error("--watchdog-grace must be > 0")

    playlist_uri = playlist_url_to_uri(args.playlist_url)

    sp = spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=os.environ["SPOTIFY_CLIENT_ID"],
            client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
            redirect_uri=os.environ["SPOTIFY_REDIRECT_URI"],
            scope=SCOPES,
            cache_path=".spotify_token_cache",
        )
    )

    print(f"Opening Spotify Desktop:")
    print(playlist_uri)

    open_spotify(playlist_uri)

    device = wait_for_desktop_device(sp)

    print(
        f"Using Spotify device: "
        f"{device['name']} [{device['id']}]"
    )

    device_id = device["id"]

    sp.transfer_playback(
        device_id=device_id,
        force_play=False
    )

    time.sleep(0.5)

    sp.shuffle(
        state=False,
        device_id=device_id
    )

    sp.repeat(
        state="off",
        device_id=device_id
    )

    time.sleep(0.5)

    sp.start_playback(
        device_id=device_id,
        context_uri=playlist_uri,
        offset={"position": args.offset_position},
        position_ms=0,
    )

    print(f"Playback started at playlist position {args.offset_position}.", flush=True)

    if args.ready_file is not None:
        write_status(
            args.ready_file.expanduser().resolve(),
            {
                "status": "ready",
                "offset_position": args.offset_position,
                "device_id": device_id,
            },
        )

    if args.watchdog_file is not None:
        return monitor_playback_health(
            sp,
            args.watchdog_file.expanduser().resolve(),
            args.watchdog_grace,
        )

    if args.start_only:
        return 0

    monitor_tracks(sp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
