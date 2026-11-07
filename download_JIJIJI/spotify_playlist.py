"""
PURPOSE: Open a Spotify playlist in the Windows desktop client and start playback.

CHANGELOG:
- 2026-11-07: The playlist plays on repeat while it is recorded: after its last track Spotify goes on to
              the first one, so the recorder hears the end of the last track like any other cut. With
              repeat off Spotify stopped there and went back to the first track (progress 0), and the
              watchdog took that end for a stall (Gabriel's run of 2026-11-07: 0/1 recorded). The
              watchdog also judges the end from the last state it saw playing, and ``--pause`` turns
              repeat off again.
- 2026-11-07: ``--route-output NAME --route-state PATH``: once Spotify is open and before it plays, send
              it alone to the recorded output (spotify_output.py; the orchestrator gives the previous
              output back when the download ends).
- 2026-11-06: ``--pause --device-id ID``: pause this computer's Spotify (only if it is the one playing),
              so autoplay does not go on between recording sessions or after one is postponed. The
              ready file says the device ("starting") before playback starts, so the orchestrator can
              pause it if the start times out.
- 2026-10-30: Keep the token cache at the repo root, not in the working directory.
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
# Absolute so the authorization is reused from any working directory: inside the
# orchestrator the player has only seconds to start, too few for a browser login.
TOKEN_CACHE = Path(__file__).resolve().parent.parent / ".spotify_token_cache"
# A stop this close to where the track last seen playing was due to end is that track's natural end
# (the watchdog polls every ~0.4 s and Spotify's progress lags a little).
NATURAL_END_MS = 2000


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
    """Return 75 if Spotify remains paused away from a natural track ending.

    The end of a playlist is a natural ending too, but there Spotify goes back to its first track
    (progress 0): the end is also judged from the last state seen playing."""
    stopped_since: float | None = None
    last_playing: tuple[dict, int, float] | None = None  # (track, progress_ms, when it was seen)
    while True:
        state = sp.current_playback()
        now = time.monotonic()
        if state and state.get("is_playing"):
            stopped_since = None
            last_playing = (state.get("item") or {}, int(state.get("progress_ms") or 0), now)
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
        if (remaining_ms is None or remaining_ms > 1000) and last_playing is not None:
            seen_item, seen_progress, seen_at = last_playing
            seen_duration = int(seen_item.get("duration_ms") or 0)
            reached_ms = seen_progress + (stopped_since - seen_at) * 1000.0
            if seen_duration and seen_duration - reached_ms <= NATURAL_END_MS:
                item, duration_ms = seen_item, seen_duration
                progress_ms, remaining_ms = min(int(reached_ms), seen_duration), 0
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


def route_to_recorded_output(output: str, state_path: Path | None) -> None:
    """Spotify alone to the recorded output (spotify_output.py). If it fails, the recording says that no sound
    arrived there; it does not stop playback."""
    import spotify_output

    try:
        result = spotify_output.route(output, state_path or Path(spotify_output.STATE_NAME))
    except Exception as exc:  # noqa: BLE001
        print(f"No pude mandar Spotify a «{output}» ({exc}). Mándalo a mano: Configuración → Sonido → "
              f"Mezclador de volumen → Spotify → {output}.", file=sys.stderr, flush=True)
        return
    print(f"Spotify sale por «{result.get('device') or output}» mientras se graba.", flush=True)


def pause_device(sp, device_id: str) -> bool:
    """Pause only if this device is the one playing: never the phone that took over playback."""
    state = sp.current_playback() or {}
    device = state.get("device") or {}
    if device.get("id") != device_id or not state.get("is_playing"):
        return False
    sp.pause_playback(device_id=device_id)
    try:
        sp.repeat(state="off", device_id=device_id)  # the recording left the playlist on repeat
    except Exception:  # noqa: BLE001 (paused is what matters)
        pass
    return True


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "playlist_url",
        nargs="?",
        help="Spotify playlist URL"
    )
    parser.add_argument(
        "--pause",
        action="store_true",
        help="Pause playback on --device-id (if it is playing) and exit",
    )
    parser.add_argument(
        "--device-id",
        default=None,
        help="Spotify Connect device to pause with --pause",
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
        "--route-output",
        default="",
        help="Before playing, send Spotify alone to this output (the one recorded, e.g. \"Voicemeeter Input\")",
    )
    parser.add_argument(
        "--route-state",
        type=Path,
        default=None,
        help="Where the output Spotify had is saved, to give it back (spotify_output.py)",
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
    if args.pause and not args.device_id:
        parser.error("--pause needs --device-id")
    if not args.pause and not args.playlist_url:
        parser.error("the playlist URL is required")

    sp = spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=os.environ["SPOTIFY_CLIENT_ID"],
            client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
            redirect_uri=os.environ["SPOTIFY_REDIRECT_URI"],
            scope=SCOPES,
            cache_path=str(TOKEN_CACHE),
        )
    )
    if args.pause:
        paused = pause_device(sp, args.device_id)
        print("Spotify en pausa." if paused else "Spotify no estaba sonando en este ordenador.", flush=True)
        return 0

    playlist_uri = playlist_url_to_uri(args.playlist_url)

    print(f"Opening Spotify Desktop:")
    print(playlist_uri)

    open_spotify(playlist_uri)

    device = wait_for_desktop_device(sp)

    print(
        f"Using Spotify device: "
        f"{device['name']} [{device['id']}]"
    )

    device_id = device["id"]
    if args.ready_file is not None:  # no es «ready»: dice qué dispositivo, por si el arranque se demora
        write_status(args.ready_file.expanduser().resolve(), {"status": "starting", "device_id": device_id})
    if args.route_output:
        route_to_recorded_output(args.route_output, args.route_state)

    sp.transfer_playback(
        device_id=device_id,
        force_play=False
    )

    time.sleep(0.5)

    sp.shuffle(
        state=False,
        device_id=device_id
    )

    # On repeat: after the last track Spotify goes on to the first one, so the recorder hears the end
    # of the last track like any other cut. The orchestrator pauses Spotify once the session is
    # recorded (and ``--pause`` turns repeat off).
    sp.repeat(
        state="context",
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
    try:  # a new context must not have reset it
        sp.repeat(state="context", device_id=device_id)
    except Exception as exc:  # noqa: BLE001 (the watchdog still recognizes the end of the playlist)
        print(f"Could not confirm repeat: {exc}", file=sys.stderr, flush=True)

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
