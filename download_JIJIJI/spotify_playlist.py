import argparse
import os
import re
import time

import spotipy
from spotipy.oauth2 import SpotifyOAuth

from dotenv import load_dotenv
load_dotenv()


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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "playlist_url",
        help="Spotify playlist URL"
    )

    args = parser.parse_args()

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
        offset={"position": 0},
        position_ms=0,
    )

    print("Playback started.")

    monitor_tracks(sp)


if __name__ == "__main__":
    main()