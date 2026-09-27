#!/usr/bin/env python3
"""
Record the Windows default output through WASAPI loopback, detect playlist
track boundaries using JSON durations plus silence around each expected
boundary, and export each track as stereo MP3 CBR 320 kbps with ID3 metadata.

Usage:
    python playlist_loopback_recorder.py playlist.json
    python playlist_loopback_recorder.py playlist.json -o tracks

Requirements:
    pip install PyAudioWPatch numpy mutagen
    ffmpeg must be available in PATH
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyaudiowpatch as pyaudio
from mutagen.id3 import COMM, ID3, ID3NoHeaderError, TALB, TIT2, TPE1, TRCK, TXXX


FORMAT = pyaudio.paInt16
CHANNELS = 2
CHUNK = 1024
MP3_BITRATE = "320k"

DEFAULT_SEARCH_BEFORE = 1.50
DEFAULT_SEARCH_AFTER = 1.50

EXACT_ZERO_MIN_MS = 5.0
NEAR_SILENCE_MIN_MS = 20.0
RMS_WINDOW_MS = 20.0
RMS_HOP_MS = 5.0
SILENCE_THRESHOLDS_DBFS = (-80.0, -72.0, -65.0)

START_HISTORY_SEC = 1.0
START_CONFIRM_MS = 60.0
START_THRESHOLD_DBFS = -72.0
START_ONSET_THRESHOLD_DBFS = -82.0


@dataclass
class BoundaryResult:
    track_index: int
    playlist_position: int | None
    expected_sec: float
    detected_sec: float
    correction_sec: float
    method: str
    silence_start_sec: float | None = None
    silence_end_sec: float | None = None
    level_dbfs: float | None = None


class RollingAudio:
    """Keep only the most recent PCM frames needed for boundary analysis."""

    def __init__(self, rate: int, channels: int, keep_seconds: float):
        self.rate = rate
        self.channels = channels
        self.keep_frames = max(1, int(math.ceil(keep_seconds * rate)))
        self.blocks: deque[tuple[int, np.ndarray]] = deque()
        self.total_frames = 0

    def append(self, block: np.ndarray) -> int:
        start = self.total_frames
        self.blocks.append((start, block.copy()))
        self.total_frames += len(block)

        oldest_allowed = self.total_frames - self.keep_frames
        while self.blocks:
            block_start, arr = self.blocks[0]
            block_end = block_start + len(arr)
            if block_end > oldest_allowed:
                break
            self.blocks.popleft()
        return start

    def extract(self, start_sec: float, end_sec: float) -> tuple[np.ndarray, float]:
        start_frame = max(0, int(math.floor(start_sec * self.rate)))
        end_frame = min(self.total_frames, int(math.ceil(end_sec * self.rate)))

        if end_frame <= start_frame:
            return np.empty((0, self.channels), dtype=np.int16), start_frame / self.rate

        pieces: list[np.ndarray] = []
        actual_start: int | None = None

        for block_start, arr in self.blocks:
            block_end = block_start + len(arr)
            if block_end <= start_frame:
                continue
            if block_start >= end_frame:
                break

            left = max(start_frame, block_start) - block_start
            right = min(end_frame, block_end) - block_start
            if right <= left:
                continue

            if actual_start is None:
                actual_start = block_start + left
            pieces.append(arr[left:right])

        if not pieces or actual_start is None:
            return np.empty((0, self.channels), dtype=np.int16), start_frame / self.rate

        return np.concatenate(pieces, axis=0), actual_start / self.rate


def dbfs_from_rms(rms: float) -> float:
    if rms <= 0:
        return -math.inf
    return 20.0 * math.log10(rms / 32768.0)


def block_dbfs(block: np.ndarray) -> float:
    if len(block) == 0:
        return -math.inf
    x = block.astype(np.float64)
    rms = math.sqrt(float(np.mean(x * x)))
    return dbfs_from_rms(rms)


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if len(mask) == 0:
        return []
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return list(zip(starts, ends))


def closest_point_in_interval(expected: float, start: float, end: float) -> float:
    if expected < start:
        return start
    if expected > end:
        return end
    return expected


def rms_dbfs_curve(audio: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    if len(audio) == 0:
        return np.array([]), np.array([])

    x = audio.astype(np.float64)
    power = np.mean(x * x, axis=1)

    window = max(1, int(round(rate * RMS_WINDOW_MS / 1000.0)))
    hop = max(1, int(round(rate * RMS_HOP_MS / 1000.0)))

    if len(power) < window:
        return np.array([]), np.array([])

    cumulative = np.concatenate(([0.0], np.cumsum(power)))
    starts = np.arange(0, len(power) - window + 1, hop, dtype=np.int64)
    sums = cumulative[starts + window] - cumulative[starts]
    rms = np.sqrt(sums / window)
    dbfs = 20.0 * np.log10(np.maximum(rms / 32768.0, 1e-12))
    centers = (starts + window / 2.0) / rate
    return centers, dbfs


def detect_boundary_in_window(
    audio: np.ndarray,
    rate: int,
    window_start_sec: float,
    expected_sec: float,
    track_index: int,
    playlist_position: int | None,
) -> BoundaryResult:
    if len(audio) == 0:
        return BoundaryResult(
            track_index=track_index,
            playlist_position=playlist_position,
            expected_sec=expected_sec,
            detected_sec=expected_sec,
            correction_sec=0.0,
            method="duration_fallback",
        )

    peak = np.max(np.abs(audio.astype(np.int32)), axis=1)
    exact_runs = contiguous_runs(peak == 0)
    min_zero_samples = max(1, int(round(rate * EXACT_ZERO_MIN_MS / 1000.0)))

    candidates: list[tuple[float, float, float, float]] = []
    for start_i, end_i in exact_runs:
        if end_i - start_i < min_zero_samples:
            continue
        start_sec = window_start_sec + start_i / rate
        end_sec = window_start_sec + end_i / rate
        point = closest_point_in_interval(expected_sec, start_sec, end_sec)
        candidates.append((abs(point - expected_sec), point, start_sec, end_sec))

    if candidates:
        _, point, silence_start, silence_end = min(candidates, key=lambda x: x[0])
        return BoundaryResult(
            track_index=track_index,
            playlist_position=playlist_position,
            expected_sec=expected_sec,
            detected_sec=point,
            correction_sec=point - expected_sec,
            method="digital_zero",
            silence_start_sec=silence_start,
            silence_end_sec=silence_end,
            level_dbfs=None,
        )

    rel_times, dbfs = rms_dbfs_curve(audio, rate)
    if len(dbfs):
        abs_times = window_start_sec + rel_times
        min_windows = max(1, int(math.ceil(NEAR_SILENCE_MIN_MS / RMS_HOP_MS)))
        half_window = RMS_WINDOW_MS / 2000.0

        for threshold in SILENCE_THRESHOLDS_DBFS:
            runs = contiguous_runs(dbfs <= threshold)
            candidates2: list[tuple[float, float, float, float, float]] = []

            for start_i, end_i in runs:
                if end_i - start_i < min_windows:
                    continue
                start_sec = float(abs_times[start_i] - half_window)
                last_i = min(end_i - 1, len(abs_times) - 1)
                end_sec = float(abs_times[last_i] + half_window)
                point = closest_point_in_interval(expected_sec, start_sec, end_sec)
                level = float(np.min(dbfs[start_i:end_i]))
                candidates2.append(
                    (abs(point - expected_sec), point, start_sec, end_sec, level)
                )

            if candidates2:
                _, point, silence_start, silence_end, level = min(
                    candidates2, key=lambda x: x[0]
                )
                return BoundaryResult(
                    track_index=track_index,
                    playlist_position=playlist_position,
                    expected_sec=expected_sec,
                    detected_sec=point,
                    correction_sec=point - expected_sec,
                    method=f"silence_{threshold:.0f}_dbfs",
                    silence_start_sec=silence_start,
                    silence_end_sec=silence_end,
                    level_dbfs=level,
                )

    return BoundaryResult(
        track_index=track_index,
        playlist_position=playlist_position,
        expected_sec=expected_sec,
        detected_sec=expected_sec,
        correction_sec=0.0,
        method="duration_fallback",
    )


def find_playlist_start(
    history: np.ndarray,
    history_start_sec: float,
    rate: int,
    confirm_threshold_dbfs: float,
) -> float | None:
    """Find a sustained onset inside the recent history and return absolute seconds."""
    if len(history) == 0:
        return None

    rel_times, dbfs = rms_dbfs_curve(history, rate)
    if len(dbfs) == 0:
        return None

    needed = max(1, int(math.ceil(START_CONFIRM_MS / RMS_HOP_MS)))
    active_runs = contiguous_runs(dbfs >= confirm_threshold_dbfs)
    confirmed = [(a, b) for a, b in active_runs if b - a >= needed]
    if not confirmed:
        return None

    confirmed_start = confirmed[0][0]

    # Search slightly backwards for the first very quiet but real onset so a fade-in
    # is not clipped merely because the confirmation threshold is higher.
    pre = max(0, confirmed_start - int(round(0.35 / (RMS_HOP_MS / 1000.0))))
    earlier = np.where(dbfs[pre : confirmed_start + 1] >= START_ONSET_THRESHOLD_DBFS)[0]
    onset_i = pre + int(earlier[0]) if len(earlier) else confirmed_start

    onset_sec = history_start_sec + float(rel_times[onset_i]) - RMS_WINDOW_MS / 2000.0
    return max(history_start_sec, onset_sec)


def duration_seconds(track: dict[str, Any]) -> float:
    if track.get("duration_seconds") is not None:
        return float(track["duration_seconds"])
    if track.get("duration_ms") is not None:
        return float(track["duration_ms"]) / 1000.0
    raise ValueError(f"Track has no duration_seconds or duration_ms: {track!r}")


def sanitize_filename(text: str) -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
    text = re.sub(r"\s+", " ", text).strip().rstrip(".")
    return text or "untitled"


def flatten_for_tags(prefix: str, value: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_for_tags(child, item))
    elif isinstance(value, list):
        result[prefix] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    elif value is None:
        result[prefix] = "null"
    else:
        result[prefix] = str(value)
    return result


def set_txxx(tags: ID3, desc: str, value: str) -> None:
    tags.delall(f"TXXX:{desc}")
    tags.add(TXXX(encoding=3, desc=desc, text=[value]))


def write_metadata(
    mp3_path: Path,
    track: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    spotify = track.get("spotify") or {}
    title = str(spotify.get("title") or mp3_path.stem)
    artists = spotify.get("artists") or []
    if isinstance(artists, str):
        artists = [artists]
    artist_text = ", ".join(str(x) for x in artists)

    source_playlist = manifest.get("source_playlist") or {}
    playlist_name = source_playlist.get("name")
    playlist_url = source_playlist.get("source")
    playlist_position = track.get("playlist_position")

    try:
        tags = ID3(str(mp3_path))
    except ID3NoHeaderError:
        tags = ID3()

    for frame in ("TIT2", "TPE1", "TALB", "TRCK", "COMM"):
        tags.delall(frame)

    tags.add(TIT2(encoding=3, text=[title]))
    if artist_text:
        tags.add(TPE1(encoding=3, text=[artist_text]))
    if playlist_name:
        tags.add(TALB(encoding=3, text=[str(playlist_name)]))
    if playlist_position is not None:
        tags.add(TRCK(encoding=3, text=[str(playlist_position)]))
    if playlist_url:
        tags.add(
            COMM(
                encoding=3,
                lang="eng",
                desc="source_playlist",
                text=[str(playlist_url)],
            )
        )

    for key, value in flatten_for_tags("track", track).items():
        set_txxx(tags, key, value)

    global_manifest = {k: v for k, v in manifest.items() if k != "tracks"}
    for key, value in flatten_for_tags("manifest", global_manifest).items():
        set_txxx(tags, key, value)

    set_txxx(
        tags,
        "track.raw_json",
        json.dumps(track, ensure_ascii=False, separators=(",", ":")),
    )

    tags.save(str(mp3_path), v2_version=3)


def encode_segment(
    ffmpeg: str,
    wav_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
) -> None:
    duration = end_sec - start_sec
    if duration <= 0:
        raise ValueError(f"Invalid segment {start_sec:.6f} -> {end_sec:.6f}")

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(wav_path),
        "-ss",
        f"{start_sec:.9f}",
        "-t",
        f"{duration:.9f}",
        "-vn",
        "-map_metadata",
        "-1",
        "-ac",
        "2",
        "-c:a",
        "libmp3lame",
        "-b:a",
        MP3_BITRATE,
        "-compression_level",
        "0",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    tracks = manifest.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        raise ValueError("JSON must contain a non-empty 'tracks' array.")

    for i, track in enumerate(tracks, start=1):
        d = duration_seconds(track)
        if not math.isfinite(d) or d <= 0:
            raise ValueError(f"Invalid duration for track {i}: {d}")

    return manifest, tracks


def make_output_filename(track: dict[str, Any], fallback_index: int) -> str:
    spotify = track.get("spotify") or {}
    title = str(spotify.get("title") or f"Track {fallback_index}")
    artists = spotify.get("artists") or []
    if isinstance(artists, str):
        artists = [artists]
    artist_text = ", ".join(str(x) for x in artists) or "Unknown artist"
    pos = track.get("playlist_position")
    prefix = f"{int(pos):03d}" if isinstance(pos, (int, float)) else f"{fallback_index:03d}"
    return sanitize_filename(f"{prefix} - {artist_text} - {title}.mp3")


def check_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg was not found in PATH. Install it, for example with: "
            "winget install -e --id Gyan.FFmpeg"
        )
    return ffmpeg


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Record the Windows default output with WASAPI loopback, split a playlist "
            "using JSON durations plus silence detection, then export MP3 320 kbps."
        )
    )
    parser.add_argument("json_file", type=Path, help="Playlist JSON manifest")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("recorded_tracks"))
    parser.add_argument("--search-before", type=float, default=DEFAULT_SEARCH_BEFORE)
    parser.add_argument("--search-after", type=float, default=DEFAULT_SEARCH_AFTER)
    parser.add_argument(
        "--start-threshold-dbfs",
        type=float,
        default=START_THRESHOLD_DBFS,
        help="RMS threshold used to confirm the beginning of the first track",
    )
    parser.add_argument(
        "--keep-wav",
        action="store_true",
        help="Keep the temporary full-session WAV after successful export",
    )
    args = parser.parse_args()

    if args.search_before <= 0 or args.search_after <= 0:
        parser.error("--search-before and --search-after must be > 0")

    manifest, tracks = load_manifest(args.json_file)
    ffmpeg = check_ffmpeg()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(
        prefix="playlist_capture_", suffix=".wav", dir=str(args.output_dir)
    )
    os.close(fd)
    temp_wav = Path(temp_name)

    boundaries: list[BoundaryResult] = []
    playlist_start: float | None = None
    current_track_start: float | None = None
    next_track_index = 0
    expected_boundary: float | None = None
    capture_finished = False

    p: pyaudio.PyAudio | None = None
    stream = None
    wav_file = None

    try:
        p = pyaudio.PyAudio()
        try:
            device = p.get_default_wasapi_loopback()
        except (OSError, LookupError) as exc:
            raise RuntimeError(
                "Could not find the WASAPI loopback for the Windows default output. "
                "Run 'python -m pyaudiowpatch' to inspect devices."
            ) from exc

        rate = int(device["defaultSampleRate"])
        max_channels = int(device["maxInputChannels"])
        if max_channels < CHANNELS:
            raise RuntimeError(
                f"Default loopback device exposes {max_channels} channel(s), but stereo is required."
            )

        keep_seconds = args.search_before + args.search_after + START_HISTORY_SEC + 2.0
        rolling = RollingAudio(rate, CHANNELS, keep_seconds)

        wav_file = wave.open(str(temp_wav), "wb")
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(p.get_sample_size(FORMAT))
        wav_file.setframerate(rate)

        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=rate,
            frames_per_buffer=CHUNK,
            input=True,
            input_device_index=device["index"],
        )

        print(f"Loopback : {device['name']}")
        print(f"Format   : {rate} Hz, stereo, PCM 16 bit")
        print(f"Tracks   : {len(tracks)}")
        print(f"Output   : {args.output_dir.resolve()}")
        print()
        print("Waiting for the first track to start...")
        print("Start playlist playback now. Keep other system sounds off while recording.")

        start_confirm_blocks = 0
        blocks_needed = max(1, math.ceil((START_CONFIRM_MS / 1000.0) * rate / CHUNK))

        while not capture_finished:
            raw = stream.read(CHUNK, exception_on_overflow=False)
            wav_file.writeframesraw(raw)

            arr = np.frombuffer(raw, dtype="<i2")
            usable = len(arr) - (len(arr) % CHANNELS)
            arr = arr[:usable].reshape(-1, CHANNELS)
            rolling.append(arr)

            now_sec = rolling.total_frames / rate

            if playlist_start is None:
                if block_dbfs(arr) >= START_THRESHOLD_DBFS:
                    start_confirm_blocks += 1
                else:
                    start_confirm_blocks = 0

                if start_confirm_blocks >= blocks_needed:
                    hist_start = max(0.0, now_sec - START_HISTORY_SEC)
                    history, actual_hist_start = rolling.extract(hist_start, now_sec)
                    onset = find_playlist_start(
                        history, actual_hist_start, rate, args.start_threshold_dbfs
                    )
                    if onset is None:
                        onset = max(0.0, now_sec - start_confirm_blocks * CHUNK / rate)

                    playlist_start = onset
                    current_track_start = onset
                    next_track_index = 0
                    expected_boundary = current_track_start + duration_seconds(tracks[0])

                    spotify = tracks[0].get("spotify") or {}
                    title = spotify.get("title") or "Track 1"
                    print(f"\nStarted   : {playlist_start:.3f} s")
                    print(f"Track 1   : {title}")
                    print(f"Expected end: {expected_boundary:.3f} s")
                continue

            assert current_track_start is not None
            assert expected_boundary is not None

            if now_sec < expected_boundary + args.search_after:
                continue

            window_audio, window_start = rolling.extract(
                expected_boundary - args.search_before,
                expected_boundary + args.search_after,
            )

            track = tracks[next_track_index]
            position = track.get("playlist_position")
            result = detect_boundary_in_window(
                window_audio,
                rate,
                window_start,
                expected_boundary,
                track_index=next_track_index,
                playlist_position=position if isinstance(position, int) else None,
            )
            boundaries.append(result)

            spotify = track.get("spotify") or {}
            title = spotify.get("title") or f"Track {next_track_index + 1}"
            print(
                f"Cut {next_track_index + 1:03d}: {result.detected_sec:.3f} s  "
                f"delta={result.correction_sec:+.3f} s  {result.method}  {title}"
            )

            next_track_index += 1
            if next_track_index >= len(tracks):
                capture_finished = True
                break

            current_track_start = result.detected_sec
            expected_boundary = current_track_start + duration_seconds(tracks[next_track_index])

            spotify_next = tracks[next_track_index].get("spotify") or {}
            next_title = spotify_next.get("title") or f"Track {next_track_index + 1}"
            print(f"Track {next_track_index + 1}: {next_title}")
            print(f"Expected end: {expected_boundary:.3f} s")

        stream.stop_stream()
        stream.close()
        stream = None

        wav_file.close()
        wav_file = None

        if playlist_start is None or len(boundaries) != len(tracks):
            raise RuntimeError("Capture ended before all track boundaries were determined.")

        print("\nCapture complete. Encoding MP3 files...")

        starts = [playlist_start] + [b.detected_sec for b in boundaries[:-1]]
        ends = [b.detected_sec for b in boundaries]

        report_tracks: list[dict[str, Any]] = []
        created_files: list[Path] = []

        for i, (track, start_sec, end_sec) in enumerate(zip(tracks, starts, ends), start=1):
            filename = make_output_filename(track, i)
            output_path = args.output_dir / filename

            encode_segment(ffmpeg, temp_wav, output_path, start_sec, end_sec)
            write_metadata(output_path, track, manifest)
            created_files.append(output_path)

            declared = duration_seconds(track)
            report_tracks.append(
                {
                    "index": i,
                    "playlist_position": track.get("playlist_position"),
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "segment_duration_sec": end_sec - start_sec,
                    "declared_duration_sec": declared,
                    "duration_difference_sec": (end_sec - start_sec) - declared,
                    "output": str(output_path),
                }
            )
            print(f"[{i:03d}/{len(tracks):03d}] {filename}")

        report = {
            "source_json": str(args.json_file),
            "sample_rate": rate,
            "channels": CHANNELS,
            "playlist_start_sec": playlist_start,
            "search_before_sec": args.search_before,
            "search_after_sec": args.search_after,
            "boundaries": [asdict(b) for b in boundaries],
            "tracks": report_tracks,
        }

        report_path = args.output_dir / "segmentation_report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)

        if not args.keep_wav:
            temp_wav.unlink(missing_ok=True)
        else:
            print(f"Temporary WAV kept: {temp_wav}")

        print(f"\nDone. {len(created_files)} MP3 files written.")
        print(f"Report: {report_path}")
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        print(f"Partial capture kept at: {temp_wav}", file=sys.stderr)
        return 130

    finally:
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

        if wav_file is not None:
            try:
                wav_file.close()
            except Exception:
                pass

        if p is not None:
            try:
                p.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
