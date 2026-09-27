#!/usr/bin/env python3
"""
PURPOSE: Record a Spotify playlist through Windows WASAPI loopback and split it.

CHANGELOG:
- 2026-09-27: Reuse the Soulseek finalizer's Spotify tags, cover, and filenames.
- 2026-09-27: Add an optional ready-file handshake for process orchestration.
- 2026-09-27: Checkpoint every completed MP3 and resume at the first pending track.

Detect playlist track boundaries using JSON durations plus silence around each
expected boundary, and export each track as stereo MP3 CBR 320 kbps with ID3
metadata.

Usage:
    python playlist_loopback_recorder.py playlist.json
    python playlist_loopback_recorder.py playlist.json -o tracks

Requirements:
    pip install PyAudioWPatch numpy mutagen
    ffmpeg must be available in PATH
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

import numpy as np
import pyaudiowpatch as pyaudio

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from spotify_soulseek_download import (  # noqa: E402
    apply_spotify_metadata,
    spotify_output_filename,
    spotify_output_filenames,
    track_from_report,
)


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
STALL_THRESHOLD_DBFS = -90.0
DEFAULT_STALL_TIMEOUT = 10.0
DEFAULT_START_TIMEOUT = 180.0
STALL_EXIT_CODE = 75
PROGRESS_FILENAME = "recording_progress.json"


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


class PlaybackStalled(RuntimeError):
    """Spotify stopped producing audio before the current track completed."""


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


def write_metadata(
    mp3_path: Path,
    track: dict[str, Any],
) -> None:
    """Use exactly the same Spotify metadata and cover policy as Soulseek files."""
    spotify = track.get("spotify")
    if not isinstance(spotify, dict):
        raise ValueError("la pista del manifest no contiene metadata Spotify")
    apply_spotify_metadata(mp3_path, track_from_report(spotify))


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


def track_identity(track: dict[str, Any]) -> str:
    spotify = track.get("spotify") or {}
    spotify_id = spotify.get("spotify_id") or spotify.get("id")
    if spotify_id:
        return str(spotify_id)
    uri = spotify.get("uri")
    if uri:
        return str(uri)
    return json.dumps(
        {
            "playlist_position": track.get("playlist_position"),
            "title": spotify.get("title"),
            "artists": spotify.get("artists"),
            "duration_ms": track.get("duration_ms"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def manifest_signature(tracks: list[dict[str, Any]]) -> str:
    payload = [
        {
            "identity": track_identity(track),
            "duration_seconds": duration_seconds(track),
        }
        for track in tracks
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def playback_watchdog_stalled(path: Path | None) -> bool:
    if path is None or not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("status") == "stalled"


def new_progress(
    json_file: Path,
    tracks: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "version": 1,
        "source_json": str(json_file.resolve()),
        "manifest_signature": manifest_signature(tracks),
        "total_tracks": len(tracks),
        "completed_count": 0,
        "completed_track_ids": [],
        "status": "recording",
        "attempt": 1,
        "tracks": [],
    }


def load_resume_progress(
    path: Path,
    json_file: Path,
    tracks: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_signature = manifest_signature(tracks)
    if not path.is_file():
        return new_progress(json_file, tracks)

    try:
        progress = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"No se pudo leer el checkpoint de grabación: {path}") from exc

    if not isinstance(progress, dict):
        raise RuntimeError(f"Checkpoint de grabación inválido: {path}")
    if progress.get("manifest_signature") != expected_signature:
        raise RuntimeError(
            "El JSON de faltantes cambió desde la grabación anterior; "
            "usa otro directorio de salida o elimina el checkpoint para empezar de cero."
        )
    if progress.get("total_tracks") != len(tracks):
        raise RuntimeError("El total de pistas no coincide con el checkpoint existente.")

    completed = progress.get("completed_count")
    rows = progress.get("tracks")
    identities = progress.get("completed_track_ids")
    if not isinstance(completed, int) or not 0 <= completed <= len(tracks):
        raise RuntimeError("completed_count inválido en el checkpoint de grabación.")
    if not isinstance(rows, list) or not isinstance(identities, list):
        raise RuntimeError("Listas inválidas en el checkpoint de grabación.")
    if len(rows) != completed or len(identities) != completed:
        raise RuntimeError("Checkpoint inconsistente: cantidad de pistas completadas.")

    expected_ids = [track_identity(track) for track in tracks[:completed]]
    if identities != expected_ids:
        raise RuntimeError("Checkpoint inconsistente: cambió el orden de las pistas.")
    for row in rows:
        output = Path(str(row.get("output") or ""))
        if not output.is_file() or output.stat().st_size <= 0:
            raise RuntimeError(
                f"Falta un MP3 marcado como completado en el checkpoint: {output}"
            )

    progress["attempt"] = int(progress.get("attempt") or 1) + 1
    progress["status"] = "recording"
    return progress


def flush_wave_for_reader(wav_file: wave.Wave_write) -> None:
    """Patch and flush the WAV header so ffmpeg can read a completed prefix."""
    wav_file.writeframes(b"")
    underlying = getattr(wav_file, "_file", None)
    if underlying is not None and hasattr(underlying, "flush"):
        underlying.flush()


def encode_and_checkpoint(
    *,
    ffmpeg: str,
    temp_wav: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
    track: dict[str, Any],
    manifest: dict[str, Any],
    global_index: int,
    total_tracks: int,
    boundary: BoundaryResult,
    progress: dict[str, Any],
    progress_path: Path,
) -> None:
    encode_segment(ffmpeg, temp_wav, output_path, start_sec, end_sec)
    write_metadata(output_path, track)

    declared = duration_seconds(track)
    progress["tracks"].append(
        {
            "index": global_index + 1,
            "playlist_position": track.get("playlist_position"),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "segment_duration_sec": end_sec - start_sec,
            "declared_duration_sec": declared,
            "duration_difference_sec": (end_sec - start_sec) - declared,
            "output": str(output_path),
            "attempt": progress["attempt"],
            "boundary": asdict(boundary),
        }
    )
    progress["completed_count"] = global_index + 1
    progress["completed_track_ids"].append(track_identity(track))
    progress["status"] = "recording"
    write_json_atomic(progress_path, progress)
    print(
        f"[{global_index + 1:03d}/{total_tracks:03d}] Guardado: {output_path.name}",
        flush=True,
    )


def make_output_filename(track: dict[str, Any], fallback_index: int) -> str:
    spotify = track.get("spotify")
    if not isinstance(spotify, dict):
        raise ValueError(f"la pista {fallback_index} no contiene metadata Spotify")
    return spotify_output_filename(track_from_report(spotify))


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
    parser.add_argument(
        "--ready-file",
        type=Path,
        default=None,
        help=(
            "Write a JSON readiness signal after the loopback stream is open; "
            "used by the workflow orchestrator"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from recording_progress.json and keep completed MP3 files",
    )
    parser.add_argument(
        "--stall-timeout",
        type=float,
        default=DEFAULT_STALL_TIMEOUT,
        help=(
            "Seconds of near-digital silence that indicate stopped playback "
            f"(default {DEFAULT_STALL_TIMEOUT:g})"
        ),
    )
    parser.add_argument(
        "--start-timeout",
        type=float,
        default=DEFAULT_START_TIMEOUT,
        help=(
            "Seconds to wait for the first pending track to start "
            f"(default {DEFAULT_START_TIMEOUT:g})"
        ),
    )
    parser.add_argument(
        "--watchdog-file",
        type=Path,
        default=None,
        help="Stop the current track cleanly when Spotify writes a stalled status",
    )
    args = parser.parse_args()

    if args.search_before <= 0 or args.search_after <= 0:
        parser.error("--search-before and --search-after must be > 0")
    if args.stall_timeout <= 0 or args.start_timeout <= 0:
        parser.error("--stall-timeout and --start-timeout must be > 0")

    manifest, tracks = load_manifest(args.json_file)
    ffmpeg = check_ffmpeg()

    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / PROGRESS_FILENAME
    if progress_path.exists() and not args.resume:
        raise RuntimeError(
            f"Ya existe {progress_path}. Usa --resume para conservar lo completado."
        )
    progress = load_resume_progress(progress_path, args.json_file, tracks)
    completed_before = int(progress["completed_count"])
    total_tracks = len(tracks)
    spotify_tracks = [
        track_from_report(track.get("spotify"))
        for track in tracks
    ]
    output_filenames = spotify_output_filenames(spotify_tracks)

    if completed_before >= total_tracks:
        progress["status"] = "completed"
        write_json_atomic(progress_path, progress)
        print(f"Todas las pistas ya estaban grabadas: {completed_before}/{total_tracks}")
        return 0

    write_json_atomic(progress_path, progress)

    fd, temp_name = tempfile.mkstemp(
        prefix="playlist_capture_", suffix=".wav", dir=str(args.output_dir)
    )
    os.close(fd)
    temp_wav = Path(temp_name)

    boundaries: list[BoundaryResult] = []
    playlist_start: float | None = None
    current_track_start: float | None = None
    next_track_index = completed_before
    expected_boundary: float | None = None
    capture_finished = False
    silent_frames = 0

    p: pyaudio.PyAudio | None = None
    stream = None
    wav_file = None
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mp3-checkpoint")
    futures: list[Future[None]] = []
    exit_code = 0
    capture_error: BaseException | None = None
    stopped_reason: str | None = None

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
        print(f"Tracks   : {total_tracks}")
        print(f"Completed: {completed_before}")
        print(f"Pending  : {total_tracks - completed_before}")
        print(f"Output   : {args.output_dir.resolve()}")
        print()
        if args.ready_file is not None:
            ready_file = args.ready_file.expanduser().resolve()
            ready_file.parent.mkdir(parents=True, exist_ok=True)
            ready_file.write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "device": str(device["name"]),
                        "sample_rate": rate,
                        "tracks": total_tracks,
                        "completed_count": completed_before,
                        "start_position": completed_before,
                        "remaining_tracks": total_tracks - completed_before,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        print("Waiting for the first track to start...", flush=True)
        print(
            "Start playlist playback now. Keep other system sounds off while recording.",
            flush=True,
        )

        start_confirm_blocks = 0
        blocks_needed = max(1, math.ceil((START_CONFIRM_MS / 1000.0) * rate / CHUNK))

        while not capture_finished:
            if playback_watchdog_stalled(args.watchdog_file):
                raise PlaybackStalled("El watchdog confirmó que Spotify se detuvo.")
            for future in futures:
                if future.done():
                    future.result()

            raw = stream.read(CHUNK, exception_on_overflow=False)
            wav_file.writeframesraw(raw)

            arr = np.frombuffer(raw, dtype="<i2")
            usable = len(arr) - (len(arr) % CHANNELS)
            arr = arr[:usable].reshape(-1, CHANNELS)
            rolling.append(arr)

            now_sec = rolling.total_frames / rate

            if playlist_start is None:
                if now_sec >= args.start_timeout:
                    raise PlaybackStalled(
                        "Spotify no inició la primera pista pendiente dentro del timeout."
                    )
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
                    expected_boundary = current_track_start + duration_seconds(
                        tracks[next_track_index]
                    )

                    spotify = tracks[next_track_index].get("spotify") or {}
                    title = spotify.get("title") or f"Track {next_track_index + 1}"
                    print(f"\nStarted   : {playlist_start:.3f} s")
                    print(f"Track {next_track_index + 1}: {title}")
                    print(f"Expected end: {expected_boundary:.3f} s")
                continue

            assert current_track_start is not None
            assert expected_boundary is not None

            if block_dbfs(arr) <= STALL_THRESHOLD_DBFS:
                silent_frames += len(arr)
            else:
                silent_frames = 0
            if silent_frames / rate >= args.stall_timeout:
                spotify = tracks[next_track_index].get("spotify") or {}
                title = spotify.get("title") or f"Track {next_track_index + 1}"
                raise PlaybackStalled(
                    f"Spotify dejó de producir audio durante «{title}»."
                )

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

            flush_wave_for_reader(wav_file)
            output_path = args.output_dir / output_filenames[next_track_index]
            futures.append(
                executor.submit(
                    encode_and_checkpoint,
                    ffmpeg=ffmpeg,
                    temp_wav=temp_wav,
                    output_path=output_path,
                    start_sec=current_track_start,
                    end_sec=result.detected_sec,
                    track=track,
                    manifest=manifest,
                    global_index=next_track_index,
                    total_tracks=total_tracks,
                    boundary=result,
                    progress=progress,
                    progress_path=progress_path,
                )
            )

            next_track_index += 1
            if next_track_index >= total_tracks:
                capture_finished = True
                break

            current_track_start = result.detected_sec
            expected_boundary = current_track_start + duration_seconds(tracks[next_track_index])
            silent_frames = 0

            spotify_next = tracks[next_track_index].get("spotify") or {}
            next_title = spotify_next.get("title") or f"Track {next_track_index + 1}"
            print(f"Track {next_track_index + 1}: {next_title}")
            print(f"Expected end: {expected_boundary:.3f} s")

    except KeyboardInterrupt:
        exit_code = 130
        stopped_reason = "Interrumpido por el usuario."
    except PlaybackStalled as exc:
        exit_code = STALL_EXIT_CODE
        stopped_reason = str(exc)
    except BaseException as exc:
        capture_error = exc

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

        executor.shutdown(wait=True)
        if capture_error is None:
            try:
                for future in futures:
                    future.result()
            except BaseException as exc:
                capture_error = exc

    if capture_error is not None:
        progress["status"] = "failed"
        progress["last_error"] = str(capture_error)
        progress["last_partial_capture"] = str(temp_wav)
        write_json_atomic(progress_path, progress)
        raise capture_error

    if exit_code:
        progress["status"] = "stalled" if exit_code == STALL_EXIT_CODE else "interrupted"
        progress["last_error"] = stopped_reason
        progress["last_partial_capture"] = str(temp_wav)
        write_json_atomic(progress_path, progress)
        print(f"\n{stopped_reason}", file=sys.stderr)
        print(
            f"Conservadas: {progress['completed_count']}/{total_tracks} pistas completas.",
            file=sys.stderr,
        )
        print(
            "Relanza el orquestador: Spotify comenzará en la primera pista pendiente.",
            file=sys.stderr,
        )
        print(f"Captura parcial conservada: {temp_wav}", file=sys.stderr)
        return exit_code

    if playlist_start is None or int(progress["completed_count"]) != total_tracks:
        raise RuntimeError("La captura terminó sin completar todos los checkpoints.")

    progress["status"] = "completed"
    progress["completed_at"] = datetime.now().isoformat(timespec="seconds")
    progress.pop("last_error", None)
    write_json_atomic(progress_path, progress)

    report = {
        "source_json": str(args.json_file),
        "sample_rate": rate,
        "channels": CHANNELS,
        "search_before_sec": args.search_before,
        "search_after_sec": args.search_after,
        "attempts": progress["attempt"],
        "completed_count": progress["completed_count"],
        "tracks": progress["tracks"],
    }
    report_path = args.output_dir / "segmentation_report.json"
    write_json_atomic(report_path, report)

    if not args.keep_wav:
        temp_wav.unlink(missing_ok=True)
    else:
        print(f"Temporary WAV kept: {temp_wav}")

    print(f"\nDone. {progress['completed_count']} MP3 files available.")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
