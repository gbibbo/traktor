#!/usr/bin/env python3
"""
PURPOSE: Run the complete Spotify -> Soulseek + loopback fallback workflow.

CHANGELOG:
- 2026-11-07: When the playlist is saved, its tracks get Beatport metadata (genre, label, artists, date)
              and the Vocal tag with the library's rules (src/v4/pipeline/tag_downloads.py), run with a
              Python that has the pipeline (pipeline_python). Gabriel: the downloads only had Spotify's
              metadata. It never fails the download.
- 2026-11-07: A recorded session pauses Spotify at once (its playlist plays on repeat, see
              spotify_playlist.py) and only then deletes its temporary playlist.
- 2026-11-07: With --recording-output, the player sends Spotify alone to that output before it plays,
              and when the download ends (finished, paused, stopped or failed) Spotify gets back the
              output it had (spotify_output.py, <runs>/spotify_output_restore.json).
- 2026-11-06: --recording-output NAME (or SPOTIFY_RECORDING_OUTPUT in .env): record that output, e.g.
              "Voicemeeter Input" with Spotify alone routed to it, instead of the Windows default output
              (CONFIGURAR_OTRO_ORDENADOR.md, «Grabar sin que suene nada»).
- 2026-11-06: The output that is recorded is checked before each session (playlist_loopback_recorder.py
              --check-device), before any Spotify playlist is created: an output another program holds
              (Traktor on the DJM, 2026-11-06: "Invalid device") is said so at the start, the download goes
              on, and it is tried again every DEVICE_RETRY_S; recording starts as soon as it is free. The
              same when the recorder ends with DEVICE_UNAVAILABLE_EXIT_CODE during a session. If it is
              still busy when the download ends, the run is "resumable" with that reason.
- 2026-11-06: The downloader works on --parallel tracks at once (2 by default) through one Soulseek session
              (a sockseek daemon); a daemon left by a downloader that was cut off is closed. The private
              Spotify playlist of each recording session is removed (DELETE /me/library) as soon as the
              session is recorded, or abandoned; one that could not be removed is retried at the end.
- 2026-11-06: Track by track (Gabriel's way of using Soulseek by hand): the downloader searches each
              track on its own and downloads it at once; what it cannot get goes to the recording queue
              (recording_queue.py) as soon as it is known, and this orchestrator records the queue from
              Spotify while the search goes on: recording starts with the first track to record, not
              after searching the whole playlist. Each recording session takes what is queued (up to 10
              tracks) into its own private Spotify playlist and its own folder (recordings/NN/) with the
              recorder's checkpoint. An empty queue waits for the next track or for the end of the
              download. Spotify stopping, the player or the loopback failing, or the session playlist
              not being created stop the recording but not the download: the run ends "resumable" with
              Soulseek's tracks already in the folder. Postponing during a session stops it at once
              (what was recorded is kept) and pauses Spotify; Spotify is also paused while the queue is
              empty, so its autoplay does not play on. A new run resumes the newest unfinished run of
              the playlist in this layout; runs made before this change are not resumed: a new run
              reuses their downloaded and recorded MP3s. Consolidation only requires the expected
              tracks; an extra copy of one of them is de-duplicated as before. Spotify is paused
              before each session's recorder opens (autoplay must not be taken as the first track's
              start); a player that timed out is not retried (it may have started playing); the
              checkpoints are matched by file name inside the session folder, so a run reached by
              another path keeps its MP3s; the run is marked final before its JSON is archived.
- 2026-11-02: The playlist's music folder (from an earlier run, inside the current destination) goes
              to the downloader as --have-dir and its tracks count as saved: a new run of a playlist
              only searches and downloads what is missing.
- 2026-11-02: The downloader gets the soulseek/ folders of earlier runs of the playlist (--reuse-dir):
              what an interrupted run already downloaded is not downloaded again.
- 2026-11-02: Postponed recording (--defer-recording, or DEFER_SIGNAL in the run while it
              downloads): Soulseek's tracks go to the library, the run ends "recording_deferred"
              (exit 75), and running the playlist again records only what is pending and adds it.
- 2026-10-30: Record from Spotify the tracks Soulseek could not deliver; a run that still misses
              tracks saves what it has ("partial", exit code 3).
- 2026-10-30: Load .env before parsing so it can set the runs and playlists roots.
- 2026-10-05: Revalidate final MP3s and existing duplicates before consolidation.
- 2026-09-29: Record missing tracks sequentially in Spotify batches of up to 10.
- 2026-09-27: Consolidate successful runs into one named playlist directory.
- 2026-09-27: Initial orchestration with manifest and recorder readiness handshakes.

Usage:
    python spotify_soulseek_orchestrator.py "https://open.spotify.com/playlist/..."

After one-time environment setup, the Spotify playlist URL is the only required
argument. The script starts the track-by-track Soulseek downloader, records from
Spotify every track it queues while the downloads go on, waits for both branches
to finish and consolidates the playlist into one folder.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from recording_queue import END_FINISHED, QUEUE_NAME, QueueState, read_queue, track_id  # noqa: E402
import spotify_output  # noqa: E402

DEFAULT_RUNS_DIR = SCRIPT_DIR / "runs"
DEFAULT_PLAYLISTS_DIR = SCRIPT_DIR / "Playlists_DOWNLOAD"
POLL_SECONDS = 0.25
IDLE_POLL_SECONDS = 1.0
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"}
LAYOUT = 2                             # tema a tema (2026-11-06); los runs sin esta marca no se retoman
RECORDINGS_DIR = "recordings"          # una carpeta por tanda grabada: recordings/01, recordings/02…
SESSION_FILE = "session.json"          # la tanda: sus temas (manifiesto del grabador) y su playlist
PROGRESS_FILE = "recording_progress.json"  # checkpoint del grabador (playlist_loopback_recorder.py)
SESSION_SIZE = 10                      # temas por tanda, como las playlists de grabación de antes
PLAYLIST_CREATE_PAUSES_S = (5.0, 15.0)  # reintentos al crear la playlist de una tanda
PLAYER_ATTEMPTS = 2                    # Spotify recién abierto puede tardar más que el tope en aparecer
DEVICE_UNAVAILABLE_EXIT_CODE = 76      # playlist_loopback_recorder.py: la salida no se puede grabar
DEVICE_RETRY_S = 30.0                  # con la salida ocupada, cada cuánto se vuelve a probar
# Runs de antes del 2026-11-06 (preflight en lote): la app todavía los muestra.
FAILED_JSON_NAME = "soulseek_failed.json"
RECORDED_FAILED_DIR = "recorded_failed"
OLD_RECORDING_DIRS = ("recorded_missing", RECORDED_FAILED_DIR)
EXIT_PARTIAL = 3
EXIT_RESUMABLE = 75
# Mientras exista en el run, no se graba: si una tanda está grabando, se corta (lo grabado queda) y Spotify
# se pausa; lo de Soulseek se sigue bajando. La app lo crea o lo borra durante la descarga.
DEFER_SIGNAL = "defer_recording.signal"


class WorkflowError(RuntimeError):
    """A user-facing orchestration failure."""


class ResumeAvailable(WorkflowError):
    """Recording stopped, but completed per-track checkpoints remain reusable."""


class IncompleteAudio(WorkflowError):
    """The final audio set does not cover the whole playlist."""


class RecordingUnavailable(RuntimeError):
    """The recording cannot go on in this run (Spotify, the player, the loopback or the session playlist)."""


class OutputBusy(RecordingUnavailable):
    """The output that is recorded cannot be opened now (another program holds it): try again later."""


def playlist_id_from_url(value: str) -> str:
    """Return a Spotify playlist ID from a web URL, Markdown link, URI, or ID."""
    value = value.strip()
    match = re.search(r"open\.spotify\.com/playlist/([A-Za-z0-9]+)", value)
    if match:
        return match.group(1)
    if value.startswith("spotify:playlist:"):
        value = value.rsplit(":", 1)[-1]
    if value and value.isalnum():
        return value
    raise WorkflowError(f"No pude interpretar la playlist de Spotify: {value}")


def canonical_playlist_url(value: str) -> str:
    playlist_id = playlist_id_from_url(value)
    return f"https://open.spotify.com/playlist/{playlist_id}"


def load_environment() -> None:
    """Load the usual .env file without printing any credential values."""
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise WorkflowError(
            "Falta python-dotenv. Instala download_JIJIJI/requirements.txt."
        ) from exc

    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(SCRIPT_DIR / ".env")


def validate_environment() -> None:
    if os.name != "nt":
        raise WorkflowError(
            "La reproducción y la captura loopback de este flujo requieren Windows."
        )

    missing = [
        name
        for name in (
            "SPOTIFY_CLIENT_ID",
            "SPOTIFY_CLIENT_SECRET",
            "SPOTIFY_REDIRECT_URI",
        )
        if not os.environ.get(name)
    ]
    if missing:
        raise WorkflowError(
            "Faltan variables de entorno: " + ", ".join(missing)
        )

    required_modules = {
        "requests": "requests",
        "mutagen": "mutagen",
        "PIL": "Pillow",
        "spotipy": "spotipy",
        "numpy": "numpy",
        "pyaudiowpatch": "PyAudioWPatch",
    }
    unavailable: list[str] = []
    for module, package in required_modules.items():
        try:
            __import__(module)
        except ImportError:
            unavailable.append(package)
    if unavailable:
        raise WorkflowError(
            "Faltan paquetes Python: "
            + ", ".join(unavailable)
            + ". Instala download_JIJIJI/requirements.txt."
        )


def find_required_program(name: str) -> Path:
    found = shutil.which(name)
    if found:
        return Path(found).resolve()

    candidates = [
        REPO_ROOT / "tools" / name / f"{name}.exe",
        REPO_ROOT / "tools" / name / name,
    ]
    if name == "ffmpeg" and os.environ.get("LOCALAPPDATA"):
        winget_packages = (
            Path(os.environ["LOCALAPPDATA"])
            / "Microsoft"
            / "WinGet"
            / "Packages"
        )
        candidates.extend(
            sorted(
                winget_packages.glob("Gyan.FFmpeg_*/*/bin/ffmpeg.exe"),
                reverse=True,
            )
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise WorkflowError(f"No se encontró '{name}' en PATH ni en tools/{name}/.")


def child_environment(sockseek: Path, ffmpeg: Path) -> dict[str, str]:
    env = os.environ.copy()
    extra_dirs = [str(sockseek.parent), str(ffmpeg.parent)]
    existing = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(extra_dirs + ([existing] if existing else []))
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    return env


def read_json_dict(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _runs_for_playlist(runs_root: Path, playlist_id: str) -> list[Path]:
    if not runs_root.is_dir():
        return []
    return sorted(
        (
            path
            for path in runs_root.iterdir()
            if path.is_dir() and path.name.endswith(f"_{playlist_id}")
        ),
        key=lambda path: path.name,
        reverse=True,
    )


def preflight_failed_entirely(run_dir: Path) -> str | None:
    """Si la consulta a Soulseek falló para TODOS los temas, el error común; si no, None.

    Es un problema de la búsqueda (tiempo agotado, sesión), no de los temas: no hay que grabarlos.
    """
    reports = sorted(run_dir.glob("soulseek_availability*.json"), key=lambda path: path.stat().st_mtime)
    availability = read_json_dict(reports[-1]) if reports else None
    rows = [row for row in (availability or {}).get("tracks") or [] if isinstance(row, dict)]
    if not rows or any(row.get("status") != "error" for row in rows):
        return None
    return str(rows[0].get("error") or "error desconocido")


def resume_point(runs_root: Path, playlist_id: str) -> Path | None:
    """The newest run of the playlist, if it is a track-by-track run that did not finish.

    Interrupted, resumable (Spotify stopped), postponed or cut off while running: everything it did is
    checkpointed (downloads, recording queue, sessions), so it goes on from there. A finished, partial
    or failed run, or one from before 2026-11-06, starts a new run (which reuses its audio)."""
    for run_dir in _runs_for_playlist(runs_root, playlist_id):
        report = read_json_dict(run_dir / "orchestration_report.json") or {}
        if report.get("layout") != LAYOUT:
            return None
        if report.get("status") in {"completed", "partial", "failed"}:
            return None
        return run_dir
    return None


def wait_for_ready_file(
    process: subprocess.Popen[Any],
    path: Path,
    timeout: float,
    label: str = "proceso",
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict) and payload.get("status") == "ready":
                return
        return_code = process.poll()
        if return_code is not None:
            raise WorkflowError(
                f"El {label} terminó antes de quedar listo "
                f"(código {return_code})."
            )
        time.sleep(POLL_SECONDS)
    raise WorkflowError(f"El {label} no quedó listo dentro del tiempo esperado.")


def terminate_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    """Rewritten while the app reads it: never a half-written file (on Windows the replace waits for a
    reader to close it)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 * min(attempt + 1, 5))
    os.replace(tmp, path)


def safe_playlist_folder_name(name: str) -> str:
    """Return a portable Windows folder name for a Spotify playlist."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().rstrip(".")
    return cleaned[:120].rstrip() or "Playlist sin nombre"


def embedded_spotify_id(path: Path) -> str:
    """Read the canonical Spotify track ID required for safe de-duplication."""
    try:
        from mutagen.id3 import ID3

        tags = ID3(path)
    except Exception as exc:
        raise WorkflowError(f"MP3 sin metadata ID3 utilizable: {path}: {exc}") from exc
    for frame in tags.getall("TXXX"):
        if frame.desc == "Spotify Track ID" and frame.text:
            return str(frame.text[0])
    raise WorkflowError(f"MP3 sin Spotify Track ID: {path}")


def _collision_destination(
    playlist_dir: Path,
    source: Path,
    spotify_id: str,
) -> tuple[Path, bool]:
    """Choose the final path and report whether an identical track exists."""
    destination = playlist_dir / source.name
    if not destination.exists():
        return destination, False
    if _validate_final_mp3(destination) == spotify_id:
        return destination, True

    alternate = playlist_dir / f"{source.stem} [{spotify_id}]{source.suffix}"
    if not alternate.exists():
        return alternate, False
    if _validate_final_mp3(alternate) == spotify_id:
        return alternate, True
    raise WorkflowError(f"Colisión de audio no resoluble: {alternate}")


def _validate_final_mp3(path: Path) -> str:
    """Use the downloader's audio/size/AV gate before reading identity or moving files."""
    from spotify_soulseek_download import _validate_download

    try:
        if path.suffix.lower() != ".mp3":
            raise RuntimeError("se requiere MP3 final")
        _validate_download(path)
        spotify_id = embedded_spotify_id(path)
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", spotify_id):
            raise RuntimeError("Spotify Track ID inválido")
        return spotify_id
    except (RuntimeError, OSError) as exc:
        raise WorkflowError(f"MP3 bloqueado durante consolidación: {path}: {exc}") from exc


def consolidate_audio(
    run_dir: Path,
    soulseek_dir: Path,
    recorded_dir: Path | None,
    playlists_root: Path,
    playlist_name: str,
    expected_spotify_ids: set[str] | None = None,
    *,
    extra_recorded_dirs: tuple[Path, ...] = (),
) -> tuple[Path, dict[str, str], dict[str, Any]]:
    """Move every final MP3 into one playlist directory without duplicates.

    With `expected_spotify_ids`, a missing expected track raises IncompleteAudio before moving
    anything; without it, whatever valid audio exists is moved (partial save). Without
    `recorded_dir`, only what came from Soulseek moves (recording postponed).
    """
    run_root = run_dir.resolve()
    sources: list[tuple[str, Path]] = []
    roots = [("soulseek", soulseek_dir)] + ([("loopback", recorded_dir)] if recorded_dir is not None else [])
    roots += [("loopback", extra) for extra in extra_recorded_dirs]
    for origin, root in roots:
        if root.is_dir():
            sources.extend(
                (origin, path)
                for path in sorted(root.rglob("*.mp3"), key=lambda item: str(item).casefold())
                if path.is_file()
            )

    identified_sources = []
    for origin, source in sources:
        if source.is_symlink() or not source.resolve().is_relative_to(run_root):
            raise WorkflowError(f"Audio fuera del run rechazado: {source}")
        identified_sources.append((origin, source, _validate_final_mp3(source)))
    found_ids = {spotify_id for _, _, spotify_id in identified_sources}
    if expected_spotify_ids is not None and expected_spotify_ids - found_ids:
        # Una copia de más de un tema de la playlist no es un problema: se deduplica al moverla.
        missing = sorted(expected_spotify_ids - found_ids)
        raise IncompleteAudio(
            "El conjunto de audio final no coincide con la playlist: "
            f"faltan={missing}"
        )

    playlist_dir = playlists_root.resolve() / safe_playlist_folder_name(playlist_name)
    playlist_dir.mkdir(parents=True, exist_ok=True)
    moved_paths: dict[str, str] = {}
    rows: list[dict[str, Any]] = []

    for origin, source, spotify_id in identified_sources:
        resolved_source = source.resolve()
        if not resolved_source.is_relative_to(run_root):
            raise WorkflowError(f"Audio fuera del run rechazado: {source}")
        destination, duplicate = _collision_destination(
            playlist_dir,
            source,
            spotify_id,
        )
        source_key = str(resolved_source)
        if duplicate:
            source.unlink()
            status = "deduplicated"
        else:
            shutil.move(str(source), str(destination))
            status = "moved"
        moved_paths[source_key] = str(destination.resolve())
        rows.append(
            {
                "spotify_id": spotify_id,
                "origin": origin,
                "source": str(source),
                "destination": str(destination),
                "status": status,
            }
        )

    report = {
        "playlist_name": playlist_name,
        "playlist_directory": str(playlist_dir),
        "audio_count": len(rows),
        "moved": sum(row["status"] == "moved" for row in rows),
        "deduplicated": sum(row["status"] == "deduplicated" for row in rows),
        "tracks": rows,
    }
    return playlist_dir, moved_paths, report


def _rewrite_moved_outputs(path: Path, moved_paths: dict[str, str]) -> None:
    payload = read_json_dict(path)
    if payload is None:
        return
    for row in payload.get("tracks") or []:
        if not isinstance(row, dict) or not row.get("output"):
            continue
        try:
            key = str(Path(str(row["output"])).resolve())
        except OSError:
            continue
        if key in moved_paths:
            row["output"] = moved_paths[key]
    payload.pop("last_partial_capture", None)
    write_summary(path, payload)


def _move_retrying(source: Path, destination: Path) -> None:
    """On Windows a rename fails while the app has the JSON open (it polls every 1.5 s): retry briefly."""
    for attempt in range(40):
        try:
            shutil.move(str(source), str(destination))
            return
        except PermissionError:
            time.sleep(0.05 * min(attempt + 1, 5))
    shutil.move(str(source), str(destination))


def archive_run(run_dir: Path, moved_paths: dict[str, str]) -> Path:
    """simplify_completed_run after the run is already saved: a failure here only leaves JSON unarchived."""
    try:
        return simplify_completed_run(run_dir, moved_paths)
    except OSError as exc:
        print(f"Aviso: no se pudieron archivar todos los JSON del run ({exc}); la música ya está guardada.",
              file=sys.stderr)
        return run_dir / "diagnostics"


def simplify_completed_run(run_dir: Path, moved_paths: dict[str, str]) -> Path:
    """Keep a small readable root plus detailed JSON under diagnostics/ (the sessions keep their folders:
    diagnostics/recordings/NN/)."""
    diagnostics = run_dir / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    recordings = run_dir / RECORDINGS_DIR

    for json_path in list(run_dir.rglob("*.json")):
        if diagnostics in json_path.parents:
            continue
        if json_path.name in {
            "orchestration_report.json",
            "soulseek_missing.json",
            FAILED_JSON_NAME,
            "consolidation_report.json",
        } and json_path.parent == run_dir:
            continue
        if json_path.name in {"player_ready.json", "recorder_ready.json"}:
            json_path.unlink(missing_ok=True)
            continue
        _rewrite_moved_outputs(json_path, moved_paths)
        if recordings in json_path.parents:
            destination = diagnostics / json_path.relative_to(run_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
        else:
            destination = diagnostics / json_path.name
            if destination.exists():
                destination = diagnostics / (
                    f"{json_path.parent.name}_{json_path.name}"
                )
        _move_retrying(json_path, destination)

    for signal in run_dir.glob("*.signal"):
        signal.unlink(missing_ok=True)
    for text_file in run_dir.rglob("*.txt"):
        text_file.unlink(missing_ok=True)
    for tmp_file in run_dir.rglob("*.tmp"):
        if diagnostics not in tmp_file.parents:
            tmp_file.unlink(missing_ok=True)

    remaining_mp3 = [path for path in run_dir.rglob("*.mp3") if path.is_file()]
    if remaining_mp3:
        raise WorkflowError(
            "Quedó audio MP3 sin consolidar: "
            + ", ".join(str(path) for path in remaining_mp3)
        )
    for path in list(run_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            path.unlink()

    for directory in sorted(
        (path for path in run_dir.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        if directory != diagnostics and diagnostics not in directory.parents:
            try:
                directory.rmdir()
            except OSError:
                pass
    return diagnostics


# ---------------------------------------------------------------------------
# Tandas de grabación (recordings/NN)
# ---------------------------------------------------------------------------

def session_dirs(run_dir: Path) -> list[Path]:
    root = run_dir / RECORDINGS_DIR
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / SESSION_FILE).is_file())


def read_session(session_dir: Path) -> dict[str, Any]:
    return read_json_dict(session_dir / SESSION_FILE) or {}


def session_tracks(session: dict[str, Any]) -> list[dict[str, Any]]:
    return [t for t in session.get("tracks") or [] if isinstance(t, dict)]


def session_completed(session_dir: Path) -> int:
    progress = read_json_dict(session_dir / PROGRESS_FILE) or {}
    completed = progress.get("completed_count")
    return completed if isinstance(completed, int) and completed >= 0 else 0


def session_recorded_ids(session_dir: Path) -> list[str]:
    """Spotify IDs that the recorder checkpointed in this session, in order."""
    progress = read_json_dict(session_dir / PROGRESS_FILE) or {}
    ids = progress.get("completed_track_ids")
    return [str(i) for i in ids] if isinstance(ids, list) else []


def session_finished(session_dir: Path) -> bool:
    session = read_session(session_dir)
    return bool(session.get("abandoned")) or session_completed(session_dir) >= len(session_tracks(session))


def recording_plan(run_dir: Path, queue: QueueState) -> tuple[list[Path], list[dict[str, Any]]]:
    """(Unfinished sessions in order, queued tracks that no session holds yet)."""
    assigned: set[str] = set()
    pending: list[Path] = []
    for session_dir in session_dirs(run_dir):
        session = read_session(session_dir)
        if session.get("abandoned"):
            assigned |= set(session_recorded_ids(session_dir))
            continue
        assigned |= {track_id(t) for t in session_tracks(session)}
        if not session_finished(session_dir):
            pending.append(session_dir)
    return pending, [t for t in queue.tracks if track_id(t) not in assigned]


def pending_recordings(run_dir: Path, queue: QueueState) -> int:
    """Tracks still to record: in the queue and not yet recorded by any session."""
    pending, unassigned = recording_plan(run_dir, queue)
    return len(unassigned) + sum(
        len(session_tracks(read_session(d))) - session_completed(d) for d in pending
    )


def create_session(run_dir: Path, tracks: list[dict[str, Any]]) -> Path:
    root = run_dir / RECORDINGS_DIR
    # Todas las carpetas numeradas: una que quedó sin session.json (corte justo al crearla) no se reutiliza.
    numbers = [int(d.name) for d in root.iterdir() if d.is_dir() and d.name.isdigit()] if root.is_dir() else []
    session_dir = root / f"{max(numbers, default=0) + 1:02d}"
    session_dir.mkdir(parents=True, exist_ok=False)
    entries = [{k: v for k, v in t.items() if k not in {"type", "queued_at"}} for t in tracks]
    write_summary(session_dir / SESSION_FILE, {
        "number": int(session_dir.name),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "missing_count": len(entries),
        "playlist": None,
        "tracks": entries,
    })
    return session_dir


def relocate_checkpoint(session_dir: Path) -> None:
    """The recorder's checkpoint keeps absolute paths: if the run was moved (another runs folder, another
    computer), point them at the session's folder, where the MP3s are, before the recorder checks them."""
    path = session_dir / PROGRESS_FILE
    progress = read_json_dict(path)
    if not progress:
        return
    changed = False
    for row in progress.get("tracks") or []:
        if not isinstance(row, dict) or not row.get("output"):
            continue
        output = Path(str(row["output"]))
        here = session_dir / output.name
        if not output.is_file() and here.is_file():
            row["output"] = str(here)
            changed = True
    if changed:
        write_summary(path, progress)


def session_checkpoint_usable(session_dir: Path) -> bool:
    """The recorder can resume only if the MP3s its checkpoint lists are still there (a partial save from
    the app, or a postponed recording, may have moved them to the music folder)."""
    progress = read_json_dict(session_dir / PROGRESS_FILE)
    if progress is None:
        return True
    for row in progress.get("tracks") or []:
        name = Path(str((row or {}).get("output") or "")).name
        if not name or not (session_dir / name).is_file():
            return False
    return True


def abandon_session(session_dir: Path) -> None:
    """Its recorded tracks count as recorded; the rest go back to the queue for a new session."""
    session = read_session(session_dir)
    session["abandoned"] = True
    session["completed_ids"] = session_recorded_ids(session_dir)
    write_summary(session_dir / SESSION_FILE, session)
    forget_session_playlist(session_dir)


def discard_unfinished_recordings(run_dir: Path) -> None:
    """An MP3 the recorder did not checkpoint was cut off while being written: it never goes to the folder.
    Compared by file name inside the session's folder, so a run reached by another path keeps its MP3s."""
    for session_dir in session_dirs(run_dir):
        progress = read_json_dict(session_dir / PROGRESS_FILE) or {}
        kept = {Path(str((row or {}).get("output") or "")).name
                for row in progress.get("tracks") or [] if isinstance(row, dict)}
        for mp3 in session_dir.glob("*.mp3"):
            if mp3.name not in kept:
                mp3.unlink(missing_ok=True)


def create_spotify_playlist(source_name: str, tracks: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """A private Spotify playlist with the session's tracks, in order (the downloader's authorization)."""
    from spotify_soulseek_download import Spotify, track_from_report

    client = Spotify(os.environ.get("SPOTIFY_CLIENT_ID", ""))
    return client.create_missing_playlist(
        source_name, [track_from_report(t["spotify"]) for t in tracks], name=name
    )


def ensure_session_playlist(session_dir: Path, playlist_name: str) -> str:
    session = read_session(session_dir)
    url = (session.get("playlist") or {}).get("url")
    if url:
        return str(url)
    name = f"Grabar - {playlist_name} - {datetime.now():%Y-%m-%d %H%M} [{session_dir.name}]"
    last_error = "Spotify no devolvió la playlist"
    for attempt in range(len(PLAYLIST_CREATE_PAUSES_S) + 1):
        if attempt:
            pause = PLAYLIST_CREATE_PAUSES_S[attempt - 1]
            print(f"No se pudo crear la playlist de la tanda ({last_error}); reintento en {pause:.0f} s.", flush=True)
            time.sleep(pause)
        try:
            created = create_spotify_playlist(playlist_name, session_tracks(session), name)
        except Exception as exc:  # noqa: BLE001 (red, 429, autorización: se reintenta y, si no, se pospone)
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        if created.get("url"):
            session["playlist"] = created
            write_summary(session_dir / SESSION_FILE, session)
            return str(created["url"])
    raise RecordingUnavailable(f"No se pudo crear la playlist de Spotify para grabar ({last_error}).")


def delete_spotify_playlist(playlist: dict[str, Any]) -> None:
    """Quita de tu Spotify la playlist temporal de una tanda (DELETE /me/library: para una playlist propia,
    borrarla)."""
    from spotify_soulseek_download import Spotify

    Spotify(os.environ.get("SPOTIFY_CLIENT_ID", "")).remove_from_library([f"spotify:playlist:{playlist['id']}"])


def forget_session_playlist(session_dir: Path) -> None:
    """La playlist de la tanda ya no hace falta (grabada o abandonada): se borra. Si no se puede, queda anotado y
    se reintenta al terminar; nunca corta la descarga."""
    session = read_session(session_dir)
    playlist = session.get("playlist") or {}
    if not playlist.get("id") or session.get("playlist_deleted"):
        return
    try:
        delete_spotify_playlist(playlist)
    except Exception as exc:  # noqa: BLE001 (red, permisos: la playlist queda, nada más)
        session["playlist_delete_error"] = f"{type(exc).__name__}: {exc}"
        print(f"No se pudo borrar de tu Spotify la playlist temporal «{playlist.get('name')}» ({exc}).", flush=True)
    else:
        session["playlist_deleted"] = True
        session.pop("playlist_delete_error", None)
        print(f"Playlist temporal borrada de tu Spotify: «{playlist.get('name')}».", flush=True)
    write_summary(session_dir / SESSION_FILE, session)


def stop_leftover_daemon(run_dir: Path) -> None:
    """El daemon de sockseek de un descargador cortado sigue con la sesión de Soulseek abierta: se cierra."""
    state = run_dir / "sockseek_daemon.json"
    if not state.exists():
        return
    try:
        from spotify_soulseek_download import stop_daemon_from_state

        stop_daemon_from_state(state)
    except Exception:  # noqa: BLE001
        pass


def check_recording_device(env: dict[str, str], output: str | None = None) -> tuple[bool, str]:
    """¿Se puede grabar ahora la salida elegida (o la predeterminada)? (ok, nombre de la salida o el problema)."""
    cmd = [sys.executable, "-u", str(SCRIPT_DIR / "playlist_loopback_recorder.py"), "--check-device"]
    if output:
        cmd += ["--output-device", output]
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"No se pudo comprobar la salida de audio ({exc})."
    info: dict[str, Any] = {}
    for line in reversed(result.stdout.splitlines()):
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            info = parsed
            break
    if result.returncode == 0 and info.get("ok"):
        return True, str(info.get("device") or "")
    tail = (result.stderr or "").strip().splitlines()[-1:] or [f"código {result.returncode}"]
    return False, str(info.get("error") or f"No se pudo comprobar la salida de audio ({tail[0]}).")


def pause_spotify(env: dict[str, str], device_id: str | None) -> None:
    """Pausa Spotify en este ordenador (no en el teléfono), para que su reproducción automática no siga
    sonando cuando no hay nada que grabar. Si falla, solo queda sonando."""
    if not device_id:
        return
    cmd = [sys.executable, "-u", str(SCRIPT_DIR / "spotify_playlist.py"), "--pause", "--device-id", device_id]
    try:
        subprocess.run(cmd, env=env, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        pass


class SessionRecorder:
    """Graba las tandas: el grabador por loopback y el reproductor de Spotify, uno por tanda."""

    def __init__(self, *, run_dir: Path, env: dict[str, str], args: argparse.Namespace,
                 summary: dict[str, Any], summary_path: Path, defer_file: Path):
        self.run_dir = run_dir
        self.env = env
        self.args = args
        self.summary = summary
        self.summary_path = summary_path
        self.defer_file = defer_file
        self.device_id: str | None = None   # el Spotify de este ordenador, de la última tanda
        self.spotify_playing = False
        self.recorder: subprocess.Popen[Any] | None = None
        self.player: subprocess.Popen[Any] | None = None

    def pause(self) -> None:
        if self.spotify_playing:
            pause_spotify(self.env, self.device_id)
            self.spotify_playing = False

    def stop_processes(self) -> None:
        terminate_process(self.player)
        terminate_process(self.recorder)
        self.player = self.recorder = None

    def record(self, session_dir: Path, url: str) -> str:
        """Graba lo que falta de la tanda. "completed", o "deferred" si se pospuso durante la tanda;
        RecordingUnavailable si Spotify, el reproductor o el grabador no siguen."""
        recorder_ready = self.run_dir / "recorder_ready.json"
        player_ready = self.run_dir / "player_ready.json"
        playback_status = self.run_dir / "spotify_playback_status.json"
        for path in (recorder_ready, player_ready, playback_status):
            path.unlink(missing_ok=True)
        session = read_session(session_dir)
        total = len(session_tracks(session))
        completed = session_completed(session_dir)
        label = f"tanda {session_dir.name}"
        # Lo que siga sonando (la reproducción automática después de la tanda anterior) no puede ser el
        # comienzo que el grabador toma como primer tema: se pausa antes de abrirlo.
        self.pause()
        recorder_cmd = [
            sys.executable, "-u", str(SCRIPT_DIR / "playlist_loopback_recorder.py"),
            str(session_dir / SESSION_FILE),
            "--output-dir", str(session_dir),
            "--ready-file", str(recorder_ready),
            "--resume",
            "--watchdog-file", str(playback_status),
        ]
        if getattr(self.args, "recording_output", None):
            recorder_cmd += ["--output-device", self.args.recording_output]
        print(f"\n[grabación] Abriendo grabador ({label}: {total - completed} tema(s))...", flush=True)
        deferred = False
        try:
            self.recorder = subprocess.Popen(recorder_cmd, env=self.env)
            wait_for_ready_file(self.recorder, recorder_ready, self.args.recorder_timeout, label="grabador")
            print(f"\n[grabación] Reproduciendo en Spotify la {label} desde su tema {completed + 1}...", flush=True)
            player_cmd = [
                sys.executable, "-u", str(SCRIPT_DIR / "spotify_playlist.py"), url,
                "--offset-position", str(completed),
                "--ready-file", str(player_ready),
                "--watchdog-file", str(playback_status),
            ]
            if getattr(self.args, "recording_output", None):
                # Spotify solo a la salida que se graba; la que tenía vuelve al terminar la descarga.
                player_cmd += ["--route-output", self.args.recording_output,
                               "--route-state", str(self.run_dir.parent / spotify_output.STATE_NAME)]
            for attempt in range(PLAYER_ATTEMPTS):
                self.player = subprocess.Popen(player_cmd, env=self.env)
                try:
                    wait_for_ready_file(self.player, player_ready, self.args.recorder_timeout,
                                        label="reproductor de Spotify")
                    break
                except WorkflowError as exc:
                    gave_up = self.player.poll() is not None  # terminó solo: no llegó a reproducir nada
                    terminate_process(self.player)
                    early = read_json_dict(player_ready) or {}
                    self.device_id = early.get("device_id") or self.device_id
                    player_ready.unlink(missing_ok=True)
                    if not gave_up:
                        # Se agotó el tiempo: pudo empezar a sonar, y el grabador ya habría tomado ese
                        # comienzo. Otro intento reiniciaría el tema y correría los cortes: se pausa.
                        self.spotify_playing = True
                    elif attempt + 1 < PLAYER_ATTEMPTS:
                        # Spotify de escritorio recién abierto tarda en aparecer: un segundo intento.
                        print(f"[grabación] {exc} Reintento.", flush=True)
                        continue
                    # Si la playlist de la tanda ya no está en Spotify, la próxima vez se crea otra (y esta se borra).
                    forget_session_playlist(session_dir)
                    session = read_session(session_dir)
                    session["playlist"] = None
                    session.pop("playlist_deleted", None)
                    write_summary(session_dir / SESSION_FILE, session)
                    raise
            self.device_id = (read_json_dict(player_ready) or {}).get("device_id") or self.device_id
            self.spotify_playing = True
            self.summary["recording_session"] = session_dir.name
            write_summary(self.summary_path, self.summary)

            while self.recorder.poll() is None:
                if self.defer_file.exists() and not deferred:
                    # Posponer corta la tanda ya: el grabador guarda lo completo y se detiene.
                    deferred = True
                    print("\n[grabación] Grabación pospuesta: se corta la tanda; lo grabado se conserva.", flush=True)
                    write_summary(playback_status, {"status": "stalled", "reason": "deferred",
                                                    "error": "Grabación pospuesta desde la app"})
                player_code = self.player.poll()
                if player_code is not None and player_code not in {0, 75}:
                    write_summary(playback_status, {"status": "stalled",
                                                    "error": f"Spotify player exited with code {player_code}"})
                time.sleep(0.5)
            recorder_code = int(self.recorder.returncode or 0)
        except WorkflowError as exc:
            if self.recorder is not None and self.recorder.poll() == DEVICE_UNAVAILABLE_EXIT_CODE:
                progress = read_json_dict(session_dir / PROGRESS_FILE) or {}
                raise OutputBusy(str(progress.get("last_error") or "La salida de audio no se puede grabar.")) from exc
            raise RecordingUnavailable(str(exc)) from exc
        finally:
            self.stop_processes()
            self.summary.pop("recording_session", None)
            write_summary(self.summary_path, self.summary)

        done = session_completed(session_dir)
        if recorder_code == 0 and done >= total:
            print(f"Tanda {session_dir.name} grabada: {total} tema(s).", flush=True)
            return "completed"
        if recorder_code in {75, 130} and deferred:
            return "deferred"
        if recorder_code == DEVICE_UNAVAILABLE_EXIT_CODE:
            progress = read_json_dict(session_dir / PROGRESS_FILE) or {}
            raise OutputBusy(str(progress.get("last_error") or "La salida de audio no se puede grabar."))
        if recorder_code in {75, 130}:
            progress = read_json_dict(session_dir / PROGRESS_FILE) or {}
            reason = progress.get("last_error") or "Spotify se detuvo"
            raise RecordingUnavailable(f"{reason} Se conservaron {done}/{total} temas de la {label}.")
        raise RecordingUnavailable(f"El grabador de la {label} terminó con código {recorder_code}.")


def playlist_identity(
    run_dir: Path,
    playlist_id: str,
) -> tuple[str, set[str] | None]:
    """Playlist name and the Spotify IDs it should contain, from the newest availability report."""
    reports = sorted(
        run_dir.glob("soulseek_availability*.json"),
        key=lambda path: path.stat().st_mtime,
    )
    availability = read_json_dict(reports[-1]) if reports else None
    missing = read_json_dict(run_dir / "soulseek_missing.json") or {}
    name = str(
        ((availability or {}).get("playlist") or {}).get("name")
        or (missing.get("source_playlist") or {}).get("name")
        or playlist_id
    )
    if availability is None:
        return name, None
    expected = {
        str((row.get("spotify") or {}).get("spotify_id"))
        for row in availability.get("tracks") or []
        if isinstance(row, dict) and (row.get("spotify") or {}).get("spotify_id")
    }
    return name, expected


def write_playlist_m3u8(playlist_dir: Path, run_dir: Path) -> Path | None:
    """The playlist in Spotify's order, for Rekordbox and Traktor: UTF-8, absolute paths and CRLF, like
    src/v4/common/dj_export.py. Rewritten from the folder on every save, so tracks recorded later take
    their place in the order."""
    reports = sorted(
        [*run_dir.glob("soulseek_availability*.json"), *(run_dir / "diagnostics").glob("soulseek_availability*.json")],
        key=lambda path: path.stat().st_mtime,
    )
    availability = read_json_dict(reports[-1]) if reports else None
    if not availability or not playlist_dir.is_dir():
        return None
    files: dict[str, Path] = {}
    for path in sorted(playlist_dir.glob("*.mp3"), key=lambda item: str(item).casefold()):
        try:
            files.setdefault(embedded_spotify_id(path), path)
        except WorkflowError:
            continue
    rows = [row for row in availability.get("tracks") or [] if isinstance(row, dict)]
    lines = ["#EXTM3U"]
    for row in sorted(rows, key=lambda item: int(item.get("position") or 0)):
        spotify = row.get("spotify") or {}
        path = files.get(str(spotify.get("spotify_id")))
        if path is None:
            continue
        seconds = round((spotify.get("duration_ms") or 0) / 1000) or -1
        artists = ", ".join(str(a) for a in spotify.get("artists") or [])
        lines += [f"#EXTINF:{seconds},{artists} - {spotify.get('title') or path.stem}", str(path.resolve())]
    out = playlist_dir / f"{playlist_dir.name}.m3u8"
    with out.open("w", encoding="utf-8", newline="\r\n") as handle:
        handle.write("\n".join(lines) + "\n")
    return out


def playlist_music_folder(runs_root: Path, playlist_id: str, playlists_root: Path) -> Path | None:
    """The music folder an earlier run of this playlist saved to, if it is in the current destination."""
    root = playlists_root.resolve()
    for run_dir in _runs_for_playlist(runs_root, playlist_id):
        report = read_json_dict(run_dir / "orchestration_report.json") or {}
        consolidation = read_json_dict(run_dir / "consolidation_report.json") or {}
        folder = report.get("final_output") or consolidation.get("playlist_directory")
        if not folder:
            name, _ = playlist_identity(run_dir, playlist_id)
            folder = root / safe_playlist_folder_name(name) if name and name != playlist_id else None
        if folder and Path(folder).is_dir() and Path(folder).resolve().is_relative_to(root):
            return Path(folder).resolve()
    return None


def saved_in_folder(folder: Path) -> set[str]:
    """Spotify Track IDs of the MP3s already in a playlist's music folder."""
    found: set[str] = set()
    if not folder.is_dir():
        return found
    for path in folder.glob("*.mp3"):
        try:
            found.add(embedded_spotify_id(path))
        except WorkflowError:
            continue
    return found


def reusable_audio_dirs(runs_root: Path, playlist_id: str, current: Path) -> list[Path]:
    """Finished MP3s of earlier runs of the playlist that were never saved: downloaded (soulseek/) or
    recorded (recordings/, and recorded_missing/ and recorded_failed/ of runs from before 2026-11-06)."""
    dirs: list[Path] = []
    for previous in _runs_for_playlist(runs_root, playlist_id):
        if previous.resolve() == current.resolve():
            continue
        for name in ("soulseek", RECORDINGS_DIR, *OLD_RECORDING_DIRS):
            folder = previous / name
            if folder.is_dir() and any(folder.rglob("*.mp3")):
                dirs.append(folder)
    return dirs


def _with_saved_before(run_dir: Path, summary: dict[str, Any], consolidation: dict[str, Any]) -> dict[str, Any]:
    """Add what went to the library when recording was postponed to this consolidation's report."""
    if not summary.get("saved_spotify_ids"):
        return consolidation
    earlier = read_json_dict(run_dir / "consolidation_report.json") or {}
    tracks = [row for row in earlier.get("tracks") or [] if isinstance(row, dict)] + consolidation["tracks"]
    return {
        **consolidation,
        "tracks": tracks,
        "audio_count": len(tracks),
        "moved": int(earlier.get("moved") or 0) + consolidation["moved"],
        "deduplicated": int(earlier.get("deduplicated") or 0) + consolidation["deduplicated"],
    }


def finish_partial(
    *,
    run_dir: Path,
    audio_dirs: tuple[Path, ...],
    playlists_root: Path,
    playlist_id: str,
    summary: dict[str, Any],
    summary_path: Path,
    reason: str,
) -> int:
    """Save the valid audio that exists even though the playlist is not complete.
    `audio_dirs`: Soulseek's folder first, then the recorded ones."""
    print(f"\nNo se consiguieron todos los temas: {reason}")
    soulseek_dir, *recorded_dirs = audio_dirs
    discard_unfinished_recordings(run_dir)
    playlist_name, expected_ids = playlist_identity(run_dir, playlist_id)
    if not any(d.is_dir() and any(d.rglob("*.mp3")) for d in audio_dirs):
        if summary.get("saved_spotify_ids"):  # lo bajado ya está en la carpeta (grabación pospuesta antes)
            summary.update({"status": "partial", "reason": reason})
            write_summary(summary_path, summary)
            print(f"Motivo   : {reason}")
            return EXIT_PARTIAL
        summary["status"] = "failed"
        summary["reason"] = reason
        write_summary(summary_path, summary)
        raise WorkflowError(reason)
    summary["status"] = "consolidating"
    summary["playlist_name"] = playlist_name
    write_summary(summary_path, summary)
    playlist_dir, moved_paths, consolidation = consolidate_audio(
        run_dir,
        soulseek_dir,
        None,
        playlists_root,
        playlist_name,
        expected_spotify_ids=None,
        extra_recorded_dirs=tuple(recorded_dirs),
    )
    consolidation = _with_saved_before(run_dir, summary, consolidation)
    # Lo que ya estaba en la carpeta (o se guardó al posponer la grabación) no falta.
    found = (
        {str(row["spotify_id"]) for row in consolidation["tracks"]}
        | set(summary.get("saved_spotify_ids") or [])
        | saved_in_folder(playlist_dir)
    )
    missing_ids = sorted((expected_ids or set()) - found)
    consolidation.update({"partial": True, "reason": reason, "missing_spotify_ids": missing_ids})
    write_summary(run_dir / "consolidation_report.json", consolidation)
    write_playlist_m3u8(playlist_dir, run_dir)
    summary.pop("soulseek_output", None)
    summary.pop("recorded_output", None)
    summary.pop("playlists_root", None)
    summary.update(
        {
            "status": "partial",
            "reason": reason,
            "final_output": str(playlist_dir),
            "audio_files": consolidation["audio_count"],
            "missing_spotify_ids": missing_ids,
        }
    )
    # Ya guardado: si se corta al archivar los JSON, el run no se retoma a medio archivar.
    write_summary(summary_path, summary)
    summary["diagnostics"] = str(archive_run(run_dir, moved_paths))
    write_summary(summary_path, summary)
    total = len(expected_ids) if expected_ids else None
    print("\n=== FLUJO INCOMPLETO ===")
    print(f"Playlist : {playlist_dir}")
    print(f"Audios   : {consolidation['audio_count']}" + (f" de {total}" if total else ""))
    print(f"Faltan   : {len(missing_ids)} (Spotify IDs en {summary_path})")
    print(f"Motivo   : {reason}")
    return EXIT_PARTIAL


def finish_with_recording_pending(
    *,
    run_dir: Path,
    soulseek_dir: Path,
    playlists_root: Path,
    playlist_id: str,
    summary: dict[str, Any],
    summary_path: Path,
    pending: int,
    blocked: str | None,
) -> int:
    """The download finished but something is still to record: postponed (DEFER_SIGNAL) or stopped
    (`blocked`: Spotify, the player, the loopback). What Soulseek delivered and the finished sessions go
    to the library now; the run waits, resumable, for the rest ("Grabar ahora" / "Continuar" in the app,
    or the same command)."""
    playlist_name, _ = playlist_identity(run_dir, playlist_id)
    saved = set(summary.get("saved_spotify_ids") or [])
    summary.update({"soulseek_finished": True, "playlist_name": playlist_name})
    finished_sessions = tuple(d for d in session_dirs(run_dir) if session_finished(d))
    sources = [soulseek_dir, *finished_sessions]
    if any(d.is_dir() and any(d.rglob("*.mp3")) for d in sources):
        summary["status"] = "consolidating"
        write_summary(summary_path, summary)
        discard_unfinished_recordings(run_dir)
        playlist_dir, _, consolidation = consolidate_audio(
            run_dir, soulseek_dir, None, playlists_root, playlist_name, expected_spotify_ids=None,
            extra_recorded_dirs=finished_sessions,
        )
        consolidation = _with_saved_before(run_dir, summary, consolidation)
        write_summary(run_dir / "consolidation_report.json", consolidation)
        write_playlist_m3u8(playlist_dir, run_dir)
        saved |= {str(row["spotify_id"]) for row in consolidation["tracks"]}
        summary["final_output"] = str(playlist_dir)
    summary.update(
        {
            "status": "resumable" if blocked else "recording_deferred",
            "saved_spotify_ids": sorted(saved),
            "pending_recordings": pending,
        }
    )
    if blocked:
        summary["reason"] = blocked
    write_summary(summary_path, summary)
    print("\n=== GRABACIÓN PENDIENTE ===" if blocked is None else "\n=== GRABACIÓN DETENIDA ===")
    if summary.get("final_output"):
        print(f"Playlist : {summary['final_output']} ({len(saved)} temas ya guardados)")
    if blocked:
        message = (
            f"La grabación se detuvo ({blocked}). Quedan {pending} tema(s) por grabar de Spotify; lo bajado "
            "de Soulseek ya está en la carpeta. Pulsa «Continuar» (o relanza el mismo comando) para grabar lo "
            "pendiente."
        )
    else:
        message = (
            f"Grabación pospuesta: {pending} tema(s) por grabar de Spotify. Lo bajado de Soulseek ya está en "
            "la carpeta; cuando quieras grabar, vuelve a lanzar la playlist («Grabar ahora» en la app)."
        )
    print(f"PAUSADO: {message}", file=sys.stderr)
    return EXIT_RESUMABLE


def run_workflow(args: argparse.Namespace) -> int:
    playlist_id = playlist_id_from_url(args.playlist_url)
    playlist_url = canonical_playlist_url(args.playlist_url)
    validate_environment()

    sockseek = find_required_program("sockseek")
    ffmpeg = find_required_program("ffmpeg")
    env = child_environment(sockseek, ffmpeg)

    runs_root = args.output_root.expanduser().resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    playlists_root = args.library_root.expanduser().resolve()
    resumed = None if args.fresh else resume_point(runs_root, playlist_id)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = resumed if resumed is not None else runs_root / f"{timestamp}_{playlist_id}"
    soulseek_dir = run_dir / "soulseek"
    recordings_dir = run_dir / RECORDINGS_DIR
    queue_path = run_dir / QUEUE_NAME
    availability_json = run_dir / "soulseek_availability.json"
    soulseek_report = run_dir / "spotify_reconcile_report.json"
    summary_path = run_dir / "orchestration_report.json"
    previous_report = (read_json_dict(summary_path) or {}) if resumed is not None else {}
    if resumed is not None:
        for signal in ("recorder_ready.json", "player_ready.json", "spotify_playback_status.json"):
            (run_dir / signal).unlink(missing_ok=True)
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    defer_file = run_dir / DEFER_SIGNAL
    if args.defer_recording:
        defer_file.write_text("1\n", encoding="utf-8")
    else:
        defer_file.unlink(missing_ok=True)

    queue_before = read_queue(queue_path)
    download_finished = resumed is not None and queue_before.closed

    print("\n=== FLUJO AUTOMÁTICO SPOTIFY / SOULSEEK ===")
    print(f"Playlist : {playlist_url}")
    print(f"Run      : {run_dir}")
    if resumed is not None:
        waiting = pending_recordings(run_dir, queue_before)
        if download_finished:
            print(f"Resume   : la descarga de Soulseek ya había terminado; {waiting} tema(s) por grabar")
        else:
            print(f"Resume   : sigue la búsqueda donde quedó; {waiting} tema(s) ya en la cola para grabar")
    if args.defer_recording:
        print("Grabación: pospuesta (se baja lo de Soulseek y se guarda en la carpeta)")

    summary: dict[str, Any] = {
        "layout": LAYOUT,
        "source_playlist": playlist_url,
        "playlist_id": playlist_id,
        "run_dir": str(run_dir),
        "recording_queue": str(queue_path),
        "soulseek_output": str(soulseek_dir),
        "recorded_output": str(recordings_dir),
        "playlists_root": str(playlists_root),
        "status": "running",
        "resumed": resumed is not None,
        "recording_output": getattr(args, "recording_output", None) or None,
    }
    # Lo que ya se guardó en la carpeta (al posponer o detenerse la grabación) sigue contando.
    summary.update({
        key: previous_report[key]
        for key in ("saved_spotify_ids", "final_output", "playlist_name")
        if key in previous_report
    })
    write_summary(summary_path, summary)

    downloader: subprocess.Popen[Any] | None = None
    sessions = SessionRecorder(run_dir=run_dir, env=env, args=args, summary=summary, summary_path=summary_path,
                               defer_file=defer_file)

    def partial(reason: str) -> int:
        return finish_partial(
            run_dir=run_dir,
            audio_dirs=(soulseek_dir, recordings_dir),
            playlists_root=playlists_root,
            playlist_id=playlist_id,
            summary=summary,
            summary_path=summary_path,
            reason=reason,
        )

    try:
        if not download_finished:
            downloader_cmd = [
                sys.executable,
                "-u",
                str(SCRIPT_DIR / "spotify_soulseek_download.py"),
                playlist_url,
                "--output", str(soulseek_dir),
                "--recording-queue", str(queue_path),
                "--availability-report", str(availability_json),
                "--report", str(soulseek_report),
                "--parallel", str(args.parallel),
            ]
            # Lo ya bajado o grabado en descargas anteriores de esta playlist (cortadas antes de guardar) se
            # trae sin buscarlo.
            for folder in reusable_audio_dirs(runs_root, playlist_id, run_dir):
                downloader_cmd += ["--reuse-dir", str(folder)]
            # Lo que ya está en la carpeta de la música no se busca ni se baja, y cuenta como guardado.
            music_folder = playlist_music_folder(runs_root, playlist_id, playlists_root)
            if music_folder is not None:
                downloader_cmd += ["--have-dir", str(music_folder)]
                summary["saved_spotify_ids"] = sorted(
                    set(summary.get("saved_spotify_ids") or []) | saved_in_folder(music_folder)
                )
                write_summary(summary_path, summary)
            print("\n[1/3] Buscando en Soulseek tema a tema (lo que está se baja enseguida)...")
            downloader = subprocess.Popen(downloader_cmd, env=env)
            print("[2/3] Lo que no se consiga se graba de Spotify mientras sigue la búsqueda.", flush=True)

        blocked: str | None = None
        idle_announced = False
        output_retry_at = 0.0  # con la salida ocupada: cuándo se vuelve a probar (time.monotonic)
        output_problem: str | None = None

        def output_busy(reason: str) -> None:
            nonlocal output_retry_at, output_problem
            output_retry_at = time.monotonic() + DEVICE_RETRY_S
            if reason != output_problem:
                print(f"\n[grabación] {reason}\nLo de Soulseek se sigue bajando; vuelvo a probar la salida cada "
                      f"{DEVICE_RETRY_S:.0f} s y grabo apenas quede libre.", file=sys.stderr, flush=True)
            output_problem = reason
            summary["recording_waiting"] = reason
            write_summary(summary_path, summary)

        recording_output = getattr(args, "recording_output", None) or None
        if not defer_file.exists():
            ok, detail = check_recording_device(env, recording_output)
            if ok:
                print(f"[grabación] Se graba lo que suena por «{detail}».", flush=True)
            else:
                output_busy(detail)
        while True:
            # Primero si el descargador terminó y después la cola: así no se pierde el último tema que agregó.
            downloader_done = downloader is None or downloader.poll() is not None
            queue = read_queue(queue_path)
            pending, unassigned = recording_plan(run_dir, queue)
            can_record = blocked is None and not defer_file.exists() and time.monotonic() >= output_retry_at
            if can_record and (pending or unassigned):
                idle_announced = False
                ok, detail = check_recording_device(env, recording_output)  # antes de la playlist y de Spotify
                if not ok:
                    output_busy(detail)
                    if downloader_done:  # ya no hay nada que esperar: queda pendiente, con el motivo
                        break
                    continue
                if output_problem is not None:
                    print(f"[grabación] La salida «{detail}» quedó libre: sigo grabando.", flush=True)
                    output_problem = None
                    summary.pop("recording_waiting", None)
                    write_summary(summary_path, summary)
                session_dir = pending[0] if pending else create_session(run_dir, unassigned[:SESSION_SIZE])
                relocate_checkpoint(session_dir)
                if not session_checkpoint_usable(session_dir):
                    print(f"La tanda {session_dir.name} no se puede retomar (faltan sus MP3): lo pendiente "
                          "vuelve a la cola.", flush=True)
                    abandon_session(session_dir)
                    continue
                try:
                    playlist_name, _ = playlist_identity(run_dir, playlist_id)
                    url = ensure_session_playlist(session_dir, playlist_name)
                    outcome = sessions.record(session_dir, url)
                except OutputBusy as exc:
                    sessions.pause()
                    output_busy(str(exc))
                    if downloader is None or downloader.poll() is not None:  # la descarga ya terminó
                        break
                    continue
                except RecordingUnavailable as exc:
                    blocked = str(exc)
                    sessions.pause()
                    summary["recording_blocked"] = blocked
                    write_summary(summary_path, summary)
                    print(f"\nLa grabación se detuvo: {blocked}", file=sys.stderr, flush=True)
                    if not downloader_done:
                        print("La descarga de Soulseek sigue; lo que falte grabar queda para «Continuar».",
                              flush=True)
                    continue
                if outcome == "deferred":
                    sessions.pause()
                elif outcome == "completed":
                    # Grabada: Spotify (la playlist suena en repetición) se pausa ya, y después se borra la
                    # playlist temporal, que no hace falta más.
                    sessions.pause()
                    forget_session_playlist(session_dir)
                continue
            # Nada para grabar ahora: Spotify no sigue sonando por su cuenta.
            sessions.pause()
            if downloader_done:
                break
            if can_record and not idle_announced and not (pending or unassigned):
                print("[grabación] Cola vacía: espero el próximo tema que haya que grabar.", flush=True)
                idle_announced = True
            time.sleep(IDLE_POLL_SECONDS)

        downloader_code = downloader.returncode if downloader is not None else None
        if downloader_code is not None:
            summary["soulseek_returncode"] = int(downloader_code)
        queue = read_queue(queue_path)
        summary["recording_count"] = len(queue.tracks)
        write_summary(summary_path, summary)
        if not queue.closed:
            return partial(
                f"La descarga de Soulseek se cortó antes de terminar (código {downloader_code})."
            )
        if (queue.end or {}).get("status") != END_FINISHED:
            error = (queue.end or {}).get("error") or "error desconocido"
            return partial(
                f"La búsqueda en Soulseek dejó de funcionar ({error}): se cortó antes de terminar la playlist. "
                "Revisa que la cuenta de Soulseek no esté abierta en otra PC y vuelve a intentar: lo ya "
                "guardado no se busca de nuevo."
            )
        waiting = pending_recordings(run_dir, queue)
        if waiting:
            return finish_with_recording_pending(
                run_dir=run_dir,
                soulseek_dir=soulseek_dir,
                playlists_root=playlists_root,
                playlist_id=playlist_id,
                summary=summary,
                summary_path=summary_path,
                pending=waiting,
                blocked=None if defer_file.exists() else (blocked or output_problem),
            )

        for session_dir in session_dirs(run_dir):  # las que no se pudieron borrar al terminar su tanda
            if session_finished(session_dir):
                forget_session_playlist(session_dir)
        print("\n[3/3] Guardando en tu carpeta...")
        playlist_name, expected_ids = playlist_identity(run_dir, playlist_id)
        summary["status"] = "consolidating"
        summary["playlist_name"] = playlist_name
        summary.pop("recording_blocked", None)
        summary.pop("recording_waiting", None)
        summary.pop("reason", None)
        summary.pop("pending_recordings", None)
        write_summary(summary_path, summary)
        discard_unfinished_recordings(run_dir)
        target_folder = playlists_root / safe_playlist_folder_name(playlist_name)
        # Ya en la carpeta: al posponer la grabación, en una descarga anterior o en un guardado que se cortó.
        saved_ids = set(summary.get("saved_spotify_ids") or []) | saved_in_folder(target_folder)
        try:
            playlist_dir, moved_paths, consolidation = consolidate_audio(
                run_dir,
                soulseek_dir,
                None,
                playlists_root,
                playlist_name,
                expected_spotify_ids=expected_ids - saved_ids if expected_ids is not None else None,
                extra_recorded_dirs=(recordings_dir,),
            )
        except IncompleteAudio as exc:
            return partial(str(exc))
        consolidation = _with_saved_before(run_dir, summary, consolidation)
        consolidation_path = run_dir / "consolidation_report.json"
        write_summary(consolidation_path, consolidation)
        write_playlist_m3u8(playlist_dir, run_dir)
        complete_metadata([Path(str(row["destination"])) for row in consolidation["tracks"]
                           if row.get("destination") and Path(str(row["destination"])).is_file()], env)

        summary.pop("soulseek_output", None)
        summary.pop("recorded_output", None)
        summary.pop("playlists_root", None)
        summary["status"] = "completed"
        summary["final_output"] = str(playlist_dir)
        summary["audio_files"] = consolidation["audio_count"]
        summary["moved_files"] = consolidation["moved"]
        summary["deduplicated_files"] = consolidation["deduplicated"]
        # Ya guardado: si se corta al archivar los JSON, el run no se retoma a medio archivar.
        write_summary(summary_path, summary)
        diagnostics_dir = archive_run(run_dir, moved_paths)
        summary["diagnostics"] = str(diagnostics_dir)
        write_summary(summary_path, summary)
        print("\n=== FLUJO COMPLETADO ===")
        print(f"Playlist : {playlist_dir}")
        print(f"Audios   : {consolidation['audio_count']}")
        print(f"Debug    : {diagnostics_dir}")
        print(f"Reporte  : {summary_path}")
        return 0
    except KeyboardInterrupt:
        sessions.stop_processes()
        sessions.pause()
        summary.pop("recording_session", None)
        summary["status"] = "interrupted"
        write_summary(summary_path, summary)
        print("\nInterrumpido. Cerrando procesos hijos...", file=sys.stderr)
        return 130
    except WorkflowError:
        if summary.get("status") != "failed":
            summary["status"] = "failed"
            write_summary(summary_path, summary)
        raise
    finally:
        sessions.stop_processes()
        terminate_process(downloader)
        stop_leftover_daemon(run_dir)
        give_back_spotify_output(runs_root)


PIPELINE_MODULES = "import pandas, torch, transformers, mutagen"
TAG_DOWNLOADS = REPO_ROOT / "src" / "v4" / "pipeline" / "tag_downloads.py"


def pipeline_python(env: dict[str, str]) -> str | None:
    """Un Python con el pipeline (pandas, torch, transformers): TRAKTOR_PIPELINE_PYTHON del .env, el de la
    descarga, el del PATH o el .venv del repo. En el escritorio la app corre con un .venv mínimo y el
    pipeline está en el Python 3.11 de la Store."""
    candidates = [env.get("TRAKTOR_PIPELINE_PYTHON"), sys.executable, shutil.which("python"),
                  str(REPO_ROOT / ".venv" / "Scripts" / "python.exe"), str(REPO_ROOT / ".venv" / "bin" / "python")]
    for candidate in dict.fromkeys(c for c in candidates if c):
        try:
            if subprocess.run([candidate, "-c", PIPELINE_MODULES], env=env, capture_output=True,
                              timeout=180).returncode == 0:
                return candidate
        except (OSError, subprocess.TimeoutExpired):
            continue
    return None


def complete_metadata(files: list[Path], env: dict[str, str]) -> None:
    """Beatport (género, sello, artistas, fecha) y la etiqueta Vocal en los temas de la playlist
    (src/v4/pipeline/tag_downloads.py, las reglas de la biblioteca). Nunca corta el final de la descarga."""
    if not files:
        return
    print("\n[metadata] Completando la metadata: Beatport y voces...", flush=True)
    python = pipeline_python(env)
    if python is None:
        print("[metadata] Ningún Python de este ordenador tiene el pipeline (pandas, torch, transformers): la "
              "metadata queda como la de Spotify. Pon su ruta en TRAKTOR_PIPELINE_PYTHON del .env.", flush=True)
        return
    try:
        process = subprocess.Popen([python, "-u", str(TAG_DOWNLOADS), *map(str, files)], env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8", errors="replace")
        assert process.stdout is not None
        for line in process.stdout:
            print(line.rstrip(), flush=True)
        code = process.wait()
    except OSError as exc:
        print(f"[metadata] No se pudo completar la metadata ({exc}).", flush=True)
        return
    if code != 0:
        print(f"[metadata] Terminó con código {code}: lo que faltó se completa en la próxima descarga.", flush=True)


def give_back_spotify_output(runs_root: Path) -> None:
    """Spotify vuelve a la salida que tenía antes de grabar (spotify_output.py). Nunca corta el final."""
    try:
        message = spotify_output.restore(runs_root / spotify_output.STATE_NAME)
    except Exception as exc:  # noqa: BLE001
        message = f"No pude devolver Spotify a su salida de antes ({exc})."
    if message:
        print(f"[grabación] {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Busca y baja de Soulseek tema a tema y graba de Spotify lo que falte, mientras sigue la búsqueda."
        )
    )
    parser.add_argument("playlist_url", help="URL de una playlist de Spotify")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("SPOTIFY_DOWNLOAD_ROOT", DEFAULT_RUNS_DIR)),
        help=(
            "Directorio de runs (default: download_JIJIJI/runs o "
            "SPOTIFY_DOWNLOAD_ROOT)"
        ),
    )
    parser.add_argument(
        "--library-root",
        type=Path,
        default=Path(
            os.environ.get("SPOTIFY_PLAYLISTS_ROOT", DEFAULT_PLAYLISTS_DIR)
        ),
        help=(
            "Destino consolidado de playlists (default: "
            "download_JIJIJI/Playlists_DOWNLOAD)"
        ),
    )
    parser.add_argument(
        "--recorder-timeout",
        type=float,
        default=30.0,
        help="Segundos máximos para abrir el dispositivo loopback y empezar a reproducir",
    )
    parser.add_argument(
        "--recording-output",
        default=os.environ.get("SPOTIFY_RECORDING_OUTPUT", ""),
        help=(
            "Salida de audio que se graba (su nombre o una parte, p. ej. \"Voicemeeter Input\"; default: "
            "SPOTIFY_RECORDING_OUTPUT o la salida predeterminada de Windows). Spotify tiene que sonar por ahí"
        ),
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=2,
        help="Temas que se buscan y bajan a la vez, con una sola sesión de Soulseek (default 2, hasta 4)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignorar una sesión incompleta anterior y crear un run nuevo",
    )
    parser.add_argument(
        "--defer-recording",
        action="store_true",
        help=(
            "Bajar de Soulseek y guardar eso en la carpeta, pero no grabar: la grabación queda "
            "pendiente hasta volver a lanzar la misma playlist sin esta opción"
        ),
    )
    return parser


def main() -> int:
    try:
        # Before the parser: .env may set SPOTIFY_DOWNLOAD_ROOT / SPOTIFY_PLAYLISTS_ROOT.
        load_environment()
        args = build_parser().parse_args()
        return run_workflow(args)
    except ResumeAvailable as exc:
        print(f"PAUSADO: {exc}", file=sys.stderr)
        return EXIT_RESUMABLE
    except WorkflowError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
