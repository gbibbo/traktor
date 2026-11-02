#!/usr/bin/env python3
"""
PURPOSE: Run the complete Spotify -> Soulseek + loopback fallback workflow.

CHANGELOG:
- 2026-11-02: Postponed recording (--defer-recording, or DEFER_SIGNAL in the run while it
              downloads): Soulseek's tracks go to the library, the run ends "recording_deferred"
              (exit 75), and running the playlist again records only what is pending and adds it.
- 2026-11-02: The preflight wait never shrinks when a read of its status file (now rewritten with
              the search progress) fails.
- 2026-10-30: Wait for the preflight as long as the downloader says it needs (long playlists hit
              Sockseek's search limiter); a preflight that failed for every track stops the run
              instead of sending the whole playlist to recording, and such runs are never resumed.
- 2026-10-30: Record from Spotify the tracks Soulseek could not deliver (soulseek_failed.json,
              recorded_failed/), resumable on their own; a downloader exit code 3 no longer stalls the
              recording; a run that still misses tracks saves what it has ("partial", exit code 3).
- 2026-10-30: Load .env before parsing so it can set the runs and playlists roots.
- 2026-10-05: Revalidate final MP3s and existing duplicates before consolidation.
- 2026-09-29: Record missing tracks sequentially in Spotify batches of up to 10.
- 2026-09-27: Consolidate successful runs into one named playlist directory.
- 2026-09-27: Initial orchestration with manifest and recorder readiness handshakes.
- 2026-09-27: Resume incomplete recording sessions at the first pending track.

Usage:
    python spotify_soulseek_orchestrator.py "https://open.spotify.com/playlist/..."

After one-time environment setup, the Spotify playlist URL is the only required
argument. The script starts the Soulseek preflight/download process, waits for
its final missing-track JSON, records the generated Spotify fallback playlist,
releases the paused Soulseek downloads only after playback starts, and waits for
both branches to finish.
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
from typing import Any, Callable


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_RUNS_DIR = SCRIPT_DIR / "runs"
DEFAULT_PLAYLISTS_DIR = SCRIPT_DIR / "Playlists_DOWNLOAD"
MANIFEST_POLL_SECONDS = 0.25
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"}
FAILED_JSON_NAME = "soulseek_failed.json"       # temas que Soulseek no pudo dar: se graban de Spotify
RECORDED_FAILED_DIR = "recorded_failed"
# El descargador termina con 0 (todo bajado) o 3 (faltan temas, listados en FAILED_JSON_NAME).
DOWNLOADER_FINISHED_CODES = {0, 3}
EXIT_PARTIAL = 3
EXIT_RESUMABLE = 75
# Mientras exista en el run, no se empieza a grabar: se baja lo de Soulseek, va a la carpeta y la
# grabación queda pendiente («Grabar ahora» la retoma). La app lo crea o lo borra durante la descarga.
DEFER_SIGNAL = "defer_recording.signal"


class WorkflowError(RuntimeError):
    """A user-facing orchestration failure."""


class ResumeAvailable(WorkflowError):
    """Playback stopped, but completed per-track checkpoints remain reusable."""


class IncompleteAudio(WorkflowError):
    """The final audio set does not cover the whole playlist."""


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


def final_missing_manifest(path: Path) -> dict[str, Any] | None:
    """Return the manifest only after playlist creation has been attempted."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None
    missing_count = payload.get("missing_count")
    tracks = payload.get("tracks")
    if not isinstance(missing_count, int) or not isinstance(tracks, list):
        return None
    if missing_count != len(tracks):
        return None
    if missing_count == 0:
        return payload

    generated_many = payload.get("generated_missing_playlists")
    if isinstance(generated_many, list) and generated_many:
        if all(
            isinstance(item, dict) and (item.get("url") or item.get("error"))
            for item in generated_many
        ):
            return payload

    generated = payload.get("generated_missing_playlist")
    if isinstance(generated, dict) and (generated.get("url") or generated.get("error")):
        return payload
    return None


def missing_playlist_batches(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Return validated, contiguous playlist batches with global index ranges."""
    missing_count = manifest.get("missing_count")
    if not isinstance(missing_count, int) or missing_count < 0:
        return []
    if missing_count == 0:
        return []

    raw_batches = manifest.get("generated_missing_playlists")
    if not isinstance(raw_batches, list) or not raw_batches:
        legacy = manifest.get("generated_missing_playlist")
        if not isinstance(legacy, dict) or not legacy.get("url"):
            return []
        raw_batches = [
            {
                **legacy,
                "batch_index": 0,
                "batch_number": 1,
                "batch_count": 1,
                "start_index": 0,
                "end_index": missing_count,
                "track_count": missing_count,
            }
        ]

    batches: list[dict[str, Any]] = []
    expected_start = 0
    total_batches = len(raw_batches)
    for index, raw in enumerate(raw_batches):
        if not isinstance(raw, dict) or not raw.get("url"):
            return []
        start = raw.get("start_index")
        end = raw.get("end_index")
        count = raw.get("track_count")
        if not all(isinstance(value, int) for value in (start, end, count)):
            return []
        if start != expected_start or end <= start or count != end - start:
            return []
        if end > missing_count:
            return []
        batches.append(
            {
                **raw,
                "batch_index": index,
                "batch_number": index + 1,
                "batch_count": total_batches,
                "start_index": start,
                "end_index": end,
                "track_count": count,
            }
        )
        expected_start = end
    return batches if expected_start == missing_count else []


def batch_for_completed(
    batches: list[dict[str, Any]], completed_count: int
) -> dict[str, Any] | None:
    """Find the batch containing the next globally pending track."""
    for batch in batches:
        if int(batch["start_index"]) <= completed_count < int(batch["end_index"]):
            return batch
    return None


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


def _pending_recording(
    manifest: dict[str, Any] | None,
    progress: dict[str, Any] | None,
    *,
    require_progress: bool = True,
) -> bool:
    """Recording list with tracks still to record (and, if it started, a coherent checkpoint)."""
    if manifest is None:
        return False
    missing_count = manifest.get("missing_count")
    if not isinstance(missing_count, int) or missing_count <= 0:
        return False
    if not missing_playlist_batches(manifest):
        return False
    if progress is None:
        return not require_progress
    completed_count = progress.get("completed_count")
    if not isinstance(completed_count, int):
        return False
    if progress.get("total_tracks") != missing_count:
        return False
    if progress.get("status") == "completed":
        return False
    return 0 <= completed_count < missing_count


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


def _recording_complete(manifest: dict[str, Any], progress: dict[str, Any] | None) -> bool:
    missing_count = manifest.get("missing_count")
    if missing_count == 0:
        return True
    return bool(progress) and progress.get("completed_count") == missing_count


def resumable_run(
    runs_root: Path,
    playlist_id: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]] | None:
    """Find the newest compatible run with at least one pending recording of missing tracks."""
    for run_dir in _runs_for_playlist(runs_root, playlist_id):
        manifest = final_missing_manifest(run_dir / "soulseek_missing.json")
        progress = read_json_dict(
            run_dir / "recorded_missing" / "recording_progress.json"
        )
        if _pending_recording(manifest, progress):
            assert manifest is not None and progress is not None
            return run_dir, manifest, progress
    return None


def resume_point(
    runs_root: Path,
    playlist_id: str,
) -> tuple[str, Path, dict[str, Any], dict[str, Any] | None] | None:
    """Where to continue the newest unfinished run of this playlist.

    ("missing", run, manifest, progress): the recording of tracks missing from Soulseek stopped.
    ("failed", run, manifest, progress|None): the download finished and the recording of the
    tracks it could not deliver (FAILED_JSON_NAME) is pending.
    ("deferred", run, manifest, progress|None): Soulseek finished and its tracks are already in the
    library; what is left is recording (both lists, whatever is pending) and the final save.
    """
    for run_dir in _runs_for_playlist(runs_root, playlist_id):
        report = read_json_dict(run_dir / "orchestration_report.json") or {}
        if report.get("status") in {"completed", "partial"}:
            continue
        manifest = final_missing_manifest(run_dir / "soulseek_missing.json")
        progress = read_json_dict(run_dir / "recorded_missing" / "recording_progress.json")
        if report.get("soulseek_finished") and manifest is not None:
            return "deferred", run_dir, manifest, progress
        if _pending_recording(manifest, progress):
            assert manifest is not None
            return "missing", run_dir, manifest, progress
        failed = final_missing_manifest(run_dir / FAILED_JSON_NAME)
        failed_progress = read_json_dict(run_dir / RECORDED_FAILED_DIR / "recording_progress.json")
        if (
            manifest is not None
            and _recording_complete(manifest, progress)
            and _pending_recording(failed, failed_progress, require_progress=False)
            and preflight_failed_entirely(run_dir) is None  # runs del 2026-10-30 con el preflight caído
        ):
            assert failed is not None
            return "failed", run_dir, failed, failed_progress
    return None


def manifest_deadline(base_deadline: float, status_path: Path | None) -> float:
    """El preflight avisa cuántos temas busca y su tope: hay que esperarlo (más un margen para
    crear las playlists de faltantes), aunque supere el tope fijo del orquestador."""
    status = read_json_dict(status_path) if status_path is not None else None
    if not status:
        return base_deadline
    try:
        started = float(status["started_at"])
        preflight_timeout = float(status["timeout_s"])
    except (KeyError, TypeError, ValueError):
        return base_deadline
    elapsed = max(0.0, time.time() - started)
    return max(base_deadline, time.monotonic() - elapsed + preflight_timeout + 300.0)


def wait_for_missing_manifest(
    process: subprocess.Popen[Any],
    path: Path,
    timeout: float,
    status_path: Path | None = None,
) -> dict[str, Any]:
    base_deadline = deadline = time.monotonic() + timeout
    while True:
        # El estado se reescribe con el avance de la búsqueda: una lectura fallida no acorta el plazo.
        deadline = max(deadline, manifest_deadline(base_deadline, status_path))
        if time.monotonic() >= deadline:
            break
        payload = final_missing_manifest(path)
        if payload is not None:
            return payload
        return_code = process.poll()
        if return_code is not None:
            payload = final_missing_manifest(path)
            if payload is not None:
                return payload
            raise WorkflowError(
                "El descargador terminó antes de generar el JSON definitivo "
                f"(código {return_code})."
            )
        time.sleep(MANIFEST_POLL_SECONDS)
    raise WorkflowError(
        f"Tiempo agotado esperando el JSON definitivo de faltantes: {path}"
    )


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
        time.sleep(MANIFEST_POLL_SECONDS)
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
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def release_downloads(path: Path) -> None:
    path.write_text("start\n", encoding="utf-8")
    print("Descargas de Soulseek liberadas.")


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

    With `expected_spotify_ids`, a different final set raises IncompleteAudio before moving
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
    if expected_spotify_ids is not None and found_ids != expected_spotify_ids:
        missing = sorted(expected_spotify_ids - found_ids)
        unexpected = sorted(found_ids - expected_spotify_ids)
        raise IncompleteAudio(
            "El conjunto de audio final no coincide con la playlist: "
            f"faltan={missing}, inesperados={unexpected}"
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


def simplify_completed_run(run_dir: Path, moved_paths: dict[str, str]) -> Path:
    """Keep a small readable root plus detailed JSON under diagnostics/."""
    diagnostics = run_dir / "diagnostics"
    diagnostics.mkdir(exist_ok=True)

    for json_path in list(run_dir.rglob("*.json")):
        if diagnostics in json_path.parents:
            continue
        if json_path.name in {
            "orchestration_report.json",
            "soulseek_missing.json",
            FAILED_JSON_NAME,
            "consolidation_report.json",
        }:
            continue
        if json_path.name in {"player_ready.json", "recorder_ready.json"}:
            json_path.unlink(missing_ok=True)
            continue
        _rewrite_moved_outputs(json_path, moved_paths)
        destination = diagnostics / json_path.name
        if destination.exists():
            destination = diagnostics / (
                f"{json_path.parent.name}_{json_path.name}"
            )
        shutil.move(str(json_path), str(destination))

    for signal in run_dir.glob("*.signal"):
        signal.unlink(missing_ok=True)
    for text_file in run_dir.rglob("*.txt"):
        text_file.unlink(missing_ok=True)

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
        if directory != diagnostics:
            try:
                directory.rmdir()
            except OSError:
                pass
    return diagnostics


def record_batches(
    *,
    label: str,
    manifest_path: Path,
    batches: list[dict[str, Any]],
    total: int,
    completed_count: int,
    recorded_dir: Path,
    run_dir: Path,
    env: dict[str, str],
    args: argparse.Namespace,
    summary: dict[str, Any],
    summary_path: Path,
    keys: tuple[str, str, str],
    downloader: subprocess.Popen[Any] | None = None,
    on_first_playback: Callable[[], None] | None = None,
) -> int:
    """Record the tracks of a recording list in its Spotify batches; return the completed count.

    `keys` are the summary fields (completed tracks, completed batches, active batch).
    """
    recorder_ready = run_dir / "recorder_ready.json"
    player_ready = run_dir / "player_ready.json"
    playback_status = run_dir / "spotify_playback_status.json"
    done_key, batches_key, active_key = keys
    completed_batches: list[dict[str, Any]] = list(summary.get(batches_key) or [])
    recorder: subprocess.Popen[Any] | None = None
    player: subprocess.Popen[Any] | None = None
    first_playback = True
    try:
        while completed_count < total:
            batch = batch_for_completed(batches, completed_count)
            if batch is None:
                raise WorkflowError(
                    f"No existe un lote para la pista pendiente {completed_count + 1}."
                )
            batch_number = int(batch["batch_number"])
            batch_count = int(batch["batch_count"])
            batch_start = int(batch["start_index"])
            batch_end = int(batch["end_index"])
            local_offset = completed_count - batch_start
            tracks_this_process = batch_end - completed_count

            recorder_ready.unlink(missing_ok=True)
            player_ready.unlink(missing_ok=True)
            playback_status.unlink(missing_ok=True)

            recorder_cmd = [
                sys.executable,
                "-u",
                str(SCRIPT_DIR / "playlist_loopback_recorder.py"),
                str(manifest_path),
                "--output-dir",
                str(recorded_dir),
                "--ready-file",
                str(recorder_ready),
                "--resume",
                "--max-tracks",
                str(tracks_this_process),
                "--watchdog-file",
                str(playback_status),
            ]
            print(
                f"\n[grabación] Abriendo grabador ({label}) para lote {batch_number}/{batch_count} "
                f"({tracks_this_process} tema(s))..."
            )
            recorder = subprocess.Popen(recorder_cmd, env=env)
            wait_for_ready_file(
                recorder,
                recorder_ready,
                args.recorder_timeout,
                label="grabador",
            )

            print(
                f"\n[grabación] Reproduciendo lote Spotify ({label}) {batch_number}/{batch_count} "
                f"desde su posición {local_offset + 1}..."
            )
            player_cmd = [
                sys.executable,
                "-u",
                str(SCRIPT_DIR / "spotify_playlist.py"),
                str(batch["url"]),
                "--offset-position",
                str(local_offset),
                "--ready-file",
                str(player_ready),
                "--watchdog-file",
                str(playback_status),
            ]
            player = subprocess.Popen(player_cmd, env=env)
            wait_for_ready_file(
                player,
                player_ready,
                args.recorder_timeout,
                label="reproductor de Spotify",
            )
            if first_playback:
                first_playback = False
                if on_first_playback is not None:
                    on_first_playback()

            summary[active_key] = batch_number
            summary[done_key] = completed_count
            write_summary(summary_path, summary)

            while recorder.poll() is None:
                downloader_code_now = downloader.poll() if downloader is not None else None
                if downloader_code_now is not None and downloader_code_now not in DOWNLOADER_FINISHED_CODES:
                    write_summary(
                        playback_status,
                        {
                            "status": "stalled",
                            "error": (
                                "Soulseek downloader exited with code "
                                f"{downloader_code_now}"
                            ),
                        },
                    )
                player_code_now = player.poll()
                if player_code_now is not None and player_code_now not in {0, 75}:
                    write_summary(
                        playback_status,
                        {
                            "status": "stalled",
                            "error": f"Spotify player exited with code {player_code_now}",
                        },
                    )
                time.sleep(0.5)

            recorder_code = int(recorder.returncode or 0)
            terminate_process(player)
            player = None
            recorder = None
            summary["recorder_returncode"] = recorder_code

            progress = read_json_dict(recorded_dir / "recording_progress.json") or {}
            completed_now = progress.get("completed_count", completed_count)
            if recorder_code in {75, 130}:
                summary["status"] = "resumable"
                summary[done_key] = completed_now
                write_summary(summary_path, summary)
                terminate_process(downloader)
                raise ResumeAvailable(
                    "Spotify se detuvo. Se conservaron "
                    f"{completed_now}/{total} pistas ({label}); relanza el mismo comando "
                    "para continuar desde la primera pendiente."
                )
            if recorder_code != 0:
                raise WorkflowError(
                    f"El grabador del lote {batch_number} ({label}) terminó con código "
                    f"{recorder_code}."
                )
            if not isinstance(completed_now, int) or completed_now != batch_end:
                raise WorkflowError(
                    f"El lote {batch_number} ({label}) terminó en {completed_now}; "
                    f"se esperaba {batch_end}."
                )

            completed_batches.append(
                {
                    "batch_number": batch_number,
                    "start_index": completed_count,
                    "end_index": completed_now,
                    "status": "completed",
                }
            )
            completed_count = completed_now
            summary[done_key] = completed_count
            summary[batches_key] = completed_batches
            summary.pop(active_key, None)
            write_summary(summary_path, summary)
            print(
                f"Lote {batch_number}/{batch_count} ({label}) completo: "
                f"{completed_count}/{total} pistas grabadas."
            )
        return completed_count
    finally:
        terminate_process(player)
        terminate_process(recorder)


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


def finish_partial(
    *,
    run_dir: Path,
    audio_dirs: tuple[Path, Path, Path],
    playlists_root: Path,
    playlist_id: str,
    summary: dict[str, Any],
    summary_path: Path,
    reason: str,
) -> int:
    """Save the valid audio that exists even though the playlist is not complete."""
    print(f"\nNo se consiguieron todos los temas: {reason}")
    soulseek_dir, recorded_dir, failed_dir = audio_dirs
    playlist_name, expected_ids = playlist_identity(run_dir, playlist_id)
    if not any(d.is_dir() and any(d.rglob("*.mp3")) for d in audio_dirs):
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
        recorded_dir,
        playlists_root,
        playlist_name,
        expected_spotify_ids=None,
        extra_recorded_dirs=(failed_dir,),
    )
    consolidation = _with_saved_before(run_dir, summary, consolidation)
    found = {str(row["spotify_id"]) for row in consolidation["tracks"]}
    missing_ids = sorted((expected_ids or set()) - found)
    consolidation.update({"partial": True, "reason": reason, "missing_spotify_ids": missing_ids})
    write_summary(run_dir / "consolidation_report.json", consolidation)
    diagnostics_dir = simplify_completed_run(run_dir, moved_paths)
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
            "diagnostics": str(diagnostics_dir),
        }
    )
    write_summary(summary_path, summary)
    total = len(expected_ids) if expected_ids else None
    print("\n=== FLUJO INCOMPLETO ===")
    print(f"Playlist : {playlist_dir}")
    print(f"Audios   : {consolidation['audio_count']}" + (f" de {total}" if total else ""))
    print(f"Faltan   : {len(missing_ids)} (Spotify IDs en {summary_path})")
    print(f"Motivo   : {reason}")
    return EXIT_PARTIAL


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


def finish_deferred(
    *,
    run_dir: Path,
    soulseek_dir: Path,
    playlists_root: Path,
    playlist_id: str,
    summary: dict[str, Any],
    summary_path: Path,
    pending: int,
) -> int:
    """Recording postponed: what Soulseek delivered goes to the library now and the run waits,
    resumable, for the recording ("Grabar ahora" in the app, or the same command without
    --defer-recording)."""
    playlist_name, _ = playlist_identity(run_dir, playlist_id)
    saved = set(summary.get("saved_spotify_ids") or [])
    summary.update({"soulseek_finished": True, "playlist_name": playlist_name})
    if soulseek_dir.is_dir() and any(soulseek_dir.rglob("*.mp3")):
        summary["status"] = "consolidating"
        write_summary(summary_path, summary)
        playlist_dir, _, consolidation = consolidate_audio(
            run_dir, soulseek_dir, None, playlists_root, playlist_name, expected_spotify_ids=None
        )
        consolidation = _with_saved_before(run_dir, summary, consolidation)
        write_summary(run_dir / "consolidation_report.json", consolidation)
        saved |= {str(row["spotify_id"]) for row in consolidation["tracks"]}
        summary["final_output"] = str(playlist_dir)
    summary.update(
        {
            "status": "recording_deferred",
            "saved_spotify_ids": sorted(saved),
            "pending_recordings": pending,
        }
    )
    write_summary(summary_path, summary)
    print("\n=== GRABACIÓN POSPUESTA ===")
    if summary.get("final_output"):
        print(f"Playlist : {summary['final_output']} ({len(saved)} temas de Soulseek)")
    print(
        f"PAUSADO: Grabación pospuesta: {pending} tema(s) por grabar de Spotify. Lo bajado de "
        "Soulseek ya está en la carpeta; cuando quieras grabar, vuelve a lanzar la playlist "
        "(«Grabar ahora» en la app).",
        file=sys.stderr,
    )
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
    resume = None if args.fresh else resume_point(runs_root, playlist_id)
    mode = resume[0] if resume is not None else "fresh"
    is_resume = resume is not None
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = resume[1] if resume is not None else runs_root / f"{timestamp}_{playlist_id}"
    soulseek_dir = run_dir / "soulseek"
    recorded_dir = run_dir / "recorded_missing"
    failed_dir = run_dir / RECORDED_FAILED_DIR
    missing_json = run_dir / "soulseek_missing.json"
    failed_json = run_dir / FAILED_JSON_NAME
    preflight_status = run_dir / "preflight_status.json"
    if is_resume:
        availability_json = run_dir / f"soulseek_availability_resume_{timestamp}.json"
        downloader_missing_json = run_dir / f"soulseek_missing_resume_{timestamp}.json"
        soulseek_report = run_dir / f"spotify_reconcile_report_resume_{timestamp}.json"
    else:
        availability_json = run_dir / "soulseek_availability.json"
        downloader_missing_json = missing_json
        soulseek_report = run_dir / "spotify_reconcile_report.json"
    download_gate = run_dir / "start_soulseek_downloads.signal"
    summary_path = run_dir / "orchestration_report.json"
    previous_report = (read_json_dict(summary_path) or {}) if is_resume else {}
    if is_resume:
        for signal in ("recorder_ready.json", "player_ready.json", "spotify_playback_status.json"):
            (run_dir / signal).unlink(missing_ok=True)
        download_gate.unlink(missing_ok=True)
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    defer_file = run_dir / DEFER_SIGNAL
    if args.defer_recording:
        defer_file.write_text("1\n", encoding="utf-8")
    else:
        defer_file.unlink(missing_ok=True)

    print("\n=== FLUJO AUTOMÁTICO SPOTIFY / SOULSEEK ===")
    print(f"Playlist : {playlist_url}")
    print(f"Run      : {run_dir}")
    if mode == "deferred":
        print("Resume   : grabación pospuesta (lo de Soulseek ya está en la carpeta)")
    elif resume is not None:
        progress = resume[3] or {}
        completed = int(progress.get("completed_count") or 0)
        total = int(resume[2]["missing_count"])
        what = "pistas ya conservadas" if mode == "missing" else "temas no bajados ya grabados"
        print(f"Resume   : {completed}/{total} {what}")
    if args.defer_recording:
        print("Grabación: pospuesta (se baja lo de Soulseek y se guarda en la carpeta)")

    downloader: subprocess.Popen[Any] | None = None
    summary: dict[str, Any] = {
        "source_playlist": playlist_url,
        "playlist_id": playlist_id,
        "run_dir": str(run_dir),
        "missing_json": str(missing_json),
        "failed_json": str(failed_json),
        "soulseek_output": str(soulseek_dir),
        "recorded_output": str(recorded_dir),
        "recorded_failed_output": str(failed_dir),
        "playlists_root": str(playlists_root),
        "status": "running",
        "resumed": is_resume,
        "resume_phase": mode if is_resume else None,
    }
    if mode == "deferred":  # lo que ya se guardó al posponer sigue contando
        summary.update(
            {
                key: previous_report[key]
                for key in ("soulseek_finished", "saved_spotify_ids", "final_output", "playlist_name")
                if key in previous_report
            }
        )
    write_summary(summary_path, summary)

    def partial(reason: str) -> int:
        return finish_partial(
            run_dir=run_dir,
            audio_dirs=(soulseek_dir, recorded_dir, failed_dir),
            playlists_root=playlists_root,
            playlist_id=playlist_id,
            summary=summary,
            summary_path=summary_path,
            reason=reason,
        )

    try:
        # Lista de faltantes cuya grabación se pospuso: (manifiesto, temas ya grabados).
        missing_to_record: tuple[dict[str, Any], int] | None = None
        if mode == "deferred":
            assert resume is not None
            manifest = resume[2]
            summary["missing_count"] = int(manifest["missing_count"])
            completed_missing = int((resume[3] or {}).get("completed_count") or 0)
            if completed_missing < int(manifest["missing_count"]):
                missing_to_record = (manifest, completed_missing)
            failed_manifest = final_missing_manifest(failed_json)
            if failed_manifest is None:
                raise WorkflowError(
                    "Falta la lista de temas que Soulseek no pudo dar: empieza la playlist de cero."
                )
            failed_completed = int(
                (read_json_dict(failed_dir / "recording_progress.json") or {}).get("completed_count") or 0
            )
            print("\nLa descarga de Soulseek ya había terminado: sigo con la grabación pospuesta.")
        elif mode != "failed":
            # Una lista vieja no puede pasar por el final de esta descarga.
            failed_json.unlink(missing_ok=True)
            downloader_cmd = [
                sys.executable,
                "-u",
                str(SCRIPT_DIR / "spotify_soulseek_download.py"),
                playlist_url,
                "--output",
                str(soulseek_dir),
                "--missing-json",
                str(downloader_missing_json),
                "--failed-json",
                str(failed_json),
                "--preflight-status",
                str(preflight_status),
                "--availability-report",
                str(availability_json),
                "--report",
                str(soulseek_report),
                "--download-gate",
                str(download_gate),
                "--download-gate-timeout",
                str(args.manifest_timeout),
            ]
            if is_resume:
                downloader_cmd.append("--no-missing-playlist")
            print("\n[1/4] Analizando disponibilidad en Soulseek...")
            downloader = subprocess.Popen(downloader_cmd, env=env)

            if mode == "missing":
                assert resume is not None
                manifest = resume[2]
            else:
                manifest = wait_for_missing_manifest(
                    downloader,
                    missing_json,
                    args.manifest_timeout,
                    status_path=preflight_status,
                )
                search_error = preflight_failed_entirely(run_dir)
                if search_error is not None:
                    raise WorkflowError(
                        "La búsqueda en Soulseek falló para todos los temas "
                        f"({search_error}). No se graba nada: el problema es la búsqueda, no los "
                        "temas. Revisa que la cuenta de Soulseek no esté abierta en otra PC y "
                        "vuelve a intentar."
                    )
            missing_count = int(manifest["missing_count"])
            summary["missing_count"] = missing_count
            completed_count = 0
            if mode == "missing":
                assert resume is not None and resume[3] is not None
                completed_count = int(resume[3]["completed_count"])
                summary["completed_before_resume"] = completed_count
                print(f"\nSesión recuperada: {missing_count} tema(s) en la playlist de faltantes.")
            else:
                print(f"\nPreflight completo: {missing_count} tema(s) faltante(s).")

            if missing_count:
                batches = missing_playlist_batches(manifest)
                if not batches:
                    raw = manifest.get("generated_missing_playlists") or []
                    errors = [
                        str(item.get("error"))
                        for item in raw
                        if isinstance(item, dict) and item.get("error")
                    ]
                    detail = "; ".join(errors) or "Spotify no devolvió playlists válidas"
                    raise WorkflowError(
                        "No se pudieron obtener las playlists de faltantes: " + detail
                    )
                summary["missing_playlists"] = [
                    {
                        "batch_number": batch["batch_number"],
                        "start_index": batch["start_index"],
                        "end_index": batch["end_index"],
                        "track_count": batch["track_count"],
                        "url": batch["url"],
                    }
                    for batch in batches
                ]
                write_summary(summary_path, summary)
                if defer_file.exists():
                    print(
                        f"\n[2/4] Grabación pospuesta: {missing_count} tema(s) que no están en Soulseek "
                        "quedan para grabar después."
                    )
                    missing_to_record = (manifest, completed_count)
                else:
                    print("\n[2/4] Grabando de Spotify los temas que no están en Soulseek...")
                    record_batches(
                        label="faltantes",
                        manifest_path=missing_json,
                        batches=batches,
                        total=missing_count,
                        completed_count=completed_count,
                        recorded_dir=recorded_dir,
                        run_dir=run_dir,
                        env=env,
                        args=args,
                        summary=summary,
                        summary_path=summary_path,
                        keys=("completed_recordings", "completed_batches", "active_batch"),
                        downloader=downloader,
                        on_first_playback=lambda: release_downloads(download_gate),
                    )
            else:
                print("\nNo hay faltantes: no es necesario reproducir ni grabar Spotify.")
            if not download_gate.exists():
                release_downloads(download_gate)

            print("\n[3/4] Esperando a que terminen las descargas de Soulseek...")
            while downloader.poll() is None:
                time.sleep(0.5)
            downloader_code = int(downloader.returncode or 0)
            summary["soulseek_returncode"] = downloader_code
            write_summary(summary_path, summary)
            failed_manifest = (
                final_missing_manifest(failed_json)
                if downloader_code in DOWNLOADER_FINISHED_CODES
                else None
            )
            if failed_manifest is None:
                return partial(
                    f"La descarga Soulseek terminó con código {downloader_code} "
                    "sin la lista de temas para grabar."
                )
            failed_completed = 0
        else:
            assert resume is not None
            failed_manifest = resume[2]
            failed_completed = int((resume[3] or {}).get("completed_count") or 0)
            summary["completed_failed_before_resume"] = failed_completed
            print("\nLa descarga de Soulseek ya había terminado: sigo con la grabación pendiente.")

        failed_count = int(failed_manifest["missing_count"])
        summary["failed_count"] = failed_count
        pending = max(0, failed_count - failed_completed)
        if missing_to_record is not None:
            pending += int(missing_to_record[0]["missing_count"]) - missing_to_record[1]
        if pending and defer_file.exists():  # se puede haber pedido durante la descarga
            return finish_deferred(
                run_dir=run_dir,
                soulseek_dir=soulseek_dir,
                playlists_root=playlists_root,
                playlist_id=playlist_id,
                summary=summary,
                summary_path=summary_path,
                pending=pending,
            )
        if missing_to_record is not None:
            manifest, completed_missing = missing_to_record
            print("\n[2/4] Grabando de Spotify los temas que no están en Soulseek...")
            record_batches(
                label="faltantes",
                manifest_path=missing_json,
                batches=missing_playlist_batches(manifest),
                total=int(manifest["missing_count"]),
                completed_count=completed_missing,
                recorded_dir=recorded_dir,
                run_dir=run_dir,
                env=env,
                args=args,
                summary=summary,
                summary_path=summary_path,
                keys=("completed_recordings", "completed_batches", "active_batch"),
            )
        if failed_count > failed_completed:
            failed_batches = missing_playlist_batches(failed_manifest)
            if not failed_batches:
                raw = failed_manifest.get("generated_missing_playlists") or []
                errors = [
                    str(item.get("error"))
                    for item in raw
                    if isinstance(item, dict) and item.get("error")
                ]
                return partial(
                    "No se pudieron crear las playlists de Spotify para grabar los temas "
                    "que no se bajaron: " + ("; ".join(errors) or "Spotify no devolvió playlists válidas")
                )
            summary["failed_playlists"] = [
                {
                    "batch_number": batch["batch_number"],
                    "start_index": batch["start_index"],
                    "end_index": batch["end_index"],
                    "track_count": batch["track_count"],
                    "url": batch["url"],
                }
                for batch in failed_batches
            ]
            write_summary(summary_path, summary)
            print(
                f"\n[4/4] Grabando de Spotify {failed_count} tema(s) que Soulseek tenía "
                "pero no se pudieron bajar..."
            )
            record_batches(
                label="no bajados",
                manifest_path=failed_json,
                batches=failed_batches,
                total=failed_count,
                completed_count=failed_completed,
                recorded_dir=failed_dir,
                run_dir=run_dir,
                env=env,
                args=args,
                summary=summary,
                summary_path=summary_path,
                keys=("completed_failed_recordings", "completed_failed_batches", "active_failed_batch"),
            )

        playlist_name, expected_ids = playlist_identity(run_dir, playlist_id)
        summary["status"] = "consolidating"
        summary["playlist_name"] = playlist_name
        write_summary(summary_path, summary)
        saved_ids = set(summary.get("saved_spotify_ids") or [])  # ya en la carpeta al posponer
        try:
            playlist_dir, moved_paths, consolidation = consolidate_audio(
                run_dir,
                soulseek_dir,
                recorded_dir,
                playlists_root,
                playlist_name,
                expected_spotify_ids=expected_ids - saved_ids if expected_ids is not None else None,
                extra_recorded_dirs=(failed_dir,),
            )
        except IncompleteAudio as exc:
            return partial(str(exc))
        consolidation = _with_saved_before(run_dir, summary, consolidation)
        consolidation_path = run_dir / "consolidation_report.json"
        write_summary(consolidation_path, consolidation)
        diagnostics_dir = simplify_completed_run(run_dir, moved_paths)

        summary.pop("soulseek_output", None)
        summary.pop("recorded_output", None)
        summary.pop("playlists_root", None)
        summary["status"] = "completed"
        summary["final_output"] = str(playlist_dir)
        summary["audio_files"] = consolidation["audio_count"]
        summary["moved_files"] = consolidation["moved"]
        summary["deduplicated_files"] = consolidation["deduplicated"]
        summary["diagnostics"] = str(diagnostics_dir)
        write_summary(summary_path, summary)
        print("\n=== FLUJO COMPLETADO ===")
        print(f"Playlist : {playlist_dir}")
        print(f"Audios   : {consolidation['audio_count']}")
        print(f"Debug    : {diagnostics_dir}")
        print(f"Reporte  : {summary_path}")
        return 0
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        write_summary(summary_path, summary)
        print("\nInterrumpido. Cerrando procesos hijos...", file=sys.stderr)
        return 130
    except ResumeAvailable:
        summary["status"] = "resumable"
        write_summary(summary_path, summary)
        raise
    except WorkflowError:
        if summary.get("status") != "failed":
            summary["status"] = "failed"
            write_summary(summary_path, summary)
        raise
    finally:
        terminate_process(downloader)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Orquesta preflight/descarga Soulseek y grabación Spotify de faltantes."
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
        "--manifest-timeout",
        type=float,
        default=900.0,
        help="Segundos máximos para el preflight y la playlist de faltantes",
    )
    parser.add_argument(
        "--recorder-timeout",
        type=float,
        default=30.0,
        help="Segundos máximos para abrir el dispositivo loopback",
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
