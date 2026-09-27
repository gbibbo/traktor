#!/usr/bin/env python3
"""
PURPOSE: Run the complete Spotify -> Soulseek + loopback fallback workflow.

CHANGELOG:
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
from typing import Any


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


class WorkflowError(RuntimeError):
    """A user-facing orchestration failure."""


class ResumeAvailable(WorkflowError):
    """Playback stopped, but completed per-track checkpoints remain reusable."""


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

    generated = payload.get("generated_missing_playlist")
    if not isinstance(generated, dict):
        return None
    if generated.get("url") or generated.get("error"):
        return payload
    return None


def read_json_dict(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def resumable_run(
    runs_root: Path,
    playlist_id: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]] | None:
    """Find the newest compatible run with at least one pending recording."""
    if not runs_root.is_dir():
        return None

    candidates = sorted(
        (
            path
            for path in runs_root.iterdir()
            if path.is_dir() and path.name.endswith(f"_{playlist_id}")
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    for run_dir in candidates:
        manifest = final_missing_manifest(run_dir / "soulseek_missing.json")
        progress = read_json_dict(
            run_dir / "recorded_missing" / "recording_progress.json"
        )
        if manifest is None or progress is None:
            continue
        missing_count = manifest.get("missing_count")
        completed_count = progress.get("completed_count")
        if not isinstance(missing_count, int) or missing_count <= 0:
            continue
        if not isinstance(completed_count, int):
            continue
        if progress.get("total_tracks") != missing_count:
            continue
        if not 0 <= completed_count < missing_count:
            continue
        if progress.get("status") == "completed":
            continue
        generated = manifest.get("generated_missing_playlist") or {}
        if not isinstance(generated, dict) or not generated.get("url"):
            continue
        return run_dir, manifest, progress
    return None


def wait_for_missing_manifest(
    process: subprocess.Popen[Any],
    path: Path,
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
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
    if embedded_spotify_id(destination) == spotify_id:
        return destination, True

    alternate = playlist_dir / f"{source.stem} [{spotify_id}]{source.suffix}"
    if not alternate.exists():
        return alternate, False
    if embedded_spotify_id(alternate) == spotify_id:
        return alternate, True
    raise WorkflowError(f"Colisión de audio no resoluble: {alternate}")


def consolidate_audio(
    run_dir: Path,
    soulseek_dir: Path,
    recorded_dir: Path,
    playlists_root: Path,
    playlist_name: str,
    expected_spotify_ids: set[str] | None = None,
) -> tuple[Path, dict[str, str], dict[str, Any]]:
    """Move every final MP3 into one playlist directory without duplicates."""
    run_root = run_dir.resolve()
    sources: list[tuple[str, Path]] = []
    for origin, root in (("soulseek", soulseek_dir), ("loopback", recorded_dir)):
        if root.is_dir():
            sources.extend(
                (origin, path)
                for path in sorted(root.rglob("*.mp3"), key=lambda item: str(item).casefold())
                if path.is_file()
            )

    identified_sources = [
        (origin, source, embedded_spotify_id(source))
        for origin, source in sources
    ]
    found_ids = {spotify_id for _, _, spotify_id in identified_sources}
    if expected_spotify_ids is not None and found_ids != expected_spotify_ids:
        missing = sorted(expected_spotify_ids - found_ids)
        unexpected = sorted(found_ids - expected_spotify_ids)
        raise WorkflowError(
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


def run_workflow(args: argparse.Namespace) -> int:
    playlist_id = playlist_id_from_url(args.playlist_url)
    playlist_url = canonical_playlist_url(args.playlist_url)
    load_environment()
    validate_environment()

    sockseek = find_required_program("sockseek")
    ffmpeg = find_required_program("ffmpeg")
    env = child_environment(sockseek, ffmpeg)

    runs_root = args.output_root.expanduser().resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    playlists_root = args.library_root.expanduser().resolve()
    resume_data = None if args.fresh else resumable_run(runs_root, playlist_id)
    is_resume = resume_data is not None
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (
        resume_data[0]
        if resume_data is not None
        else runs_root / f"{timestamp}_{playlist_id}"
    )
    soulseek_dir = run_dir / "soulseek"
    recorded_dir = run_dir / "recorded_missing"
    missing_json = run_dir / "soulseek_missing.json"
    if is_resume:
        availability_json = run_dir / f"soulseek_availability_resume_{timestamp}.json"
        downloader_missing_json = run_dir / f"soulseek_missing_resume_{timestamp}.json"
        soulseek_report = run_dir / f"spotify_reconcile_report_resume_{timestamp}.json"
    else:
        availability_json = run_dir / "soulseek_availability.json"
        downloader_missing_json = missing_json
        soulseek_report = run_dir / "spotify_reconcile_report.json"
    recorder_ready = run_dir / "recorder_ready.json"
    player_ready = run_dir / "player_ready.json"
    playback_status = run_dir / "spotify_playback_status.json"
    download_gate = run_dir / "start_soulseek_downloads.signal"
    summary_path = run_dir / "orchestration_report.json"
    if is_resume:
        recorder_ready.unlink(missing_ok=True)
        player_ready.unlink(missing_ok=True)
        playback_status.unlink(missing_ok=True)
        download_gate.unlink(missing_ok=True)
    else:
        run_dir.mkdir(parents=True, exist_ok=False)

    print("\n=== FLUJO AUTOMÁTICO SPOTIFY / SOULSEEK ===")
    print(f"Playlist : {playlist_url}")
    print(f"Run      : {run_dir}")
    if is_resume:
        assert resume_data is not None
        completed = int(resume_data[2]["completed_count"])
        total = int(resume_data[2]["total_tracks"])
        print(f"Resume   : {completed}/{total} pistas ya conservadas")

    downloader: subprocess.Popen[Any] | None = None
    recorder: subprocess.Popen[Any] | None = None
    player: subprocess.Popen[Any] | None = None
    summary: dict[str, Any] = {
        "source_playlist": playlist_url,
        "playlist_id": playlist_id,
        "run_dir": str(run_dir),
        "missing_json": str(missing_json),
        "soulseek_output": str(soulseek_dir),
        "recorded_output": str(recorded_dir),
        "playlists_root": str(playlists_root),
        "status": "running",
        "resumed": is_resume,
    }
    write_summary(summary_path, summary)

    try:
        downloader_cmd = [
            sys.executable,
            "-u",
            str(SCRIPT_DIR / "spotify_soulseek_download.py"),
            playlist_url,
            "--output",
            str(soulseek_dir),
            "--missing-json",
            str(downloader_missing_json),
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
        print("\n[1/3] Analizando disponibilidad en Soulseek...")
        downloader = subprocess.Popen(downloader_cmd, env=env)

        if is_resume:
            assert resume_data is not None
            manifest = resume_data[1]
        else:
            manifest = wait_for_missing_manifest(
                downloader,
                missing_json,
                args.manifest_timeout,
            )
        missing_count = int(manifest["missing_count"])
        summary["missing_count"] = missing_count
        if is_resume:
            summary["completed_before_resume"] = int(
                resume_data[2]["completed_count"]
            )
        if is_resume:
            print(f"\nSesión recuperada: {missing_count} tema(s) en la playlist de faltantes.")
        else:
            print(f"\nPreflight completo: {missing_count} tema(s) faltante(s).")

        if missing_count:
            generated = manifest.get("generated_missing_playlist") or {}
            missing_playlist_url = str(generated.get("url") or "")
            if not missing_playlist_url:
                detail = generated.get("error") or "Spotify no devolvió una URL"
                raise WorkflowError(
                    "No se pudo obtener la playlist de faltantes: " + str(detail)
                )
            summary["missing_playlist_url"] = missing_playlist_url
            completed_count = (
                int(resume_data[2]["completed_count"])
                if resume_data is not None
                else 0
            )
            write_summary(summary_path, summary)

            recorder_cmd = [
                sys.executable,
                "-u",
                str(SCRIPT_DIR / "playlist_loopback_recorder.py"),
                str(missing_json),
                "--output-dir",
                str(recorded_dir),
                "--ready-file",
                str(recorder_ready),
                "--resume",
                "--watchdog-file",
                str(playback_status),
            ]
            print("\n[2/3] Abriendo el grabador loopback...")
            recorder = subprocess.Popen(recorder_cmd, env=env)
            wait_for_ready_file(
                recorder,
                recorder_ready,
                args.recorder_timeout,
                label="grabador",
            )

            print("\n[3/3] Iniciando la playlist Spotify de faltantes...")
            player_cmd = [
                sys.executable,
                "-u",
                str(SCRIPT_DIR / "spotify_playlist.py"),
                missing_playlist_url,
                "--offset-position",
                str(completed_count),
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
            release_downloads(download_gate)
        else:
            print("\nNo hay faltantes: no es necesario reproducir ni grabar Spotify.")
            release_downloads(download_gate)

        while True:
            downloader_running = downloader.poll() is None
            recorder_running = recorder is not None and recorder.poll() is None
            recorder_current_code = recorder.returncode if recorder is not None else None
            player_current_code = player.poll() if player is not None else None
            if (
                player_current_code is not None
                and player_current_code not in {0, 75}
                and recorder_running
            ):
                write_summary(
                    playback_status,
                    {
                        "status": "stalled",
                        "error": f"Spotify player exited with code {player_current_code}",
                    },
                )
            if recorder is not None and not recorder_running:
                terminate_process(player)
            if recorder_current_code in {75, 130}:
                progress = read_json_dict(recorded_dir / "recording_progress.json") or {}
                completed_now = progress.get("completed_count", completed_count)
                summary["status"] = "resumable"
                summary["completed_recordings"] = completed_now
                summary["recorder_returncode"] = recorder_current_code
                write_summary(summary_path, summary)
                terminate_process(downloader)
                raise ResumeAvailable(
                    "Spotify se detuvo. Se conservaron "
                    f"{completed_now}/{missing_count} pistas; relanza el mismo comando "
                    "para continuar desde la primera pendiente."
                )
            if not downloader_running and not recorder_running:
                break
            time.sleep(0.5)

        downloader_code = int(downloader.returncode or 0)
        recorder_code = int(recorder.returncode or 0) if recorder else None
        summary["soulseek_returncode"] = downloader_code
        summary["recorder_returncode"] = recorder_code

        if downloader_code != 0 or (recorder_code is not None and recorder_code != 0):
            summary["status"] = "failed"
            write_summary(summary_path, summary)
            raise WorkflowError(
                "Una rama del flujo falló: "
                f"Soulseek={downloader_code}, grabador={recorder_code}."
            )

        availability = read_json_dict(availability_json)
        if availability is None:
            raise WorkflowError(
                f"No se pudo leer el reporte final de disponibilidad: {availability_json}"
            )
        playlist_data = availability.get("playlist") or {}
        playlist_name = str(
            playlist_data.get("name")
            or (manifest.get("source_playlist") or {}).get("name")
            or playlist_id
        )
        expected_ids = {
            str((row.get("spotify") or {}).get("spotify_id"))
            for row in availability.get("tracks") or []
            if isinstance(row, dict) and (row.get("spotify") or {}).get("spotify_id")
        }

        summary["status"] = "consolidating"
        summary["playlist_name"] = playlist_name
        write_summary(summary_path, summary)
        playlist_dir, moved_paths, consolidation = consolidate_audio(
            run_dir,
            soulseek_dir,
            recorded_dir,
            playlists_root,
            playlist_name,
            expected_spotify_ids=expected_ids,
        )
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
        summary["status"] = "failed"
        write_summary(summary_path, summary)
        raise
    finally:
        terminate_process(player)
        terminate_process(recorder)
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run_workflow(args)
    except ResumeAvailable as exc:
        print(f"PAUSADO: {exc}", file=sys.stderr)
        return 75
    except WorkflowError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
