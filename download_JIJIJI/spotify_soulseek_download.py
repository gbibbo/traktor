#!/usr/bin/env python3
"""
spotify_soulseek_download.py
============================

PURPOSE: Reconcile a Spotify playlist against Soulseek and download matches.

CHANGELOG:
- 2026-11-06: Several tracks at once (--parallel, 2 by default) through one Soulseek session: a sockseek
              daemon started for the run takes every search and transfer (separate sockseek processes
              each log in with the same account and cut each other off). A transfer that receives
              nothing for TRANSFER_STALL_SECONDS is dropped and the next file is tried, and the daemon
              job is cancelled; candidates with a free upload slot and a faster peer go first; up to
              MAX_TRANSFER_ATTEMPTS files per track when transfers fail, within a time budget; a track
              whose files all failed to arrive is searched again at the end before it goes to
              recording. --sockseek-mock-dir runs everything against local files (sockseek's mock).
              Spotify.remove_from_library (DELETE /me/library) removes the temporary playlists.
- 2026-11-06: Track by track, as when searching by hand: each track is searched on its own (one sockseek
              process) and, if it is there, downloaded right away. The batch preflight opened 20 searches
              at once, spent the 34 allowed in under 2 minutes and then sat idle ~2 minutes, over and over
              (what looked like Soulseek being stuck). What cannot be had goes to --recording-queue as soon as
              it is known, so the orchestrator records it while the search goes on. The 34 searches / 220 s
              limit is kept across processes and runs (--search-log); a failed search is retried once, and
              after 3 tracks in a row the run stops without sending them to recording. A resumed run keeps
              what was searched, downloaded or queued. Removed: the batch preflight, --download-gate,
              --failed-json, --preflight-status and --preflight-timeout.
- 2026-11-02: --have-dir (the playlist's music folder, read-only) and --reuse-dir tracks are neither
              searched nor downloaded: a new run only searches what is missing.
- 2026-11-02: Preflight without --strict-title: each file's title and version are judged with the
              verification's own rules, reading the file name as its tag would be. --strict-title
              wanted Spotify's «Title - Version» literally and lost «Title (Version)» (29 of 46
              "missing" in a 152-track playlist). Sockseek's length tolerance reaches 1.9× (the
              Extended rule); the per-track range stays in the classification.
- 2026-11-02: --reuse-dir: finished MP3s of earlier runs of the playlist are checked again and reused,
              not downloaded again nor listed as missing.
- 2026-11-02: The preflight reports its progress as it goes (sockseek --progress-json: searched and
              found, in --preflight-status and every 10 tracks in the output) and leaves Sockseek's
              own log next to it (sockseek_preflight.log).
- 2026-11-02: Spotify authorization link for the app to show (copy it to another browser), 5 min to
              authorize; reading a playlist prints its progress and every rate-limit wait.
- 2026-11-02: A failed Sockseek transfer prints its exit code and the tail of Sockseek's output
              (it used to be skipped in silence and its temporary log deleted).
- 2026-10-30: Preflight timeout scales with the playlist (Sockseek's 34 searches / 220 s limiter made
              every playlist above ~100 tracks time out at the fixed 600 s); --preflight-status tells
              the orchestrator how long to wait; a preflight that failed for every track is not sent
              to recording.
- 2026-10-30: Try up to MAX_CANDIDATE_TRANSFERS candidates per track, the Spotify duration
              and album first, verifying each one before accepting it (it used to keep the first
              file that transferred, and a rejection lost the track).
- 2026-10-30: Split combined artist tags ("A & B", "A e B") and accept an exact ISRC even when the
              artist tag is written differently.
- 2026-10-30: --failed-json: tracks that could not be downloaded (or whose preflight failed) go to
              private Spotify playlists so the orchestrator can record them.
- 2026-10-05: Monitor transfers, bound ffmpeg and verify/rebuild cover images.
- 2026-10-05: Gate exact Soulseek files by size, content and mandatory antivirus.
- 2026-09-29: Split missing tracks into private Spotify playlists of 10 tracks.
- 2026-09-27: Match the Windows Media-compatible Soulseek ID3 frame profile.
- 2026-09-27: Expose the canonical Spotify metadata/cover writer for recordings.
- 2026-09-27: Add an optional pre-download gate for external orchestration.

Recorre una playlist de Spotify tema a tema: busca cada uno en Soulseek y, si está,
lo baja en el momento; lo que no se consigue va a la cola de grabación (o, sin cola, a
una lista y una playlist Spotify privada con los faltantes). Prefiere versiones Extended
cuando existen y verifica de forma conservadora la identidad del tema, escribe metadata
de Spotify + cover oficial y deja todo en MP3 320 kbps.

Requisitos:
  - Python 3.10+
  - pip install requests mutagen
  - ffmpeg en el PATH
  - sockseek (o sldl) en el PATH
  - App de Spotify con Redirect URI exactamente:
        http://127.0.0.1:48721/callback

Uso rápido:
  export SPOTIFY_CLIENT_ID=TU_CLIENT_ID
  # PowerShell: $env:SPOTIFY_CLIENT_ID="TU_CLIENT_ID"

  python spotify_soulseek_download_fixed.py \
      "https://open.spotify.com/playlist/XXXX" \
      -o ./descargas

La primera vez se abre el navegador para autorizar Spotify.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import http.server
import io
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import urllib.parse
import urllib.request
import warnings
from dataclasses import asdict, dataclass, fields
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

try:
    import mutagen
    from mutagen.flac import FLAC
    from mutagen.id3 import (
        APIC,
        ID3,
        TALB,
        TDRC,
        TIT2,
        TPE1,
        TPE2,
        TPOS,
        TRCK,
        TSRC,
        TXXX,
    )
    from mutagen.mp3 import MP3
    from mutagen.wave import WAVE
except ImportError:
    print("Instala mutagen: pip install mutagen", file=sys.stderr)
    sys.exit(1)

try:
    import requests
except ImportError:
    print("Instala requests: pip install requests", file=sys.stderr)
    sys.exit(1)

import socket  # noqa: E402

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from recording_queue import END_FINISHED, END_SEARCH_FAILED, append_track, close_queue, read_queue  # noqa: E402


# ---------------------------------------------------------------------------
# Spotify
# ---------------------------------------------------------------------------

AUTHORIZE_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
API_ROOT = "https://api.spotify.com/v1"
DEFAULT_REDIRECT = "http://127.0.0.1:48721/callback"
AUTH_WAIT_S = 300                 # tiempo para autorizar, también copiando el enlace a otro navegador
SCOPES = "playlist-read-private playlist-read-collaborative playlist-modify-private"
REQUIRED_SCOPES = set(SCOPES.split())
DEFAULT_MISSING_PLAYLIST_BATCH_SIZE = 10
DURATION_TOLERANCE_S = 8          # misma versión que Spotify (no Extended)
MAX_CANDIDATE_TRANSFERS = 4       # archivos que se llegan a bajar por tema hasta que uno pase la verificación
MAX_TRANSFER_ATTEMPTS = 8         # transferencias por tema contando las que fallan por la red (no llegó nada)
TRACK_DOWNLOAD_BUDGET_S = 420     # tiempo por tema: después se deja para el segundo intento o para grabar
TRANSFER_STALL_SECONDS = 45       # sin recibir ni un byte: el usuario no responde o nos tiene en cola
DEFAULT_PARALLEL = 2              # temas a la vez (búsqueda y descarga), con una sola sesión de Soulseek
MAX_PARALLEL = 4
DAEMON_STATE_NAME = "sockseek_daemon.json"  # PID y puerto del daemon de la descarga, para cerrarlo si queda
# Soulseek banea la cuenta 30 min si se pasan 34 búsquedas cada 220 s (el límite de sockseek). Con un
# proceso de sockseek por búsqueda, la cuenta la lleva SearchLimiter, en un archivo del ordenador.
SOULSEEK_SEARCHES_PER_WINDOW = 34
SOULSEEK_SEARCH_WINDOW_S = 220
SEARCH_LOG_NAME = "traktor_soulseek_searches.json"
DEFAULT_SEARCH_TIMEOUT_S = 90.0      # conectar (hasta 20 s) + 5 s de búsqueda de sockseek, con margen
SEARCH_ATTEMPTS = 2                  # una búsqueda que falla se reintenta una vez
SEARCH_RETRY_PAUSE_S = 15.0
MAX_CONSECUTIVE_SEARCH_ERRORS = 3    # tantos temas seguidos sin búsqueda: el problema es la búsqueda
EXIT_SEARCH_FAILED = 4
DOWNLOADED_STATUSES = {"finalized", "already_finalized"}
ALLOWED_AUDIO_FORMATS = {"mp3", "flac", "wav"}
SIZE_MARGIN = 0.15
TAG_ALLOWANCE_BYTES = 2 * 1024 * 1024
MAX_SOURCE_BYTES = 512 * 1024 * 1024
TRANSFER_POLL_SECONDS = 0.1
TRANSFER_TIMEOUT_SECONDS = 360
MIN_DISK_FREE_BYTES = 64 * 1024 * 1024
MAX_PROCESS_LOG_BYTES = 2 * 1024 * 1024
FAILED_TRANSFER_LOG_LINES = 12    # lo último que dijo sockseek cuando una transferencia falla
_CREDENTIAL_LINE = re.compile(r"pass|token|secret", re.IGNORECASE)
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
FFMPEG_TIMEOUT_SECONDS = 180
MAX_COVER_PIXELS = 16_000_000
_DANGEROUS_SUFFIX = re.compile(
    r"\.(?:exe|com|scr|msi|dll|bat|cmd|ps1|vbs|js|jse|wsf|lnk|url|hta|jar|zip|rar|7z)(?:$|[.\s])",
    re.IGNORECASE,
)


def _positive_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def _safe_audio_filename(filename: str) -> bool:
    parts = filename.replace("\\", "/").split("/")
    return bool(
        filename
        and all(part and part not in {".", ".."} and ":" not in part for part in parts)
        and not any(unicodedata.category(ch).startswith("C") for ch in filename)
        and Path(parts[-1]).suffix.lower().lstrip(".") in ALLOWED_AUDIO_FORMATS
        and not _DANGEROUS_SUFFIX.search(parts[-1])
        and not parts[-1].endswith((" ", "."))
    )


def _audio_size_error(candidate: dict[str, Any]) -> str | None:
    """Conservative byte bounds; peer metadata is untrusted, not proof of safety."""
    size = _positive_number(candidate.get("size_bytes"))
    length = _positive_number(candidate.get("length_s"))
    fmt = candidate.get("format")
    if size is None or length is None or size != int(size) or length > 86400:
        return "tamaño/duración ausentes o inválidos"
    if size > MAX_SOURCE_BYTES:
        return "archivo supera el límite de 512 MiB"
    if fmt == "mp3":
        bitrate = _positive_number(candidate.get("bitrate_kbps"))
        if candidate.get("bitrate_kbps") is not None and bitrate is None:
            return "bitrate inválido"
        if bitrate is not None and not 315 <= bitrate <= 325:
            return "MP3 fuera del rango de 320 kbps"
        expected = length * (bitrate or 320) * 1000 / 8
        lower, upper = expected * (1 - SIZE_MARGIN), expected * (1 + SIZE_MARGIN)
    elif fmt in {"flac", "wav"}:
        rate = _positive_number(candidate.get("sample_rate"))
        depth = _positive_number(candidate.get("bit_depth"))
        channels = _positive_number(candidate.get("channels", 2))
        if (rate is None or depth is None or channels not in {1, 2}
                or not 8000 <= rate <= 192000 or depth not in {8, 16, 24, 32}):
            return "parámetros PCM ausentes o fuera de rango"
        expected = length * rate * depth * channels / 8
        # Soulseek does not advertise channels; allow mono with the stereo estimate.
        lower = expected * (0.01 if fmt == "flac" else 0.5 * (1 - SIZE_MARGIN))
        upper = expected * (1 + SIZE_MARGIN)
    else:
        return "formato no permitido"
    if not round(lower) <= size <= round(upper) + TAG_ALLOWANCE_BYTES:
        return "tamaño no proporcional a duración/calidad"
    return None


def _antivirus_command(path: Path) -> list[str]:
    if os.name == "nt":
        platform = Path(os.environ.get("PROGRAMDATA", "C:/ProgramData")) / "Microsoft/Windows Defender/Platform"
        candidates = sorted(platform.glob("*/MpCmdRun.exe"), reverse=True)
        candidates.append(Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Windows Defender/MpCmdRun.exe")
        for executable in candidates:
            if executable.is_file():
                # No remediation: code 0 must mean no detection, not a repaired file.
                return [str(executable), "-Scan", "-ScanType", "3", "-File", str(path.resolve()), "-DisableRemediation"]
    else:
        executable = shutil.which("clamscan")
        if executable:
            return [
                executable, "--no-summary", "--max-filesize=512M",
                "--max-scansize=1024M", "--alert-exceeds-max=yes",
                "--", str(path.resolve()),
            ]
    raise RuntimeError("antivirus requerido: Microsoft Defender en Windows o clamscan en Linux")


def _scan_antivirus(path: Path) -> None:
    try:
        result = subprocess.run(
            _antivirus_command(path), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=120, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("no se pudo completar el análisis antivirus; archivo bloqueado") from exc
    if result.returncode != 0 or not path.is_file():
        raise RuntimeError(f"antivirus detectó una amenaza o falló (código {result.returncode}); archivo bloqueado")


def _check_antivirus_ready() -> None:
    with tempfile.TemporaryDirectory(prefix="soulseek_antivirus_") as directory:
        probe = Path(directory) / "probe.txt"
        probe.write_text("Soulseek antivirus readiness check\n", encoding="utf-8")
        _scan_antivirus(probe)


_PRINT_LOCK = threading.Lock()
_TRACK_TAG = threading.local()


def say(*parts: Any, file: Any = None) -> None:
    """print para lo que pasa con un tema: con varios a la vez, cada línea lleva el número del tema."""
    text = " ".join(str(part) for part in parts)
    tag = getattr(_TRACK_TAG, "value", None)
    if tag:
        text = "\n".join(f"{tag:>5} {line}" if line.strip() else line for line in text.split("\n"))
    with _PRINT_LOCK:
        print(text, file=file or sys.stdout, flush=True)


# ------------------------------------------------------------------ una sola sesión de Soulseek
_SOCKSEEK_REMOTE: str | None = None      # URL del daemon de esta descarga
_SOCKSEEK_MOCK_DIR: Path | None = None   # pruebas sin Soulseek: archivos locales (--mock-files-dir de sockseek)
_DAEMON: "SockseekDaemon | None" = None


def _sockseek_session_args(tmp_dir: Path, binary: str) -> list[str]:
    """Cómo entra un proceso de sockseek a Soulseek: por el daemon de la descarga (una sola sesión para
    todo), con archivos locales de prueba, o con su propia sesión (credenciales sin perfiles ni comandos)."""
    if _SOCKSEEK_REMOTE:
        return ["--no-config", f"--remote={_SOCKSEEK_REMOTE}"]
    if _SOCKSEEK_MOCK_DIR is not None:
        return ["--no-config", f"--mock-files-dir={_SOCKSEEK_MOCK_DIR}", "--no-listen"]
    config = tmp_dir / "sockseek.conf"
    _write_sockseek_safe_config(config, binary)
    return [f"--config={config}"]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _process_command_line(pid: int) -> str | None:
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                 f"(Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}').CommandLine"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() or None
    try:
        return Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return None


def stop_daemon_from_state(state_path: Path) -> bool:
    """Cierra el daemon que dejó una descarga cortada (su sesión de Soulseek cortaría la de la próxima). Solo si
    el proceso es ese daemon: sockseek, «daemon» y el mismo puerto en su línea de comandos."""
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        pid, port = int(state["pid"]), int(state["port"])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    command = _process_command_line(pid) or ""
    stopped = False
    if "sockseek" in command.lower() and " daemon" in command and f"--server-port {port}" in command:
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, timeout=20,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            else:
                os.kill(pid, 9)
            stopped = True
        except (OSError, subprocess.TimeoutExpired):
            pass
    state_path.unlink(missing_ok=True)
    return stopped


class SockseekDaemon:
    """El daemon de sockseek de una descarga: una sola sesión de Soulseek para todas las búsquedas y
    transferencias, que llegan como procesos de sockseek con --remote. Dos sesiones con la misma cuenta se
    cortan entre sí; así se pueden hacer varias cosas a la vez."""

    def __init__(self, binary: str, *, state_path: Path, log_path: Path, mock_dir: Path | None = None):
        self.binary = binary
        self.state_path = state_path
        self.log_path = log_path
        self.mock_dir = mock_dir
        self.process: subprocess.Popen[bytes] | None = None
        self.url: str | None = None
        self._tmp: tempfile.TemporaryDirectory[str] | None = None
        self._log: Any = None

    def start(self, timeout: float = 30.0) -> str:
        self._tmp = tempfile.TemporaryDirectory(prefix="sockseek_daemon_")
        port = _free_port()
        cmd = [self.binary, "daemon", "--server-port", str(port)]
        if self.mock_dir is not None:
            cmd += ["--no-config", f"--mock-files-dir={self.mock_dir}", "--no-listen"]
        else:
            config = Path(self._tmp.name) / "sockseek.conf"
            _write_sockseek_safe_config(config, self.binary)
            cmd.append(f"--config={config}")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("ab")
        self.process = subprocess.Popen(cmd, cwd=self._tmp.name, stdin=subprocess.DEVNULL, stdout=self._log,
                                        stderr=subprocess.STDOUT)
        self.url = f"http://127.0.0.1:{port}"
        write_report(self.state_path, {"pid": self.process.pid, "port": port, "url": self.url,
                                       "started_at": time.time()})
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.stop()
                raise RuntimeError(f"el daemon de sockseek terminó al arrancar (código {self.process.returncode})")
            try:
                self._request("GET", "/api/server/status", timeout=2)
                return self.url
            except (OSError, ValueError):
                time.sleep(0.3)
        self.stop()
        raise RuntimeError("el daemon de sockseek no respondió a tiempo")

    def _request(self, method: str, path: str, timeout: float = 5) -> Any:
        assert self.url is not None
        request = urllib.request.Request(self.url + path, method=method, data=b"" if method == "POST" else None)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
        return json.loads(body) if body else None

    def cancel_matching(self, query_text: str) -> int:
        """Cancela las transferencias del daemon que siguen vivas para ese archivo (por si cortar el proceso
        de sockseek no las cancela)."""
        if not query_text:
            return 0
        cancelled = 0
        try:
            jobs = self._request("GET", "/api/jobs") or []
            for job in jobs:
                if (isinstance(job, dict) and job.get("kind") == "song" and job.get("lifecycleState") != "Terminal"
                        and query_text in {job.get("queryText"), job.get("itemName")}):
                    self._request("POST", f"/api/jobs/{job.get('jobId')}/cancel")
                    cancelled += 1
        except (OSError, ValueError):
            pass
        return cancelled

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self._log is not None:
            self._log.close()
            self._log = None
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None
        self.state_path.unlink(missing_ok=True)


def use_daemon(daemon: "SockseekDaemon | None") -> None:
    global _SOCKSEEK_REMOTE, _DAEMON
    _DAEMON = daemon
    _SOCKSEEK_REMOTE = daemon.url if daemon is not None else None


def use_mock_dir(path: Path | None) -> None:
    global _SOCKSEEK_MOCK_DIR
    _SOCKSEEK_MOCK_DIR = path


def _print_failed_transfer_log(code: int, log: Any) -> None:
    """Without this the temporary log vanishes and a failed transfer leaves no trace; lines that
    mention credentials are left out."""
    log.seek(max(0, os.fstat(log.fileno()).st_size - 4096))
    text = _ANSI_ESCAPE.sub("", log.read().decode("utf-8", errors="replace"))
    lines = [line.strip() for line in re.split(r"[\r\n]+", text)
             if line.strip() and not _CREDENTIAL_LINE.search(line)]
    say(f"    ! sockseek terminó con código {code}\n" + "\n".join(
        f"      sockseek> {line}" for line in lines[-FAILED_TRANSFER_LOG_LINES:]))


def _run_sockseek_transfer(cmd: list[str], incoming: Path, expected_bytes: int, config: Path) -> int:
    """Watch received bytes while the CLI runs; always reap it before cleaning files. A transfer that receives
    nothing new for TRANSFER_STALL_SECONDS (the peer does not answer or keeps us queued) is dropped."""
    limit = min(expected_bytes, MAX_SOURCE_BYTES)
    if limit <= 0 or shutil.disk_usage(incoming).free < limit + MIN_DISK_FREE_BYTES:
        raise RuntimeError("espacio libre insuficiente para la transferencia")
    root = incoming.resolve()
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + TRANSFER_TIMEOUT_SECONDS
        last_received, last_progress = -1, time.monotonic()
        try:
            while True:
                received = 0
                for path in incoming.rglob("*"):
                    if path == config:
                        continue
                    if path.is_symlink() or not path.resolve().is_relative_to(root):
                        raise RuntimeError("ruta de transferencia no permitida")
                    if path.is_dir():
                        continue
                    try:
                        received += path.stat().st_size
                    except FileNotFoundError:
                        continue  # Sockseek can rename .incomplete while we poll.
                if received > limit:
                    raise RuntimeError("transferencia excedió los bytes anunciados; cancelada")
                if os.fstat(log.fileno()).st_size > MAX_PROCESS_LOG_BYTES:
                    raise RuntimeError("salida excesiva del descargador; transferencia cancelada")
                if shutil.disk_usage(incoming).free < MIN_DISK_FREE_BYTES:
                    raise RuntimeError("espacio libre agotándose; transferencia cancelada")
                if time.monotonic() >= deadline:
                    raise RuntimeError("timeout de transferencia Soulseek")
                if received != last_received:
                    last_received, last_progress = received, time.monotonic()
                elif time.monotonic() - last_progress >= TRANSFER_STALL_SECONDS:
                    raise RuntimeError(f"la transferencia no avanzó en {TRANSFER_STALL_SECONDS:.0f} s (el usuario "
                                       "no responde o nos tiene en cola); pruebo otro archivo")
                code = process.poll()
                if code is not None:
                    if code != 0:
                        _print_failed_transfer_log(code, log)
                    return code
                try:
                    process.wait(timeout=TRANSFER_POLL_SECONDS)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def _check_audio_header(path: Path) -> None:
    if path.is_symlink() or not _safe_audio_filename(path.name):
        raise RuntimeError("nombre/formato de archivo no permitido")
    with path.open("rb") as stream:
        header = stream.read(12)
    fmt = path.suffix.lower()
    valid = (
        (fmt == ".mp3" and (header.startswith(b"ID3") or
         (len(header) >= 2 and header[0] == 0xff and header[1] & 0xe0 == 0xe0)))
        or (fmt == ".flac" and header.startswith(b"fLaC"))
        or (fmt == ".wav" and header.startswith(b"RIFF") and header[8:12] == b"WAVE")
    )
    if not valid:
        raise RuntimeError("contenido no coincide con el formato de audio permitido")


def _validate_download(path: Path, advertised: dict[str, Any] | None = None) -> LocalFile:
    """Scan before invoking audio parsers, then verify actual size and audio info."""
    _check_audio_header(path)
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise RuntimeError("archivo supera el límite de 512 MiB")
    if advertised and path.stat().st_size != int(advertised["size_bytes"]):
        raise RuntimeError("tamaño recibido distinto del anunciado")
    _scan_antivirus(path)
    local = inspect(path)
    if local is None:
        raise RuntimeError("contenido de audio inválido")
    error = _audio_size_error({
        "format": local.format, "length_s": local.length_s,
        "size_bytes": path.stat().st_size, "bitrate_kbps": local.bitrate_kbps,
        "sample_rate": local.sample_rate,
        "bit_depth": local.bit_depth,
        "channels": local.channels,
    })
    if error:
        raise RuntimeError(error)
    return local


def token_cache_path() -> Path:
    if os.name == "nt" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "traktor_spotify_token.json"
    return Path.home() / ".config" / "traktor_spotify_token.json"


@dataclass
class Track:
    spotify_id: str
    uri: str
    url: str
    title: str
    artists: list[str]
    album: str
    album_id: str | None
    album_artists: list[str]
    release_date: str | None
    track_number: int | None
    total_tracks: int | None
    disc_number: int | None
    duration_ms: int
    isrc: str | None
    cover_url: str | None
    explicit: bool = False
    label: str | None = None
    copyrights: list[str] | None = None
    playlist_id: str | None = None
    playlist_name: str | None = None
    added_at: str | None = None

    @property
    def primary_artist(self) -> str:
        return self.artists[0] if self.artists else "Unknown"

    @property
    def duration_s(self) -> float:
        return self.duration_ms / 1000.0


@dataclass
class LocalFile:
    path: Path
    length_s: float | None
    bitrate_kbps: float | None
    format: str
    tag_title: str | None = None
    tag_artists: list[str] | None = None
    isrc: str | None = None
    sample_rate: int | None = None
    bit_depth: int | None = None
    channels: int | None = None


@dataclass
class AvailabilityCheck:
    status: str
    result_count: int
    candidates: list[dict[str, Any]]
    quality_unknown: bool = False
    error: str | None = None

    @property
    def downloadable(self) -> bool:
        return self.status in {"available", "available_quality_unknown"}


class _Callback(http.server.BaseHTTPRequestHandler):
    code = None
    state = None
    error = None

    def do_GET(self):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        type(self).code = q.get("code", [None])[0]
        type(self).state = q.get("state", [None])[0]
        type(self).error = q.get("error", [None])[0]
        body = b"OK - puedes cerrar esta pestana."
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class Spotify:
    def __init__(self, client_id: str, redirect: str = DEFAULT_REDIRECT):
        if not client_id:
            raise ValueError("Spotify Client ID vacío")
        self.client_id = client_id
        self.redirect = redirect
        self.cache = token_cache_path()
        self.session = requests.Session()
        self._token: dict[str, Any] | None = None
        self._album_cache: dict[str, dict[str, Any]] = {}
        self.on_auth_url: Callable[[str], None] | None = None  # la app muestra el enlace para copiarlo

    def _save(self, token: dict[str, Any]):
        token = dict(token)
        token["expires_at"] = time.time() + max(
            30, float(token.get("expires_in", 3600)) - 60
        )
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        self.cache.write_text(json.dumps(token, indent=2), encoding="utf-8")
        try:
            os.chmod(self.cache, 0o600)
        except OSError:
            pass
        self._token = token

    def _load(self) -> dict[str, Any] | None:
        if not self.cache.exists():
            return None
        try:
            data = json.loads(self.cache.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def _refresh(self, refresh: str) -> dict[str, Any]:
        r = self.session.post(
            TOKEN_URL,
            data={
                "client_id": self.client_id,
                "grant_type": "refresh_token",
                "refresh_token": refresh,
            },
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        data.setdefault("refresh_token", refresh)
        previous = self._token or self._load() or {}
        if not data.get("scope") and previous.get("scope"):
            data["scope"] = previous["scope"]
        self._save(data)
        return data

    def _login(self) -> dict[str, Any]:
        parsed = urllib.parse.urlparse(self.redirect)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1"}:
            raise ValueError(
                "Usa un Redirect URI loopback explícito, por ejemplo "
                f"{DEFAULT_REDIRECT}"
            )

        verifier = secrets.token_urlsafe(64)
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .rstrip(b"=")
            .decode()
        )
        state = secrets.token_urlsafe(24)
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect,
            "scope": SCOPES,
            "code_challenge_method": "S256",
            "code_challenge": challenge,
            "state": state,
        }
        url = AUTHORIZE_URL + "?" + urllib.parse.urlencode(params)
        port = parsed.port or 80

        _Callback.code = _Callback.state = _Callback.error = None
        server = http.server.HTTPServer(("127.0.0.1", port), _Callback)
        t = threading.Thread(target=server.handle_request, daemon=True)
        t.start()
        print("Abre esta URL si el navegador no se abre solo:\n", url)
        print(f"(Tienes {AUTH_WAIT_S // 60} minutos; sirve cualquier navegador de este ordenador.)")
        if self.on_auth_url is not None:
            self.on_auth_url(url)
        try:
            import webbrowser

            webbrowser.open(url)
        except Exception:
            pass

        t.join(timeout=AUTH_WAIT_S)
        server.server_close()
        if _Callback.error:
            raise RuntimeError(f"Auth Spotify falló: {_Callback.error}")
        if not _Callback.code or _Callback.state != state:
            raise RuntimeError(
                "No se recibió el código de autorización de Spotify o falló state"
            )

        r = self.session.post(
            TOKEN_URL,
            data={
                "client_id": self.client_id,
                "grant_type": "authorization_code",
                "code": _Callback.code,
                "redirect_uri": self.redirect,
                "code_verifier": verifier,
            },
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        self._save(data)
        return data

    def _has_required_scopes(self, token: dict[str, Any] | None) -> bool:
        if not token:
            return False
        granted = set(str(token.get("scope") or "").split())
        return REQUIRED_SCOPES.issubset(granted)

    def token(self) -> str:
        tok = self._token or self._load()
        # A refresh token cannot grant scopes that were not authorized originally.
        # If an older cache predates playlist-modify-private, force a fresh PKCE login.
        if tok and not self._has_required_scopes(tok):
            tok = None
            self._token = None
        if (
            tok
            and tok.get("access_token")
            and float(tok.get("expires_at", 0)) > time.time()
        ):
            self._token = tok
            return str(tok["access_token"])
        if tok and tok.get("refresh_token"):
            try:
                self._token = self._refresh(str(tok["refresh_token"]))
                return str(self._token["access_token"])
            except requests.RequestException:
                # El refresh puede haber expirado/revocado. Reautorizar es más seguro
                # que reciclar un access token inválido.
                self._token = None
        self._token = self._login()
        return str(self._token["access_token"])

    def _force_refresh_after_401(self) -> str:
        tok = self._token or self._load() or {}
        refresh = tok.get("refresh_token")
        if refresh:
            try:
                self._token = self._refresh(str(refresh))
                return str(self._token["access_token"])
            except requests.RequestException:
                pass
        self._token = self._login()
        return str(self._token["access_token"])

    def _get(self, path: str, params: dict | None = None) -> dict[str, Any]:
        url = path if path.startswith("http") else API_ROOT + path
        headers = {"Authorization": f"Bearer {self.token()}"}

        for attempt in range(4):
            r = self.session.get(url, params=params, headers=headers, timeout=25)

            if r.status_code == 401:
                headers = {
                    "Authorization": f"Bearer {self._force_refresh_after_401()}"
                }
                params = params
                continue

            if r.status_code == 429 and attempt < 3:
                retry_after = r.headers.get("Retry-After", "1")
                try:
                    delay = max(1.0, min(float(retry_after), 30.0))
                except ValueError:
                    delay = 1.0
                print(f"Spotify pide esperar {delay:.0f} s (demasiados pedidos seguidos)…", flush=True)
                time.sleep(delay)
                continue

            if 500 <= r.status_code < 600 and attempt < 3:
                time.sleep(1.0 + attempt)
                continue

            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict):
                raise RuntimeError(f"Respuesta inesperada de Spotify para {url}")
            return data

        raise RuntimeError(f"Spotify no respondió correctamente: {url}")

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = path if path.startswith("http") else API_ROOT + path
        headers = {
            "Authorization": f"Bearer {self.token()}",
            "Content-Type": "application/json",
        }

        for attempt in range(4):
            r = self.session.post(url, json=payload, headers=headers, timeout=25)

            if r.status_code == 401:
                headers["Authorization"] = (
                    f"Bearer {self._force_refresh_after_401()}"
                )
                continue

            if r.status_code == 429 and attempt < 3:
                retry_after = r.headers.get("Retry-After", "1")
                try:
                    delay = max(1.0, min(float(retry_after), 30.0))
                except ValueError:
                    delay = 1.0
                print(f"Spotify pide esperar {delay:.0f} s (demasiados pedidos seguidos)…", flush=True)
                time.sleep(delay)
                continue

            if 500 <= r.status_code < 600 and attempt < 3:
                time.sleep(1.0 + attempt)
                continue

            r.raise_for_status()
            if not r.content:
                return {}
            data = r.json()
            if not isinstance(data, dict):
                raise RuntimeError(f"Respuesta inesperada de Spotify para {url}")
            return data

        raise RuntimeError(f"Spotify no respondió correctamente: {url}")

    def create_missing_playlist(
        self,
        source_playlist_name: str,
        tracks: list[Track],
        *,
        name: str | None = None,
    ) -> dict[str, str]:
        # Preserve the source playlist exactly, including duplicate occurrences.
        # Spotify accepts repeated URIs in a playlist, so every missing occurrence
        # is appended in the same order in which it appeared in the source.
        uris = [track.uri for track in tracks if track.uri]

        if not uris:
            raise ValueError("No hay tracks para crear la playlist de faltantes")

        playlist_name = name or (
            f"No disponibles en Soulseek - {source_playlist_name} - "
            + time.strftime("%Y-%m-%d %H%M")
        )
        created = self._post(
            "/me/playlists",
            {
                "name": playlist_name,
                "public": False,
                "description": (
                    "Temas para grabar de Spotify: no se consiguieron en Soulseek. "
                    f"Fuente: {source_playlist_name}."
                ),
            },
        )
        playlist_id = str(created.get("id") or "")
        if not playlist_id:
            raise RuntimeError("Spotify no devolvió ID para la playlist creada")

        for start in range(0, len(uris), 100):
            self._post(
                f"/playlists/{playlist_id}/items",
                {"uris": uris[start : start + 100]},
            )

        external_urls = created.get("external_urls") or {}
        url = str(
            external_urls.get("spotify")
            or f"https://open.spotify.com/playlist/{playlist_id}"
        )
        return {"id": playlist_id, "name": playlist_name, "url": url}

    def create_missing_playlists(
        self,
        source_playlist_name: str,
        tracks: list[Track],
        *,
        name: str | None = None,
        batch_size: int = DEFAULT_MISSING_PLAYLIST_BATCH_SIZE,
    ) -> list[dict[str, Any]]:
        """Create ordered private playlists small enough for bounded recordings."""
        if batch_size <= 0:
            raise ValueError("batch_size debe ser mayor que cero")
        if not tracks:
            raise ValueError("No hay tracks para crear playlists de faltantes")

        total_batches = (len(tracks) + batch_size - 1) // batch_size
        base_name = name or (
            f"No disponibles en Soulseek - {source_playlist_name} - "
            + time.strftime("%Y-%m-%d %H%M")
        )
        playlists: list[dict[str, Any]] = []
        for batch_index, start in enumerate(range(0, len(tracks), batch_size)):
            end = min(start + batch_size, len(tracks))
            batch_name = (
                base_name
                if total_batches == 1
                else f"{base_name} [{batch_index + 1:02d}/{total_batches:02d}]"
            )
            created = self.create_missing_playlist(
                source_playlist_name,
                tracks[start:end],
                name=batch_name,
            )
            playlists.append(
                {
                    **created,
                    "batch_index": batch_index,
                    "batch_number": batch_index + 1,
                    "batch_count": total_batches,
                    "start_index": start,
                    "end_index": end,
                    "track_count": end - start,
                }
            )
        return playlists

    def remove_from_library(self, uris: list[str]) -> None:
        """Quita de la biblioteca (para una playlist propia, la borra): DELETE /me/library, que reemplazó a
        «dejar de seguir una playlist» en febrero de 2026. Hasta 40 URIs por pedido."""
        url = API_ROOT + "/me/library"
        for start in range(0, len(uris), 40):
            params = {"uris": ",".join(uris[start:start + 40])}
            headers = {"Authorization": f"Bearer {self.token()}"}
            for attempt in range(4):
                r = self.session.delete(url, params=params, headers=headers, timeout=25)
                if r.status_code == 401:
                    headers["Authorization"] = f"Bearer {self._force_refresh_after_401()}"
                    continue
                if r.status_code == 429 and attempt < 3:
                    try:
                        delay = max(1.0, min(float(r.headers.get("Retry-After", "1")), 30.0))
                    except ValueError:
                        delay = 1.0
                    time.sleep(delay)
                    continue
                if 500 <= r.status_code < 600 and attempt < 3:
                    time.sleep(1.0 + attempt)
                    continue
                r.raise_for_status()
                break

    def own_playlists(self) -> list[dict[str, Any]]:
        """Las playlists de la biblioteca del usuario (GET /me/playlists)."""
        found: list[dict[str, Any]] = []
        url: str | None = "/me/playlists"
        params: dict[str, Any] | None = {"limit": 50}
        while url:
            page = self._get(url, params=params)
            params = None
            found += [item for item in page.get("items") or [] if isinstance(item, dict)]
            url = str(page["next"]) if page.get("next") else None
        return found

    def _album_details(self, album_id: str | None) -> dict[str, Any]:
        if not album_id:
            return {}
        if album_id not in self._album_cache:
            self._album_cache[album_id] = self._get(f"/albums/{album_id}")
        return self._album_cache[album_id]

    def playlist_tracks(self, playlist: str) -> tuple[str, list[Track]]:
        pid = playlist.strip()
        if "spotify:playlist:" in pid:
            pid = pid.rsplit(":", 1)[-1]
        elif "open.spotify.com" in pid:
            parts = [p for p in urllib.parse.urlparse(pid).path.split("/") if p]
            if len(parts) >= 2 and parts[0] == "playlist":
                pid = parts[1]

        if not pid or not all(ch.isalnum() for ch in pid):
            raise ValueError(f"No pude interpretar el ID de playlist: {playlist}")

        info = self._get(f"/playlists/{pid}", params={"fields": "id,name"})
        name = str(info.get("name") or pid)
        tracks: list[Track] = []
        url: str | None = f"/playlists/{pid}/items"
        params: dict[str, Any] | None = {
            "limit": 50,
            "offset": 0,
            "additional_types": "track",
        }

        while url:
            page = self._get(url, params=params)
            params = None
            for item in page.get("items") or []:
                if not isinstance(item, dict):
                    continue
                t = item.get("track") or item.get("item")
                if not isinstance(t, dict) or not t.get("id") or t.get("is_local"):
                    continue

                album = t.get("album") or {}
                album_id = album.get("id")
                album_extra = self._album_details(album_id)
                images = album.get("images") or album_extra.get("images") or []
                cover = (
                    images[0].get("url")
                    if images and isinstance(images[0], dict)
                    else None
                )
                ext = t.get("external_ids") or {}
                copyrights = [
                    str(x.get("text"))
                    for x in album_extra.get("copyrights") or []
                    if isinstance(x, dict) and x.get("text")
                ]

                tracks.append(
                    Track(
                        spotify_id=str(t["id"]),
                        uri=str(t.get("uri") or f"spotify:track:{t['id']}"),
                        url=str(
                            (t.get("external_urls") or {}).get("spotify")
                            or f"https://open.spotify.com/track/{t['id']}"
                        ),
                        title=str(t.get("name") or ""),
                        artists=[
                            str(a["name"])
                            for a in t.get("artists") or []
                            if a.get("name")
                        ],
                        album=str(album.get("name") or ""),
                        album_id=str(album_id) if album_id else None,
                        album_artists=[
                            str(a["name"])
                            for a in album.get("artists") or []
                            if a.get("name")
                        ],
                        release_date=album.get("release_date"),
                        track_number=t.get("track_number"),
                        total_tracks=album.get("total_tracks"),
                        disc_number=t.get("disc_number"),
                        duration_ms=int(t.get("duration_ms") or 0),
                        isrc=str(ext.get("isrc")) if ext.get("isrc") else None,
                        cover_url=str(cover) if cover else None,
                        explicit=bool(t.get("explicit")),
                        label=(
                            str(album_extra.get("label"))
                            if album_extra.get("label")
                            else None
                        ),
                        copyrights=copyrights,
                        playlist_id=pid,
                        playlist_name=name,
                        added_at=item.get("added_at"),
                    )
                )
            url = str(page.get("next")) if page.get("next") else None
            # Un pedido por álbum: en playlists largas tarda, y así se ve que avanza.
            print(f"Leyendo la playlist en Spotify: {len(tracks)}/{page.get('total') or '?'} temas…", flush=True)

        return name, tracks


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

_VERSION_RE = [
    ("extended", re.compile(r"\bextended(?:\s+(?:mix|version))?\b", re.I)),
    ("radio", re.compile(r"\bradio\s+(?:edit|mix|version)\b", re.I)),
    ("club", re.compile(r"\bclub\s+(?:mix|version)\b", re.I)),
    ("original", re.compile(r"\boriginal\s+mix\b", re.I)),
    ("remix", re.compile(r"\bremix\b", re.I)),
    ("edit", re.compile(r"\bedit\b", re.I)),
    ("live", re.compile(r"\blive\b", re.I)),
    ("acoustic", re.compile(r"\bacoustic\b", re.I)),
    ("instrumental", re.compile(r"\binstrumental\b", re.I)),
    ("bootleg", re.compile(r"\bbootleg\b", re.I)),
    ("vip", re.compile(r"\bvip\b", re.I)),
    ("cover", re.compile(r"\bcover\b", re.I)),
    ("dub", re.compile(r"\bdub\b", re.I)),
    ("sped_up", re.compile(r"\bsped\s*up\b", re.I)),
    ("slowed", re.compile(r"\bslowed\b", re.I)),
]

_CONFLICT = {
    "radio",
    "club",
    "remix",
    "edit",
    "live",
    "acoustic",
    "instrumental",
    "bootleg",
    "vip",
    "cover",
    "dub",
    "sped_up",
    "slowed",
}

# Si Spotify explicita una variante de este grupo, el candidato debe contenerla.
_REQUIRED_TARGET_VARIANTS = _CONFLICT | {"extended"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.casefold().replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def variants(s: str) -> set[str]:
    return {name for name, pat in _VERSION_RE if pat.search(s or "")}


def strip_variants(s: str) -> str:
    for _, pat in _VERSION_RE:
        s = pat.sub(" ", s)
    return norm(s)


def strip_extended_only(s: str) -> str:
    out = s
    for name, pat in _VERSION_RE:
        if name == "extended":
            out = pat.sub(" ", out)
    return norm(out)


def sim(a: str, b: str) -> float:
    a = norm(a)
    b = norm(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # No premiar automáticamente substrings. "Sun" y "Sunrise" no son
    # identidad suficiente para esta tarea.
    return SequenceMatcher(None, a, b).ratio()


def _candidate_stem(f: LocalFile) -> str:
    stem = f.path.stem
    return re.sub(r"^\s*\d{1,3}[\s._-]+", "", stem)


def _candidate_title(track: Track, f: LocalFile) -> str:
    if f.tag_title:
        return f.tag_title
    stem = _candidate_stem(f)
    normalized = norm(stem)
    for artist in sorted(track.artists, key=len, reverse=True):
        a = norm(artist)
        if a:
            normalized = re.sub(rf"\b{re.escape(a)}\b", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


_ARTIST_SEPARATORS = re.compile(
    r"\s*(?:[,;/&+]|\b(?:and|e|y|x|feat|ft|featuring|with|vs)\b\.?)\s*", re.IGNORECASE
)


def artist_names(value: str) -> list[str]:
    """El nombre completo y sus partes: «Toquinho e Vinícius» -> [toquinho e vinicius, toquinho, vinicius].
    Conservar el nombre completo hace que separar nunca baje la similitud de un dúo con nombre propio."""
    names = [norm(value), *(norm(part) for part in _ARTIST_SEPARATORS.split(value or ""))]
    return list(dict.fromkeys(name for name in names if name))


def _artist_similarity(track: Track, f: LocalFile, *, split: bool = True) -> float:
    target = [norm(a) for a in track.artists if norm(a)]
    if split:
        local = [name for a in (f.tag_artists or []) for name in artist_names(a)]
    else:
        local = [norm(a) for a in (f.tag_artists or []) if norm(a)]
    if local:
        return max((sim(a, b) for a in target for b in local), default=0.0)

    filename = norm(f.path.name)
    exact_presence = [
        1.0
        for a in target
        if a and re.search(rf"\b{re.escape(a)}\b", filename)
    ]
    return max(exact_presence, default=0.0)


def score_candidate(
    track: Track,
    f: LocalFile,
    prefer_extended: bool = True,
) -> tuple[bool, float, bool, list[str]]:
    """Devuelve (aceptado, score, es_extended, razones)."""
    reasons: list[str] = []

    fmt = f.format.casefold()
    if fmt == "mp3":
        if (f.bitrate_kbps or 0) < 315:
            return False, 0.0, False, [
                f"MP3 < 320 kbps ({f.bitrate_kbps or 0:.1f})"
            ]
        quality = 3.0
    elif fmt == "flac":
        quality = 2.0
    elif fmt == "wav":
        quality = 1.0
    else:
        return False, 0.0, False, [f"formato no soportado: {fmt}"]

    title_raw = _candidate_title(track, f)
    base_t = strip_variants(track.title)
    base_c = strip_variants(title_raw)
    title_s = sim(track.title, title_raw)
    base_s = sim(base_t, base_c)
    art_s = _artist_similarity(track, f)
    # El ISRC identifica la grabación: si coincide, un artista escrito de otra forma en la etiqueta
    # («Toquinho e Vinícius» frente a Toquinho, Chico Buarque) no la descarta.
    isrc_exact = bool(track.isrc and f.isrc and norm(track.isrc) == norm(f.isrc))

    if base_s < 0.88:
        return False, 0.0, False, [f"título base mismatch ({base_s:.3f})"]
    if art_s < 0.88 and not isrc_exact:
        return False, 0.0, False, [f"artista mismatch ({art_s:.3f})"]
    if art_s < 0.88:
        reasons.append(f"artista distinto en la etiqueta ({art_s:.3f}), aceptado por ISRC exacto")

    tv = variants(track.title)
    cv = variants(title_raw + " " + _candidate_stem(f))

    missing_required = sorted((tv & _REQUIRED_TARGET_VARIANTS) - cv)
    if missing_required:
        return False, 0.0, False, [
            "faltan variantes exigidas por Spotify: " + ", ".join(missing_required)
        ]

    # Extended es la única variante adicional permitida porque el objetivo del
    # script es preferirla cuando Spotify contiene la versión corta.
    conflicts = sorted((cv - tv) & _CONFLICT)
    if conflicts:
        return False, 0.0, False, [
            "versión conflictiva: " + ", ".join(conflicts)
        ]

    # Para un remix/club/etc. no basta con compartir la palabra "Remix".
    # Comparamos la identidad completa quitando únicamente "Extended" para
    # permitir, por ejemplo, "DJ Foo Remix" -> "DJ Foo Extended Remix".
    identity_variants = tv & _CONFLICT
    if identity_variants:
        version_identity = sim(
            strip_extended_only(track.title),
            strip_extended_only(title_raw),
        )
        if version_identity < 0.93:
            return False, 0.0, False, [
                f"identidad de versión/remixer mismatch ({version_identity:.3f})"
            ]
        reasons.append(f"version_identity={version_identity:.3f}")

    explicit_extended = "extended" in cv and "extended" not in tv
    inferred_extended = False
    length = f.length_s
    dur = track.duration_s
    dur_s = 0.0

    if length is not None and dur > 0:
        delta = length - dur
        ratio = length / dur

        if "extended" in tv:
            # Spotify ya apunta a una Extended. Debe tener duración cercana.
            if abs(delta) > 15:
                return False, 0.0, False, [
                    f"duración mismatch para Extended de Spotify ({delta:+.1f}s)"
                ]
            dur_s = max(0.0, 1.0 - abs(delta) / 15.0)
        elif explicit_extended:
            if delta < 12 or ratio > 1.9:
                return False, 0.0, False, [
                    f"extended con duración rara ({delta:+.1f}s, {ratio:.2f}x)"
                ]
            dur_s = 1.0
        elif abs(delta) <= DURATION_TOLERANCE_S:
            dur_s = max(0.0, 1.0 - abs(delta) / DURATION_TOLERANCE_S)
        elif (
            prefer_extended
            and delta >= 20
            and ratio <= 1.9
            and base_s >= 0.985
            # Sin separar «A & B»: una Extended inferida es la regla más arriesgada y no se amplía
            # (con la separación entraba otra versión, p. ej. «Tomara» con Maria Creuza, +54 s).
            and _artist_similarity(track, f, split=False) >= 0.985
            and not (cv - {"original"})
        ):
            inferred_extended = True
            dur_s = 0.9
            reasons.append(f"extended inferido (+{delta:.1f}s)")
        else:
            return False, 0.0, False, [f"duración mismatch ({delta:+.1f}s)"]
    else:
        reasons.append("sin duración local: confianza reducida")

    is_ext = explicit_extended or inferred_extended or "extended" in tv

    isrc_match = False
    if track.isrc and f.isrc:
        isrc_match = norm(track.isrc) == norm(f.isrc)
        if not isrc_match and not is_ext:
            return False, 0.0, False, [
                f"ISRC distinto: local={f.isrc} Spotify={track.isrc}"
            ]
        if isrc_match:
            reasons.append("ISRC exacto")
        else:
            reasons.append(
                "ISRC distinto permitido sólo porque la Extended puede tener ISRC propio"
            )

    score = (
        45.0 * base_s
        + 25.0 * art_s
        + 15.0 * title_s
        + 10.0 * dur_s
        + quality
    )

    if isrc_match and not is_ext:
        score += 40.0
    if is_ext and prefer_extended and "extended" not in tv:
        score += 20.0

    reasons += [
        f"base={base_s:.3f}",
        f"art={art_s:.3f}",
        f"title={title_s:.3f}",
        f"score={score:.3f}",
    ]
    return True, score, is_ext, reasons


# ---------------------------------------------------------------------------
# Inspección local
# ---------------------------------------------------------------------------

def _frame_text(value: Any) -> str | None:
    if value is None:
        return None
    text = getattr(value, "text", value)
    if isinstance(text, (list, tuple)):
        return str(text[0]).strip() if text else None
    if isinstance(text, bytes):
        return text.decode("utf-8", errors="replace").strip()
    rendered = str(text).strip()
    return rendered or None


def inspect(path: Path) -> LocalFile | None:
    try:
        audio = mutagen.File(path)
        if audio is None or not getattr(audio, "info", None):
            return None
        if not isinstance(audio, {".mp3": MP3, ".flac": FLAC, ".wav": WAVE}.get(path.suffix.lower(), ())):
            return None
        info = audio.info
        br = getattr(info, "bitrate", None)
        tags = getattr(audio, "tags", None)

        def txt(*keys):
            if not tags:
                return None
            for k in keys:
                try:
                    v = tags.get(k)
                except Exception:
                    v = None
                rendered = _frame_text(v)
                if rendered:
                    return rendered
            return None

        artist_raw = txt("TPE1", "artist", "ARTIST")
        artists = None
        if artist_raw:
            artists = [
                part.strip()
                for part in re.split(r"\s*(?:;|\u001f)\s*", artist_raw)
                if part.strip()
            ]

        return LocalFile(
            path=path,
            length_s=float(getattr(info, "length", 0) or 0) or None,
            bitrate_kbps=(float(br) / 1000.0) if br else None,
            format=path.suffix.lstrip(".").lower(),
            tag_title=txt("TIT2", "title", "TITLE"),
            tag_artists=artists,
            isrc=txt("TSRC", "isrc", "ISRC"),
            sample_rate=getattr(info, "sample_rate", None),
            bit_depth=getattr(info, "bits_per_sample", None),
            channels=getattr(info, "channels", None),
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Soulseek
# ---------------------------------------------------------------------------
# Esta sección conserva deliberadamente el mecanismo del script original.

def find_sockseek() -> str:
    for name in ("sockseek", "sldl"):
        p = shutil.which(name)
        if p:
            return p
    raise RuntimeError(
        "No se encontró 'sockseek' ni 'sldl' en el PATH.\n"
        "Descárgalo de https://github.com/fiso64/sockseek/releases"
    )


def _write_sockseek_safe_config(destination: Path, binary: str) -> None:
    """Keep global login only; never inherit hooks, album mode or fallback commands."""
    locations = [Path.home() / ".config/sockseek/sockseek.conf"]
    if os.environ.get("APPDATA"):
        locations.append(Path(os.environ["APPDATA"]) / "sockseek/sockseek.conf")
    if os.environ.get("XDG_CONFIG_HOME"):
        locations.append(Path(os.environ["XDG_CONFIG_HOME"]) / "sockseek/sockseek.conf")
    locations.append(Path(binary).resolve().parent / "sockseek.conf")
    source = next((p for p in locations if p.is_file()), None)
    if source is None:
        raise RuntimeError("falta sockseek.conf con username/password globales")
    login: dict[str, str] = {}
    for line in source.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            break  # Profiles cannot enable commands or replace credentials.
        key, separator, value = stripped.partition("=")
        if separator and key.strip().lower() in {"username", "password"}:
            login[key.strip().lower()] = value.strip()
    if not all(login.get(key) for key in ("username", "password")):
        raise RuntimeError("sockseek.conf debe tener username/password en la sección global")
    destination.write_text("\n".join(f"{key} = {value}" for key, value in login.items()) + "\n", encoding="utf-8")
    destination.chmod(0o600)


def _sockseek_query(track: Track) -> str:
    query = f"{track.primary_artist} - {track.title}"
    if track.duration_ms:
        query += f", length={max(1, track.duration_ms // 1000)}"
    return query


def _availability_candidate_summary(item: dict[str, Any]) -> dict[str, Any]:
    user = item.get("User") if isinstance(item.get("User"), dict) else {}
    file_info = item.get("File") if isinstance(item.get("File"), dict) else {}
    filename = str(file_info.get("Filename") or "")
    suffix = Path(filename.replace("\\", "/")).suffix.casefold()
    return {
        "username": user.get("Username"),
        "upload_speed_mib_s": user.get("UploadSpeed"),
        "has_free_upload_slot": user.get("HasFreeUploadSlot"),
        "filename": filename,
        "format": suffix.lstrip("."),
        "length_s": file_info.get("Length"),
        "bitrate_kbps": file_info.get("Bitrate"),
        "sample_rate": file_info.get("SampleRate"),
        "bit_depth": file_info.get("BitDepth"),
        "size_bytes": file_info.get("Size"),
    }


def _title_from_filename(name: str, track: Track) -> str:
    """What the title tag of a Soulseek file would say, from its name: without the track number, the
    leading «Artist - » or the trailing « - Artist 124» of DJ pools, and with underscores as spaces. The
    remixer stays (it is often also a Spotify artist: «Sun - Rafael Extended Remix»)."""
    stem = re.sub(r"^\s*\d{1,3}[\s._-]+", "", Path(name).stem).replace("_", " ")
    if "[" not in track.title:  # «[www.dj-promo.org]», «[Material]», «[soulseek]»: no son parte del título
        stem = re.sub(r"\s*\[[^\]]*\]\s*", " ", stem).strip()
    parts = [p for p in re.split(r"\s+-\s+", stem) if p.strip()]
    if len(parts) == 1:
        parts = [p for p in re.split(r"(?<=\S)-(?=\S)", stem) if p.strip()]  # scene: artista-titulo (mix)
    # «Artista - Disco - 06 - Título»: el título es lo que sigue al número de pista (que puede ser «365»).
    numbered = [i for i, p in enumerate(parts[:-1]) if re.fullmatch(r"\d{1,3}", p.strip())]
    if numbered:
        parts = parts[numbered[-1] + 1:]

    def credits_only(part: str) -> bool:
        return not variants(part) and any(
            norm(a) and re.search(rf"\b{re.escape(norm(a))}\b", norm(part)) for a in track.artists
        )

    if len(parts) > 1 and credits_only(parts[0]):
        parts = parts[1:]
    if len(parts) > 1 and credits_only(parts[-1]):
        parts = parts[:-1]
    title = " - ".join(parts).strip()
    scene_group = re.search(r"-([a-z0-9]{2,8})$", title, re.I)  # «07-esh_-_peter_gunn-zzzz»
    if " " not in Path(name).stem and scene_group and norm(scene_group.group(1)) not in norm(track.title).split():
        title = title[: scene_group.start()].strip()
    # BPM y tonalidad de los pools («… Extended Mix 12A 124»), salvo que sean parte del título de Spotify.
    trailing = re.search(r"(?:\s+(?:\d{1,2}[AB]|\d{2,3}))+\s*$", title)
    if trailing and title[: trailing.start()].strip() and not norm(track.title).endswith(norm(trailing.group())):
        title = title[: trailing.start()].strip()
    return title


def _availability_title_error(track: Track, candidate: dict[str, Any]) -> str | None:
    """The same verification as after the download, judged by the file name and what the peer announces,
    so that another song or another version is not taken as available. Quality is checked elsewhere; the
    artist is in the path (--strict-artist)."""
    name = Path(str(candidate.get("filename") or "").replace("\\", "/")).name
    local = LocalFile(
        path=Path(name),
        length_s=candidate.get("length_s"),
        bitrate_kbps=None,
        format="flac",
        tag_title=_title_from_filename(name, track),
        tag_artists=list(track.artists),
    )
    accepted, _, _, reasons = score_candidate(track, local)
    return None if accepted else (reasons[0] if reasons else "no coincide con el tema")


def _classify_availability_raw(raw: Any, track: Track | None = None, length_tol: int = 60) -> AvailabilityCheck:
    """Classify one Sockseek JSON result array without touching the network."""
    if not isinstance(raw, list):
        return AvailabilityCheck(
            status="error",
            result_count=0,
            candidates=[],
            error="sockseek --print json-all no devolvió una lista JSON para el tema",
        )

    summaries = [
        _availability_candidate_summary(item)
        for item in raw
        if isinstance(item, dict)
    ]

    usable: list[dict[str, Any]] = []
    quality_unknown = False
    for candidate in summaries:
        error = None
        if not _safe_audio_filename(candidate["filename"]):
            error = "nombre/formato no permitido"
        elif not isinstance(candidate["username"], str) or not candidate["username"]:
            error = "usuario ausente o inválido"
        else:
            error = _audio_size_error(candidate)
        if not error and track is not None:
            length = float(candidate["length_s"])
            if track.duration_s <= 0:
                error = "duración Spotify ausente o inválida"
            elif length < max(1, track.duration_s - max(0, length_tol)) or length > max(
                track.duration_s + max(0, length_tol), track.duration_s * 1.9
            ):
                error = "duración anunciada fuera del rango del tema/Extended"
            else:
                error = _availability_title_error(track, candidate)
        if error:
            candidate["rejection_reason"] = error
            continue
        fmt = str(candidate.get("format") or "").casefold()
        bitrate = candidate.get("bitrate_kbps")
        if fmt in {"flac", "wav"}:
            usable.append(candidate)
        elif fmt == "mp3":
            if bitrate is None:
                # Some Soulseek clients do not advertise bitrate. Do not create a false
                # negative; the downloaded file will still be verified locally.
                quality_unknown = True
                usable.append(candidate)
            else:
                try:
                    if float(bitrate) >= 315.0:
                        usable.append(candidate)
                except (TypeError, ValueError):
                    quality_unknown = True
                    usable.append(candidate)

    if usable:
        status = "available_quality_unknown" if quality_unknown else "available"
        return AvailabilityCheck(
            status=status,
            result_count=len(usable),
            candidates=usable[:10],
            quality_unknown=quality_unknown,
        )

    return AvailabilityCheck(
        status="missing",
        result_count=0,
        candidates=summaries[:10],
    )


def _decode_concatenated_json_values(stdout: str) -> list[Any]:
    """Decode consecutive compact JSON values emitted by Sockseek JobList printing.

    Sockseek prints one JSON value per child job for multi-item inputs.  The values
    are compact JSON arrays separated only by whitespace/newlines, so parsing the
    complete stdout with json.loads() is not sufficient.
    """
    text = (stdout or "").lstrip("\ufeff")
    decoder = json.JSONDecoder()
    values: list[Any] = []
    pos = 0
    n = len(text)
    while pos < n:
        while pos < n and text[pos].isspace():
            pos += 1
        if pos >= n:
            break
        value, end = decoder.raw_decode(text, pos)
        values.append(value)
        pos = end
    return values


def _write_sockseek_preflight_csv(path: Path, tracks: list[Track]) -> None:
    """Write a structured multi-song input preserving Spotify playlist order."""
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Artist", "Title", "Length"])
        writer.writeheader()
        for track in tracks:
            writer.writerow(
                {
                    "Artist": track.primary_artist,
                    "Title": track.title,
                    "Length": max(1, int(round(track.duration_s))) if track.duration_ms else "",
                }
            )


def _search_outcome(line: bytes) -> bool | None:
    """--progress-json line of a finished search: True if it found files, False if not; None otherwise."""
    if not line.startswith(b"{") or b"track_state" not in line[:64]:  # los resultados son listas largas
        return None
    try:
        event = json.loads(line)
    except ValueError:
        return None
    data = event.get("data") if isinstance(event, dict) and event.get("type") == "track_state" else None
    if not isinstance(data, dict) or data.get("lifecycleState") != "Terminal":
        return None
    return data.get("terminalOutcome") == "Succeeded"


def check_soulseek_availability_batch(
    tracks: list[Track],
    *,
    length_tol: int = 60,
    timeout: float = 600.0,
    on_progress: Callable[[int, int], None] | None = None,
    log_file: Path | None = None,
) -> list[AvailabilityCheck]:
    """Preflight all tracks in one Sockseek process and one Soulseek session.

    No audio is downloaded because --print json-all sets Sockseek to print-only
    mode.  The CSV row order is the Spotify playlist order, and Sockseek's JobList
    renderer emits one compact JSON array per child job in that same order, all at the end.
    --progress-json adds one event line per search as it happens: `on_progress(searched,
    found)` follows them, and the events are left out of the results.
    """
    if not tracks:
        return []

    bin_ = find_sockseek()
    with tempfile.TemporaryDirectory(prefix="sockseek_preflight_") as tmp:
        batch_csv = Path(tmp) / "spotify_preflight.csv"
        _write_sockseek_preflight_csv(batch_csv, tracks)
        session_args = _sockseek_session_args(Path(tmp), bin_)
        # El rango de duración real (hasta 1,9× para una Extended) lo aplica _classify_availability_raw; con
        # la tolerancia de antes (60 s) sockseek ya descartaba la Original Mix larga de un radio edit.
        sockseek_tol = max([length_tol] + [math.ceil(0.9 * t.duration_s) for t in tracks if t.duration_s > 0])

        cmd = [
            bin_,
            str(batch_csv),
            *session_args,
            # El título lo decide _classify_availability_raw con las reglas de la verificación. --strict-title
            # exigía el «Título - Versión» de Spotify al pie de la letra y perdía «Título (Versión)»: en una
            # playlist de 152 temas, 29 de los 46 «no disponibles» estaban (2026-11-02).
            "--pref-strict-title",
            "--strict-artist",
            f"--length-tol={sockseek_tol}",
            f"--pref-length-tol={length_tol}",
            # Required in preflight because the finalizer only supports these formats.
            "--format=flac,wav,mp3",
            "--min-bitrate=315",
            "--pref-format=flac,wav,mp3",
            "--pref-min-bitrate=320",
            "--print",
            "json-all",
            "--progress-json",
        ]
        if log_file is not None:
            cmd.append(f"--log-file={log_file}")

        stdout_lines: list[bytes] = []
        stderr_chunks: list[bytes] = []
        searched = found = 0

        def read_stdout(stream: Any) -> None:
            nonlocal searched, found
            for line in iter(stream.readline, b""):
                stdout_lines.append(line)
                outcome = _search_outcome(line)
                if outcome is None:
                    continue
                searched, found = min(searched + 1, len(tracks)), found + int(outcome)
                if on_progress is not None:
                    on_progress(searched, found)

        timed_out = False
        try:
            process = subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            readers = [
                threading.Thread(target=read_stdout, args=(process.stdout,), daemon=True),
                threading.Thread(target=lambda: stderr_chunks.append(process.stderr.read()), daemon=True),
            ]
            for reader in readers:
                reader.start()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                for reader in readers:
                    reader.join(timeout=10)
        except FileNotFoundError:
            return [
                AvailabilityCheck(
                    status="error",
                    result_count=0,
                    candidates=[],
                    error="sockseek no encontrado",
                )
                for _ in tracks
            ]
        if timed_out:
            message = f"timeout del preflight Soulseek batch ({timeout:.0f}s)"
            return [
                AvailabilityCheck(
                    status="error",
                    result_count=0,
                    candidates=[],
                    error=message,
                )
                for _ in tracks
            ]

        stdout = b"".join(stdout_lines).decode("utf-8", errors="replace").lstrip("\ufeff").strip()
        stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace")
        try:
            raw_values = [
                value for value in _decode_concatenated_json_values(stdout)
                if not (isinstance(value, dict) and "type" in value)  # eventos de --progress-json
            ]
        except json.JSONDecodeError as exc:
            detail = (stderr or stdout or "sin salida").strip()
            message = f"salida JSON batch inválida de sockseek: {exc}; {detail[:500]}"
            return [
                AvailabilityCheck(
                    status="error",
                    result_count=0,
                    candidates=[],
                    error=message,
                )
                for _ in tracks
            ]

        if len(raw_values) != len(tracks):
            detail = stderr.strip()
            message = (
                "sockseek devolvió un número inesperado de resultados de preflight: "
                f"{len(raw_values)} para {len(tracks)} temas"
            )
            if detail:
                message += f"; {detail[:500]}"
            return [
                AvailabilityCheck(
                    status="error",
                    result_count=0,
                    candidates=[],
                    error=message,
                )
                for _ in tracks
            ]

        return [
            _classify_availability_raw(raw, track, length_tol)
            for raw, track in zip(raw_values, tracks)
        ]


def check_soulseek_availability(
    track: Track,
    *,
    length_tol: int = 60,
    timeout: float = 90.0,
) -> AvailabilityCheck:
    """Compatibility helper for a one-track preflight."""
    return check_soulseek_availability_batch(
        [track], length_tol=length_tol, timeout=timeout
    )[0]

def write_missing_list(path: Path, missing: list[tuple[int, Track]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Temas no disponibles en Soulseek durante el preflight",
        "=" * 54,
        "",
    ]
    if not missing:
        lines.append("Ninguno.")
    else:
        for position, track in missing:
            lines.append(
                f"{position:4d}. {track.primary_artist} - {track.title} | {track.url}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _duration_hms(duration_ms: int) -> str:
    total_seconds = max(0, int(round(duration_ms / 1000.0)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def write_missing_json(
    path: Path,
    *,
    source_playlist_name: str,
    source_playlist: str,
    missing: list[tuple[int, Track]],
    generated_playlist: dict[str, Any] | None = None,
    generated_playlists: list[dict[str, Any]] | None = None,
    playlist_batch_size: int = DEFAULT_MISSING_PLAYLIST_BATCH_SIZE,
) -> None:
    """Write missing tracks in original playlist order with full Spotify metadata."""
    tracks = [recording_entry(position, track) for position, track in missing]

    playlists = generated_playlists
    if playlists is None and generated_playlist is not None:
        playlists = [generated_playlist]
    legacy_playlist = playlists[0] if playlists and len(playlists) == 1 else None
    payload = {
        "source_playlist": {
            "name": source_playlist_name,
            "source": source_playlist,
        },
        "generated_missing_playlist": legacy_playlist,
        "generated_missing_playlists": playlists,
        "playlist_batch_size": playlist_batch_size,
        "missing_count": len(tracks),
        "order": "source_playlist_order",
        "tracks": tracks,
    }
    write_report(path, payload)


def rank_candidates(
    track: Track,
    candidates: list[dict[str, Any]],
    prefer_extended: bool = True,
) -> list[dict[str, Any]]:
    """Orden en que se prueban los candidatos del preflight.

    1. Extended explícita en el nombre, si se prefiere y Spotify no apunta ya a una Extended.
    2. La duración de Spotify (± DURATION_TOLERANCE_S): la misma versión.
    3. El resto (puede pasar como Extended inferida o ser otra versión).
    Dentro de cada grupo, primero los que tienen el título en el nombre del archivo (no solo en la
    carpeta: «Wave (1967)/04 - Batidinha»), después los de un usuario con un lugar libre para subir (los
    demás nos ponen en cola: en el log del 2026-11-02, 33 de 52 transferencias fallidas fueron esperas sin
    avance), los de una carpeta con el nombre del álbum de Spotify, los de duración más cercana (salvo en
    las Extended) y los de usuarios más rápidos; si empatan, el orden de sockseek.
    """
    duration = track.duration_s
    album = norm(track.album)
    title = norm(strip_variants(track.title))
    spotify_extended = "extended" in variants(track.title)

    def key(item: tuple[int, dict[str, Any]]) -> tuple[int, int, int, int, float, float, int]:
        index, candidate = item
        raw = str(candidate.get("filename") or "")
        name = norm(raw)
        file_name = norm(re.split(r"[\\/]", raw)[-1])
        length = candidate.get("length_s")
        delta = abs(float(length) - duration) if isinstance(length, (int, float)) and duration > 0 else None
        if prefer_extended and not spotify_extended and re.search(r"\bextended\b", name):
            tier = 0
        elif delta is not None and delta <= DURATION_TOLERANCE_S:
            tier = 1
        else:
            tier = 2
        titled = bool(title) and re.search(rf"\b{re.escape(title)}\b", file_name) is not None
        same_album = bool(album) and re.search(rf"\b{re.escape(album)}\b", name) is not None
        closeness = 0.0 if tier == 0 or delta is None else delta
        busy = 0 if candidate.get("has_free_upload_slot") else 1
        speed = candidate.get("upload_speed_mib_s")
        speed = float(speed) if isinstance(speed, (int, float)) and not isinstance(speed, bool) else 0.0
        return (tier, 0 if titled else 1, busy, 0 if same_album else 1, closeness, -speed, index)

    return [candidate for _, candidate in sorted(enumerate(candidates), key=key)]


def download_with_sockseek(
    track: Track,
    out_dir: Path,
    length_tol: int = 60,
    *,
    availability: AvailabilityCheck | None = None,
    verify: Callable[[Path], bool] | None = None,
    max_transfers: int = MAX_CANDIDATE_TRANSFERS,
    prefer_extended: bool = True,
    max_attempts: int = MAX_TRANSFER_ATTEMPTS,
    budget_s: float = TRACK_DOWNLOAD_BUDGET_S,
    avoid: set[tuple[str, str]] | None = None,
    outcome: dict[str, int] | None = None,
) -> list[Path]:
    """Download only preflight-approved single files, scan before exposing to parsers.

    With `verify`, each file that passes the safety gates is checked as the right track; a rejected
    file is deleted and the next candidate is tried, up to `max_transfers` files that arrived and
    `max_attempts` transfers in all (a transfer that fails brings nothing: it does not use up the
    verification attempts), within `budget_s`. `avoid` holds (user, file) pairs that already failed and
    gets the new failures; `outcome` gets the counts ("attempts", "arrived").
    """
    bin_ = find_sockseek()
    out_dir.mkdir(parents=True, exist_ok=True)
    _check_antivirus_ready()  # Require a working scanner before any network transfer.
    check = availability or check_soulseek_availability(track, length_tol=length_tol)
    if not check.downloadable:
        if check.error:
            raise RuntimeError(check.error)
        return []
    arrived = attempts = 0
    started = time.monotonic()
    counts = outcome if outcome is not None else {}
    counts.update(attempts=0, arrived=0)
    for candidate in rank_candidates(track, check.candidates, prefer_extended):
        if arrived >= max_transfers or attempts >= max_attempts:
            break
        if time.monotonic() - started >= budget_s:
            say(f"    ! se agotó el tiempo para este tema ({budget_s / 60:.0f} min)")
            break
        if not _safe_audio_filename(str(candidate.get("filename") or "")) or _audio_size_error(candidate):
            continue
        username = candidate.get("username")
        if not isinstance(username, str) or not username:
            continue
        remote_path = candidate["filename"].replace("\\", "/")
        if avoid is not None and (username, remote_path) in avoid:
            continue
        link = "slsk://" + urllib.parse.quote(username, safe="") + "/" + urllib.parse.quote(remote_path, safe="/")
        attempts += 1
        counts["attempts"] = attempts
        destination: Path | None = None
        # Each attempt is isolated. Rejected and incomplete payloads are removed.
        with tempfile.TemporaryDirectory(prefix="incoming_", dir=out_dir) as tmp:
            incoming = Path(tmp)
            safe_config = incoming / "sockseek.conf"
            session_args = _sockseek_session_args(incoming, bin_)
            cmd = [
                bin_, link, "--input-type=soulseek", "--song",
                *session_args,
                "--format=flac,wav,mp3", "--min-bitrate=315",
                "--yt-dlp=false", "--album=false",
                "--upgrade-to-album=false", "--aggregate=false",
                "--album-art=default", "--album-art-only=false",
                "--name-format={slsk-filename}", "--no-write-index",
                "--write-playlist=false", "--no-skip-existing",
                f"--output-dir={incoming}",
            ]
            try:
                code = _run_sockseek_transfer(cmd, incoming, int(candidate["size_bytes"]), safe_config)
                if code != 0:
                    if avoid is not None:
                        avoid.add((username, remote_path))
                    continue
                files = [p for p in incoming.rglob("*") if p.is_file() and p != safe_config]
                if len(files) != 1:
                    say(f"    ! sockseek dejó {len(files)} archivos en vez de 1; candidato descartado")
                    continue
                path = files[0]
                if not path.resolve().is_relative_to(incoming.resolve()):
                    raise RuntimeError("archivo fuera del directorio de recepción")
                arrived += 1
                counts["arrived"] = arrived
                _validate_download(path, candidate)
                destination = Path(tempfile.mkdtemp(prefix="verified_", dir=out_dir)) / path.name
                shutil.move(str(path), destination)
            except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
                say(f"    ! candidato bloqueado: {exc}")
                if avoid is not None:
                    avoid.add((username, remote_path))
                if _DAEMON is not None:  # que el daemon no siga bajando lo que se dejó
                    _DAEMON.cancel_matching(Path(remote_path).stem)
        if destination is None:
            continue
        if verify is None or verify(destination):
            return [destination]
        say(f"    ✗ no pasó la verificación: {destination.name}; pruebo otro candidato")
        shutil.rmtree(destination.parent, ignore_errors=True)
    return []


# ---------------------------------------------------------------------------
# Finalización MP3 + metadata + cover
# ---------------------------------------------------------------------------

def safe_name(s: str) -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", s)
    s = re.sub(r"\s+", " ", s).strip().rstrip(".")
    return s or "Unknown"


def track_from_report(report: dict[str, Any]) -> Track:
    """Rebuild a Track from the Spotify object stored in soulseek_missing.json."""
    if not isinstance(report, dict):
        raise TypeError("la metadata Spotify de la pista no es un objeto JSON")
    allowed = {field.name for field in fields(Track)}
    values = {key: value for key, value in report.items() if key in allowed}
    try:
        return Track(**values)
    except TypeError as exc:
        raise ValueError(f"metadata Spotify incompleta en el manifest: {exc}") from exc


def spotify_output_filename(track: Track, extended: bool = False) -> str:
    """Return the same canonical filename used for finalized Soulseek files."""
    title = safe_name(track.title)
    if extended and "extended" not in track.title.casefold():
        title += " [Extended]"
    return f"{safe_name(track.primary_artist)} - {title}.mp3"


def spotify_output_filenames(tracks: list[Track]) -> list[str]:
    """Plan canonical filenames, including Soulseek-style Spotify ID collisions."""
    owners: dict[str, str] = {}
    filenames: list[str] = []
    for track in tracks:
        filename = spotify_output_filename(track)
        key = filename.casefold()
        owner = owners.get(key)
        if owner is None:
            owners[key] = track.spotify_id
        elif owner != track.spotify_id:
            stem = Path(filename).stem
            filename = f"{stem} [{track.spotify_id}].mp3"
            key = filename.casefold()
            alternate_owner = owners.get(key)
            if alternate_owner not in (None, track.spotify_id):
                raise RuntimeError(f"colisión de salida no resoluble: {filename}")
            owners[key] = track.spotify_id
        filenames.append(filename)
    return filenames


def download_cover(
    url: str | None,
    *,
    retries: int = 3,
    max_bytes: int = 15 * 1024 * 1024,
) -> tuple[bytes, str] | None:
    if not url:
        return None
    if urllib.parse.urlparse(url).scheme != "https":
        raise RuntimeError("la carátula debe descargarse por HTTPS")

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "traktor-spotify-import/2.0"},
            )
            with urllib.request.urlopen(req, timeout=15) as r:
                data = r.read(max_bytes + 1)
                if len(data) > max_bytes:
                    raise RuntimeError("cover demasiado grande")
                ctype = (
                    (r.headers.get("Content-Type") or "image/jpeg")
                    .split(";")[0]
                    .strip()
                )
                if ctype not in ("image/jpeg", "image/png", "image/webp"):
                    raise RuntimeError(f"Content-Type de cover inesperado: {ctype}")
                if not data:
                    raise RuntimeError("cover vacío")
                return _validated_cover(data, ctype)
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.5 * (attempt + 1))

    raise RuntimeError(f"no se pudo descargar el cover de Spotify: {last_error}")


def _validated_cover(data: bytes, mime: str) -> tuple[bytes, str]:
    """Scan and decode a bounded image; return fresh JPEG bytes without extra payloads."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("falta Pillow; instala download_JIJIJI/requirements.txt") from exc
    formats = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}
    if mime not in formats or not data or len(data) > 15 * 1024 * 1024:
        raise RuntimeError("carátula inválida o demasiado grande")
    with tempfile.TemporaryDirectory(prefix="spotify_cover_") as directory:
        path = Path(directory) / "cover.bin"
        path.write_bytes(data)
        _scan_antivirus(path)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data), formats=[formats[mime]]) as image:
                    if image.format != formats[mime] or image.width * image.height > MAX_COVER_PIXELS:
                        raise ValueError("formato/dimensiones fuera de rango")
                    if getattr(image, "n_frames", 1) != 1:
                        raise ValueError("carátula animada no permitida")
                    image.verify()
                with Image.open(io.BytesIO(data), formats=[formats[mime]]) as image:
                    image.load()
                    image.thumbnail((2048, 2048))
                    clean = image.convert("RGB")
                    clean.info.clear()
                    output = io.BytesIO()
                    clean.save(output, format="JPEG", quality=90)
                    rebuilt = output.getvalue()
        except (OSError, ValueError, Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
            raise RuntimeError("contenido de carátula inválido") from exc
        if len(rebuilt) > TAG_ALLOWANCE_BYTES:
            raise RuntimeError("carátula reconstruida supera 2 MiB")
        path.write_bytes(rebuilt)
        _scan_antivirus(path)
    return rebuilt, "image/jpeg"


def _existing_spotify_track_id(path: Path) -> str | None:
    try:
        tags = ID3(path)
        for frame in tags.getall("TXXX"):
            if frame.desc == "Spotify Track ID" and frame.text:
                return str(frame.text[0])
    except Exception:
        return None
    return None


def reusable_downloads(dirs: list[Path]) -> dict[str, Path]:
    """Finished MP3s from earlier runs of this playlist, by Spotify Track ID (the first one found wins)."""
    found: dict[str, Path] = {}
    for directory in dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.mp3"), key=lambda p: str(p).casefold()):
            if path.is_symlink() or not path.is_file():
                continue
            spotify_id = _existing_spotify_track_id(path)
            if spotify_id and spotify_id not in found:
                found[spotify_id] = path
    return found


def reuse_download(previous: Path, out: Path, spotify_id: str, *, require_cover: bool) -> Path:
    """Bring an earlier run's finished MP3 into this run after checking it again (320 kbps, Spotify ID,
    cover). It is moved, not copied: the earlier run was superseded and two copies would only use disk."""
    _validate_existing_output(previous, spotify_id, require_cover=require_cover)
    dest = out / previous.name
    if dest.resolve() == previous.resolve():
        return dest
    if dest.exists():
        _validate_existing_output(dest, spotify_id, require_cover=require_cover)
        return dest
    out.mkdir(parents=True, exist_ok=True)
    shutil.move(str(previous), str(dest))
    return dest


def _validate_existing_output(
    path: Path,
    spotify_id: str,
    *,
    require_cover: bool,
) -> None:
    try:
        audio = MP3(path)
        if audio.info.bitrate < 315_000:
            raise RuntimeError(f"bitrate existente bajo: {audio.info.bitrate}")
        tags = ID3(path)
    except Exception as exc:
        raise RuntimeError(
            f"output existente inválido; usa --overwrite para regenerarlo: {path}: {exc}"
        ) from exc

    stored_id = None
    for frame in tags.getall("TXXX"):
        if frame.desc == "Spotify Track ID" and frame.text:
            stored_id = str(frame.text[0])
            break
    if stored_id != spotify_id:
        raise RuntimeError(
            f"output existente pertenece a otro Spotify Track ID: {path}"
        )
    if require_cover and not tags.getall("APIC"):
        raise RuntimeError(
            f"output existente no contiene cover; usa --overwrite para regenerarlo: {path}"
        )


def _unique_destination(
    out_dir: Path,
    track: Track,
    extended: bool,
    overwrite: bool,
) -> tuple[Path, bool]:
    base = out_dir / spotify_output_filename(track, extended)

    if not base.exists():
        return base, False

    existing_id = _existing_spotify_track_id(base)
    if existing_id == track.spotify_id and not overwrite:
        return base, True

    if overwrite:
        return base, False

    # Otro tema con el mismo nombre de archivo (antes era un NameError: `title` no existía).
    alt = out_dir / f"{base.stem} [{track.spotify_id}].mp3"
    if alt.exists():
        alt_id = _existing_spotify_track_id(alt)
        if alt_id == track.spotify_id:
            return alt, True
        raise RuntimeError(f"colisión de salida no resoluble: {alt}")
    return alt, False


def spotify_output_destination(
    out_dir: Path,
    track: Track,
    *,
    extended: bool = False,
    overwrite: bool = False,
) -> tuple[Path, bool]:
    """Resolve filename collisions using the canonical Soulseek output policy."""
    return _unique_destination(out_dir, track, extended, overwrite)


def write_tags(
    mp3_path: Path,
    track: Track,
    source: LocalFile,
    extended: bool,
    cover: tuple[bytes, str] | None,
):
    tags = ID3()
    tags.add(TIT2(encoding=3, text=track.title))
    tags.add(TPE1(encoding=3, text=track.artists or [track.primary_artist]))
    if track.album_artists:
        tags.add(TPE2(encoding=3, text=track.album_artists))
    tags.add(TALB(encoding=3, text=track.album))
    if track.release_date:
        tags.add(TDRC(encoding=3, text=track.release_date))
    if track.track_number:
        tr = str(track.track_number)
        if track.total_tracks:
            tr += f"/{track.total_tracks}"
        tags.add(TRCK(encoding=3, text=tr))
    if track.disc_number:
        tags.add(TPOS(encoding=3, text=str(track.disc_number)))

    # No atribuir a una Extended el ISRC de la versión corta de Spotify.
    tag_isrc: str | None
    if extended:
        tag_isrc = source.isrc
    else:
        tag_isrc = track.isrc or source.isrc
    if tag_isrc:
        tags.add(TSRC(encoding=3, text=tag_isrc))

    # Keep this profile identical to the Soulseek files known to be readable by
    # both Windows Media Player Legacy and the current Windows Media Player.
    # The modern player can fail to expose APIC when numerous auxiliary TXXX or
    # copyright/publisher frames precede the artwork.
    custom = {
        "Spotify Track ID": track.spotify_id,
        "Spotify URI": track.uri,
        "Spotify URL": track.url,
        "Spotify Album ID": track.album_id or "",
        "Version": "Extended" if extended else "Spotify-length",
    }
    for desc, val in custom.items():
        if val:
            tags.add(TXXX(encoding=3, desc=desc, text=val))

    if cover:
        data, mime = cover
        tags.add(APIC(encoding=3, mime=mime, type=3, desc="Cover", data=data))

    tags.save(mp3_path, v2_version=3)


def apply_spotify_metadata(
    mp3_path: Path,
    track: Track,
    *,
    require_cover: bool = True,
) -> None:
    """Apply the canonical Spotify tags and official cover to an existing MP3."""
    if not mp3_path.is_file():
        raise RuntimeError(f"el MP3 a etiquetar no existe: {mp3_path}")

    source = inspect(mp3_path)
    if source is None or source.format != "mp3":
        raise RuntimeError(f"no se pudo inspeccionar el MP3 grabado: {mp3_path}")
    if (source.bitrate_kbps or 0) < 315:
        raise RuntimeError(
            f"MP3 grabado por debajo de 320 kbps: {source.bitrate_kbps or 0:.1f}"
        )

    cover = download_cover(track.cover_url)
    if require_cover and cover is None:
        raise RuntimeError("Spotify no proporcionó cover utilizable")
    write_tags(mp3_path, track, source, extended=False, cover=cover)


def finalize(
    src: Path,
    out_dir: Path,
    track: Track,
    source: LocalFile,
    extended: bool,
    *,
    overwrite: bool = False,
    require_cover: bool = True,
) -> tuple[Path, bool]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest, already_exists = _unique_destination(
        out_dir,
        track,
        extended,
        overwrite,
    )
    if already_exists:
        _check_audio_header(dest)
        _scan_antivirus(dest)
        _validate_existing_output(
            dest,
            track.spotify_id,
            require_cover=require_cover,
        )
        return dest, True

    if src.resolve() != source.path.resolve():
        raise RuntimeError("el source seleccionado cambió antes de finalizar")
    if not src.is_file():
        raise RuntimeError(f"el source seleccionado no existe: {src}")

    checked = _validate_download(src)
    if source.length_s and checked.length_s and abs(source.length_s - checked.length_s) > 2:
        raise RuntimeError("la duración del source cambió después del matching")
    if src.suffix.lower() == ".mp3" and (checked.bitrate_kbps or 0) < 315:
        raise RuntimeError(
            f"MP3 source cayó por debajo de 320 kbps: {checked.bitrate_kbps or 0:.1f}"
        )

    cover = download_cover(track.cover_url)
    if require_cover and cover is None:
        raise RuntimeError("Spotify no proporcionó cover utilizable")

    with tempfile.TemporaryDirectory(prefix="ssdl_") as tmp:
        tmp_mp3 = Path(tmp) / dest.name

        if src.suffix.lower() == ".mp3":
            # Mantener intactos los frames de audio. Sólo se reemplaza ID3 después.
            shutil.copy2(src, tmp_mp3)
        elif src.suffix.lower() in {".flac", ".wav"}:
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-max_alloc", str(64 * 1024 * 1024),
                "-loglevel",
                "error",
                "-y",
                "-threads", "2",
                "-protocol_whitelist", "file",
                "-f", src.suffix.lower().lstrip("."),
                "-i",
                str(src),
                "-map", "0:a:0",
                "-vn",
                "-map_metadata",
                "-1",
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "320k",
                "-threads", "2",
                "-t", str(checked.length_s + 2),
                "-fs", str(min(MAX_SOURCE_BYTES, math.ceil((checked.length_s + 2) * 40000) + TAG_ALLOWANCE_BYTES)),
                str(tmp_mp3),
            ]
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=FFMPEG_TIMEOUT_SECONDS,
                    stdin=subprocess.DEVNULL,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("ffmpeg excedió el tiempo permitido; conversión cancelada") from exc
            except FileNotFoundError:
                raise RuntimeError("ffmpeg no está en el PATH")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"ffmpeg falló (código {e.returncode})") from e
        else:
            raise RuntimeError(f"formato no soportado al finalizar: {src.suffix}")

        try:
            info = MP3(tmp_mp3).info
            if info.bitrate < 315_000:
                raise RuntimeError(f"bitrate final bajo: {info.bitrate}")
            if not _positive_number(info.length) or abs(info.length - checked.length_s) > 2:
                raise RuntimeError("duración final incompatible con el source; posible truncamiento")
        except Exception as e:
            raise RuntimeError(f"no se pudo verificar MP3: {e}")

        write_tags(tmp_mp3, track, source, extended, cover)
        if tmp_mp3.stat().st_size > MAX_SOURCE_BYTES:
            raise RuntimeError("MP3 final supera el límite de 512 MiB")
        _scan_antivirus(tmp_mp3)

        if dest.exists():
            if overwrite:
                dest.unlink()
            else:
                raise RuntimeError(f"la salida apareció durante la finalización: {dest}")
        shutil.move(str(tmp_mp3), dest)

    return dest, False


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------

def _track_report(track: Track) -> dict[str, Any]:
    data = asdict(track)
    data["duration_seconds"] = round(track.duration_s, 3)
    data["duration_hms"] = _duration_hms(track.duration_ms)
    return data


def recording_entry(position: int, track: Track) -> dict[str, Any]:
    """Un tema en el formato que graba playlist_loopback_recorder.py (y que lista --missing-json)."""
    return {
        "playlist_position": position,
        "duration_ms": track.duration_ms,
        "duration_seconds": round(track.duration_s, 3),
        "duration_hms": _duration_hms(track.duration_ms),
        "spotify": _track_report(track),
    }


def _candidate_report(
    candidate: LocalFile,
    accepted: bool,
    score: float,
    extended: bool,
    reasons: list[str],
) -> dict[str, Any]:
    return {
        "path": str(candidate.path),
        "format": candidate.format,
        "length_s": candidate.length_s,
        "bitrate_kbps": candidate.bitrate_kbps,
        "tag_title": candidate.tag_title,
        "tag_artists": candidate.tag_artists,
        "isrc": candidate.isrc,
        "accepted": accepted,
        "score": round(score, 5),
        "extended": extended,
        "reasons": reasons,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _replace_report(path: Path, payload: dict[str, Any], *, attempts: int = 5) -> None:
    """Rewrite a report others poll without ever exposing a half-written file. On Windows the
    replace fails while a reader has it open: retry briefly, and skip this update if it persists
    (a later update rewrites the whole file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(min(0.05 * (attempt + 1), 0.5))
    tmp.unlink(missing_ok=True)


def _read_report(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Búsqueda tema a tema
# ---------------------------------------------------------------------------

def default_search_log() -> Path:
    """Las búsquedas recientes de este ordenador, junto al token de Spotify (%APPDATA%)."""
    return token_cache_path().with_name(SEARCH_LOG_NAME)


class SearchLimiter:
    """Soulseek bloquea la cuenta 30 minutos si se pasan las 34 búsquedas cada 220 s. El limitador de
    sockseek vive dentro de un proceso, y ahora cada búsqueda es un proceso: la cuenta se lleva aquí, en
    un archivo del ordenador, para que tampoco se pase una descarga que empieza justo después de otra."""

    def __init__(
        self,
        path: Path | None,
        *,
        per_window: int = SOULSEEK_SEARCHES_PER_WINDOW,
        window_s: float = SOULSEEK_SEARCH_WINDOW_S,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.path = path
        self.per_window = per_window
        self.window_s = window_s
        self.clock = clock
        self.sleep = sleep
        self.stamps: list[float] = []
        self._lock = threading.Lock()  # varios temas a la vez: una cuenta para todos

    def _recent(self) -> list[float]:
        now = self.clock()
        stamps = list(self.stamps)
        if self.path is not None:
            stored = _read_report(self.path).get("searches")
            if isinstance(stored, list):
                stamps += [float(s) for s in stored if isinstance(s, (int, float)) and not isinstance(s, bool)]
        # Una hora futura (el reloj se atrasó) no puede frenar las búsquedas para siempre.
        return sorted({s for s in stamps if now - self.window_s < s <= now + 1})

    def wait_s(self) -> float:
        recent = self._recent()
        if len(recent) < self.per_window:
            return 0.0
        return max(0.0, recent[-self.per_window] + self.window_s - self.clock())

    def acquire(self, on_wait: Callable[[float], None] | None = None) -> float:
        """Espera turno y anota la búsqueda; devuelve su marca, para `finish`."""
        announced = False
        while True:
            with self._lock:
                wait = self.wait_s()
                if wait <= 0:
                    stamp = self.clock()
                    self.stamps = self._recent() + [stamp]
                    self._save()
                    return stamp
            if on_wait is not None and not announced:
                on_wait(wait)
                announced = True
            self.sleep(min(wait, 5.0))

    def finish(self, stamp: float) -> None:
        """La búsqueda terminó: cuenta desde ahora. Sockseek se conecta antes de buscar (hasta 20 s), así que
        la hora de inicio subestima cuándo salió la búsqueda; la de fin, no."""
        with self._lock:
            stamps = self._recent()
            if stamp in stamps:
                stamps.remove(stamp)
            self.stamps = stamps + [self.clock()]
            self._save()

    def _save(self) -> None:
        if self.path is None:
            return
        # Un temporal propio de este proceso. Una descarga a la vez (la app no lanza dos).
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps({"searches": self.stamps}), encoding="utf-8")
            for attempt in range(10):
                try:
                    os.replace(tmp, self.path)
                    return
                except PermissionError:
                    time.sleep(0.05 * (attempt + 1))
        except OSError:
            pass  # sin archivo, el límite se respeta igual dentro de esta descarga
        tmp.unlink(missing_ok=True)


def _append_sockseek_log(source: Path | None, destination: Path | None, track: Track) -> None:
    if source is None or destination is None or not source.is_file():
        return
    try:
        text = source.read_text(encoding="utf-8", errors="replace")
        with _PRINT_LOCK, destination.open("a", encoding="utf-8") as handle:
            handle.write(f"\n===== {track.primary_artist} - {track.title} =====\n{text}")
    except OSError:
        pass


def search_track(
    track: Track,
    *,
    limiter: SearchLimiter,
    timeout: float = DEFAULT_SEARCH_TIMEOUT_S,
    length_tol: int = 60,
    log_file: Path | None = None,
    attempts: int = SEARCH_ATTEMPTS,
    retry_pause: float = SEARCH_RETRY_PAUSE_S,
    on_wait: Callable[[float], None] | None = None,
) -> AvailabilityCheck:
    """Una búsqueda en Soulseek para un tema (un proceso de sockseek, como cuando se busca a mano). Si
    falla (tiempo agotado, sesión), se reintenta una vez; el error que quede lo decide quien llama."""
    check = AvailabilityCheck(status="error", result_count=0, candidates=[], error="sin búsqueda")
    for attempt in range(max(1, attempts)):
        if attempt:
            say(f"    ! la búsqueda falló ({check.error}); reintento en {retry_pause:.0f} s")
            time.sleep(retry_pause)
        stamp = limiter.acquire(on_wait)
        with tempfile.TemporaryDirectory(prefix="sockseek_search_") as tmp:
            own_log = Path(tmp) / "sockseek.log" if log_file is not None else None
            try:
                check = check_soulseek_availability_batch(
                    [track], length_tol=length_tol, timeout=timeout, log_file=own_log
                )[0]
            finally:
                limiter.finish(stamp)
            _append_sockseek_log(own_log, log_file, track)
        if check.status != "error":
            return check
    return check


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Spotify playlist → tema a tema: búsqueda en Soulseek y, si está, descarga en el momento "
            "(MP3 320 con verificación, metadata + cover); lo que no se consigue va a la cola para grabarlo"
        )
    )
    ap.add_argument("playlist", help="URL, URI o ID de la playlist de Spotify")
    ap.add_argument("-o", "--output", required=True, help="Carpeta de salida de los MP3 finales")
    ap.add_argument(
        "--client-id",
        default=os.environ.get("SPOTIFY_CLIENT_ID"),
        help="Spotify Client ID (o variable SPOTIFY_CLIENT_ID)",
    )
    ap.add_argument("--staging", default=None, help="Carpeta temporal de descargas Soulseek")
    ap.add_argument("--length-tol", type=int, default=60, help="Tolerancia de duración para sockseek (default 60s)")
    ap.add_argument("--no-extended", action="store_true", help="No preferir versiones Extended")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Buscar todos los temas y generar los reportes, sin crear playlist en Spotify ni descargar audio",
    )
    ap.add_argument(
        "--preflight-only",
        action="store_true",
        help="Buscar todos los temas y crear la playlist Spotify de los que faltan, sin descargar audio",
    )
    ap.add_argument("--max", type=int, default=None, help="Limitar número de tracks (pruebas)")
    ap.add_argument("--report", default=None, help="Ruta del JSON de auditoría de las descargas")
    ap.add_argument(
        "--availability-report",
        default=None,
        help="Ruta del JSON de la búsqueda, tema a tema (default: output/soulseek_availability.json)",
    )
    ap.add_argument(
        "--recording-queue",
        default=None,
        help=(
            "Cola de grabación (JSON por línea, ver recording_queue.py): cada tema que haya que grabar de "
            "Spotify se agrega apenas se sabe, para que el orquestador lo grabe mientras sigue la búsqueda. "
            "Con una cola existente (descarga retomada), lo que ya está en ella no se vuelve a buscar"
        ),
    )
    ap.add_argument(
        "--search-log",
        default=None,
        help="Archivo con las búsquedas recientes, para el límite de Soulseek (default: %%APPDATA%%)",
    )
    ap.add_argument(
        "--parallel",
        type=int,
        default=DEFAULT_PARALLEL,
        help=(
            f"Temas a la vez, cada uno buscado y bajado por su lado, con una sola sesión de Soulseek "
            f"(default {DEFAULT_PARALLEL}, hasta {MAX_PARALLEL}; 1 = de a uno)"
        ),
    )
    ap.add_argument(
        "--no-daemon",
        action="store_true",
        help="Sin el daemon de sockseek: cada búsqueda y descarga abre su propia sesión (de a un tema)",
    )
    ap.add_argument(
        "--sockseek-mock-dir",
        default=None,
        help="Pruebas sin Soulseek: busca y baja de esta carpeta local (--mock-files-dir de sockseek)",
    )
    ap.add_argument(
        "--search-timeout",
        type=float,
        default=DEFAULT_SEARCH_TIMEOUT_S,
        help=f"Segundos máximos de una búsqueda (default {DEFAULT_SEARCH_TIMEOUT_S:.0f})",
    )
    ap.add_argument(
        "--missing-list",
        default=None,
        help="Sin cola de grabación: TXT de lo que hay que grabar (default: output/soulseek_missing.txt)",
    )
    ap.add_argument(
        "--missing-json",
        default=None,
        help=(
            "Sin cola de grabación: JSON de lo que hay que grabar, con metadata Spotify completa y orden de "
            "playlist (default: output/soulseek_missing.json)"
        ),
    )
    ap.add_argument(
        "--no-missing-playlist",
        action="store_true",
        help="Sin cola de grabación: no crear playlist privada de Spotify con lo que hay que grabar",
    )
    ap.add_argument("--missing-playlist-name", default=None, help="Nombre de la playlist Spotify de faltantes")
    ap.add_argument(
        "--missing-playlist-batch-size",
        type=int,
        default=DEFAULT_MISSING_PLAYLIST_BATCH_SIZE,
        help=f"Cantidad maxima de temas por playlist Spotify de faltantes (default {DEFAULT_MISSING_PLAYLIST_BATCH_SIZE})",
    )
    ap.add_argument("--overwrite", action="store_true", help="Permitir reemplazar un output existente con el mismo nombre")
    ap.add_argument("--allow-missing-cover", action="store_true", help="No fallar si Spotify no entrega/descarga el cover")
    ap.add_argument(
        "--reuse-dir",
        action="append",
        default=[],
        help=(
            "Carpeta con MP3 ya terminados de una descarga anterior de esta playlist (repetible): "
            "esos temas no se buscan, no se bajan de nuevo ni se dan por faltantes; se traen a esta descarga"
        ),
    )
    ap.add_argument(
        "--have-dir",
        action="append",
        default=[],
        help=(
            "Carpeta de la música de esta playlist (repetible, no se modifica): lo que ya está ahí, por "
            "Spotify Track ID, no se busca ni se baja"
        ),
    )
    args = ap.parse_args()

    if args.dry_run and args.preflight_only:
        print("Usa sólo uno de --dry-run o --preflight-only.", file=sys.stderr)
        return 2
    if args.missing_playlist_batch_size <= 0:
        ap.error("--missing-playlist-batch-size debe ser mayor que cero")
    if not args.client_id:
        print("Falta Client ID. Usa --client-id o exporta SPOTIFY_CLIENT_ID", file=sys.stderr)
        return 2

    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report).expanduser().resolve() if args.report else out / "spotify_reconcile_report.json"
    availability_report_path = (
        Path(args.availability_report).expanduser().resolve()
        if args.availability_report
        else out / "soulseek_availability.json"
    )
    queue_path = Path(args.recording_queue).expanduser().resolve() if args.recording_queue else None
    missing_list_path = (
        Path(args.missing_list).expanduser().resolve() if args.missing_list else out / "soulseek_missing.txt"
    )
    missing_json_path = (
        Path(args.missing_json).expanduser().resolve() if args.missing_json else out / "soulseek_missing.json"
    )
    downloads_enabled = not (args.dry_run or args.preflight_only)

    print("Conectando a Spotify…")
    try:
        sp = Spotify(args.client_id)
        pl_name, tracks = sp.playlist_tracks(args.playlist)
    except Exception as exc:
        print(f"Spotify falló: {exc}", file=sys.stderr)
        return 2

    if args.max:
        tracks = tracks[: args.max]
    print(f"Playlist: «{pl_name}» — {len(tracks)} temas")

    try:
        find_sockseek()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2
    if downloads_enabled and not shutil.which("ffmpeg"):
        print("ffmpeg no está en el PATH", file=sys.stderr)
        return 2

    # Lo que ya está no se busca ni se baja: en la carpeta de la música (--have-dir, no se toca) o bajado
    # en una descarga anterior que se cortó antes de guardar (--reuse-dir, se trae a esta descarga).
    saved = reusable_downloads([Path(d).expanduser().resolve() for d in args.have_dir])
    # Con cola (descarga retomada), también lo ya terminado en la propia carpeta: el reporte pudo no alcanzar
    # a registrar el último tema antes del corte.
    reusable = reusable_downloads(([out] if args.recording_queue else [])
                                  + [Path(d).expanduser().resolve() for d in args.reuse_dir])
    # Una descarga retomada (con cola): lo ya buscado, bajado o puesto en la cola sigue como estaba.
    queued_ids: set[str] = set(read_queue(queue_path).ids()) if queue_path is not None else set()
    previous_rows: dict[str, dict[str, Any]] = {}
    previous_downloads: dict[str, dict[str, Any]] = {}
    if queue_path is not None:
        for row in _read_report(availability_report_path).get("tracks") or []:
            if isinstance(row, dict) and row.get("status") != "pending":
                previous_rows.setdefault(str((row.get("spotify") or {}).get("spotify_id")), row)
        for row in _read_report(report_path).get("tracks") or []:
            if isinstance(row, dict):
                previous_downloads[str((row.get("spotify") or {}).get("spotify_id"))] = row

    rows: list[dict[str, Any]] = []
    for position, track in enumerate(tracks, 1):
        row = {"position": position, "spotify": _track_report(track), "status": "pending", "result_count": 0,
               "quality_unknown": False, "top_candidates": [], "error": None}
        earlier = previous_rows.get(track.spotify_id)
        if track.spotify_id in saved:
            row.update(status="available", note="ya en tu carpeta")
        elif earlier is not None:
            row.update({k: earlier.get(k) for k in ("status", "result_count", "quality_unknown", "top_candidates",
                                                    "error", "note") if k in earlier})
        rows.append(row)
    downloads: dict[int, dict[str, Any]] = {}
    current: dict[int, dict[str, Any]] = {}  # los temas en curso (uno por trabajador): buscando, bajando, esperando
    counts = {"searched": 0, "found": 0, "done": 0}
    missing_playlists: list[dict[str, Any]] | None = None
    staging = Path(args.staging or tempfile.mkdtemp(prefix="soulseek_staging_")).expanduser().resolve()
    staging_is_temporary = args.staging is None
    log_file = availability_report_path.with_name("sockseek_preflight.log") if queue_path is not None else None
    lock = threading.RLock()  # filas, cola, reportes y contadores, compartidos por los trabajadores
    parallel = max(1, min(MAX_PARALLEL, int(args.parallel or 1)))

    def summary(final: bool = False) -> dict[str, Any]:
        statuses = [row["status"] for row in rows]
        return {
            "playlist": {"name": pl_name, "source": args.playlist},
            "total_tracks": len(tracks),
            "checked": sum(s != "pending" for s in statuses),
            "available": sum(s in {"available", "available_quality_unknown"} for s in statuses),
            "missing": statuses.count("missing"),
            "errors": statuses.count("error"),
            "preflight_mode": "track_by_track",
            "parallel": parallel,
            "current": None if final else ([dict(c) for c in current.values()] or None),
            "tracks": rows,
        }

    def publish(final: bool = False) -> None:
        with lock:
            _replace_report(availability_report_path, summary(final), attempts=60 if final else 5)

    def publish_downloads(final: bool = False) -> None:
        with lock:
            ordered = [downloads[p] for p in sorted(downloads)]
            failed_now = sum(row["status"] not in DOWNLOADED_STATUSES for row in ordered)
            _replace_report(report_path, {
                "playlist": {"name": pl_name, "source": args.playlist},
                "availability_report": str(availability_report_path),
                "recording_queue": str(queue_path) if queue_path else None,
                "output": str(out),
                "staging": str(staging),
                "staging_is_temporary": staging_is_temporary,
                "prefer_extended": not args.no_extended,
                "completed": len(ordered) - failed_now,
                "failed": failed_now,
                "tracks": ordered,
            }, attempts=60 if final else 5)

    def set_step(position: int, step: str | None, **extra: Any) -> None:
        with lock:
            if step is None:
                current.pop(position, None)
            else:
                current[position] = {"position": position, "step": step, **extra}
        publish()

    to_record: list[tuple[int, Track]] = []

    def queue_for_recording(position: int, track: Track, origin: str, reason: str) -> None:
        with lock:
            if track.spotify_id in queued_ids:
                return
            if queue_path is not None:
                append_track(queue_path, recording_entry(position, track), origin=origin, reason=reason)
            queued_ids.add(track.spotify_id)
            to_record.append((position, track))
        say(f"    ↻ se grabará de Spotify: {reason}")

    def download_row(position: int, track: Track, availability: AvailabilityCheck | None) -> dict[str, Any]:
        row: dict[str, Any] = {"position": position, "spotify": _track_report(track), "status": "pending",
                               "candidates": []}
        if availability is not None:
            row["preflight"] = {"status": availability.status, "result_count": availability.result_count,
                                "quality_unknown": availability.quality_unknown}
        return row

    print("\n=== TEMA A TEMA: se busca cada uno en Soulseek y, si está, se baja enseguida ===")
    if queue_path is not None:
        print("Lo que no se consiga se graba de Spotify mientras sigue la búsqueda.")
    limiter = SearchLimiter(Path(args.search_log).expanduser().resolve() if args.search_log else default_search_log())
    held_errors: list[tuple[int, Track, str]] = []  # se mandan a grabar solo si después otra búsqueda funciona
    consecutive_errors = 0
    last_ok: Track | None = None  # el último tema que se pudo buscar en esta descarga
    aborted: str | None = None
    retry_later: list[tuple[int, Track]] = []  # Soulseek los tenía pero no llegó ningún archivo: otra búsqueda al final
    avoid: set[tuple[str, str]] = set()  # (usuario, archivo) que ya fallaron: no se vuelven a pedir

    # Una sola sesión de Soulseek para todo (el daemon de sockseek); sin ella, de a un tema.
    mock_dir = Path(args.sockseek_mock_dir).expanduser().resolve() if args.sockseek_mock_dir else None
    use_mock_dir(mock_dir)
    daemon_state = availability_report_path.with_name(DAEMON_STATE_NAME)
    stop_daemon_from_state(daemon_state)  # el de una descarga cortada: su sesión cortaría la de esta
    daemon: SockseekDaemon | None = None
    if not args.no_daemon:
        daemon = SockseekDaemon(find_sockseek(), state_path=daemon_state,
                                log_path=availability_report_path.with_name("sockseek_daemon.log"),
                                mock_dir=mock_dir)
        try:
            daemon.start()
        except (RuntimeError, OSError) as exc:
            print(f"! No se pudo abrir la sesión compartida de Soulseek ({exc}): sigo de a un tema.", flush=True)
            daemon = None
    if daemon is None:
        parallel = 1  # dos sesiones con la misma cuenta se cortan entre sí
    use_daemon(daemon)
    print(f"Soulseek: {'una sola sesión para todo' if daemon else 'una sesión por búsqueda y descarga'}; "
          f"{parallel} tema(s) a la vez.", flush=True)

    # Un tema repetido en la playlist se resuelve una vez (la primera aparición) y se copia al final.
    work: list[tuple[int, Track]] = []
    repeated: list[tuple[int, Track, int]] = []
    first_position: dict[str, int] = {}
    for position, track in enumerate(tracks, 1):
        if track.spotify_id in first_position:
            repeated.append((position, track, first_position[track.spotify_id]))
        else:
            first_position[track.spotify_id] = position
            work.append((position, track))

    def process_track(position: int, track: Track, second_try: bool) -> None:
        nonlocal consecutive_errors, last_ok, aborted
        row = rows[position - 1]
        tid = track.spotify_id
        say(f"[{position}/{len(tracks)} | Spotify #{position}] {track.primary_artist} - {track.title}"
            + ("  (segundo intento)" if second_try else ""))

        if not second_try:
            in_folder = saved.get(tid)
            if in_folder is not None:
                say(f"    → ya está en tu carpeta: {in_folder.name}")
                with lock:
                    downloads[position] = {**download_row(position, track, None), "status": "already_finalized",
                                           "output": str(in_folder),
                                           "selected": {"path": str(in_folder),
                                                        "reasons": ["ya estaba en tu carpeta de música"]}}
                return
            earlier = previous_downloads.get(tid)
            if (earlier and earlier.get("status") in DOWNLOADED_STATUSES and earlier.get("output")
                    and Path(str(earlier["output"])).is_file()):
                say(f"    → ya bajado: {Path(str(earlier['output'])).name}")
                with lock:
                    downloads[position] = {**earlier, "position": position}
                return
            if tid in queued_ids:
                say("    ↻ ya está en la cola para grabar de Spotify")
                if earlier is not None:
                    with lock:
                        downloads[position] = {**earlier, "position": position}
                return
            previous = reusable.get(tid)
            if previous is not None and downloads_enabled:
                try:
                    dest = reuse_download(previous, out, tid, require_cover=not args.allow_missing_cover)
                except (RuntimeError, OSError) as exc:
                    say(f"    ! no se pudo reutilizar {previous.name}: {exc}; lo busco")
                else:
                    say(f"    → reutilizado de una descarga anterior: {dest.name}")
                    with lock:
                        row.update(status="available", note="bajado en una descarga anterior")
                        downloads[position] = {**download_row(position, track, None), "status": "already_finalized",
                                               "output": str(dest),
                                               "selected": {"path": str(previous),
                                                            "reasons": ["reutilizado de una descarga anterior"]}}
                    publish_downloads()
                    return

        def on_wait(seconds: float) -> None:
            say(f"    Soulseek permite {SOULSEEK_SEARCHES_PER_WINDOW} búsquedas cada {SOULSEEK_SEARCH_WINDOW_S} s: "
                f"espero {seconds:.0f} s antes de buscar este.")
            set_step(position, "wait", until=time.time() + seconds)

        set_step(position, "search")
        check = search_track(track, limiter=limiter, timeout=args.search_timeout, length_tol=args.length_tol,
                             log_file=log_file, on_wait=on_wait)
        set_step(position, "search")
        flush: list[tuple[int, Track, str]] = []
        with lock:
            row.update(status=check.status, result_count=check.result_count, quality_unknown=check.quality_unknown,
                       top_candidates=check.candidates, error=check.error)
            if check.status == "error":
                consecutive_errors += 1
                held_errors.append((position, track, check.error or "error desconocido"))
                if consecutive_errors >= MAX_CONSECUTIVE_SEARCH_ERRORS and aborted is None:
                    aborted = check.error or "error desconocido"
            else:
                consecutive_errors = 0
                last_ok = track
                counts["searched"] += 1
                flush, held_errors[:] = list(held_errors), []
        for held_position, held_track, error in flush:  # la búsqueda funciona: el error era de ese tema
            queue_for_recording(held_position, held_track, "search_error", f"La búsqueda en Soulseek falló ({error})")
        if check.status == "error":
            say(f"    ! la búsqueda en Soulseek falló: {check.error}")
            return
        if check.status == "missing":
            say("    ✗ no está en Soulseek")
            queue_for_recording(position, track, "missing", "No está en Soulseek")
            return
        with lock:
            counts["found"] += 1
        say(f"    ✓ en Soulseek: {check.result_count} archivo(s) utilizable(s)"
            + (" (bitrate no anunciado)" if check.quality_unknown else ""))
        if not downloads_enabled:
            return

        set_step(position, "download")
        drow = download_row(position, track, check)
        best: LocalFile | None = None
        best_score = -1.0
        best_ext = False
        best_reasons: list[str] = []

        def verify(path: Path) -> bool:
            """Cada archivo bajado se verifica antes de aceptarlo; si no pasa, se prueba otro."""
            nonlocal best, best_score, best_ext, best_reasons
            local = inspect(path)
            if not local:
                drow["candidates"].append({"path": str(path), "accepted": False, "reasons": ["no se pudo leer el audio"]})
                return False
            accepted, score, extended, reasons = score_candidate(track, local, prefer_extended=not args.no_extended)
            drow["candidates"].append(_candidate_report(local, accepted, score, extended, reasons))
            if accepted:
                best, best_score, best_ext, best_reasons = local, score, extended, reasons
            return accepted

        outcome: dict[str, int] = {}
        try:
            files = download_with_sockseek(track, staging, length_tol=args.length_tol, availability=check,
                                           verify=verify, prefer_extended=not args.no_extended, avoid=avoid,
                                           outcome=outcome)
        except Exception as exc:
            say(f"    ✗ error descarga: {exc}")
            drow.update(status="download_error", error=str(exc))
            files = []
        if drow["status"] == "pending" and (not files or best is None):
            if drow["candidates"]:
                say(f"    ✗ ninguno de los {len(drow['candidates'])} archivos bajados pasó la verificación")
                drow["status"] = "verification_failed"
            else:
                say("    ✗ Soulseek tenía resultados, pero la descarga no produjo archivo")
                drow["status"] = "availability_changed_or_download_failed"
        if drow["status"] == "pending":
            assert best is not None
            say(f"    ✓ elegido: {best.path.name}" + ("  [Extended]" if best_ext else ""))
            drow["selected"] = {"path": str(best.path), "score": best_score, "extended": best_ext, "reasons": best_reasons}
            try:
                dest, already_exists = finalize(best.path, out, track, best, best_ext, overwrite=args.overwrite,
                                                require_cover=not args.allow_missing_cover)
                say(f"    → {'ya existía: ' if already_exists else ''}{dest.name}")
                drow.update(status="already_finalized" if already_exists else "finalized", output=str(dest))
            except Exception as exc:
                say(f"    ✗ finalize: {exc}")
                drow.update(status="finalize_error", error=str(exc))
        nothing_arrived = drow["status"] == "availability_changed_or_download_failed" and not outcome.get("arrived")
        if drow["status"] not in DOWNLOADED_STATUSES and nothing_arrived and not second_try:
            # Ningún usuario llegó a mandar el archivo (en cola, desconectado, sin avanzar): otra búsqueda al
            # final, con otros usuarios, antes de grabarlo.
            drow["retry_pending"] = True
            with lock:
                retry_later.append((position, track))
            say("    ↺ no llegó ningún archivo: se vuelve a buscar al final")
        elif drow["status"] not in DOWNLOADED_STATUSES:
            queue_for_recording(position, track, "failed", _download_failure_reason(drow))
        with lock:
            downloads[position] = drow
        publish_downloads()

    fatal: list[BaseException] = []

    def run_pool(items: list[tuple[int, Track]], second_try: bool = False) -> None:
        index = 0

        def worker() -> None:
            nonlocal index
            while True:
                with lock:
                    if aborted is not None or fatal or index >= len(items):
                        return
                    position, track = items[index]
                    index += 1
                _TRACK_TAG.value = f"#{position}" if parallel > 1 else None
                try:
                    process_track(position, track, second_try)
                except BaseException as exc:  # un error inesperado corta la descarga (no se manda nada a grabar por él)
                    with lock:
                        fatal.append(exc)
                    return
                finally:
                    set_step(position, None)
                    _TRACK_TAG.value = None
                with lock:
                    counts["done"] += 1
                    done = counts["done"]
                if done % 10 == 0:
                    say(f"Soulseek: {done} temas vistos, {counts['found']} encontrados, {len(queued_ids)} para grabar")

        threads = [threading.Thread(target=worker, name=f"tema-{n + 1}", daemon=True)
                   for n in range(max(1, min(parallel, len(items))))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if fatal:
            raise fatal[0]

    publish()
    try:
        run_pool(work)
        if aborted is None and retry_later:
            retry_later.sort(key=lambda item: item[0])
            print(f"\n=== SEGUNDO INTENTO: {len(retry_later)} tema(s) que Soulseek tenía pero no llegaron ===", flush=True)
            for position, _ in retry_later:
                downloads[position].pop("retry_pending", None)
            run_pool(list(retry_later), second_try=True)
        if aborted is None and held_errors:
            # Los últimos temas fallaron y no hay otro después que pruebe que la búsqueda funciona: se prueba
            # con uno que sí se encontró. Si tampoco funciona, el problema es la búsqueda y no se graban.
            probe = None
            if last_ok is not None:
                print(f"Compruebo que la búsqueda en Soulseek funciona con «{last_ok.title}»…", flush=True)
                probe = search_track(last_ok, limiter=limiter, timeout=args.search_timeout,
                                     length_tol=args.length_tol, log_file=log_file)
            if probe is not None and probe.status != "error":
                for held_position, held_track, error in list(held_errors):
                    queue_for_recording(held_position, held_track, "search_error",
                                        f"La búsqueda en Soulseek falló ({error})")
            else:
                aborted = held_errors[-1][2]
    finally:
        use_daemon(None)
        if daemon is not None:
            daemon.stop()

    for position, track, first in repeated:  # lo que se resolvió en su primera aparición
        say(f"[{position}/{len(tracks)} | Spotify #{position}] {track.primary_artist} - {track.title}\n"
            "    = repetido en la playlist: ya se resolvió más arriba")
        rows[position - 1].update({k: rows[first - 1].get(k) for k in ("status", "result_count", "quality_unknown",
                                                                       "top_candidates", "error")},
                                  note="repetido en la playlist")
        if first in downloads:
            downloads[position] = {**downloads[first], "position": position}
    current.clear()
    publish(final=True)
    publish_downloads(final=True)

    staging_cleanup_error: str | None = None
    if staging_is_temporary:
        try:
            if staging.exists():
                shutil.rmtree(staging)
        except OSError as exc:
            staging_cleanup_error = str(exc)

    if aborted is not None:
        print(
            f"\nERROR: la búsqueda en Soulseek falló en {max(1, len(held_errors))} tema(s) seguido(s) ({aborted}). Se corta "
            "aquí: el problema es la búsqueda, no los temas, y esos no se mandan a grabar. Revisa que la cuenta de "
            "Soulseek no esté abierta en otra PC y vuelve a intentar.",
            file=sys.stderr,
        )
    if queue_path is not None:
        close_queue(queue_path, END_SEARCH_FAILED if aborted else END_FINISHED, aborted)
    else:
        to_record.sort(key=lambda item: item[0])
        write_missing_list(missing_list_path, to_record)
        write_missing_json(missing_json_path, source_playlist_name=pl_name, source_playlist=args.playlist,
                           missing=to_record, playlist_batch_size=args.missing_playlist_batch_size)
        if to_record and not args.dry_run and not args.no_missing_playlist:
            print("\nCreando playlists privadas de Spotify con lo que hay que grabar…")
            try:
                missing_playlists = sp.create_missing_playlists(
                    pl_name, [track for _, track in to_record], name=args.missing_playlist_name,
                    batch_size=args.missing_playlist_batch_size)
                for batch in missing_playlists:
                    print(f"[{batch['batch_number']}/{batch['batch_count']}] {batch['track_count']} tema(s): {batch['url']}")
            except Exception as exc:
                print(f"No se pudieron crear las playlists de faltantes: {exc}", file=sys.stderr)
                missing_playlists = [{"id": "", "name": "", "url": "", "error": str(exc)}]
            write_missing_json(missing_json_path, source_playlist_name=pl_name, source_playlist=args.playlist,
                               missing=to_record, generated_playlists=missing_playlists,
                               playlist_batch_size=args.missing_playlist_batch_size)
        print(f"JSON de lo que hay que grabar: {missing_json_path}")

    ok = sum(row["status"] in DOWNLOADED_STATUSES for row in downloads.values())
    failed = sum(row["status"] not in DOWNLOADED_STATUSES for row in downloads.values())
    print(f"\nListo. {ok} tema(s) bajados o ya guardados en {out}; {len(queued_ids)} para grabar de Spotify.")
    print(f"Reporte: {report_path}")
    if aborted is not None:
        return EXIT_SEARCH_FAILED
    return 0 if failed == 0 and staging_cleanup_error is None else 3


def _download_failure_reason(row: dict[str, Any]) -> str:
    return {
        "verification_failed": "Ningún archivo de Soulseek pasó la verificación",
        "download_error": "Falló la descarga de Soulseek",
        "availability_changed_or_download_failed": "Soulseek tenía el tema, pero no se pudo bajar",
        "finalize_error": "No se pudo guardar el MP3",
    }.get(str(row.get("status")), "No se pudo bajar de Soulseek")


if __name__ == "__main__":
    raise SystemExit(main())
