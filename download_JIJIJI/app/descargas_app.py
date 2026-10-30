"""
PURPOSE: App local para bajar playlists de Spotify (Soulseek + grabación por loopback) sin línea de
         comandos. Sirve download_JIJIJI/app/index.html en una ventana propia (Edge --app) y desde ahí:
           - pegar el enlace de una playlist y verla (portada, temas, duración);
           - elegir la salida de audio de Windows (la que suena y la que se graba), probarla con un
             tono que además se graba para medirlo, y ver el nivel de lo que está sonando;
           - elegir la carpeta de destino de la música (y la de trabajo de los runs);
           - lanzar, seguir tema a tema, detener y reanudar spotify_soulseek_orchestrator.py;
           - ver el historial de runs, también los lanzados desde la terminal.
         Solo escucha en 127.0.0.1; las acciones exigen el token de la página y un Host/Origin local.
         Una sola instancia: si ya está abierta, el acceso directo solo vuelve a abrir la ventana.
         Se apaga sola cuando no hay ventana abierta ni descarga en curso.
         Uso: .venv\\Scripts\\pythonw.exe download_JIJIJI\\app\\descargas_app.py [--port N] [--no-window]
CHANGELOG:
  - 2026-10-30: Creación inicial.
  - 2026-10-30: Temas que Soulseek no pudo dar: «se grabará» y su grabación (recorded_failed/);
                run «incompleto» (partial) con lo que sí llegó a la carpeta.
"""
from __future__ import annotations

import argparse
import collections
import colorsys
import ctypes
import io
import json
import math
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
import urllib.request
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

APP_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = APP_DIR.parent
REPO_ROOT = DOWNLOAD_DIR.parent
sys.path.insert(0, str(DOWNLOAD_DIR))

APP_ID = "traktor-descargas"
CONFIG_DIR = Path(os.environ.get("APPDATA") or (Path.home() / ".config")) / "traktor_descargas"
DEFAULT_PORT = 8790
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
RUN_DIR_RE = re.compile(r"^(\d{8})-(\d{6})_([A-Za-z0-9]+)$")
STALE_SECONDS = 20 * 60          # un run "running" sin escribir nada en 20 min se considera cortado
IDLE_SHUTDOWN_SECONDS = 90       # sin ventana abierta ni descarga: la app se apaga
ACTIVE_STATUSES = {"running", "consolidating"}
DOWNLOADED = {"finalized", "already_finalized"}
COVER_HOSTS = (".scdn.co", ".spotifycdn.com")

import spotify_soulseek_orchestrator as orch  # noqa: E402


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def json_candidates(run_dir: Path, pattern: str) -> list[Path]:
    """Archivos del run que coinciden, en la raíz o (si el run ya terminó) en diagnostics/, por fecha."""
    found = [*run_dir.glob(pattern), *(run_dir / "diagnostics").glob(pattern)]
    return sorted((p for p in found if p.is_file()), key=lambda p: p.stat().st_mtime)


def newest_mtime(folder: Path) -> float:
    newest = 0.0
    for root, _dirs, files in os.walk(folder):
        for name in files:
            try:
                newest = max(newest, os.stat(os.path.join(root, name)).st_mtime)
            except OSError:
                pass
    return newest


def run_started(run_dir: Path) -> float | None:
    m = RUN_DIR_RE.match(run_dir.name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").timestamp()
    except ValueError:
        return None


def is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def console_python() -> str:
    """python.exe (con consola oculta) para los procesos de descarga, aunque la app corra con pythonw."""
    exe = Path(sys.executable)
    candidate = exe.with_name("python.exe")
    return str(candidate if candidate.is_file() else exe)


# ---------------------------------------------------------------------------
# Ajustes
# ---------------------------------------------------------------------------

def default_settings() -> dict[str, str]:
    orch.load_environment()  # el .env puede fijar SPOTIFY_PLAYLISTS_ROOT / SPOTIFY_DOWNLOAD_ROOT
    return {
        "library_root": str(Path(os.environ.get("SPOTIFY_PLAYLISTS_ROOT") or orch.DEFAULT_PLAYLISTS_DIR)),
        "runs_root": str(Path(os.environ.get("SPOTIFY_DOWNLOAD_ROOT") or orch.DEFAULT_RUNS_DIR)),
    }


def load_settings(path: Path) -> dict[str, str]:
    settings = default_settings()
    saved = read_json(path) or {}
    for key in settings:
        if isinstance(saved.get(key), str) and saved[key].strip():
            settings[key] = saved[key]
    return settings


# ---------------------------------------------------------------------------
# Estado de un run (lee los archivos que escribe el orquestador; no toca nada)
# ---------------------------------------------------------------------------

# Solo lo que la verificación comprobó: un ISRC distinto puede ser una reedición de la misma grabación
# y un artista "distinto" puede estar escrito de otra forma en la etiqueta («A & B»).
_REASONS = [
    (re.compile(r"^ISRC distinto"), "ISRC distinto al de Spotify"),
    (re.compile(r"^artista mismatch"), "el artista de la etiqueta no coincide"),
    (re.compile(r"^duraci.n mismatch \(([-+]?\d+(?:\.\d+)?)s\)"), "duración distinta ({0} s)"),
    (re.compile(r"^t.tulo (base )?mismatch|^title mismatch", re.I), "el título no coincide"),
    (re.compile(r"^identidad de versi.n", re.I), "otra versión o remix"),
]
_FAILURES = {
    "verification_failed": "El archivo de Soulseek no pasó la verificación",
    "download_error": "Falló la descarga",
    "availability_changed_or_download_failed": "Ya no estaba disponible en Soulseek",
    "finalize_error": "No se pudo guardar el MP3",
}


def friendly_failure(row: dict[str, Any]) -> str:
    base = _FAILURES.get(str(row.get("status")), "No se pudo bajar")
    reasons: list[str] = []
    for candidate in row.get("candidates") or []:
        if not isinstance(candidate, dict) or candidate.get("accepted"):
            continue
        for reason in candidate.get("reasons") or []:
            text = str(reason)
            for rx, label in _REASONS:
                m = rx.search(text)
                if m:
                    text = label.format(*(f"{float(g):+.0f}" for g in m.groups()))
                    break
            if text not in reasons:
                reasons.append(text)
    if row.get("error") and not reasons:
        reasons.append(str(row["error"])[:140])
    return base + (": " + ", ".join(reasons[:3]) if reasons else "")


def track_meta(spotify: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": spotify.get("spotify_id"),
        "title": spotify.get("title") or "",
        "artists": [str(a) for a in spotify.get("artists") or []],
        "album": spotify.get("album") or "",
        "cover": spotify.get("cover_url"),
        "duration_ms": int(spotify.get("duration_ms") or 0),
    }


def run_state(run_dir: Path, *, alive: bool | None = None, current_download: int | None = None,
              now: float | None = None) -> dict[str, Any]:
    """Resumen del run y estado de cada tema. `alive` None = adivinarlo por la fecha de los archivos."""
    now = time.time() if now is None else now
    report = read_json(run_dir / "orchestration_report.json") or {}
    raw_status = str(report.get("status") or "unknown")
    availability_files = json_candidates(run_dir, "soulseek_availability*.json")
    availability = read_json(availability_files[-1]) if availability_files else None
    missing = read_json(run_dir / "soulseek_missing.json")
    failed_list = read_json(run_dir / orch.FAILED_JSON_NAME)  # lo que Soulseek no pudo dar: se graba
    progress = read_json(run_dir / "recorded_missing" / "recording_progress.json")
    failed_progress = read_json(run_dir / orch.RECORDED_FAILED_DIR / "recording_progress.json")
    for archived in json_candidates(run_dir / "diagnostics", "*recording_progress.json"):
        # Run terminado: los dos checkpoints se archivan con nombres que dependen del orden;
        # se reconocen por la lista que grabaron.
        data = read_json(archived) or {}
        if Path(str(data.get("source_json") or "")).name == orch.FAILED_JSON_NAME:
            failed_progress = failed_progress or data
        else:
            progress = progress or data
    consolidation = read_json(run_dir / "consolidation_report.json")

    if alive is None:
        alive = raw_status in ACTIVE_STATUSES and now - newest_mtime(run_dir) < STALE_SECONDS
    if raw_status == "completed":
        status = "done"
    elif raw_status == "partial":
        status = "partial"
    elif raw_status == "resumable":
        status = "paused"
    elif raw_status == "failed":
        status = "failed"
    elif raw_status in ACTIVE_STATUSES or raw_status == "unknown":
        status = "running" if alive else "stopped"
    else:  # interrupted
        status = "stopped"
    if status == "running":
        phase = "consolidating" if raw_status == "consolidating" else ("transfer" if availability else "preflight")
    else:
        phase = None

    downloads: dict[int, dict[str, Any]] = {}
    for path in json_candidates(run_dir, "spotify_reconcile_report*.json"):  # en orden: el último manda
        for row in (read_json(path) or {}).get("tracks") or []:
            if isinstance(row, dict) and isinstance(row.get("position"), int):
                downloads[row["position"]] = row
    def positions(manifest: dict[str, Any] | None) -> list[int]:
        return [int(t["playlist_position"]) for t in (manifest or {}).get("tracks") or []
                if isinstance(t, dict) and isinstance(t.get("playlist_position"), int)]

    recorded = {int(t["playlist_position"]): t for p in (progress, failed_progress) for t in (p or {}).get("tracks") or []
                if isinstance(t, dict) and isinstance(t.get("playlist_position"), int)}
    missing_positions = positions(missing)
    failed_positions = positions(failed_list)
    completed_recordings = int((progress or {}).get("completed_count") or 0)
    completed_failed = int((failed_progress or {}).get("completed_count") or 0)
    consolidated = {str(r.get("spotify_id")): r for r in (consolidation or {}).get("tracks") or [] if isinstance(r, dict)}

    recording_now = None
    recording_since = None
    if phase == "transfer" and (run_dir / "recorder_ready.json").exists():
        if completed_recordings < len(missing_positions):
            recording_now = missing_positions[completed_recordings]
            folder = "recorded_missing"
        elif failed_list is not None and completed_failed < len(failed_positions):
            recording_now = failed_positions[completed_failed]
            folder = orch.RECORDED_FAILED_DIR
        if recording_now is not None:
            marks = [run_dir / "recorder_ready.json", run_dir / folder / "recording_progress.json"]
            recording_since = max((p.stat().st_mtime for p in marks if p.exists()), default=None)

    rows = (availability or {}).get("tracks") or []
    available_order = [int(r["position"]) for r in rows if isinstance(r, dict)
                       and r.get("status") in {"available", "available_quality_unknown"}
                       and isinstance(r.get("position"), int) and r["position"] not in missing_positions]
    pending_downloads = [p for p in available_order if p not in downloads]
    downloading_now = None
    if phase == "transfer" and pending_downloads:
        if current_download in pending_downloads:
            downloading_now = current_download
        elif (run_dir / "start_soulseek_downloads.signal").exists():
            downloading_now = pending_downloads[0]

    tracks: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("position"), int):
            continue
        pos = row["position"]
        meta = track_meta(row.get("spotify") or {})
        detail = None
        done = consolidated.get(str(meta["id"]))
        if done:
            state = "recorded" if done.get("origin") == "loopback" else "downloaded"
            detail = Path(str(done.get("destination") or "")).name or None
        elif pos in recorded:
            state = "recorded"
        elif pos in missing_positions:
            state = "recording" if pos == recording_now else "to_record"
        elif pos in downloads and downloads[pos].get("status") in DOWNLOADED:
            state = "downloaded"
            detail = Path(str(downloads[pos].get("output") or "")).name or None
        elif pos in failed_positions or (pos in downloads and status == "running"):
            # No se pudo bajar: el orquestador lo graba de Spotify (al terminar las descargas).
            state = "recording" if pos == recording_now else "to_record"
            reason = friendly_failure(downloads[pos]) if pos in downloads else "La búsqueda en Soulseek falló"
            detail = reason + ". Se graba de Spotify" + ("" if failed_list is not None else " al terminar las descargas")
        elif pos in downloads:
            state = "failed"
            detail = friendly_failure(downloads[pos])
        elif pos == downloading_now:
            state = "downloading"
        elif row.get("status") in {"available", "available_quality_unknown"}:
            state = "queued"
        elif status == "running":
            state = "to_record"
            detail = "La búsqueda en Soulseek falló. Se graba de Spotify al terminar las descargas"
        else:
            state = "search_error"
            detail = str(row.get("error") or "Error al buscar en Soulseek")
        tracks.append({"pos": pos, **meta, "state": state, "detail": detail})

    partial = bool((consolidation or {}).get("partial"))
    missing_ids = set((consolidation or {}).get("missing_spotify_ids") or [])
    if missing_ids:  # run incompleto ya cerrado: lo que no llegó a la carpeta
        for track in tracks:
            if track["id"] in missing_ids and track["state"] not in {"downloaded", "recorded"}:
                track["state"] = "failed"
                track["detail"] = "No se consiguió: " + str(report.get("reason") or "falló también la grabación")
    counts = collections.Counter(t["state"] for t in tracks)
    name = ((availability or {}).get("playlist") or {}).get("name") \
        or ((missing or {}).get("source_playlist") or {}).get("name") \
        or report.get("playlist_name") or (consolidation or {}).get("playlist_name")
    playlist_id = report.get("playlist_id") or (RUN_DIR_RE.match(run_dir.name) or [None, None, None, None])[3]
    return {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "playlist_id": playlist_id,
        "url": report.get("source_playlist") or (f"https://open.spotify.com/playlist/{playlist_id}" if playlist_id else None),
        "name": name,
        "status": status,
        "raw_status": raw_status,
        "phase": phase,
        "alive": bool(alive),
        "started": run_started(run_dir),
        "total": len(tracks),
        "counts": {k: counts.get(k, 0) for k in ("downloaded", "recorded", "failed", "queued", "downloading",
                                                  "to_record", "recording", "search_error")},
        "missing_count": len(missing_positions) if missing is not None else None,
        "recording_since": recording_since,
        "final_output": report.get("final_output"),
        "reason": report.get("reason"),
        "saved_to": consolidation.get("playlist_directory") if partial else None,  # guardado parcial
        "saved_count": consolidation.get("audio_count") if partial else None,
        "tracks": tracks,
    }


def list_runs(runs_root: Path) -> list[Path]:
    if not runs_root.is_dir():
        return []
    return sorted((p for p in runs_root.iterdir() if p.is_dir() and RUN_DIR_RE.match(p.name)),
                  key=lambda p: p.name, reverse=True)


def history(runs_root: Path, playlists: dict[str, Any], *, alive_run: str | None = None,
            limit: int = 40) -> list[dict[str, Any]]:
    """Último run de cada playlist, el más nuevo primero."""
    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    for run_dir in list_runs(runs_root):
        pid = RUN_DIR_RE.match(run_dir.name).group(3)
        if pid in seen:
            continue
        seen.add(pid)
        state = run_state(run_dir, alive=True if run_dir.name == alive_run else None)
        cached = playlists.get(pid) or {}
        first_cover = next((t["cover"] for t in state["tracks"] if t.get("cover")), None)
        items.append({
            "run": state["run"], "playlist_id": pid, "url": state["url"],
            "name": cached.get("name") or state["name"] or "Playlist",
            "cover": cached.get("cover") or first_cover, "status": state["status"], "alive": state["alive"],
            "started": state["started"], "total": state["total"],
            "done": state["counts"]["downloaded"] + state["counts"]["recorded"],
            "failed": state["counts"]["failed"], "final_output": state["final_output"],
        })
        if len(items) >= limit:
            break
    return items


# ---------------------------------------------------------------------------
# Procesos de Windows: todo lo que lanzó una descarga, para no dejar huérfanos (sin cerrar Spotify)
# ---------------------------------------------------------------------------
# Un Job Object no sirve: el Python de la Microsoft Store ya corre dentro del suyo. Se sigue el árbol
# por PID padre, guardando la hora de creación de cada proceso para no confundir un PID reutilizado.

def _kernel32():
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    return k32


class _ProcessEntry(ctypes.Structure):
    from ctypes import wintypes as _w
    _fields_ = [("dwSize", _w.DWORD), ("cntUsage", _w.DWORD), ("th32ProcessID", _w.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", _w.DWORD), ("cntThreads", _w.DWORD),
                ("th32ParentProcessID", _w.DWORD), ("pcPriClassBase", ctypes.c_long), ("dwFlags", _w.DWORD),
                ("szExeFile", ctypes.c_wchar * 260)]


def process_table() -> dict[int, tuple[int, str]]:
    """{pid: (pid del padre, nombre del ejecutable)} de todos los procesos."""
    if os.name != "nt":
        return {}
    k32 = _kernel32()
    k32.Process32FirstW.argtypes = k32.Process32NextW.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ProcessEntry)]
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    if not snap or snap == ctypes.c_void_p(-1).value:
        return {}
    table: dict[int, tuple[int, str]] = {}
    try:
        entry = _ProcessEntry()
        entry.dwSize = ctypes.sizeof(_ProcessEntry)
        ok = k32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            table[int(entry.th32ProcessID)] = (int(entry.th32ParentProcessID), entry.szExeFile)
            ok = k32.Process32NextW(snap, ctypes.byref(entry))
    finally:
        k32.CloseHandle(snap)
    return table


def process_created(pid: int) -> int | None:
    """Hora de creación (FILETIME) del proceso vivo, o None si ya no existe."""
    if os.name != "nt":
        return None
    from ctypes import wintypes
    k32 = _kernel32()
    handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        times = [wintypes.FILETIME() for _ in range(4)]
        if not k32.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            return None
        exit_code = wintypes.DWORD()
        k32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        if exit_code.value != 259:  # STILL_ACTIVE
            return None
        return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
    finally:
        k32.CloseHandle(handle)


def terminate_pid(pid: int) -> bool:
    if os.name != "nt":
        return False
    k32 = _kernel32()
    handle = k32.OpenProcess(0x0001, False, pid)  # PROCESS_TERMINATE
    if not handle:
        return False
    try:
        return bool(k32.TerminateProcess(handle, 1))
    finally:
        k32.CloseHandle(handle)


class ProcessTree:
    """Recuerda los descendientes de la descarga (orquestador, descargador, grabador, sockseek, ffmpeg)."""

    def __init__(self, root_pid: int):
        self.root = root_pid
        self.seen: dict[int, tuple[str, int | None]] = {}
        self.lock = threading.Lock()

    def scan(self) -> None:
        table = process_table()
        with self.lock:
            family = {self.root, *self.seen}
            grew = True
            while grew:
                grew = False
                for pid, (ppid, name) in table.items():
                    if ppid in family and pid not in family and pid != ppid:
                        family.add(pid)
                        self.seen[pid] = (name, process_created(pid))
                        grew = True

    def kill_leftovers(self, keep: Callable[[str], bool]) -> list[str]:
        """Cierra lo que siga vivo salvo lo que `keep(nombre)` conserve (Spotify)."""
        killed: list[str] = []
        with self.lock:
            items = list(self.seen.items())
        for pid, (name, created) in items:
            if keep(name) or created is None or process_created(pid) != created:
                continue  # se conserva, ya terminó o el PID ahora es de otro proceso
            if terminate_pid(pid):
                killed.append(name)
        return killed


def keep_spotify(name: str) -> bool:
    return name.lower() == "spotify.exe"


# ---------------------------------------------------------------------------
# Una descarga lanzada desde la app
# ---------------------------------------------------------------------------

_DOWNLOAD_LINE = re.compile(r"^\[(\d+)/(\d+) \| Spotify #(\d+)\]")


def build_job_command(stop_file: Path, url: str, settings: dict[str, str], fresh: bool) -> list[str]:
    cmd = [console_python(), "-u", str(APP_DIR / "run_job.py"), str(stop_file), url,
           "--output-root", settings["runs_root"], "--library-root", settings["library_root"]]
    return cmd + (["--fresh"] if fresh else [])


class DownloadJob:
    def __init__(self, url: str, settings: dict[str, str], fresh: bool):
        self.id = secrets.token_hex(4)
        self.url = orch.canonical_playlist_url(url)
        self.playlist_id = orch.playlist_id_from_url(url)
        self.settings = dict(settings)
        self.fresh = fresh
        self.lines: collections.deque[str] = collections.deque(maxlen=800)
        self.started = time.time()
        self.ended: float | None = None
        self.returncode: int | None = None
        self.run_dir: Path | None = None
        self.current_download: int | None = None
        self.error: str | None = None
        self.paused: str | None = None
        self.auth_url: str | None = None
        self.stop_requested = False
        self.stop_file = Path(tempfile.gettempdir()) / f"traktor_descargas_stop_{self.id}.signal"
        self.proc: subprocess.Popen[str] | None = None
        self.tree: ProcessTree | None = None
        self.killed: list[str] = []

    def start(self) -> None:
        Path(self.settings["runs_root"]).mkdir(parents=True, exist_ok=True)
        Path(self.settings["library_root"]).mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        self.proc = subprocess.Popen(
            build_job_command(self.stop_file, self.url, self.settings, self.fresh), cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", env=env, creationflags=CREATE_NO_WINDOW)
        self.tree = ProcessTree(self.proc.pid)
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._watch_tree, daemon=True).start()

    def _watch_tree(self) -> None:
        while self.alive and self.tree:
            self.tree.scan()
            time.sleep(2)

    def _cleanup(self) -> None:
        """Nada de la descarga queda corriendo al terminar (sockseek mantiene la sesión de Soulseek)."""
        if self.tree:
            self.tree.scan()
            self.killed += self.tree.kill_leftovers(keep_spotify)
            if self.killed:
                self.lines.append("Procesos que quedaban abiertos, cerrados: " + ", ".join(self.killed))

    def _read(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        expect_auth_url = False
        for raw in self.proc.stdout:
            line = raw.rstrip("\r\n")
            self.lines.append(line)
            text = line.strip()
            if expect_auth_url and text.startswith("https://"):
                self.auth_url, expect_auth_url = text, False
            if text.startswith("Abre esta URL si el navegador no se abre solo"):
                expect_auth_url = True
            elif text.startswith("Run      :") or text.startswith("Run :"):
                self.run_dir = Path(text.split(":", 1)[1].strip())
            elif text.startswith("ERROR:"):
                self.error = text[len("ERROR:"):].strip()
            elif text.startswith("PAUSADO:"):
                self.paused = text[len("PAUSADO:"):].strip()
            elif text.startswith("Descargas de Soulseek liberadas") or text.startswith("Señal recibida"):
                self.auth_url = None
            m = _DOWNLOAD_LINE.match(text)
            if m:
                self.current_download = int(m.group(3))
        returncode = self.proc.wait()
        self._cleanup()
        self.stop_file.unlink(missing_ok=True)
        self.ended = time.time()
        self.returncode = returncode

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.returncode is None

    def stop(self) -> None:
        """Como un Ctrl+C: el orquestador marca el run como interrumpido; si no termina en 20 s, se cierra todo."""
        self.stop_requested = True
        if self.tree:
            self.tree.scan()
        self.stop_file.write_text("stop\n", encoding="utf-8")

        def force() -> None:
            deadline = time.time() + 20
            while self.proc and self.proc.poll() is None and time.time() < deadline:
                time.sleep(0.5)
            if self.proc and self.proc.poll() is None:
                self._cleanup()
                self.proc.kill()
        threading.Thread(target=force, daemon=True).start()

    def public(self) -> dict[str, Any]:
        if self.alive:
            outcome = "running"
        elif self.returncode == 0:
            outcome = "done"
        elif self.returncode == 75:
            outcome = "paused"
        elif self.returncode == orch.EXIT_PARTIAL:
            outcome = "partial"
        elif self.returncode == 130 or self.stop_requested:
            outcome = "stopped"
        else:
            outcome = "failed"
        return {"id": self.id, "url": self.url, "playlist_id": self.playlist_id, "fresh": self.fresh,
                "alive": self.alive, "outcome": outcome, "returncode": self.returncode,
                "started": self.started, "ended": self.ended, "run": self.run_dir.name if self.run_dir else None,
                "error": self.error, "paused": self.paused, "auth_url": self.auth_url,
                "stop_requested": self.stop_requested, "settings": self.settings}


# ---------------------------------------------------------------------------
# Spotify: vista previa de la playlist con una autorización propia de la app
# ---------------------------------------------------------------------------

class NeedsAuth(Exception):
    pass


def http_get_json(url: str, timeout: float = 8.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (TRAKTOR ML descargas)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (URL fija de Spotify)
        data = json.loads(resp.read(2_000_000).decode("utf-8"))
    return data if isinstance(data, dict) else {}


def cover_color(url: str | None) -> str | None:
    """Color de fondo para el encabezado, sacado de la portada (como hace Spotify)."""
    if not url:
        return None
    host = urllib.parse.urlparse(url).hostname or ""
    if not url.startswith("https://") or not host.endswith(COVER_HOSTS):
        return None
    try:
        from PIL import Image
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:  # noqa: S310
            data = resp.read(4_000_000)
        img = Image.open(io.BytesIO(data)).convert("RGB").resize((24, 24))
        pixels = list(img.getdata())
    except Exception:  # noqa: BLE001 (sin color no pasa nada)
        return None
    best, weight_sum = [0.0, 0.0, 0.0], 0.0
    for r, g, b in pixels:
        h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
        w = 0.15 + s * (1 - abs(l - 0.5) * 2)  # prefiere colores vivos, ni negros ni blancos
        best = [best[0] + r * w, best[1] + g * w, best[2] + b * w]
        weight_sum += w
    r, g, b = (c / weight_sum / 255 for c in best)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    r, g, b = colorsys.hls_to_rgb(h, min(max(l, 0.30), 0.42), min(max(s, 0.35), 0.75))
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


class SpotifyPreview:
    def __init__(self, token_path: Path):
        self.token_path = token_path
        self.connecting = False
        self.connect_error: str | None = None
        self._client = None

    def client(self):
        if self._client is None:
            import spotify_soulseek_download as dl
            orch.load_environment()
            client_id = os.environ.get("SPOTIFY_CLIENT_ID") or ""
            self._client = dl.Spotify(client_id)
            self._client.cache = self.token_path  # no comparte el token del descargador
        return self._client

    def authorized(self) -> bool:
        token = read_json(self.token_path) or {}
        return bool(token.get("refresh_token"))

    def connect(self) -> None:
        if self.connecting:
            return
        self.connecting, self.connect_error = True, None

        def target() -> None:
            try:
                self.client().token()  # abre el navegador si hace falta
            except Exception as exc:  # noqa: BLE001
                self.connect_error = f"{type(exc).__name__}: {exc}"
            finally:
                self.connecting = False
        threading.Thread(target=target, daemon=True).start()

    def _ensure_token(self) -> None:
        sp = self.client()
        token = sp._token or sp._load() or {}
        if not token.get("refresh_token"):
            raise NeedsAuth()
        if float(token.get("expires_at", 0)) <= time.time():
            try:
                sp._token = sp._refresh(str(token["refresh_token"]))
            except Exception as exc:  # noqa: BLE001 (refresh vencido: hay que volver a conectar)
                raise NeedsAuth() from exc

    def playlist(self, playlist_id: str) -> dict[str, Any]:
        self._ensure_token()
        sp = self.client()
        try:
            info = sp._get(f"/playlists/{playlist_id}",
                           params={"fields": "id,name,description,images,owner(display_name),external_urls"})
        except Exception:  # noqa: BLE001 (algunas cuentas no aceptan fields)
            info = sp._get(f"/playlists/{playlist_id}")
        tracks: list[dict[str, Any]] = []
        url: str | None = f"/playlists/{playlist_id}/items"
        params: dict[str, Any] | None = {"limit": 50, "offset": 0, "additional_types": "track"}
        while url and len(tracks) < 2000:
            page = sp._get(url, params=params)
            params = None
            for item in page.get("items") or []:
                t = (item or {}).get("track") or (item or {}).get("item")
                if not isinstance(t, dict) or not t.get("id") or t.get("is_local"):
                    continue
                images = (t.get("album") or {}).get("images") or []
                small = min(images, key=lambda i: i.get("width") or 9999) if images else {}
                tracks.append({"pos": len(tracks) + 1, "id": t["id"], "title": t.get("name") or "",
                               "artists": [a.get("name") for a in t.get("artists") or [] if a.get("name")],
                               "album": (t.get("album") or {}).get("name") or "", "cover": small.get("url"),
                               "duration_ms": int(t.get("duration_ms") or 0), "explicit": bool(t.get("explicit"))})
            url = page.get("next")
        images = info.get("images") or []
        return {"name": info.get("name"), "description": re.sub(r"<[^>]+>", "", info.get("description") or ""),
                "owner": (info.get("owner") or {}).get("display_name"),
                "cover": images[0].get("url") if images else None, "tracks": tracks}


def oembed(playlist_url: str) -> dict[str, Any]:
    """Nombre y portada sin autorización (solo playlists públicas)."""
    try:
        data = http_get_json("https://open.spotify.com/oembed?url=" + urllib.parse.quote(playlist_url, safe=""))
    except Exception:  # noqa: BLE001
        return {}
    return {"name": data.get("title"), "cover": data.get("thumbnail_url")}


# ---------------------------------------------------------------------------
# Audio: salidas de Windows, prueba con tono y medidor del loopback
# ---------------------------------------------------------------------------

def run_audio_script(*args: str) -> list[dict[str, Any]]:
    out = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(APP_DIR / "audio_devices.ps1"), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        creationflags=CREATE_NO_WINDOW)
    if out.returncode != 0:
        raise RuntimeError((out.stderr or out.stdout or "No se pudo leer el audio de Windows").strip()[-400:])
    data = json.loads(out.stdout.strip() or "[]")
    return data if isinstance(data, list) else [data]


def _chime(rate: int, channels: int) -> bytes:
    """Dos notas suaves (ding-dong) de 0,6 s cada una, a -12 dBFS."""
    import numpy as np
    samples = []
    for freq in (880.0, 659.25):
        t = np.arange(int(rate * 0.6)) / rate
        env = np.minimum(1, t / 0.01) * np.exp(-t * 5)
        samples.append(np.sin(2 * math.pi * freq * t) * env)
    mono = (np.concatenate(samples) * 0.25).astype(np.float32)
    frames = np.zeros((mono.size, channels), dtype=np.float32)
    frames[:, : min(2, channels)] = mono[:, None]
    return frames.tobytes()


class AudioMonitor:
    """Escucha el loopback de la salida predeterminada y guarda el nivel del último bloque."""

    def __init__(self):
        self.lock = threading.Lock()
        self.until = 0.0
        self.db = -90.0
        self.peak_db = -90.0
        self.last_block = 0.0
        self.device: str | None = None
        self.error: str | None = None
        self.thread: threading.Thread | None = None

    def level(self) -> dict[str, Any]:
        active = time.time() < self.until and self.thread is not None and self.thread.is_alive()
        db = self.db if time.time() - self.last_block < 0.4 else -90.0
        return {"active": active, "db": round(db, 1), "peak_db": round(self.peak_db, 1),
                "device": self.device, "error": self.error, "remaining": max(0, int(self.until - time.time()))}

    def start(self, seconds: float = 25) -> None:
        with self.lock:
            self.until = time.time() + seconds
            self.peak_db, self.error = -90.0, None
            if self.thread and self.thread.is_alive():
                return
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self) -> None:
        self.until = 0.0

    def _run(self) -> None:
        try:
            import numpy as np
            import pyaudiowpatch as pyaudio
            p = pyaudio.PyAudio()
            try:
                loop = p.get_default_wasapi_loopback()
                self.device = str(loop["name"]).replace(" [Loopback]", "")
                stream = p.open(format=pyaudio.paInt16, channels=int(loop["maxInputChannels"]),
                                rate=int(loop["defaultSampleRate"]), input=True,
                                input_device_index=int(loop["index"]), frames_per_buffer=2048)
                try:
                    while time.time() < self.until:
                        if stream.get_read_available() <= 0:
                            time.sleep(0.03)
                            continue
                        block = np.frombuffer(stream.read(min(stream.get_read_available(), 4096),
                                                          exception_on_overflow=False), dtype=np.int16)
                        if block.size:
                            rms = float(np.sqrt(np.mean((block.astype(np.float32) / 32768) ** 2)))
                            self.db = 20 * math.log10(max(rms, 1e-6))
                            self.peak_db = max(self.peak_db, self.db)
                            self.last_block = time.time()
                finally:
                    stream.close()
            finally:
                p.terminate()
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"


def audio_test() -> dict[str, Any]:
    """Suena un tono por la salida predeterminada y lo graba por su loopback, igual que el grabador."""
    import numpy as np
    import pyaudiowpatch as pyaudio
    p = pyaudio.PyAudio()
    try:
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        out = p.get_device_info_by_index(int(wasapi["defaultOutputDevice"]))
        loop = p.get_default_wasapi_loopback()
        captured: list[Any] = []

        def on_block(data, _frames, _info, _status):
            captured.append(np.frombuffer(data, dtype=np.int16).copy())
            return (None, pyaudio.paContinue)
        rec = p.open(format=pyaudio.paInt16, channels=int(loop["maxInputChannels"]),
                     rate=int(loop["defaultSampleRate"]), input=True, input_device_index=int(loop["index"]),
                     frames_per_buffer=1024, stream_callback=on_block)
        channels = int(out["maxOutputChannels"]) or 2
        rate = int(out["defaultSampleRate"])
        play = p.open(format=pyaudio.paFloat32, channels=channels, rate=rate, output=True,
                      output_device_index=int(out["index"]))
        try:
            play.write(_chime(rate, channels))
            time.sleep(0.3)
        finally:
            play.stop_stream()
            play.close()
            rec.stop_stream()
            rec.close()
        audio = np.concatenate(captured) if captured else np.zeros(1, dtype=np.int16)
        peak = float(np.max(np.abs(audio.astype(np.float32)))) / 32768
        peak_db = 20 * math.log10(max(peak, 1e-6))
        stereo = int(loop["maxInputChannels"]) >= 2
        return {"device": out["name"], "peak_db": round(peak_db, 1), "recorded": peak_db > -45, "stereo": stereo}
    finally:
        p.terminate()


# ---------------------------------------------------------------------------
# Comprobaciones del entorno (sin mostrar ninguna clave)
# ---------------------------------------------------------------------------

def soulseek_account_ok() -> bool:
    paths = [Path.home() / ".config/sockseek/sockseek.conf"]
    if os.environ.get("APPDATA"):
        paths.append(Path(os.environ["APPDATA"]) / "sockseek/sockseek.conf")
    for path in paths:
        if not path.is_file():
            continue
        found: set[str] = set()
        for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            stripped = line.strip()
            if stripped.startswith("["):
                break
            key, sep, value = stripped.partition("=")
            if sep and key.strip().lower() in {"username", "password"} and value.strip():
                found.add(key.strip().lower())
        return found == {"username", "password"}
    return False


def program_ok(name: str) -> bool:
    try:
        orch.find_required_program(name)
        return True
    except orch.WorkflowError:
        return False


def process_running(image: str) -> bool:
    if os.name != "nt":
        return False
    out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {image}", "/FO", "CSV", "/NH"], capture_output=True,
                         text=True, errors="replace", creationflags=CREATE_NO_WINDOW)
    return image.lower() in out.stdout.lower()


def antivirus_ok() -> bool | None:
    if os.name != "nt":
        return None
    out = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                          "(Get-MpComputerStatus).AntivirusEnabled"], capture_output=True, text=True,
                         errors="replace", timeout=30, creationflags=CREATE_NO_WINDOW)
    text = out.stdout.strip().lower()
    return True if text == "true" else (False if text == "false" else None)


def environment_checks(preview: SpotifyPreview) -> list[dict[str, Any]]:
    orch.load_environment()
    keys = all(os.environ.get(k) for k in ("SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET", "SPOTIFY_REDIRECT_URI"))
    av = antivirus_ok()
    return [
        {"key": "keys", "label": "Claves de Spotify (.env)", "ok": keys,
         "fix": "Copia el .env a la raíz del repo (CONFIGURAR_OTRO_ORDENADOR.md, paso 6a)."},
        {"key": "soulseek", "label": "Cuenta de Soulseek", "ok": soulseek_account_ok(),
         "fix": "Copia sockseek.conf a %APPDATA%\\sockseek (paso 5)."},
        {"key": "sockseek", "label": "sockseek", "ok": program_ok("sockseek"),
         "fix": "Descarga sockseek 3.0.5 en tools\\sockseek (paso 4)."},
        {"key": "ffmpeg", "label": "ffmpeg", "ok": program_ok("ffmpeg"),
         "fix": "winget install -e --id Gyan.FFmpeg (paso 1)."},
        {"key": "antivirus", "label": "Antivirus (Defender)", "ok": av,
         "fix": "Activa Microsoft Defender: sin antivirus no se descarga nada (paso 9)."},
        {"key": "playback", "label": "Spotify autorizado para reproducir", "ok": (REPO_ROOT / ".spotify_token_cache").is_file(),
         "fix": "Haz el paso 8a una vez: spotify_playlist.py <playlist> --start-only."},
        {"key": "preview", "label": "Vista previa de playlists", "ok": preview.authorized(),
         "fix": "Pulsa «Conectar» para ver los temas antes de bajar."},
        {"key": "spotify_app", "label": "Spotify de escritorio abierto", "ok": process_running("Spotify.exe"),
         "fix": "Abre Spotify con tu cuenta Premium antes de grabar."},
    ]


# ---------------------------------------------------------------------------
# Estado de la app
# ---------------------------------------------------------------------------

class App:
    def __init__(self, config_dir: Path = CONFIG_DIR):
        self.config_dir = config_dir
        self.settings_path = config_dir / "settings.json"
        self.playlists_path = config_dir / "playlists.json"
        self.settings = load_settings(self.settings_path)
        self.token = secrets.token_urlsafe(24)
        self.lock = threading.Lock()
        self.job: DownloadJob | None = None
        self.preview = SpotifyPreview(config_dir / "spotify_ui_token.json")
        self.monitor = AudioMonitor()
        self.checks: list[dict[str, Any]] = []
        self.checks_at = 0.0
        self.port = 0
        self.last_seen = time.time()
        self.saving: dict[str, Any] | None = None
        self.playlists: dict[str, Any] = read_json(self.playlists_path) or {}

    # ---- ajustes y carpetas ----
    def runs_root(self) -> Path:
        return Path(self.settings["runs_root"]).expanduser()

    def library_root(self) -> Path:
        return Path(self.settings["library_root"]).expanduser()

    def update_settings(self, changes: dict[str, Any]) -> dict[str, str]:
        with self.lock:
            if self.job and self.job.alive:
                raise RuntimeError("Espera a que termine la descarga para cambiar las carpetas.")
            for key in ("library_root", "runs_root"):
                if key in changes:
                    path = Path(str(changes[key])).expanduser()
                    if not path.is_absolute():
                        raise ValueError("Elige una carpeta completa, por ejemplo D:\\Música\\Descargas.")
                    path.mkdir(parents=True, exist_ok=True)
                    self.settings[key] = str(path.resolve())
            write_json(self.settings_path, self.settings)
        return self.settings

    def pick_folder(self, which: str) -> dict[str, Any]:
        if which not in ("library_root", "runs_root"):
            raise ValueError("Carpeta desconocida")
        title = ("Elige dónde guardar la música descargada" if which == "library_root"
                 else "Elige la carpeta de trabajo (temporales de cada descarga)")
        code = ("import sys, tkinter as tk; from tkinter import filedialog; r = tk.Tk(); r.withdraw(); "
                "r.attributes('-topmost', True); "
                "p = filedialog.askdirectory(initialdir=sys.argv[1], title=sys.argv[2], mustexist=False); "
                "print(p or '')")
        initial = self.settings[which] if Path(self.settings[which]).exists() else str(Path.home())
        out = subprocess.run([console_python(), "-c", code, initial, title], capture_output=True, text=True,
                             encoding="utf-8", creationflags=CREATE_NO_WINDOW).stdout.strip()
        if not out:
            return {"cancelled": True, "settings": self.settings}
        return {"cancelled": False, "settings": self.update_settings({which: out})}

    def open_folder(self, path: str) -> None:
        target = Path(path).expanduser()
        roots = [self.library_root(), self.runs_root()]
        if not any(is_inside(target, root) for root in roots):
            raise ValueError("Solo se abren las carpetas de descargas.")
        if not target.exists():
            raise ValueError("La carpeta ya no existe.")
        os.startfile(str(target))  # noqa: S606 (Explorador de Windows)

    def disk(self) -> dict[str, Any]:
        root = self.library_root()
        probe = next((p for p in [root, *root.parents] if p.exists()), None)
        if probe is None:
            return {}
        usage = shutil.disk_usage(probe)
        return {"free_gb": round(usage.free / 2**30, 1), "total_gb": round(usage.total / 2**30, 1)}

    # ---- runs ----
    def live_external(self) -> dict[str, Any] | None:
        """Una descarga en curso que no lanzó esta app (por ejemplo, desde la terminal)."""
        own = self.job.run_dir.name if self.job and self.job.run_dir else None
        for run_dir in list_runs(self.runs_root())[:6]:
            if run_dir.name == own:
                continue
            report = read_json(run_dir / "orchestration_report.json") or {}
            if report.get("status") in ACTIVE_STATUSES and time.time() - newest_mtime(run_dir) < STALE_SECONDS:
                return {"run": run_dir.name, "playlist_id": report.get("playlist_id")}
        return None

    def busy_reason(self) -> str | None:
        if self.job and self.job.alive:
            return "Hay una descarga en curso."
        if self.live_external():
            return "Hay una descarga en curso lanzada desde la terminal."
        return None

    def run_view(self, run_name: str) -> dict[str, Any]:
        if not RUN_DIR_RE.match(run_name):
            raise ValueError("Run inválido")
        run_dir = self.runs_root() / run_name
        if not run_dir.is_dir():
            raise ValueError("Ese run ya no existe")
        job = self.job if self.job and self.job.run_dir and self.job.run_dir.name == run_name else None
        state = run_state(run_dir, alive=job.alive if job else None,
                          current_download=job.current_download if job else None)
        if job:
            state["job"] = job.public()
        state["playlist"] = self.playlists.get(str(state["playlist_id"])) or {}
        return state

    def history(self) -> list[dict[str, Any]]:
        alive_run = self.job.run_dir.name if self.job and self.job.alive and self.job.run_dir else None
        return history(self.runs_root(), self.playlists, alive_run=alive_run)

    def remember_playlist(self, playlist_id: str, meta: dict[str, Any]) -> None:
        entry = {k: meta.get(k) for k in ("name", "cover", "color", "owner") if meta.get(k)}
        if entry:
            self.playlists[playlist_id] = {**self.playlists.get(playlist_id, {}), **entry}
            write_json(self.playlists_path, self.playlists)

    # ---- acciones ----
    def preview_playlist(self, url: str) -> dict[str, Any]:
        try:
            playlist_id = orch.playlist_id_from_url(url)
        except orch.WorkflowError as exc:
            raise ValueError("Eso no parece un enlace de playlist de Spotify.") from exc
        canonical = orch.canonical_playlist_url(url)
        result: dict[str, Any] = {"playlist_id": playlist_id, "url": canonical, "needs_auth": False, "tracks": None}
        meta = dict(self.playlists.get(playlist_id) or {})
        try:
            full = self.preview.playlist(playlist_id)
            meta.update({k: v for k, v in full.items() if k != "tracks" and v})
            result["tracks"] = full["tracks"]
        except NeedsAuth:
            result["needs_auth"] = True
            meta.update({k: v for k, v in oembed(canonical).items() if v})
        except Exception as exc:  # noqa: BLE001
            result["error"] = f"Spotify no devolvió la playlist ({type(exc).__name__}). ¿Es privada o el enlace está mal?"
            meta.update({k: v for k, v in oembed(canonical).items() if v})
        if meta.get("cover") and not meta.get("color"):
            meta["color"] = cover_color(meta["cover"])
        self.remember_playlist(playlist_id, meta)
        result["meta"] = meta
        latest = next((r for r in list_runs(self.runs_root()) if r.name.endswith("_" + playlist_id)), None)
        result["latest_run"] = self.run_view(latest.name) if latest else None
        return result

    def start(self, url: str, fresh: bool) -> dict[str, Any]:
        with self.lock:
            reason = self.busy_reason()
            if reason:
                raise RuntimeError(reason)
            missing = [c["label"] for c in self.checks_now() if c["key"] in {"keys", "soulseek", "sockseek", "ffmpeg"}
                       and not c["ok"]]
            if missing:
                raise RuntimeError("Falta configurar: " + ", ".join(missing) + ".")
            self.monitor.stop()
            job = DownloadJob(url, self.settings, fresh)
            job.start()
            self.job = job
        return job.public()

    def stop(self) -> dict[str, Any]:
        if not (self.job and self.job.alive):
            raise RuntimeError("No hay ninguna descarga de la app en curso.")
        self.job.stop()
        return self.job.public()

    def save_partial(self, run_name: str) -> dict[str, Any]:
        """Run que terminó con error: lleva a la carpeta los MP3 que sí están, con la validación del orquestador
        (audio, tamaño, antivirus, Spotify Track ID) y sin duplicar lo que ya esté en la carpeta."""
        if not RUN_DIR_RE.match(run_name):
            raise ValueError("Run inválido")
        run_dir = self.runs_root() / run_name
        with self.lock:
            if self.saving and self.saving.get("status") == "running":
                raise RuntimeError("Ya se están guardando temas.")
            if self.busy_reason():
                raise RuntimeError("Espera a que termine la descarga en curso.")
            state = run_state(run_dir, alive=False)
            if state["status"] != "failed":
                raise ValueError("Esto solo es para descargas que terminaron con error.")
            if state["saved_to"]:
                raise ValueError("Esos temas ya se guardaron.")
            self.saving = {"run": run_name, "status": "running"}
        name, library = str(state["name"] or state["playlist_id"]), self.library_root()

        def target() -> None:
            try:
                folder, _moved, report = orch.consolidate_audio(
                    run_dir, run_dir / "soulseek", run_dir / "recorded_missing", library, name,
                    expected_spotify_ids=None, extra_recorded_dirs=(run_dir / orch.RECORDED_FAILED_DIR,))
                report["partial"] = True
                write_json(run_dir / "consolidation_report.json", report)
                self.saving = {"run": run_name, "status": "done", "folder": str(folder), "count": report["audio_count"]}
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self.saving = {"run": run_name, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        threading.Thread(target=target, daemon=True).start()
        return self.saving

    def checks_now(self, refresh: bool = False) -> list[dict[str, Any]]:
        if refresh or not self.checks or time.time() - self.checks_at > 60:
            self.checks = environment_checks(self.preview)
            self.checks_at = time.time()
        return self.checks

    def devices(self) -> list[dict[str, Any]]:
        return run_audio_script("-List")

    def set_device(self, device_id: str) -> list[dict[str, Any]]:
        reason = self.busy_reason()
        if reason:
            raise RuntimeError(reason + " Cambiar la salida ahora cortaría la grabación.")
        return run_audio_script("-SetDefault", device_id)

    def test_audio(self) -> dict[str, Any]:
        reason = self.busy_reason()
        if reason:
            raise RuntimeError(reason + " El tono de prueba quedaría grabado.")
        return audio_test()

    def state(self) -> dict[str, Any]:
        self.last_seen = time.time()
        return {"settings": self.settings, "disk": self.disk(), "job": self.job.public() if self.job else None,
                "external": self.live_external(), "preview_auth": self.preview.authorized(),
                "connecting": self.preview.connecting, "connect_error": self.preview.connect_error,
                "monitor": self.monitor.level(), "saving": self.saving}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def make_handler(app: App):
    page_template = (APP_DIR / "index.html").read_text(encoding="utf-8")

    class Handler(BaseHTTPRequestHandler):
        server_version = "TraktorDescargas/1"

        def log_message(self, fmt, *args):  # silencioso salvo errores
            if len(args) > 1 and str(args[1]).startswith(("4", "5")):
                sys.stderr.write("[HTTP] " + fmt % args + "\n")

        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").lower()
            return host in (f"127.0.0.1:{app.port}", f"localhost:{app.port}")

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: Any) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):  # noqa: N802
            if not self._host_ok():
                return self._json(403, {"error": "host"})
            parsed = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed.query)
            path = parsed.path
            try:
                if path in ("/", "/index.html"):
                    app.last_seen = time.time()
                    body = page_template.replace("__APP_TOKEN__", app.token).encode("utf-8")
                    return self._send(200, body, "text/html; charset=utf-8")
                if path == "/icon.svg":
                    return self._send(200, (APP_DIR / "icon.svg").read_bytes(), "image/svg+xml")
                if path == "/api/ping":
                    return self._json(200, {"app": APP_ID})
                if path == "/api/state":
                    return self._json(200, app.state())
                if path == "/api/history":
                    return self._json(200, {"items": app.history()})
                if path == "/api/run":
                    return self._json(200, app.run_view(query.get("id", [""])[0]))
                if path == "/api/log":
                    lines = list(app.job.lines)[-500:] if app.job else []
                    return self._json(200, {"lines": lines})
                if path == "/api/checks":
                    return self._json(200, {"checks": app.checks_now(refresh="refresh" in query)})
                if path == "/api/audio/devices":
                    return self._json(200, {"devices": app.devices(), "busy": app.busy_reason()})
                if path == "/api/audio/level":
                    return self._json(200, app.monitor.level())
            except (ValueError, RuntimeError) as exc:
                return self._json(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return self._json(404, {"error": "no existe"})

        def do_POST(self):  # noqa: N802
            origin = self.headers.get("Origin")
            if (not self._host_ok() or self.headers.get("X-App-Token") != app.token
                    or (origin and origin not in (f"http://127.0.0.1:{app.port}", f"http://localhost:{app.port}"))):
                return self._json(403, {"error": "No autorizado"})
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return self._json(400, {"error": "JSON inválido"})
            path = urllib.parse.urlparse(self.path).path
            app.last_seen = time.time()
            try:
                if path == "/api/preview":
                    return self._json(200, app.preview_playlist(str(body.get("url") or "")))
                if path == "/api/start":
                    return self._json(200, app.start(str(body.get("url") or ""), bool(body.get("fresh"))))
                if path == "/api/stop":
                    return self._json(200, app.stop())
                if path == "/api/save-partial":
                    return self._json(200, app.save_partial(str(body.get("run") or "")))
                if path == "/api/settings":
                    return self._json(200, {"settings": app.update_settings(body)})
                if path == "/api/pick-folder":
                    return self._json(200, app.pick_folder(str(body.get("which") or "")))
                if path == "/api/open-folder":
                    app.open_folder(str(body.get("path") or ""))
                    return self._json(200, {"ok": True})
                if path == "/api/spotify/connect":
                    app.preview.connect()
                    return self._json(200, {"ok": True})
                if path == "/api/audio/default":
                    return self._json(200, {"devices": app.set_device(str(body.get("id") or ""))})
                if path == "/api/audio/test":
                    return self._json(200, app.test_audio())
                if path == "/api/audio/monitor":
                    if body.get("stop"):
                        app.monitor.stop()
                    else:
                        reason = app.busy_reason()
                        if reason:
                            raise RuntimeError(reason)
                        app.monitor.start()
                    return self._json(200, app.monitor.level())
            except (ValueError, RuntimeError, KeyError, orch.WorkflowError) as exc:
                return self._json(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return self._json(404, {"error": "no existe"})
    return Handler


def free_port(preferred: int) -> int:
    for port in [preferred, *range(preferred + 1, preferred + 30), 0]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("No hay puertos libres")


def open_window(url: str) -> None:
    """Ventana propia sin barra del navegador (Edge o Chrome en modo app); si no, el navegador."""
    candidates = [
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
    ]
    for exe in candidates:
        if exe.is_file():
            subprocess.Popen([str(exe), f"--app={url}", "--window-size=1340,900"], creationflags=CREATE_NO_WINDOW)
            return
    webbrowser.open(url)


def running_instance(info_path: Path) -> str | None:
    info = read_json(info_path) or {}
    port = info.get("port")
    if not isinstance(port, int):
        return None
    try:
        data = http_get_json(f"http://127.0.0.1:{port}/api/ping", timeout=1.5)
    except Exception:  # noqa: BLE001
        return None
    return f"http://127.0.0.1:{port}/" if data.get("app") == APP_ID else None


def redirect_output(log_path: Path) -> None:
    """Con pythonw no hay consola: los mensajes van a un log en %APPDATA%."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() and log_path.stat().st_size > 2_000_000:
        log_path.unlink()
    stream = open(log_path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115 (vive lo que vive la app)
    sys.stdout = sys.stderr = stream


def main() -> int:
    parser = argparse.ArgumentParser(description="App local para bajar playlists de Spotify")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR,
                        help="Ajustes, caché y token de la vista previa (para pruebas: otra carpeta)")
    args = parser.parse_args()
    redirect_output(args.config_dir / "app.log")
    info_path = args.config_dir / "server.json"

    existing = running_instance(info_path)
    if existing:
        if not args.no_window:
            open_window(existing)
        return 0

    app = App(args.config_dir)
    app.port = free_port(args.port)
    httpd = ThreadingHTTPServer(("127.0.0.1", app.port), make_handler(app))
    url = f"http://127.0.0.1:{app.port}/"
    write_json(info_path, {"port": app.port, "pid": os.getpid()})
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Descargas abierto en {url}", flush=True)
    threading.Thread(target=app.checks_now, daemon=True).start()
    if not args.no_window:
        threading.Timer(0.3, open_window, args=(url,)).start()

    def idle_watchdog() -> None:
        while True:
            time.sleep(10)
            busy = (app.job and app.job.alive) or app.monitor.level()["active"]
            if not busy and time.time() - app.last_seen > IDLE_SHUTDOWN_SECONDS:
                print(f"[{datetime.now():%H:%M:%S}] Sin ventana abierta: cerrando.", flush=True)
                httpd.shutdown()
                return
    threading.Thread(target=idle_watchdog, daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        if (read_json(info_path) or {}).get("pid") == os.getpid():
            info_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
