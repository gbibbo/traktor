"""
PURPOSE: Send Spotify alone to the output that is recorded (e.g. «Voicemeeter Input») while a download
         records, and give it back the output it had once the download ends, without the user doing
         anything. Windows keeps one output per app (Settings → Sound → Volume mixer); it is read and set
         through app/audio_devices.ps1 -App / -SetApp. The output Spotify had is saved in
         <runs>/spotify_output_restore.json before the first change, so a run that was cut off still gives
         it back at the end of the next one. Windows only changes it while Spotify is open: with Spotify
         closed the file stays for the next time.

CHANGELOG:
- 2026-11-07: Creation (Gabriel: Voicemeeter for Spotify is chosen on its own when recording starts and
              the previous output comes back when it ends).
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve().parent / "app" / "audio_devices.ps1"
STATE_NAME = "spotify_output_restore.json"
APP = "Spotify"
NOT_RUNNING_EXIT = 3  # audio_devices.ps1: the app is not open


class AppNotRunning(RuntimeError):
    pass


def _call(*args: str) -> dict[str, Any]:
    cmd = ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
           "-File", str(SCRIPT), *args]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=60, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    error = (result.stderr or "").strip().splitlines()
    if result.returncode == NOT_RUNNING_EXIT:
        raise AppNotRunning(error[-1] if error else "La app no está abierta.")
    if result.returncode != 0:
        raise RuntimeError(error[-1] if error else f"audio_devices.ps1 terminó con código {result.returncode}")
    lines = (result.stdout or "").strip().splitlines()
    return json.loads(lines[-1]) if lines else {}


def current(app: str = APP) -> dict[str, Any]:
    """{app, pids, device_id, device}: device_id "" is Windows' default output."""
    return _call("-App", app)


def _read_state(path: Path) -> dict[str, Any] | None:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return state if isinstance(state, dict) else None


def route(target: str, state_path: Path, app: str = APP) -> dict[str, Any]:
    """Send the app to `target` (a name or part of it). The first time, save the output it had, to give it
    back with restore(): later sessions of the same download (or a run after a cut-off one) keep that one."""
    result = _call("-SetApp", app, "-Device", target)
    if _read_state(state_path) is None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = state_path.with_name(state_path.name + ".tmp")
        temporary.write_text(json.dumps({
            "app": app,
            "previous_id": result.get("previous_id") or "",
            "previous": result.get("previous") or "",
            "routed_id": result.get("device_id") or "",
            "routed": result.get("device") or "",
            "at": datetime.now().isoformat(timespec="seconds"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(state_path)
    return result


def restore(state_path: Path) -> str | None:
    """Give the app back the output it had before route(). Returns what happened, for the log, or None when
    there was nothing to give back. If the user chose another output in the meantime, that one stays."""
    state = _read_state(state_path)
    if state is None:
        return None
    app = str(state.get("app") or APP)
    previous = str(state.get("previous") or "") or "la salida predeterminada de Windows"
    try:
        now = current(app)
        if (now.get("device_id") or "") != (state.get("routed_id") or ""):
            state_path.unlink(missing_ok=True)
            return None
        _call("-SetApp", app, "-Device", str(state.get("previous_id") or "") or "default")
    except AppNotRunning:
        return f"{app} está cerrado: le devuelvo su salida («{previous}») la próxima vez."
    except Exception as exc:  # noqa: BLE001 (the download is over: it only stays routed)
        return f"No pude devolver {app} a «{previous}» ({exc}): queda saliendo por «{state.get('routed')}»."
    state_path.unlink(missing_ok=True)
    return f"{app} vuelve a sonar por «{previous}»."
