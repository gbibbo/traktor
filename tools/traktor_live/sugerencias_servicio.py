"""
PURPOSE: Sugerencias en vivo sin abrir nada a mano. Un proceso sin ventana arranca al iniciar sesión
         (Windows: acceso directo a pythonw en la carpeta Inicio; macOS: un LaunchAgent) y espera. Cuando
         se abre Traktor, empieza una sesión de sugerencias_vivo.run_session (receptor de la sonda,
         controlador MIDI, botón). Cuando Traktor se cierra, la termina y suelta el controlador.
         Avisa con una notificación del sistema, siempre sin sonido para que nada salga por el equipo de
         sonido: cuando queda listo y cuando hay un problema. Todo queda en
         artifacts/v4/traktor_live/servicio.log.
         Subcomandos: instalar (deja el inicio automático y lo arranca ya), quitar, estado y correr (lo
         que corre el inicio automático; también sirve para probarlo en primer plano).
CHANGELOG:
  - 2026-11-06: Creación (Gabriel: que no haga falta correr un script cada vez que abre Traktor).
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import os
import plistlib
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from src.v4.common.traktor_nml import traktor_running  # noqa: E402
from tools.traktor_live import sugerencias_vivo as sv  # noqa: E402
from tools.traktor_live import traktor_estado as te  # noqa: E402

NAME = "TRAKTOR ML Sugerencias"
LOG_FILE = te.DEFAULT_LOG_DIR / "servicio.log"
PID_FILE = te.DEFAULT_LOG_DIR / "servicio.pid"
STOP_FILE = te.DEFAULT_LOG_DIR / "servicio.stop"
MAC_LABEL = "com.traktorml.sugerencias"
MAC_PLIST = Path.home() / "Library" / "LaunchAgents" / f"{MAC_LABEL}.plist"
POLL_S = 2.0                    # cada cuánto se mira si Traktor está abierto
RETRY_S = 30.0                  # si una sesión no pudo empezar (puerto ocupado...), cuándo reintentar
NOTIFY_REPEAT_S = 120.0         # el mismo aviso no se repite antes de esto
LOG_MAX_BYTES = 1_000_000
CREATE_NO_WINDOW = 0x08000000
DETACHED_PROCESS = 0x00000008
# AppUserModelID de PowerShell: Windows muestra la notificación a su nombre (no hace falta registrar una app)
POWERSHELL_AUMID = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"


# ---------------------------------------------------------------- notificaciones y registro

def xml_escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&apos;"))


def toast_xml(title: str, text: str) -> str:
    """Notificación de Windows, sin sonido: el sonido de sistema podría salir por la placa del DJ."""
    return ("<toast><visual><binding template='ToastGeneric'>"
            f"<text>{xml_escape(title)}</text><text>{xml_escape(text)}</text>"
            "</binding></visual><audio silent='true'/></toast>")


def toast_command(title: str, text: str) -> List[str]:
    script = (
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null\n"
        "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null\n"
        "$x = New-Object Windows.Data.Xml.Dom.XmlDocument\n"
        f"$x.LoadXml('{toast_xml(title, text).replace(chr(39), chr(39) * 2)}')\n"      # ' -> '' en PowerShell
        "$t = [Windows.UI.Notifications.ToastNotification]::new($x)\n"
        f"[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('{POWERSHELL_AUMID}').Show($t)\n"
    )
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return ["powershell.exe", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-EncodedCommand", encoded]


def mac_notify_command(title: str, text: str) -> List[str]:
    def q(s: str) -> str:
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return ["osascript", "-e", f"display notification {q(text)} with title {q(title)}"]


def notify(title: str, text: str) -> None:
    """Notificación del sistema, sin esperar y sin abrir ventanas. Si falla, solo queda en el registro."""
    try:
        if sys.platform == "win32":
            subprocess.Popen(toast_command(title, text), creationflags=CREATE_NO_WINDOW,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform == "darwin":
            subprocess.Popen(mac_notify_command(title, text), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass


class ServiceSay:
    """El say() del servicio: cada línea al registro con la hora; los avisos, los errores y el «Listo»
    también como notificación, sin repetir el mismo texto antes de NOTIFY_REPEAT_S."""

    def __init__(self, notifier: Callable[[str, str], None] = notify, clock=time.monotonic):
        self.notifier, self.clock = notifier, clock
        self._sent = {}

    def __call__(self, level: str, text: str) -> None:
        print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} [{sv.LEVELS.get(level, level.upper())}] {text}", flush=True)
        if level in ("aviso", "error") or (level == "ok" and text.startswith("Listo")):
            short = text.split(". Sugerencias en")[0]
            now = self.clock()
            if now - self._sent.get(short, -1e9) >= NOTIFY_REPEAT_S:
                self._sent[short] = now
                self.notifier("Sugerencias en vivo" + (" - problema" if level != "ok" else ""), short)


def open_log() -> None:
    """stdout y stderr al registro (pythonw no tiene consola); si pasa de 1 MB, el anterior va a .1."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if LOG_FILE.exists() and LOG_FILE.stat().st_size > LOG_MAX_BYTES:
        LOG_FILE.replace(LOG_FILE.with_suffix(".log.1"))
    f = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = f


# ---------------------------------------------------------------- un solo servicio a la vez

def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = k32.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(k32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            k32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def running_pid() -> Optional[int]:
    try:
        pid = int(PID_FILE.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None
    return pid if pid_alive(pid) else None


def terminate(pid: int) -> None:
    if sys.platform == "win32":
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = k32.OpenProcess(0x0001, False, pid)          # PROCESS_TERMINATE
        if handle:
            k32.TerminateProcess(handle, 1)
            k32.CloseHandle(handle)
    else:
        import signal
        os.kill(pid, signal.SIGTERM)


# ---------------------------------------------------------------- el supervisor

class Supervisor:
    """Arranca una sesión cuando Traktor se abre y la termina cuando se cierra. Si una sesión termina
    sola con error (puerto ocupado, colección que falta...), reintenta a los RETRY_S si Traktor sigue."""

    def __init__(self, is_running: Callable[[], bool], session: Callable[[threading.Event], int],
                 say: Callable[[str, str], None], clock=time.monotonic):
        self.is_running, self.session, self.say, self.clock = is_running, session, say, clock
        self.thread: Optional[threading.Thread] = None
        self.stop: Optional[threading.Event] = None
        self.result: Optional[int] = None
        self.retry_at = 0.0

    def _run(self, stop: threading.Event) -> None:
        try:
            self.result = self.session(stop)
        except Exception:                                   # que un error no tumbe el servicio
            self.say("error", "La sesión se cortó por un error: ver servicio.log")
            traceback.print_exc()
            self.result = 1

    def step(self) -> None:
        running = self.is_running()
        if self.thread is not None and not self.thread.is_alive():
            self.thread = None
            if self.result:
                self.retry_at = self.clock() + RETRY_S
        if running and self.thread is None and self.clock() >= self.retry_at:
            self.say("info", "Traktor abierto: empiezo las sugerencias")
            self.stop, self.result = threading.Event(), None
            self.thread = threading.Thread(target=self._run, args=(self.stop,), daemon=True)
            self.thread.start()
        elif not running and self.thread is not None:
            self.say("info", "Traktor cerrado: termino las sugerencias y suelto el controlador")
            self.stop.set()
            self.thread.join(timeout=10)
            self.thread, self.retry_at = None, 0.0
        elif not running:
            self.retry_at = 0.0

    def shutdown(self) -> None:
        if self.thread is not None:
            self.stop.set()
            self.thread.join(timeout=10)
            self.thread = None


def serve_forever(say: Callable[[str, str], None], is_running: Callable[[], bool] = traktor_running,
                  session: Optional[Callable[[threading.Event], int]] = None,
                  stop_file: Path = STOP_FILE, poll_s: float = POLL_S) -> None:
    """El bucle del servicio: hasta que aparezca stop_file (lo deja «quitar»)."""
    if session is None:
        def session(stop: threading.Event) -> int:
            return sv.run_session(sv.SETTINGS_FILE, stop, say=say)
    sup = Supervisor(is_running, session, say)
    try:
        while not stop_file.exists():
            try:
                sup.step()
            except Exception:
                traceback.print_exc()
            time.sleep(poll_s)
    finally:
        sup.shutdown()


def run_service() -> int:
    """Lo que corre el inicio automático: un solo servicio a la vez, registro en servicio.log."""
    open_log()
    other = running_pid()
    if other is not None and other != os.getpid():
        print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} [INFO] Ya hay un servicio corriendo (PID {other}): salgo")
        return 0
    STOP_FILE.unlink(missing_ok=True)
    PID_FILE.write_text(str(os.getpid()), encoding="ascii")
    say = ServiceSay()
    say("info", f"Servicio de sugerencias en vivo iniciado (PID {os.getpid()}); espera a que se abra Traktor")
    try:
        serve_forever(say)
    finally:
        say("info", "Servicio detenido")
        if running_pid() in (None, os.getpid()):
            PID_FILE.unlink(missing_ok=True)
        STOP_FILE.unlink(missing_ok=True)
    return 0


# ---------------------------------------------------------------- instalar, quitar, estado

def startup_dir() -> Path:
    return Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))) / "Microsoft" / "Windows" \
        / "Start Menu" / "Programs" / "Startup"


def shortcut_path() -> Path:
    return startup_dir() / f"{NAME}.lnk"


def service_python() -> Path:
    """pythonw.exe al lado del Python que corre esto (Windows, sin ventana); en macOS, el mismo Python."""
    exe = Path(sys.executable)
    if sys.platform == "win32":
        w = exe.with_name("pythonw.exe")
        if w.exists():
            return w
    return exe


def shortcut_script(lnk: Path, target: Path, arguments: str, workdir: Path) -> str:
    """PowerShell que crea el acceso directo (WScript.Shell, sin permisos de administrador)."""
    def q(s) -> str:
        return "'" + str(s).replace("'", "''") + "'"
    return ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut(" + q(lnk) + ")\n"
            f"$s.TargetPath = {q(target)}\n$s.Arguments = {q(arguments)}\n$s.WorkingDirectory = {q(workdir)}\n"
            f"$s.WindowStyle = 7\n$s.Description = {q('Sugerencias en vivo en Traktor (TRAKTOR ML)')}\n$s.Save()\n")


def mac_plist(python: Path, script: Path, log: Path) -> bytes:
    return plistlib.dumps({
        "Label": MAC_LABEL,
        "ProgramArguments": [str(python), str(script), "correr"],
        "WorkingDirectory": str(REPO_ROOT),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},           # se reinicia solo si se cae, no si se lo detiene
        "StandardOutPath": str(log),
        "StandardErrorPath": str(log),
        "ProcessType": "Interactive",
    })


def start_detached() -> None:
    script = Path(__file__).resolve()
    if sys.platform == "win32":
        subprocess.Popen([str(service_python()), str(script), "correr"], cwd=str(REPO_ROOT),
                         creationflags=DETACHED_PROCESS | CREATE_NO_WINDOW, close_fds=True)


def install() -> int:
    script = Path(__file__).resolve()
    if sys.platform == "win32":
        lnk = shortcut_path()
        lnk.parent.mkdir(parents=True, exist_ok=True)
        res = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                              shortcut_script(lnk, service_python(), f'"{script}" correr', REPO_ROOT)],
                             capture_output=True, text=True)
        if res.returncode != 0 or not lnk.exists():
            print(f"[ERROR] No pude crear el acceso directo de inicio: {res.stderr.strip()}")
            return 1
        print(f"[OK] Inicio automático: {lnk}")
        if running_pid() is None:
            start_detached()
            print("[OK] Servicio arrancado: no hace falta reiniciar")
        else:
            print(f"[INFO] El servicio ya estaba corriendo (PID {running_pid()})")
    elif sys.platform == "darwin":
        MAC_PLIST.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["launchctl", "unload", str(MAC_PLIST)], capture_output=True)
        MAC_PLIST.write_bytes(mac_plist(Path(sys.executable), script, te.DEFAULT_LOG_DIR / "servicio_launchd.log"))
        res = subprocess.run(["launchctl", "load", "-w", str(MAC_PLIST)], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[ERROR] launchctl load falló: {res.stderr.strip()}")
            return 1
        print(f"[OK] Inicio automático: {MAC_PLIST} (ya está corriendo)")
    else:
        print("[ERROR] El inicio automático es para Windows y macOS")
        return 1
    print("[INFO] Al abrir Traktor aparece una notificación «Sugerencias en vivo: Listo». Registro: "
          f"{LOG_FILE}")
    return 0


def stop_service(timeout: float = 8.0) -> None:
    pid = running_pid()
    if pid is None:
        return
    STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
    STOP_FILE.write_text("quitar", encoding="ascii")
    end = time.time() + timeout
    while time.time() < end and pid_alive(pid):
        time.sleep(0.25)
    if pid_alive(pid):
        terminate(pid)
    print(f"[OK] Servicio detenido (PID {pid})")


def uninstall() -> int:
    if sys.platform == "darwin" and MAC_PLIST.exists():
        subprocess.run(["launchctl", "unload", "-w", str(MAC_PLIST)], capture_output=True)
        MAC_PLIST.unlink()
        print(f"[OK] Quité {MAC_PLIST}")
    stop_service()
    if sys.platform == "win32" and shortcut_path().exists():
        shortcut_path().unlink()
        print(f"[OK] Quité {shortcut_path()}")
    STOP_FILE.unlink(missing_ok=True)
    print("[OK] Sin inicio automático. Para usarlo a mano: «Sugerencias en vivo.bat» / «.command»")
    return 0


def status() -> int:
    auto = shortcut_path() if sys.platform == "win32" else MAC_PLIST
    pid = running_pid()
    print(f"[INFO] Inicio automático: {'sí' if auto.exists() else 'no'} ({auto})")
    print(f"[INFO] Servicio: {'corriendo, PID ' + str(pid) if pid else 'no está corriendo'}")
    print(f"[INFO] Traktor: {'abierto' if traktor_running() else 'cerrado'}")
    if LOG_FILE.exists():
        print(f"[INFO] Últimas líneas de {LOG_FILE}:")
        for line in LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]:
            print("   " + line)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Sugerencias en vivo automáticas: arrancan solas con Traktor")
    ap.add_argument("accion", nargs="?", default="estado", choices=["instalar", "quitar", "estado", "correr"])
    args = ap.parse_args()
    return {"instalar": install, "quitar": uninstall, "estado": status, "correr": run_service}[args.accion]()


if __name__ == "__main__":
    sys.exit(main())
