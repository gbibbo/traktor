"""
PURPOSE: Tests del servicio automático de las sugerencias en vivo (tools/traktor_live/sugerencias_servicio.py):
         el supervisor arranca y termina la sesión con Traktor y reintenta si falla, el bucle termina con
         el archivo de parada, las notificaciones van sin sonido y con el texto escapado, los avisos no se
         repiten, el acceso directo de Windows y el LaunchAgent de macOS tienen lo que tienen que tener, y
         el control de un solo proceso. Corren en Windows, macOS y Linux.
CHANGELOG:
  - 2026-11-06: Creación inicial.
"""
import base64
import plistlib
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.traktor_live import sugerencias_servicio as ss  # noqa: E402


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_supervisor_follows_traktor_and_retries_after_a_failed_start():
    traktor = {"open": False}
    started, said = [], []
    outcome = {"code": 1}

    def session(stop):
        started.append(stop)
        if outcome["code"]:
            return outcome["code"]                       # p. ej. el puerto ocupado
        stop.wait(5)
        return 0

    clock = FakeClock()
    sup = ss.Supervisor(lambda: traktor["open"], session, lambda lvl, txt: said.append(txt), clock=clock)
    sup.step()
    assert started == []                                 # Traktor cerrado: nada
    traktor["open"] = True
    sup.step()
    sup.thread.join(2)
    assert len(started) == 1
    sup.step()                                           # terminó con error: espera RETRY_S
    assert len(started) == 1 and sup.thread is None
    clock.t += ss.RETRY_S + 1
    outcome["code"] = 0
    sup.step()
    assert len(started) == 2 and sup.thread.is_alive()
    traktor["open"] = False
    sup.step()                                           # Traktor cerrado: la sesión se detiene
    assert started[1].is_set() and sup.thread is None
    assert any("Traktor cerrado" in s for s in said)


def test_service_loop_ends_with_the_stop_file(tmp_path):
    stop_file = tmp_path / "servicio.stop"
    sessions = []

    def session(stop):
        sessions.append(stop)
        stop.wait(10)
        return 0

    runner = threading.Thread(target=ss.serve_forever, kwargs=dict(
        say=lambda lvl, txt: None, is_running=lambda: True, session=session, stop_file=stop_file, poll_s=0.05))
    runner.start()
    time.sleep(0.3)
    stop_file.write_text("quitar")
    runner.join(5)
    assert not runner.is_alive() and len(sessions) == 1 and sessions[0].is_set()


def test_notifications_are_silent_escaped_and_not_repeated():
    xml = ss.toast_xml("Sugerencias en vivo", 'No veo «K2» <&> "ya"')
    assert "<audio silent='true'/>" in xml and "&lt;&amp;&gt;" in xml and "&quot;ya&quot;" in xml
    cmd = ss.toast_command("T", "it's")
    script = base64.b64decode(cmd[-1]).decode("utf-16-le")
    assert "-EncodedCommand" in cmd and "it&apos;s" in script and "silent=''true''" in script
    sent = []
    clock = FakeClock()
    say = ss.ServiceSay(notifier=lambda t, x: sent.append(x), clock=clock)
    say("info", "Colección: ...")                         # lo normal no notifica
    say("ok", "Listo. Controlador: XONE:K2 MIDI. Botón: Ch15.Note.D0. Sugerencias en C:\\X.nml.")
    say("aviso", "No veo el controlador")
    say("aviso", "No veo el controlador")                 # repetido enseguida: no
    clock.t += ss.NOTIFY_REPEAT_S + 1
    say("aviso", "No veo el controlador")
    assert sent == ["Listo. Controlador: XONE:K2 MIDI. Botón: Ch15.Note.D0", "No veo el controlador",
                    "No veo el controlador"]
    assert ss.mac_notify_command("T", 'a "b"')[2] == 'display notification "a \\"b\\"" with title "T"'


def test_windows_shortcut_and_mac_launch_agent():
    script = ss.shortcut_script(Path("C:/x/Startup/TRAKTOR ML Sugerencias.lnk"), Path("C:/v/pythonw.exe"),
                                '"C:/repo/tools/traktor_live/sugerencias_servicio.py" correr', Path("C:/it's repo"))
    assert "CreateShortcut('C:" in script and "$s.WindowStyle = 7" in script
    assert "it''s repo" in script and "correr'" in script and script.rstrip().endswith("$s.Save()")
    plist = plistlib.loads(ss.mac_plist(Path("/r/.venv/bin/python"), Path("/r/tools/traktor_live/s.py"),
                                        Path("/r/log.txt")))
    assert plist["Label"] == ss.MAC_LABEL and plist["ProgramArguments"][-1] == "correr"
    assert plist["RunAtLoad"] is True and plist["KeepAlive"] == {"SuccessfulExit": False}


def test_pid_checks():
    assert ss.pid_alive(__import__("os").getpid())
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    assert not ss.pid_alive(p.pid) and not ss.pid_alive(0)
