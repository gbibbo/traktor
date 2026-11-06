"""
PURPOSE: Receptor local de traktor-api-client (mod QML del controlador Kontrol D2, ver
         tools/traktor_live/traktor_qml/). Traktor envía por HTTP a localhost:8080 qué tema se carga en
         cada deck (con su ruta), si suena, por dónde va, el master y, con el mezclador interno, qué
         canal está al aire. Este script guarda ese estado, imprime una línea por cambio y anota cada
         evento en un .jsonl para evaluar después (cargado no es lo mismo que tocado).
         Solo biblioteca estándar; escucha en 127.0.0.1 y ::1, nunca en la red.
CHANGELOG:
  - 2026-11-04: Creación inicial (prueba de instalación del mod en Traktor 4.4.1).
  - 2026-11-04: Ctrl+C apaga los dos servidores sin excepción en el hilo de ::1; tonalidad desde key.
  - 2026-11-04: browserProbe/<lado>: la sonda de la pantalla del D2 (creada, nodo de su navegador).
  - 2026-11-04: Órdenes para la sonda (POST /enqueue, GET /command/<lado>) y GET /state, para probar si
                el navegador del D2 mueve el árbol de Traktor.
  - 2026-11-04: Puerto exclusivo: un segundo receptor avisa y sale (antes Windows lo dejaba abrir en el
                mismo puerto y el viejo seguía recibiendo).
  - 2026-11-04: Orden "goto" (ruta completa) y wait_result() para esperar el reporte de una orden.
  - 2026-11-05: Con show_browser=False tampoco imprime el "goto" fallido (lo avisa sugerencias_vivo.py).
  - 2026-11-05: open_servers() para este receptor y sugerencias_vivo.py; ::1 solo si el equipo tiene IPv6.
  - 2026-11-06: log_event(): el registro de la sesión con un solo candado para todos los que escriben.
"""
from __future__ import annotations

import argparse
import datetime as dt
import errno
import json
import signal
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)

DEFAULT_LOG_DIR = REPO_ROOT / "artifacts" / "v4" / "traktor_live"
KINDS = ("deckLoaded", "updateDeck", "updateChannel", "updateMasterClock", "browserProbe")
PROBE_COMMANDS = ("exit", "select", "enter", "report", "goto")   # solo navegación: ninguna carga temas


def parse_endpoint(path: str) -> Optional[Tuple[str, str]]:
    """"/deckLoaded/A" -> ("deckLoaded", "A"); "/updateMasterClock" -> ("updateMasterClock", "")."""
    parts = [p for p in path.split("?")[0].split("/") if p]
    if not parts or parts[0] not in KINDS:
        return None
    return parts[0], (parts[1] if len(parts) > 1 else "")


def _clock(seconds) -> str:
    try:
        s = int(float(seconds))
    except (TypeError, ValueError):
        return "?"
    return f"{s // 60}:{s % 60:02d}"


class State:
    """Último estado conocido de decks, canales y master. apply() devuelve las líneas a imprimir."""

    def __init__(self, verbose: bool = False, show_browser: bool = True):
        self.decks: Dict[str, dict] = {}
        self.channels: Dict[str, dict] = {}
        self.master: dict = {}
        self.probe: Dict[str, dict] = {}
        self.commands: List[dict] = []
        self.next_id = 0
        self.results: Dict[int, dict] = {}
        self.verbose = verbose
        self.show_browser = show_browser          # imprimir cada cambio de carpeta del navegador del D2
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)

    def apply(self, kind: str, target: str, data: dict) -> List[str]:
        with self.lock:
            if kind == "deckLoaded":
                self.decks[target] = dict(data)
                if not data.get("filePath"):
                    return [f"Deck {target}: vacío"]
                return [f"Deck {target} cargó: {data.get('artist', '')} - {data.get('title', '')}  "
                        f"[{float(data.get('bpm') or 0):.1f} BPM, {data.get('keyText') or data.get('key') or '?'}]  {data.get('filePath')}"]
            if kind == "updateDeck":
                deck = self.decks.setdefault(target, {})
                lines = []
                if "isPlaying" in data and data["isPlaying"] != deck.get("isPlaying"):
                    lines.append(f"Deck {target} {'▶ suena' if data['isPlaying'] else '⏸ parado'} en "
                                 f"{_clock(data.get('elapsedTime', deck.get('elapsedTime')))}")
                elif self.verbose and "elapsedTime" not in data:
                    lines.append(f"Deck {target}: {data}")
                deck.update(data)
                return lines
            if kind == "updateChannel":
                ch = self.channels.setdefault(target, {})
                lines = []
                if "isOnAir" in data and data["isOnAir"] != ch.get("isOnAir"):
                    lines.append(f"Canal {target} {'AL AIRE' if data['isOnAir'] else 'fuera del aire'}")
                if self.verbose and "onAirLevel" in data:
                    lines.append(f"Canal {target} nivel {data['onAirLevel']:.2f}")
                ch.update(data)
                return lines
            if kind == "updateMasterClock":
                changed = (data.get("deck"), round(float(data.get("bpm") or 0), 1)) != \
                          (self.master.get("deck"), round(float(self.master.get("bpm") or 0), 1))
                self.master = dict(data)
                return [f"Master: deck {data.get('deck') or '-'}, {float(data.get('bpm') or 0):.1f} BPM"] if changed else []
            if kind == "browserProbe":              # sonda de la pantalla del D2 (ApiBrowserProbe.qml)
                self.probe[target] = dict(data)
                where = (f"«{data.get('path') or '(vacío)'}»  [{data.get('count')} elementos, "
                         f"índice {data.get('currentIndex')}, {'lista de temas' if data.get('isContentList') else 'nodos'}]")
                reason = data.get("reason")
                if data.get("id") and reason in ("done", "refused", "gotoDone", "gotoFailed"):
                    self.results[int(data["id"])] = dict(data)      # para wait_result()
                    self.cond.notify_all()
                if reason == "screenLoaded":
                    return [f"Pantalla del D2 ({target}) creada; su navegador está en {where}"]
                if reason == "pathChanged":
                    return [f"Navegador del D2 ({target}) en {where}"] if self.show_browser else []
                if reason == "done":
                    return [f"Orden «{data.get('cmd')}» hecha; navegador del D2 en {where}"] if self.show_browser else []
                if reason == "refused":
                    return [f"Orden «{data.get('cmd')}» rechazada (lo elegido no es una carpeta) en {where}"]
                if reason == "gotoDone":
                    return [f"Navegador del D2 llegó a {where}"] if self.show_browser else []
                if reason == "gotoFailed":              # sugerencias_vivo.py avisa por su cuenta
                    return [f"[AVISO] El navegador del D2 no pudo llegar: {data.get('error')}"] if self.show_browser else []
                if reason == "error":
                    return [f"[AVISO] La sonda dio un error: {data.get('error')}"]
                return [f"Navegador del D2 ({target}) sigue en {where}"] if self.verbose else []
        return []

    def wait_result(self, cmd_id: int, timeout: float) -> Optional[dict]:
        """Reporte de la sonda para la orden cmd_id (None si no llegó a tiempo)."""
        with self.cond:
            self.cond.wait_for(lambda: cmd_id in self.results, timeout)
            return self.results.pop(cmd_id, None)

    def enqueue(self, cmd: str, arg=None) -> int:
        """Orden para la sonda (la primera pantalla que pregunte la toma)."""
        with self.lock:
            self.next_id += 1
            self.commands.append({"id": self.next_id, "cmd": cmd, "arg": arg})
            return self.next_id

    def pop_command(self) -> Optional[dict]:
        with self.lock:
            return self.commands.pop(0) if self.commands else None


_LOG_LOCK = threading.Lock()


def log_event(log_path: Optional[Path], kind: str, target: str, data: dict) -> None:
    """Una línea JSON en el registro de la sesión (la usan el receptor y sugerencias_vivo.py, que anota
    cada pulsación con los temas sugeridos)."""
    if log_path is None:
        return
    line = json.dumps({"t": dt.datetime.now().isoformat(timespec="milliseconds"), "kind": kind,
                       "target": target, "data": data}, ensure_ascii=False)
    with _LOG_LOCK, open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def make_handler(state: State, log_path: Optional[Path]):

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, body: bytes = b"", ctype: str = "application/json") -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            """/command/<lado>: la próxima orden para la sonda ({} si no hay); /state: último estado."""
            if self.path.startswith("/command/"):
                cmd = state.pop_command()
                self._reply(json.dumps(cmd or {}).encode("utf-8"))
                if cmd and state.verbose:
                    print(f"[{dt.datetime.now():%H:%M:%S}] Orden «{cmd['cmd']}» {cmd.get('arg') if cmd.get('arg') is not None else ''} "
                          f"entregada a la sonda", flush=True)
            elif self.path == "/state":
                with state.lock:
                    snap = {"decks": state.decks, "channels": state.channels, "master": state.master,
                            "probe": state.probe, "pending": state.commands}
                    body = json.dumps(snap, ensure_ascii=False).encode("utf-8")
                self._reply(body)
            else:
                self.send_error(404)

        def do_POST(self):  # noqa: N802 (nombre de http.server)
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length > 0 else b""
            if self.path == "/enqueue":           # orden para la sonda: {"cmd": ..., "arg": ...}
                try:
                    req = json.loads(raw.decode("utf-8") or "{}")
                    ok = req.get("cmd") in PROBE_COMMANDS
                except json.JSONDecodeError:
                    ok = False
                if not ok:
                    self.send_error(400)
                    return
                self._reply(json.dumps({"id": state.enqueue(req["cmd"], req.get("arg"))}).encode("utf-8"))
                return
            self._reply()
            ep = parse_endpoint(self.path)
            if ep is None:
                print(f"[AVISO] Ruta desconocida: {self.path}")
                return
            text = raw.decode("utf-8", errors="replace")
            try:
                data = json.loads(text) if text else {}
            except json.JSONDecodeError:
                print(f"[AVISO] JSON ilegible en {self.path}: {text[:200]!r}")
                data = {"_raw": text}
            now = dt.datetime.now()
            log_event(log_path, ep[0], ep[1], data)
            if "_raw" in data:
                return
            for line in state.apply(ep[0], ep[1], data):
                print(f"[{now:%H:%M:%S}] {line}", flush=True)

        def log_message(self, *args):  # silencia el log por pedido de http.server
            pass

    return Handler


def port_in_use(exc: OSError) -> bool:
    return getattr(exc, "winerror", None) in (10048, 10013) or exc.errno == errno.EADDRINUSE


class ExclusiveServer(ThreadingHTTPServer):
    """Puerto exclusivo: con SO_REUSEADDR (lo que pone http.server), Windows dejó abrir un segundo receptor
    en el mismo puerto y el tráfico siguió yendo al viejo. Así el segundo falla al arrancar."""
    allow_reuse_address = False

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class _Server6(ExclusiveServer):
    address_family = socket.AF_INET6


def open_servers(port: int, handler) -> List[ThreadingHTTPServer]:
    """Servidores en 127.0.0.1 y, si el equipo tiene IPv6, en ::1 ("localhost" puede resolver primero a
    ::1). Si el puerto está ocupado, cierra lo abierto y deja pasar el OSError (ver port_in_use)."""
    servers: List[ThreadingHTTPServer] = [ExclusiveServer(("127.0.0.1", port), handler)]
    try:
        servers.append(_Server6(("::1", port), handler))
    except OSError as exc:
        if port_in_use(exc):
            for s in servers:
                s.server_close()
            raise
    return servers


def serve(servers: List[ThreadingHTTPServer], stop: threading.Event) -> None:
    """Atiende en hilos hasta que stop se activa (Ctrl+C) y después apaga cada servidor antes de
    cerrarlo: cerrar el socket con serve_forever corriendo tiraba una excepción en el hilo."""
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in servers]
    for t in threads:
        t.start()
    try:
        while not stop.wait(0.25):
            pass
    finally:
        for s in servers:
            s.shutdown()
            s.server_close()
        for t in threads:
            t.join(timeout=2)


def main() -> int:
    ap = argparse.ArgumentParser(description="Recibe el estado de Traktor desde traktor-api-client")
    ap.add_argument("--port", type=int, default=8080, help="Puerto (el de ApiClient.js, 8080)")
    ap.add_argument("--log", default=None, help="Archivo .jsonl de eventos (por defecto artifacts/v4/traktor_live/)")
    ap.add_argument("--no-log", action="store_true", help="No anotar eventos")
    ap.add_argument("--verbose", action="store_true", help="Mostrar también niveles y cambios menores")
    args = ap.parse_args()

    log_path = None
    if not args.no_log:
        log_path = Path(args.log) if args.log else DEFAULT_LOG_DIR / f"traktor_estado_{dt.datetime.now():%Y%m%d_%H%M%S}.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
    state = State(verbose=args.verbose)
    handler = make_handler(state, log_path)
    try:
        servers = open_servers(args.port, handler)
    except OSError as exc:
        if port_in_use(exc):
            print(f"[ERROR] El puerto {args.port} ya lo usa otro programa (¿otra ventana con este receptor?). "
                  "Cerrarla con Ctrl+C y volver a correr esto.")
            return 1
        raise
    print(f"[INFO] Escuchando en localhost:{args.port}. Carga un tema en Traktor; Ctrl+C para salir.")
    if log_path is not None:
        print(f"[INFO] Eventos en {log_path}")
    stop = threading.Event()
    previous = signal.signal(signal.SIGINT, lambda *_: stop.set())
    try:
        serve(servers, stop)
    finally:
        signal.signal(signal.SIGINT, previous)
    loaded = ", ".join(f"{d}={v.get('title', '')}" for d, v in sorted(state.decks.items()))
    print(f"[INFO] Fin. Decks: {loaded or 'sin datos'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
