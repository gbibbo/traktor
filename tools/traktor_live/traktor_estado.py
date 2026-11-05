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
"""
from __future__ import annotations

import argparse
import datetime as dt
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

    def __init__(self, verbose: bool = False):
        self.decks: Dict[str, dict] = {}
        self.channels: Dict[str, dict] = {}
        self.master: dict = {}
        self.probe: Dict[str, dict] = {}
        self.commands: List[dict] = []
        self.next_id = 0
        self.verbose = verbose
        self.lock = threading.Lock()

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
                if reason == "screenLoaded":
                    return [f"Pantalla del D2 ({target}) creada; su navegador está en {where}"]
                if reason == "pathChanged":
                    return [f"Navegador del D2 ({target}) en {where}"]
                if reason == "done":
                    return [f"Orden «{data.get('cmd')}» hecha; navegador del D2 en {where}"]
                if reason == "refused":
                    return [f"Orden «{data.get('cmd')}» rechazada (lo elegido no es una carpeta) en {where}"]
                if reason == "error":
                    return [f"[AVISO] La sonda dio un error: {data.get('error')}"]
                return [f"Navegador del D2 ({target}) sigue en {where}"] if self.verbose else []
        return []

    def enqueue(self, cmd: str, arg=None) -> int:
        """Orden para la sonda (la primera pantalla que pregunte la toma)."""
        with self.lock:
            self.next_id += 1
            self.commands.append({"id": self.next_id, "cmd": cmd, "arg": arg})
            return self.next_id

    def pop_command(self) -> Optional[dict]:
        with self.lock:
            return self.commands.pop(0) if self.commands else None


def make_handler(state: State, log_path: Optional[Path]):
    log_lock = threading.Lock()

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
                if cmd:
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
            if self.path == "/enqueue":           # orden para la sonda: {"cmd": "exit"|"select"|"enter"|"report", "arg": n}
                try:
                    req = json.loads(raw.decode("utf-8") or "{}")
                    ok = req.get("cmd") in ("exit", "select", "enter", "report")
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
            if log_path is not None:
                with log_lock, open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"t": now.isoformat(timespec="milliseconds"), "kind": ep[0],
                                        "target": ep[1], "data": data}, ensure_ascii=False) + "\n")
            if "_raw" in data:
                return
            for line in state.apply(ep[0], ep[1], data):
                print(f"[{now:%H:%M:%S}] {line}", flush=True)

        def log_message(self, *args):  # silencia el log por pedido de http.server
            pass

    return Handler


class _Server6(ThreadingHTTPServer):
    address_family = socket.AF_INET6


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
    servers = [ThreadingHTTPServer(("127.0.0.1", args.port), handler)]
    try:                                        # "localhost" puede resolver primero a ::1
        servers.append(_Server6(("::1", args.port), handler))
    except OSError:
        pass
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
