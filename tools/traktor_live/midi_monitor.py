"""
PURPOSE: Monitor MIDI mínimo (Windows con winmm, macOS con python-rtmidi; ver midi_io.py). Lista las
         entradas MIDI, abre una por nombre y muestra cada mensaje (CC, notas) y, al terminar, el mínimo
         y el máximo de cada control. Sirve para dos pruebas de las sugerencias en vivo: el recorrido de
         los faders del DJM (CC 17-20 en el canal 1) y si este programa puede leer el controlador
         mientras Traktor lo tiene abierto (si el sistema no comparte el puerto, abrirlo da "en uso").
         El reloj MIDI (el DJM manda el tempo de su contador de BPM) se cuenta aparte y se resume en BPM.
CHANGELOG:
  - 2026-11-04: Creación inicial.
  - 2026-11-04: Ctrl+C por un manejador de señal: en la prueba del K2 el KeyboardInterrupt se escapó
                sin el resumen.
  - 2026-11-04: MidiInput (abrir y cerrar con callback) separado de monitor() para sugerencias_vivo.py.
  - 2026-11-05: La entrada MIDI pasa a midi_io.py (winmm o python-rtmidi): corre también en macOS.
"""
from __future__ import annotations

import argparse
import queue
import re
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from tools.traktor_live.midi_io import BACKENDS, default_backend, list_inputs, open_input  # noqa: E402,F401

SYSTEM = {0xF8: "clock", 0xFA: "start", 0xFB: "continue", 0xFC: "stop", 0xFE: "active_sensing"}
QUIET_KINDS = {"clock", "active_sensing"}   # llegan decenas por segundo: se cuentan, no se imprimen


def decode(packed: int) -> Tuple[str, int, int, int]:
    """Mensaje corto empaquetado (midi_io) -> (tipo, canal 1-16 o 0 si es de sistema, dato1, dato2)."""
    status, d1, d2 = packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF
    if status >= 0xF0:
        return SYSTEM.get(status, f"system_{status:02X}"), 0, d1, d2
    kind = {0x80: "note_off", 0x90: "note_on", 0xA0: "aftertouch", 0xB0: "cc", 0xC0: "program",
            0xD0: "pressure", 0xE0: "pitch"}[status & 0xF0]
    if kind == "note_on" and d2 == 0:
        kind = "note_off"
    return kind, (status & 0x0F) + 1, d1, d2


def clock_bpm(times: List[float]) -> Optional[float]:
    """BPM del reloj MIDI (24 pulsos por negra) a partir de los instantes de los pulsos."""
    if len(times) < 25:
        return None
    return 60.0 * (len(times) - 1) / (24.0 * (times[-1] - times[0]))


def describe(kind: str, ch: int, d1: int, d2: int) -> str:
    if kind == "cc":
        return f"Ch{ch:02d}.CC.{d1:03d} = {d2:3d} ({100 * d2 / 127:5.1f} %)"
    if kind in ("note_on", "note_off"):
        return f"Ch{ch:02d}.Note.{d1:03d} {'on ' if kind == 'note_on' else 'off'} vel {d2}"
    if ch == 0:
        return f"sistema: {kind}"
    return f"Ch{ch:02d} {kind} {d1} {d2}"


def summarize(events: List[Tuple[str, int, int, int]]) -> Dict[Tuple[int, int], Tuple[int, int, int]]:
    """(canal, CC) -> (mínimo, máximo, cantidad de mensajes)."""
    out: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
    for kind, ch, d1, d2 in events:
        if kind != "cc":
            continue
        lo, hi, n = out.get((ch, d1), (127, 0, 0))
        out[(ch, d1)] = (min(lo, d2), max(hi, d2), n + 1)
    return out


def find_input(names: List[str], text: str) -> Optional[int]:
    """Entrada cuyo nombre tiene una palabra que empieza con text ("K2" -> "XONE:K2 MIDI", nunca
    "DJM-750MK2", aunque el K2 no esté conectado). None si no hay ninguna o hay más de una."""
    t = text.lower()
    hits = [i for i, n in enumerate(names) if any(w.startswith(t) for w in re.split(r"[^0-9a-z]+", n.lower()))]
    return hits[0] if len(hits) == 1 else None


def monitor(index: int, seconds: Optional[float], quiet: bool, backend: Optional[str] = None) -> int:
    midi = open_input(index, backend)
    events = midi.events
    error = midi.open()
    if error:
        print(f"[ERROR] No pude abrir la entrada: {error}")
        return 2
    print("[OK] Escuchando. Ctrl+C para terminar." if seconds is None else f"[OK] Escuchando {seconds:.0f} s.")
    seen: List[Tuple[str, int, int, int]] = []
    clock: List[float] = []
    t_end = None if seconds is None else time.perf_counter() + seconds
    # Ctrl+C solo pide terminar: un KeyboardInterrupt en medio de la espera se escapaba sin resumen
    stop = threading.Event()
    previous = signal.signal(signal.SIGINT, lambda *_: stop.set())
    try:
        while not stop.is_set() and (t_end is None or time.perf_counter() < t_end):
            try:
                packed, t = events.get(timeout=0.1)
            except queue.Empty:
                continue
            ev = decode(packed)
            if ev[0] == "clock":
                clock.append(t)
            if ev[0] in QUIET_KINDS:
                continue
            seen.append(ev)
            if not quiet:
                print("  " + describe(*ev), flush=True)
    finally:
        signal.signal(signal.SIGINT, previous)
        midi.close()
    print(f"[INFO] {len(seen)} mensajes (sin contar {len(clock)} pulsos de reloj)")
    bpm = clock_bpm(clock)
    if bpm is not None:
        print(f"[INFO] Reloj MIDI: {bpm:.1f} BPM")
    for (ch, cc), (lo, hi, n) in sorted(summarize(seen).items()):
        print(f"  Ch{ch:02d}.CC.{cc:03d}: mínimo {lo:3d}, máximo {hi:3d} ({n} mensajes)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Monitor MIDI para las pruebas en vivo")
    ap.add_argument("--port", default=None, help="Parte del nombre de la entrada, p. ej. DJM o K2")
    ap.add_argument("--seconds", type=float, default=None, help="Escuchar N segundos (sin esto, hasta Ctrl+C)")
    ap.add_argument("--quiet", action="store_true", help="Solo el resumen final")
    ap.add_argument("--backend", choices=BACKENDS, default=None,
                    help=f"Motor MIDI (por defecto {default_backend()} en este sistema)")
    args = ap.parse_args()
    try:
        names = list_inputs(args.backend)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 1
    print("[INFO] Entradas MIDI:")
    for i, n in enumerate(names):
        print(f"  {i}: {n}")
    if args.port is None:
        return 0
    index = find_input(names, args.port)
    if index is None:
        print(f"[ERROR] '{args.port}' no identifica una sola entrada de la lista")
        return 1
    print(f"[INFO] Abriendo {index}: {names[index]}")
    return monitor(index, args.seconds, args.quiet, args.backend)


if __name__ == "__main__":
    sys.exit(main())
