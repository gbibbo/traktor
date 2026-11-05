"""
PURPOSE: Entrada MIDI de las herramientas en vivo, en Windows y en macOS. Dos motores con la misma forma:
         "winmm" (Windows, por ctypes, sin paquetes extra; el que se usa por defecto en Windows) y
         "rtmidi" (python-rtmidi: CoreMIDI en macOS, ALSA en Linux, y también Windows si se pide).
         Cada mensaje corto llega a events como (mensaje empaquetado status | d1 << 8 | d2 << 16,
         instante de time.perf_counter()), el formato de winmm, para que decode() sirva con los dos.
         En Windows (Windows MIDI Services) y en macOS (CoreMIDI) varios programas pueden leer la misma
         entrada: Traktor y este programa leen el controlador a la vez.
CHANGELOG:
  - 2026-11-05: Creación: la entrada winmm sale de midi_monitor.py y se suma python-rtmidi para macOS.
"""
from __future__ import annotations

import queue
import sys
import time
from typing import List, Optional, Sequence, Tuple

MIM_DATA = 0x3C3
CALLBACK_FUNCTION = 0x30000
WINMM_ERRORS = {2: "dispositivo inexistente", 4: "en uso por otro programa (el sistema no comparte el puerto)",
                6: "sin driver", 7: "sin memoria"}
RTMIDI_MISSING = ("falta python-rtmidi: instalarlo con «python -m pip install python-rtmidi==1.5.8» "
                  "(en Windows no hace falta: se usa winmm)")
CLIENT_NAME = "TRAKTOR ML"
BACKENDS = ("winmm", "rtmidi")


def default_backend(platform: Optional[str] = None) -> str:
    return "winmm" if (platform or sys.platform) == "win32" else "rtmidi"


def pack(message: Sequence[int]) -> Optional[int]:
    """Lista de bytes de rtmidi -> mensaje empaquetado como el de winmm (None si no es un mensaje corto)."""
    if not message or len(message) > 3 or message[0] == 0xF0:
        return None
    b = list(message) + [0, 0]
    return (b[0] & 0xFF) | ((b[1] & 0xFF) << 8) | ((b[2] & 0xFF) << 16)


# ---------------------------------------------------------------- winmm (Windows)

def _winmm():
    import ctypes
    return ctypes.WinDLL("winmm")


def _winmm_list() -> List[str]:
    import ctypes
    from ctypes import wintypes

    class MIDIINCAPSW(ctypes.Structure):
        _fields_ = [("wMid", wintypes.WORD), ("wPid", wintypes.WORD), ("vDriverVersion", wintypes.UINT),
                    ("szPname", wintypes.WCHAR * 32), ("dwSupport", wintypes.DWORD)]

    mm = _winmm()
    names = []
    for i in range(mm.midiInGetNumDevs()):
        caps = MIDIINCAPSW()
        if mm.midiInGetDevCapsW(ctypes.c_size_t(i), ctypes.byref(caps), ctypes.sizeof(caps)) == 0:
            names.append(caps.szPname)
        else:
            names.append(f"<entrada {i} sin datos>")
    return names


class WinMMInput:
    """Entrada abierta con callback de winmm."""

    def __init__(self, index: int):
        import ctypes
        from ctypes import wintypes
        self.index = index
        self.events: "queue.SimpleQueue[Tuple[int, float]]" = queue.SimpleQueue()
        self._ctypes = ctypes
        self._mm = _winmm()
        self._handle = wintypes.HANDLE()
        proc_type = ctypes.WINFUNCTYPE(None, wintypes.HANDLE, wintypes.UINT, ctypes.c_size_t, ctypes.c_size_t,
                                       ctypes.c_size_t)

        def on_msg(_h, msg, _inst, p1, _p2):
            if msg == MIM_DATA:
                self.events.put((int(p1), time.perf_counter()))

        self._proc = proc_type(on_msg)                 # referencia viva mientras la entrada esté abierta
        self._mm.midiInOpen.argtypes = [ctypes.POINTER(wintypes.HANDLE), wintypes.UINT, proc_type,
                                        ctypes.c_size_t, wintypes.DWORD]
        self._open = False

    def open(self) -> Optional[str]:
        """None si quedó abierta; si no, el motivo."""
        err = self._mm.midiInOpen(self._ctypes.byref(self._handle), self.index, self._proc, 0, CALLBACK_FUNCTION)
        if err:
            return WINMM_ERRORS.get(err, f"código {err}")
        self._mm.midiInStart(self._handle)
        self._open = True
        return None

    def close(self) -> None:
        if self._open:
            self._mm.midiInStop(self._handle)
            self._mm.midiInReset(self._handle)
            self._mm.midiInClose(self._handle)
            self._open = False


# ---------------------------------------------------------------- python-rtmidi (macOS, Linux, Windows)

def _rtmidi():
    try:
        import rtmidi  # type: ignore
    except ImportError as exc:
        raise RuntimeError(RTMIDI_MISSING) from exc
    return rtmidi


def _rtmidi_list() -> List[str]:
    rtmidi = _rtmidi()
    m = rtmidi.MidiIn(name=CLIENT_NAME)
    try:
        return list(m.get_ports())
    finally:
        m.delete()


class RtMidiInput:
    """Entrada abierta con python-rtmidi (callback en su propio hilo)."""

    def __init__(self, index: int):
        self.index = index
        self.events: "queue.SimpleQueue[Tuple[int, float]]" = queue.SimpleQueue()
        self._m = None

    def _on_msg(self, event, _data=None) -> None:
        message, _delta = event
        packed = pack(message)
        if packed is not None:
            self.events.put((packed, time.perf_counter()))

    def open(self) -> Optional[str]:
        try:
            rtmidi = _rtmidi()
        except RuntimeError as exc:
            return str(exc)
        m = rtmidi.MidiIn(name=CLIENT_NAME)
        try:
            m.ignore_types(sysex=True, timing=False, active_sense=False)   # el monitor cuenta el reloj
            m.set_callback(self._on_msg)
            m.open_port(self.index, name=f"{CLIENT_NAME} entrada")
        except Exception as exc:                        # rtmidi.SystemError, InvalidPortError
            m.delete()
            return f"{type(exc).__name__}: {exc}"
        self._m = m
        return None

    def close(self) -> None:
        if self._m is not None:
            self._m.cancel_callback()
            self._m.close_port()
            self._m.delete()
            self._m = None


# ---------------------------------------------------------------- uso común

def list_inputs(backend: Optional[str] = None) -> List[str]:
    """Nombres de las entradas MIDI, en el orden de sus índices."""
    b = backend or default_backend()
    return _winmm_list() if b == "winmm" else _rtmidi_list()


def open_input(index: int, backend: Optional[str] = None):
    """Entrada sin abrir (llamar a .open(), que devuelve None o el motivo del error)."""
    b = backend or default_backend()
    return WinMMInput(index) if b == "winmm" else RtMidiInput(index)
