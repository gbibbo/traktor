"""
PURPOSE: Tests de tools/traktor_live/midi_monitor.py sin hardware: decodificación de mensajes de winmm,
         resumen de mínimo y máximo por control, BPM del reloj MIDI y búsqueda de la entrada por nombre.
CHANGELOG:
  - 2026-11-04: Creación inicial.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.traktor_live.midi_monitor import clock_bpm, decode, describe, find_input, summarize  # noqa: E402


def _packed(status, d1=0, d2=0):
    return status | (d1 << 8) | (d2 << 16)


def test_decode_cc_note_and_system():
    assert decode(_packed(0xB0, 17, 64)) == ("cc", 1, 17, 64)            # fader 1 del DJM a la mitad
    assert decode(_packed(0x9F, 36, 127)) == ("note_on", 16, 36, 127)
    assert decode(_packed(0x90, 36, 0)) == ("note_off", 1, 36, 0)        # note on con velocidad 0
    assert decode(_packed(0xF8)) == ("clock", 0, 0, 0)
    assert describe(*decode(_packed(0xB0, 20, 127))) == "Ch01.CC.020 = 127 (100.0 %)"


def test_summarize_min_max_per_control():
    evs = [decode(_packed(0xB0, 17, v)) for v in (0, 64, 127)] + [decode(_packed(0xB0, 18, 5))]
    evs.append(decode(_packed(0x90, 36, 127)))
    assert summarize(evs) == {(1, 17): (0, 127, 3), (1, 18): (5, 5, 1)}


def test_clock_bpm():
    pulses = [i * (60.0 / (124.0 * 24)) for i in range(97)]             # 4 negras a 124 BPM
    assert abs(clock_bpm(pulses) - 124.0) < 1e-6
    assert clock_bpm(pulses[:10]) is None


def test_find_input_by_name_part():
    names = ["DJM-750MK2", "XONE:K2 MIDI"]
    assert find_input(names, "k2") == 1                  # palabra "K2", no el final de "750MK2"
    assert find_input(names, "DJM") == 0 and find_input(names, "xone") == 1
    assert find_input(names, "750") == 0
    assert find_input(["DJM-750MK2"], "k2") is None       # sin el K2 conectado no abre el DJM
    assert find_input(names, "S4") is None
    assert find_input(["XONE:K2 MIDI", "XONE:K2 MIDI 2"], "xone") is None   # ambiguo
