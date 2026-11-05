"""
PURPOSE: Tests de tools/traktor_live/mapeo_tsi.py con un DeviceIO.Config.Controller armado a mano (dos
         dispositivos, un control con dos asignaciones) y de los mapeos que quedan en el repo
         (tools/traktor_live/mapeos/): el K2 con D0 libre y el teclado.
CHANGELOG:
  - 2026-11-05: Creación inicial.
"""
import struct
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.traktor_live import mapeo_tsi as m  # noqa: E402

MAPEOS = REPO_ROOT / "tools" / "traktor_live" / "mapeos"


def w(s: str) -> bytes:
    return struct.pack(">I", len(s)) + s.encode("utf-16-be")


def ch(tag: str, body: bytes) -> bytes:
    return tag.encode("ascii") + struct.pack(">I", len(body)) + body


def device(kind: str, port: str, controls) -> bytes:
    """DEVI con una asignación por control: controls = [(id, "Ch15.Note.D0"), ...]."""
    cmas = struct.pack(">I", len(controls)) + b"".join(
        ch("CMAI", struct.pack(">III", cid, 0, 1000 + cid) + ch("CMAD", b"\0" * 8)) for cid, _ in controls)
    dcbm = struct.pack(">I", len(controls)) + b"".join(ch("DCBM", struct.pack(">I", cid) + w(nm))
                                                       for cid, nm in controls)
    ddat = (ch("DDIF", b"\0" * 4) + ch("DDIV", w("3.2.0 60") + struct.pack(">I", 31)) + ch("DDIC", w(""))
            + ch("DDPT", w(port) + w(port)) + ch("DDDC", ch("DDCI", struct.pack(">I", 0)))
            + ch("DDCB", ch("CMAS", cmas) + ch("DCBM", dcbm)) + ch("DVST", b"\0" * 12))
    return w(kind) + ch("DDAT", ddat)


def blob(*devis: bytes) -> bytes:
    devs = struct.pack(">I", len(devis)) + b"".join(ch("DEVI", d) for d in devis)
    return ch("DIOM", ch("DIOI", struct.pack(">I", 1)) + ch("DEVS", devs))


K2 = [(65, "Ch15.CC.020"), (89, "Ch15.Note.D0"), (13, "Ch15.Note.C#0"), (125, "Ch15.Note.D0")]


def test_read_devices_and_rebuild_identical():
    b = blob(device("Generic MIDI", "XONE:K2", K2), device("Generic MIDI", "DJM", [(1, "Ch01.CC.017")]))
    devis = m.devices(b)
    assert len(devis) == 2 and m.build(b, devis) == b
    k2 = m.info(devis[0])
    assert (k2.kind, k2.version, k2.in_port, k2.out_port) == ("Generic MIDI", "3.2.0 60", "XONE:K2", "XONE:K2")
    assert len(k2.assignments) == 4 and k2.assigned("Ch15.Note.D0") == 2 and k2.assigned("Ch15.Note.C#0") == 1


def test_free_a_button_and_keep_the_rest():
    b = blob(device("Generic MIDI", "XONE:K2", K2))
    devi, removed = m.free_controls(m.devices(b)[0], ["Ch15.Note.D0"])
    assert removed == 2
    out = m.build(b, [devi])
    k2 = m.info(m.devices(out)[0])                     # vuelve a leerse entero: las cantidades cierran
    assert k2.assigned("Ch15.Note.D0") == 0 and sorted(k2.assignments) == [13, 65]
    assert k2.controls == {65: "Ch15.CC.020", 13: "Ch15.Note.C#0"}
    assert m.free_controls(devi, ["Ch15.Note.D0"])[1] == 0


def test_ports_and_the_tsi_file_round_trip(tmp_path):
    b = blob(device("Generic MIDI", "XONE:K2", K2))
    devi = m.set_ports(m.devices(b)[0], "XONE:K2 MIDI", "XONE:K2 MIDI")
    out = m.build(b, [devi])
    path = tmp_path / "k2.tsi"
    path.write_bytes(m.tsi_text(m.ENTRIES["controlador"], out).encode("utf-8"))
    text = path.read_bytes()
    assert text.startswith(b'<?xml version="1.0" encoding="utf-8" standalone="no"?>\r\n<NIXML>')
    again = m.read_entry(path, "DeviceIO.Config.Controller")
    assert again == out and m.info(m.devices(again)[0]).in_port == "XONE:K2 MIDI"


def test_a_count_that_does_not_close_is_refused():
    good = device("Generic MIDI", "XONE:K2", K2)
    bad = good.replace(b"CMAS" + good.split(b"CMAS")[1][:4] + struct.pack(">I", 4),
                       b"CMAS" + good.split(b"CMAS")[1][:4] + struct.pack(">I", 5))
    with pytest.raises(ValueError):
        m.info(bad)


def test_the_mappings_in_the_repo():
    k2_path = MAPEOS / "Xone K2 - TRAKTOR ML (D0 libre).tsi"
    k2_blob = m.read_entry(k2_path, "DeviceIO.Config.Controller")
    devis = m.devices(k2_blob)
    assert len(devis) == 1 and m.build(k2_blob, devis) == k2_blob
    k2 = m.info(devis[0])
    assert k2.kind == "Generic MIDI" and k2.in_port == "XONE:K2 MIDI" and len(k2.assignments) == 126
    assert k2.assigned("Ch15.Note.D0") == 0 and k2.assigned("Ch15.Note.C#0") == 1
    assert k2_path.read_bytes().count(b"\r\n") == 8   # CRLF como los de NI (y *.tsi -text en git)
    kb = m.info(m.devices(m.read_entry(MAPEOS / "Teclado - TRAKTOR ML.tsi", "DeviceIO.Config.Keyboard"))[0])
    assert kb.kind == "Generic Keyboard" and len(kb.assignments) == 56
