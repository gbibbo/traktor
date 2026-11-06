"""
PURPOSE: Tests de tools/traktor_live/mapeo_tsi.py con un DeviceIO.Config.Controller armado a mano (dos
         dispositivos; D0 con la preescucha y el Modifier #2, el encoder derecho con la lista y el
         "Seek" de la preescucha condicionado a M2) y de los mapeos que quedan en el repo
         (tools/traktor_live/mapeos/): el K2 con D0 libre y el teclado.
CHANGELOG:
  - 2026-11-05: Creación inicial.
  - 2026-11-06: CMAD con sus campos (modo, valor, condiciones): --detalle y --quitar; el K2 del repo
                queda sin la preescucha de D0 ni su "Seek" en el encoder derecho (125 asignaciones).
"""
import struct
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.traktor_live import mapeo_tsi as m  # noqa: E402

MAPEOS = REPO_ROOT / "tools" / "traktor_live" / "mapeos"
M1, M2 = 2548, 2549


def w(s: str) -> bytes:
    return struct.pack(">I", len(s)) + s.encode("utf-16-be")


def ch(tag: str, body: bytes) -> bytes:
    return tag.encode("ascii") + struct.pack(">I", len(body)) + body


def cmad(mode: int, value: int, cond1=(0, 0), cond2=(0, 0)) -> bytes:
    """CMAD de 120 bytes como los de Traktor: 12 enteros, comentario vacío, condiciones y relleno."""
    head = struct.pack(">12i", 4, 0, mode, 0, 0, 0, 0, 0, 0, 0, 1, value)
    conds = struct.pack(">4i", cond1[0], cond1[1], cond2[1], cond2[0])
    body = head + w("") + conds
    return ch("CMAD", body + b"\0" * (120 - len(body)))


def device(kind: str, port: str, rows) -> bytes:
    """DEVI con una asignación por fila: (id, control, comando, modo, valor, condición1, condición2)."""
    cmas = struct.pack(">I", len(rows)) + b"".join(
        ch("CMAI", struct.pack(">III", cid, 0, cmd) + cmad(mode, value, c1, c2))
        for cid, _, cmd, mode, value, c1, c2 in rows)
    dcbm = struct.pack(">I", len(rows)) + b"".join(ch("DCBM", struct.pack(">I", cid) + w(nm))
                                                    for cid, nm, *_ in rows)
    ddat = (ch("DDIF", b"\0" * 4) + ch("DDIV", w("3.2.0 60") + struct.pack(">I", 31)) + ch("DDIC", w(""))
            + ch("DDPT", w(port) + w(port)) + ch("DDDC", ch("DDCI", struct.pack(">I", 0)))
            + ch("DDCB", ch("CMAS", cmas) + ch("DCBM", dcbm)) + ch("DVST", b"\0" * 12))
    return w(kind) + ch("DDAT", ddat)


def blob(*devis: bytes) -> bytes:
    devs = struct.pack(">I", len(devis)) + b"".join(ch("DEVI", d) for d in devis)
    return ch("DIOM", ch("DIOI", struct.pack(">I", 1)) + ch("DEVS", devs))


K2 = [  # como el mapeo de Gabriel: D0 = preescucha + Modifier #2; con M2, el encoder derecho recorre la preescucha
    (65, "Ch15.CC.021", 3200, 4, 0, (M1, 0), (0, 0)),
    (66, "Ch15.CC.021", 211, 4, 0, (M1, 0), (M2, 1)),
    (89, "Ch15.Note.D0", 3137, 0, 0, (M1, 0), (0, 0)),
    (13, "Ch15.Note.C#0", 3329, 3, 2, (M1, 0), (0, 0)),
    (125, "Ch15.Note.D0", M2, 2, 1, (0, 0), (0, 0)),
]


def test_read_devices_and_rebuild_identical():
    b = blob(device("Generic MIDI", "XONE:K2", K2), device("Generic MIDI", "DJM", [K2[0]]))
    devis = m.devices(b)
    assert len(devis) == 2 and m.build(b, devis) == b
    k2 = m.info(devis[0])
    assert (k2.kind, k2.version, k2.in_port, k2.out_port) == ("Generic MIDI", "3.2.0 60", "XONE:K2", "XONE:K2")
    assert len(k2.assignments) == 5 and k2.assigned("Ch15.Note.D0") == 2 and k2.assigned("Ch15.Note.C#0") == 1


def test_assignment_details_with_names_and_conditions():
    rows = {(a.control, a.command): a for a in m.assignments(m.devices(blob(device("Generic MIDI", "K2", K2)))[0])}
    seek = rows[("Ch15.CC.021", 211)]
    assert seek.conditions == [("M1", 0), ("M2", 1)] and "Seek Position (Preview Player)" in seek.describe()
    mod = rows[("Ch15.Note.D0", M2)]
    assert (mod.mode, mod.value, mod.conditions) == (2, 1, []) and "Modifier #2" in mod.describe()
    assert rows[("Ch15.Note.D0", 3137)].describe().endswith("M1=0")


def test_free_a_button_and_remove_one_assignment():
    b = blob(device("Generic MIDI", "XONE:K2", K2))
    devi, removed = m.free_controls(m.devices(b)[0], ["Ch15.Note.D0"])
    assert removed == 2
    devi, removed = m.remove_assignments(devi, "Ch15.CC.021", 211)
    assert removed == 1
    out = m.build(b, [devi])
    k2 = m.info(m.devices(out)[0])                     # vuelve a leerse entero: las cantidades cierran
    assert k2.assigned("Ch15.Note.D0") == 0 and sorted(k2.assignments) == [13, 65]
    assert k2.controls == {65: "Ch15.CC.021", 13: "Ch15.Note.C#0"}
    assert [a.command for a in m.assignments(devi) if a.control == "Ch15.CC.021"] == [3200]
    assert m.free_controls(devi, ["Ch15.Note.D0"])[1] == 0 and m.remove_assignments(devi, "Ch15.CC.021", 211)[1] == 0


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
    i = good.index(b"CMAS") + 8                         # la cantidad de asignaciones
    bad = good[:i] + struct.pack(">I", 6) + good[i + 4:]
    with pytest.raises(ValueError):
        m.info(bad)


def test_the_mappings_in_the_repo():
    k2_path = MAPEOS / "Xone K2 - TRAKTOR ML (D0 libre).tsi"
    k2_blob = m.read_entry(k2_path, "DeviceIO.Config.Controller")
    devis = m.devices(k2_blob)
    assert len(devis) == 1 and m.build(k2_blob, devis) == k2_blob
    k2 = m.info(devis[0])
    assert k2.kind == "Generic MIDI" and k2.in_port == "XONE:K2 MIDI" and len(k2.assignments) == 125
    assert k2.assigned("Ch15.Note.D0") == 0 and k2.assigned("Ch15.Note.C#0") == 1
    rows = m.assignments(devis[0])
    assert not any(a.command == M2 or ("M2", 1) in a.conditions for a in rows)      # nadie fija ni usa M2
    assert {a.command for a in rows if a.control == "Ch15.CC.021"} == {3200, 8}     # la lista sigue
    assert k2_path.read_bytes().count(b"\r\n") == 8   # CRLF como los de NI (y *.tsi -text en git)
    kb = m.info(m.devices(m.read_entry(MAPEOS / "Teclado - TRAKTOR ML.tsi", "DeviceIO.Config.Keyboard"))[0])
    assert kb.kind == "Generic Keyboard" and len(kb.assignments) == 56
