"""
PURPOSE: Exporta mapeos de controlador o de teclado de Traktor a un .tsi propio, que otro ordenador
         importa desde Preferences > Controller Manager, y libera los botones que se usan para otra
         cosa (por ejemplo el D0 del K2, que abre las sugerencias en vivo). Lee la configuración de
         Traktor (Traktor Settings.tsi) o cualquier .tsi, y escribe un archivo con el mismo formato que
         los mapeos de NI: solo la entrada DeviceIO.Config.Controller o DeviceIO.Config.Keyboard, más
         Flavour y Version. No toca la configuración de Traktor.
         Formato (big-endian): bloque = 4 letras + largo uint32 + contenido; texto = uint32 de
         caracteres + UTF-16BE. DIOM > DIOI (versión) y DEVS (cantidad + DEVI). DEVI = tipo (texto) +
         DDAT > DDIF, DDIV (versión), DDIC (comentario), DDPT (puertos), DDDC (controles posibles),
         DDCB > CMAS (cantidad + CMAI: una asignación, que empieza con el id de su control) y DCBM
         (cantidad + DCBM: id del control + su nombre, p. ej. "Ch15.Note.D0"), DVST.
         CMAI = id del control, tipo, número de comando y un bloque CMAD con enteros de 4 bytes. Campos
         del CMAD deducidos del mapeo de Gabriel (2026-11-06; coinciden con lo que muestra Traktor):
         2 = modo, 3 = deck, 11 = valor, 12 = comentario (texto) y, después del comentario, la
         condición 1 (modificador, valor) y la condición 2 (valor, modificador). Modificador #n =
         comando 2547 + n.
CHANGELOG:
  - 2026-11-05: Creación: el mapeo del K2 de Gabriel (con D0 libre) y su teclado van al repo.
  - 2026-11-06: --detalle (asignaciones de un control con nombre de comando y condiciones) y --quitar
                CONTROL:COMANDO (una sola asignación). D0 tenía, además de la preescucha, el Modifier #2
                que habilita "Seek Position" de la preescucha en el encoder derecho: se quita también.
"""
from __future__ import annotations

import argparse
import base64
import struct
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from src.v4.common.traktor_nml import find_collection  # noqa: E402

ENTRIES = {"controlador": "DeviceIO.Config.Controller", "teclado": "DeviceIO.Config.Keyboard"}
# Nombres de los comandos que aparecen en el mapeo del K2 de Gabriel. Datos sueltos, revisados contra
# commands.xml de github.com/ivanz/Traktor.Mapping (AGPL: no se copia el archivo).
COMMAND_NAMES = {
    8: "Monitor Volume Adjust", 17: "Monitor Mix Adjust", 100: "Play/Pause", 211: "Seek Position (Preview Player)",
    319: "Filter On", 320: "Filter Adjust", 2351: "Move", 3076: "Load Selected (Deck)",
    3137: "Load Selected (Preview Player)", 3200: "Select Up/Down (Browser List)",
    3328: "Select Up/Down (Browser Tree)", 3329: "Select Expand/Collapse (Browser Tree)",
    **{2547 + n: f"Modifier #{n}" for n in range(1, 9)},
}
Chunk = Tuple[str, bytes]


def split(b: bytes) -> List[Chunk]:
    """Bloques seguidos (etiqueta, contenido); falla si el largo no cierra exacto."""
    out, i = [], 0
    while i < len(b):
        if i + 8 > len(b):
            raise ValueError(f"bloque cortado en {i}")
        tag = b[i:i + 4].decode("ascii")
        n = struct.unpack(">I", b[i + 4:i + 8])[0]
        if i + 8 + n > len(b):
            raise ValueError(f"bloque {tag} más largo que su contenedor")
        out.append((tag, b[i + 8:i + 8 + n]))
        i += 8 + n
    return out


def join(chunks: Sequence[Chunk]) -> bytes:
    return b"".join(tag.encode("ascii") + struct.pack(">I", len(body)) + body for tag, body in chunks)


def wstr(b: bytes, i: int = 0) -> Tuple[str, int]:
    n = struct.unpack(">I", b[i:i + 4])[0]
    return b[i + 4:i + 4 + 2 * n].decode("utf-16-be"), i + 4 + 2 * n


def counted(body: bytes) -> Tuple[int, List[Chunk]]:
    """Contenedor con cantidad: uint32 + bloques (DEVS, CMAS, DCBM). Revisa que la cantidad cierre."""
    count = struct.unpack(">I", body[:4])[0]
    items = split(body[4:])
    if count != len(items):
        raise ValueError(f"dice {count} elementos y tiene {len(items)}")
    return count, items


def counted_body(items: Sequence[Chunk]) -> bytes:
    return struct.pack(">I", len(items)) + join(items)


def replace(chunks: List[Chunk], tag: str, body: bytes) -> List[Chunk]:
    hits = [i for i, (t, _) in enumerate(chunks) if t == tag]
    if len(hits) != 1:
        raise ValueError(f"esperaba un {tag} y hay {len(hits)}")
    out = list(chunks)
    out[hits[0]] = (tag, body)
    return out


def body_of(chunks: Sequence[Chunk], tag: str) -> bytes:
    hits = [b for t, b in chunks if t == tag]
    if len(hits) != 1:
        raise ValueError(f"esperaba un {tag} y hay {len(hits)}")
    return hits[0]


# ---------------------------------------------------------------- dispositivos

def devices(blob: bytes) -> List[bytes]:
    """Contenido de cada DEVI de un DIOM."""
    (tag, diom), = split(blob)
    if tag != "DIOM":
        raise ValueError(f"no es un DIOM sino {tag}")
    _, items = counted(body_of(split(diom), "DEVS"))
    return [b for t, b in items if t == "DEVI"]


def build(blob: bytes, devis: Sequence[bytes]) -> bytes:
    """El mismo DIOM (su DIOI) con estos dispositivos."""
    (_, diom), = split(blob)
    parts = replace(split(diom), "DEVS", counted_body([("DEVI", d) for d in devis]))
    return join([("DIOM", join(parts))])


def _devi_parts(devi: bytes) -> Tuple[bytes, List[Chunk]]:
    _, k = wstr(devi)
    return devi[:k], split(devi[k:])


@dataclass
class DeviceInfo:
    kind: str
    version: str
    comment: str
    in_port: str
    out_port: str
    controls: Dict[int, str]          # id -> "Ch15.Note.D0"
    assignments: List[int]            # id del control de cada asignación

    def assigned(self, name: str) -> int:
        ids = {i for i, n in self.controls.items() if n == name}
        return sum(1 for a in self.assignments if a in ids)


def info(devi: bytes) -> DeviceInfo:
    head, parts = _devi_parts(devi)
    ddat = split(body_of(parts, "DDAT"))
    in_port, k = wstr(body_of(ddat, "DDPT"))
    out_port, _ = wstr(body_of(ddat, "DDPT"), k)
    controls: Dict[int, str] = {}
    assignments: List[int] = []
    ddcb = [c for c in ddat if c[0] == "DDCB"]
    if ddcb:
        cb = split(ddcb[0][1])
        _, cmas = counted(body_of(cb, "CMAS"))
        assignments = [struct.unpack(">I", b[:4])[0] for _, b in cmas]
        _, dcbm = counted(body_of(cb, "DCBM"))
        for _, b in dcbm:
            controls[struct.unpack(">I", b[:4])[0]] = wstr(b, 4)[0]
    return DeviceInfo(kind=wstr(head)[0], version=wstr(body_of(ddat, "DDIV"))[0],
                      comment=wstr(body_of(ddat, "DDIC"))[0], in_port=in_port, out_port=out_port,
                      controls=controls, assignments=assignments)


def free_controls(devi: bytes, names: Sequence[str]) -> Tuple[bytes, int]:
    """El dispositivo sin las asignaciones de esos controles (y sin sus definiciones en DCBM). Devuelve
    el dispositivo nuevo y cuántas asignaciones quitó. Los ids no se renumeran: no van seguidos."""
    wanted = set(names)
    return _drop(devi, lambda name, _cmd: name in wanted)


@dataclass
class Assignment:
    control: str
    command: int
    mode: int
    deck: int
    value: int
    conditions: List[Tuple[str, int]]     # [("M1", 0), ("M2", 1)]

    def describe(self) -> str:
        name = COMMAND_NAMES.get(self.command, "comando sin nombre conocido")
        cond = " y ".join(f"{m}={v}" for m, v in self.conditions) or "sin condición"
        return f"{self.control}: {name} [{self.command}] modo {self.mode}, deck {self.deck}, valor {self.value}, {cond}"


def _modifier(cmd: int) -> Optional[str]:
    return f"M{cmd - 2547}" if 2548 <= cmd <= 2555 else None


def assignments(devi: bytes) -> List[Assignment]:
    """Cada asignación con su control, comando, modo, deck, valor y condiciones de modificador."""
    _, parts = _devi_parts(devi)
    cb = split(body_of(split(body_of(parts, "DDAT")), "DDCB"))
    _, dcbm = counted(body_of(cb, "DCBM"))
    names = {struct.unpack(">I", b[:4])[0]: wstr(b, 4)[0] for _, b in dcbm}
    out = []
    for _, b in counted(body_of(cb, "CMAS"))[1]:
        cid, _kind, cmd = struct.unpack(">III", b[:12])
        (_, cmad), = split(b[12:])
        head = struct.unpack(">12i", cmad[:48])
        _, k = wstr(cmad, 48)                        # comentario
        c1_mod, c1_val, c2_val, c2_mod = struct.unpack(">4i", cmad[k:k + 16])
        conds = [(m, v) for m, v in ((_modifier(c1_mod), c1_val), (_modifier(c2_mod), c2_val)) if m]
        out.append(Assignment(names.get(cid, "?"), cmd, head[2], head[3], head[11], conds))
    return out


def _drop(devi: bytes, gone_assignment) -> Tuple[bytes, int]:
    """El dispositivo sin las asignaciones para las que gone_assignment(nombre, comando) es verdad, y
    sin las definiciones DCBM que ya no use ninguna."""
    head, parts = _devi_parts(devi)
    ddat = split(body_of(parts, "DDAT"))
    cb = split(body_of(ddat, "DDCB"))
    _, dcbm = counted(body_of(cb, "DCBM"))
    names = {struct.unpack(">I", b[:4])[0]: wstr(b, 4)[0] for _, b in dcbm}
    _, cmas = counted(body_of(cb, "CMAS"))
    keep_cmas = []
    for t, b in cmas:
        cid, _kind, cmd = struct.unpack(">III", b[:12])
        if not gone_assignment(names.get(cid, ""), cmd):
            keep_cmas.append((t, b))
    used = {struct.unpack(">I", b[:4])[0] for _, b in keep_cmas}
    keep_dcbm = [(t, b) for t, b in dcbm if struct.unpack(">I", b[:4])[0] in used]
    cb = replace(replace(cb, "CMAS", counted_body(keep_cmas)), "DCBM", counted_body(keep_dcbm))
    ddat = replace(ddat, "DDCB", join(cb))
    parts = replace(parts, "DDAT", join(ddat))
    return head + join(parts), len(cmas) - len(keep_cmas)


def remove_assignments(devi: bytes, control: str, command: int) -> Tuple[bytes, int]:
    """Quita solo las asignaciones de ese control con ese comando (el resto del control queda)."""
    return _drop(devi, lambda name, cmd: name == control and cmd == command)


def set_ports(devi: bytes, in_port: str, out_port: str) -> bytes:
    def w(s: str) -> bytes:
        return struct.pack(">I", len(s)) + s.encode("utf-16-be")
    head, parts = _devi_parts(devi)
    ddat = replace(split(body_of(parts, "DDAT")), "DDPT", w(in_port) + w(out_port))
    return head + join(replace(parts, "DDAT", join(ddat)))


# ---------------------------------------------------------------- archivos .tsi

def read_entry(tsi: Path, entry: str) -> bytes:
    for e in ET.parse(tsi).getroot().iter("Entry"):
        if e.get("Name") == entry:
            return base64.b64decode(e.get("Value"))
    raise ValueError(f"{tsi} no tiene {entry}")


def tsi_text(entry: str, blob: bytes) -> str:
    """Como los mapeos de NI: XML con la entrada, Flavour -1 y Version 0, finales CRLF."""
    lines = ['<?xml version="1.0" encoding="utf-8" standalone="no"?>', "<NIXML>", "  <TraktorSettings>",
             f'    <Entry Name="{entry}" Type="3" Value="{base64.b64encode(blob).decode("ascii")}"></Entry>',
             '    <Entry Name="Flavour" Type="1" Value="-1"></Entry>',
             '    <Entry Name="Version" Type="1" Value="0"></Entry>', "  </TraktorSettings>", "</NIXML>", ""]
    return "\r\n".join(lines)


def traktor_settings() -> Optional[Path]:
    """Traktor Settings.tsi de la versión de Traktor cuya colección es la más reciente."""
    collection = find_collection()
    if collection is None:
        return None
    p = collection.parent / "Traktor Settings.tsi"
    return p if p.exists() else None


def describe(i: int, d: DeviceInfo, controls: Sequence[str]) -> str:
    text = (f"  {i}: {d.kind} «{d.comment}» (versión {d.version}), puertos {d.in_port or '-'} / "
            f"{d.out_port or '-'}, {len(d.assignments)} asignaciones")
    for c in controls:
        text += f"; {c}: {d.assigned(c)}"
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description="Exporta un mapeo de Traktor a .tsi (y libera botones)")
    ap.add_argument("--desde", default=None, help="Traktor Settings.tsi u otro .tsi (por defecto el de Traktor)")
    ap.add_argument("--tipo", choices=sorted(ENTRIES), default="controlador")
    ap.add_argument("--dispositivo", type=int, default=None, help="Índice del dispositivo (ver --listar)")
    ap.add_argument("--liberar", action="append", default=[],
                    help='Control a dejar sin asignaciones, p. ej. "Ch15.Note.D0" (se puede repetir)')
    ap.add_argument("--quitar", action="append", default=[],
                    help='Una sola asignación, "CONTROL:COMANDO", p. ej. "Ch15.CC.021:211" (se puede repetir)')
    ap.add_argument("--detalle", action="append", default=[],
                    help='Mostrar las asignaciones de un control ("todos" = todas) y salir')
    ap.add_argument("--puertos", default=None, help='Puertos de entrada y salida, "Entrada|Salida"')
    ap.add_argument("--salida", default=None, help="Archivo .tsi a escribir")
    ap.add_argument("--listar", action="store_true", help="Mostrar los dispositivos y salir")
    args = ap.parse_args()

    src = Path(args.desde) if args.desde else traktor_settings()
    if src is None or not src.exists():
        print("[ERROR] No encontré Traktor Settings.tsi: indicarlo con --desde")
        return 1
    entry = ENTRIES[args.tipo]
    blob = read_entry(src, entry)
    devis = devices(blob)
    print(f"[INFO] {src} ({entry}):")
    for i, d in enumerate(devis):
        print(describe(i, info(d), args.liberar))
    if args.detalle:
        for i, d in enumerate(devis):
            if args.dispositivo is not None and i != args.dispositivo:
                continue
            rows = [a for a in assignments(d) if "todos" in args.detalle or a.control in args.detalle]
            for a in sorted(rows, key=lambda a: (a.control, a.command)):
                print(f"  [{i}] {a.describe()}")
        return 0
    if args.listar:
        return 0
    if args.dispositivo is None or not 0 <= args.dispositivo < len(devis) or not args.salida:
        print("[ERROR] Indicar --dispositivo (un índice de la lista) y --salida")
        return 1
    devi = devis[args.dispositivo]
    if args.liberar:
        devi, removed = free_controls(devi, args.liberar)
        print(f"[OK] Quité {removed} asignaciones de {', '.join(args.liberar)}")
    for item in args.quitar:
        control, _, command = item.rpartition(":")
        if not control or not command.isdigit():
            print(f'[ERROR] --quitar espera "CONTROL:COMANDO", p. ej. "Ch15.CC.021:211", no «{item}»')
            return 1
        devi, removed = remove_assignments(devi, control, int(command))
        print(f"[OK] Quité {removed} asignación(es) de {control} con "
              f"{COMMAND_NAMES.get(int(command), 'el comando ' + command)}")
    if args.puertos:
        in_port, _, out_port = args.puertos.partition("|")
        devi = set_ports(devi, in_port, out_port or in_port)
    out_blob = build(blob, [devi])
    check = info(devices(out_blob)[0])                    # el archivo nuevo se vuelve a leer entero
    out = Path(args.salida)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(tsi_text(entry, out_blob).encode("utf-8"))
    print(f"[OK] {out}:")
    print(describe(0, check, args.liberar))
    return 0


if __name__ == "__main__":
    sys.exit(main())
