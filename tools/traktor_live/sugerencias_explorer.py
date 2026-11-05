"""
PURPOSE: Prueba de factibilidad del refresco sin clics de la lista de sugerencias. El nodo Explorer de
         Traktor lee del disco las playlists (.m3u8, .nml) de cualquier carpeta, y Explorer > Archive
         muestra los .nml de su carpeta History. Si Traktor vuelve a leer el archivo cada vez que se
         selecciona su favorito (comando "Favorites Selector", mapeable a MIDI), un botón del K2 basta
         para ver la lista nueva. Cada --round escribe otros N temas de la colección, con las mismas
         rutas y, en el .nml, las entradas copiadas tal cual de collection.nml, para que Traktor los
         reconozca. No toca collection.nml ni la configuración; no pisa ni borra archivos que no haya
         escrito este script.
CHANGELOG:
  - 2026-11-04: Creación inicial (el refresco del nodo iTunes solo existe en su menú contextual).
  - 2026-11-04: Traktor mostró un solo nodo para el .m3u8 y el .nml con el mismo nombre: cada formato
                lleva su nombre ("... M3U", "... NML", también dentro del .nml) y se borran los viejos.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from src.v4.common.traktor_nml import entry_key, entry_path, find_collection  # noqa: E402
from tools.traktor_live.sugerencias_itunes import PLAYLIST_NAME, Track, collection_tracks, pick  # noqa: E402

MARKER = "TRAKTOR-ML-SUGERENCIAS"               # comentario en el .m3u8
UUID_PREFIX = "54524b4d4c53554c"                # "TRKMLSUL": UUID de la playlist del .nml
DEFAULT_DIR = Path.home() / "Music" / "TRAKTOR ML Sugerencias"


def m3u8_text(tracks: List[Track], round_no: int) -> str:
    """M3U extendido con rutas absolutas de Windows, tal como las guarda la colección."""
    lines = ["#EXTM3U", f"#{MARKER} ronda {round_no}"]
    for t in tracks:
        lines.append(f"#EXTINF:{int(round(t.seconds))},{t.artist} - {t.title}")
        lines.append(str(t.path))
    return "\r\n".join(lines) + "\r\n"


def playlist_uuid(round_no: int) -> str:
    return UUID_PREFIX + f"{round_no:016x}"


def nml_bytes(entries: List[ET.Element], round_no: int, version: str = "20", name: str = PLAYLIST_NAME) -> bytes:
    """Playlist NML como las del History de Traktor: las entradas de la colección (copiadas tal cual,
    con cues y grilla) y una playlist que las lista en orden por su clave de ruta."""
    root = ET.Element("NML", VERSION=version)
    ET.SubElement(root, "HEAD", COMPANY="www.native-instruments.com", PROGRAM="Traktor Pro 4")
    coll = ET.SubElement(root, "COLLECTION", ENTRIES=str(len(entries)))
    for e in entries:
        coll.append(copy.deepcopy(e))
    ET.SubElement(root, "SETS", ENTRIES="0")
    playlists = ET.SubElement(root, "PLAYLISTS")
    folder = ET.SubElement(playlists, "NODE", TYPE="FOLDER", NAME="$ROOT")
    sub = ET.SubElement(folder, "SUBNODES", COUNT="1")
    node = ET.SubElement(sub, "NODE", TYPE="PLAYLIST", NAME=name)
    pl = ET.SubElement(node, "PLAYLIST", ENTRIES=str(len(entries)), TYPE="LIST", UUID=playlist_uuid(round_no))
    for e in entries:
        ET.SubElement(ET.SubElement(pl, "ENTRY"), "PRIMARYKEY", TYPE="TRACK", KEY=entry_key(e))
    ET.SubElement(root, "INDEXING")
    return (b'<?xml version="1.0" encoding="UTF-8" standalone="no" ?>\n'
            + ET.tostring(root, encoding="utf-8", xml_declaration=False))


def is_ours(path: Path) -> bool:
    try:
        if path.suffix.lower() == ".m3u8":
            return f"#{MARKER} " in path.read_text(encoding="utf-8-sig")[:300]
        pl = ET.parse(path).getroot().find(".//PLAYLIST")
        return pl is not None and (pl.get("UUID") or "").startswith(UUID_PREFIX)
    except Exception:
        return False


def write_atomic(data: bytes, out: Path) -> None:
    """Escribe a un temporal y reemplaza de una vez: Traktor nunca lee un archivo a medio escribir."""
    if out.exists() and not is_ours(out):
        raise FileExistsError(f"{out} existe y no lo escribió este script: no lo piso")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_bytes(data)
    for attempt in range(10):                   # Traktor puede tener el archivo abierto un instante
        try:
            os.replace(tmp, out)
            return
        except PermissionError:
            time.sleep(0.2 * (attempt + 1))
    raise PermissionError(f"No pude reemplazar {out} (¿abierto por otro programa?)")


def remove_ours(path: Path) -> bool:
    if not path.exists():
        return False
    if not is_ours(path):
        raise FileExistsError(f"{path} no lo escribió este script: no lo borro")
    path.unlink()
    return True


def entries_by_path(collection: Path) -> Dict[str, ET.Element]:
    out = {}
    for e in ET.parse(collection).getroot().find("COLLECTION").findall("ENTRY"):
        if e.find("LOCATION") is not None:
            out.setdefault(str(entry_path(e)).lower(), e)     # la primera, como hace Traktor
    return out


def targets(out_dir: Path, archive_dir: Optional[Path]) -> List[Path]:
    """Un nombre por formato: Traktor mostró un solo nodo para dos archivos con el mismo nombre."""
    paths = [out_dir / f"{PLAYLIST_NAME} M3U.m3u8", out_dir / f"{PLAYLIST_NAME} NML.nml"]
    if archive_dir is not None:
        paths.append(archive_dir / f"{PLAYLIST_NAME} NML.nml")
    return paths


def legacy_targets(out_dir: Path, archive_dir: Optional[Path]) -> List[Path]:
    """Nombres de la primera versión (los dos formatos con el mismo nombre)."""
    paths = [out_dir / f"{PLAYLIST_NAME}.m3u8", out_dir / f"{PLAYLIST_NAME}.nml"]
    return paths + ([archive_dir / f"{PLAYLIST_NAME}.nml"] if archive_dir is not None else [])


def main() -> int:
    ap = argparse.ArgumentParser(description="Lista de sugerencias de prueba como playlist en el disco (Explorer)")
    ap.add_argument("--round", type=int, default=1, help="Ronda: cada una escribe otros temas (1, 2, 3...)")
    ap.add_argument("--n", type=int, default=10, help="Temas por ronda")
    ap.add_argument("--collection", default=None, help="collection.nml (por defecto la de Traktor más reciente)")
    ap.add_argument("--dir", default=str(DEFAULT_DIR), help="Carpeta de las playlists (.m3u8 y .nml)")
    ap.add_argument("--archive", action="store_true",
                    help="Escribir también el .nml en la carpeta History de Traktor (Explorer > Archive)")
    ap.add_argument("--remove", action="store_true", help="Borrar las playlists de prueba (solo las de este script)")
    ap.add_argument("--dry-run", action="store_true", help="Mostrar la ronda sin escribir")
    args = ap.parse_args()

    collection = Path(args.collection) if args.collection else find_collection()
    if collection is None or not collection.exists():
        print("[ERROR] No encontré collection.nml de Traktor")
        return 1
    out_dir = Path(args.dir)
    archive_dir = collection.parent / "History" if args.archive or args.remove else None
    paths = targets(out_dir, archive_dir)

    if args.remove:
        for p in paths + legacy_targets(out_dir, archive_dir):
            print(f"[INFO] {'Borrado' if remove_ours(p) else 'No existía'}: {p}")
        if out_dir.exists() and not any(out_dir.iterdir()):
            out_dir.rmdir()
        return 0

    tracks = collection_tracks(collection)
    chosen = pick(tracks, args.round, args.n)
    by_path = entries_by_path(collection)
    entries = [by_path[str(t.path).lower()] for t in chosen]
    version = ET.parse(collection).getroot().get("VERSION", "20")
    print(f"[INFO] Ronda {args.round}, playlist \"{PLAYLIST_NAME}\" ({len(tracks)} temas con cues propios):")
    for i, t in enumerate(chosen, start=1):
        print(f"  {i:2d}. {t.artist} - {t.title}  [{t.bpm} BPM]")
    if args.dry_run:
        print("[INFO] --dry-run: no escribí nada")
        return 0
    write_atomic(m3u8_text(chosen, args.round).encode("utf-8"), paths[0])
    for p in paths[1:]:
        write_atomic(nml_bytes(entries, args.round, version, name=p.stem), p)
    for p in legacy_targets(out_dir, archive_dir):
        if p.exists() and is_ours(p):
            p.unlink()
            print(f"[INFO] Borré el nombre viejo {p}")
    for p in paths:
        print(f"[OK] Escrito {p}")
    print(f"[OK] Ronda {args.round} a las {time.strftime('%H:%M:%S')}: el primer tema es {chosen[0].artist} - {chosen[0].title}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
