"""
PURPOSE: Prueba de factibilidad de la lista de sugerencias dentro de Traktor. Escribe un XML con el
         formato de la biblioteca de iTunes y una playlist "TRAKTOR ML Sugerencias" de N temas de la
         colección de Traktor (las mismas rutas, para que Traktor use sus cues y su grilla). Traktor
         muestra ese XML en su nodo iTunes; la prueba averigua si lo vuelve a leer sin reiniciarse y
         con qué acción (refresh, colapsar y expandir, un favorito). Cada --round escribe otros N temas,
         con "Ronda R · i/N" en el comentario para reconocer qué ronda está mostrando Traktor.
         No toca collection.nml ni la configuración de Traktor: escribe en la ruta que Traktor ya tiene
         configurada (Browser.Dir.ITunesMusic de Traktor Settings.tsi), y no pisa ni borra un XML
         que no haya escrito este script.
CHANGELOG:
  - 2026-11-04: Creación inicial (prueba del nodo iTunes como lista que se refresca desde Traktor).
  - 2026-11-04: Primera prueba de Gabriel: solo el Refresh del nodo relee el XML (y lo pliega); Traktor
                muestra los datos de su colección y del XML solo llena lo vacío (el comentario de la
                ronda apareció en los temas sin comentario). Porcentaje de prueba en Composer, Grouping
                y Album Artist para ver cuál llega a una columna vacía.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import plistlib
import random
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Optional
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from src.v4.common.traktor_nml import entry_path, find_collection, has_own_cues  # noqa: E402

PLAYLIST_NAME = "TRAKTOR ML Sugerencias"
LIBRARY_ID = "54524B4D4C53554C"          # "TRKMLSUL": marca los XML escritos por este script
SETTINGS_KEY = "Browser.Dir.ITunesMusic"
SEED = 20261104
# Traktor muestra los datos de su colección y llena desde el XML solo los campos vacíos: Comments llega
# a los temas sin comentario. Para el porcentaje se prueban tres campos del XML con un prefijo cada uno,
# para ver cuál cae en una columna vacía de la colección (Comment 2, Producer o Mix).
SCORE_PROBE_KEYS = (("Composer", "C"), ("Grouping", "G"), ("Album Artist", "AA"))
KINDS = {".mp3": "MPEG audio file", ".flac": "FLAC audio file", ".m4a": "AAC audio file",
         ".aiff": "AIFF audio file", ".aif": "AIFF audio file", ".wav": "WAV audio file"}


@dataclass(frozen=True)
class Track:
    path: Path
    title: str
    artist: str
    seconds: float
    bpm: str
    key: str


def settings_itunes_path(tsi: Path) -> Optional[Path]:
    """Ruta del XML de iTunes que Traktor tiene configurada (Preferences > File Management)."""
    for _, el in ET.iterparse(tsi):
        if el.tag == "Entry" and el.get("Name") == SETTINGS_KEY:
            value = (el.get("Value") or "").strip()
            return Path(value) if value else None
        el.clear()
    return None


def collection_tracks(collection: Path, own_cues_only: bool = True) -> List[Track]:
    """Temas de la colección cuyo archivo existe; por defecto solo los que tienen cues propios, para
    que se vea al cargarlos si Traktor usa la entrada de su colección. Orden estable por ruta."""
    out = []
    for e in ET.parse(collection).getroot().find("COLLECTION").findall("ENTRY"):
        if e.find("LOCATION") is None:
            continue
        path = entry_path(e)
        if own_cues_only and not has_own_cues(e):
            continue
        if not path.exists():
            continue
        info = e.find("INFO")
        tempo = e.find("TEMPO")
        out.append(Track(path=path, title=e.get("TITLE", path.stem), artist=e.get("ARTIST", ""),
                         seconds=float(info.get("PLAYTIME_FLOAT") or info.get("PLAYTIME") or 0) if info is not None else 0.0,
                         bpm=f"{float(tempo.get('BPM')):.0f}" if tempo is not None and tempo.get("BPM") else "?",
                         key=info.get("KEY", "?") if info is not None else "?"))
    return sorted(out, key=lambda t: str(t.path).lower())


def pick(tracks: List[Track], round_no: int, n: int, seed: int = SEED) -> List[Track]:
    """Bloque round_no de n temas de un orden mezclado fijo: rondas distintas no repiten temas."""
    order = list(tracks)
    random.Random(seed).shuffle(order)
    block = order[(round_no - 1) * n: round_no * n]
    if len(block) < n:
        raise ValueError(f"La ronda {round_no} necesita {round_no * n} temas y hay {len(tracks)}")
    return block


def file_url(path: Path) -> str:
    """Location de iTunes en Windows: file://localhost/C:/... con espacios y acentos codificados."""
    return "file://localhost/" + quote(PureWindowsPath(str(path)).as_posix(), safe="/:")


def _pid(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16].upper()


def probe_score(rank: int) -> int:
    """Porcentaje de mentira para la prueba: baja con el puesto (95, 92, 89...)."""
    return max(5, 95 - 3 * (rank - 1))


def build_library(tracks: List[Track], round_no: int) -> Dict[str, object]:
    """Biblioteca de iTunes mínima: los temas, la playlist maestra y la de sugerencias en orden."""
    items: Dict[str, object] = {}
    for i, t in enumerate(tracks, start=1):
        tid = 1000 + i
        probe = {key: f"{prefix} {probe_score(i)} %" for key, prefix in SCORE_PROBE_KEYS}
        items[str(tid)] = {
            **probe,
            "Track ID": tid, "Name": t.title, "Artist": t.artist,
            "Kind": KINDS.get(t.path.suffix.lower(), "Audio file"),
            "Total Time": int(round(t.seconds * 1000)),
            "Comments": f"Ronda {round_no} · {i}/{len(tracks)}",
            "Persistent ID": _pid(str(t.path).lower()), "Track Type": "File",
            "Location": file_url(t.path),
        }
    playlist_items = [{"Track ID": 1000 + i} for i in range(1, len(tracks) + 1)]
    return {
        "Major Version": 1, "Minor Version": 1, "Application Version": "12.13.0.9",
        "Date": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, microsecond=0),
        "Features": 5, "Show Content Ratings": True, "Library Persistent ID": LIBRARY_ID,
        "Tracks": items,
        "Playlists": [
            {"Name": "Library", "Master": True, "Playlist ID": 1, "Playlist Persistent ID": _pid("library"),
             "Visible": False, "All Items": True, "Playlist Items": playlist_items},
            {"Name": PLAYLIST_NAME, "Playlist ID": 2, "Playlist Persistent ID": _pid(PLAYLIST_NAME),
             "All Items": True, "Playlist Items": playlist_items},
        ],
    }


def is_ours(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return plistlib.load(f).get("Library Persistent ID") == LIBRARY_ID
    except Exception:
        return False


def write_library(lib: Dict[str, object], out: Path) -> None:
    """Escribe a un temporal y reemplaza de una vez: Traktor nunca lee un archivo a medio escribir."""
    if out.exists() and not is_ours(out):
        raise FileExistsError(f"{out} existe y no lo escribió este script: no lo piso")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as f:
        plistlib.dump(lib, f, fmt=plistlib.FMT_XML, sort_keys=False)
    for attempt in range(10):             # Traktor puede tener el archivo abierto un instante
        try:
            os.replace(tmp, out)
            return
        except PermissionError:
            time.sleep(0.2 * (attempt + 1))
    raise PermissionError(f"No pude reemplazar {out} (¿abierto por otro programa?)")


def remove_library(out: Path) -> bool:
    if not out.exists():
        return False
    if not is_ours(out):
        raise FileExistsError(f"{out} no lo escribió este script: no lo borro")
    out.unlink()
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Lista de sugerencias de prueba en el nodo iTunes de Traktor")
    ap.add_argument("--round", type=int, default=1, help="Ronda: cada una escribe otros temas (1, 2, 3...)")
    ap.add_argument("--n", type=int, default=10, help="Temas por ronda")
    ap.add_argument("--collection", default=None, help="collection.nml (por defecto la de Traktor más reciente)")
    ap.add_argument("--out", default=None, help="XML a escribir (por defecto el que Traktor tiene configurado)")
    ap.add_argument("--any", action="store_true", help="Aceptar temas sin cues propios")
    ap.add_argument("--remove", action="store_true", help="Borrar el XML de prueba (solo si es de este script)")
    ap.add_argument("--dry-run", action="store_true", help="Mostrar la ronda sin escribir")
    args = ap.parse_args()

    collection = Path(args.collection) if args.collection else find_collection()
    if collection is None or not collection.exists():
        print("[ERROR] No encontré collection.nml de Traktor")
        return 1
    out = Path(args.out) if args.out else None
    if out is None:
        tsi = collection.parent / "Traktor Settings.tsi"
        out = settings_itunes_path(tsi) if tsi.exists() else None
    if out is None:
        print("[ERROR] Traktor no tiene ruta de iTunes configurada: pasar --out")
        return 1

    if args.remove:
        print(f"[INFO] {'Borrado' if remove_library(out) else 'No existía'}: {out}")
        return 0

    tracks = collection_tracks(collection, own_cues_only=not args.any)
    chosen = pick(tracks, args.round, args.n)
    print(f"[INFO] Colección: {collection} ({len(tracks)} temas {'con cues propios' if not args.any else ''})")
    print(f"[INFO] Ronda {args.round}, playlist \"{PLAYLIST_NAME}\" en el nodo iTunes de Traktor:")
    for i, t in enumerate(chosen, start=1):
        print(f"  {i:2d}. {t.artist} - {t.title}  [{t.bpm} BPM, {t.key}]")
    if args.dry_run:
        print(f"[INFO] --dry-run: no escribí {out}")
        return 0
    write_library(build_library(chosen, args.round), out)
    print(f"[OK] Escrito {out} ({time.strftime('%H:%M:%S')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
