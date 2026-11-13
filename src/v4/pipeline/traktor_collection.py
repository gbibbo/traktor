"""
PURPOSE: Pone las playlists de un paquete portátil (usb_package.py) directamente en la colección de
         Traktor de esta computadora (collection.nml), con Traktor cerrado. Hace falta porque
         "Import Playlist" de Traktor toma una sola playlist de un NML con carpetas, y porque un tema
         importado desde un NML queda con los datos del NML, sin leer la carátula del archivo
         (Traktor la guarda en su caché propia solo al agregar el archivo desde el disco).
         Dos pasos, cada uno con respaldo previo de collection.nml:
           - prepare: de los temas del paquete que ya están en la colección, saca los que no tienen
             carátula en Traktor pero sí en el archivo, y no tienen cue points propios (Traktor los
             vuelve a leer, con carátula, al agregar la carpeta Musica); a los demás les actualiza
             título, artista, álbum, género, sello, mezcla y fecha desde los tags del archivo, sin
             tocar análisis, cues ni carátula.
           - playlists: agrega (o reemplaza) la carpeta del paquete con todas sus playlists en orden,
             apuntando a las entradas de la colección; si falta algún tema en la colección, no
             escribe nada y lo informa.
         Entre los dos pasos, en Traktor: arrastrar la carpeta Musica del paquete a Track Collection.
         Aparte, bpm-doble: los temas del paquete cuyo BPM de Traktor es la mitad (o el doble) del de sus tags
         pasan al BPM del tag (Gabriel, 2026-11-13: corregirlos en Traktor).
CHANGELOG:
  - 2026-10-30: Creación inicial (las playlists del pendrive no entraban en Traktor 4.4).
  - 2026-11-04: find_collection, traktor_running, entry_key, entry_path y _has_own_cues pasan a
                src/v4/common/traktor_nml.py (sin pandas, para las herramientas en vivo).
  - 2026-11-13: bpm-doble (26 temas con el BPM a la mitad no pasaban el filtro de BPM del vivo).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.tags import read_release_tags, read_tags  # noqa: E402
from src.v4.common.traktor_nml import (  # noqa: E402,F401
    entry_key, entry_path, find_collection, has_own_cues as _has_own_cues, traktor_running,
)
from src.v4.pipeline.usb_package import _root_label, nml_dir_prefix, read_package_specs  # noqa: E402

DECLARATION = '<?xml version="1.0" encoding="UTF-8" standalone="no" ?>'
HALF_BPM_TOL = 0.03                     # bpm-doble: x2 o x0,5 del BPM de Traktor a un 3 % de la referencia
KEEP_BPM = (90.0, 180.0)                # un BPM de Traktor en este rango no se toca
CLUB_BPM = (90.0, 160.0)                # y el corregido tiene que caer en este


def _k(s: str) -> str:
    """Clave de comparación de rutas: NFC y minúsculas (Windows no distingue mayúsculas)."""
    return unicodedata.normalize("NFC", s).lower()


def load(path: Path) -> ET.ElementTree:
    return ET.parse(path)


def save(tree: ET.ElementTree, path: Path) -> None:
    """Escribe con la declaración de Traktor, a un temporal y después reemplaza."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(DECLARATION + "\n")
        f.write(ET.tostring(tree.getroot(), encoding="unicode"))
        f.write("\n")
    ET.parse(tmp)                       # se vuelve a leer antes de reemplazar
    os.replace(tmp, path)


def backup(path: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    dst = backup_dir / f"collection_{time.strftime('%Y%m%d_%H%M%S')}.nml"
    shutil.copy2(path, dst)
    return dst


def package_prefix(package_dir: Path) -> str:
    vol, dirp = nml_dir_prefix(PureWindowsPath(str(package_dir)))
    return vol + dirp + "Musica/:"


def _file_has_cover(path: Path) -> bool:
    from src.v4.pipeline.package_tags import cover_state
    return path.exists() and cover_state(path) in ("ok", "mime")


def _release_date(tag_date: Optional[str]) -> str:
    """'2020-11-12' -> '2020/11/12' (formato de Traktor); otra cosa -> ''."""
    import re
    m = re.match(r"\s*(\d{4})-(\d{2})-(\d{2})", tag_date or "")
    return f"{int(m.group(1))}/{int(m.group(2))}/{int(m.group(3))}" if m else ""


def update_entry(e: ET.Element, path: Path) -> List[str]:
    """Texto de la entrada desde los tags del archivo. Devuelve los campos que cambiaron."""
    t, rel = read_tags(path), read_release_tags(path)
    changed = []

    def put(node: ET.Element, attr: str, value: Optional[str], name: str):
        if value and node.get(attr) != value:
            node.set(attr, value)
            changed.append(name)

    put(e, "TITLE", t["tag_title"], "title")
    put(e, "ARTIST", t["tag_artist"], "artist")
    if t["tag_album"]:
        album = e.find("ALBUM")
        if album is None:
            album = ET.Element("ALBUM")
            album.tail = "\n"
            e.insert(list(e).index(e.find("LOCATION")) + 1, album)
        put(album, "TITLE", t["tag_album"], "album")
    info = e.find("INFO")
    if info is None:
        info = ET.SubElement(e, "INFO")
        info.tail = "\n"
    put(info, "GENRE", t["tag_genre"], "genre")
    put(info, "LABEL", t["tag_label"], "label")
    put(info, "REMIXER", rel["tag_remixer"], "remixer")
    put(info, "RELEASE_DATE", _release_date(rel["tag_date"]), "released")
    return changed


def prepare(collection: Path, package_dir: Path, backup_dir: Path, dry_run: bool = False) -> Dict[str, int]:
    tree = load(collection)
    root = tree.getroot()
    coll = root.find("COLLECTION")
    prefix = _k(package_prefix(package_dir))
    removed, kept, updated = [], 0, 0
    for e in list(coll.findall("ENTRY")):
        if not _k(entry_key(e)).startswith(prefix):
            continue
        path = entry_path(e)
        info = e.find("INFO")
        no_cover = info is None or not info.get("COVERARTID")
        if no_cover and not _has_own_cues(e) and _file_has_cover(path):
            coll.remove(e)
            removed.append(_k(entry_key(e)))
            continue
        kept += 1
        if path.exists() and update_entry(e, path):
            updated += 1
    # referencias a las entradas sacadas, en playlists del usuario
    refs = 0
    gone = set(removed)
    for pl in root.iter("PLAYLIST"):
        for ent in list(pl.findall("ENTRY")):
            pk = ent.find("PRIMARYKEY")
            if pk is not None and _k(pk.get("KEY", "")) in gone:
                pl.remove(ent)
                refs += 1
        pl.set("ENTRIES", str(len(pl.findall("ENTRY"))))
    coll.set("ENTRIES", str(len(coll.findall("ENTRY"))))
    stats = {"removed": len(removed), "kept": kept, "updated": updated, "playlist_refs_removed": refs}
    if not dry_run:
        backup(collection, backup_dir)
        save(tree, collection)
    return stats


def _folder(name: str) -> Tuple[ET.Element, ET.Element]:
    node = ET.Element("NODE", TYPE="FOLDER", NAME=name)
    sub = ET.SubElement(node, "SUBNODES", COUNT="0")
    node.tail = sub.tail = "\n"
    return node, sub


def add_playlists(collection: Path, package_dir: Path, backup_dir: Path,
                  dry_run: bool = False) -> Dict[str, object]:
    tree = load(collection)
    root = tree.getroot()
    keys = {_k(entry_key(e)): entry_key(e) for e in root.find("COLLECTION").findall("ENTRY")}
    org, specs, placements = read_package_specs(package_dir)
    rel_by_uid = {p.track_uid: p.rel_path for p in placements}
    vol, dirp = nml_dir_prefix(PureWindowsPath(str(package_dir)))

    def key_of(uid: str) -> str:
        return vol + dirp + rel_by_uid[uid].replace("\\", "/:")

    missing = sorted({key_of(u) for s in specs for u in s.track_uids if _k(key_of(u)) not in keys})
    stats: Dict[str, object] = {"playlists": len(specs), "tracks": len({u for s in specs for u in s.track_uids}),
                                "missing": len(missing), "missing_examples": missing[:5]}
    if missing:
        return stats

    label = _root_label(package_dir)
    top = root.find("PLAYLISTS/NODE")                       # $ROOT
    top_sub = top.find("SUBNODES")
    for old in [n for n in top_sub.findall("NODE") if n.get("NAME") == label]:
        top_sub.remove(old)
    base, base_sub = _folder(label)
    folders: Dict[str, ET.Element] = {}
    for s in specs:
        parent = base_sub
        if s.folder:
            if s.folder not in folders:
                fnode, folders[s.folder] = _folder(s.folder)
                base_sub.append(fnode)
            parent = folders[s.folder]
        node = ET.SubElement(parent, "NODE", TYPE="PLAYLIST", NAME=s.name)
        uuid = hashlib.md5(f"{label}/{s.folder}/{s.name}".encode("utf-8")).hexdigest()
        plist = ET.SubElement(node, "PLAYLIST", ENTRIES=str(len(s.track_uids)), TYPE="LIST", UUID=uuid)
        node.tail = plist.tail = "\n"
        for uid in s.track_uids:
            ent = ET.SubElement(plist, "ENTRY")
            pk = ET.SubElement(ent, "PRIMARYKEY", TYPE="TRACK", KEY=keys[_k(key_of(uid))])
            ent.tail = pk.tail = "\n"
    for sub in [base_sub, *folders.values()]:
        sub.set("COUNT", str(len(sub.findall("NODE"))))
    top_sub.append(base)
    top_sub.set("COUNT", str(len(top_sub.findall("NODE"))))
    stats["folder"] = label
    if not dry_run:
        backup(collection, backup_dir)
        save(tree, collection)
    return stats


def octave_fix(traktor_bpm: float, ref_bpm: Optional[float], tol: float = HALF_BPM_TOL) -> Optional[float]:
    """El BPM corregido si Traktor leyó la mitad o el doble del de referencia (dentro de tol); None si no.
    Solo mueve un BPM fuera de 90-180 a uno de pista (90-160): en 9 temas de la biblioteca la referencia era la
    que estaba a la mitad (Dax J, «The Invisible Man»: Traktor 133, tags 66,5), y ese BPM de Traktor no se toca."""
    if not traktor_bpm or not ref_bpm or traktor_bpm <= 0 or ref_bpm <= 0:
        return None
    if KEEP_BPM[0] <= traktor_bpm <= KEEP_BPM[1]:
        return None
    for factor in (2.0, 0.5):
        fixed = traktor_bpm * factor
        if abs(fixed - ref_bpm) <= tol * ref_bpm and CLUB_BPM[0] <= fixed <= CLUB_BPM[1]:
            return fixed
    return None


def reference_bpm(package_dir: Path, data_dir: Path) -> Dict[str, float]:
    """BPM de referencia por clave de entrada del paquete: el de Beatport confirmado y, si no hay, el de los tags
    originales (features/bpm_key.parquet). Los tags del paquete no sirven: ya llevan el BPM que leyó Traktor."""
    import pandas as pd
    by_uid: Dict[str, float] = {}
    bk = data_dir / "features" / "bpm_key.parquet"
    if bk.exists():
        for uid, bpm in pd.read_parquet(bk, columns=["track_uid", "bpm"]).itertuples(index=False):
            if bpm == bpm and bpm:
                by_uid[str(uid)] = float(bpm)
    bp = data_dir / "features" / "beatport.parquet"
    if bp.exists():
        table = pd.read_parquet(bp, columns=["track_uid", "confirmed", "bp_bpm"])
        for uid, ok, bpm in table.itertuples(index=False):
            if ok is True and bpm == bpm and bpm:
                by_uid[str(uid)] = float(bpm)
    _, _, placements = read_package_specs(package_dir)
    vol, dirp = nml_dir_prefix(PureWindowsPath(str(package_dir)))
    return {_k(vol + dirp + p.rel_path.replace("\\", "/:")): by_uid[p.track_uid]
            for p in placements if p.track_uid in by_uid}


def fix_half_bpm(collection: Path, package_dir: Path, backup_dir: Path, reference: Dict[str, float],
                 dry_run: bool = False) -> Dict[str, object]:
    """Temas del paquete con el BPM de Traktor a la mitad (o al doble) del de referencia: el BPM de Traktor pasa a
    ser el doble (o la mitad). Es lo mismo que el botón x2 de la grilla: el ancla de la grilla no se mueve."""
    tree = load(collection)
    prefix = _k(package_prefix(package_dir))
    changes: List[Tuple[str, float, float]] = []
    for e in tree.getroot().find("COLLECTION").findall("ENTRY"):
        key = _k(entry_key(e))
        if not key.startswith(prefix):
            continue
        tempo = e.find("TEMPO")
        try:
            bpm = float(tempo.get("BPM")) if tempo is not None and tempo.get("BPM") else 0.0
        except ValueError:
            continue
        path = entry_path(e)
        fixed = octave_fix(bpm, reference.get(key))
        if fixed is not None:
            tempo.set("BPM", f"{fixed:.6f}")
            changes.append((path.name, round(bpm, 2), round(fixed, 2)))
    stats: Dict[str, object] = {"changed": len(changes), "examples": changes}
    if changes and not dry_run:
        stats["backup"] = str(backup(collection, backup_dir))
        save(tree, collection)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Playlists de un paquete portátil en la colección de Traktor")
    ap.add_argument("cmd", choices=("prepare", "playlists", "bpm-doble"))
    ap.add_argument("--package", required=True, help="Carpeta del paquete, p. ej. C:\\TRAKTOR ML")
    ap.add_argument("--collection", default=None, help="collection.nml (por defecto la de Traktor más reciente)")
    ap.add_argument("--backup-dir", default=str(REPO_ROOT / "artifacts" / "v4" / "traktor_backups"))
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "artifacts" / "v4" / "datasets" / "musica"),
                    help="bpm-doble: carpeta del dataset con features/bpm_key.parquet y beatport.parquet")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    collection = Path(args.collection) if args.collection else find_collection()
    if collection is None or not collection.exists():
        print("[ERROR] No encontré collection.nml de Traktor")
        return 1
    if not args.dry_run and traktor_running():
        print("[ERROR] Traktor está abierto: cerrarlo antes (guarda la colección al salir)")
        return 1
    package = Path(args.package)
    print(f"[INFO] Colección: {collection}")
    if args.cmd == "prepare":
        stats = prepare(collection, package, Path(args.backup_dir), args.dry_run)
    elif args.cmd == "bpm-doble":
        reference = reference_bpm(package, Path(args.data_dir))
        print(f"[INFO] BPM de referencia para {len(reference)} temas del paquete")
        stats = fix_half_bpm(collection, package, Path(args.backup_dir), reference, args.dry_run)
        for name, old, new in stats["examples"]:
            print(f"   {old:>7} -> {new:>7}  {name}")
        stats = {k: v for k, v in stats.items() if k != "examples"}
    else:
        stats = add_playlists(collection, package, Path(args.backup_dir), args.dry_run)
    print(f"[INFO] {stats}")
    if stats.get("missing"):
        print("[ERROR] Hay temas del paquete que no están en la colección: en Traktor, arrastrar la carpeta "
              f"Musica de {package} a Track Collection y cerrar Traktor antes de este paso")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
