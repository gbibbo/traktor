"""
PURPOSE: Escritores de playlists para software de DJ a partir de una lista ordenada de playlists:
         - rekordbox.xml (Rekordbox: Preferencias > Avanzado > Base de datos > rekordbox xml, y
           luego "Importar playlist" desde el árbol "rekordbox xml"),
         - NML de Traktor (clic derecho en Playlists > Import Playlist),
         - M3U8 UTF-8 con rutas absolutas (importable en Rekordbox, Traktor y otros).
         Rutas absolutas de Windows tomadas de catalog.source_path (la biblioteca tiene
         subcarpetas, así que carpeta raíz + nombre de archivo ya no alcanza).
CHANGELOG:
  - 2026-09-27: Creación inicial (MVP Rekordbox + Traktor). Filtra caracteres de control prohibidos en XML.
"""
from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import PureWindowsPath
from typing import Dict, List
from urllib.parse import quote

import pandas as pd

_KIND = {".mp3": "MP3 File", ".wav": "WAV File", ".aif": "AIFF File", ".aiff": "AIFF File",
         ".flac": "FLAC File", ".m4a": "M4A File"}


@dataclass
class PlaylistSpec:
    """Una playlist ordenada dentro de una carpeta (folder vacío = raíz del export)."""
    folder: str
    name: str
    track_uids: List[str] = field(default_factory=list)


_XML_INVALID = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff￾￿]")


def _s(value) -> str:
    """Texto seguro para atributos: None/NaN -> ''; sin caracteres prohibidos en XML 1.0
    (algunos tags traen controles, p. ej. un comentario con \\x1a)."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return _XML_INVALID.sub("", str(value))


def rekordbox_location(path: str) -> str:
    """'C:\\Música\\#1 x.mp3' -> 'file://localhost/C:/M%C3%BAsica/%231%20x.mp3'."""
    posix = PureWindowsPath(path).as_posix()
    return "file://localhost/" + quote(posix, safe="/:")


def traktor_location(path: str) -> Dict[str, str]:
    """Partes de LOCATION de Traktor y la PRIMARYKEY ('C:/:dir/:sub/:file.mp3')."""
    p = PureWindowsPath(path)
    volume = p.drive  # 'C:'
    dirs = [d for d in p.parts[1:-1]]
    dir_str = "/:" + "".join(f"{d}/:" for d in dirs)
    return {"VOLUME": volume, "DIR": dir_str, "FILE": p.name, "KEY": f"{volume}{dir_str}{p.name}"}


def _label(row: pd.Series) -> str:
    artist, title = _s(row.get("artist")).strip(), _s(row.get("title")).strip()
    return f"{artist} - {title}" if artist and title else PureWindowsPath(_s(row.get("source_path"))).stem


def write_m3u8(out_path, tracks: pd.DataFrame, uids: List[str]) -> None:
    """M3U extendido UTF-8, rutas absolutas, en el orden de uids."""
    lines = ["#EXTM3U"]
    for uid in uids:
        row = tracks.loc[uid]
        dur = int(round(float(row["duration_s"]))) if pd.notna(row.get("duration_s")) else -1
        lines += [f"#EXTINF:{dur},{_label(row)}", _s(row["source_path"])]
    with open(out_path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")


def _write_xml(root: ET.Element, out_path, declaration: str) -> None:
    """Escribe con la declaración XML exacta que usa cada programa (comillas dobles)."""
    ET.indent(root)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(declaration + "\n")
        f.write(ET.tostring(root, encoding="unicode"))
        f.write("\n")


def _folder_tree(playlists: List[PlaylistSpec]) -> Dict[str, List[PlaylistSpec]]:
    tree: Dict[str, List[PlaylistSpec]] = {}
    for pl in playlists:
        tree.setdefault(pl.folder, []).append(pl)
    return tree


def write_rekordbox_xml(out_path, tracks: pd.DataFrame, playlists: List[PlaylistSpec], root_name: str) -> None:
    """rekordbox.xml con la colección de los temas usados y las playlists bajo la carpeta root_name.
    Sin AverageBpm/Tonality: Rekordbox analiza BPM, grilla y tonalidad al importar."""
    used = list(dict.fromkeys(uid for pl in playlists for uid in pl.track_uids))
    track_id = {uid: i + 1 for i, uid in enumerate(used)}

    root = ET.Element("DJ_PLAYLISTS", Version="1.0.0")
    ET.SubElement(root, "PRODUCT", Name="rekordbox", Version="6.0.0", Company="AlphaTheta")
    coll = ET.SubElement(root, "COLLECTION", Entries=str(len(used)))
    for uid in used:
        row = tracks.loc[uid]
        path = _s(row["source_path"])
        attrs = {
            "TrackID": str(track_id[uid]),
            "Name": _s(row.get("title")) or PureWindowsPath(path).stem,
            "Artist": _s(row.get("artist")),
            "Album": _s(row.get("tag_album")),
            "Genre": _s(row.get("tag_genre")),
            "Label": _s(row.get("tag_label")),
            "Comments": _s(row.get("tag_comment")),
            "Kind": _KIND.get(PureWindowsPath(path).suffix.lower(), ""),
            "Size": str(int(row["filesize_bytes"])) if pd.notna(row.get("filesize_bytes")) else "",
            "TotalTime": str(int(round(float(row["duration_s"])))) if pd.notna(row.get("duration_s")) else "",
            "Location": rekordbox_location(path),
        }
        ET.SubElement(coll, "TRACK", {k: v for k, v in attrs.items() if v != ""})

    pls = ET.SubElement(root, "PLAYLISTS")
    top = ET.SubElement(pls, "NODE", Type="0", Name="ROOT", Count="1")
    tree = _folder_tree(playlists)
    base = ET.SubElement(top, "NODE", Type="0", Name=root_name, Count="0")
    n_children = 0
    for folder, items in tree.items():
        parent = base
        if folder:
            parent = ET.SubElement(base, "NODE", Type="0", Name=folder, Count=str(len(items)))
            n_children += 1
        for pl in items:
            node = ET.SubElement(parent, "NODE", Name=pl.name, Type="1", KeyType="0", Entries=str(len(pl.track_uids)))
            for uid in pl.track_uids:
                ET.SubElement(node, "TRACK", Key=str(track_id[uid]))
            if not folder:
                n_children += 1
    base.set("Count", str(n_children))
    _write_xml(root, out_path, '<?xml version="1.0" encoding="UTF-8"?>')


def write_traktor_nml(out_path, tracks: pd.DataFrame, playlists: List[PlaylistSpec], root_name: str) -> None:
    """NML de Traktor con las entradas usadas y una carpeta root_name con subcarpetas/playlists."""
    used = list(dict.fromkeys(uid for pl in playlists for uid in pl.track_uids))
    root = ET.Element("NML", VERSION="19")
    ET.SubElement(root, "HEAD", COMPANY="www.native-instruments.com", PROGRAM="Traktor")
    ET.SubElement(root, "MUSICFOLDERS")
    coll = ET.SubElement(root, "COLLECTION", ENTRIES=str(len(used)))
    keys = {}
    for uid in used:
        row = tracks.loc[uid]
        loc = traktor_location(_s(row["source_path"]))
        keys[uid] = loc["KEY"]
        entry = ET.SubElement(coll, "ENTRY", TITLE=_s(row.get("title")), ARTIST=_s(row.get("artist")))
        ET.SubElement(entry, "LOCATION", DIR=loc["DIR"], FILE=loc["FILE"], VOLUME=loc["VOLUME"], VOLUMEID="")
        if _s(row.get("tag_album")):
            ET.SubElement(entry, "ALBUM", TITLE=_s(row.get("tag_album")))
        info = {"GENRE": _s(row.get("tag_genre")), "LABEL": _s(row.get("tag_label")),
                "COMMENT": _s(row.get("tag_comment"))}
        if pd.notna(row.get("duration_s")):
            info["PLAYTIME"] = str(int(round(float(row["duration_s"]))))
        ET.SubElement(entry, "INFO", {k: v for k, v in info.items() if v != ""})
    ET.SubElement(root, "SETS", ENTRIES="0")

    pls = ET.SubElement(root, "PLAYLISTS")
    rootnode = ET.SubElement(pls, "NODE", TYPE="FOLDER", NAME="$ROOT")
    rootsub = ET.SubElement(rootnode, "SUBNODES", COUNT="1")
    base = ET.SubElement(rootsub, "NODE", TYPE="FOLDER", NAME=root_name)
    tree = _folder_tree(playlists)
    base_sub = ET.SubElement(base, "SUBNODES", COUNT="0")
    n_children = 0

    def _playlist(parent_sub, pl: PlaylistSpec):
        node = ET.SubElement(parent_sub, "NODE", TYPE="PLAYLIST", NAME=pl.name)
        uuid = hashlib.md5(f"{root_name}/{pl.folder}/{pl.name}".encode("utf-8")).hexdigest()
        plist = ET.SubElement(node, "PLAYLIST", ENTRIES=str(len(pl.track_uids)), TYPE="LIST", UUID=uuid)
        for uid in pl.track_uids:
            e = ET.SubElement(plist, "ENTRY")
            ET.SubElement(e, "PRIMARYKEY", TYPE="TRACK", KEY=keys[uid])

    for folder, items in tree.items():
        if folder:
            fnode = ET.SubElement(base_sub, "NODE", TYPE="FOLDER", NAME=folder)
            fsub = ET.SubElement(fnode, "SUBNODES", COUNT=str(len(items)))
            for pl in items:
                _playlist(fsub, pl)
            n_children += 1
        else:
            for pl in items:
                _playlist(base_sub, pl)
                n_children += 1
    base_sub.set("COUNT", str(n_children))
    _write_xml(root, out_path, '<?xml version="1.0" encoding="UTF-8" standalone="no" ?>')
