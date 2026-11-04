"""
PURPOSE: Paquete portátil de una organización para tocar con Rekordbox o Traktor desde un pendrive o
         disco externo en cualquier computadora Windows. Copia cada tema a
         <destino>/<paquete>/Musica/<carpeta>/<playlist>/<archivo> (la música queda ordenada como
         las playlists) y escribe:
           - traktor.nml con las rutas del destino actual (Import Playlist en Traktor),
           - _plantilla.nml + _preparar_traktor.ps1 (traktor_prepare.ps1) + "Preparar Traktor.bat":
             en otra computadora arman traktor.nml con los temas que ya están en su colección de
             Traktor (ya analizados; ficha copiada tal cual) y, para los que no, la copia del
             paquete en la letra y carpeta donde esté,
           - rekordbox.xml con las rutas del destino actual, y _plantilla_rekordbox.xml +
             _preparar_rekordbox.ps1 (rekordbox_prepare.ps1) + "Preparar Rekordbox.bat", que lo
             rearman con la letra y la carpeta donde esté el paquete en otra computadora,
           - Playlists m3u8/ con rutas relativas (respaldo para otros programas),
           - _organizacion.csv (playlist, posición, track_uid, archivo) y LEEME.txt.
         Reanudable: no vuelve a copiar archivos que ya están con el mismo tamaño. Verifica tamaños
         siempre y, con --verify hash, el SHA256 de cada copia contra el original.
         --refresh <paquete> rehace todo lo que no es audio desde el propio paquete
         (_organizacion.csv y los tags actuales de cada archivo), sin la organización ni la
         biblioteca original: para después de corregir los tags con package_tags.py.
CHANGELOG:
  - 2026-11-04: --refresh <paquete> --from-org: rehace playlists, NML, rekordbox.xml, m3u8 y _organizacion.csv
                desde la versión actual de la organización cuando la biblioteca ES el paquete (package_library.py):
                cada tema queda en el archivo donde está, aunque cambie de playlist (moverlo podría perder sus
                cue points en Traktor o Rekordbox). Nunca copia ni borra audio.
  - 2026-10-30: Rekordbox en el paquete (plantilla + Preparar Rekordbox.bat) y --refresh (con
                --refresh, artista, título, mezcla y año del NML/XML salen de los tags de cada
                archivo). El tamaño del LEEME sale de las copias del paquete. El LEEME avisa que
                Traktor importa una sola playlist del NML y remite a traktor_collection.py;
                read_package_specs lee playlists y ubicaciones de _organizacion.csv.
  - 2026-10-10: Preparar Traktor usa los temas de la colección local (ya analizados) y deja la copia
                del paquete solo para los que faltan; el CSV suma nombre, carpeta y duración originales.
  - 2026-10-10: Creación inicial (Gabriel toca con Traktor desde el pendrive en otra computadora).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.config_loader import load_config  # noqa: E402
from src.v4.common.path_resolver import resolve_dataset_artifacts  # noqa: E402
from src.v4.common.dj_export import (  # noqa: E402
    PlaylistSpec, rekordbox_location, write_m3u8, write_rekordbox_xml, write_traktor_nml,
)

PACKAGE_NAME = "TRAKTOR ML"
MUSIC_DIR = "Musica"
M3U_DIR = "Playlists m3u8"
VOL_TOKEN, DIR_TOKEN = "@@VOL@@", "@@DIR@@"
RB_TOKEN = "@@ROOT@@"              # unidad y carpeta del paquete dentro de cada Location de rekordbox.xml
_PLACEHOLDER_ROOT = r"Z:\__PKG__"  # raíz ficticia que la plantilla reemplaza por la real
MAX_PATH = 240                     # margen bajo el límite clásico de 260 de Windows


# ---------------------------------------------------------------------------
# Nombres y ubicación de cada tema dentro del paquete
# ---------------------------------------------------------------------------

def safe_name(name: str, max_len: int = 60) -> str:
    """Nombre válido en FAT32/NTFS: sin \\ / : * ? " < > | ni controles, sin punto o espacio final.
    ' · ' y '/' (separadores de los nombres de playlist) pasan a ' - '."""
    s = name.replace(" · ", " - ").replace(" / ", " - ")
    s = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", s)
    s = re.sub(r"\s+", " ", s).strip()[:max_len]
    return s.rstrip(" .") or "_"


def safe_filename(filename: str, max_len: int = 120) -> str:
    """Nombre de archivo válido conservando la extensión."""
    dot = filename.rfind(".")
    stem, ext = (filename[:dot], filename[dot:]) if dot > 0 else (filename, "")
    return safe_name(stem, max_len=max_len - len(ext)) + ext


@dataclass
class Placement:
    track_uid: str
    folder: str      # carpeta de la organización ('' = raíz)
    playlist: str
    position: int    # 1-based dentro de la playlist
    rel_path: str    # relativo a la carpeta del paquete, con '\\'
    source_path: str


def plan_layout(specs: List[PlaylistSpec], tracks: pd.DataFrame) -> List[Placement]:
    """Ubicación de cada tema: Musica/<carpeta>/<playlist>/<archivo>. Un tema que aparece en
    varias playlists se copia una sola vez (en la primera). Nombres repetidos dentro de una misma
    carpeta reciben ' (2)', ' (3)'..."""
    placements: List[Placement] = []
    seen: Dict[str, str] = {}
    used: Dict[str, set] = {}
    for spec in specs:
        parts = [MUSIC_DIR] + ([safe_name(spec.folder)] if spec.folder else []) + [safe_name(spec.name)]
        d = "\\".join(parts)
        taken = used.setdefault(d.lower(), set())
        for i, uid in enumerate(spec.track_uids, start=1):
            src = str(tracks.loc[uid, "source_path"])
            if uid in seen:
                rel = seen[uid]
            else:
                fn = safe_filename(PureWindowsPath(src).name)
                dot = fn.rfind(".")
                stem, ext = (fn[:dot], fn[dot:]) if dot > 0 else (fn, "")
                k = 2
                while fn.lower() in taken:
                    fn = f"{stem} ({k}){ext}"
                    k += 1
                taken.add(fn.lower())
                rel = f"{d}\\{fn}"
                seen[uid] = rel
            placements.append(Placement(uid, spec.folder, spec.name, i, rel, src))
    return placements


def unique_files(placements: List[Placement]) -> List[Placement]:
    out, seen = [], set()
    for p in placements:
        if p.track_uid not in seen:
            seen.add(p.track_uid)
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# NML con raíz reemplazable
# ---------------------------------------------------------------------------

def nml_dir_prefix(package_dir: PureWindowsPath) -> Tuple[str, str]:
    """('E:', '/:carpeta/:TRAKTOR ML/:') para un paquete en E:\\carpeta\\TRAKTOR ML."""
    return package_dir.drive, "/:" + "".join(f"{d}/:" for d in package_dir.parts[1:])


def write_template_nml(out_path: Path, tracks: pd.DataFrame, specs: List[PlaylistSpec],
                       placements: List[Placement], root_label: str) -> str:
    """Plantilla NML con @@VOL@@ y @@DIR@@ en lugar de la unidad y la carpeta del paquete.
    Devuelve el texto de la plantilla."""
    ph = PureWindowsPath(_PLACEHOLDER_ROOT)
    t = tracks.copy()
    for p in unique_files(placements):
        t.loc[p.track_uid, "source_path"] = str(ph / p.rel_path)
    tmp = out_path.with_suffix(".tmp")
    write_traktor_nml(tmp, t, specs, root_label)
    text = tmp.read_text(encoding="utf-8")
    tmp.unlink()
    vol, dirp = nml_dir_prefix(ph)
    n_files = len(unique_files(placements))
    for old, new, expected in ((f'VOLUME="{vol}"', f'VOLUME="{VOL_TOKEN}"', n_files),
                               (f'DIR="{dirp}', f'DIR="{DIR_TOKEN}', n_files),
                               (f'KEY="{vol}{dirp}', f'KEY="{VOL_TOKEN}{DIR_TOKEN}', len(placements))):
        assert text.count(old) == expected, (old, text.count(old), expected)
        text = text.replace(old, new)
    assert "__PKG__" not in text
    out_path.write_text(text, encoding="utf-8", newline="\n")
    return text


def fill_template(template: str, package_dir: PureWindowsPath) -> str:
    """Lo mismo que hace _preparar_traktor.ps1: unidad y carpeta reales (con escape XML)."""
    from xml.sax.saxutils import escape
    vol, dirp = nml_dir_prefix(package_dir)
    return template.replace(VOL_TOKEN, escape(vol, {'"': "&quot;"})).replace(DIR_TOKEN, escape(dirp, {'"': "&quot;"}))


def write_rekordbox_template(out_path: Path, tracks: pd.DataFrame, specs: List[PlaylistSpec],
                             placements: List[Placement], root_label: str) -> str:
    """Plantilla rekordbox.xml con Location="file://localhost/@@ROOT@@/Musica/..." en cada tema."""
    ph = PureWindowsPath(_PLACEHOLDER_ROOT)
    t = tracks.copy()
    for p in unique_files(placements):
        t.loc[p.track_uid, "source_path"] = str(ph / p.rel_path)
    tmp = out_path.with_suffix(".tmp")
    write_rekordbox_xml(tmp, t, specs, root_label)
    text = tmp.read_text(encoding="utf-8")
    tmp.unlink()
    old = 'Location="' + rekordbox_location(str(ph)) + "/"
    assert text.count(old) == len(unique_files(placements)), (old, text.count(old))
    text = text.replace(old, f'Location="file://localhost/{RB_TOKEN}/')
    assert "__PKG__" not in text
    out_path.write_text(text, encoding="utf-8", newline="\n")
    return text


def fill_rekordbox(template: str, package_dir: PureWindowsPath) -> str:
    """Lo mismo que hace _preparar_rekordbox.ps1: 'E:/otra%20carpeta/TRAKTOR%20ML' en cada Location."""
    from urllib.parse import quote
    root = package_dir.drive + "".join("/" + quote(d, safe="") for d in package_dir.parts[1:])
    return template.replace(RB_TOKEN, root)


# ---------------------------------------------------------------------------
# Scripts para la otra computadora
# ---------------------------------------------------------------------------

PS1_PATH = Path(__file__).with_name("traktor_prepare.ps1")  # se copia como _preparar_traktor.ps1
RB_PS1_PATH = Path(__file__).with_name("rekordbox_prepare.ps1")  # se copia como _preparar_rekordbox.ps1

BAT = (
    "@echo off\r\n"
    "rem Arma traktor.nml con los temas de la coleccion de Traktor de esta computadora.\r\n"
    "powershell -NoProfile -ExecutionPolicy Bypass -File \"%~dp0_preparar_traktor.ps1\"\r\n"
    "echo.\r\n"
    "pause\r\n"
)
BAT_RB = (
    "@echo off\r\n"
    "rem Arma rekordbox.xml con la letra y la carpeta donde esta este paquete.\r\n"
    "powershell -NoProfile -ExecutionPolicy Bypass -File \"%~dp0_preparar_rekordbox.ps1\"\r\n"
    "echo.\r\n"
    "pause\r\n"
)


def leeme(root_label: str, n_tracks: int, n_playlists: int, size_gb: float) -> str:
    return f"""TRAKTOR ML - organizacion para Rekordbox y Traktor
=================================================

Contenido: {n_tracks} temas ({size_gb:.1f} GB) en {n_playlists} playlists, carpeta "{root_label}".

Musica\\                Los temas, ordenados en carpetas igual que las playlists.
Preparar Rekordbox.bat  Arma rekordbox.xml para esta computadora (ver abajo).
rekordbox.xml           Las playlists para Rekordbox (con el orden sugerido).
Preparar Traktor.bat    Arma traktor.nml para esta computadora.
traktor.nml             Las playlists para Traktor (Traktor importa solo una: ver abajo).
Playlists m3u8\\         Las mismas playlists en formato m3u8 (respaldo para otros programas).
_organizacion.csv       Lista de temas por playlist (lo usan los .bat).
_informe.txt            Lo escribe Preparar Traktor.bat: que archivo usa cada tema.

Las playlists guardan la ruta completa de cada tema, con la letra del pendrive (D:, E:...), que
cambia de una computadora a otra. Por eso, en cada computadora nueva, primero el .bat.
Recomendado: copiar esta carpeta entera a C:\\TRAKTOR ML y preparar desde ahi; asi la ruta no
cambia aunque el pendrive cambie de letra (y no hace falta tenerlo enchufado para tocar).

Como cargarlo en Rekordbox (Windows, con controladora)
------------------------------------------------------
1. Doble clic en "Preparar Rekordbox.bat". Escribe rekordbox.xml con la carpeta donde esta este
   paquete y muestra cuantos temas encontro ("Archivos que faltan: 0" = todo bien).
2. Rekordbox: Preferencias > Avanzado > Base de datos > rekordbox xml: elegir el rekordbox.xml
   de esta carpeta. Preferencias > Vista > Diseno: activar "rekordbox xml".
3. En el arbol de la izquierda: rekordbox xml > Playlists > clic derecho sobre la carpeta
   "{root_label}" > Importar playlist. Rekordbox analiza los temas al importarlos.
   Ordenar por la columna "#" para ver el orden sugerido.

Como cargarlo en Traktor (Windows)
----------------------------------
Ojo: "Import Playlist" de Traktor 4 toma UNA sola playlist de traktor.nml, y los temas que entran
asi no muestran la caratula. Lo que funciona:
1. Abrir Traktor y arrastrar la carpeta Musica de este paquete a "Track Collection" (Traktor lee
   titulos y caratulas). Cerrar Traktor con la X y esperar a que se cierre del todo.
2. Las {n_playlists} playlists, en una PC con el repo TRAKTOR ML y Traktor cerrado:
   python src/v4/pipeline/traktor_collection.py playlists --package "<esta carpeta>"
   Aparece la carpeta "{root_label}" debajo de Playlists, con el orden sugerido.
   Sin el repo: en Traktor, Explorer > esta carpeta > Musica (cada subcarpeta es una playlist).
"Preparar Traktor.bat" arma traktor.nml para esta PC (usa los temas que Traktor ya tiene
analizados); sirve para importar una playlist suelta.
"""


# ---------------------------------------------------------------------------
# Copia y verificación
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def copy_files(files: List[Placement], package_dir: Path) -> Tuple[int, int]:
    """Copia lo que falta (o tiene otro tamaño). Escribe a .part y renombra. -> (copiados, ya estaban)."""
    copied = skipped = 0
    done_bytes, t0 = 0, time.time()
    for i, p in enumerate(files, start=1):
        src, dst = Path(p.source_path), package_dir / p.rel_path
        size = src.stat().st_size
        if dst.exists() and dst.stat().st_size == size:
            skipped += 1
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            part = dst.with_name(dst.name + ".part")
            shutil.copyfile(src, part)
            os.replace(part, dst)
            copied += 1
            done_bytes += size
        if i % 50 == 0 or i == len(files):
            el = time.time() - t0
            rate = done_bytes / el / 1e6 if el > 0 else 0.0
            print(f"[COPY] {i}/{len(files)}  copiados={copied} ya estaban={skipped}  {rate:.0f} MB/s", flush=True)
    return copied, skipped


def verify_files(files: List[Placement], package_dir: Path, mode: str) -> List[str]:
    """Problemas encontrados ([] = todo bien). mode: 'size' o 'hash'."""
    problems = []
    for i, p in enumerate(files, start=1):
        src, dst = Path(p.source_path), package_dir / p.rel_path
        if not dst.exists():
            problems.append(f"falta: {p.rel_path}")
        elif dst.stat().st_size != src.stat().st_size:
            problems.append(f"tamaño distinto: {p.rel_path}")
        elif mode == "hash" and _sha256(dst) != _sha256(src):
            problems.append(f"contenido distinto: {p.rel_path}")
        if mode == "hash" and (i % 100 == 0 or i == len(files)):
            print(f"[VERIFY] {i}/{len(files)}  problemas={len(problems)}", flush=True)
    return problems


def stale_files(files: List[Placement], package_dir: Path) -> List[Path]:
    """Archivos dentro de Musica/ que ya no pertenecen a la organización (de un paquete anterior)."""
    keep = {(package_dir / p.rel_path).resolve() for p in files}
    music = package_dir / MUSIC_DIR
    if not music.exists():
        return []
    return [f for f in music.rglob("*") if f.is_file() and f.resolve() not in keep]


# ---------------------------------------------------------------------------
# Paquete
# ---------------------------------------------------------------------------

def _text(value) -> str:
    return str(value) if value is not None and pd.notna(value) else ""


def write_package_files(package_dir: Path, tracks: pd.DataFrame, specs: List[PlaylistSpec],
                        placements: List[Placement], root_label: str) -> None:
    """Todo lo que no es audio: NML, plantilla, scripts, m3u8, CSV y LEEME."""
    files = unique_files(placements)
    template = write_template_nml(package_dir / "_plantilla.nml", tracks, specs, placements, root_label)
    real = PureWindowsPath(str(package_dir.resolve()))
    (package_dir / "traktor.nml").write_text(fill_template(template, real), encoding="utf-8", newline="\n")
    (package_dir / "_preparar_traktor.ps1").write_text(PS1_PATH.read_text(encoding="utf-8"),
                                                       encoding="utf-8-sig", newline="\r\n")
    (package_dir / "Preparar Traktor.bat").write_bytes(BAT.encode("ascii"))
    rb_template = write_rekordbox_template(package_dir / "_plantilla_rekordbox.xml", tracks, specs, placements,
                                           root_label)
    (package_dir / "rekordbox.xml").write_text(fill_rekordbox(rb_template, real), encoding="utf-8", newline="\n")
    (package_dir / "_preparar_rekordbox.ps1").write_text(RB_PS1_PATH.read_text(encoding="utf-8"),
                                                         encoding="utf-8-sig", newline="\r\n")
    (package_dir / "Preparar Rekordbox.bat").write_bytes(BAT_RB.encode("ascii"))

    # m3u8 con rutas relativas a la carpeta de cada playlist
    m3u_root = package_dir / M3U_DIR
    if m3u_root.exists():
        shutil.rmtree(m3u_root)
    rel_by_uid = {p.track_uid: p.rel_path for p in files}
    for spec in specs:
        sub = [safe_name(spec.folder)] if spec.folder else []
        out_dir = m3u_root.joinpath(*sub)
        out_dir.mkdir(parents=True, exist_ok=True)
        up = "..\\" * (1 + len(sub))
        t = tracks.loc[spec.track_uids].copy()
        t["source_path"] = [up + rel_by_uid[u] for u in spec.track_uids]
        t = t[~t.index.duplicated()]
        write_m3u8(out_dir / f"{safe_name(spec.name)}.m3u8", t, spec.track_uids)

    with open(package_dir / "_organizacion.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["carpeta", "playlist", "posicion", "track_uid", "archivo", "artista", "titulo", "origen",
                    "nombre_original", "carpeta_original", "duracion_s"])
        for p in placements:
            row = tracks.loc[p.track_uid]
            src = PureWindowsPath(p.source_path)
            rel = row.get("rel_path")
            orig_dir = str(PureWindowsPath(str(rel)).parent) if isinstance(rel, str) and rel else ""
            dur = row.get("duration_s")
            w.writerow([p.folder, p.playlist, p.position, p.track_uid, p.rel_path,
                        _text(row.get("artist")), _text(row.get("title")), p.source_path,
                        src.name, "" if orig_dir == "." else orig_dir,
                        f"{float(dur):.2f}" if pd.notna(dur) else ""])

    size_gb = sum((package_dir / p.rel_path).stat().st_size for p in files) / 1e9
    (package_dir / "LEEME.txt").write_text(
        leeme(root_label, len(files), len(specs), size_gb), encoding="utf-8-sig", newline="\r\n")


def _root_label(package_dir: Path) -> str:
    """Nombre de la carpeta raíz de las playlists, leído de la plantilla NML del paquete."""
    import xml.etree.ElementTree as ET
    text = (package_dir / "_plantilla.nml").read_text(encoding="utf-8")
    node = ET.fromstring(text[text.index("<NML"):]).find("PLAYLISTS/NODE/SUBNODES/NODE")
    if node is None or not node.get("NAME"):
        raise ValueError("no encontré la carpeta raíz de las playlists en _plantilla.nml")
    return node.get("NAME")


def read_package_specs(package_dir: Path) -> Tuple[pd.DataFrame, List[PlaylistSpec], List[Placement]]:
    """Playlists (en orden) y ubicación de cada tema, leídas de _organizacion.csv del paquete."""
    org = pd.read_csv(package_dir / "_organizacion.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    specs: List[PlaylistSpec] = []
    by_key: Dict[Tuple[str, str], PlaylistSpec] = {}
    for r in org.itertuples():
        if (r.carpeta, r.playlist) not in by_key:
            by_key[(r.carpeta, r.playlist)] = PlaylistSpec(r.carpeta, r.playlist, [])
            specs.append(by_key[(r.carpeta, r.playlist)])
    for r in sorted(org.itertuples(), key=lambda r: int(r.posicion)):
        by_key[(r.carpeta, r.playlist)].track_uids.append(r.track_uid)
    placements = [Placement(r.track_uid, r.carpeta, r.playlist, int(r.posicion), r.archivo, r.origen)
                  for r in org.itertuples()]
    return org, specs, placements


def refresh_package(package_dir: Path) -> int:
    """Rehace NML, rekordbox.xml, plantillas, scripts, m3u8, CSV y LEEME de un paquete existente
    a partir de su _organizacion.csv y de los tags actuales de sus archivos (sin copiar audio)."""
    from src.v4.common.tags import read_release_tags, read_tags
    org, specs, placements = read_package_specs(package_dir)
    files = unique_files(placements)
    missing = [p.rel_path for p in files if not (package_dir / p.rel_path).exists()]
    if missing:
        print(f"[ERROR] {len(missing)} archivos del paquete no están, p. ej. {missing[:3]}")
        return 1
    first = {r.track_uid: r for r in org.drop_duplicates("track_uid").itertuples()}
    rows = []
    for p in files:
        f, o = package_dir / p.rel_path, first[p.track_uid]
        t, rel = read_tags(f), read_release_tags(f)
        rows.append({"track_uid": p.track_uid, "source_path": str(f),
                     "artist": _text(t["tag_artist"]) or o.artista, "title": _text(t["tag_title"]) or o.titulo,
                     "tag_album": t["tag_album"], "tag_genre": t["tag_genre"], "tag_label": t["tag_label"],
                     "tag_comment": t["tag_comment"], "tag_remixer": rel["tag_remixer"], "tag_date": rel["tag_date"],
                     "duration_s": float(o.duracion_s) if o.duracion_s else float("nan"),
                     "filesize_bytes": f.stat().st_size,
                     "rel_path": f"{o.carpeta_original}\\{o.nombre_original}" if o.carpeta_original else o.nombre_original})
    tracks = pd.DataFrame(rows).set_index("track_uid")
    write_package_files(package_dir, tracks, specs, placements, _root_label(package_dir))
    print(f"[INFO] Paquete rehecho: {len(files)} temas, {len(specs)} playlists -> {package_dir}")
    return 0


def refresh_from_org(package_dir: Path, dataset_name: str, org_name: str, config: Optional[dict] = None) -> int:
    """Las playlists del paquete desde la versión actual de la organización, sin mover ni copiar audio: la
    biblioteca es el paquete (su Musica/, package_library.py) y catalog.source_path dice dónde está cada tema."""
    from src.v4.common.tags import read_release_tags, read_tags
    from src.v4.pipeline.phase5_export import run_export
    config = config or load_config(None)
    with tempfile.TemporaryDirectory() as tmp:
        _, specs, _, _ = run_export(dataset_name, config, formats=("traktor",), out_root=Path(tmp),
                                    org_name=org_name, return_specs=True)
    artifacts = resolve_dataset_artifacts(dataset_name, config)
    cat = pd.read_parquet(artifacts / "catalog.parquet").drop_duplicates("track_uid").set_index("track_uid")
    music = (Path(package_dir) / "Musica").resolve()
    # El origen de cada tema (columnas origen y nombre_original del CSV) se conserva: Preparar Traktor
    # (traktor_prepare.ps1) reconoce por nombre y carpeta originales los temas que Traktor ya analizó.
    index = Path(package_dir) / "_organizacion.csv"
    prev, prev_dur = {}, {}
    if index.exists():
        old = pd.read_csv(index, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        if {"track_uid", "origen"} <= set(old.columns):
            prev = dict(zip(old["track_uid"], old["origen"]))
        if {"track_uid", "duracion_s"} <= set(old.columns):  # la misma duración que ya tenía (sin redondeos nuevos)
            prev_dur = {u: float(d) for u, d in zip(old["track_uid"], old["duracion_s"]) if d}
    placements: List[Placement] = []
    outside = []
    for spec in specs:
        for k, uid in enumerate(spec.track_uids, 1):
            src = Path(str(cat.at[uid, "source_path"])).resolve()
            if music not in src.parents:
                outside.append(str(src))
                continue
            rel = "Musica\\" + str(PureWindowsPath(src.relative_to(music)))
            placements.append(Placement(uid, spec.folder, spec.name, k, rel, prev.get(uid) or str(src)))
    files = unique_files(placements)
    missing = [p.rel_path for p in files if not (Path(package_dir) / p.rel_path).exists()]
    if outside or missing:
        print(f"[ERROR] {len(outside)} temas fuera de {music} y {len(missing)} archivos que no están, "
              f"p. ej. {(outside + missing)[:3]}")
        return 1
    rows = []
    for p in files:
        f, c = Path(package_dir) / p.rel_path, cat.loc[p.track_uid]
        t, rel = read_tags(f), read_release_tags(f)
        orig = c.get("orig_rel_path") if isinstance(c.get("orig_rel_path"), str) else c.get("rel_path")
        rows.append({"track_uid": p.track_uid, "source_path": str(f),
                     "artist": _text(t["tag_artist"]) or _text(c.get("artist")),
                     "title": _text(t["tag_title"]) or _text(c.get("title")),
                     "tag_album": t["tag_album"], "tag_genre": t["tag_genre"], "tag_label": t["tag_label"],
                     "tag_comment": t["tag_comment"], "tag_remixer": rel["tag_remixer"], "tag_date": rel["tag_date"],
                     "duration_s": prev_dur.get(p.track_uid, float(c.get("duration_s")) if pd.notna(c.get("duration_s"))
                                               else float("nan")),
                     "filesize_bytes": f.stat().st_size,
                     "rel_path": str(PureWindowsPath(str(orig))) if isinstance(orig, str) else ""})
    tracks = pd.DataFrame(rows).set_index("track_uid")
    write_package_files(Path(package_dir), tracks, specs, placements, _root_label(Path(package_dir)))
    print(f"[INFO] Paquete rehecho desde '{org_name}': {len(files)} temas, {len(specs)} playlists -> {package_dir}")
    return 0


def build_package(dataset_name: str, org_name: str, dest: Path, org_version: Optional[int] = None,
                  package_name: str = PACKAGE_NAME, verify: str = "size", prune: bool = False,
                  dry_run: bool = False, config: Optional[dict] = None) -> int:
    from src.v4.pipeline.phase5_export import run_export
    config = config or load_config(None)
    with tempfile.TemporaryDirectory() as tmp:
        _, specs, tracks, root_label = run_export(dataset_name, config, formats=("traktor",),
                                                  out_root=Path(tmp), org_name=org_name,
                                                  org_version=org_version, return_specs=True)
    package_dir = Path(dest) / package_name
    placements = plan_layout(specs, tracks)
    files = unique_files(placements)
    missing_src = [p.source_path for p in files if not Path(p.source_path).exists()]
    longest = max(len(str(package_dir / p.rel_path)) for p in files)
    need = sum(Path(p.source_path).stat().st_size for p in files if Path(p.source_path).exists())
    print(f"[INFO] {len(specs)} playlists, {len(files)} temas, {need / 1e9:.2f} GB -> {package_dir}")
    print(f"[INFO] Ruta más larga en el destino: {longest} caracteres")
    if missing_src:
        print(f"[ERROR] {len(missing_src)} temas no existen en el origen, p. ej. {missing_src[:3]}")
        return 1
    if longest > MAX_PATH:
        print(f"[ERROR] Rutas de más de {MAX_PATH} caracteres; usar un destino más corto")
        return 1
    if dry_run:
        return 0

    package_dir.mkdir(parents=True, exist_ok=True)
    present = sum((package_dir / p.rel_path).stat().st_size for p in files if (package_dir / p.rel_path).exists())
    free = shutil.disk_usage(package_dir).free
    if need - present > free:
        print(f"[ERROR] Falta espacio: hacen falta {(need - present) / 1e9:.1f} GB y hay {free / 1e9:.1f} GB")
        return 1

    copied, skipped = copy_files(files, package_dir)
    write_package_files(package_dir, tracks, specs, placements, root_label)
    stale = stale_files(files, package_dir)
    if stale:
        if prune:
            for f in stale:
                f.unlink()
            print(f"[INFO] Borrados {len(stale)} archivos de un paquete anterior")
        else:
            print(f"[WARN] {len(stale)} archivos en {MUSIC_DIR}/ ya no están en la organización (--prune los borra)")

    problems = verify_files(files, package_dir, verify)
    print(f"[INFO] Copiados {copied}, ya estaban {skipped}. Verificación ({verify}): "
          f"{'OK' if not problems else f'{len(problems)} problemas'}")
    for msg in problems[:20]:
        print("  ", msg)
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Paquete portátil (música + traktor.nml) de una organización")
    ap.add_argument("--dataset-name", default="musica")
    ap.add_argument("--org-name", default=None)
    ap.add_argument("--org-version", type=int, default=None)
    ap.add_argument("--dest", default=None, help="Unidad o carpeta destino, p. ej. D:\\")
    ap.add_argument("--refresh", default=None, metavar="PAQUETE",
                    help="Rehace los archivos que no son audio de un paquete existente desde sus tags actuales")
    ap.add_argument("--from-org", action="store_true",
                    help="Con --refresh: playlists desde la versión actual de --org-name (la biblioteca es el paquete)")
    ap.add_argument("--package-name", default=PACKAGE_NAME)
    ap.add_argument("--verify", choices=("size", "hash"), default="size")
    ap.add_argument("--prune", action="store_true", help="Borra de Musica/ lo que ya no está en la organización")
    ap.add_argument("--dry-run", action="store_true", help="Solo planifica y revisa rutas y espacio")
    ap.add_argument("--low-priority", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    if args.refresh and args.from_org:
        if not args.org_name:
            ap.error("--from-org necesita --org-name")
        return refresh_from_org(Path(args.refresh), args.dataset_name, args.org_name,
                                load_config(Path(args.config) if args.config else None))
    if args.refresh:
        return refresh_package(Path(args.refresh))
    if not args.org_name or not args.dest:
        ap.error("--org-name y --dest son obligatorios (salvo con --refresh)")
    if args.low_priority:
        from src.v4.pipeline.extract_representations import lower_priority
        lower_priority()
    config = load_config(Path(args.config) if args.config else None)
    return build_package(args.dataset_name, args.org_name, Path(args.dest), args.org_version,
                         args.package_name, args.verify, args.prune, args.dry_run, config)


if __name__ == "__main__":
    sys.exit(main())
