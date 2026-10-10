"""
PURPOSE: Paquete portátil de una organización para tocar con Traktor desde un pendrive o disco
         externo en cualquier computadora Windows. Copia cada tema a
         <destino>/<paquete>/Musica/<carpeta>/<playlist>/<archivo> (la música queda ordenada como
         las playlists) y escribe:
           - traktor.nml con las rutas del destino actual (Import Playlist en Traktor),
           - _plantilla.nml + _preparar_traktor.ps1 (traktor_prepare.ps1) + "Preparar Traktor.bat":
             en otra computadora arman traktor.nml con los temas que ya están en su colección de
             Traktor (ya analizados; ficha copiada tal cual) y, para los que no, la copia del
             paquete en la letra y carpeta donde esté,
           - Playlists m3u8/ con rutas relativas (respaldo para otros programas),
           - _organizacion.csv (playlist, posición, track_uid, archivo) y LEEME.txt.
         Reanudable: no vuelve a copiar archivos que ya están con el mismo tamaño. Verifica tamaños
         siempre y, con --verify hash, el SHA256 de cada copia contra el original.
CHANGELOG:
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
from src.v4.common.dj_export import PlaylistSpec, write_m3u8, write_traktor_nml  # noqa: E402

PACKAGE_NAME = "TRAKTOR ML"
MUSIC_DIR = "Musica"
M3U_DIR = "Playlists m3u8"
VOL_TOKEN, DIR_TOKEN = "@@VOL@@", "@@DIR@@"
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


# ---------------------------------------------------------------------------
# Scripts para la otra computadora
# ---------------------------------------------------------------------------

PS1_PATH = Path(__file__).with_name("traktor_prepare.ps1")  # se copia como _preparar_traktor.ps1

BAT = (
    "@echo off\r\n"
    "rem Arma traktor.nml con los temas de la coleccion de Traktor de esta computadora.\r\n"
    "powershell -NoProfile -ExecutionPolicy Bypass -File \"%~dp0_preparar_traktor.ps1\"\r\n"
    "echo.\r\n"
    "pause\r\n"
)


def leeme(root_label: str, n_tracks: int, n_playlists: int, size_gb: float) -> str:
    return f"""TRAKTOR ML - organizacion para Traktor
=====================================

Contenido: {n_tracks} temas ({size_gb:.1f} GB) en {n_playlists} playlists, carpeta "{root_label}".

Musica\\             Los temas, ordenados en carpetas igual que las playlists.
traktor.nml          Las playlists para Traktor (con el orden sugerido).
Preparar Traktor.bat Arma traktor.nml para esta computadora (ver abajo).
Playlists m3u8\\      Las mismas playlists en formato m3u8 (respaldo para otros programas).
_organizacion.csv    Lista de temas por playlist (lo usa Preparar Traktor.bat).
_informe.txt         Lo escribe Preparar Traktor.bat: que archivo usa cada tema.

Como cargarlo en Traktor (Windows)
----------------------------------
1. Cerrar Traktor si esta abierto (asi su coleccion en disco esta al dia).
2. Doble clic en "Preparar Traktor.bat". Busca cada tema en la coleccion de Traktor de esta
   computadora: los que ya estan se usan tal cual (con su analisis y sus cue points); los que no,
   salen de la copia de este paquete. Muestra cuantos hay de cada tipo.
   - "Copia del paquete: 0": todo es local; el pendrive no hace falta para tocar.
   - Si hay temas de la copia del paquete y no queres tocar con el pendrive enchufado: copiar esta
     carpeta entera a la raiz de un disco (C:\\TRAKTOR ML o D:\\TRAKTOR ML; no dentro de Documentos:
     las rutas quedarian largas) y volver a hacer doble clic en Preparar Traktor.bat desde ahi.
3. Abrir Traktor. En el panel izquierdo: clic derecho sobre "Playlists" > "Import Playlist" >
   elegir traktor.nml de la carpeta donde corriste Preparar Traktor.bat.
4. Aparece la carpeta "{root_label}" con las playlists. Ordenar por la columna "#" para ver
   el orden sugerido. Solo los temas de la copia del paquete necesitan analisis.
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

    size_gb = sum(Path(p.source_path).stat().st_size for p in files) / 1e9
    (package_dir / "LEEME.txt").write_text(
        leeme(root_label, len(files), len(specs), size_gb), encoding="utf-8-sig", newline="\r\n")


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
    ap.add_argument("--org-name", required=True)
    ap.add_argument("--org-version", type=int, default=None)
    ap.add_argument("--dest", required=True, help="Unidad o carpeta destino, p. ej. D:\\")
    ap.add_argument("--package-name", default=PACKAGE_NAME)
    ap.add_argument("--verify", choices=("size", "hash"), default="size")
    ap.add_argument("--prune", action="store_true", help="Borra de Musica/ lo que ya no está en la organización")
    ap.add_argument("--dry-run", action="store_true", help="Solo planifica y revisa rutas y espacio")
    ap.add_argument("--low-priority", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    if args.low_priority:
        from src.v4.pipeline.extract_representations import lower_priority
        lower_priority()
    config = load_config(Path(args.config) if args.config else None)
    return build_package(args.dataset_name, args.org_name, Path(args.dest), args.org_version,
                         args.package_name, args.verify, args.prune, args.dry_run, config)


if __name__ == "__main__":
    sys.exit(main())
