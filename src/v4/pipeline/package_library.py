"""
PURPOSE: La biblioteca pasa a ser el paquete de usb_package.py (DECISIONS 2026-11-04: la laptop se retira y el
         pendrive manda). `adopt` hace que el pipeline trabaje sobre los archivos del paquete:
           - el catálogo apunta a cada archivo del paquete por track_uid (_organizacion.csv): rel_path relativo
             a <paquete>/Musica y source_path con su ruta real; la ruta original queda en orig_rel_path y la
             carpeta original («tu carpeta» en la página) en folder;
           - artista, título y los tags que se muestran o exportan se releen de los archivos del paquete (los
             corrigió package_tags.py); BPM, tonalidad y energía no se tocan;
           - las filas sin archivo en el paquete (copias descartadas, temas sin embedding) salen del catálogo;
           - duplicates.csv y duplicate_decisions.json siguen las rutas nuevas de lo que se conservó;
           - Música/ en la raíz del repo pasa a ser un junction a <paquete>/Musica, así que la app deja el modo
             de solo lectura y «Agregar música nueva» copia ahí.
         Antes de escribir, copia catalog.parquet, duplicates.csv, duplicate_decisions.json y moved_copies.csv a
         artifacts/<dataset>/respaldo_package_library_<fecha>/. Nunca mueve, copia ni borra audio.
         Uso: package_library.py adopt --package "D:\\TRAKTOR ML" [--dry-run]
CHANGELOG:
  - 2026-11-04: Creación.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path, PureWindowsPath
from typing import Dict, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.config_loader import load_config  # noqa: E402
from src.v4.common.duplicates import DECISIONS_FILE  # noqa: E402
from src.v4.common.path_resolver import resolve_dataset_artifacts  # noqa: E402
from src.v4.common.tags import read_tags  # noqa: E402

PACKAGE_INDEX = "_organizacion.csv"
LIBRARY_LINK = REPO_ROOT / "Música"  # datasets.musica.audio_root en config/v4.yaml
SHOWN_TAGS = ("tag_artist", "tag_title", "tag_album", "tag_genre", "tag_label", "tag_comment")
BACKED_UP = ("catalog.parquet", "duplicates.csv", DECISIONS_FILE, "moved_copies.csv")


def is_package_library(library: Path) -> bool:
    """¿La biblioteca es la carpeta Musica de un paquete (con su _organizacion.csv al lado)?"""
    library = Path(library).resolve()
    return library.name == "Musica" and (library.parent / PACKAGE_INDEX).is_file()


def new_paths(package: Path, catalog: pd.DataFrame) -> Dict[str, Path]:
    """track_uid → archivo del paquete, para los temas del catálogo que están en el paquete."""
    music = (Path(package) / "Musica").resolve()
    index = pd.read_csv(Path(package) / PACKAGE_INDEX, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    known = set(catalog["track_uid"])
    out = {}
    for uid, rel in zip(index["track_uid"], index["archivo"]):
        f = (Path(package) / PureWindowsPath(rel).as_posix()).resolve()
        if uid in known and music in f.parents and f.is_file():
            out[uid] = f
    return out


def rebase(catalog: pd.DataFrame, files: Dict[str, Path], music: Path, read_tags_fn=read_tags) -> pd.DataFrame:
    """El catálogo con cada tema en su archivo del paquete; sin las filas que no tienen archivo."""
    cat = catalog[catalog["track_uid"].isin(files)].copy().reset_index(drop=True)
    if "orig_rel_path" not in cat.columns:
        cat["orig_rel_path"] = cat["rel_path"]
    for col in SHOWN_TAGS:
        if col not in cat.columns:
            cat[col] = None
        cat[col] = cat[col].astype(object)
    for i, uid in enumerate(cat["track_uid"]):
        f = files[uid]
        cat.at[i, "rel_path"] = f.relative_to(music).as_posix()
        cat.at[i, "source_path"] = str(f)
        cat.at[i, "filename"] = f.name
        cat.at[i, "filesize_bytes"] = f.stat().st_size
        tags = read_tags_fn(f)
        for col in SHOWN_TAGS:
            cat.at[i, col] = tags.get(col)
        if tags.get("tag_artist"):
            cat.at[i, "artist"] = tags["tag_artist"]
        if tags.get("tag_title"):
            cat.at[i, "title"] = tags["tag_title"]
    return cat


def link_state(music: Path, link: Path = LIBRARY_LINK) -> str:
    """'falta', 'ya estaba' o error si en su lugar hay otra cosa (una carpeta de verdad no se toca)."""
    if not os.path.lexists(link):
        return "falta"
    if Path(link).resolve() == Path(music).resolve():
        return "ya estaba"
    raise RuntimeError(f"{link} ya existe y no es un enlace a {music}: no se toca")


def link_library(music: Path, link: Path = LIBRARY_LINK) -> str:
    """Música/ del repo → <paquete>/Musica (junction)."""
    music = Path(music).resolve()
    if link_state(music, link) == "ya estaba":
        return "ya estaba"
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(music)], check=True, capture_output=True)
    else:
        os.symlink(music, link, target_is_directory=True)
    return "creado"


def adopt(package: Path, dataset_name: str = "musica", config: Optional[dict] = None, dry_run: bool = False,
          link: Path = LIBRARY_LINK) -> Dict:
    config = config or load_config(None)
    artifacts = resolve_dataset_artifacts(dataset_name, config)
    music = (Path(package) / "Musica").resolve()
    link_state(music, link)  # antes de escribir nada: si en su lugar hay otra cosa, no se hace nada
    catalog = pd.read_parquet(artifacts / "catalog.parquet")
    files = new_paths(package, catalog)
    old_rel = dict(zip(catalog["track_uid"], catalog["rel_path"]))
    summary = {"catalogo": len(catalog), "en_el_paquete": len(files), "sin_archivo": int((~catalog["track_uid"].isin(files)).sum())}
    if dry_run:
        return summary
    backup = artifacts / f"respaldo_package_library_{dt.datetime.now():%Y%m%d_%H%M%S_%f}"
    backup.mkdir()
    for name in BACKED_UP:
        if (artifacts / name).exists():
            shutil.copy2(artifacts / name, backup / name)
    catalog[~catalog["track_uid"].isin(files)].to_parquet(backup / "catalogo_sin_archivo.parquet", index=False)

    new = rebase(catalog, files, music)
    new_rel = dict(zip(new["track_uid"], new["rel_path"]))
    moved = {old_rel[u]: new_rel[u] for u in files}
    new.to_parquet(artifacts / "catalog.parquet", index=False)
    dup_path = artifacts / "duplicates.csv"
    if dup_path.exists():
        dups = pd.read_csv(dup_path)
        dups["kept_rel_path"] = dups["kept_rel_path"].map(lambda r: moved.get(r, r))
        dups.to_csv(dup_path, index=False)
    dec_path = artifacts / DECISIONS_FILE
    if dec_path.exists():
        data = json.loads(dec_path.read_text(encoding="utf-8"))
        for d in data.get("decisions", []):
            for t in d.get("tracks", []):
                t["rel_path"] = moved.get(t.get("rel_path"), t.get("rel_path"))
        dec_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    summary.update({"respaldo": str(backup), "enlace": link_library(music, link)})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="La biblioteca pasa a ser el paquete del pendrive")
    parser.add_argument("mode", choices=("adopt",))
    parser.add_argument("--package", required=True)
    parser.add_argument("--dataset-name", default="musica")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    out = adopt(Path(args.package), args.dataset_name, load_config(Path(args.config) if args.config else None),
                args.dry_run)
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
