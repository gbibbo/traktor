"""
PURPOSE: Catálogo central de dataset. Single source of truth para metadata de tracks.
         Soporta dos modos de hashing: full (SHA256 archivo completo, default) y
         fast (SHA256 de primeros N bytes + filesize).
CHANGELOG:
  - 2026-09-27: Biblioteca con subcarpetas: recursive, rel_path/folder, lectura de tags
                (read_tags), hash "audio" estable ante ediciones de tags, y deduplicado por
                track_uid prefiriendo la copia organizada sobre carpetas "copia"/"old N".
  - 2026-02-28: Creación inicial V4. Hash modes: full (default) y fast.
"""
import hashlib
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import soundfile as sf

from src.v4.config import TRACK_UID_BYTES_TO_READ
from src.v4.common.audio_utils import get_audio_files
from src.v4.common.path_resolver import resolve_artifacts_root, resolve_dataset_artifacts
from src.v4.common.tags import audio_payload_hash, read_tags

# Carpetas que Gabriel usa como copia de respaldo desordenada ("... - copia", "old 4")
_COPY_LIKE = re.compile(r"(^|\W)(copia|copy)(\W|$)|^old(\s*\d+)?$", re.IGNORECASE)


def compute_track_uid(filepath: Path, mode: str = "full", bytes_to_read: int = TRACK_UID_BYTES_TO_READ) -> str:
    """
    Calcular hash estable de contenido.

    mode="full" (default robusto): SHA256 streaming de TODO el archivo.
      64 chars hex. Sin colisiones prácticas. Más lento pero canónico.
    mode="fast": SHA256(primeros bytes_to_read bytes + filesize_bytes como string).
      Para datasets masivos donde performance importa.
    mode="audio": SHA256 del audio sin bloques de tags (tags.audio_payload_hash): editar
      artista, comentario, etc. no cambia el uid.

    Returns: hex string de 64 chars (SHA256 completo, nunca truncado).
    """
    filepath = Path(filepath)
    if mode == "audio":
        return audio_payload_hash(filepath)
    filesize = filepath.stat().st_size

    h = hashlib.sha256()
    if mode == "fast":
        with open(filepath, "rb") as f:
            chunk = f.read(bytes_to_read)
        h.update(chunk)
        h.update(str(filesize).encode())
    else:
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                h.update(chunk)

    return h.hexdigest()  # 64 chars, nunca truncar


def _parse_artist_title(filename: str) -> tuple[str, str]:
    """
    Extraer artist y title de nombre de archivo tipo "Artist - Title.mp3".
    Retorna ("", "") si el formato no coincide.
    """
    stem = Path(filename).stem
    parts = stem.split(" - ", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return "", stem.strip()


def _normalize_filename(filename: str) -> str:
    """Normalizar filename para merge: lowercase, sin espacios extra, sin extensión."""
    stem = Path(filename).stem
    return re.sub(r"\s+", " ", stem.strip().lower())


def _get_duration(filepath: Path) -> Optional[float]:
    """Duración en segundos con soundfile; si libsndfile no abre el archivo (p. ej. MP3 con
    "bad map offset"), con mutagen. Retorna None si ambos fallan."""
    try:
        info = sf.info(str(filepath))
        return info.duration
    except Exception:
        pass
    try:
        import mutagen
        audio = mutagen.File(str(filepath))
        length = float(audio.info.length) if audio is not None else 0.0
        return length if length > 0 else None
    except Exception:
        return None


def _copy_penalty(rel_path: str) -> int:
    """1 si alguna carpeta de la ruta parece una copia de respaldo; 0 si no."""
    parts = Path(rel_path).parts[:-1]
    return int(any(_COPY_LIKE.search(p.strip()) for p in parts))


def dedupe_by_uid(catalog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Una fila por track_uid. Gana la ruta fuera de carpetas tipo copia; luego la más corta.
    Devuelve (catálogo, duplicados descartados con la ruta que se conservó)."""
    if catalog.empty or not catalog["track_uid"].duplicated().any():
        return catalog, pd.DataFrame(columns=["track_uid", "rel_path", "kept_rel_path"])
    key = catalog["rel_path"] if "rel_path" in catalog.columns else catalog["filename"]
    ranked = catalog.assign(_pen=key.map(_copy_penalty), _len=key.str.len(), _key=key)
    ranked = ranked.sort_values(["track_uid", "_pen", "_len", "_key"])
    keep_mask = ~ranked["track_uid"].duplicated(keep="first")
    kept = ranked[keep_mask]
    dropped = ranked[~keep_mask][["track_uid", "_key"]].rename(columns={"_key": "rel_path"})
    dropped = dropped.merge(kept[["track_uid", "_key"]].rename(columns={"_key": "kept_rel_path"}), on="track_uid")
    kept = kept.drop(columns=["_pen", "_len", "_key"]).sort_values(key.name).reset_index(drop=True)
    return kept, dropped.reset_index(drop=True)


def build_catalog(
    audio_dir: Path,
    dataset_name: str,
    config: dict,
    metadata_df: Optional[pd.DataFrame] = None,
    recursive: bool = False,
    with_tags: bool = False,
    hash_mode: Optional[str] = None,
) -> pd.DataFrame:
    """
    Escanear directorio y construir catálogo.

    Columnas mínimas: track_uid, filename, source_path, duration_s, filesize_bytes,
                      artist, title, rel_path, folder.
    with_tags=True agrega columnas tag_* (tags.read_tags), usa artista/título de los tags
    cuando existen y llena beatport_genre_norm con el género del tag (naming de phase3).
    Si metadata_df está disponible, merge por filename normalizado.
    Duplicados exactos (mismo track_uid) se reducen a una fila; los descartados van a
    duplicates.csv junto al catálogo.
    Guarda en artifacts/v4/datasets/<dataset_name>/catalog.parquet.

    Returns: DataFrame del catálogo (solo tracks válidos con duration_s no None).
    """
    audio_dir = Path(audio_dir)
    hashing_cfg = config.get("hashing", {})
    hash_mode = hash_mode or hashing_cfg.get("mode", "full")
    hash_bytes = hashing_cfg.get("fast_bytes_to_read", TRACK_UID_BYTES_TO_READ)

    audio_files = get_audio_files(audio_dir, recursive=recursive)

    rows = []
    n_failed = 0
    for filepath in audio_files:
        duration = _get_duration(filepath)
        if duration is None:
            n_failed += 1
            continue
        artist, title = _parse_artist_title(filepath.name)
        uid = compute_track_uid(filepath, mode=hash_mode, bytes_to_read=hash_bytes)
        rel = filepath.relative_to(audio_dir)
        row = {
            "track_uid": uid,
            "filename": filepath.name,
            "source_path": str(filepath),
            "rel_path": rel.as_posix(),
            "folder": rel.parent.as_posix() if rel.parent != Path(".") else "",
            "duration_s": duration,
            "filesize_bytes": filepath.stat().st_size,
            "artist": artist,
            "title": title,
        }
        if with_tags:
            tags = read_tags(filepath)
            row.update(tags)
            row["artist"] = tags["tag_artist"] or artist
            row["title"] = tags["tag_title"] or title
            row["beatport_genre_norm"] = tags["tag_genre"]
        rows.append(row)

    catalog = pd.DataFrame(rows)
    catalog, dropped = dedupe_by_uid(catalog)

    # Merge con metadata externa si disponible
    if metadata_df is not None and len(catalog) > 0:
        catalog = _merge_metadata(catalog, metadata_df)

    # Guardar
    artifacts_dir = resolve_dataset_artifacts(dataset_name, config)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / "catalog.parquet"
    catalog.to_parquet(out_path, index=False)
    if not dropped.empty:
        dropped.to_csv(artifacts_dir / "duplicates.csv", index=False)

    print(f"[INFO] Catalog: {len(catalog)} tracks OK, {n_failed} failed, "
          f"{len(dropped)} duplicate files dropped → {out_path}")
    return catalog


def _merge_metadata(catalog: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge metadata externa. Intentar por filename normalizado.
    Loguear cuántos matchearon.
    """
    if "filename" not in metadata_df.columns:
        return catalog

    metadata_df = metadata_df.copy()
    metadata_df["_norm_filename"] = metadata_df["filename"].apply(_normalize_filename)
    catalog["_norm_filename"] = catalog["filename"].apply(_normalize_filename)

    meta_cols = [c for c in metadata_df.columns if c not in ("filename", "_norm_filename")]
    merged = catalog.merge(
        metadata_df[["_norm_filename"] + meta_cols],
        on="_norm_filename",
        how="left",
        suffixes=("", "_meta"),
    )
    n_matched = merged[meta_cols[0]].notna().sum() if meta_cols else 0
    print(f"[INFO] Metadata merge: {n_matched}/{len(catalog)} tracks matched")
    merged = merged.drop(columns=["_norm_filename"])
    return merged


def load_catalog(dataset_name: str, config: dict) -> pd.DataFrame:
    """Cargar catálogo existente desde artifacts."""
    artifacts_dir = resolve_dataset_artifacts(dataset_name, config)
    path = artifacts_dir / "catalog.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {path}")
    return pd.read_parquet(path)


def update_catalog_columns(dataset_name: str, config: dict, updates: pd.DataFrame) -> pd.DataFrame:
    """
    Agregar/actualizar columnas al catálogo existente.
    updates debe tener columna 'track_uid' para el join.
    Guarda y retorna el catálogo actualizado.
    """
    catalog = load_catalog(dataset_name, config)
    if "track_uid" not in updates.columns:
        raise ValueError("updates DataFrame must have 'track_uid' column")

    update_cols = [c for c in updates.columns if c != "track_uid"]
    for col in update_cols:
        if col in catalog.columns:
            catalog = catalog.drop(columns=[col])
    catalog = catalog.merge(updates[["track_uid"] + update_cols], on="track_uid", how="left")

    artifacts_dir = resolve_dataset_artifacts(dataset_name, config)
    catalog.to_parquet(artifacts_dir / "catalog.parquet", index=False)
    return catalog
