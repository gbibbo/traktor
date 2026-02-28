"""
PURPOSE: Catálogo central de dataset. Single source of truth para metadata de tracks.
         Soporta dos modos de hashing: full (SHA256 archivo completo, default) y
         fast (SHA256 de primeros N bytes + filesize).
CHANGELOG:
  - 2026-02-28: Creación inicial V4. Hash modes: full (default) y fast.
"""
import hashlib
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import soundfile as sf

from src.v4.config import TRACK_UID_BYTES_TO_READ
from src.v4.common.path_resolver import resolve_artifacts_root, resolve_dataset_artifacts


def compute_track_uid(filepath: Path, mode: str = "full", bytes_to_read: int = TRACK_UID_BYTES_TO_READ) -> str:
    """
    Calcular hash estable de contenido.

    mode="full" (default robusto): SHA256 streaming de TODO el archivo.
      64 chars hex. Sin colisiones prácticas. Más lento pero canónico.
    mode="fast": SHA256(primeros bytes_to_read bytes + filesize_bytes como string).
      Para datasets masivos donde performance importa.

    Returns: hex string de 64 chars (SHA256 completo, nunca truncado).
    """
    filepath = Path(filepath)
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
    """Obtener duración en segundos con soundfile. Retorna None si falla."""
    try:
        info = sf.info(str(filepath))
        return info.duration
    except Exception:
        return None


def build_catalog(
    audio_dir: Path,
    dataset_name: str,
    config: dict,
    metadata_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Escanear directorio y construir catálogo.

    Columnas mínimas: track_uid, filename, source_path, duration_s, filesize_bytes,
                      artist, title.
    Si metadata_df está disponible, merge por filename normalizado.
    Guarda en artifacts/v4/datasets/<dataset_name>/catalog.parquet.

    Returns: DataFrame del catálogo (solo tracks válidos con duration_s no None).
    """
    audio_dir = Path(audio_dir)
    hashing_cfg = config.get("hashing", {})
    hash_mode = hashing_cfg.get("mode", "full")
    hash_bytes = hashing_cfg.get("fast_bytes_to_read", TRACK_UID_BYTES_TO_READ)

    extensions = (".mp3", ".wav", ".flac", ".aiff", ".aif", ".m4a")
    audio_files = sorted([
        f for f in audio_dir.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ])

    rows = []
    n_failed = 0
    for filepath in audio_files:
        duration = _get_duration(filepath)
        if duration is None:
            n_failed += 1
            continue
        artist, title = _parse_artist_title(filepath.name)
        uid = compute_track_uid(filepath, mode=hash_mode, bytes_to_read=hash_bytes)
        rows.append({
            "track_uid": uid,
            "filename": filepath.name,
            "source_path": str(filepath),
            "duration_s": duration,
            "filesize_bytes": filepath.stat().st_size,
            "artist": artist,
            "title": title,
        })

    catalog = pd.DataFrame(rows)

    # Merge con metadata externa si disponible
    if metadata_df is not None and len(catalog) > 0:
        catalog = _merge_metadata(catalog, metadata_df)

    # Guardar
    artifacts_dir = resolve_dataset_artifacts(dataset_name, config)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / "catalog.parquet"
    catalog.to_parquet(out_path, index=False)

    print(f"[INFO] Catalog: {len(catalog)} tracks OK, {n_failed} failed → {out_path}")
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
