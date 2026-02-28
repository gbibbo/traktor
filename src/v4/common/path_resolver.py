"""
PURPOSE: Resolver rutas de audio, artifacts, caches y logs de forma portátil (laptop + HPC).
         Precedencia siempre: env var > YAML > default en REPO_ROOT.
CHANGELOG:
  - 2026-02-28: Creación inicial V4.
"""
import os
from pathlib import Path
from typing import Optional

from src.v4.config import REPO_ROOT


def resolve_artifacts_root(config: dict) -> Path:
    """
    Precedencia:
      1. env TRAKTOR_ARTIFACTS_ROOT
      2. config.paths.artifacts_root
      3. REPO_ROOT/artifacts/v4/datasets/
    """
    env_val = os.environ.get("TRAKTOR_ARTIFACTS_ROOT")
    if env_val:
        return Path(env_val)
    yaml_val = config.get("paths", {}).get("artifacts_root")
    if yaml_val:
        return Path(yaml_val)
    return REPO_ROOT / "artifacts" / "v4" / "datasets"


def resolve_dataset_audio_root(
    dataset_name: str,
    config: dict,
    cli_override: Optional[str] = None,
) -> Path:
    """
    Busca en orden:
      1. cli_override (argumento de línea de comandos)
      2. config.datasets[dataset_name].audio_root
      3. Cada directorio en env TRAKTOR_AUDIO_ROOTS (separado por ":") / dataset_name
      4. Cada directorio en config.paths.audio_roots / dataset_name
      5. REPO_ROOT/data/raw_audio/dataset_name/

    Si no encuentra ninguno, lanza FileNotFoundError con las rutas probadas.
    """
    tried = []

    if cli_override:
        p = Path(cli_override)
        if p.exists():
            return p
        tried.append(str(p))

    ds_cfg = config.get("datasets", {}).get(dataset_name, {}) or {}
    ds_audio = ds_cfg.get("audio_root")
    if ds_audio:
        p = Path(ds_audio)
        if p.exists():
            return p
        tried.append(str(p))

    env_roots = os.environ.get("TRAKTOR_AUDIO_ROOTS", "")
    for root in env_roots.split(":"):
        root = root.strip()
        if not root:
            continue
        p = Path(root) / dataset_name
        if p.exists():
            return p
        tried.append(str(p))

    yaml_roots = config.get("paths", {}).get("audio_roots", []) or []
    for root in yaml_roots:
        if not root:
            continue
        p = Path(root) / dataset_name
        if p.exists():
            return p
        tried.append(str(p))

    default = REPO_ROOT / "data" / "raw_audio" / dataset_name
    if default.exists():
        return default
    tried.append(str(default))

    raise FileNotFoundError(
        f"Audio root for dataset '{dataset_name}' not found. Tried:\n"
        + "\n".join(f"  - {t}" for t in tried)
    )


def resolve_dataset_metadata(dataset_name: str, config: dict) -> Optional[Path]:
    """Buscar metadata CSV externa. Retorna None si no hay."""
    ds_cfg = config.get("datasets", {}).get(dataset_name, {}) or {}
    val = ds_cfg.get("metadata_csv")
    if val:
        p = Path(val)
        return p if p.exists() else None
    return None


def resolve_dataset_manifest(dataset_name: str, config: dict) -> Optional[Path]:
    """Si existe, retorna ruta a manifest CSV (local/http)."""
    ds_cfg = config.get("datasets", {}).get(dataset_name, {}) or {}
    val = ds_cfg.get("manifest_csv")
    if val:
        p = Path(val)
        return p if p.exists() else None
    return None


def resolve_hf_cache(config: dict) -> Optional[Path]:
    """
    Precedencia:
      1. env TRAKTOR_HF_CACHE
      2. env TRAKTOR_CACHE_ROOT/hf
      3. config.paths.hf_cache
      4. None (HuggingFace usará su default)
    """
    val = os.environ.get("TRAKTOR_HF_CACHE")
    if val:
        return Path(val)
    cache_root = os.environ.get("TRAKTOR_CACHE_ROOT")
    if cache_root:
        return Path(cache_root) / "hf"
    yaml_val = config.get("paths", {}).get("hf_cache")
    if yaml_val:
        return Path(yaml_val)
    return None


def resolve_torch_cache(config: dict) -> Optional[Path]:
    """
    Precedencia:
      1. env TRAKTOR_TORCH_CACHE
      2. env TRAKTOR_CACHE_ROOT/torch
      3. config.paths.torch_cache
      4. None
    """
    val = os.environ.get("TRAKTOR_TORCH_CACHE")
    if val:
        return Path(val)
    cache_root = os.environ.get("TRAKTOR_CACHE_ROOT")
    if cache_root:
        return Path(cache_root) / "torch"
    yaml_val = config.get("paths", {}).get("torch_cache")
    if yaml_val:
        return Path(yaml_val)
    return None


def resolve_logs_root(dataset_name: str, config: dict) -> Path:
    """
    Directorio para logs JSONL.
    Por defecto: <artifacts_root>/<dataset_name>/logs/
    """
    artifacts_root = resolve_artifacts_root(config)
    return artifacts_root / dataset_name / "logs"


def resolve_dataset_artifacts(dataset_name: str, config: dict) -> Path:
    """Directorio raíz de artifacts para un dataset."""
    return resolve_artifacts_root(config) / dataset_name
