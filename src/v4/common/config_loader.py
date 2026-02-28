"""
PURPOSE: Cargar configuración V4 con cascada: CLI --config > env TRAKTOR_CONFIG > config/v4.yaml.
         Rutas pueden overridearse también con env vars (ver path_resolver).
CHANGELOG:
  - 2026-02-28: Creación inicial V4.
"""
import os
import re
from pathlib import Path
from typing import Optional

import yaml

from src.v4.config import DEFAULT_CONFIG_PATH


def _expand_env_vars(obj):
    """Expandir ${VAR} en strings del dict recursivamente."""
    if isinstance(obj, str):
        def _replace(match):
            var = match.group(1)
            return os.environ.get(var, match.group(0))
        return re.sub(r"\$\{([^}]+)\}", _replace, obj)
    elif isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(i) for i in obj]
    return obj


def load_config(config_path: Optional[Path] = None) -> dict:
    """
    Cargar configuración V4.

    Orden de precedencia:
      1. config_path (argumento explícito)
      2. Variable de entorno TRAKTOR_CONFIG
      3. REPO_ROOT/config/v4.yaml

    Las rutas del YAML pueden overridearse con env vars:
      - TRAKTOR_ARTIFACTS_ROOT
      - TRAKTOR_AUDIO_ROOTS (lista separada por ":")
      - TRAKTOR_CACHE_ROOT (root para caches) o TRAKTOR_HF_CACHE / TRAKTOR_TORCH_CACHE

    Dentro del YAML, strings con ${VAR} se expanden con os.environ.
    Valores null se dejan como None (se resuelven en path_resolver).
    """
    if config_path is None:
        env_path = os.environ.get("TRAKTOR_CONFIG")
        config_path = Path(env_path) if env_path else DEFAULT_CONFIG_PATH

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    config = _expand_env_vars(config)
    return config
