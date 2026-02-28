"""
PURPOSE: Logging consistente (JSONL) + helpers para manifests reproducibles en V4.
         Cada fase escribe un archivo .jsonl en logs/ con eventos estandarizados.
CHANGELOG:
  - 2026-02-28: Creación inicial V4.
"""
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Optional


def compute_config_hash(obj: dict) -> str:
    """
    Hash corto (SHA1, primeros 8 chars) de un dict canonizado.
    Sirve para identificar una configuración en nombre de archivos.
    """
    canonical = json.dumps(obj, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(canonical.encode()).hexdigest()[:8]


def get_git_commit(repo_root: Path) -> str:
    """Retorna hash del commit actual o 'unknown' si no hay git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_slurm_job_id() -> Optional[str]:
    """Lee SLURM_ARRAY_JOB_ID (preferido) o SLURM_JOB_ID si existen."""
    return (
        os.environ.get("SLURM_ARRAY_JOB_ID")
        or os.environ.get("SLURM_JOB_ID")
        or None
    )


def open_phase_log(logs_root: Path, phase_name: str) -> IO:
    """
    Crear archivo de log JSONL para una fase.
    Nombre: logs/<phase_name>_<timestamp_utc>.jsonl

    Returns: file handle abierto para escritura (caller debe cerrarlo).
    """
    logs_root = Path(logs_root)
    logs_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = logs_root / f"{phase_name}_{ts}.jsonl"
    return open(log_path, "w", encoding="utf-8", buffering=1)  # line-buffered


def log_event(fh: IO, event: dict) -> None:
    """
    Escribir un evento JSONL al file handle.
    Agrega automáticamente timestamp_utc (ISO 8601).

    Schema mínimo recomendado:
      - timestamp_utc (auto)
      - phase: str
      - dataset_name: str
      - event_type: str  (e.g. 'track_ok', 'track_failed', 'phase_start', 'phase_end')
    Opcionales:
      - track_uid, filepath, status, duration_ms, error
    """
    event = dict(event)
    event["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    fh.write(json.dumps(event, ensure_ascii=False) + "\n")
