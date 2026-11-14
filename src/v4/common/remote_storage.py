"""
PURPOSE: Almacenamiento remoto de TRAKTOR ML: el Google Drive de Gabriel, compartido entre esta PC y los
         trabajos de gpuq (Colab y Kaggle). En esta PC y en Kaggle se usa rclone; en Colab, la API del Drive
         desde Python (drive_api.py), porque Colab recupera toda VM que ejecuta rclone (use_drive_api). Un solo
         lugar resuelve la raíz:
           1. TRAKTOR_REMOTE, si está definida (p. ej. "erclm:traktor_ml" o una carpeta local en tests);
           2. si no, "<remoto de GPUQ_REMOTE>:traktor_ml" (en la VM gpuq define GPUQ_REMOTE, hoy
              "erclm:gpuq-test"; los datos de TRAKTOR van a su propia carpeta del mismo Drive);
           3. si no, "erclm:traktor_ml" (el nombre del remoto en el rclone.conf de esta PC).
         Reusa gpuq.storage (subida atómica verificada por tamaño y MD5, Checkpoints): en la VM gpuq lo
         pone en PYTHONPATH; en esta PC se importa del clon del repo de gpuq (GPUQ_HOME, por defecto la
         carpeta hermana del repo `gpuq-personal`). Instrucciones: docs/GPUQ.md.
CHANGELOG:
  - 2026-11-09: primera versión (puesta en marcha de gpuq en este repo).
  - 2026-11-13: TRAKTOR_DRIVE_API=1 abre el Drive por su API desde Python (drive_api.py) en lugar de rclone,
                porque Colab recupera toda VM que ejecuta rclone a los ~12 min.
  - 2026-11-13: en una VM de gpuq en Colab (GPUQ_PLATFORM=colab) ese camino es el predeterminado, después de
                que E8 viviera 20 min bajando tars reales y subiendo parciales por él.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

DEFAULT_RCLONE_REMOTE = "erclm"
DEFAULT_FOLDER = "traktor_ml"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _env(env=None):
    return os.environ if env is None else env


def is_local(base: str) -> bool:
    """Same rule as gpuq.storage.Remote.from_env: 'name:path' is rclone, anything else a local folder."""
    return ":" not in base.split("/")[0] or os.path.isabs(base)


def gpuq_home(env=None) -> Path:
    """Clone of the gpuq ops repo on this computer (it holds gpuq.toml and the gpuq package)."""
    return Path(_env(env).get("GPUQ_HOME") or REPO_ROOT.parent / "gpuq-personal")


def import_gpuq_storage(env=None):
    """gpuq.storage, from PYTHONPATH (on a gpuq VM) or from the local clone of the ops repo."""
    try:
        from gpuq import storage
        return storage
    except ImportError:
        pass
    home = gpuq_home(env)
    if not (home / "gpuq" / "storage.py").is_file():
        raise ImportError(f"gpuq no se puede importar: cloná gbibbo/gpuq-personal en {home} o definí GPUQ_HOME")
    sys.path.insert(0, str(home))
    from gpuq import storage
    return storage


def remote_root(env=None) -> str:
    """Root of TRAKTOR ML's remote storage, as rclone 'remote:folder' or a local folder."""
    env = _env(env)
    if env.get("TRAKTOR_REMOTE"):
        return env["TRAKTOR_REMOTE"].rstrip("/\\")
    gq = env.get("GPUQ_REMOTE") or ""
    if gq and is_local(gq):
        return str(Path(gq) / DEFAULT_FOLDER)
    name = gq.split(":", 1)[0] if gq else DEFAULT_RCLONE_REMOTE
    return f"{name}:{DEFAULT_FOLDER}"


def find_rclone(env=None):
    """rclone executable: RCLONE, then PATH, then a winget install not yet on this shell's PATH."""
    env = _env(env)
    if env.get("RCLONE"):
        return env["RCLONE"]
    found = shutil.which("rclone")
    if found:
        return found
    local = env.get("LOCALAPPDATA")
    if local:
        hits = sorted((Path(local) / "Microsoft" / "WinGet" / "Packages").glob("Rclone.Rclone_*/*/rclone.exe"))
        if hits:
            return str(hits[-1])
    return None


def use_drive_api(env=None) -> bool:
    """Whether open_remote() talks to Google Drive through its API from Python instead of through rclone.
    By default only on a gpuq VM on Colab, which reclaims every VM that runs rclone about 12 min after assignment
    (docs/reports/colab_corte_13min_2026-11-13.md; the API path lived 20 min there, E8). TRAKTOR_DRIVE_API=1
    forces it anywhere and =0 forbids it."""
    env = _env(env)
    forced = env.get("TRAKTOR_DRIVE_API")
    if forced in ("0", "1"):
        return forced == "1"
    return env.get("GPUQ_PLATFORM") == "colab"


def open_remote(env=None):
    """A gpuq Remote rooted at remote_root(): put/get/ls with verified, atomic transfers."""
    env = _env(env)
    storage = import_gpuq_storage(env)
    root = remote_root(env)
    if is_local(root):
        return storage.LocalRemote(root)
    if use_drive_api(env):
        from . import drive_api
        return drive_api.make_drive_remote(storage, root, drive_api.rclone_config_path(env))
    exe = find_rclone(env)
    if not exe:
        raise storage.StorageError("no encuentro rclone: winget install Rclone.Rclone (docs/GPUQ.md)")
    return storage.RcloneRemote(root, config=env.get("RCLONE_CONFIG"), rclone=exe)
