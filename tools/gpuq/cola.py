"""
PURPOSE: Encolar y seguir trabajos de gpuq desde este repo. Envuelve el comando `gpuq` del clon del repo de
         la cola (GPUQ_HOME, por defecto la carpeta hermana `gpuq-personal`, que tiene gpuq.toml):
           python tools/gpuq/cola.py add <id> --cmd "python tools/gpuq/smoke_drive.py run --run x"
           python tools/gpuq/cola.py status | list | logs <id> | cancel <id>
         `add` fija --repo en este repo (sacado de `git remote get-url origin`) y --ref en el commit actual,
         y se niega si ese commit no está en GitHub (la VM no podría bajarlo). El resto de las opciones pasa
         tal cual a `gpuq add` (--gpu, --platform, --setup, --no-setup, --depends-on, --timeout-h, ...).
         Cada clase nueva de trabajo necesita antes el OK de Gabriel en el chat (AGENTS.md). Guía: docs/GPUQ.md.
CHANGELOG:
  - 2026-11-09: primera versión (puesta en marcha de gpuq en este repo); el gpuq hijo escribe en UTF-8.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from src.v4.common.remote_storage import gpuq_home  # noqa: E402

PASS_THROUGH = ("status", "list", "logs", "cancel")


def repo_slug(url: str) -> str:
    """owner/name from a GitHub remote URL (https or ssh)."""
    m = re.search(r"github\.com[:/]+([^/]+)/([^/]+?)(?:\.git)?/?$", url.strip())
    if not m:
        raise ValueError(f"no reconozco el remoto de GitHub {url!r}")
    return f"{m.group(1)}/{m.group(2)}"


def _git(*args: str) -> str:
    r = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"git {args[0]} falló: {r.stderr.strip()[-400:]}")
    return r.stdout.strip()


def has_flag(argv: list, name: str) -> bool:
    return any(a == name or a.startswith(name + "=") for a in argv)


def build_add(argv: list, repo: str, sha: str) -> list:
    """`gpuq add` arguments: the user's, plus --repo and --ref unless given."""
    out = ["add", *argv]
    if not has_flag(argv, "--repo"):
        out += ["--repo", repo]
    if not has_flag(argv, "--ref"):
        out += ["--ref", sha]
    return out


def run_gpuq(args: list) -> int:
    home = gpuq_home()
    if not (home / "gpuq.toml").is_file():
        raise SystemExit(f"no encuentro gpuq.toml en {home}: cloná gbibbo/gpuq-personal ahí o definí GPUQ_HOME")
    env = {**os.environ, "PYTHONPATH": str(home) + os.pathsep + os.environ.get("PYTHONPATH", ""),
           "PYTHONIOENCODING": "utf-8"}  # logs with accents, also on cp1252 consoles
    return subprocess.call([sys.executable, "-m", "gpuq", *args], cwd=str(home), env=env)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.split("CHANGELOG")[0])
        return 0
    if argv[0] in PASS_THROUGH:
        return run_gpuq(argv)
    if argv[0] != "add":
        raise SystemExit(f"comando desconocido {argv[0]!r}: add, {', '.join(PASS_THROUGH)}")
    repo = repo_slug(_git("remote", "get-url", "origin"))
    sha = _git("rev-parse", "HEAD")
    if not has_flag(argv, "--ref"):
        _git("fetch", "-q", "origin")
        if not _git("branch", "-r", "--contains", sha):
            raise SystemExit(f"el commit {sha[:12]} no está en GitHub: hacé git push antes de encolar")
        if _git("status", "--porcelain", "--untracked-files=no"):
            print("aviso: hay cambios sin commit; la VM corre el commit, no tu copia de trabajo")
    full = build_add(argv[1:], repo, sha)
    print("gpuq " + " ".join(full))
    return run_gpuq(full)


if __name__ == "__main__":
    sys.exit(main())
