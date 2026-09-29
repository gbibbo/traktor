"""
PURPOSE: Chequeo determinista antes de cada commit (hook pre-commit de git). Bloquea lo que AGENTS.md
         prohíbe commitear: audio, embeddings y tablas generadas (.npy, .parquet), pesos y checkpoints
         de modelos, archivos de credenciales (.env, cachés de tokens de Spotify, claves privadas),
         archivos de más de 5 MB y contenido que lleve una credencial: un token con formato conocido,
         un literal asignado a un nombre de secreto, o el valor exacto de un secreto de los .env y
         cachés de tokens locales (git-ignorados).
         Decide existencia y contenencia, no significado: ante la duda no bloquea, porque una falsa
         alarma enseña a saltearse el chequeo. Nunca imprime el valor de una credencial.
         Se activa una vez por clon: `git config core.hooksPath tools/git_hooks`.
         Sin argumentos revisa lo staged (lo que entraría al commit); `--all` revisa todos los
         archivos versionados y los nuevos no ignorados del árbol de trabajo.
CHANGELOG:
  - 2026-09-29: Creación inicial.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Tuple

MAX_BYTES = 5 * 1024 * 1024

FORBIDDEN_SUFFIXES = {
    # audio
    ".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".aac", ".m4a", ".opus", ".wma", ".alac",
    # arrays y tablas generadas por el pipeline
    ".npy", ".npz", ".parquet",
    # pesos y checkpoints de modelos
    ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".h5", ".pkl", ".joblib", ".pb",
}
ENV_TEMPLATES = {".env.example", ".env.sample", ".env.template", ".env.dist"}
PRIVATE_KEY_NAMES = {"id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"}
PRIVATE_KEY_SUFFIXES = (".pem", ".key", ".p12", ".pfx")

TOKEN_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("clave privada", re.compile(r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----")),
    ("API key de Anthropic", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    ("API key de OpenAI", re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{32,}")),
    ("token de GitHub", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})")),
    ("access key de AWS", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("token de Slack", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}")),
    ("API key de Google", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}")),
]
# Un literal entre comillas asignado a un nombre de secreto: client_secret = "…", "api_key": "…".
SECRET_ASSIGNMENT = re.compile(
    r"""(?i)\b[\w.\-]*(?:secret|token|passw(?:or)?d|api[_\-]?key|access[_\-]?key|private[_\-]?key)[\w.\-]*"""
    r"""["']?\s*[:=]\s*[rbuf]{0,2}["']([^"'\s]{16,})["']"""
)
PLACEHOLDER = re.compile(r"(?i)your|example|placeholder|changeme|dummy|fake|redacted|xxx|[<>${}%*]")
SECRET_KEY_NAME = re.compile(r"(?i)secret|token|passw|pwd|api_?key|auth|credential|private")


@dataclass(frozen=True)
class Problem:
    path: str
    line: Optional[int]
    reason: str

    def render(self) -> str:
        where = self.path if self.line is None else f"{self.path}:{self.line}"
        return f"  {where}: {self.reason}"


def git(repo: Path, *args: str, stdin: Optional[bytes] = None) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], input=stdin, capture_output=True, check=True).stdout


def split_z(raw: bytes) -> List[str]:
    return [p.decode("utf-8", "surrogateescape") for p in raw.split(b"\0") if p]


def path_problem(path: str) -> Optional[str]:
    """Motivo por el que la ruta no se puede commitear, solo por su nombre."""
    name = PurePosixPath(path).name.lower()
    suffix = PurePosixPath(name).suffix
    if suffix in FORBIDDEN_SUFFIXES:
        return f"extensión {suffix} (audio, embeddings, tablas o pesos generados)"
    if name == ".env" or (name.startswith(".env.") and name not in ENV_TEMPLATES):
        return "archivo .env con credenciales"
    if name in {".spotify_token_cache", ".cache"} or name.startswith(".cache-"):
        return "caché de tokens de Spotify"
    if name in PRIVATE_KEY_NAMES or name.endswith(PRIVATE_KEY_SUFFIXES):
        return "clave privada"
    return None


def looks_like_secret(value: str) -> bool:
    """Un literal de 16+ caracteres parece secreto si mezcla letras y dígitos y no es un marcador,
    una URL, un nombre de variable de entorno ni un nombre de archivo."""
    if PLACEHOLDER.search(value) or "://" in value:
        return False
    if not (re.search(r"[A-Za-z]", value) and re.search(r"\d", value)):
        return False
    if re.fullmatch(r"[A-Z][A-Z0-9_]*", value):
        return False
    if re.fullmatch(r"[\w./\\\-]+\.[A-Za-z]{2,5}", value):
        return False
    if re.fullmatch(r"[a-z]+(?:[_\-][a-z0-9]+)+", value):
        return False
    return True


def parse_env_file(text: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


def local_secrets(repo: Path) -> Dict[str, str]:
    """Valores de secretos de los .env y cachés de tokens git-ignorados del repo: valor → origen."""
    ignored = split_z(git(repo, "ls-files", "-z", "--others", "--ignored", "--exclude-standard", "--directory"))
    secrets: Dict[str, str] = {}
    for rel in ignored:
        if rel.endswith("/"):
            continue
        name = PurePosixPath(rel).name.lower()
        file = repo / rel
        try:
            text = file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if name == ".env" or (name.startswith(".env.") and name not in ENV_TEMPLATES):
            for key, value in parse_env_file(text).items():
                if SECRET_KEY_NAME.search(key) and len(value) >= 8:
                    secrets[value] = f"el valor de {key} de {rel}"
        elif name in {".spotify_token_cache", ".cache"} or name.startswith(".cache-"):
            try:
                data = json.loads(text)
            except ValueError:
                continue
            if isinstance(data, dict):
                for key, value in data.items():
                    if "token" in str(key).lower() and isinstance(value, str) and len(value) >= 20:
                        secrets[value] = f"el valor de {key} de {rel}"
    return secrets


def content_problems(path: str, data: bytes, secrets: Dict[str, str]) -> List[Problem]:
    if b"\0" in data[:8192]:
        return []
    problems: List[Problem] = []
    for number, line in enumerate(data.decode("utf-8", "replace").splitlines(), start=1):
        for label, pattern in TOKEN_PATTERNS:
            if pattern.search(line):
                problems.append(Problem(path, number, f"parece un {label}"))
        for match in SECRET_ASSIGNMENT.finditer(line):
            if looks_like_secret(match.group(1)):
                problems.append(Problem(path, number, "literal asignado a un nombre de secreto"))
        for value, origin in secrets.items():
            if value in line:
                problems.append(Problem(path, number, f"contiene {origin}"))
    return problems


def check(paths: Iterable[str], size_of, read, secrets: Dict[str, str]) -> List[Problem]:
    problems: List[Problem] = []
    for path in paths:
        reason = path_problem(path)
        if reason:
            problems.append(Problem(path, None, reason))
            continue
        size = size_of(path)
        if size is None:
            continue
        if size > MAX_BYTES:
            problems.append(Problem(path, None, f"pesa {size / 1024 / 1024:.1f} MB (tope {MAX_BYTES // 1024 // 1024} MB)"))
            continue
        problems.extend(content_problems(path, read(path), secrets))
    return problems


def staged(repo: Path, secrets: Dict[str, str]) -> List[Problem]:
    paths = split_z(git(repo, "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMRT"))
    if not paths:
        return []
    query = "".join(f":{p}\n" for p in paths).encode("utf-8", "surrogateescape")
    checks = git(repo, "cat-file", "--batch-check=%(objecttype) %(objectsize)", stdin=query).decode().splitlines()
    sizes: Dict[str, Optional[int]] = {}
    for path, row in zip(paths, checks):
        kind, _, size = row.partition(" ")
        sizes[path] = int(size) if kind == "blob" else None
    return check(paths, sizes.get, lambda p: git(repo, "cat-file", "blob", f":{p}"), secrets)


def worktree(repo: Path, secrets: Dict[str, str]) -> List[Problem]:
    tracked = split_z(git(repo, "ls-files", "-z"))
    untracked = split_z(git(repo, "ls-files", "-z", "--others", "--exclude-standard"))
    paths = [p for p in tracked + untracked if (repo / p).is_file()]
    return check(paths, lambda p: (repo / p).stat().st_size, lambda p: (repo / p).read_bytes(), secrets)


def main(argv: Optional[List[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.split("CHANGELOG")[0])
    parser.add_argument("--all", action="store_true", help="revisar todo el árbol de trabajo en vez de lo staged")
    parser.add_argument("--repo", type=Path, default=None, help="raíz del repo (por defecto, la del directorio actual)")
    args = parser.parse_args(argv)

    repo = args.repo or Path(git(Path.cwd(), "rev-parse", "--show-toplevel").decode().strip())
    secrets = local_secrets(repo)
    problems = worktree(repo, secrets) if args.all else staged(repo, secrets)
    if not problems:
        if args.all:
            print("check_staged: sin problemas en el árbol de trabajo.")
        return 0

    what = "El árbol de trabajo tiene" if args.all else "El commit tiene"
    print(f"check_staged: {what} cosas que AGENTS.md no permite commitear:", file=sys.stderr)
    for problem in problems:
        print(problem.render(), file=sys.stderr)
    if not args.all:
        print(
            "Sacalas del commit con `git restore --staged <ruta>` (o `git rm --cached <ruta>` si ya estaba\n"
            "versionada). Si es una falsa alarma, se corrige tools/git_hooks/check_staged.py; el chequeo no\n"
            "se saltea.",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
