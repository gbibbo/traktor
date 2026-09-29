"""
PURPOSE: Tests de tools/git_hooks/check_staged.py sobre repos git temporales: bloquea audio, arrays,
         pesos, .env, cachés de tokens, archivos grandes y contenido con credenciales (sin imprimir el
         valor); deja pasar código normal, plantillas .env.example, marcadores y el borrado de un archivo
         prohibido; y el hook pre-commit real bloquea un `git commit`.
         Los tokens de prueba se arman en tiempo de ejecución para que este archivo no dispare el chequeo.
CHANGELOG:
  - 2026-09-29: Creación inicial.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.git_hooks import check_staged  # noqa: E402

SPOTIFY_SECRET = "9f3c" + "a1b2c3d4e5f60718293a4b5c6d7e"


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        capture_output=True, text=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_text(".env\n.spotify_token_cache\n", encoding="utf-8")
    (tmp_path / ".env").write_text(
        f"SPOTIFY_CLIENT_ID=abc\nSPOTIFY_CLIENT_SECRET={SPOTIFY_SECRET}\nSPOTIFY_REDIRECT_URI=http://127.0.0.1:8888\n",
        encoding="utf-8",
    )
    git(tmp_path, "add", ".gitignore")
    git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def stage(repo: Path, rel: str, content) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    assert git(repo, "add", "-f", rel).returncode == 0


def run(repo: Path, capsys) -> tuple:
    code = check_staged.main(["--repo", str(repo)])
    return code, capsys.readouterr().err


def test_normal_code_passes(repo, capsys):
    stage(repo, "src/a.py", 'client_secret = os.environ["SPOTIFY_CLIENT_SECRET"]\nprint("hola")\n')
    stage(repo, ".env.example", "SPOTIFY_CLIENT_SECRET=your-secret-here\n")
    assert run(repo, capsys) == (0, "")


@pytest.mark.parametrize("rel", ["data/emb.npy", "x/Track.MP3", "a/b.parquet", "m/model.safetensors"])
def test_forbidden_suffixes_blocked(repo, capsys, rel):
    stage(repo, rel, b"\x00\x01")
    code, err = run(repo, capsys)
    assert code == 1 and rel in err


@pytest.mark.parametrize("rel", [".env", "download_JIJIJI/.env.local", ".spotify_token_cache", ".cache-gabriel", "id_rsa"])
def test_credential_files_blocked(repo, capsys, rel):
    stage(repo, rel, "x=1\n")
    code, err = run(repo, capsys)
    assert code == 1 and rel in err


def test_large_file_blocked(repo, capsys, monkeypatch):
    monkeypatch.setattr(check_staged, "MAX_BYTES", 100)
    stage(repo, "notes.txt", "a" * 200)
    code, err = run(repo, capsys)
    assert code == 1 and "notes.txt" in err and "MB" in err


def test_local_env_secret_blocked_without_printing_it(repo, capsys):
    stage(repo, "download_JIJIJI/run.py", f'CFG = {{"id": "abc", "s": "{SPOTIFY_SECRET}"}}\n')
    code, err = run(repo, capsys)
    assert code == 1
    assert "download_JIJIJI/run.py:1" in err and "SPOTIFY_CLIENT_SECRET" in err
    assert SPOTIFY_SECRET not in err


def test_known_token_format_blocked(repo, capsys):
    token = "gh" + "p_" + "A1b2C3d4" * 5
    stage(repo, "README.md", f"usar {token} para clonar\n")
    code, err = run(repo, capsys)
    assert code == 1 and "README.md:1" in err and token not in err


def test_secret_assignment_literal(repo, capsys):
    stage(repo, "a.py", 'api_key = "' + "Zq8" + 'x7Lm2Pw9Rt4Vb6Nc"\n')
    code, err = run(repo, capsys)
    assert code == 1 and "a.py:1" in err


@pytest.mark.parametrize("line", [
    'client_secret = "<tu-client-secret-de-spotify>"',
    '"access_token": "SPOTIFY_ACCESS_TOKEN_VALUE"',
    'token_path = "cache/spotify_token_2026.json"',
    'password = "not_a_real_password_123"',
])
def test_placeholders_pass(repo, capsys, line):
    stage(repo, "b.py", line + "\n")
    assert run(repo, capsys) == (0, "")


def test_removing_forbidden_file_passes(repo, capsys):
    stage(repo, "old/emb.npy", b"\x00")
    git(repo, "commit", "-q", "--no-verify", "-m", "legacy")
    assert git(repo, "rm", "-q", "--cached", "old/emb.npy").returncode == 0
    assert run(repo, capsys) == (0, "")


def test_pre_commit_hook_blocks_git_commit(repo):
    hooks = (REPO_ROOT / "tools" / "git_hooks").as_posix()
    git(repo, "config", "core.hooksPath", hooks)
    env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))

    stage(repo, "ok.py", "x = 1\n")
    ok = subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com",
                         "commit", "-q", "-m", "ok"], capture_output=True, text=True, env=env)
    assert ok.returncode == 0, ok.stderr

    stage(repo, "track.wav", b"RIFF\x00")
    bad = subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com",
                          "commit", "-q", "-m", "bad"], capture_output=True, text=True, env=env)
    assert bad.returncode != 0 and "track.wav" in bad.stderr
