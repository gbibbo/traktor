"""
PURPOSE: Tests de tools/repo/subir.py sobre un repositorio temporal con un remoto: sube solo las rutas nombradas y deja
         intacto lo que otra sesión tiene preparado o sin guardar; archivos nuevos y borrados; candado ocupado y candado
         viejo; rama cambiada; mensaje con atribución; push que falla sin perder el commit; y el hook real, que rechaza
         un `git commit` directo y acepta el de la herramienta.
CHANGELOG:
  - 2026-11-13: Creación inicial.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.repo import subir  # noqa: E402


def git(repo: Path, *args: str, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8", env=env)


def lineas(repo: Path, *args: str) -> list:
    return [x for x in git(repo, *args).stdout.splitlines() if x]


@pytest.fixture
def clon(tmp_path: Path, monkeypatch) -> Path:
    remoto = tmp_path / "remoto.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remoto)], check=True)
    repo = tmp_path / "clon"
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    git(repo, "config", "user.name", "t")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "core.autocrlf", "false")
    git(repo, "remote", "add", "origin", str(remoto))
    for nombre in ("a.txt", "b.txt", "c.txt", "d.txt"):
        (repo / nombre).write_text("uno\n", encoding="utf-8")
    (repo / ".gitignore").write_text("datos/\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "init")
    assert git(repo, "push", "-q", "-u", "origin", "main").returncode == 0
    monkeypatch.chdir(repo)
    monkeypatch.setattr(subir, "PUSH_ESPERAS", (0,))
    monkeypatch.delenv("TRAKTOR_SUBIR", raising=False)
    return repo


def cabeza(repo: Path) -> str:
    return git(repo, "rev-parse", "HEAD").stdout.strip()


def test_sube_solo_las_rutas_nombradas_y_deja_lo_ajeno(clon, capsys):
    (clon / "a.txt").write_text("mío\n", encoding="utf-8")
    (clon / "b.txt").write_text("de otra sesión, preparado\n", encoding="utf-8")
    git(clon, "add", "b.txt")
    (clon / "c.txt").write_text("de otra sesión, sin preparar\n", encoding="utf-8")
    antes = cabeza(clon)

    assert subir.main(["-m", "mine only", "a.txt"]) == 0

    assert cabeza(clon) != antes
    assert lineas(clon, "show", "--name-only", "--format=", "HEAD") == ["a.txt"]
    assert lineas(clon, "diff", "--cached", "--name-only") == ["b.txt"]
    assert lineas(clon, "diff", "--name-only") == ["c.txt"]
    assert (clon / "b.txt").read_text(encoding="utf-8") == "de otra sesión, preparado\n"
    assert git(clon.parent / "remoto.git", "rev-parse", "main").stdout.strip() == cabeza(clon)
    assert "PUSH hecho" in capsys.readouterr().out
    assert not (clon / ".git" / subir.CANDADO).exists()


def test_archivo_nuevo_borrado_y_carpeta(clon):
    (clon / "pkg").mkdir()
    (clon / "pkg" / "nuevo.py").write_text("x = 1\n", encoding="utf-8")
    (clon / "d.txt").unlink()
    (clon / "suelto.txt").write_text("no nombrado\n", encoding="utf-8")

    assert subir.main(["-m", "new and deleted", "pkg", "d.txt"]) == 0

    assert sorted(lineas(clon, "show", "--name-status", "--format=", "HEAD")) == ["A\tpkg/nuevo.py", "D\td.txt"]
    assert lineas(clon, "ls-files", "--others", "--exclude-standard") == ["suelto.txt"]


def test_nada_que_subir_no_crea_commit(clon, capsys):
    antes = cabeza(clon)
    assert subir.main(["-m", "nothing", "a.txt"]) == 0
    assert cabeza(clon) == antes
    assert "nada que subir" in capsys.readouterr().out


def test_candado_ocupado_y_candado_viejo(clon, capsys):
    (clon / "a.txt").write_text("mío\n", encoding="utf-8")
    candado = clon / ".git" / subir.CANDADO
    candado.mkdir()
    (candado / "info.json").write_text(json.dumps({"t": time.time(), "mensaje": "otra"}), encoding="utf-8")
    antes = cabeza(clon)

    assert subir.main(["-m", "blocked", "--esperar", "0.5", "a.txt"]) == 3
    assert cabeza(clon) == antes and candado.exists()
    assert "otra sesión está subiendo" in capsys.readouterr().err

    (candado / "info.json").write_text(json.dumps({"t": time.time() - subir.VIEJO_S - 10}), encoding="utf-8")
    assert subir.main(["-m", "after stale lock", "a.txt"]) == 0
    assert cabeza(clon) != antes and not candado.exists()


def test_rama_cambiada_en_el_arbol_compartido(clon, capsys):
    git(clon, "checkout", "-q", "-b", "otra")
    (clon / "a.txt").write_text("mío\n", encoding="utf-8")
    assert subir.main(["-m", "wrong branch", "a.txt"]) == 2
    assert "otra" in capsys.readouterr().err
    assert lineas(clon, "diff", "--name-only") == ["a.txt"]


def test_rechaza_atribucion_raiz_e_ignorados(clon, capsys):
    (clon / "a.txt").write_text("mío\n", encoding="utf-8")
    assert subir.main(["-m", "x\n\nCo-Authored-By: alguien", "a.txt"]) == 2
    assert subir.main(["-m", "all", "."]) == 2
    (clon / "datos").mkdir()
    (clon / "datos" / "audio.bin").write_bytes(b"\x00")
    assert subir.main(["-m", "data", "datos/audio.bin"]) == 2
    assert subir.main(["-m", "missing", "no_existe.txt"]) == 2
    capsys.readouterr()
    assert lineas(clon, "diff", "--name-only") == ["a.txt"]


def test_push_que_falla_no_pierde_el_commit(clon, capsys):
    git(clon, "remote", "set-url", "origin", str(clon.parent / "no_existe.git"))
    (clon / "a.txt").write_text("mío\n", encoding="utf-8")
    antes = cabeza(clon)
    assert subir.main(["-m", "offline", "a.txt"]) == 0
    assert cabeza(clon) != antes
    assert "PUSH pendiente" in capsys.readouterr().out

    git(clon, "remote", "set-url", "origin", str(clon.parent / "remoto.git"))
    assert subir.main(["-m", "nothing new", "a.txt"]) == 0
    assert git(clon.parent / "remoto.git", "rev-parse", "main").stdout.strip() == cabeza(clon)


def test_el_hook_real_rechaza_el_commit_directo_y_acepta_la_herramienta(clon, monkeypatch):
    git(clon, "config", "core.hooksPath", (REPO_ROOT / "tools" / "git_hooks").as_posix())
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
    (clon / "a.txt").write_text("mío\n", encoding="utf-8")
    git(clon, "add", "a.txt")
    antes = cabeza(clon)

    directo = git(clon, "commit", "-q", "-m", "raw")
    assert directo.returncode != 0 and "subir.py" in directo.stderr
    assert cabeza(clon) == antes

    assert subir.main(["-m", "through the tool", "a.txt"]) == 0
    assert cabeza(clon) != antes
