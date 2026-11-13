"""
PURPOSE: El único camino para hacer commit y push en este repositorio. Varias sesiones de agentes comparten el mismo
         árbol de trabajo, el mismo índice de Git y la misma rama; sin coordinación, una sesión sube lo que otra dejó
         preparado, o tiene que separar a mano los bloques de un documento compartido. Esta herramienta:
         - espera su turno con un candado en la carpeta de Git (uno solo para todos los árboles del repositorio);
         - hace el commit SOLO de las rutas que se le nombran, con un índice propio (`git commit --only`): lo que otra
           sesión tenga preparado o sin guardar en otros archivos queda como estaba;
         - corre el chequeo pre-commit sobre ese commit y no sobre lo ajeno;
         - sube a la rama remota y, si la red falla, deja el commit local y lo sube en la corrida siguiente;
         - suelta el candado siempre.
         No integra ni reescribe historia: si el remoto tiene commits que este clon no tiene, se detiene y lo dice.
         Solo biblioteca estándar.
         Uso: python tools/repo/subir.py -m "mensaje" <ruta> [<ruta> ...] [--sin-push] [--esperar SEG]
              python tools/repo/subir.py --estado
         Salidas: 0 hecho (o nada que subir); 2 uso o ruta inválida; 3 candado ocupado; 4 el remoto se adelantó;
                  5 el commit fue rechazado (chequeo pre-commit o Git).
CHANGELOG:
  - 2026-11-13: Creación inicial (Gabriel: que las cuatro sesiones se coordinen solas al subir, sin choques ni pérdidas).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

CANDADO = "traktor_subir.lock"
ESPERA_S = 300                      # cuánto se espera el turno
VIEJO_S = 900                       # un candado más viejo que esto quedó de una corrida muerta: se rompe
REINTENTOS_INDICE = 15              # otra sesión puede tener el índice tomado un instante (git status, git add)
PUSH_ESPERAS = (0, 5, 15)           # segundos antes de cada intento de push
ATRIBUCION = re.compile(r"co-authored-by:|generated with|claude\.ai/code|🤖", re.I)


class Falla(Exception):
    def __init__(self, codigo: int, texto: str):
        super().__init__(texto)
        self.codigo = codigo


def correr(args: Sequence[str], cwd: Path, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def git(args: Sequence[str], cwd: Path, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    """Como correr(), pero vuelve a probar si otra sesión tiene el índice tomado en ese instante."""
    r = correr(args, cwd, env)
    for i in range(REINTENTOS_INDICE):
        if r.returncode == 0 or "index.lock" not in (r.stderr or ""):
            break
        time.sleep(0.4 + 0.2 * i)
        r = correr(args, cwd, env)
    return r


def salida(args: Sequence[str], cwd: Path) -> str:
    r = git(args, cwd)
    if r.returncode:
        raise Falla(5, f"git {' '.join(args)}: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout


class Candado:
    """Exclusión entre sesiones: crear una carpeta es atómico en todos los sistemas de archivos."""

    def __init__(self, comun: Path, mensaje: str = ""):
        self.dir = comun / CANDADO
        self.info = self.dir / "info.json"
        self.mensaje = mensaje
        self.mio = False

    def leer(self) -> Optional[dict]:
        try:
            return json.loads(self.info.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def edad(self) -> float:
        info = self.leer()
        try:
            return time.time() - (info["t"] if info and "t" in info else self.dir.stat().st_mtime)
        except OSError:
            return 0.0

    def romper(self) -> None:
        for paso in (lambda: self.info.unlink(), lambda: self.dir.rmdir()):
            try:
                paso()
            except OSError:
                pass

    def tomar(self, espera: float) -> None:
        fin = time.time() + espera
        avisado = False
        while True:
            try:
                os.mkdir(self.dir)
            except FileExistsError:
                edad = self.edad()
                if edad > VIEJO_S:
                    print(f"[subir] candado de hace {edad:.0f} s, de una corrida que no terminó: lo retiro", flush=True)
                    self.romper()
                    continue
                if time.time() >= fin:
                    quien = (self.leer() or {}).get("mensaje", "")
                    raise Falla(3, f"otra sesión está subiendo desde hace {edad:.0f} s"
                                   + (f" («{quien[:60]}»)" if quien else "")
                                   + ". No se perdió nada: vuelve a correr este mismo comando en un minuto.")
                if not avisado:
                    print("[subir] otra sesión está subiendo; espero mi turno...", flush=True)
                    avisado = True
                time.sleep(2)
                continue
            self.mio = True
            self.info.write_text(json.dumps({"t": time.time(), "pid": os.getpid(), "equipo": socket.gethostname(),
                                             "mensaje": self.mensaje}, ensure_ascii=False), encoding="utf-8")
            return

    def soltar(self) -> None:
        if self.mio:
            self.romper()
            self.mio = False


def contexto(cwd: Path) -> dict:
    raiz = Path(salida(["rev-parse", "--show-toplevel"], cwd).strip())
    comun = Path(salida(["rev-parse", "--git-common-dir"], raiz).strip())
    gitdir = Path(salida(["rev-parse", "--git-dir"], raiz).strip())
    comun = comun if comun.is_absolute() else (raiz / comun)
    gitdir = gitdir if gitdir.is_absolute() else (raiz / gitdir)
    rama = git(["symbolic-ref", "--short", "-q", "HEAD"], raiz).stdout.strip()
    return {"raiz": raiz, "comun": comun.resolve(), "gitdir": gitdir.resolve(), "rama": rama,
            "principal": (raiz / ".git").is_dir()}


def validar(ctx: dict, rutas: Sequence[str], mensaje: str, rama_pedida: Optional[str]) -> List[str]:
    raiz: Path = ctx["raiz"]
    if not mensaje.strip():
        raise Falla(2, "falta el mensaje del commit (-m)")
    if ATRIBUCION.search(mensaje):
        raise Falla(2, "el mensaje lleva atribución de IA o un enlace de sesión; AGENTS.md lo prohíbe. Quítalo.")
    if not rutas:
        raise Falla(2, "nombra las rutas que cambiaste; esta herramienta nunca sube «todo»")
    if not ctx["rama"]:
        raise Falla(2, "HEAD no está en una rama. No se sube desde un HEAD suelto.")
    esperada = rama_pedida or ("main" if ctx["principal"] else ctx["rama"])
    if ctx["rama"] != esperada:
        raise Falla(2, f"el árbol está en la rama «{ctx['rama']}» y se esperaba «{esperada}». En el árbol compartido no "
                       "se cambia de rama: otra sesión lo movió. Avísale (SendMessage) para que vuelva a «main»; el "
                       "trabajo en ramas va en un `git worktree`. Si de verdad corresponde esta rama, usa --rama.")
    for marca in ("MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "rebase-merge", "rebase-apply"):
        if (ctx["gitdir"] / marca).exists():
            raise Falla(2, f"hay una operación de Git a medio hacer ({marca}). Se termina o se aborta antes de subir.")
    limpias = []
    for r in rutas:
        p = (Path.cwd() / r).resolve() if not Path(r).is_absolute() else Path(r).resolve()
        try:
            rel = p.relative_to(raiz.resolve())
        except ValueError:
            raise Falla(2, f"{r}: está fuera del repositorio")
        if str(rel) in (".", ""):
            raise Falla(2, "la raíz del repositorio no es una ruta válida: nombra archivos o carpetas concretas")
        limpias.append(rel.as_posix())
    conocidas = set(salida(["ls-files", "-z", "--", *limpias], raiz).split("\0"))
    for rel in limpias:
        en_disco = (raiz / rel).exists()
        seguida = any(c == rel or c.startswith(rel.rstrip("/") + "/") for c in conocidas if c)
        if not en_disco and not seguida:
            raise Falla(2, f"{rel}: no existe y Git no la conoce")
    return limpias


def subir(ctx: dict, rutas: List[str], mensaje: str, con_push: bool) -> dict:
    raiz: Path = ctx["raiz"]
    env = dict(os.environ, TRAKTOR_SUBIR="1")
    nuevas = [x for x in salida(["ls-files", "--others", "--exclude-standard", "-z", "--", *rutas], raiz).split("\0") if x]
    ignoradas = [x for x in salida(["ls-files", "--others", "--ignored", "--exclude-standard", "-z", "--", *rutas],
                                   raiz).split("\0") if x and (raiz / x).is_file()]
    for r in rutas:
        if r in ignoradas:
            raise Falla(2, f"{r}: está ignorada por .gitignore (datos, audio, artefactos). Eso no va a Git.")
    if nuevas:
        r = git(["add", "--intent-to-add", "--", *nuevas], raiz)
        if r.returncode:
            raise Falla(5, f"no pude registrar los archivos nuevos: {r.stderr.strip()}")
    hay = bool(salida(["status", "--porcelain", "--", *rutas], raiz).strip())
    sha = None
    if hay:
        r = git(["commit", "--only", "-m", mensaje, "--", *rutas], raiz, env)
        if r.returncode:
            raise Falla(5, "el commit fue rechazado. Tus cambios siguen en el árbol.\n" + (r.stderr.strip() or r.stdout.strip()))
        sha = salida(["rev-parse", "HEAD"], raiz).strip()
    estado_push = "no pedido"
    if con_push:
        estado_push = empujar(ctx)
    return {"sha": sha, "push": estado_push, "nuevas": len(nuevas)}


def empujar(ctx: dict) -> str:
    raiz: Path = ctx["raiz"]
    rama = ctx["rama"]
    remoto = git(["config", f"branch.{rama}.remote"], raiz).stdout.strip() or "origin"
    if not git(["remote"], raiz).stdout.split():
        return "sin remoto configurado"
    ultimo = ""
    for espera in PUSH_ESPERAS:
        time.sleep(espera)
        r = correr(["push", remoto, f"{rama}:{rama}"], raiz)
        if r.returncode == 0:
            return "hecho"
        ultimo = (r.stderr or r.stdout).strip()
        if re.search(r"non-fast-forward|fetch first|\[rejected\]", ultimo):
            raise Falla(4, f"el remoto «{remoto}/{rama}» tiene commits que este clon no tiene. El commit quedó guardado "
                           "en local y no se perdió nada, pero no integro solo: con el árbol compartido en uso, un merge "
                           "puede pisar trabajo sin guardar de otra sesión. Avisa a las otras sesiones (SendMessage) y "
                           "que una sola, con todas en pausa, corra `git fetch` y `git merge --ff-only` o un merge normal.\n"
                           + ultimo)
    return "pendiente (la red o GitHub fallaron; el commit está en local y sale con la próxima corrida): " + ultimo[-200:]


def estado(ctx: dict) -> int:
    raiz: Path = ctx["raiz"]
    c = Candado(ctx["comun"])
    info = c.leer() if c.dir.exists() else None
    print(f"rama: {ctx['rama'] or '(HEAD suelto)'} | árbol principal: {'sí' if ctx['principal'] else 'no'}")
    print("candado: " + (f"tomado hace {c.edad():.0f} s («{(info or {}).get('mensaje', '')[:60]}»)" if c.dir.exists() else "libre"))
    adelante = git(["rev-list", "--count", "@{u}..HEAD"], raiz)
    print("commits sin subir: " + (adelante.stdout.strip() if adelante.returncode == 0 else "(la rama no tiene remoto)"))
    print(git(["-c", "core.quotepath=off", "status", "--short"], raiz, dict(os.environ, GIT_OPTIONAL_LOCKS="0")).stdout.rstrip()
          or "(árbol limpio)")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="Commit y push coordinados entre sesiones: solo las rutas nombradas")
    ap.add_argument("rutas", nargs="*", help="archivos o carpetas que cambiaste")
    ap.add_argument("-m", "--mensaje", default="", help="mensaje del commit, en inglés, sin atribución de IA")
    ap.add_argument("--sin-push", action="store_true", help="solo el commit")
    ap.add_argument("--esperar", type=float, default=ESPERA_S, help="segundos de espera por el candado")
    ap.add_argument("--rama", default=None, help="rama esperada, si no es main en el árbol principal")
    ap.add_argument("--estado", action="store_true", help="muestra la rama, el candado y lo que hay sin subir")
    args = ap.parse_args(argv)
    try:
        ctx = contexto(Path.cwd())
        if args.estado:
            return estado(ctx)
        rutas = validar(ctx, args.rutas, args.mensaje, args.rama)
        candado = Candado(ctx["comun"], args.mensaje)
        candado.tomar(args.esperar)
        try:
            r = subir(ctx, rutas, args.mensaje, not args.sin_push)
        finally:
            candado.soltar()
    except Falla as f:
        print(f"[subir] {f}", file=sys.stderr, flush=True)
        return f.codigo
    if r["sha"]:
        print(f"[subir] COMMIT {r['sha']} ({len(rutas)} ruta(s), {r['nuevas']} archivo(s) nuevo(s)) | PUSH {r['push']}")
    else:
        print(f"[subir] nada que subir en esas rutas: ya están en HEAD | PUSH {r['push']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
