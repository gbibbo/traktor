"""
PURPOSE: App local para organizar la colección desde el navegador, pensada para cualquier persona
         (plan docs/plans/20260929_app_local.md). Sirve la interfaz de revisión
         (tools/playlist_review/template.html) con la organización con nombre (organize.py) y ejecuta
         desde ahí lo que antes pedía la línea de comandos:
           - agregar música nueva: selector de carpetas de Windows, copia a la biblioteca si hace falta,
             análisis con progreso (catálogo, BPM, sonido, voces) y ubicación en las playlists,
             manteniendo la organización actual o reorganizando todo;
           - fusiones de temas (siempre en la misma playlist): borradores y aplicación;
           - exportar para Rekordbox y Traktor; deshacer el último cambio.
         Solo escucha en 127.0.0.1; las acciones exigen el token de la página y un Host/Origin local.
         Solo biblioteca estándar (http.server): nada nuevo que instalar.
CHANGELOG:
  - 2026-09-29: Creación inicial.
  - 2026-09-29: Reordenar temas de una playlist a mano (/api/reorder) y eliminar fusiones
                (/api/remove-fusion), cada uno como versión nueva que se puede deshacer.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.audio_utils import get_audio_files  # noqa: E402
from src.v4.common.config_loader import load_config  # noqa: E402
from src.v4.common.path_resolver import resolve_dataset_artifacts, resolve_dataset_audio_root  # noqa: E402
from src.v4.pipeline import organize  # noqa: E402
from tools.playlist_review.build_review_page import history_lines, make_data, render  # noqa: E402

AUDIO_TYPES = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac", ".aif": "audio/aiff",
               ".aiff": "audio/aiff", ".m4a": "audio/mp4"}
ADD_STEPS = ["Copiar a tu biblioteca", "Leer la colección", "BPM y tonalidad", "Analizar el sonido de cada tema",
             "Detectar voces", "Ubicar los temas en las playlists"]
_PROGRESS = [re.compile(r"(\d+)/(\d+) temas"), re.compile(r"BPM estimado (\d+)/(\d+)")]


# ---------------------------------------------------------------------------
# Tareas en segundo plano
# ---------------------------------------------------------------------------

class Job:
    """Una operación larga con pasos; la interfaz la consulta por /api/job/<id>."""

    def __init__(self, kind: str, steps: List[str]):
        self.id = secrets.token_hex(6)
        self.kind, self.steps = kind, steps
        self.step, self.frac, self.detail = 0, 0.0, ""
        self.status, self.result, self.error = "running", None, None
        self.started = time.time()
        self.step_started = time.time()

    def set(self, step: Optional[int] = None, frac: Optional[float] = None, detail: Optional[str] = None) -> None:
        if step is not None and step != self.step:
            self.step, self.frac, self.step_started = step, 0.0, time.time()
        if frac is not None:
            self.frac = max(0.0, min(1.0, frac))
        if detail is not None:
            self.detail = detail

    def public(self) -> Dict:
        eta = None
        if self.status == "running" and 0.02 < self.frac < 1:
            el = time.time() - self.step_started
            eta = int(el / self.frac * (1 - self.frac))
        return {"id": self.id, "kind": self.kind, "steps": self.steps, "step": self.step, "frac": round(self.frac, 3),
                "detail": self.detail, "status": self.status, "result": self.result, "error": self.error,
                "eta_s": eta}


def run_script(args: List[str], job: Job, step: int) -> None:
    """Corre un script del pipeline y traduce sus líneas 'n/N temas' a progreso del paso."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    proc = subprocess.Popen([sys.executable, *args], cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", env=env)
    tail: List[str] = []
    for line in proc.stdout:
        tail = (tail + [line.rstrip()])[-30:]
        for rx in _PROGRESS:
            m = rx.search(line)
            if m and int(m.group(2)) > 0:
                job.set(step, int(m.group(1)) / int(m.group(2)), f"{m.group(1)} de {m.group(2)} temas")
    if proc.wait() != 0:
        raise RuntimeError("Falló " + Path(args[0]).name + ":\n" + "\n".join(tail[-12:]))


def copy_tree(src: Path, dst: Path, job: Job, step: int) -> None:
    files = [p for p in src.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files) or 1
    done = 0
    for p in files:
        target = dst / p.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        done += p.stat().st_size
        job.set(step, done / total, f"{done / 2**20:.0f} de {total / 2**20:.0f} MB")


# ---------------------------------------------------------------------------
# Estado de la app
# ---------------------------------------------------------------------------

class App:
    def __init__(self, dataset: str, org_name: str, config_path: Optional[str] = None):
        self.config = load_config(Path(config_path) if config_path else None)
        self.dataset, self.org_name = dataset, org_name
        self.artifacts = resolve_dataset_artifacts(dataset, self.config)
        self.library = resolve_dataset_audio_root(dataset, self.config).resolve()
        self.store = organize.OrgStore(self.artifacts, org_name)
        if not self.store.exists():
            raise SystemExit(f"No existe la organización '{org_name}' en {self.artifacts / 'orgs'}")
        self.token = secrets.token_urlsafe(24)
        self.jobs: Dict[str, Job] = {}
        self.lock = threading.Lock()
        self.port = 0

    # ---- página ----
    def page(self) -> str:
        data = make_data(self.artifacts, self.dataset, [self.org_name], [])
        data["app"] = {"token": self.token, "org": self.org_name, "library": str(self.library),
                       "drafts": self.store.drafts(), "can_undo": self._can_undo()}
        return render(data)

    def _can_undo(self) -> bool:
        meta = self.store.meta()
        return any(h["version"] < meta["current_version"] for h in meta["history"])

    def running(self) -> Optional[Job]:
        return next((j for j in self.jobs.values() if j.status == "running"), None)

    def start(self, kind: str, steps: List[str], fn: Callable[[Job], Dict]) -> Job:
        with self.lock:
            if self.running():
                raise RuntimeError("Ya hay una tarea en curso; esperá a que termine.")
            job = Job(kind, steps)
            self.jobs[job.id] = job

        def target():
            try:
                job.result = fn(job)
                job.status = "done"
            except Exception as exc:  # noqa: BLE001
                job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
        threading.Thread(target=target, daemon=True).start()
        return job

    # ---- carpetas ----
    def pick_folder(self) -> Dict:
        code = ("import sys, tkinter as tk; from tkinter import filedialog; r = tk.Tk(); r.withdraw(); "
                "r.attributes('-topmost', True); "
                "p = filedialog.askdirectory(initialdir=sys.argv[1], title=sys.argv[2], mustexist=True); print(p or '')")
        out = subprocess.run([sys.executable, "-c", code, str(self.library), "Elegí la carpeta con la música nueva"],
                             capture_output=True, text=True, encoding="utf-8").stdout.strip()
        if not out:
            return {"cancelled": True}
        return self.describe_folder(out)

    def describe_folder(self, path: str) -> Dict:
        folder = Path(path).resolve()
        if not folder.is_dir():
            raise ValueError(f"No existe la carpeta: {folder}")
        files = get_audio_files(folder, recursive=True)
        inside = folder == self.library or self.library in folder.parents
        known = 0
        if inside and folder != self.library:  # temas que ya están en esta organización
            import pandas as pd
            cat = pd.read_parquet(self.artifacts / "catalog.parquet", columns=["track_uid", "rel_path"])
            in_org = set(self.store.load()[0]["track_uid"])
            organized = set(cat.loc[cat["track_uid"].isin(in_org), "rel_path"])
            known = sum((p.relative_to(self.library).as_posix() in organized) for p in files)
        size = sum(p.stat().st_size for p in files)
        return {"path": str(folder), "name": folder.name, "inside": inside, "is_library": folder == self.library,
                "n_audio": len(files), "n_known": int(known), "size_mb": round(size / 2**20),
                "dest": str(folder) if inside else str(self._copy_destination(folder))}

    def _copy_destination(self, folder: Path) -> Path:
        dest, k = self.library / folder.name, 2
        while dest.exists():
            dest, k = self.library / f"{folder.name} ({k})", k + 1
        return dest

    # ---- operaciones ----
    def add_collection(self, path: str, keep: bool) -> Job:
        info = self.describe_folder(path)
        if info["is_library"]:
            raise ValueError("Elegí una subcarpeta con la música nueva, no la biblioteca entera.")
        if info["n_audio"] == 0:
            raise ValueError("La carpeta no tiene archivos de audio.")

        steps = ADD_STEPS if not info["inside"] else ADD_STEPS[1:]  # sin copia si ya está en la biblioteca
        S = {name: k for k, name in enumerate(steps)}

        def fn(job: Job) -> Dict:
            folder = Path(info["path"])
            if not info["inside"]:
                dest = Path(info["dest"])
                copy_tree(folder, dest, job, S["Copiar a tu biblioteca"])
                folder = dest
            scope = folder.relative_to(self.library).as_posix()
            pipe = REPO_ROOT / "src" / "v4" / "pipeline"
            ds = ["--dataset-name", self.dataset]
            step = S["Leer la colección"]
            job.set(step, 0, "Leyendo tags y buscando duplicados…")
            run_script([str(pipe / "phase0_ingest.py"), *ds, "--scope", scope], job, step)
            step = S["BPM y tonalidad"]
            job.set(step, 0, "")
            run_script([str(pipe / "phase1_tags.py"), *ds, "--estimate-missing"], job, step)
            step = S["Analizar el sonido de cada tema"]
            job.set(step, 0, "")
            backend = self.store.meta()["params"]["rep"].split("_")[0]
            run_script([str(pipe / "extract_representations.py"), *ds, "--models", backend, "--folder", scope,
                        "--min-duration", "90", "--max-duration", "900"], job, step)
            step = S["Detectar voces"]
            job.set(step, 0, "")
            if backend == "clap":
                run_script([str(pipe / "tag_vocals.py"), *ds, "--method", "clap", "--write-tags"], job, step)
            step = S["Ubicar los temas en las playlists"]
            job.set(step, 0, "Esto puede tardar un minuto…")
            lib = organize.Library(self.artifacts, self.store.meta()["params"]["rep"])
            if keep:
                v, counts = organize.add(self.store, lib, scope)
            else:
                meta = self.store.meta()
                scopes = organize.org_scopes(meta)
                if not any(x == "" or (scope + "/").startswith(x.rstrip("/") + "/") for x in scopes):
                    scopes.append(scope)
                v = organize.build(self.store, lib, meta["params"], scopes, action="build")
                counts = {"added": None}
            job.set(step, 1, "")
            return {"version": v, "counts": counts, "scope": scope, "keep": keep}
        return self.start("add", steps, fn)

    def save_drafts(self, drafts: List[Dict]) -> None:
        clean = [{"name": str(d.get("name") or f"Fusión {i + 1}")[:60], "color": int(d.get("color", i)) % organize.FUSION_COLORS,
                  "tracks": [str(t) for t in d.get("tracks", [])]} for i, d in enumerate(drafts)]
        self.store.save_drafts(clean)

    def apply_fusions(self, keep: bool) -> Job:
        all_drafts = self.store.drafts()
        drafts = [d for d in all_drafts if len(d["tracks"]) >= 2]
        waiting = [d for d in all_drafts if len(d["tracks"]) < 2]  # siguen en armado (falta otro tema)
        if not drafts:
            raise ValueError("No hay fusiones pendientes con al menos dos temas.")

        def fn(job: Job) -> Dict:
            job.set(0, 0, "Reacomodando…")
            lib = organize.Library(self.artifacts, self.store.meta()["params"]["rep"])
            v, info = organize.link_groups(self.store, lib, drafts, rebuild=not keep)
            self.store.save_drafts(waiting)
            moved = sum(len(r["moved"]) for r in info["groups"]) if info["mode"] == "frozen" else None
            job.set(0, 1, "")
            return {"version": v, "keep": keep, "moved": moved, "n_fusions": len(drafts)}
        return self.start("fusions", ["Rehacer las playlists con las fusiones"], fn)

    def reorder(self, l1: int, l2: int, tracks: List[str]) -> Dict:
        """Orden a mano de una playlist (rápido: sin tarea en segundo plano)."""
        with self.lock:
            if self.running():
                raise RuntimeError("Hay una tarea en curso; esperá a que termine.")
            v = organize.reorder(self.store, l1, l2, [str(t) for t in tracks])
        meta = self.store.meta()
        return {"version": v, "history": history_lines(meta), "can_undo": self._can_undo()}

    def remove_fusion(self, name: str) -> Dict:
        """Quita la fusión: la aplicada, con una versión nueva (los temas no se mueven), y sus borradores."""
        with self.lock:
            if self.running():
                raise RuntimeError("Hay una tarea en curso; esperá a que termine.")
            applied = any(f["name"] == name for f in self.store.fusions())
            v = organize.remove_fusion(self.store, name) if applied else None
            drafts = self.store.drafts()
            if not applied and not any(d["name"] == name for d in drafts):
                raise ValueError(f"No hay una fusión llamada «{name}»")
            self.store.save_drafts([d for d in drafts if d["name"] != name])
        return {"version": v, "applied": applied}

    def export(self) -> Job:
        def fn(job: Job) -> Dict:
            from src.v4.pipeline.phase5_export import run_export
            job.set(0, 0.1, "Escribiendo las playlists…")
            out = run_export(self.dataset, self.config, formats=("m3u8", "rekordbox", "traktor"),
                             out_root=self.artifacts / "exports", org_name=self.org_name)
            job.set(0, 1, "")
            return {"folder": str(Path(out).resolve()), "rekordbox": str((Path(out) / "rekordbox.xml").resolve()),
                    "traktor": str((Path(out) / "traktor.nml").resolve())}
        return self.start("export", ["Preparar las playlists para Rekordbox y Traktor"], fn)

    def open_folder(self, path: str) -> None:
        target = Path(path).resolve()
        if not (self.artifacts.resolve() in target.parents or self.library in target.parents or target == self.library):
            raise ValueError("Solo se abren carpetas de la biblioteca o de los exports.")
        if hasattr(os, "startfile"):
            os.startfile(str(target))  # noqa: S606 (Explorador de Windows)
        else:
            subprocess.Popen(["xdg-open", str(target)])


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "TraktorML/1"

        def log_message(self, fmt, *args):  # silencioso salvo errores
            if args and str(args[1]).startswith(("4", "5")):
                sys.stderr.write("[HTTP] " + fmt % args + "\n")

        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").lower()
            return host in (f"127.0.0.1:{app.port}", f"localhost:{app.port}")

        def _json(self, code: int, obj) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if not self._host_ok():
                return self._json(403, {"error": "host"})
            path = urllib.parse.urlparse(self.path).path
            if path in ("/", "/index.html"):
                body = app.page().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return self.wfile.write(body)
            if path.startswith("/audio/"):
                return self._audio(urllib.parse.unquote(path[len("/audio/"):]))
            if path.startswith("/api/job/"):
                job = app.jobs.get(path.rsplit("/", 1)[-1])
                return self._json(200, job.public()) if job else self._json(404, {"error": "no existe"})
            return self._json(404, {"error": "no existe"})

        def _audio(self, rel: str):
            target = (app.library / rel).resolve()
            if app.library not in target.parents or not target.is_file():
                return self._json(404, {"error": "no existe"})
            size = target.stat().st_size
            ctype = AUDIO_TYPES.get(target.suffix.lower()) or mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            start, end = 0, size - 1
            rng = self.headers.get("Range")
            m = re.match(r"bytes=(\d*)-(\d*)$", rng or "")
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:  # sufijo: los últimos N bytes
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                if start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            with open(target, "rb") as f:
                f.seek(start)
                remaining = end - start + 1
                try:
                    while remaining > 0:
                        chunk = f.read(min(1 << 16, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                    pass  # el navegador cortó (saltó a otro punto del tema)

        def do_POST(self):  # noqa: N802
            origin = self.headers.get("Origin")
            if (not self._host_ok() or self.headers.get("X-App-Token") != app.token
                    or (origin and origin not in (f"http://127.0.0.1:{app.port}", f"http://localhost:{app.port}"))):
                return self._json(403, {"error": "No autorizado"})
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": "JSON inválido"})
            path = urllib.parse.urlparse(self.path).path
            try:
                if path == "/api/pick-folder":
                    return self._json(200, app.pick_folder())
                if path == "/api/describe-folder":
                    return self._json(200, app.describe_folder(body["path"]))
                if path == "/api/add-collection":
                    return self._json(200, app.add_collection(body["path"], bool(body.get("keep", True))).public())
                if path == "/api/drafts":
                    app.save_drafts(body.get("fusions", []))
                    return self._json(200, {"ok": True})
                if path == "/api/apply-fusions":
                    return self._json(200, app.apply_fusions(bool(body.get("keep", True))).public())
                if path == "/api/reorder":
                    return self._json(200, app.reorder(int(body["l1"]), int(body["l2"]), list(body["tracks"])))
                if path == "/api/remove-fusion":
                    return self._json(200, app.remove_fusion(str(body["name"])))
                if path == "/api/export":
                    return self._json(200, app.export().public())
                if path == "/api/undo":
                    if app.running():
                        raise RuntimeError("Hay una tarea en curso.")
                    v = app.store.undo()
                    return self._json(200, {"version": v})
                if path == "/api/open-folder":
                    app.open_folder(body["path"])
                    return self._json(200, {"ok": True})
            except (ValueError, RuntimeError, KeyError) as exc:
                return self._json(400, {"error": str(exc)})
            return self._json(404, {"error": "no existe"})
    return Handler


def free_port(preferred: int) -> int:
    for port in [preferred, *range(preferred + 1, preferred + 50), 0]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("No hay puertos libres")


def serve(app: App, port: int = 8765, open_browser: bool = True) -> ThreadingHTTPServer:
    app.port = free_port(port)
    httpd = ThreadingHTTPServer(("127.0.0.1", app.port), make_handler(app))
    url = f"http://127.0.0.1:{app.port}/"
    print(f"[INFO] TRAKTOR ML abierto en {url}  (cerrá esta ventana para salir)", flush=True)
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    return httpd


def main() -> int:
    parser = argparse.ArgumentParser(description="App local: revisar y organizar la colección desde el navegador")
    parser.add_argument("--dataset-name", default="musica")
    parser.add_argument("--org-name", default="biblioteca")
    parser.add_argument("--config", default=None)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    app = App(args.dataset_name, args.org_name, args.config)
    httpd = serve(app, args.port, not args.no_browser)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
