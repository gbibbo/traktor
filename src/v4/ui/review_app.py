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
  - 2026-09-29: Agregar música nueva busca temas repetidos (dedupe.py): lo que la regla resuelve queda
                con una sola copia y las otras se mueven a Música/_copias.
  - 2026-09-29: /cover/<tema>: carátula del archivo (o de su carpeta) para la columna de la tabla.
  - 2026-11-02: Pestaña «Organización» de la app de descargas (download_JIJIJI/app/descargas_app.py), que
                la lanza y la muestra dentro de su ventana: /api/ping (¿hay una tarea en curso?) y
                --exit-on-stdin-eof (se cierra cuando se cierra la app, al terminar la tarea en curso).
  - 2026-11-04: «Agregar música nueva» cuenta como ya organizadas las copias idénticas (duplicates.csv) y
                las descartadas por dedupe: en la biblioteca real, «2020 new - copia» decía 246 temas nuevos
                y no había ninguno.
  - 2026-11-04: La etiqueta Vocal de la música agregada sale de clap_probe (la regla vigente), no de 'clap':
                el método viejo le habría quitado la marca a 473 temas de la biblioteca y puesto a 94.
  - 2026-11-04: Sin la carpeta Música/ (un ordenador con solo los datos de la organización), la música sale
                del paquete de usb_package.py (<unidad>:\\TRAKTOR ML: primero la unidad de los datos de la
                organización, después discos fijos y extraíbles; o --package):
                cada tema se busca por track_uid en su _organizacion.csv. Así la organización se ve y se
                escucha en solo lectura: los cambios se hacen donde está la biblioteca, para no tener dos
                organizaciones que se separan.
  - 2026-11-04: Con la biblioteca en el pendrive (package_library.py: Música/ es un junction a <paquete>/Musica),
                la app vuelve a permitir cambios, y «Preparar para Rekordbox y Traktor» y «Agregar música nueva»
                además actualizan el pendrive: sus playlists (usb_package.refresh_from_org, sin mover audio) y el
                lugar de cada tema guardado en su archivo (file_embeddings.py).
  - 2026-11-07: Los procesos que lanza la app arrancan con stdin=DEVNULL, sin heredar el pipe que lee
                exit_with_parent. «Agregar música nueva» de Gabriel quedó trabado en «Leer la colección»: el
                hijo seguía sin arrancar (13 MB, 0 s de CPU) a los 30 min. Que el pipe heredado sea la causa es
                una hipótesis: en una prueba aislada traba al hijo, con la cadena exacta de la app no.
  - 2026-11-07: «Agregar música nueva» completa lo que leen las playlists y las sugerencias en vivo: BPM y
                tonalidad también de la colección de Traktor (phase1_tags --traktor) y el paso «Género y
                subgénero» (MAEST, beatport_lookup de la carpeta sumado a features/beatport,
                beatport_subgenres, genre_model --predict; lo que falle se avisa sin cortar). Para una carpeta
                ya agregada, «Completar BPM, tonalidad y género» (/api/complete-metadata) sin tocar las
                playlists (Gabriel: los temas descargados no tenían la metadata de la colección).
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
from pathlib import Path, PureWindowsPath
from typing import Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.audio_utils import get_audio_files  # noqa: E402
from src.v4.common.config_loader import load_config  # noqa: E402
from src.v4.common.path_resolver import resolve_dataset_artifacts, resolve_dataset_audio_root  # noqa: E402
from src.v4.common.tags import read_cover  # noqa: E402
from src.v4.pipeline import organize  # noqa: E402
# Importados al arrancar, no a mitad de una tarea: el 2026-11-07 la app se congeló justo al importar dedupe
# mientras otro hilo leía la entrada estándar (lo que sigue hace --parent-pid, sin leerla)
from src.v4.pipeline import dedupe, file_embeddings, phase5_export  # noqa: E402,F401
from src.v4.pipeline.usb_package import PACKAGE_NAME  # noqa: E402
from tools.playlist_review.build_review_page import history_lines, make_data, render  # noqa: E402

AUDIO_TYPES = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac", ".aif": "audio/aiff",
               ".aiff": "audio/aiff", ".m4a": "audio/mp4"}
ADD_STEPS = ["Copiar a tu biblioteca", "Leer los temas", "BPM y tonalidad (de Traktor)", "Embeddings del sonido",
             "Voces", "Género y subgénero", "Temas repetidos", "Ubicar en las playlists"]
COMPLETE_STEPS = ["BPM y tonalidad (de Traktor)", "Género y subgénero"]
_PROGRESS = [re.compile(r"(\d+)/(\d+) temas"), re.compile(r"BPM estimado (\d+)/(\d+)"),
             re.compile(r"\[beatport\] (\d+)/(\d+)")]
BEATPORT_PART = "beatport_agregado"     # búsqueda de la carpeta agregada, antes de sumarla a features/beatport


# ---------------------------------------------------------------------------
# Tareas en segundo plano
# ---------------------------------------------------------------------------

STALL_S = 600.0     # sin escribir nada y sin usar CPU durante 10 min: el script está trabado
STALL_RETRIES = 1   # trabado: se lanza una vez más (los pasos se pueden repetir: lo hecho se reutiliza)


class Cancelled(RuntimeError):
    """Gabriel tocó «Cancelar»."""


class Job:
    """Una operación larga con pasos; la interfaz la consulta por /api/job/<id>."""

    def __init__(self, kind: str, steps: List[str]):
        self.id = secrets.token_hex(6)
        self.kind, self.steps = kind, steps
        self.step, self.frac, self.detail = 0, 0.0, ""
        self.status, self.result, self.error = "running", None, None
        self.started = time.time()
        self.step_started = time.time()
        self.proc: Optional[subprocess.Popen] = None   # el script que corre ahora (para cancelar)
        self.cancel_requested = False

    def check(self) -> None:
        """Entre pasos: si se pidió cancelar, se corta acá (lo hecho queda y se reutiliza la próxima vez)."""
        if self.cancel_requested:
            raise Cancelled("cancelado")

    def cancel(self) -> None:
        self.cancel_requested = True
        if self.proc is not None:
            _kill_tree(self.proc)

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


def _cpu_seconds(pid: int) -> Optional[float]:
    """CPU usada por el proceso y sus hijos (psutil); None si no se puede saber."""
    try:
        import psutil
        p = psutil.Process(pid)
        return sum(sum(q.cpu_times()[:2]) for q in [p, *p.children(recursive=True)])
    except Exception:  # noqa: BLE001
        return None


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        import psutil
        p = psutil.Process(proc.pid)
        for q in p.children(recursive=True):
            q.kill()
        p.kill()
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except OSError:
            pass


def run_script(args: List[str], job: Job, step: int, stall_s: Optional[float] = None) -> None:
    """Corre un script del pipeline y traduce sus líneas 'n/N temas' a progreso del paso. Si se traba (ni escribe
    ni usa CPU durante stall_s) se corta y se lanza una vez más; «Cancelar» lo corta enseguida."""
    stall_s = STALL_S if stall_s is None else stall_s
    name = Path(args[0]).name
    for attempt in range(STALL_RETRIES + 1):
        if _run_once(args, job, step, stall_s):
            return
        if attempt < STALL_RETRIES:
            print(f"[agregar] {name} no avanzó en {stall_s / 60:.0f} min: lo vuelvo a lanzar", flush=True)
            job.set(step, 0, "Se había trabado: lo vuelvo a intentar…")
    raise RuntimeError(f"{name} se trabó {STALL_RETRIES + 1} veces (sin avanzar durante {stall_s / 60:.0f} min). "
                       "Lo hecho hasta acá queda guardado: volvé a agregar la carpeta y sigue desde ahí.")


def _run_once(args: List[str], job: Job, step: int, stall_s: float) -> bool:
    """True si terminó; False si se trabó (y ya se cortó)."""
    import queue
    job.check()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    # stdin=DEVNULL: lanzada desde la app de descargas, nuestra entrada es un pipe que exit_with_parent está leyendo.
    # En una prueba aislada, un hijo que heredaba un pipe así se trababa al arrancar (HIPÓTESIS de la traba del
    # 2026-11-07, «Leer la colección» 30 min sin avanzar; con la cadena exacta de la app no se reprodujo)
    proc = subprocess.Popen([sys.executable, *args], cwd=REPO_ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env)
    job.proc = proc
    lines: "queue.Queue[Optional[str]]" = queue.Queue()

    def pump() -> None:
        for raw in proc.stdout:
            lines.put(raw)
        lines.put(None)
    threading.Thread(target=pump, daemon=True).start()
    tail: List[str] = []
    active, cpu = time.monotonic(), _cpu_seconds(proc.pid)
    try:
        while True:
            if job.cancel_requested:
                _kill_tree(proc)
                proc.wait()
                raise Cancelled("cancelado")
            try:
                line = lines.get(timeout=2.0)
            except queue.Empty:
                now_cpu = _cpu_seconds(proc.pid)
                if now_cpu is not None and (cpu is None or now_cpu > cpu + 0.5):
                    cpu, active = now_cpu, time.monotonic()   # trabaja aunque no escriba
                if time.monotonic() - active > stall_s:
                    _kill_tree(proc)
                    proc.wait()
                    return False
                continue
            if line is None:
                break
            active = time.monotonic()
            tail = (tail + [line.rstrip()])[-30:]
            for rx in _PROGRESS:
                m = rx.search(line)
                if m and int(m.group(2)) > 0:
                    job.set(step, int(m.group(1)) / int(m.group(2)), f"{m.group(1)} de {m.group(2)} temas")
    finally:
        job.proc = None
    code = proc.wait()
    if job.cancel_requested:  # «Cancelar» lo cortó (el proceso muerto no es un error del script)
        raise Cancelled("cancelado")
    if code != 0:
        raise RuntimeError("Falló " + Path(args[0]).name + ":\n" + "\n".join(tail[-12:]))
    return True


def package_index(pkg: Path) -> Dict[str, tuple]:
    """track_uid -> (archivo, carpeta, playlist) del _organizacion.csv del pendrive (lo que va en el lugar)."""
    import pandas as pd
    index = Path(pkg) / "_organizacion.csv"
    if not index.is_file():
        return {}
    df = pd.read_csv(index, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    cols = [c for c in ("archivo", "carpeta", "playlist") if c in df.columns]
    return {r[0]: tuple(r[1:]) for r in df[["track_uid", *cols]].itertuples(index=False)}


def changed_places(store, since_version: int) -> set:
    """Temas nuevos o que cambiaron de playlist, posición o lugar en el mapa desde since_version."""
    cols = ["l1", "l2", "position", "x", "y"]
    cur = store.load()[0].drop_duplicates("track_uid").set_index("track_uid")[cols]
    try:
        old = store.load(since_version)[0].drop_duplicates("track_uid").set_index("track_uid")[cols]
    except OSError:
        return set(cur.index)
    common = cur.index.intersection(old.index)
    a, b = cur.loc[common], old.loc[common]
    moved = (a[["l1", "l2", "position"]] != b[["l1", "l2", "position"]]).any(axis=1) | \
        ((a[["x", "y"]] - b[["x", "y"]]).abs() > 1e-6).any(axis=1)
    return set(cur.index.difference(old.index)) | set(common[moved.to_numpy()])


def files_under(folder: Path) -> List[Path]:
    return [p for p in folder.rglob("*") if p.is_file()]


def is_copy_of(dest: Path, src: Path) -> bool:
    """dest es una copia (entera o cortada a medias) de src: cada archivo suyo está en src con el mismo nombre y
    a lo sumo el mismo tamaño. Así, repetir un agregado que se cortó completa esa copia en vez de hacer otra."""
    for p in files_under(dest):
        q = src / p.relative_to(dest)
        if not q.is_file() or p.stat().st_size > q.stat().st_size:
            return False
    return True


def copy_tree(src: Path, dst: Path, job: Job, step: int, skip: Optional[set] = None) -> None:
    """Copia lo que falte. No se copia de nuevo un archivo ya copiado (mismo tamaño) ni uno de `skip` (rutas
    relativas de temas que ya están en la biblioteca)."""
    skip = skip or set()
    files = [p for p in files_under(src) if p.relative_to(src).as_posix() not in skip]
    total = sum(p.stat().st_size for p in files) or 1
    done = 0
    for p in files:
        job.check()
        target = dst / p.relative_to(src)
        st = p.stat()
        if not (target.is_file() and target.stat().st_size == st.st_size):
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".copiando")
            shutil.copy2(p, partial)
            os.replace(partial, target)   # un corte a mitad no deja un archivo de audio truncado
        done += st.st_size
        job.set(step, done / total, f"{done / 2**20:.0f} de {total / 2**20:.0f} MB")


# ---------------------------------------------------------------------------
# Música del paquete (usb_package.py): la organización en un ordenador sin la carpeta Música/
# ---------------------------------------------------------------------------

PACKAGE_INDEX = "_organizacion.csv"  # usb_package.py: carpeta, playlist, track_uid y archivo de cada tema
READONLY = ("En este ordenador la organización es para ver y escuchar, con la música del paquete. Agregar música, "
            "fusionar, ordenar y preparar para Rekordbox/Traktor se hace en el ordenador que tiene la biblioteca "
            "(Música/), para no tener dos organizaciones distintas.")


def drive_order(kinds: Dict[str, int], prefer: Optional[str] = None) -> List[str]:
    """Unidades donde buscar el paquete: primero `prefer` (la de los datos de la organización: lo que va junto,
    junto), después los discos fijos y después los extraíbles; nunca unidades de red ni lectoras."""
    fixed = [d for d in sorted(kinds) if kinds[d] == 3]      # DRIVE_FIXED
    removable = [d for d in sorted(kinds) if kinds[d] == 2]  # DRIVE_REMOVABLE
    first = [prefer] if prefer in fixed + removable else []
    return first + [d for d in fixed + removable if d != prefer]


def find_package(prefer_near: Optional[Path] = None, name: str = PACKAGE_NAME) -> Optional[Path]:
    """Un paquete en la raíz de una unidad (<unidad>:\\TRAKTOR ML\\_organizacion.csv). `prefer_near`: una ruta
    (los datos de la organización, aunque se llegue por un junction) cuya unidad se mira primero."""
    if os.name != "nt":
        return None
    import ctypes
    k32 = ctypes.windll.kernel32
    mask = k32.GetLogicalDrives()
    drives = [f"{chr(65 + i)}:\\" for i in range(26) if mask >> i & 1]
    kinds = {d: k32.GetDriveTypeW(ctypes.c_wchar_p(d)) for d in drives}
    prefer = None
    if prefer_near is not None:
        try:
            prefer = Path(prefer_near).resolve().anchor.upper()
        except OSError:
            pass
    for d in drive_order(kinds, prefer):
        if (Path(d) / name / PACKAGE_INDEX).is_file():
            return Path(d) / name
    return None


class PackageAudio:
    """Los temas de la organización en la copia del paquete. Ahí tienen otros nombres y carpetas (una por playlist):
    se encuentran por track_uid (el hash del audio, que los tags no cambian)."""

    def __init__(self, root: Path, artifacts: Path, org_uids: List[str]):
        import pandas as pd
        self.root = Path(root).resolve()
        index = pd.read_csv(self.root / PACKAGE_INDEX, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        self.by_uid: Dict[str, Path] = {uid: self.root / PureWindowsPath(rel).as_posix()
                                        for uid, rel in zip(index["track_uid"], index["archivo"])
                                        if rel and ".." not in PureWindowsPath(rel).parts}
        cat = pd.read_parquet(artifacts / "catalog.parquet", columns=["track_uid", "rel_path"])
        self.by_rel = {rel: self.by_uid[uid] for uid, rel in zip(cat["track_uid"], cat["rel_path"]) if uid in self.by_uid}
        self.total = len(org_uids)
        self.found = sum(1 for u in org_uids if u in self.by_uid and self.by_uid[u].is_file())

    def file(self, rel: str) -> Optional[Path]:
        path = self.by_rel.get(rel)
        return path if path is not None and path.is_file() else None

    def public(self) -> Dict:
        return {"root": str(self.root), "found": self.found, "total": self.total}


# ---------------------------------------------------------------------------
# Estado de la app
# ---------------------------------------------------------------------------

class App:
    package: Optional[PackageAudio] = None  # sin Música/: la música del paquete, y solo lectura
    readonly = False

    def __init__(self, dataset: str, org_name: str, config_path: Optional[str] = None, package: Optional[str] = None):
        self.config = load_config(Path(config_path) if config_path else None)
        self.dataset, self.org_name = dataset, org_name
        self.artifacts = resolve_dataset_artifacts(dataset, self.config)
        self.store = organize.OrgStore(self.artifacts, org_name)
        if not self.store.exists():
            raise SystemExit(f"No existe la organización '{org_name}' en {self.artifacts / 'orgs'}")
        try:
            self.library = resolve_dataset_audio_root(dataset, self.config).resolve()
        except FileNotFoundError:
            root = Path(package) if package else find_package(self.artifacts)
            if root is None or not (root / PACKAGE_INDEX).is_file():
                raise
            self.library = None
            self.package = PackageAudio(root, self.artifacts, list(self.store.load()[0]["track_uid"]))
            self.readonly = True
            print(f"[INFO] Sin la carpeta de música: los temas salen del paquete {self.package.root} "
                  f"({self.package.found} de {self.package.total}); solo lectura.", flush=True)
        self.token = secrets.token_urlsafe(24)
        self.jobs: Dict[str, Job] = {}
        self.lock = threading.Lock()
        self.port = 0

    def package_dir(self) -> Optional[Path]:
        """El paquete del pendrive cuando la biblioteca es su carpeta Musica (package_library.py): entonces cada
        cambio rehace sus playlists y guarda en cada archivo su lugar en la organización."""
        from src.v4.pipeline.package_library import is_package_library
        return self.library.parent if self.library is not None and is_package_library(self.library) else None

    def _update_package(self, job: "Job", step: int, since_version: Optional[int] = None) -> Dict:
        """Playlists del pendrive y, en cada archivo, su embedding y su lugar. Con since_version, solo se escriben
        los temas nuevos o que cambiaron de lugar desde esa versión: el resto ya tiene lo mismo."""
        from src.v4.pipeline import file_embeddings, usb_package
        pkg = self.package_dir()
        before = package_index(pkg)
        job.set(step, 0.05, "Rehaciendo las playlists del pendrive…")
        if usb_package.refresh_from_org(pkg, self.dataset, self.org_name, self.config) != 0:
            raise RuntimeError("No se pudieron rehacer las playlists del pendrive: el detalle está en la consola.")
        job.check()
        only = None
        if since_version is not None:
            after = package_index(pkg)
            only = changed_places(self.store, since_version) | {u for u, v in after.items() if before.get(u) != v}
        job.set(step, 0.5, "Guardando en cada tema su lugar en la organización…")
        plan, rep = file_embeddings.package_plan(pkg, self.artifacts, self.org_name, only_uids=only)
        result = file_embeddings.write_package(plan, rep)
        if result["counts"]["errores"]:
            raise RuntimeError(f"{result['counts']['errores']} temas no se pudieron actualizar, p. ej. "
                               f"{result['problems'][0]['archivo']}: {result['problems'][0]['motivo']}")
        job.set(step, 1, "")
        return {"package": str(pkg), "updated": result["counts"]["escritos"]}

    # ---- página ----
    def page(self) -> str:
        data = make_data(self.artifacts, self.dataset, [self.org_name], [],
                         paths=self.package.by_uid if self.package else None)
        data["app"] = {"token": self.token, "org": self.org_name,
                       "library": str(self.package.root if self.package else self.library),
                       "drafts": self.store.drafts(), "can_undo": self._can_undo(),
                       "readonly": READONLY if self.readonly else None,
                       "package": self.package.public() if self.package else None}
        return render(data)

    def audio_file(self, rel: str) -> Optional[Path]:
        """El archivo de un tema por su ruta en la biblioteca (rel_path): en Música/ o, sin ella, en el paquete."""
        if self.package is not None:
            return self.package.file(rel)
        target = (self.library / rel).resolve()
        return target if self.library in target.parents and target.is_file() else None

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
            except Cancelled:
                job.status = "cancelled"
            except Exception as exc:  # noqa: BLE001
                job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
        threading.Thread(target=target, daemon=True).start()
        return job

    def cancel(self) -> Dict:
        job = self.running()
        if job is None:
            return {"cancelled": False}
        job.cancel()
        return {"cancelled": True, "id": job.id}

    # ---- carpetas ----
    def pick_folder(self) -> Dict:
        code = ("import sys, tkinter as tk; from tkinter import filedialog; r = tk.Tk(); r.withdraw(); "
                "r.attributes('-topmost', True); "
                "p = filedialog.askdirectory(initialdir=sys.argv[1], title=sys.argv[2], mustexist=True); print(p or '')")
        out = subprocess.run([sys.executable, "-c", code, str(self.library), "Elegí la carpeta con la música nueva"],
                             stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8").stdout.strip()
        if not out:
            return {"cancelled": True}
        return self.describe_folder(out)

    def describe_folder(self, path: str) -> Dict:
        folder = Path(path).resolve()
        if not folder.is_dir():
            raise ValueError(f"No existe la carpeta: {folder}")
        files = get_audio_files(folder, recursive=True)
        inside = folder == self.library or self.library in folder.parents
        known, homes, present = 0, [], []
        if folder != self.library:
            library = self._library_files()
            if inside:  # temas que ya están en esta organización
                rels = set(library.loc[library["organized"], "rel_path"])
                known = sum((p.relative_to(self.library).as_posix() in rels) for p in files)
            else:  # fuera de la biblioteca: ¿ya se copiaron antes? (mismo nombre y tamaño, o el mismo audio)
                for p in files:
                    hit = self._library_copy(p, library)
                    if hit is not None:
                        known += bool(hit["organized"])
                        homes.append(Path(str(hit["rel_path"])).parent.as_posix())
                        present.append(p.relative_to(folder).as_posix())
        size = sum(p.stat().st_size for p in files)
        info = {"path": str(folder), "name": folder.name, "inside": inside, "is_library": folder == self.library,
                "n_audio": len(files), "n_known": int(known), "size_mb": round(size / 2**20),
                "n_traktor": self._traktor_analyzed(files), "dest": str(folder), "dest_kind": "inside",
                "complete_path": str(folder) if inside else None, "skip": present}
        if not inside and folder != self.library:
            home = max(set(homes), key=homes.count) if homes else None
            if home is not None:  # lo nuevo va junto a lo que ya está; lo ya copiado no se copia de nuevo
                info.update(dest=str(self.library / home), dest_kind="home", complete_path=str(self.library / home))
            else:
                dest = self._copy_destination(folder)
                info.update(dest=str(dest), dest_kind="resume" if dest.exists() else "new")
        return info

    def _library_files(self):
        """Los temas de la biblioteca (catálogo, más las copias idénticas que el catálogo no repite): rel_path,
        nombre, tamaño y si ya están organizados (en la organización, o descartados por dedupe porque su gemela
        está)."""
        import pandas as pd
        from src.v4.common.duplicates import dropped
        cat = pd.read_parquet(self.artifacts / "catalog.parquet")
        cat = cat.reindex(columns=["track_uid", "rel_path", "filesize_bytes"])  # un catálogo viejo puede no tener tamaño
        same_audio = self.artifacts / "duplicates.csv"
        if same_audio.is_file():
            cat = pd.concat([cat, pd.read_csv(same_audio, usecols=["track_uid", "rel_path"])])
        in_org = set(self.store.load()[0]["track_uid"]) | set(dropped(self.artifacts))
        cat = cat.copy()
        cat["organized"] = cat["track_uid"].isin(in_org)
        cat["name"] = cat["rel_path"].map(lambda r: Path(str(r)).name.casefold())
        return cat

    def _library_copy(self, path: Path, library) -> Optional[Dict]:
        """La fila de la biblioteca que es este mismo tema, o None. Mismo nombre y tamaño; si el tamaño cambió
        (Traktor escribe sus datos en el archivo), se compara el audio. Entre varias, la organizada."""
        same_name = library[library["name"] == path.name.casefold()]
        if same_name.empty:
            return None
        hit = same_name[same_name["filesize_bytes"] == path.stat().st_size]
        if hit.empty:
            from src.v4.common.tags import audio_payload_hash
            hit = same_name[same_name["track_uid"] == audio_payload_hash(path)]
        if hit.empty:
            return None
        return hit.sort_values("organized", ascending=False).iloc[0].to_dict()

    def _traktor_analyzed(self, files: List[Path]) -> Optional[int]:
        """Cuántos de estos archivos tienen BPM en la colección de Traktor (por nombre); None sin Traktor."""
        try:
            from src.v4.common.traktor_nml import find_collection
            from src.v4.pipeline.phase1_tags import traktor_bpm_key
            collection = find_collection()
            if collection is None:
                return None
            entries = traktor_bpm_key(collection)
        except Exception:  # noqa: BLE001 — es un aviso
            return None
        return sum(any(bpm > 0 for bpm, _, _ in entries.get(p.name.casefold(), [])) for p in files)

    def _copy_destination(self, folder: Path) -> Path:
        """Una carpeta con ese nombre que ya es una copia (entera o a medias) de esta, de un intento anterior que
        se cortó, se completa en vez de hacer «nombre (2)»."""
        dest, k = self.library / folder.name, 2
        while dest.exists() and not is_copy_of(dest, folder):
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
        pkg = self.package_dir()
        if pkg is not None:
            steps = steps + ["Actualizar el pendrive"]
        S = {name: k for k, name in enumerate(steps)}

        def fn(job: Job) -> Dict:
            since = self.store.meta()["current_version"]   # el pendrive reescribe solo lo que cambió desde acá
            folder = Path(info["path"])
            if not info["inside"]:
                dest = Path(info["dest"])
                copy_tree(folder, dest, job, S["Copiar a tu biblioteca"], skip=set(info.get("skip") or []))
                folder = dest
            scope = folder.relative_to(self.library).as_posix()
            pipe = REPO_ROOT / "src" / "v4" / "pipeline"
            ds = ["--dataset-name", self.dataset]
            step = S["Leer los temas"]
            job.set(step, 0, "Leyendo los tags de la carpeta…")
            run_script([str(pipe / "phase0_ingest.py"), *ds, "--scope", scope], job, step)
            notes = self._bpm_key(scope, job, S["BPM y tonalidad (de Traktor)"])
            step = S["Embeddings del sonido"]
            job.set(step, 0, "Los que vienen de la descarga ya están calculados: se suman a la colección…")
            backend = self.store.meta()["params"]["rep"].split("_")[0]
            run_script([str(pipe / "extract_representations.py"), *ds, "--models", backend, "--folder", scope,
                        "--min-duration", "90", "--max-duration", "900", "--incremental"], job, step)
            step = S["Voces"]
            job.set(step, 0, "")
            if backend == "clap":
                # clap_probe, la regla vigente (DECISIONS 2026-10-05), solo sobre la carpeta (--scope): antes leía y
                # comparaba los comentarios de toda la biblioteca.
                run_script([str(pipe / "tag_vocals.py"), *ds, "--method", "clap_probe", "--write-tags",
                            "--scope", scope], job, step)
            notes += self._genres(scope, job, S["Género y subgénero"])
            job.check()
            step = S["Temas repetidos"]
            job.set(step, 0, "Comparando con tu colección…")
            dups = dedupe.auto_resolve(self.artifacts, self.library, self.org_name, scope)
            job.check()
            step = S["Ubicar en las playlists"]
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
            if dups["resolved"]:  # la copia buena puede ser una que ya estaba: sale la vieja
                v = dedupe.apply(self.artifacts, self.org_name)
                dedupe.move_copies(self.artifacts, self.library,
                                   protected=dedupe.protected_roots(self.config, self.dataset))
            counts["repeated"] = sum(len(g["tracks"]) - 1 for g in dups["resolved"])
            counts["to_decide"] = len(dups["pending"])
            job.set(step, 1, "")
            result = {"version": v, "counts": counts, "scope": scope, "keep": keep, "notes": notes}
            if pkg is not None:  # playlists del pendrive y lugar en el archivo de los temas nuevos o que se movieron
                result.update(self._update_package(job, S["Actualizar el pendrive"], since_version=since if keep else None))
            return result
        return self.start("add", steps, fn)

    def complete_metadata(self, path: str) -> Job:
        """BPM, tonalidad y género de una carpeta que ya está organizada (p. ej. después de analizarla en
        Traktor), sin tocar las playlists."""
        info = self.describe_folder(path)
        if not info["inside"] or info["is_library"]:
            raise ValueError("Elegí una subcarpeta de tu biblioteca que ya esté agregada.")
        scope = Path(info["path"]).relative_to(self.library).as_posix()
        steps = COMPLETE_STEPS

        def fn(job: Job) -> Dict:
            notes = self._bpm_key(scope, job, 0)
            notes += self._genres(scope, job, 1)
            return {"scope": scope, "notes": notes}
        return self.start("complete", steps, fn)

    def _bpm_key(self, scope: str, job: Job, step: int) -> List[str]:
        """features/bpm_key: tags, después lo que Traktor analizó (collection.nml: no lo escribe en el archivo),
        y el BPM que siga faltando se estima del audio. Avisa si temas de la carpeta quedaron sin tonalidad."""
        import pandas as pd
        job.set(step, 0, "")
        run_script([str(REPO_ROOT / "src" / "v4" / "pipeline" / "phase1_tags.py"), "--dataset-name", self.dataset,
                    "--traktor", "--estimate-missing"], job, step)
        try:
            cat = pd.read_parquet(self.artifacts / "catalog.parquet", columns=["track_uid", "rel_path"])
            uids = cat.loc[cat["rel_path"].str.startswith(scope.rstrip("/") + "/"), "track_uid"]
            table = pd.read_parquet(self.artifacts / "features" / "bpm_key.parquet", columns=["track_uid", "key"])
            n = int(table.loc[table["track_uid"].isin(uids), "key"].isna().sum())
        except Exception:  # noqa: BLE001 — un aviso no corta el agregado
            return []
        if not n:
            return []
        return [f"{'1 tema quedó' if n == 1 else f'{n} temas quedaron'} sin tonalidad: Traktor todavía no "
                f"{'lo analizó' if n == 1 else 'los analizó'}. Cargá la carpeta en Traktor y analizala (Analyze); "
                "después elegila de nuevo en «Agregar música nueva» y tocá «Completar BPM, tonalidad y género»."]

    def _genres(self, scope: str, job: Job, step: int) -> List[str]:
        """Género (de Beatport, o del modelo si Beatport no tiene el tema) y subgénero de la música agregada: las
        tablas features/beatport, beatport_genres y genre_pred que leen las playlists y las sugerencias en vivo,
        con los scripts de siempre. Cada parte que falle se avisa sin cortar el agregado."""
        import pandas as pd
        from src.v4.common.beatport import available
        from src.v4.pipeline.beatport_lookup import save

        pipe = REPO_ROOT / "src" / "v4" / "pipeline"
        ds = ["--dataset-name", self.dataset]
        features = self.artifacts / "features"
        notes: List[str] = []

        def attempt(what: str, fn: Callable[[], None]) -> bool:
            try:
                fn()
                return True
            except Exception as exc:  # noqa: BLE001
                notes.append(f"{what}: no se pudo ({str(exc).splitlines()[0][:160]}).")
                return False

        job.set(step, 0, "Escuchando el estilo de cada tema…")  # MAEST: el modelo de género lo necesita
        attempt("Estilo de cada tema", lambda: run_script(
            [str(pipe / "extract_representations.py"), *ds, "--models", "maesthf", "--folder", scope,
             "--min-duration", "90", "--max-duration", "900", "--incremental"], job, step))
        job.set(step, 0, "Buscando en Beatport…")
        online, why = available()
        offline = [] if online else ["--offline"]
        if not online:
            notes.append(f"Beatport {why}: el género salió de lo que ya estaba guardado o del modelo. Volvé a "
                         "agregar la carpeta más tarde para buscarlo de nuevo.")

        def lookup() -> None:
            run_script([str(pipe / "beatport_lookup.py"), *ds, "--folder", scope, "--output", BEATPORT_PART,
                        *offline], job, step)
            # solo las filas de esta carpeta: el resto de features/beatport no se toca
            new = pd.read_parquet(features / f"{BEATPORT_PART}.parquet")
            old = pd.read_parquet(features / "beatport.parquet")
            save(pd.concat([old[~old["track_uid"].isin(new["track_uid"])], new], ignore_index=True),
                 features, "beatport")
            for suffix in (".parquet", ".csv", "_summary.json"):
                (features / f"{BEATPORT_PART}{suffix}").unlink(missing_ok=True)

        if attempt("Beatport", lookup):
            job.set(step, 0.5, "Subgénero de Beatport…")
            attempt("Subgénero", lambda: run_script([str(pipe / "beatport_subgenres.py"), *ds, *offline], job, step))
        job.set(step, 0.8, "Género de los temas que Beatport no tiene…")
        attempt("Género del modelo", lambda: run_script([str(pipe / "genre_model.py"), *ds, "--predict"], job, step))
        job.set(step, 1, "")
        return notes

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
        pkg = self.package_dir()

        def fn(job: Job) -> Dict:
            from src.v4.pipeline.phase5_export import run_export
            job.set(0, 0.1, "Escribiendo las playlists…")
            out = run_export(self.dataset, self.config, formats=("m3u8", "rekordbox", "traktor"),
                             out_root=self.artifacts / "exports", org_name=self.org_name)
            job.set(0, 1, "")
            result = {"folder": str(Path(out).resolve()), "rekordbox": str((Path(out) / "rekordbox.xml").resolve()),
                      "traktor": str((Path(out) / "traktor.nml").resolve())}
            if pkg is not None:  # y las del pendrive (Preparar Rekordbox/Traktor.bat del paquete)
                result.update(self._update_package(job, 1))
            return result
        steps = ["Preparar las playlists para Rekordbox y Traktor"] + (["Actualizar el pendrive"] if pkg else [])
        return self.start("export", steps, fn)

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
            if path.startswith("/cover/"):
                return self._cover(urllib.parse.unquote(path[len("/cover/"):]))
            if path == "/api/ping":  # la app de descargas: no se cierra con una tarea en curso
                return self._json(200, {"app": "traktor-ml", "org": app.org_name, "busy": app.running() is not None,
                                        "readonly": app.readonly})
            if path.startswith("/api/job/"):
                job = app.jobs.get(path.rsplit("/", 1)[-1])
                return self._json(200, job.public()) if job else self._json(404, {"error": "no existe"})
            return self._json(404, {"error": "no existe"})

        def _cover(self, rel: str):
            target = app.audio_file(rel)
            if target is None:
                return self._json(404, {"error": "no existe"})
            found = read_cover(target)
            if not found:
                return self._json(404, {"error": "sin carátula"})
            data, ctype = found
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "private, max-age=3600")
            self.end_headers()
            self.wfile.write(data)

        def _audio(self, rel: str):
            target = app.audio_file(rel)
            if target is None:
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
            if app.readonly:  # todas las acciones cambian la organización o necesitan Música/
                return self._json(400, {"error": READONLY})
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
                if path == "/api/complete-metadata":
                    return self._json(200, app.complete_metadata(body["path"]).public())
                if path == "/api/cancel":
                    return self._json(200, app.cancel())
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


def stdin_closed() -> bool:
    """¿Se cerró la entrada estándar? Sin dejar una lectura pendiente: en Windows, con un hilo bloqueado leyendo el
    pipe de la entrada, la app se trabó dos veces el 2026-11-07 (un script que no arrancaba y, después, la app
    entera al importar dedupe a mitad de una tarea). PeekNamedPipe mira sin bloquear."""
    if sys.stdin is None:
        return False
    try:
        fd = sys.stdin.fileno()
    except (OSError, ValueError, AttributeError):  # no es un archivo del sistema (p. ej. en memoria): leer no bloquea
        try:
            return sys.stdin.read() == ""
        except (OSError, ValueError):
            return False
    if os.name == "nt":
        import ctypes
        import msvcrt
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.PeekNamedPipe.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p,
                                      ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        try:
            handle = msvcrt.get_osfhandle(fd)
        except OSError:
            return False
        avail = wintypes.DWORD(0)
        if not k32.PeekNamedPipe(handle, None, 0, None, ctypes.byref(avail), None):
            return ctypes.get_last_error() in (109, 232, 233)  # pipe roto / cerrándose / sin conectar
        if avail.value:  # la app de descargas no escribe nada; si algo llega, se descarta
            os.read(fd, avail.value)
        return False
    import select
    ready, _, _ = select.select([fd], [], [], 0)
    return bool(ready) and os.read(fd, 4096) == b""


def process_alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except ImportError:
        pass
    except Exception:  # noqa: BLE001 (psutil.NoSuchProcess y similares)
        return False
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(handle, ctypes.byref(code))
        k32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def exit_with_parent(app: App, httpd: ThreadingHTTPServer, parent_pid: Optional[int] = None,
                     watch_stdin: bool = True, poll_s: float = 1.0) -> None:
    """Se cierra cuando la app de descargas cierra la entrada estándar (no escribe nada en ella) o cuando su proceso
    ya no está (se cayó). Una tarea a medias (agregar música, fusiones, export) termina antes de cerrar."""
    while not ((watch_stdin and stdin_closed()) or (parent_pid and not process_alive(parent_pid))):
        time.sleep(poll_s)
    while app.running():
        time.sleep(2)
    httpd.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description="App local: revisar y organizar la colección desde el navegador")
    parser.add_argument("--dataset-name", default="musica")
    parser.add_argument("--org-name", default="biblioteca")
    parser.add_argument("--config", default=None)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--exit-on-stdin-eof", action="store_true",
                        help="Cerrarse cuando se cierra la entrada estándar (la app que la lanzó se cerró o se cayó), "
                             "después de la tarea en curso (sin dejar una lectura pendiente)")
    parser.add_argument("--parent-pid", type=int, default=None,
                        help="Cerrarse cuando termina ese proceso (la app que la lanzó), después de la tarea en curso")
    parser.add_argument("--package", default=None,
                        help="Sin la carpeta de música: el paquete de usb_package.py del que salen los temas "
                             "(si no, se busca <unidad>:\\TRAKTOR ML: primero en la unidad de los datos de la "
                             "organización, después en los discos fijos y en los extraíbles)")
    args = parser.parse_args()
    app = App(args.dataset_name, args.org_name, args.config, args.package)
    httpd = serve(app, args.port, not args.no_browser)
    if args.parent_pid or args.exit_on_stdin_eof:
        threading.Thread(target=exit_with_parent, args=(app, httpd, args.parent_pid, args.exit_on_stdin_eof),
                         daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
