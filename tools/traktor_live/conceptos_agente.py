"""
PURPOSE: Agente local de «Traktor Concept AI» (docs/reports/traktor_concept_ai_cloudflare_2026-11-09.md). El DJ
         escribe en el teléfono qué quiere escuchar; el Worker de Cloudflare arma un concepto (frases en inglés
         para CLAP y condiciones) y lo deja en una cola. Este proceso consulta la cola por HTTPS, sin abrir
         puertos, ordena la biblioteca por parecido texto-audio y devuelve los temas. Audio, embeddings y rutas
         se quedan en este ordenador: a Cloudflare solo van título, artista, BPM, tonalidad, Vocal y parecido.
         - Ranking: coseno entre el promedio de las frases positivas (CLAP de texto) y el embedding clap_full
           del índice de recomendación (indice_recomendacion.py). Sirve para ordenar; no es una probabilidad.
         - Candidatos: los del recomendador, es decir los temas del paquete que están en la colección de
           Traktor (con su BPM y su tonalidad de Traktor); sin colección, todos los del índice.
         - Condiciones aplicadas: voces (etiqueta Vocal de clap_probe), BPM cerca del tema que suena (±20, la
           regla de Parecidos) y fuera el tema principal y lo tocado (registro de la sesión en vivo).
         - Sin aplicar, y lo dice en el teléfono: la energía relativa (no hay coordenada validada: plan,
           sección 11) y las frases negativas (DECISIONS 2026-11-09: restarlas en CLAP cambia la lista sin
           sacar lo excluido).
         - Cada lista se escribe en «TRAKTOR ML Concepto NML.nml», en la carpeta de las sugerencias en vivo.
         CLAP ocupa cerca de 1 GB: se carga con el primer concepto y, sin conceptos por 20 min, el proceso
         termina y libera la memoria (el servicio lo vuelve a lanzar, liviano). Corre con prioridad baja y dos
         hilos, para no competir con el audio de Traktor.
         Subcomandos: correr (la cola, en primer plano), probar (una frase, sin red), servicio (lo que corre
         el inicio automático: el agente solo mientras Traktor está abierto), instalar, quitar, estado.
         URL y DEVICE_TOKEN: variables TRAKTOR_CONCEPT_URL y TRAKTOR_CONCEPT_DEVICE_TOKEN, o
         artifacts/v4/traktor_live/conceptos_agente.json ({"url": ..., "device_token": ...}), fuera de Git.
CHANGELOG:
  - 2026-11-09: Creación inicial.
"""
from __future__ import annotations

import argparse
import datetime as dt
import http.client
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from src.v4.common.traktor_nml import find_collection, traktor_running  # noqa: E402
from tools.traktor_live import recomendador as rc  # noqa: E402
from tools.traktor_live import sugerencias_servicio as ss  # noqa: E402
from tools.traktor_live import sugerencias_vivo as sv  # noqa: E402
from tools.traktor_live import traktor_estado as te  # noqa: E402
from tools.traktor_live.sugerencias_explorer import entries_by_path, nml_bytes, write_atomic  # noqa: E402

DEFAULT_URL = "https://traktor-concept-ai.gabobibbo2.workers.dev"
USER_AGENT = "TRAKTOR-ML-conceptos/1"       # con el agente por defecto de urllib, Cloudflare puede rechazar
NAME = "TRAKTOR ML Conceptos"
CONFIG_FILE = te.DEFAULT_LOG_DIR / "conceptos_agente.json"
LOG_FILE = te.DEFAULT_LOG_DIR / "conceptos_agente.log"
PID_FILE = te.DEFAULT_LOG_DIR / "conceptos_agente.pid"
STOP_FILE = te.DEFAULT_LOG_DIR / "conceptos_agente.stop"
RESULTS_LOG = te.DEFAULT_LOG_DIR / "conceptos_resultados.jsonl"
VOCAL_FILE = REPO_ROOT / "artifacts" / "v4" / "datasets" / "musica" / "features" / "vocals_clap_probe.parquet"
NML_NAME = "TRAKTOR ML Concepto NML"
MAX_TRACKS = 30                             # el tope del Worker y el tamaño de la lista de Parecidos
POLL_S = 4.0
IDLE_EXIT_S = 20 * 60.0
SESSION_FRESH_S = 600.0                     # un registro de sesión más viejo no es la sesión de ahora
THREADS = 2
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
LOG_MAX_BYTES = 1_000_000
ENERGY_TEXT = {"maintain": "mantener", "increase": "subir", "decrease": "bajar"}
VOICE_WORDS = re.compile(r"\b(vocals?|voices?|voz|voces|sing\w*|lyrics?|rapp\w*)\b", re.IGNORECASE)
TRANSIENT = (OSError, http.client.HTTPException, ValueError)   # red caída, respuesta cortada o ilegible
Say = Callable[[str, str], None]


class NoAutorizado(Exception):
    """El Worker rechazó el DEVICE_TOKEN."""


class ResultadoRechazado(Exception):
    """El Worker no aceptó el resultado (HTTP 400): reintentarlo igual no sirve."""


class ConceptoInvalido(ValueError):
    """El trabajo no trae un concepto usable. El texto se muestra en el teléfono."""


class BibliotecaNoDisponible(RuntimeError):
    """Falta el índice de recomendación. El texto se muestra en el teléfono: sin rutas."""


# ---------------------------------------------------------------- configuración y Worker

@dataclass(frozen=True)
class Ajustes:
    url: str
    token: str


def load_ajustes(path: Path = CONFIG_FILE, env: Optional[Dict[str, str]] = None) -> Optional[Ajustes]:
    """Variables de entorno sobre el archivo local. None si no hay token."""
    env = os.environ if env is None else env
    data: dict = {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        pass
    token = env.get("TRAKTOR_CONCEPT_DEVICE_TOKEN") or data.get("device_token") or ""
    url = env.get("TRAKTOR_CONCEPT_URL") or data.get("url") or DEFAULT_URL
    return Ajustes(url=str(url).rstrip("/"), token=str(token)) if token else None


class WorkerClient:
    """Las dos rutas del ordenador. Nunca escribe el token en mensajes ni en el registro."""

    def __init__(self, url: str, token: str, timeout: float = 20.0):
        self.url, self.token, self.timeout = url.rstrip("/"), token, timeout

    def _post(self, path: str, body: dict) -> Tuple[int, dict]:
        request = urllib.request.Request(
            self.url + path, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json", "X-Device-Token": self.token, "User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status, json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, json.loads(exc.read().decode("utf-8") or "{}")
            except ValueError:
                return exc.code, {}

    def claim(self, job_id: Optional[str] = None) -> Optional[dict]:
        """El trabajo más viejo en cola (o solo ese job_id), ya reclamado; None si no hay."""
        status, data = self._post("/api/device/claim", {"job_id": job_id} if job_id else {})
        if status == 401:
            raise NoAutorizado()
        if status != 200:
            raise OSError(f"claim: HTTP {status} {data.get('error', '')}")
        return data.get("job")

    def complete(self, job_id: str, attempts: int, status: str, result: Optional[dict] = None,
                 error: Optional[str] = None) -> bool:
        """True si quedó guardado; False si el Worker dice que el reclamo ya no vale (409)."""
        body: dict = {"id": job_id, "attempts": attempts, "status": status}
        if status == "done":
            body["result"] = result
        else:
            body["error"] = error
        code, data = self._post("/api/device/complete", body)
        if code == 401:
            raise NoAutorizado()
        if code == 409:
            return False
        if code == 400:
            raise ResultadoRechazado(str(data.get("error", "")))
        if code != 200:
            raise OSError(f"complete: HTTP {code} {data.get('error', '')}")
        return True


# ---------------------------------------------------------------- concepto y sesión

@dataclass(frozen=True)
class Concepto:
    name: str
    positive: Tuple[str, ...]
    negative: Tuple[str, ...] = ()
    energy: str = "unspecified"
    vocals: str = "unspecified"
    bpm: str = "unspecified"


def parse_concepto(raw) -> Concepto:
    """El concepto de un trabajo (versión 1 del contrato del Worker) o ConceptoInvalido."""
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ConceptoInvalido("Este concepto tiene un formato que el ordenador no conoce.")
    positive = raw.get("positive_prompts")
    negative = raw.get("negative_prompts") or []
    constraints = raw.get("constraints") or {}
    if not isinstance(positive, list) or not positive or not all(isinstance(p, str) and p.strip() for p in positive):
        raise ConceptoInvalido("El concepto no trae frases para buscar.")
    if not isinstance(negative, list) or not isinstance(constraints, dict):
        raise ConceptoInvalido("El concepto está mal formado.")
    return Concepto(name=str(raw.get("name") or "Concepto"), positive=tuple(p.strip() for p in positive),
                    negative=tuple(str(p).strip() for p in negative),
                    energy=str(constraints.get("energy", "unspecified")),
                    vocals=str(constraints.get("vocals", "unspecified")),
                    bpm=str(constraints.get("bpm", "unspecified")))


@dataclass
class Sesion:
    principal: Optional[dict] = None
    tocados: List[str] = field(default_factory=list)


def leer_sesion(log_dir: Path = te.DEFAULT_LOG_DIR, running: Callable[[], bool] = traktor_running,
                now: Callable[[], float] = time.time) -> Sesion:
    """Tema principal y tocados, del registro de la sesión en vivo más reciente. Vacío si Traktor está
    cerrado o el registro no se escribió hace rato: esa no es la sesión de ahora."""
    if not running():
        return Sesion()
    logs = list(Path(log_dir).glob("sugerencias_vivo_*.jsonl"))
    if not logs:
        return Sesion()
    newest = max(logs, key=lambda p: p.stat().st_mtime)
    if now() - newest.stat().st_mtime > SESSION_FRESH_S:
        return Sesion()
    sesion = Sesion()
    with open(newest, encoding="utf-8", errors="replace") as f:
        for line in f:
            head = line[:80]
            if '"kind": "principal"' not in head and '"kind": "tocado"' not in head:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue                                # la última línea puede estar a medio escribir
            data = event.get("data") or {}
            if event.get("kind") == "principal":
                sesion.principal = data
            elif data.get("path"):
                sesion.tocados.append(str(data["path"]))
    return sesion


# ---------------------------------------------------------------- biblioteca y ranking

class CodificadorTexto:
    """CLAP de texto (el mismo modelo que los embeddings de audio). Se carga la primera vez que hace falta."""

    def __init__(self):
        self.backend = None

    def _load(self) -> None:
        os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
        os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")    # el modelo ya está en el disco: sin esperar a la red
        import torch
        from src.v4.pipeline.extract_representations import ClapBackend, lower_priority
        lower_priority()
        torch.set_num_threads(THREADS)
        self.backend = ClapBackend(None)

    def __call__(self, frases: Sequence[str]):
        if self.backend is None:
            self._load()
        return self.backend.embed_text(list(frases))


@dataclass
class Biblioteca:
    items: List[rc.Item]
    matrix: object                              # numpy, una fila por item, con norma 1
    vocal: Dict[str, bool]
    recommender: rc.Recommender
    nml_version: Optional[str] = None           # None: sin colección de Traktor, no se escribe playlist
    descripcion: str = ""


def cargar_biblioteca(index_path: Path = rc.DEFAULT_INDEX, collection: Optional[Path] = None,
                      vocal_path: Path = VOCAL_FILE, say: Optional[Say] = None) -> Biblioteca:
    import numpy as np
    say = say or (lambda level, text: None)
    collection = collection if collection is not None else find_collection()
    entries = entries_by_path(collection) if collection is not None and Path(collection).exists() else {}
    rec, text = rc.load_recommender(entries, Path(index_path), MAX_TRACKS)
    version = None
    if rec is not None:
        items = rec.candidates
        version = ET.parse(collection).getroot().get("VERSION", "20")
    else:
        try:
            _, todos = rc.load_index(Path(index_path))
        except (OSError, ValueError, KeyError) as exc:
            raise BibliotecaNoDisponible("El ordenador no tiene el índice de la biblioteca.") from exc
        rec, items = rc.Recommender(todos, {}, MAX_TRACKS), todos
    con_vector = [it for it in items if it.vec is not None]
    if not con_vector:
        raise BibliotecaNoDisponible("El índice de la biblioteca no tiene embeddings.")
    matrix = np.array([it.vec for it in con_vector], dtype=np.float32)
    vocal: Dict[str, bool] = {}
    try:
        import pandas as pd
        table = pd.read_parquet(vocal_path, columns=["track_uid", "is_vocal"])
        vocal = {str(u): bool(v) for u, v in zip(table["track_uid"], table["is_vocal"])}
    except (OSError, ValueError, KeyError, ImportError) as exc:
        say("aviso", f"Sin etiqueta Vocal ({type(exc).__name__}): el filtro de voces no se va a aplicar")
    return Biblioteca(items=con_vector, matrix=matrix, vocal=vocal, recommender=rec, nml_version=version,
                      descripcion=text)


def filtrar(concepto: Concepto, items: Sequence[rc.Item], vocal: Dict[str, bool], sesion: Sesion,
            recommender: rc.Recommender) -> Tuple[List[bool], List[str]]:
    """Qué temas pasan y las notas para el teléfono: lo aplicado y, sobre todo, lo que no se aplicó."""
    principal = sesion.principal or {}
    main = recommender.identify(str(principal["path"])) if principal.get("path") else None
    fuera = {it.uid for it in (recommender.identify(p) for p in sesion.tocados) if it is not None}
    if main is not None:
        fuera.add(main.uid)
    keep = [it.uid not in fuera for it in items]
    notas: List[str] = []

    if concepto.vocals in ("present", "absent"):
        if vocal:
            wanted = concepto.vocals == "present"
            keep = [k and vocal.get(it.uid) is wanted for k, it in zip(keep, items)]
            notas.append(f"{'Con voces' if wanted else 'Sin voces'} (etiqueta Vocal): {sum(keep)} temas.")
        else:
            notas.append("Voces: sin aplicar, en este ordenador no está la etiqueta Vocal.")

    if concepto.bpm == "near_current":
        tempo = float(principal.get("tempo") or 1.0) or 1.0
        base = float(principal.get("bpm") or 0.0)
        if base <= 0 and main is not None and main.bpm:
            base = main.bpm
        if base > 0:
            actual = base * tempo
            keep = [k and it.bpm is not None and abs(it.bpm - actual) <= rc.BPM_RANGE for k, it in zip(keep, items)]
            notas.append(f"BPM a ±{rc.BPM_RANGE:.0f} de {actual:.0f}: {sum(keep)} temas.")
        else:
            notas.append("BPM cerca del actual: sin aplicar, no hay un tema sonando.")

    if concepto.energy in ENERGY_TEXT:
        notas.append(f"Energía ({ENERGY_TEXT[concepto.energy]}): sin aplicar, todavía no hay una medida validada.")

    negativas = [n for n in concepto.negative if not (concepto.vocals == "absent" and VOICE_WORDS.search(n))]
    if negativas:
        notas.append(("Exclusiones sin aplicar (CLAP no las separa bien): " + "; ".join(negativas))[:200])
    if fuera:
        notas.append(f"Fuera de la lista: {len(fuera)} ya tocados o sonando.")
    return keep, notas


def titulo_artista(it: rc.Item) -> Tuple[str, str]:
    entry = it.entry
    title = (entry.get("TITLE") if entry is not None else "") or it.title
    artist = (entry.get("ARTIST") if entry is not None else "") or it.artist
    if not title:
        title = Path(str(it.rel).replace("\\", "/")).stem or it.uid[:12]
    return title[:160], (artist or "")[:120]         # holgado para el tope de 24 KB del Worker


def resolver(concepto: Concepto, bib: Biblioteca, codificar: Callable[[Sequence[str]], object], sesion: Sesion,
             n: int = MAX_TRACKS) -> Tuple[dict, List[rc.Item]]:
    """(resultado para el Worker, los temas elegidos en orden)."""
    import numpy as np
    frases = np.asarray(codificar(concepto.positive), dtype=np.float64)
    query = frases.mean(axis=0)
    query = query / np.linalg.norm(query)
    sims = np.asarray(bib.matrix @ query, dtype=np.float64)
    keep, notas = filtrar(concepto, bib.items, bib.vocal, sesion, bib.recommender)
    passing = np.flatnonzero(np.asarray(keep, dtype=bool))
    order = passing[np.argsort(-sims[passing], kind="stable")][:n]
    tracks = []
    chosen = []
    for i in order:
        it = bib.items[int(i)]
        title, artist = titulo_artista(it)
        track = {"track_id": it.uid, "title": title, "artist": artist,
                 "score": round(float(np.clip(sims[int(i)], -1.0, 1.0)), 4)}
        if it.bpm and 0 < it.bpm <= 400:
            track["bpm"] = round(float(it.bpm), 1)
        if it.key and it.key != "?":
            track["key"] = it.key
        if it.uid in bib.vocal:
            track["vocal"] = bib.vocal[it.uid]
        tracks.append(track)
        chosen.append(it)
    return {"tracks": tracks, "pool": len(bib.items), "passed": int(passing.size), "notes": notas[:8]}, chosen


def carpeta_sugerencias() -> Path:
    return Path(sv.load_settings().get("dir") or sv.default_live_dir())


def escribir_nml(bib: Biblioteca, chosen: Sequence[rc.Item], carpeta: Optional[Path] = None) -> Optional[Path]:
    """La lista como playlist del Explorer de Traktor, con las entradas copiadas de la colección."""
    entries = [it.entry for it in chosen if it.entry is not None]
    if bib.nml_version is None or not entries:
        return None
    destino = (carpeta or carpeta_sugerencias()) / f"{NML_NAME}.nml"
    write_atomic(nml_bytes(entries, int(time.time()), bib.nml_version, name=NML_NAME), destino)
    return destino


# ---------------------------------------------------------------- el trabajo

class Agente:
    """Atiende trabajos reclamados: arma la biblioteca con el primero y devuelve el resultado al Worker."""

    def __init__(self, client: WorkerClient, say: Say,
                 cargar: Optional[Callable[[], Biblioteca]] = None,
                 codificar: Optional[Callable[[Sequence[str]], object]] = None,
                 sesion: Callable[[], Sesion] = leer_sesion,
                 nml: Callable[[Biblioteca, Sequence[rc.Item]], Optional[Path]] = escribir_nml,
                 results_log: Optional[Path] = RESULTS_LOG, sleep: Callable[[float], None] = time.sleep):
        self.client, self.say = client, say
        self.cargar = cargar or (lambda: cargar_biblioteca(say=say))
        self.codificar = codificar or CodificadorTexto()
        self.sesion, self.nml, self.results_log, self.sleep = sesion, nml, results_log, sleep
        self.bib: Optional[Biblioteca] = None

    def _procesar(self, job: dict) -> dict:
        concepto = parse_concepto(job.get("concept"))
        t0 = time.perf_counter()
        if self.bib is None:
            self.bib = self.cargar()
            self.say("info", f"Biblioteca: {len(self.bib.items)} temas. {self.bib.descripcion}")
        result, chosen = resolver(concepto, self.bib, self.codificar, self.sesion())
        try:
            destino = self.nml(self.bib, chosen)
            if destino is None:
                result["notes"] = (result["notes"] + ["No se escribió la playlist: no encontré la colección de Traktor."])[:8]
        except OSError as exc:
            self.say("aviso", f"No pude escribir la playlist de Traktor: {type(exc).__name__}: {exc}")
            result["notes"] = (result["notes"] + ["No se pudo escribir la playlist en Traktor."])[:8]
        ms = round((time.perf_counter() - t0) * 1000)
        self.say("ok", f"«{concepto.name}»: {len(result['tracks'])} temas de {result['passed']} ({ms} ms)")
        self._anotar(job, concepto, result, ms)
        return result

    def _anotar(self, job: dict, concepto: Concepto, result: dict, ms: int) -> None:
        if self.results_log is None:
            return
        try:
            self.results_log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.results_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": dt.datetime.now().isoformat(timespec="seconds"), "job": job.get("id"),
                                    "concepto": job.get("concept"), "notas": result["notes"], "pool": result["pool"],
                                    "pasan": result["passed"], "ms": ms,
                                    "uids": [t["track_id"] for t in result["tracks"]],
                                    "parecido": [t["score"] for t in result["tracks"]]}, ensure_ascii=False) + "\n")
        except OSError as exc:
            self.say("aviso", f"No pude anotar el resultado: {type(exc).__name__}")

    def atender(self, job: dict) -> None:
        job_id, attempts = job["id"], job["attempts"]
        try:
            result = self._procesar(job)
        except (ConceptoInvalido, BibliotecaNoDisponible) as exc:
            self.say("aviso", f"Trabajo {job_id}: {exc}")
            self._completar(job_id, attempts, "failed", error=str(exc))
            return
        except Exception:                                # noqa: BLE001 (un trabajo no tumba el agente)
            traceback.print_exc()
            self.say("error", f"Trabajo {job_id}: error interno, ver {LOG_FILE.name}")
            self._completar(job_id, attempts, "failed", error="Error en el ordenador; queda en su registro local.")
            return
        self._completar(job_id, attempts, "done", result=result)

    def _completar(self, job_id: str, attempts: int, status: str, result: Optional[dict] = None,
                   error: Optional[str] = None) -> None:
        for wait in (0, 2, 5, 15, 30):                   # menos que el lease: si no alcanza, se reintenta el trabajo
            if wait:
                self.sleep(wait)
            try:
                if not self.client.complete(job_id, attempts, status, result=result, error=error):
                    self.say("aviso", f"Trabajo {job_id}: el reclamo ya no vale (lease vencido o ya terminado)")
                return
            except ResultadoRechazado as exc:
                self.say("error", f"Trabajo {job_id}: el Worker rechazó el resultado: {exc}")
                if status == "done":
                    status, result, error = "failed", None, "El servidor rechazó el resultado del ordenador."
                    continue
                return
            except NoAutorizado:
                self.say("error", "El Worker rechazó el DEVICE_TOKEN al guardar el resultado")
                return
            except TRANSIENT as exc:
                self.say("info", f"Trabajo {job_id}: no pude guardar el resultado ({type(exc).__name__}), reintento")
        self.say("aviso", f"Trabajo {job_id}: sin conexión; vuelve a la cola cuando venza el lease")


def bucle(agente: Agente, stop: threading.Event, poll_s: float = POLL_S, idle_exit_s: Optional[float] = None,
          job_id: Optional[str] = None, una_vez: bool = False, clock: Callable[[], float] = time.monotonic) -> int:
    """Consulta la cola hasta que stop se active. Devuelve el código de salida: 0 bien (también al liberar
    memoria), 2 token rechazado."""
    last_job = clock()
    fallos = 0
    while not stop.is_set():
        try:
            job = agente.client.claim(job_id)
            if fallos >= 3:
                agente.say("ok", "Conexión con Cloudflare recuperada")
            fallos = 0
        except NoAutorizado:
            agente.say("error", "El Worker rechazó el DEVICE_TOKEN: revisar conceptos_agente.json")
            return 2
        except TRANSIENT as exc:
            fallos += 1
            if fallos == 3:
                agente.say("aviso", f"Sin conexión con Cloudflare ({type(exc).__name__}): sigo intentando")
            if una_vez:
                return 1
            stop.wait(min(60.0, poll_s * (2 ** min(fallos, 4))))
            continue
        if job is None:
            if una_vez:
                agente.say("info", "No hay trabajos en cola")
                return 0
            if idle_exit_s is not None and agente.bib is not None and clock() - last_job > idle_exit_s:
                agente.say("info", "Sin conceptos hace rato: libero la memoria de CLAP")
                return 0
            stop.wait(poll_s)
            continue
        agente.say("info", f"Trabajo {job['id']} (intento {job['attempts']})")
        agente.atender(job)
        last_job = clock()
        if una_vez:
            return 0
    return 0


# ---------------------------------------------------------------- comandos

def console_say(level: str, text: str) -> None:
    print(f"{dt.datetime.now():%H:%M:%S} [{level.upper()}] {text}", flush=True)


def cmd_correr(args: argparse.Namespace) -> int:
    ajustes = load_ajustes()
    if ajustes is None:
        print(f"[ERROR] Falta el DEVICE_TOKEN: {CONFIG_FILE} o TRAKTOR_CONCEPT_DEVICE_TOKEN")
        return 1
    stop = threading.Event()
    agente = Agente(WorkerClient(ajustes.url, ajustes.token), console_say)
    console_say("info", f"Consultando {ajustes.url} cada {args.cada:g} s")
    try:
        return bucle(agente, stop, poll_s=args.cada, idle_exit_s=args.salir_sin_uso, job_id=args.job, una_vez=args.una_vez)
    except KeyboardInterrupt:
        return 0


def cmd_probar(args: argparse.Namespace) -> int:
    """Una frase contra la biblioteca real, sin red y sin escribir nada salvo con --nml."""
    concepto = Concepto(name="prueba", positive=tuple(args.frase), negative=tuple(args.excluir), vocals=args.voces,
                        bpm="near_current" if args.bpm_cerca else "unspecified", energy=args.energia)
    t0 = time.perf_counter()
    bib = cargar_biblioteca(say=console_say)
    t1 = time.perf_counter()
    codificar = CodificadorTexto()
    result, chosen = resolver(concepto, bib, codificar, leer_sesion(), n=args.n)
    t2 = time.perf_counter()
    resolver(concepto, bib, codificar, leer_sesion(), n=args.n)
    t3 = time.perf_counter()
    print(f"[INFO] {bib.descripcion}")
    print(f"[INFO] biblioteca {t1 - t0:.1f} s; primera búsqueda (con carga de CLAP) {t2 - t1:.1f} s; "
          f"siguiente {1000 * (t3 - t2):.0f} ms; pasan {result['passed']} de {result['pool']}")
    for note in result["notes"]:
        print(f"[NOTA] {note}")
    for i, t in enumerate(result["tracks"], 1):
        print(f"{i:>3}  {t['score']:.3f}  {t.get('bpm', 0):>5.0f}  {t.get('key', '?'):>3}  "
              f"{'V' if t.get('vocal') else ' '}  {t['artist'][:30]} - {t['title'][:60]}")
    if args.nml:
        print(f"[OK] Playlist: {escribir_nml(bib, chosen)}")
    return 0


def _abrir_registro():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if LOG_FILE.exists() and LOG_FILE.stat().st_size > LOG_MAX_BYTES:
        LOG_FILE.replace(LOG_FILE.with_suffix(".log.1"))
    f = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = f
    return f


def _notificar(title: str, text: str) -> None:
    ss.notify(title.replace("Sugerencias en vivo", "Conceptos"), text)


def sesion_hija(log, stop: threading.Event, popen=subprocess.Popen) -> int:
    """El agente en un proceso aparte, mientras dure stop: al terminar, Windows recupera toda su memoria."""
    command = [str(ss.service_python()), str(Path(__file__).resolve()), "correr", "--salir-sin-uso", str(IDLE_EXIT_S)]
    kwargs = {"creationflags": BELOW_NORMAL_PRIORITY_CLASS | ss.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
    child = popen(command, cwd=str(REPO_ROOT), stdout=log, stderr=log, stdin=subprocess.DEVNULL, **kwargs)
    try:
        while not stop.wait(1.0):
            code = child.poll()
            if code is not None:
                return int(code)
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
    return 0


def _pid_corriendo() -> Optional[int]:
    try:
        pid = int(PID_FILE.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None
    return pid if ss.pid_alive(pid) else None


def cmd_servicio(_args: argparse.Namespace) -> int:
    """Lo que corre el inicio automático: el agente solo mientras Traktor está abierto."""
    log = _abrir_registro()
    otro = _pid_corriendo()
    if otro is not None and otro != os.getpid():
        print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} [INFO] Ya hay un servicio de conceptos (PID {otro}): salgo")
        return 0
    STOP_FILE.unlink(missing_ok=True)
    PID_FILE.write_text(str(os.getpid()), encoding="ascii")
    say = ss.ServiceSay(notifier=_notificar)
    say("info", f"Servicio de conceptos iniciado (PID {os.getpid()}); espera a que se abra Traktor")

    def session(stop: threading.Event) -> int:
        code = sesion_hija(log, stop)
        if code == 2:
            say("error", "El Worker rechazó el DEVICE_TOKEN: revisar conceptos_agente.json")
        return code

    try:
        ss.serve_forever(say, session=session, stop_file=STOP_FILE)
    finally:
        say("info", "Servicio de conceptos detenido")
        if _pid_corriendo() in (None, os.getpid()):
            PID_FILE.unlink(missing_ok=True)
        STOP_FILE.unlink(missing_ok=True)
    return 0


def _acceso_directo() -> Path:
    return ss.startup_dir() / f"{NAME}.lnk"


def _arrancar_servicio() -> None:
    subprocess.Popen([str(ss.service_python()), str(Path(__file__).resolve()), "servicio"], cwd=str(REPO_ROOT),
                     creationflags=ss.DETACHED_PROCESS | ss.CREATE_NO_WINDOW, close_fds=True)


def cmd_instalar(_args: argparse.Namespace) -> int:
    if sys.platform != "win32":
        print("[ERROR] El inicio automático del agente es para Windows")
        return 1
    if load_ajustes() is None:
        print(f"[ERROR] Primero hace falta el DEVICE_TOKEN en {CONFIG_FILE}")
        return 1
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
        print(f"[ERROR] Este Python no tiene torch y transformers ({sys.executable}): usar el del pipeline")
        return 1
    lnk = _acceso_directo()
    lnk.parent.mkdir(parents=True, exist_ok=True)
    script = ss.shortcut_script(lnk, ss.service_python(), f'"{Path(__file__).resolve()}" servicio', REPO_ROOT)
    res = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                         capture_output=True, text=True)
    if res.returncode != 0 or not lnk.exists():
        print(f"[ERROR] No pude crear el acceso directo de inicio: {res.stderr.strip()}")
        return 1
    print(f"[OK] Inicio automático: {lnk}")
    if _pid_corriendo() is None:
        _arrancar_servicio()
        print("[OK] Servicio arrancado: al abrir Traktor empieza a consultar la cola")
    else:
        print(f"[INFO] El servicio ya estaba corriendo (PID {_pid_corriendo()})")
    return 0


def cmd_quitar(_args: argparse.Namespace) -> int:
    pid = _pid_corriendo()
    if pid is not None:
        STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
        STOP_FILE.write_text("quitar", encoding="ascii")
        end = time.time() + 8.0
        while time.time() < end and ss.pid_alive(pid):
            time.sleep(0.25)
        if ss.pid_alive(pid):
            ss.terminate(pid)
        print(f"[OK] Servicio detenido (PID {pid})")
    if _acceso_directo().exists():
        _acceso_directo().unlink()
        print(f"[OK] Quité {_acceso_directo()}")
    STOP_FILE.unlink(missing_ok=True)
    return 0


def cmd_estado(_args: argparse.Namespace) -> int:
    ajustes = load_ajustes()
    pid = _pid_corriendo()
    print(f"[INFO] Inicio automático: {'sí' if _acceso_directo().exists() else 'no'}")
    print(f"[INFO] Servicio: {'corriendo, PID ' + str(pid) if pid else 'no está corriendo'}")
    print(f"[INFO] Traktor: {'abierto' if traktor_running() else 'cerrado'}")
    print(f"[INFO] Worker: {ajustes.url if ajustes else 'sin configurar'}; token: {'sí' if ajustes else 'no'}")
    if LOG_FILE.exists():
        print(f"[INFO] Últimas líneas de {LOG_FILE}:")
        for line in LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]:
            print("   " + line)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Agente local de los conceptos que se escriben desde el teléfono")
    sub = ap.add_subparsers(dest="accion", required=True)
    correr = sub.add_parser("correr", help="Consultar la cola en primer plano")
    correr.add_argument("--cada", type=float, default=POLL_S, help="Segundos entre consultas")
    correr.add_argument("--una-vez", action="store_true", help="Atender como mucho un trabajo y salir")
    correr.add_argument("--job", default=None, help="Reclamar solo este trabajo (pruebas: no consume otros)")
    correr.add_argument("--salir-sin-uso", type=float, default=None, help="Con CLAP cargado y sin trabajos por estos segundos, salir")
    correr.set_defaults(func=cmd_correr)
    probar = sub.add_parser("probar", help="Una frase contra la biblioteca, sin red")
    probar.add_argument("--frase", action="append", required=True, help="Frase en inglés para CLAP (repetible)")
    probar.add_argument("--excluir", action="append", default=[], help="Se anota como sin aplicar")
    probar.add_argument("--voces", choices=("unspecified", "present", "absent"), default="unspecified")
    probar.add_argument("--energia", choices=("unspecified", "maintain", "increase", "decrease"), default="unspecified")
    probar.add_argument("--bpm-cerca", action="store_true")
    probar.add_argument("--n", type=int, default=MAX_TRACKS)
    probar.add_argument("--nml", action="store_true", help="Escribir también la playlist de Traktor")
    probar.set_defaults(func=cmd_probar)
    for name, func in (("servicio", cmd_servicio), ("instalar", cmd_instalar), ("quitar", cmd_quitar), ("estado", cmd_estado)):
        sub.add_parser(name).set_defaults(func=func)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
