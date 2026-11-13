"""
PURPOSE: El ordenador en «Traktor Concept AI»: del concepto escrito en el teléfono a una playlist nueva en Traktor
         (arquitectura: cloudflare/README.md). El Worker tiene el LLM y decide; este proceso es el oído: consulta la cola por
         HTTPS, sin abrir puertos, y hace dos tipos de trabajo.
         - sondeo: busca cada texto candidato y cuenta qué encontró (géneros, BPM, voz, ejemplos y «suena a»: las palabras
           del vocabulario que más describen esos temas). Con eso el LLM juzga sin oír.
         - final: busca con los textos elegidos y escribe la playlist «Concepto <fecha> <nombre>.nml» en la carpeta de
           las sugerencias en vivo, que Traktor muestra en su Explorer.
         La búsqueda es buscar(), la única pieza que decide qué temas salen: recibe el formulario del teléfono
         (formulario.py) y lo ejecuta (ejecutor.py): filtros, referencia («lo que suena», del registro del vivo, o un
         tema nombrado), puntaje de Parecidos o coseno CLAP del texto, corte. Un formulario con solo texto da la lista de
         siempre: coseno entre el promedio de los textos (CLAP) y el embedding clap_full del índice de recomendación.
         Audio, embeddings y rutas se quedan en este ordenador: a Cloudflare solo van título, artista, género, BPM,
         tonalidad, Vocal, palabras del vocabulario, el tema que suena (título, artista, BPM, tonalidad) y el catálogo
         de la biblioteca en números (formulario.catalogo()). Los dos últimos viajan en el pedido de trabajo a la cola:
         el tema que suena cuando cambia (y cada 5 min, para que el teléfono sepa que el ordenador sigue), el catálogo
         una vez por proceso.
         CLAP ocupa cerca de 1 GB: se carga con el primer trabajo y, sin trabajos por 20 min, el proceso termina y libera la
         memoria (el servicio lo vuelve a lanzar, liviano). Corre con prioridad baja y dos hilos, para no competir con el
         audio de Traktor.
         Subcomandos: correr (la cola, en primer plano), probar (una frase, sin red), servicio (lo que corre el inicio
         automático: el agente solo mientras Traktor está abierto), instalar, quitar, estado.
         URL y DEVICE_TOKEN: variables TRAKTOR_CONCEPT_URL y TRAKTOR_CONCEPT_DEVICE_TOKEN, o
         artifacts/v4/traktor_live/conceptos_agente.json ({"url": ..., "device_token": ...}), fuera de Git.
CHANGELOG:
  - 2026-11-09: Creación inicial (un trabajo: concepto -> lista, con filtros de voces y BPM).
  - 2026-11-09: Rearmado para el circuito con el LLM (Gabriel: el trabajo es la arquitectura, no los filtros): trabajos
                de sondeo y final, buscar() como pieza cambiable y una playlist nueva por concepto. Sin filtros.
  - 2026-11-13: Línea B del plan v2 (B1): trabajos de versión 3 con formulario, buscar() ejecuta el formulario (con el
                recomendador del vivo y sus ajustes), el sondeo busca dentro de los filtros, la lista dice cuántos temas
                pasaron cada filtro; el tema que suena y el catálogo viajan al Worker. CLAP se libera por inactividad
                solo si se cargó (la biblioteca sola es liviana).
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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from src.v4.common.traktor_nml import find_collection, traktor_running  # noqa: E402
from src.v4.evaluation.clap_texto_frases import VOCABULARIO  # noqa: E402
from tools.traktor_live import ejecutor as ej  # noqa: E402
from tools.traktor_live import formulario as fm  # noqa: E402
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
MAX_TRACKS = 30                             # el tope del Worker y el tamaño de la lista de Parecidos
EJEMPLOS = 6                                # temas de muestra por búsqueda de prueba
SUENA_A = 6                                 # palabras «suena a» por búsqueda de prueba
GRUPOS_FUERA = {"ocasion", "estructura"}    # no describen cómo suena un tema
POLL_S = 4.0
POLL_ACTIVO_S = 1.0                         # mientras dura una conversación con el LLM
ACTIVO_S = 45.0
IDLE_EXIT_S = 20 * 60.0
THREADS = 2
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
LOG_MAX_BYTES = 1_000_000
SESION_FRESCA_S = 20 * 60.0                 # un registro del vivo más quieto que esto no es la sesión de ahora
SONANDO_LATIDO_S = 5 * 60.0                 # el tema que suena se reenvía aunque no cambie: el teléfono sabe que el ordenador sigue
TRANSIENT =(OSError, http.client.HTTPException, ValueError)   # red caída, respuesta cortada o ilegible
Say = Callable[[str, str], None]


class NoAutorizado(Exception):
    """El Worker rechazó el DEVICE_TOKEN."""


class ResultadoRechazado(Exception):
    """El Worker no aceptó el resultado (HTTP 400): reintentarlo igual no sirve."""


class TareaInvalida(ValueError):
    """El trabajo no se puede interpretar. El texto se muestra en el teléfono."""


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

    def __init__(self, url: str, token: str, timeout: float = 90.0):
        self.url, self.token, self.timeout = url.rstrip("/"), token, timeout    # complete espera al LLM

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

    def claim(self, job_id: Optional[str] = None, extra: Optional[dict] = None) -> Optional[dict]:
        """El trabajo más viejo en cola (o solo ese job_id), ya reclamado; None si no hay. extra: lo que el ordenador le
        cuenta al Worker de paso («sonando», «catalogo»)."""
        body = dict(extra or {})
        if job_id:
            body["job_id"] = job_id
        status, data = self._post("/api/device/claim", body)
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


# ---------------------------------------------------------------- tarea

@dataclass(frozen=True)
class Tarea:
    fase: str                                   # "sondeo" o "final"
    nombre: str
    textos: Tuple[Tuple[str, str], ...]         # (id, texto en inglés)
    formulario: Optional[dict] = None           # versión 3: el formulario sin la casilla texto (la llenan los textos)


def _textos(raw) -> Tuple[Tuple[str, str], ...]:
    textos = raw.get("textos")
    if not isinstance(textos, list) or not all(
            isinstance(t, dict) and isinstance(t.get("id"), str) and isinstance(t.get("texto"), str) and t["texto"].strip()
            for t in textos):
        raise TareaInvalida("El trabajo no trae textos para buscar.")
    return tuple((t["id"], t["texto"].strip()) for t in textos)


def parse_tarea(raw) -> Tarea:
    """El trabajo de la cola. El formato 1 (conceptos guardados antes del circuito) es directamente un final. El 3 trae
    el formulario: un preset llega como final sin textos; un texto, como el 2, con el formulario de los presets."""
    if not isinstance(raw, dict):
        raise TareaInvalida("El trabajo tiene un formato que el ordenador no conoce.")
    if raw.get("version") == 1:
        frases = raw.get("positive_prompts")
        if isinstance(frases, list) and frases and all(isinstance(f, str) and f.strip() for f in frases):
            return Tarea("final", str(raw.get("name") or "Concepto"),
                         tuple((chr(65 + i), f.strip()) for i, f in enumerate(frases)))
        raise TareaInvalida("El concepto no trae frases para buscar.")
    if raw.get("version") not in (2, 3) or raw.get("fase") not in ("sondeo", "final"):
        raise TareaInvalida("El trabajo tiene un formato que el ordenador no conoce.")
    textos = _textos(raw)
    nombre = str(raw.get("nombre") or "Concepto")
    if raw["version"] == 2:
        if not textos:
            raise TareaInvalida("El trabajo no trae textos para buscar.")
        return Tarea(raw["fase"], nombre, textos)
    if not isinstance(raw.get("formulario", {}), dict):
        raise TareaInvalida("El trabajo no trae un formulario.")
    form, _ = fm.validar(raw.get("formulario", {}))      # el Worker ya validó; lo que no está en las listas se descarta
    if raw["fase"] == "sondeo" and not textos:
        raise TareaInvalida("El trabajo no trae textos para buscar.")
    if not textos and fm.es_vacio(form):
        raise TareaInvalida("El pedido no tiene nada que buscar.")
    return Tarea(raw["fase"], nombre, textos, form)


# ---------------------------------------------------------------- biblioteca y búsqueda

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
    nml_version: Optional[str] = None           # None: sin colección de Traktor, no se escribe playlist
    descripcion: str = ""
    rec: Optional[rc.Recommender] = None        # el de Parecidos, con los ajustes del vivo; None sin colección


def cargar_biblioteca(index_path: Path = rc.DEFAULT_INDEX, collection: Optional[Path] = None,
                      vocal_path: Path = VOCAL_FILE, say: Optional[Say] = None,
                      ajustes_vivo: Optional[dict] = None) -> Biblioteca:
    """Candidatos = los temas del paquete que están en la colección de Traktor; sin colección, todo el índice. El
    recomendador lleva los pesos y opciones de sugerencias_vivo.json, para que «lo que suena» dé lo mismo que Parecidos."""
    import numpy as np
    say = say or (lambda level, text: None)
    collection = collection if collection is not None else find_collection()
    entries = entries_by_path(collection) if collection is not None and Path(collection).exists() else {}
    vivo = sv.load_settings() if ajustes_vivo is None else ajustes_vivo
    rec, text = rc.load_recommender(entries, Path(index_path), MAX_TRACKS, vivo.get("pesos"), options=vivo.get("opciones"))
    version = None
    if rec is not None:
        items = rec.candidates
        version = ET.parse(collection).getroot().get("VERSION", "20")
    else:
        try:
            _, items = rc.load_index(Path(index_path))
        except (OSError, ValueError, KeyError) as exc:
            raise BibliotecaNoDisponible("El ordenador no tiene el índice de la biblioteca.") from exc
    con_vector = [it for it in items if it.vec is not None]
    if not con_vector:
        raise BibliotecaNoDisponible("El índice de la biblioteca no tiene embeddings.")
    vocal: Dict[str, bool] = {}
    try:
        import pandas as pd
        table = pd.read_parquet(vocal_path, columns=["track_uid", "is_vocal"])
        vocal = {str(u): bool(v) for u, v in zip(table["track_uid"], table["is_vocal"])}
    except (OSError, ValueError, KeyError, ImportError) as exc:
        say("aviso", f"Sin etiqueta Vocal ({type(exc).__name__}): la voz no se va a informar")
    return Biblioteca(items=con_vector, matrix=np.array([it.vec for it in con_vector], dtype=np.float32), vocal=vocal,
                      nml_version=version, descripcion=text, rec=rec)


def buscar(bib: Biblioteca, formulario: dict, sesion: Optional[ej.Sesion] = None,
           codificar: Optional[Callable[[Sequence[str]], object]] = None) -> ej.Resultado:
    """EL BUSCADOR: qué temas salen para un formulario del teléfono, en orden, y qué hizo para elegirlos (ejecutor.py).
    Los dos casos de uso entran acá: sin referencia (antes de empezar a tocar) y con el set en curso (la sesión del vivo:
    lo que suena y lo ya tocado). Ni el Worker ni el LLM se enteran de cómo se elige."""
    return ej.ejecutar(formulario, bib, sesion, codificar)


class SesionVivo:
    """Lee el registro de la sesión en vivo más reciente (sugerencias_vivo_*.jsonl) de a poco: el tema principal y lo ya
    tocado. Vacío si el registro no se escribió hace rato: esa no es la sesión de ahora."""

    def __init__(self, log_dir: Path = te.DEFAULT_LOG_DIR, now: Callable[[], float] = time.time,
                 fresca_s: float = SESION_FRESCA_S):
        self.log_dir, self.now, self.fresca_s = Path(log_dir), now, fresca_s
        self.path: Optional[Path] = None
        self.offset = 0
        self.sesion = ej.Sesion()

    def actual(self) -> ej.Sesion:
        try:
            logs = list(self.log_dir.glob("sugerencias_vivo_*.jsonl"))
            newest = max(logs, key=lambda p: p.stat().st_mtime) if logs else None
            if newest is None or self.now() - newest.stat().st_mtime > self.fresca_s:
                return ej.Sesion()
            if newest != self.path or newest.stat().st_size < self.offset:
                self.path, self.offset, self.sesion = newest, 0, ej.Sesion()
            with open(newest, "rb") as f:
                f.seek(self.offset)
                data = f.read()
        except OSError:
            return ej.Sesion()
        fin = data.rfind(b"\n") + 1                     # la última línea puede estar a medio escribir
        self.offset += fin
        for raw in data[:fin].splitlines():
            head = raw[:80]
            if b'"kind": "principal"' not in head and b'"kind": "tocado"' not in head:
                continue
            try:
                event = json.loads(raw.decode("utf-8", errors="replace"))
            except ValueError:
                continue
            datos = event.get("data") or {}
            if event.get("kind") == "principal":
                self.sesion.principal = datos
            elif datos.get("path"):
                self.sesion.tocados.append(str(datos["path"]))
        return ej.Sesion(dict(self.sesion.principal) if self.sesion.principal else None, list(self.sesion.tocados))


def sonando_publico(sesion: ej.Sesion) -> Optional[dict]:
    """Lo que el teléfono muestra arriba: título, artista, BPM y tonalidad del tema principal. Sin ruta."""
    p = sesion.principal
    if not p:
        return None
    try:
        bpm = round(float(p.get("bpm") or 0) * float(p.get("tempo") or 1.0), 1)
    except (TypeError, ValueError):
        bpm = 0.0
    clave = str(p.get("clave_resultante") or p.get("clave_original") or "")
    return {"titulo": str(p.get("title") or "")[:160], "artista": str(p.get("artist") or "")[:120],
            "bpm": bpm if 0 < bpm <= 400 else None, "clave": clave if re.fullmatch(r"(1[0-2]|[1-9])[AB]", clave) else None,
            "deck": str(p.get("deck") or "")[:2]}


class Oido:
    """Dice con palabras cómo suena un grupo de temas: las del vocabulario que más se acercan a ese grupo y no al resto."""

    def __init__(self, bib: Biblioteca, codificar: Callable[[Sequence[str]], object], vocabulario: Optional[Sequence[str]] = None):
        import numpy as np
        self.np = np
        self.bib = bib
        self.palabras = list(vocabulario) if vocabulario is not None else [
            p for grupo, lista in VOCABULARIO.items() if grupo not in GRUPOS_FUERA for p in lista]
        self.W = np.asarray(codificar(self.palabras), dtype=np.float64)
        self.media = np.asarray(bib.matrix, dtype=np.float64).mean(axis=0)

    def suena_a(self, indices: Sequence[int], k: int = SUENA_A) -> List[str]:
        np = self.np
        if not len(indices):
            return []
        centro = np.asarray(self.bib.matrix[list(indices)], dtype=np.float64).mean(axis=0)
        return [self.palabras[int(i)] for i in np.argsort(-(self.W @ (centro - self.media)), kind="stable")[:k]]


def titulo_artista(it: rc.Item) -> Tuple[str, str]:
    entry = it.entry
    title = (entry.get("TITLE") if entry is not None else "") or it.title
    artist = (entry.get("ARTIST") if entry is not None else "") or it.artist
    if not title:
        title = Path(str(it.rel).replace("\\", "/")).stem or it.uid[:12]
    return title[:160], (artist or "")[:120]         # holgado para el tope de 24 KB del Worker


def resumir(indices: Sequence[int], bib: Biblioteca, oido: Oido) -> dict:
    """Lo que el LLM necesita saber de una búsqueda: qué trajo, sin puntajes (no dicen si el concepto existe)."""
    import numpy as np
    items = [bib.items[i] for i in indices]
    generos = Counter(it.genre for it in items if it.genre)
    bpms = sorted(it.bpm for it in items if it.bpm)
    voces = [bib.vocal[it.uid] for it in items if it.uid in bib.vocal]
    temas = []
    for it in items[:EJEMPLOS]:
        title, artist = titulo_artista(it)
        temas.append(f"{artist} - {title}"[:140] if artist else title[:140])
    return {"generos": {g[:60]: n for g, n in generos.most_common(5)},
            "bpm": None if not bpms else [round(float(np.quantile(bpms, q))) for q in (0.1, 0.5, 0.9)],
            "voz": None if not voces else round(sum(voces) / len(voces), 2),
            "suena_a": oido.suena_a(indices), "temas": temas}


def formulario_de(tarea: Tarea, textos: Sequence[str], n: int = MAX_TRACKS) -> dict:
    """El formulario de la tarea (el de los presets, o vacío en las versiones 1 y 2) con estos textos."""
    form = dict(tarea.formulario) if tarea.formulario is not None else fm.vacio()
    form["texto"] = list(textos)[: fm.contrato()["texto"]["max"]]
    form["largo"] = min(int(form.get("largo") or n), n)
    return form


def sondear(tarea: Tarea, bib: Biblioteca, codificar: Callable, oido: Oido, sesion: Optional[ej.Sesion] = None) -> dict:
    """Una búsqueda por texto, separadas, para que el LLM pueda compararlas. Cada una pasa por los filtros del formulario."""
    busquedas: Dict[str, dict] = {}
    conjuntos = []
    pasan = len(bib.items)
    for ident, texto in tarea.textos:
        res = buscar(bib, formulario_de(tarea, [texto]), sesion, codificar)
        busquedas[ident] = resumir(res.indices, bib, oido)
        conjuntos.append(set(res.indices))
        pasan = res.pasos[-1]["quedan"]
    comunes = set.intersection(*conjuntos) if conjuntos else set()
    return {"busquedas": busquedas, "comunes": len(comunes), "pool": len(bib.items), "pasan": pasan}


def crear_lista(tarea: Tarea, bib: Biblioteca, codificar: Callable, n: int = MAX_TRACKS,
                sesion: Optional[ej.Sesion] = None) -> Tuple[dict, List[rc.Item]]:
    """La lista definitiva: el formulario con el promedio de los textos elegidos, y qué hizo el ejecutor."""
    form = formulario_de(tarea, [t for _, t in tarea.textos], n)
    res = buscar(bib, form, sesion, codificar)
    tracks, chosen = [], []
    for i, score in zip(res.indices, res.puntajes):
        it = bib.items[i]
        title, artist = titulo_artista(it)
        track = {"track_id": it.uid, "title": title, "artist": artist, "score": round(score, 4)}
        if it.bpm and 0 < it.bpm <= 400:
            track["bpm"] = round(float(it.bpm), 1)
        if it.key and it.key != "?":
            track["key"] = it.key
        if it.uid in bib.vocal:
            track["vocal"] = bib.vocal[it.uid]
        tracks.append(track)
        chosen.append(it)
    result = {"tracks": tracks, "pool": len(bib.items), "passed": res.pasos[-1]["quedan"],
              "pasos": res.pasos[:12], "aplicado": fm.describir(form)[:12]}
    if res.ignorado:
        result["ignorado"] = res.ignorado[:8]
    if res.referencia:
        result["referencia"] = res.referencia
    if res.candidatos:
        result["candidatos"] = res.candidatos
    return result, chosen


def carpeta_sugerencias() -> Path:
    return Path(sv.load_settings().get("dir") or sv.default_live_dir())


def nombre_playlist(nombre: str, ahora: Optional[dt.datetime] = None) -> str:
    """«Concepto 1109-2315 Fiesta con trompetas»: una playlist nueva por concepto, ordenadas por fecha."""
    limpio = re.sub(r"\s+", " ", re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", nombre)).strip(" .") or "Sin nombre"
    return f"Concepto {(ahora or dt.datetime.now()):%m%d-%H%M} {limpio[:60].strip(' .')}"


def escribir_nml(bib: Biblioteca, chosen: Sequence[rc.Item], nombre: str, carpeta: Optional[Path] = None) -> Optional[str]:
    """La playlist nueva, con las entradas copiadas de la colección. Devuelve su nombre; None sin colección."""
    entries = [it.entry for it in chosen if it.entry is not None]
    if bib.nml_version is None or not entries:
        return None
    titulo = nombre_playlist(nombre)
    write_atomic(nml_bytes(entries, int(time.time()), bib.nml_version, name=titulo), (carpeta or carpeta_sugerencias()) / f"{titulo}.nml")
    return titulo


# ---------------------------------------------------------------- el trabajo

class Agente:
    """Atiende trabajos reclamados: arma la biblioteca con el primero y devuelve el resultado al Worker."""

    def __init__(self, client: WorkerClient, say: Say, cargar: Optional[Callable[[], Biblioteca]] = None,
                 codificar: Optional[Callable[[Sequence[str]], object]] = None,
                 nml: Callable[[Biblioteca, Sequence[rc.Item], str], Optional[str]] = escribir_nml,
                 results_log: Optional[Path] = RESULTS_LOG, sleep: Callable[[float], None] = time.sleep,
                 sesion: Optional[Callable[[], ej.Sesion]] = None, estado: bool = False,
                 clock: Callable[[], float] = time.monotonic):
        self.client, self.say = client, say
        self.cargar = cargar or (lambda: cargar_biblioteca(say=say))
        self.codificar = codificar or CodificadorTexto()
        self.nml, self.results_log, self.sleep = nml, results_log, sleep
        self.sesion = sesion or (lambda: ej.Sesion())
        self.estado, self.clock = estado, clock          # estado: contarle al Worker qué suena y el catálogo
        self.bib: Optional[Biblioteca] = None
        self.oido: Optional[Oido] = None
        self._catalogo_enviado = False
        self._sonando: Tuple[Optional[str], float] = ("sin enviar", 0.0)

    def _preparar_bib(self) -> Biblioteca:
        if self.bib is None:
            self.bib = self.cargar()
            self.say("info", f"Biblioteca: {len(self.bib.items)} temas. {self.bib.descripcion}")
        return self.bib

    def _preparar(self) -> None:
        """Biblioteca y oído (CLAP): el sondeo los necesita; una lista de presets sin texto no carga CLAP."""
        self._preparar_bib()
        if self.oido is None:
            self.oido = Oido(self.bib, self.codificar)

    def para_worker(self) -> Optional[dict]:
        """Lo que va de paso en el próximo pedido de trabajo: el catálogo (una vez) y el tema que suena (si cambió o
        pasaron 5 min). None si no hay nada que contar."""
        if not self.estado:
            return None
        extra: dict = {}
        if not self._catalogo_enviado:
            try:
                bib = self._preparar_bib()
                extra["catalogo"] = fm.catalogo(bib.items, bib.vocal, dt.datetime.now().isoformat(timespec="seconds"))
            except (BibliotecaNoDisponible, OSError, ValueError) as exc:
                self.say("aviso", f"Sin catálogo para el teléfono ({type(exc).__name__})")
                self._catalogo_enviado = True
        sonando = sonando_publico(self.sesion())
        firma = None if sonando is None else json.dumps(sonando, sort_keys=True, ensure_ascii=False)
        previa, cuando = self._sonando
        if firma != previa or self.clock() - cuando > SONANDO_LATIDO_S:      # también sin tema: el ordenador sigue
            extra["sonando"] = sonando
        return extra or None

    def enviado(self, extra: Optional[dict]) -> None:
        """El Worker recibió extra: no se repite."""
        if not extra:
            return
        if "catalogo" in extra:
            self._catalogo_enviado = True
        if "sonando" in extra:
            s = extra["sonando"]
            self._sonando = (None if s is None else json.dumps(s, sort_keys=True, ensure_ascii=False), self.clock())

    def _procesar(self, job: dict) -> dict:
        tarea = parse_tarea(job.get("concept"))
        t0 = time.perf_counter()
        sesion = self.sesion()
        if tarea.fase == "sondeo":
            self._preparar()
            result = sondear(tarea, self.bib, self.codificar, self.oido, sesion)
            texto = f"sondeo de {len(tarea.textos)} textos"
        else:
            self._preparar_bib()
            result, chosen = crear_lista(tarea, self.bib, self.codificar, sesion=sesion)
            try:
                titulo = self.nml(self.bib, chosen, tarea.nombre)
                if titulo:
                    result["playlist"] = titulo
                else:
                    result["notes"] = ["No se escribió la playlist: no encontré la colección de Traktor."]
            except OSError as exc:
                self.say("aviso", f"No pude escribir la playlist de Traktor: {type(exc).__name__}: {exc}")
                result["notes"] = ["No se pudo escribir la playlist en Traktor."]
            texto = f"playlist «{result.get('playlist', '?')}», {len(result['tracks'])} temas"
        ms = round((time.perf_counter() - t0) * 1000)
        self.say("ok", f"«{tarea.nombre}»: {texto} ({ms} ms)")
        self._anotar(job, tarea, result, ms)
        return result

    def _anotar(self, job: dict, tarea: Tarea, result: dict, ms: int) -> None:
        if self.results_log is None:
            return
        try:
            self.results_log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.results_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": dt.datetime.now().isoformat(timespec="seconds"), "job": job.get("id"), "fase": tarea.fase,
                                    "concepto": job.get("concept"), "resultado": result, "ms": ms}, ensure_ascii=False) + "\n")
        except OSError as exc:
            self.say("aviso", f"No pude anotar el resultado: {type(exc).__name__}")

    def atender(self, job: dict) -> None:
        job_id, attempts = job["id"], job["attempts"]
        try:
            result = self._procesar(job)
        except (TareaInvalida, BibliotecaNoDisponible) as exc:
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
          job_id: Optional[str] = None, una_vez: bool = False, clock: Callable[[], float] = time.monotonic,
          activo_s: float = ACTIVO_S, poll_activo_s: float = POLL_ACTIVO_S) -> int:
    """Consulta la cola hasta que stop se active. Devuelve el código de salida: 0 bien (también al liberar memoria),
    2 token rechazado. Con --una-vez y un job_id, sigue ese concepto hasta que no quede nada que hacer con él."""
    last_job = clock()
    fallos = 0
    atendidos = 0
    while not stop.is_set():
        try:
            extra = agente.para_worker()
            job = agente.client.claim(job_id, extra) if extra else agente.client.claim(job_id)
            agente.enviado(extra)
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
            if una_vez and (job_id is None or atendidos):
                agente.say("info", "No hay trabajos en cola" if not atendidos else "Listo")
                return 0
            if una_vez and clock() - last_job > activo_s:
                agente.say("info", "Ese trabajo no está en cola")
                return 0
            if idle_exit_s is not None and agente.oido is not None and clock() - last_job > idle_exit_s:
                agente.say("info", "Sin conceptos hace rato: libero la memoria de CLAP")
                return 0
            stop.wait(poll_activo_s if clock() - last_job < activo_s and (atendidos or una_vez) else poll_s)
            continue
        agente.say("info", f"Trabajo {job['id']} ({(job.get('concept') or {}).get('fase', 'final')}, intento {job['attempts']})")
        agente.atender(job)
        last_job = clock()
        atendidos += 1
        if una_vez and job_id is None:
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
    agente = Agente(WorkerClient(ajustes.url, ajustes.token), console_say, sesion=SesionVivo().actual, estado=True)
    console_say("info", f"Consultando {ajustes.url} cada {args.cada:g} s")
    try:
        return bucle(agente, stop, poll_s=args.cada, idle_exit_s=args.salir_sin_uso, job_id=args.job, una_vez=args.una_vez)
    except KeyboardInterrupt:
        return 0


def cmd_probar(args: argparse.Namespace) -> int:
    """Unas frases contra la biblioteca real, sin red y sin escribir nada salvo con --nml."""
    tarea = Tarea("final", args.nombre, tuple((chr(65 + i), f) for i, f in enumerate(args.frase)))
    t0 = time.perf_counter()
    bib = cargar_biblioteca(say=console_say)
    t1 = time.perf_counter()
    codificar = CodificadorTexto()
    oido = Oido(bib, codificar)
    t2 = time.perf_counter()
    sondeo = sondear(Tarea("sondeo", args.nombre, tarea.textos), bib, codificar, oido)
    result, chosen = crear_lista(tarea, bib, codificar, n=args.n)
    t3 = time.perf_counter()
    print(f"[INFO] {bib.descripcion}")
    print(f"[INFO] biblioteca {t1 - t0:.1f} s; CLAP y vocabulario {t2 - t1:.1f} s; búsquedas {1000 * (t3 - t2):.0f} ms")
    for ident, texto in tarea.textos:
        b = sondeo["busquedas"][ident]
        print(f"[{ident}] {texto}\n    géneros {b['generos']}  bpm {b['bpm']}  voz {b['voz']}\n    suena a {b['suena_a']}")
    for i, t in enumerate(result["tracks"], 1):
        print(f"{i:>3}  {t['score']:.3f}  {t.get('bpm', 0):>5.0f}  {t.get('key', '?'):>3}  {'V' if t.get('vocal') else ' '}  {t['artist'][:30]} - {t['title'][:60]}")
    if args.nml:
        print(f"[OK] Playlist: {escribir_nml(bib, chosen, args.nombre)}")
    return 0


def cmd_formulario(args: argparse.Namespace) -> int:
    """Un formulario contra la biblioteca real, sin red: qué pasó cada filtro y la lista. CLAP solo si hay texto."""
    raw = json.loads(Path(args.formulario).read_text(encoding="utf-8") if Path(args.formulario).suffix == ".json"
                     and Path(args.formulario).exists() else args.formulario)
    bib = cargar_biblioteca(say=console_say)
    form, problemas = fm.validar(raw, generos_validos={it.genre for it in bib.items if it.genre})
    for p in problemas:
        print(f"[AVISO] {p}")
    sesion = SesionVivo().actual() if args.sesion else ej.Sesion()
    if args.sonando:
        principal = next((it for it in bib.items if ej.puntaje_tema(args.sonando, it) >= ej.TEMA_CLARO), None)
        if principal is None:
            print(f"[ERROR] No encontré «{args.sonando}» en la biblioteca")
            return 1
        sesion = ej.Sesion({"path": str(principal.path or principal.rel), "bpm": principal.bpm, "tempo": 1.0,
                            "key": principal.key, "title": principal.title, "artist": principal.artist}, sesion.tocados)
    t0 = time.perf_counter()
    textos = tuple((chr(65 + i), t) for i, t in enumerate(form["texto"]))
    result, _ = crear_lista(Tarea("final", "prueba", textos, form), bib, CodificadorTexto(), sesion=sesion)
    print(f"[INFO] {1000 * (time.perf_counter() - t0):.0f} ms; entendido: {' · '.join(result['aplicado'])}")
    for p in result["pasos"]:
        print(f"   {p['quedan']:>5}  {p['paso']}")
    for i in result.get("ignorado", []):
        print(f"[IGNORADO] {i['casilla']}: {i['motivo']}")
    for c in result.get("candidatos", []):
        print(f"[CANDIDATO] {c['artista']} - {c['titulo']}")
    for i, t in enumerate(result["tracks"], 1):
        print(f"{i:>3}  {t['score']:.3f}  {t.get('bpm', 0):>5.0f}  {t.get('key', '?'):>3}  {'V' if t.get('vocal') else ' '}  {t['artist'][:30]} - {t['title'][:60]}")
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
    ap = argparse.ArgumentParser(description="El ordenador en el circuito de conceptos escritos en el teléfono")
    sub = ap.add_subparsers(dest="accion", required=True)
    correr = sub.add_parser("correr", help="Consultar la cola en primer plano")
    correr.add_argument("--cada", type=float, default=POLL_S, help="Segundos entre consultas")
    correr.add_argument("--una-vez", action="store_true", help="Sin --job: atender un trabajo y salir. Con --job: seguir ese concepto hasta el final")
    correr.add_argument("--job", default=None, help="Reclamar solo este trabajo (pruebas: no consume otros)")
    correr.add_argument("--salir-sin-uso", type=float, default=None, help="Con CLAP cargado y sin trabajos por estos segundos, salir")
    correr.set_defaults(func=cmd_correr)
    probar = sub.add_parser("probar", help="Unas frases contra la biblioteca, sin red")
    probar.add_argument("--frase", action="append", required=True, help="Texto en inglés (repetible: se promedian)")
    probar.add_argument("--nombre", default="prueba")
    probar.add_argument("--n", type=int, default=MAX_TRACKS)
    probar.add_argument("--nml", action="store_true", help="Escribir también la playlist de Traktor")
    probar.set_defaults(func=cmd_probar)
    formulario = sub.add_parser("formulario", help="Un formulario del teléfono contra la biblioteca, sin red")
    formulario.add_argument("formulario", help="JSON del formulario, o un archivo .json")
    formulario.add_argument("--sesion", action="store_true", help="Leer lo que suena del registro del vivo")
    formulario.add_argument("--sonando", default=None, help="Simular que suena este tema (texto que lo nombre)")
    formulario.set_defaults(func=cmd_formulario)
    for name, func in (("servicio", cmd_servicio), ("instalar", cmd_instalar), ("quitar", cmd_quitar), ("estado", cmd_estado)):
        sub.add_parser(name).set_defaults(func=func)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
