"""
PURPOSE: A0 de la línea A (docs/plans/linea_a_protocolo_a0_a1.md, fijado en el commit a6fc0f3): las vistas de todos
         los audios de DJSets v1 (CLAP, MAEST capa 7 y los 400 estilos Discogs, 4 segmentos de 30 s) que V7 no cubre,
         leídas de los shards del dataset (djsets_v1/audio/), y las vistas de los 33 321 temas juntadas por track_id.
           correr     un trabajo de gpuq: por cada shard pedido sube <destino>/<fuente>/<shard>.npz con recording_id,
                      segundos, clap (n, 4, 512), l7 (n, 4, 768), styles (n, 4, 400) y meta; en YouTube además
                      comienzo_s, duracion_s y, para el control (md5(recording_id) % 10 == 0), control_id con
                      clap_orig, l7_orig y styles_orig. Se retoma: salta los shards ya subidos y sigue desde el
                      parcial (<shard>.parcial.npz, cada 96 temas, nunca más de 10 min sin guardar).
           planear    los shards que faltan (sin V7 y sin salida en el destino), en trabajos de una hora o menos, con
                      los comandos de cola.py.
           juntar     (CPU, esta PC) V7 + A0 -> <salida>/: indice.csv, clap.npy, l7.npy, styles.npy (33 321, 4, d) en
                      el orden de tracks.csv, youtube_control.npz y LEEME.json; comprueba cobertura, finitos y
                      alineación (umbrales 1 a 3 y 6 del protocolo).
           verificar  umbral 4 (una salida de A0 frente a la de V7 del mismo shard) y umbral 5 (los 166 de E15).
         Ventana: Beatport, la de V7 y E15 (el preview entero en 4 segmentos de 30 s); YouTube, la que tendría su
         preview: comienzo 0,40 × D, largo mín(120 s, D − comienzo), recodificada a MP3 CBR de 96 kbps con LAME en
         estéreo (como codec_clap.mp3_ida_y_vuelta, E17) antes de los modelos; el control lleva también la vista con
         el códec original. Modelos: la clase Modelos de V7 (vistas_previews.py), el mismo código que E15.
           python tools/dj_sets/vistas_djsets.py correr --shards beatport_060,youtube_000 [--destino djsets_v1_vistas/a0]
                                                        [--max-temas N] [--device auto]
           python tools/dj_sets/vistas_djsets.py planear [--destino djsets_v1_vistas/a0]
           python tools/dj_sets/vistas_djsets.py juntar [--destino djsets_v1_vistas/a0] [--cache DIR]
           python tools/dj_sets/vistas_djsets.py verificar --a0 <shard.npz> --v7 <tanda.npz> | --vistas DIR --e15 DIR
CHANGELOG:
  - 2026-11-13: Creación (A0).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from tools.dj_sets.vistas_previews import N_SEG, SEG_S, cortar, leer_mono  # noqa: E402

DATASET = "djsets_v1"
TRACKS = f"{DATASET}/metadata/tracks.csv"
SHARDS = f"{DATASET}/metadata/shards.csv"
RAIZ = "djsets_v1_vistas"
DESTINO = f"{RAIZ}/a0"
SALIDA = f"{RAIZ}/vistas"
V7 = "v7_vistas"
DIMS = (("clap", 512), ("l7", 768), ("styles", 400))
INICIO_FRAC = 0.40           # Beatport empieza el preview en el 40 % del tema (0,400 en 30 254 previews)
LARGO_S = 120.0              # y dura 120 s (28 765 de 30 255)
KBPS = 96                    # los previews son MP3 de 96 kbps
SR_YT = 44100
CONTROL_MOD = 10             # control: md5(recording_id) % 10 == 0, unas 306 grabaciones de YouTube
PARCIAL_CADA = 96            # unos 3 min en Kaggle: nunca más de 10 min sin guardar (plan, 2.5)
MAX_RETARDO = 4096           # = codec_clap.MAX_RETARDO
S_POR_TEMA = {"beatport": 2.0, "youtube": 3.0}   # Kaggle, V7 medido; YouTube estimado (lo corrige el trabajo corto)
MIN_POR_TRABAJO = 55.0
HILOS = min(4, os.cpu_count() or 2)   # preparar el audio de YouTube cuesta 2,5 a 3 s por tema en CPU (prueba del 11-13)


# ---------------------------------------------------------------------------
# Población, ventana y audio
# ---------------------------------------------------------------------------

def leer_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def stem(shard: str) -> str:
    """audio/beatport/beatport_060.tar -> beatport_060."""
    return Path(shard).name[:-4] if shard.endswith(".tar") else Path(shard).name


def fuente_de(shard: str) -> str:
    return stem(shard).split("_")[0]


def poblacion(tracks: Sequence[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """{shard: [{member, recording_id, fuente}]}: una entrada por grabación con audio, en el orden de tracks.csv."""
    out: Dict[str, List[Dict[str, str]]] = {}
    vistos = set()
    for t in tracks:
        r = t.get("recording_id", "")
        if t.get("has_audio") != "1" or not r or r in vistos:
            continue
        vistos.add(r)
        out.setdefault(t["audio_shard"], []).append({"member": t["audio_member"], "recording_id": r,
                                                     "fuente": t["audio_source"]})
    return out


def ventana(duracion_s: float, frac: float = INICIO_FRAC, largo_s: float = LARGO_S) -> Tuple[float, float]:
    """(comienzo, largo) en segundos de la ventana de un preview dentro de un tema de duracion_s."""
    comienzo = frac * duracion_s
    return comienzo, max(0.0, min(largo_s, duracion_s - comienzo))


def es_control(recording_id: str, mod: int = CONTROL_MOD) -> bool:
    return int(hashlib.md5(recording_id.encode("utf-8")).hexdigest(), 16) % mod == 0


def n_seg_validos(segundos: float) -> int:
    """Segmentos de 30 s con 15 s de audio o más (el 0 siempre cuenta): protocolo A1, «el vector de un tema»."""
    if segundos < 15.0:
        return 1
    return int(max(1, min(N_SEG, np.floor((segundos - 15.0) / SEG_S) + 1)))


def ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("falta ffmpeg (hace falta para el audio de YouTube)")
    return exe


def decodificar_estereo(path: Path, sr: int = SR_YT) -> np.ndarray:
    """(n, 2) float32 a sr, con ffmpeg."""
    raw = subprocess.run([ffmpeg(), "-v", "error", "-i", str(path), "-ac", "2", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<f4").reshape(-1, 2).copy()


def retardo(ref: np.ndarray, y: np.ndarray, sr: int, max_lag: int = MAX_RETARDO) -> int:
    """= codec_clap.retardo (copiada para que el trabajo no importe la evaluación entera; un test las compara)."""
    a = int(5 * sr)
    L = int(2 * sr)
    r = ref[a:a + L]
    w = y[max(0, a - max_lag):a + L + max_lag]
    if len(r) < L or np.abs(r).sum() == 0:
        return 0
    c = np.correlate(w, r, mode="valid")
    return int(np.argmax(c)) - (a - max(0, a - max_lag))


def mp3_ida_y_vuelta(x: np.ndarray, sr: int, kbps: int = KBPS) -> np.ndarray:
    """= codec_clap.mp3_ida_y_vuelta: el audio estéreo (n, 2) a MP3 CBR con LAME y de vuelta, alineado y del mismo largo."""
    enc = subprocess.run([ffmpeg(), "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "2", "-i", "-",
                          "-c:a", "libmp3lame", "-b:a", f"{kbps}k", "-f", "mp3", "-"],
                         input=np.ascontiguousarray(x, dtype="<f4").tobytes(), capture_output=True, check=True).stdout
    dec = subprocess.run([ffmpeg(), "-v", "error", "-f", "mp3", "-i", "-", "-ac", "2", "-ar", str(sr), "-f", "f32le", "-"],
                         input=enc, capture_output=True, check=True).stdout
    y = np.frombuffer(dec, dtype="<f4").reshape(-1, 2)
    lag = retardo(x.mean(axis=1), y.mean(axis=1), sr)
    y = y[lag:] if lag >= 0 else np.pad(y, ((-lag, 0), (0, 0)))
    n = len(x)
    return (np.pad(y, ((0, n - len(y)), (0, 0))) if len(y) < n else y[:n]).astype(np.float32)


def audio_youtube(path: Path, control: bool, sr: int = SR_YT) -> Dict[str, object]:
    """La ventana del preview de un tema entero: mono recodificado a 96 kbps y, si es de control, mono original."""
    st = decodificar_estereo(path, sr)
    dur = len(st) / sr
    comienzo, largo = ventana(dur)
    a = int(round(comienzo * sr))
    w = st[a:a + int(round(largo * sr))]
    if not len(w):
        raise ValueError("ventana vacía")
    return {"x": mp3_ida_y_vuelta(w, sr).mean(axis=1), "sr": sr, "comienzo_s": comienzo, "duracion_s": dur,
            "segundos": len(w) / sr, "orig": w.mean(axis=1).astype(np.float32) if control else None}


def audio_beatport(path: Path) -> Dict[str, object]:
    x, sr = leer_mono(path)
    return {"x": x, "sr": sr, "segundos": len(x) / sr, "orig": None}


# ---------------------------------------------------------------------------
# Un shard
# ---------------------------------------------------------------------------

def _vacio(n: int = 0) -> Dict[str, np.ndarray]:
    return {k: np.zeros((n, N_SEG, d), np.float32) for k, d in DIMS}


def _cat(partes: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {k: np.concatenate([p[k] for p in partes]) if partes else _vacio()[k] for k, _ in DIMS}


class Estado:
    """Lo hecho en un shard: filas principales, control y fallidos (lo que guarda el parcial)."""

    def __init__(self) -> None:
        self.ids: List[str] = []
        self.segundos: List[float] = []
        self.comienzo: List[float] = []
        self.duracion: List[float] = []
        self.emb: List[Dict[str, np.ndarray]] = []
        self.ctrl_ids: List[str] = []
        self.ctrl: List[Dict[str, np.ndarray]] = []
        self.fallidos: List[Dict[str, str]] = []

    def listos(self) -> set:
        return set(self.ids) | {f["recording_id"] for f in self.fallidos}

    def arrays(self) -> Dict[str, np.ndarray]:
        e, c = _cat(self.emb), _cat(self.ctrl)
        out = {"recording_id": np.asarray(self.ids, dtype=str), "segundos": np.asarray(self.segundos, np.float32),
               "comienzo_s": np.asarray(self.comienzo, np.float32), "duracion_s": np.asarray(self.duracion, np.float32),
               "control_id": np.asarray(self.ctrl_ids, dtype=str),
               "fallidos": np.asarray(json.dumps(self.fallidos, ensure_ascii=False))}
        out.update(e)
        out.update({f"{k}_orig": v for k, v in c.items()})
        return out

    @classmethod
    def desde(cls, z) -> "Estado":
        s = cls()
        s.ids = [str(x) for x in z["recording_id"]]
        s.segundos = [float(x) for x in z["segundos"]]
        s.comienzo = [float(x) for x in z["comienzo_s"]]
        s.duracion = [float(x) for x in z["duracion_s"]]
        s.emb = [{k: np.asarray(z[k]) for k, _ in DIMS}]
        s.ctrl_ids = [str(x) for x in z["control_id"]]
        s.ctrl = [{k: np.asarray(z[f"{k}_orig"]) for k, _ in DIMS}]
        s.fallidos = json.loads(str(z["fallidos"]))
        return s


def procesar_shard(remote, shard: str, miembros: Sequence[Dict[str, str]], modelos, scratch: Path,
                   max_temas: Optional[int] = None, lote: int = 16, commit: str = "", destino: str = DESTINO,
                   cada: int = PARCIAL_CADA, leer_yt: Callable = audio_youtube,
                   leer_bp: Callable = audio_beatport, hilos: int = HILOS) -> Dict[str, object]:
    """Baja el shard, embebe cada grabación de la población y sube <destino>/<fuente>/<stem>.npz. El audio se prepara
    en `hilos` hilos mientras la GPU procesa el lote anterior; el orden de las filas es el del .tar. Cada `cada` temas
    sube un parcial; si la máquina se corta, el intento siguiente sigue desde ahí."""
    t0 = time.perf_counter()
    fuente, nombre = fuente_de(shard), stem(shard)
    miembros = list(miembros)[:max_temas]
    base = f"{destino}/{fuente}/{nombre}"
    est = Estado()
    if remote.exists(f"{base}.parcial.npz"):
        previo = scratch / f"{nombre}.parcial_previo.npz"
        remote.get(f"{base}.parcial.npz", previo)
        with np.load(previo) as z:
            est = Estado.desde(z)
        previo.unlink(missing_ok=True)
        print(f"    {nombre}: sigo desde el parcial, {len(est.ids)} hechos", flush=True)
    listos = est.listos()
    por_miembro = {m["member"]: m for m in miembros if m["recording_id"] not in listos}
    tar_local = scratch / f"{nombre}.tar"
    remote.get(f"{DATASET}/{shard}", tar_local)
    t_bajada = time.perf_counter() - t0
    pendientes: List[Tuple[str, Dict[str, object]]] = []
    ultimo = [len(est.ids)]
    t_audio = [0.0]

    def guardar_parcial() -> None:
        loc = scratch / f"{nombre}.parcial.npz"
        np.savez(loc, **est.arrays())
        remote.put(loc, f"{base}.parcial.npz")
        loc.unlink(missing_ok=True)
        ultimo[0] = len(est.ids)

    def vaciar() -> None:
        if not pendientes:
            return
        items = [(a["x"], a["sr"]) for _, a in pendientes]
        ctrl = [(r, a) for r, a in pendientes if a["orig"] is not None]
        e = modelos.embed(items + [(a["orig"], a["sr"]) for _, a in ctrl])
        n = len(items)
        est.emb.append({k: e[k][:n] for k, _ in DIMS})
        if ctrl:
            est.ctrl.append({k: e[k][n:] for k, _ in DIMS})
            est.ctrl_ids.extend(r for r, _ in ctrl)
        for r, a in pendientes:
            est.ids.append(r)
            est.segundos.append(float(a["segundos"]))
            est.comienzo.append(float(a.get("comienzo_s", np.nan)))
            est.duracion.append(float(a.get("duracion_s", np.nan)))
        pendientes.clear()
        if cada and len(est.ids) - ultimo[0] >= cada:
            guardar_parcial()

    def preparar(tmp: str, i: int, nombre_m: str, datos: bytes, fila: Dict[str, str]) -> Tuple[Dict[str, object], float]:
        """Audio listo para los modelos (en un hilo: ffmpeg corre en su proceso y suelta el GIL)."""
        ta = time.perf_counter()
        arch = Path(tmp) / f"{i:05d}_{Path(nombre_m).name}"
        try:
            arch.write_bytes(datos)
            r = fila["recording_id"]
            a = leer_yt(arch, es_control(r)) if fila["fuente"] == "youtube" else leer_bp(arch)
            if not len(a["x"]) or not np.isfinite(a["x"]).all():
                raise ValueError("audio vacío o no finito")
            return a, time.perf_counter() - ta
        finally:
            arch.unlink(missing_ok=True)

    cola: deque = deque()

    def recoger(hasta: int) -> None:
        """Toma los resultados en el orden del .tar hasta que queden `hasta` en vuelo; la GPU corre mientras los
        hilos preparan los siguientes."""
        while len(cola) > hasta:
            r, fut = cola.popleft()
            try:
                a, dt = fut.result()
                t_audio[0] += dt
                pendientes.append((r, a))
            except Exception as exc:  # noqa: BLE001
                est.fallidos.append({"recording_id": r, "error": f"{type(exc).__name__}: {exc}"[:200]})
            if len(pendientes) >= lote:
                vaciar()
                print(f"    {nombre}: {len(est.ids)}/{len(miembros)} ({time.perf_counter() - t0:.0f} s)", flush=True)

    try:
        with tarfile.open(tar_local) as tar, tempfile.TemporaryDirectory(dir=scratch) as tmp, \
                ThreadPoolExecutor(max_workers=max(1, hilos)) as ex:
            for i, m in enumerate(tar):
                fila = por_miembro.get(m.name)
                if fila is None:
                    continue
                cola.append((fila["recording_id"], ex.submit(preparar, tmp, i, m.name, tar.extractfile(m).read(), fila)))
                recoger(2 * max(1, hilos))
            recoger(0)
            vaciar()
    finally:
        tar_local.unlink(missing_ok=True)
    faltan = [m["recording_id"] for m in miembros if m["recording_id"] not in est.listos()]
    est.fallidos += [{"recording_id": r, "error": "no está en el .tar"} for r in faltan]
    arr = est.arrays()
    if not all(np.isfinite(arr[k]).all() for k, _ in DIMS) or not all(np.isfinite(arr[f"{k}_orig"]).all() for k, _ in DIMS):
        raise ValueError(f"{nombre}: embeddings no finitos")
    seg = time.perf_counter() - t0
    meta = {"shard": shard, "fuente": fuente, "poblacion": len(miembros), "hechos": len(est.ids),
            "control": len(est.ctrl_ids), "fallidos": est.fallidos, "commit": commit,
            "ventana": ("V4: el preview entero en 4 segmentos de 30 s (V7, E15)" if fuente == "beatport" else
                        f"comienzo {INICIO_FRAC:.2f} × D, largo mín({LARGO_S:.0f} s, D − comienzo), 4 segmentos de "
                        f"30 s; recodificada a MP3 CBR {KBPS} kbps (LAME, estéreo); control con el códec original"),
            "modelos": getattr(modelos, "info", {}), "segundos": round(seg, 1), "segundos_bajada": round(t_bajada, 1),
            "segundos_audio": round(t_audio[0], 1), "gpuq_job": os.environ.get("GPUQ_JOB_ID"),
            "plataforma": os.environ.get("GPUQ_PLATFORM")}
    npz = scratch / f"{nombre}.npz"
    arr.pop("fallidos")
    np.savez_compressed(npz, meta=np.asarray(json.dumps(meta, ensure_ascii=False)), **arr)
    remote.put(npz, f"{base}.npz")
    npz.unlink(missing_ok=True)
    remote.delete(f"{base}.parcial.npz")
    return meta


# ---------------------------------------------------------------------------
# correr y planear
# ---------------------------------------------------------------------------

def _tablas(remote, scratch: Path) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    tr, sh = scratch / "tracks.csv", scratch / "shards.csv"
    remote.get(TRACKS, tr)
    remote.get(SHARDS, sh)
    return leer_csv(tr), leer_csv(sh)


def resolver(pedidos: Sequence[str], shards: Sequence[Dict[str, str]]) -> List[str]:
    """beatport_060 o audio/beatport/beatport_060.tar -> la ruta del shard en shards.csv."""
    por_stem = {stem(s["shard"]): s["shard"] for s in shards}
    out = []
    for p in pedidos:
        s = por_stem.get(stem(p.strip()))
        if not s:
            raise SystemExit(f"shard desconocido: {p}")
        out.append(s)
    return out


def correr(pedidos: Sequence[str], max_temas: Optional[int], device: str, hf_cache: Optional[str], lote: int,
           destino: str, hilos: int = HILOS) -> int:
    from src.v4.common.remote_storage import open_remote
    from tools.dj_sets.vistas_previews import Modelos
    remote = open_remote()
    scratch = Path(os.environ.get("GPUQ_SCRATCH") or tempfile.gettempdir()) / "a0"
    scratch.mkdir(parents=True, exist_ok=True)
    tracks, shards = _tablas(remote, scratch)
    pob = poblacion(tracks)
    pend = []
    for s in resolver(pedidos, shards):
        hechos = remote.ls(f"{destino}/{fuente_de(s)}")
        if f"{stem(s)}.npz" not in hechos:
            pend.append(s)
    if any(fuente_de(s) == "youtube" for s in pend):
        ff = ffmpeg()
        enc = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        if "libmp3lame" not in enc:
            raise RuntimeError("el ffmpeg de esta máquina no tiene libmp3lame")
        print(f"ffmpeg: {ff}", flush=True)
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(pend)} shards por hacer de {len(pedidos)} pedidos; dispositivo {device}; {hilos} hilos de audio; "
          f"{os.cpu_count()} CPU", flush=True)
    if not pend:
        return 0
    modelos = Modelos(hf_cache, device)
    print(f"modelos listos: {json.dumps(modelos.info, ensure_ascii=False)[:300]}", flush=True)
    commit = os.environ.get("GPUQ_CODE_SHA") or ""
    t0, total = time.perf_counter(), 0
    for s in pend:
        meta = procesar_shard(remote, s, pob.get(s, []), modelos, scratch, max_temas, lote, commit, destino, hilos=hilos)
        total += meta["hechos"]
        el = time.perf_counter() - t0
        print(f"  {stem(s)}: {meta['hechos']}/{meta['poblacion']} (+{meta['control']} de control), "
              f"{len(meta['fallidos'])} fallidos, {meta['segundos']:.0f} s (bajada {meta['segundos_bajada']:.0f} s, "
              f"audio {meta['segundos_audio']:.0f} s) | {total} en {el / 60:.1f} min, {el / max(total, 1):.2f} s por tema",
              flush=True)
    print(f"[OK] {total} temas en {len(pend)} shards", flush=True)
    return 0


def pendientes(shards: Sequence[Dict[str, str]], v7_hechos: set, destino_hechos: Dict[str, set]) -> List[str]:
    """Shards del dataset sin salida de V7 (por su tanda de origen) ni salida en el destino."""
    out = []
    for s in shards:
        origen = s["origin"]
        cubierto = origen.startswith("beatport_previews/") and origen[len("beatport_previews/"):-4] + ".npz" in v7_hechos
        if not cubierto and f"{stem(s['shard'])}.npz" not in destino_hechos.get(s["source"], set()):
            out.append(s["shard"])
    return out


def agrupar(shards: Sequence[str], pob: Dict[str, List[dict]], s_por_tema: Dict[str, float] = S_POR_TEMA,
            minutos: float = MIN_POR_TRABAJO) -> List[List[str]]:
    """Trabajos de `minutos` o menos de trabajo útil, sin mezclar fuentes, en el orden de los shards."""
    trabajos: List[List[str]] = []
    for fuente in ("beatport", "youtube"):
        actual, t = [], 0.0
        for s in shards:
            if fuente_de(s) != fuente:
                continue
            n = len(pob.get(s, []))
            extra = n * (1 + 1 / CONTROL_MOD) if fuente == "youtube" else n
            dt = extra * s_por_tema[fuente] / 60
            if actual and t + dt > minutos:
                trabajos.append(actual)
                actual, t = [], 0.0
            actual.append(s)
            t += dt
        if actual:
            trabajos.append(actual)
    return trabajos


def planear(destino: str, primero: int) -> int:
    from src.v4.common.remote_storage import open_remote
    remote = open_remote()
    with tempfile.TemporaryDirectory() as d:
        tracks, shards = _tablas(remote, Path(d))
    pob = poblacion(tracks)
    v7 = {f"{c}/{n}" for c in {s["origin"].split("/")[1] for s in shards if s["origin"].startswith("beatport_previews/")}
          for n in remote.ls(f"{V7}/{c}")}
    hechos = {f: set(remote.ls(f"{destino}/{f}")) for f in ("beatport", "youtube")}
    pend = pendientes(shards, v7, hechos)
    trabajos = agrupar(pend, pob)
    n = sum(len(pob.get(s, [])) for s in pend)
    print(f"{len(pend)} shards sin vistas ({n} grabaciones) en {len(trabajos)} trabajos de {MIN_POR_TRABAJO:.0f} min o menos")
    for i, t in enumerate(trabajos):
        lista = ",".join(stem(s) for s in t)
        print(f'python tools/gpuq/cola.py add traktor-a0-{primero + i} --cmd "python tools/dj_sets/vistas_djsets.py '
              f'correr --shards {lista} --destino {destino}" --gpu T4 --platform auto --no-setup --timeout-h 2')
    return 0


# ---------------------------------------------------------------------------
# juntar y verificar
# ---------------------------------------------------------------------------

def _bajar(remote, rel: str, cache: Path) -> Path:
    """Copia local de rel en cache (se reusa si el tamaño coincide con el del remoto)."""
    loc = cache / rel
    st = remote.stat(rel)
    if st is None:
        raise FileNotFoundError(rel)
    if not (loc.exists() and loc.stat().st_size == int(st["size"])):
        loc.parent.mkdir(parents=True, exist_ok=True)
        remote.get(rel, loc)
    return loc


def cargar_v7(archivos: Sequence[Path]) -> Dict[str, Tuple[Dict[str, np.ndarray], float, str]]:
    """{recording_id: (vistas (4, d), segundos, origen)} de las salidas de V7 (bp_id -> bp_<id>)."""
    out = {}
    for f in archivos:
        with np.load(f) as z:
            A = {k: np.asarray(z[k]) for k, _ in DIMS}
            seg = np.asarray(z["segundos"])
            job = json.loads(str(z["meta"])).get("gpuq_job") or ""
            for i, b in enumerate(z["bp_id"]):
                out.setdefault(f"bp_{b}", ({k: A[k][i] for k in A}, float(seg[i]), f"V7 {job} {f.parent.name}/{f.stem}"))
    return out


def cargar_a0(archivos: Sequence[Path]) -> Tuple[Dict[str, Tuple[Dict[str, np.ndarray], float, str]], Dict[str, dict], List[dict]]:
    """Filas principales, control de YouTube y fallidos de las salidas de A0."""
    out, ctrl, fallidos = {}, {}, []
    for f in archivos:
        with np.load(f) as z:
            meta = json.loads(str(z["meta"]))
            A = {k: np.asarray(z[k]) for k, _ in DIMS}
            seg, com, dur = (np.asarray(z[c]) for c in ("segundos", "comienzo_s", "duracion_s"))
            for i, r in enumerate(z["recording_id"]):
                out[str(r)] = ({k: A[k][i] for k in A}, float(seg[i]), f"A0 {meta.get('gpuq_job') or ''} {f.stem}")
                if meta["fuente"] == "youtube":
                    ctrl.setdefault(str(r), {})["ventana"] = (float(com[i]), float(dur[i]), float(seg[i]))
            O = {k: np.asarray(z[f"{k}_orig"]) for k, _ in DIMS}
            for i, r in enumerate(z["control_id"]):
                ctrl.setdefault(str(r), {})["orig"] = {k: O[k][i] for k in O}
            fallidos += [dict(x, shard=meta["shard"]) for x in meta["fallidos"]]
    return out, ctrl, fallidos


def armar_vistas(tracks: Sequence[Dict[str, str]], filas: Dict[str, Tuple[Dict[str, np.ndarray], float, str]]
                 ) -> Tuple[List[Dict[str, str]], Dict[str, np.ndarray], List[str]]:
    """Índice y arrays (n, 4, d) en el orden de tracks.csv (temas con audio); temas sin vistas aparte (fila en cero)."""
    aud = [t for t in tracks if t.get("has_audio") == "1"]
    arr = _vacio(len(aud))
    indice, sin = [], []
    for i, t in enumerate(aud):
        r = t["recording_id"]
        f = filas.get(r)
        if f is None:
            sin.append(t["track_id"])
            indice.append({"track_id": t["track_id"], "recording_id": r, "fuente": t["audio_source"], "origen": "",
                           "segundos": "", "n_seg_validos": "0"})
            continue
        v, seg, origen = f
        for k, _ in DIMS:
            arr[k][i] = v[k]
        indice.append({"track_id": t["track_id"], "recording_id": r, "fuente": t["audio_source"], "origen": origen,
                       "segundos": f"{seg:.3f}", "n_seg_validos": str(n_seg_validos(seg))})
    return indice, arr, sin


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def juntar(remote, destino: str, salida: str, cache: Path, subir: bool = True) -> Dict[str, object]:
    cache.mkdir(parents=True, exist_ok=True)
    tracks = leer_csv(_bajar(remote, TRACKS, cache))
    shards = leer_csv(_bajar(remote, SHARDS, cache))
    corridas = sorted({s["origin"].split("/")[1] for s in shards if s["origin"].startswith("beatport_previews/")})
    f_v7 = [_bajar(remote, f"{V7}/{c}/{n}", cache) for c in corridas for n in sorted(remote.ls(f"{V7}/{c}"))
            if n.endswith(".npz") and not n.endswith(".parcial.npz")]
    f_a0 = [_bajar(remote, f"{destino}/{fu}/{n}", cache) for fu in ("beatport", "youtube")
            for n in sorted(remote.ls(f"{destino}/{fu}")) if n.endswith(".npz") and not n.endswith(".parcial.npz")]
    filas = cargar_v7(f_v7)
    a0, ctrl, fallidos = cargar_a0(f_a0)
    pisados = sorted(set(filas) & set(a0))
    filas.update(a0)                                   # A0 manda sobre V7 si hay las dos (no debería)
    indice, arr, sin = armar_vistas(tracks, filas)
    n_aud = len(indice)
    rep = {"temas_con_audio": n_aud, "con_vistas": n_aud - len(sin), "sin_vistas": len(sin),
           "fallidos_a0": len(fallidos), "fallidos_ejemplo": fallidos[:10], "v7_y_a0_a_la_vez": len(pisados),
           "archivos_v7": len(f_v7), "archivos_a0": len(f_a0)}
    # Umbrales 1 a 3 y 6 del protocolo
    rep["umbral_1_cobertura"] = len(sin) <= 0.005 * n_aud
    rep["umbral_2_finitos"] = bool(all(np.isfinite(arr[k]).all() for k, _ in DIMS))
    por_rec: Dict[str, int] = {}
    alineado = True
    for i, row in enumerate(indice):
        j = por_rec.setdefault(row["recording_id"], i)
        if j != i and not all(np.array_equal(arr[k][i], arr[k][j]) for k, _ in DIMS):
            alineado = False
    rep["umbral_3_alineacion"] = alineado and [r["track_id"] for r in indice] == [t["track_id"] for t in tracks if t.get("has_audio") == "1"]
    vent = [v["ventana"] for v in ctrl.values() if "ventana" in v]
    rep["umbral_6_ventana_youtube"] = bool(vent) and all(abs(c / d - INICIO_FRAC) < 1e-3 and abs(s - min(LARGO_S, (1 - INICIO_FRAC) * d)) < 0.05
                                                         for c, d, s in vent) if vent else None
    rep["youtube_con_ventana"] = len(vent)
    salida_local = cache / "_salida"
    if salida_local.exists():
        shutil.rmtree(salida_local)
    salida_local.mkdir(parents=True)
    with open(salida_local / "indice.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(indice[0]))
        w.writeheader()
        w.writerows(indice)
    for k, _ in DIMS:
        np.save(salida_local / f"{k}.npy", arr[k])
    cid = sorted(r for r, v in ctrl.items() if "orig" in v)
    np.savez_compressed(salida_local / "youtube_control.npz", recording_id=np.asarray(cid, dtype=str),
                        **{f"{k}_orig": np.stack([ctrl[r]["orig"][k] for r in cid]) if cid else _vacio()[k] for k, _ in DIMS})
    leeme = {"que": "Vistas de DJSets v1 por tema (A0, docs/plans/linea_a_protocolo_a0_a1.md)", "dataset": DATASET,
             "orden": "tracks.csv, temas con has_audio = 1 (indice.csv)", "forma": {k: [n_aud, N_SEG, d] for k, d in DIMS},
             "ventana_beatport": "el preview entero en 4 segmentos de 30 s desde el comienzo, con ceros (V7, E15)",
             "ventana_youtube": f"comienzo {INICIO_FRAC} × D, largo mín({LARGO_S:.0f} s, D − comienzo); recodificada a "
                                f"MP3 CBR {KBPS} kbps (LAME, estéreo); youtube_control.npz: el códec original",
             "n_seg_validos": "segmentos con 15 s de audio o más (el 0 siempre)", "reporte": rep,
             "commit_juntar": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                                             text=True).stdout.strip()}
    leeme["md5"] = {p.name: _md5(p) for p in sorted(salida_local.iterdir())}
    (salida_local / "LEEME.json").write_text(json.dumps(leeme, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if subir:
        for p in sorted(salida_local.iterdir()):
            remote.put(p, f"{salida}/{p.name}")
    print(json.dumps(rep, ensure_ascii=False, indent=1), flush=True)
    return rep


def cosenos(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Coseno fila a fila de (n, 4, d) con (n, 4, d) -> (n, 4)."""
    return np.sum(A * B, axis=-1) / (np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1) + 1e-12)


def resumen_cos(c: np.ndarray, ok: np.ndarray) -> Dict[str, float]:
    v = c[ok]
    return {"n": int(v.size), "min": round(float(v.min()), 6), "mediana": round(float(np.median(v)), 6)} if v.size else {"n": 0}


def verificar_v7(a0: Path, v7: Path, mediana: float = 0.9999, minimo: float = 0.999) -> Dict[str, object]:
    """Umbral 4: la salida de A0 de un shard frente a la de V7 de la misma tanda, por vista y segmento."""
    z, w = np.load(a0), np.load(v7)
    idx = {f"bp_{b}": i for i, b in enumerate(w["bp_id"])}
    comunes = [(i, idx[str(r)]) for i, r in enumerate(z["recording_id"]) if str(r) in idx]
    ia, iv = (np.asarray(x) for x in zip(*comunes))
    seg = np.asarray(z["segundos"])[ia]
    ok = np.stack([seg >= SEG_S * k + 1e-3 for k in range(N_SEG)], axis=1)   # segmentos con algo de audio
    out = {"comunes": len(comunes), "solo_a0": int(len(z["recording_id"]) - len(comunes)),
           "segundos_iguales": bool(np.allclose(seg, np.asarray(w["segundos"])[iv], atol=1e-3))}
    for k, _ in DIMS:
        out[k] = resumen_cos(cosenos(z[k][ia], w[k][iv]), ok)
    out["pasa"] = out["segundos_iguales"] and all(out[k]["mediana"] >= mediana and out[k]["min"] >= minimo for k, _ in DIMS)
    return out


def verificar_e15(vistas: Path, e15: Path, mediana: float = 0.999, minimo: float = 0.99) -> Dict[str, object]:
    """Umbral 5: los previews de E15 en vistas/ frente a e15/emb/<bp_id>.npz (prev_*), por vista y segmento."""
    indice = leer_csv(vistas / "indice.csv")
    arr = {k: np.load(vistas / f"{k}.npy", mmap_mode="r") for k, _ in DIMS}
    fila = {}
    for i, r in enumerate(indice):
        fila.setdefault(r["recording_id"], i)
    cos = {k: [] for k, _ in DIMS}
    n = 0
    for f in sorted(e15.glob("*.npz")):
        i = fila.get(f"bp_{f.stem}")
        if i is None:
            continue
        e = np.load(f)
        n += 1
        for k, _ in DIMS:
            cos[k].append(cosenos(np.asarray(arr[k][i])[None], e[f"prev_{k}"][None])[0])
    out: Dict[str, object] = {"e15_encontrados": n}
    for k, _ in DIMS:
        c = np.concatenate(cos[k]) if cos[k] else np.zeros(0)
        out[k] = resumen_cos(c, np.isfinite(c))
    out["pasa"] = n > 0 and all(out[k]["mediana"] >= mediana and out[k]["min"] >= minimo for k, _ in DIMS)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="A0: vistas de todo DJSets v1 (trabajo de gpuq y armado por tema)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("correr")
    c.add_argument("--shards", required=True, help="beatport_060,youtube_000,...")
    c.add_argument("--max-temas", type=int, default=None, help="Solo los primeros N de cada shard (prueba chica)")
    c.add_argument("--device", default="auto")
    c.add_argument("--hf-cache", default=None)
    c.add_argument("--lote", type=int, default=16, help="Temas por pasada de los modelos")
    c.add_argument("--destino", default=DESTINO)
    c.add_argument("--hilos", type=int, default=HILOS, help="Hilos que preparan el audio")
    p = sub.add_parser("planear")
    p.add_argument("--destino", default=DESTINO)
    p.add_argument("--primero", type=int, default=1, help="Número del primer trabajo (traktor-a0-N)")
    j = sub.add_parser("juntar")
    j.add_argument("--destino", default=DESTINO)
    j.add_argument("--salida", default=SALIDA)
    j.add_argument("--cache", default=str(Path(tempfile.gettempdir()) / "a0_cache"))
    j.add_argument("--no-subir", action="store_true")
    v = sub.add_parser("verificar")
    v.add_argument("--a0")
    v.add_argument("--v7")
    v.add_argument("--vistas")
    v.add_argument("--e15", default=None, help="Carpeta de embeddings de E15 (<dj_sets>/e15/emb); obligatoria con --vistas")
    a = ap.parse_args()
    if a.cmd == "correr":
        return correr([x for x in a.shards.split(",") if x], a.max_temas, a.device, a.hf_cache, a.lote, a.destino, a.hilos)
    if a.cmd == "planear":
        return planear(a.destino, a.primero)
    if a.cmd == "juntar":
        from src.v4.common.remote_storage import open_remote
        juntar(open_remote(), a.destino, a.salida, Path(a.cache), subir=not a.no_subir)
        return 0
    if not a.a0 and not (a.vistas and a.e15):
        ap.error("verificar necesita --a0 y --v7, o --vistas y --e15")
    out = verificar_v7(Path(a.a0), Path(a.v7)) if a.a0 else verificar_e15(Path(a.vistas), Path(a.e15))
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
