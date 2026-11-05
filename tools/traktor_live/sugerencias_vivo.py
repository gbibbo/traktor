"""
PURPOSE: Sugerencias en vivo dentro de Traktor con un botón del K2 (D0 del canal 15, sin otra función en
         Traktor). Junta el receptor de traktor_estado.py (estado de los decks y el navegador de la
         pantalla del D2, que es el mismo árbol que ve el DJ) y la entrada MIDI del K2, que Windows
         comparte con Traktor. Al apretar el botón: guarda en qué carpeta y en qué tema está el árbol,
         escribe 10 temas nuevos en el .nml de la carpeta de sugerencias y lleva el árbol a esa playlist
         pasando por su carpeta (Traktor relee un .nml del Explorer al entrar desde otra carpeta). Al
         apretarlo otra vez, vuelve a la carpeta y al tema guardados. Si el DJ se fue de las sugerencias
         por su cuenta, el botón vuelve a mostrarle 10 nuevas.
         Los temas son de prueba (rondas de la colección con cues propios): el cálculo de las
         sugerencias, con la trayectoria del set, viene después.
CHANGELOG:
  - 2026-11-04: Creación inicial.
  - 2026-11-05: El botón es Ch15.Note.D0 sin aprender (la primera prueba aprendió C#0, el que despliega
                el panel); --aprender sigue para cambiarlo y dice el nombre de Traktor. El .nml va a
                <unidad>:\\TRAKTOR ML Sugerencias: la pantalla del D2 vio vacía C:\\Users\\bibbo. Si una
                ida o una vuelta falla, la pulsación siguiente la repite sin pisar el lugar guardado.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import signal
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from src.v4.common.traktor_nml import find_collection  # noqa: E402
from tools.traktor_live import traktor_estado as te  # noqa: E402
from tools.traktor_live.midi_monitor import MidiInput, decode, find_input, list_inputs  # noqa: E402
from tools.traktor_live.sugerencias_explorer import entries_by_path, nml_bytes, write_atomic  # noqa: E402
from tools.traktor_live.sugerencias_itunes import PLAYLIST_NAME, collection_tracks, pick  # noqa: E402

BUTTON_FILE = te.DEFAULT_LOG_DIR / "boton_k2.json"          # solo si se cambió con --aprender
DEFAULT_BUTTON = {"channel": 15, "note": 14}                # Ch15.Note.D0 en el mapeo de Gabriel
NML_NAME = f"{PLAYLIST_NAME} NML"
# En la raíz de la unidad del usuario: la pantalla del D2 vio vacía C:\Users\bibbo (2026-11-05)
LIVE_DIR = Path(Path.home().anchor) / PLAYLIST_NAME
DEBOUNCE_S = 0.4
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def traktor_name(channel: int, note: int) -> str:
    """Nombre que muestra el Controller Manager de Traktor: nota 13 = C#0, 14 = D0."""
    return f"Ch{channel:02d}.Note.{NOTE_NAMES[note % 12]}{note // 12 - 1}"


def explorer_path(nml: Path) -> List[str]:
    """Ruta del navegador de Traktor hasta la playlist: Browser | Explorer | C: | ... | <nombre>."""
    parts = [p for p in nml.parent.parts]
    drive = parts[0].rstrip("\\/")
    return ["Browser", "Explorer", drive] + parts[1:] + [nml.stem]


def same_name(a: str, b: str) -> bool:
    """Igual que sameName de ApiBrowserProbe.qml: sin mayúsculas o con una "s" final de diferencia."""
    x, y = a.lower(), b.lower()
    return x == y or x + "s" == y or y + "s" == x


def same_path(a: List[str], b: List[str]) -> bool:
    return len(a) == len(b) and all(same_name(x, y) for x, y in zip(a, b))


class Toggle:
    """Qué hace cada pulsación: ir a las sugerencias (guardando dónde estaba) o volver."""

    def __init__(self, target: List[str]):
        self.target = target
        self.saved: Optional[Tuple[List[str], int]] = None
        self.retry: Optional[Tuple[str, List[str], int]] = None

    def failed(self, action: str, path: List[str], index: int) -> None:
        """No llegó: la pulsación siguiente repite lo mismo (el árbol quedó a medio camino, no es un
        lugar del DJ que haya que guardar)."""
        self.retry = (action, path, index)

    def decide(self, path: str, index: int) -> Tuple[str, List[str], int]:
        if self.retry is not None:
            retry, self.retry = self.retry, None
            return retry
        cur = [p for p in path.split(" | ") if p]
        if same_path(cur, self.target):
            if self.saved is not None:
                saved, self.saved = self.saved, None
                return "volver", saved[0], saved[1]
            return "sugerencias", self.target, 0          # sin lugar guardado: 10 nuevas
        self.saved = (cur, index)
        return "sugerencias", self.target, 0


def load_button() -> Optional[dict]:
    try:
        b = json.loads(BUTTON_FILE.read_text(encoding="utf-8"))
        return {"channel": int(b["channel"]), "note": int(b["note"])}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def learn_button(midi: MidiInput, stop: threading.Event) -> Optional[dict]:
    print("[INFO] Aprieta el botón del K2 que va a abrir las sugerencias...")
    while not stop.is_set():
        try:
            packed, _ = midi.events.get(timeout=0.2)
        except Exception:
            continue
        kind, ch, note, vel = decode(packed)
        if kind == "note_on" and vel > 0:
            b = {"channel": ch, "note": note}
            BUTTON_FILE.parent.mkdir(parents=True, exist_ok=True)
            BUTTON_FILE.write_text(json.dumps(b), encoding="utf-8")
            print(f"[OK] Botón: {traktor_name(ch, note)} (guardado en {BUTTON_FILE})")
            return b
    return None


class Suggestions:
    """Escribe cada ronda de prueba en el .nml (las entradas copiadas de la colección)."""

    def __init__(self, collection: Path, out_dir: Path, n: int):
        self.tracks = collection_tracks(collection)
        self.by_path = entries_by_path(collection)
        self.version = ET.parse(collection).getroot().get("VERSION", "20")
        self.nml = out_dir / f"{NML_NAME}.nml"
        self.n = n
        self.seed = int(time.time())
        self.round = 0
        self.rounds = max(1, len(self.tracks) // n)

    def next(self) -> List:
        self.round += 1
        chosen = pick(self.tracks, (self.round - 1) % self.rounds + 1, self.n, self.seed)
        entries = [self.by_path[str(t.path).lower()] for t in chosen]
        write_atomic(nml_bytes(entries, self.round, self.version, name=NML_NAME), self.nml)
        return chosen


def main() -> int:
    ap = argparse.ArgumentParser(description="Sugerencias en vivo en Traktor con un botón del K2")
    ap.add_argument("--port", type=int, default=8080, help="Puerto del receptor (el de ApiClient.js)")
    ap.add_argument("--midi", default="K2", help="Entrada MIDI del K2 (palabra del nombre)")
    ap.add_argument("--aprender", action="store_true",
                    help="Elegir otro botón del K2 apretándolo (por defecto Ch15.Note.D0)")
    ap.add_argument("--n", type=int, default=10, help="Temas por lista")
    ap.add_argument("--dir", default=str(LIVE_DIR), help="Carpeta del .nml de sugerencias")
    ap.add_argument("--collection", default=None, help="collection.nml (por defecto la de Traktor)")
    args = ap.parse_args()

    collection = Path(args.collection) if args.collection else find_collection()
    if collection is None or not collection.exists():
        print("[ERROR] No encontré collection.nml de Traktor")
        return 1
    sugg = Suggestions(collection, Path(args.dir), args.n)
    if not sugg.nml.exists():
        sugg.next()                                         # para que la playlist exista en el Explorer
    target = explorer_path(sugg.nml)

    names = list_inputs()
    idx = find_input(names, args.midi)
    if idx is None:
        print(f"[ERROR] No encontré una sola entrada MIDI con «{args.midi}»: {names}")
        return 1
    midi = MidiInput(idx)
    error = midi.open()
    if error:
        print(f"[ERROR] No pude abrir {names[idx]}: {error}")
        return 1

    state = te.State(show_browser=False)
    log_path = te.DEFAULT_LOG_DIR / f"sugerencias_vivo_{dt.datetime.now():%Y%m%d_%H%M%S}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = te.make_handler(state, log_path)
    servers: List = []
    try:
        servers.append(te.ExclusiveServer(("127.0.0.1", args.port), handler))
        servers.append(te._Server6(("::1", args.port), handler))
    except OSError as exc:
        for s in servers:
            s.server_close()
        midi.close()
        if te.port_in_use(exc):
            print(f"[ERROR] El puerto {args.port} ya lo usa otro programa (¿traktor_estado.py abierto?). "
                  "Cerrarlo con Ctrl+C y volver a correr esto.")
            return 1
        raise

    stop = threading.Event()
    previous = signal.signal(signal.SIGINT, lambda *_: stop.set())
    serve_thread = threading.Thread(target=te.serve, args=(servers, stop), daemon=True)
    serve_thread.start()
    try:
        if args.aprender:
            button = learn_button(midi, stop)
            if button is None:
                return 0
        else:
            button = load_button() or DEFAULT_BUTTON
        print(f"[INFO] Listo. Botón: {traktor_name(button['channel'], button['note'])}. "
              f"Sugerencias en {sugg.nml}. Ctrl+C para salir.")
        toggle = Toggle(target)
        last = 0.0
        while not stop.is_set():
            try:
                packed, t = midi.events.get(timeout=0.1)
            except Exception:
                continue
            kind, ch, note, vel = decode(packed)
            if not (kind == "note_on" and vel > 0 and ch == button["channel"] and note == button["note"]):
                continue
            if t - last < DEBOUNCE_S:
                continue
            last = t
            press(state, toggle, sugg)
    finally:
        stop.set()
        signal.signal(signal.SIGINT, previous)
        serve_thread.join(timeout=3)
        midi.close()
    print("[INFO] Fin.")
    return 0


def press(state: te.State, toggle: Toggle, sugg: Suggestions) -> None:
    t0 = time.perf_counter()
    rid = state.enqueue("report")
    rep = state.wait_result(rid, 2.0)
    if rep is None:
        print("[AVISO] La pantalla del D2 no responde: ¿Traktor abierto, con el Kontrol D2 agregado y la sonda "
              "instalada (instalar_api_client.py --sonda)?")
        return
    action, path, index = toggle.decide(rep.get("path", ""), int(rep.get("currentIndex", -1)))
    if action == "sugerencias":
        chosen = sugg.next()
        print(f"[{dt.datetime.now():%H:%M:%S}] Sugerencias (lista {sugg.round}): "
              + "; ".join(f"{t.artist} - {t.title}" for t in chosen[:3]) + " ...")
    gid = state.enqueue("goto", {"path": path, "index": index})
    res = state.wait_result(gid, 12.0)
    ms = (time.perf_counter() - t0) * 1000
    now = f"{dt.datetime.now():%H:%M:%S}"
    if res is not None and res.get("reason") == "gotoDone":
        where = " › ".join(path[1:])
        print(f"[{now}] {'En las sugerencias' if action == 'sugerencias' else 'De vuelta en ' + where}"
              f" ({ms:.0f} ms)")
        return
    toggle.failed(action, path, index)
    why = "el navegador del D2 no contestó a tiempo" if res is None else res.get("error")
    print(f"[{now}] [AVISO] No pude llegar ({why}). Apretando otra vez lo reintento.")


if __name__ == "__main__":
    sys.exit(main())
