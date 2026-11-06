"""
PURPOSE: Sugerencias en vivo dentro de Traktor con un botón del controlador MIDI, en Windows y macOS
         (guía completa: docs/SUGERENCIAS_EN_VIVO.md). Junta el receptor de traktor_estado.py (estado de
         los decks y el navegador de la pantalla del D2, que es el mismo árbol que ve el DJ) y la entrada
         MIDI del controlador (midi_io.py), que el sistema comparte con Traktor. Al apretar el botón:
         guarda en qué carpeta está el árbol, escribe N temas nuevos en el .nml de la carpeta de
         sugerencias y lleva el árbol a esa playlist pasando por su carpeta (Traktor relee un .nml del
         Explorer al entrar desde otra carpeta). Al apretarlo otra vez, vuelve a la carpeta guardada, y
         Traktor conserva el tema que estaba elegido. Si el DJ se fue de las sugerencias por su cuenta,
         el botón vuelve a mostrarle N nuevas.
         Se adapta sin tocar el código: controlador (--midi), botón (--aprender, nota o CC), cantidad
         (--n), carpeta (--dir), motor MIDI (--backend). Lo elegido queda en
         artifacts/v4/traktor_live/sugerencias_vivo.json, junto con el camino del árbol hasta la
         playlist si Traktor lo muestra distinto del calculado (navegar una vez a mano hasta la playlist
         y apretar el botón lo enseña).
         Los temas son de prueba (rondas de la colección con cues propios): el cálculo de las
         sugerencias, con la trayectoria del set, viene después.
CHANGELOG:
  - 2026-11-04: Creación inicial.
  - 2026-11-05: El botón es Ch15.Note.D0 sin aprender (la primera prueba aprendió C#0, el que despliega
                el panel); --aprender sigue para cambiarlo y dice el nombre de Traktor. El .nml va a
                <unidad>:\\TRAKTOR ML Sugerencias: la pantalla del D2 vio vacía C:\\Users\\bibbo. Si una
                ida o una vuelta falla, la pulsación siguiente la repite sin pisar el lugar guardado.
  - 2026-11-05: Probado por Gabriel en Traktor 4.4.1 (Windows): ida 0,5-0,65 s, vuelta 0,6-0,8 s.
  - 2026-11-05: Windows y macOS (midi_io.py; en Mac el .nml va a /Users/Shared/TRAKTOR ML Sugerencias),
                cualquier controlador y botón (nota o CC), ajustes guardados en sugerencias_vivo.json y
                camino del árbol aprendido si Traktor no muestra la carpeta como se calculó.
  - 2026-11-05: En otro ordenador: si la colección tiene menos de N temas con cues propios, las listas de
                prueba salen de toda la colección (antes se caía); si ni así alcanzan, un mensaje claro.
                Al arrancar avisa si a Traktor le falta la sonda, con el comando para instalarla.
  - 2026-11-06: Cada pulsación queda en el registro de la sesión (kind "boton": ida o vuelta, desde y
                hacia dónde, si llegó, ms y, en una ida, los temas sugeridos), para la prueba tocando.
  - 2026-11-06: run_session() (la usa también el servicio automático, sugerencias_servicio.py) y
                MidiWatcher: si el controlador aparece tarde o se desenchufa y vuelve, se reabre solo.
                Los avisos van por say(nivel, texto).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import queue
import signal
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from src.v4.common.traktor_nml import DEFAULT_MAC_BOOT_VOLUME, find_collection, mac_boot_volume  # noqa: E402
from tools.traktor_live import instalar_api_client as inst  # noqa: E402
from tools.traktor_live import traktor_estado as te  # noqa: E402
from tools.traktor_live.midi_io import BACKENDS, default_backend, list_inputs, open_input  # noqa: E402
from tools.traktor_live.midi_monitor import decode, find_input  # noqa: E402
from tools.traktor_live.sugerencias_explorer import entries_by_path, nml_bytes, write_atomic  # noqa: E402
from tools.traktor_live.sugerencias_itunes import PLAYLIST_NAME, collection_tracks, pick  # noqa: E402

SETTINGS_FILE = te.DEFAULT_LOG_DIR / "sugerencias_vivo.json"
NML_NAME = f"{PLAYLIST_NAME} NML"
DEFAULT_MIDI = "K2"
DEBOUNCE_S = 0.4
MIDI_POLL_S = 2.0                     # cada cuánto se revisa si el controlador se desenchufó o apareció
LEVELS = {"info": "INFO", "ok": "OK", "aviso": "AVISO", "error": "ERROR"}
Say = Callable[[str, str], None]      # (nivel, texto): la ventana lo imprime, el servicio además notifica
NOTE_NAMES =("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


@dataclass(frozen=True)
class Button:
    """Botón del controlador: una nota (note_on con velocidad > 0) o un CC (valor >= 64)."""
    kind: str          # "note" o "cc"
    channel: int       # 1-16
    number: int        # nota o número de CC

    def name(self) -> str:
        """Como lo muestra el Controller Manager de Traktor: nota 13 = C#0, 14 = D0."""
        if self.kind == "cc":
            return f"Ch{self.channel:02d}.CC.{self.number:03d}"
        return f"Ch{self.channel:02d}.Note.{NOTE_NAMES[self.number % 12]}{self.number // 12 - 1}"

    def pressed(self, kind: str, ch: int, d1: int, d2: int) -> bool:
        if ch != self.channel or d1 != self.number:
            return False
        return (kind == "note_on" and d2 > 0) if self.kind == "note" else (kind == "cc" and d2 >= 64)

    @staticmethod
    def from_event(kind: str, ch: int, d1: int, d2: int) -> Optional["Button"]:
        """El botón que mandó este mensaje, si parece un botón apretado."""
        if kind == "note_on" and d2 > 0:
            return Button("note", ch, d1)
        if kind == "cc" and d2 >= 64:
            return Button("cc", ch, d1)
        return None


DEFAULT_BUTTON = Button("note", 15, 14)             # Ch15.Note.D0 en el mapeo del K2 de Gabriel


def traktor_name(channel: int, note: int) -> str:
    return Button("note", channel, note).name()


def default_live_dir(platform: Optional[str] = None, home: Optional[str] = None) -> str:
    """Carpeta del .nml de sugerencias. Windows: raíz de la unidad del usuario (la pantalla del D2 vio
    vacía C:\\Users\\bibbo). macOS: /Users/Shared, que todos pueden escribir y no pasa por la carpeta
    personal. Otros sistemas: la carpeta personal."""
    system = platform or sys.platform
    home = home or str(Path.home())
    if system == "win32":
        return str(PureWindowsPath(PureWindowsPath(home).anchor) / PLAYLIST_NAME)
    if system == "darwin":
        return str(PurePosixPath("/Users/Shared") / PLAYLIST_NAME)
    return str(PurePosixPath(home) / PLAYLIST_NAME)


def explorer_path(nml, platform: Optional[str] = None, boot_volume: Optional[str] = None) -> List[str]:
    """Ruta del navegador de Traktor hasta la playlist. Windows: Browser | Explorer | C: | ... | <nombre>.
    macOS: Browser | Explorer | <disco> | ... | <nombre>, con el disco de arranque para lo que cuelga de
    "/" y el nombre del disco para /Volumes/<disco>/... (si Traktor lo muestra distinto, se aprende)."""
    system = platform or sys.platform
    if system == "win32":
        p = PureWindowsPath(str(nml))
        return ["Browser", "Explorer", p.drive] + list(p.parts[1:-1]) + [p.stem]
    p = PurePosixPath(str(nml))
    parts = list(p.parts[1:-1])
    if parts[:1] == ["Volumes"] and len(parts) >= 2:
        volume, rest = parts[1], parts[2:]
    else:
        volume = boot_volume or (mac_boot_volume() if system == "darwin" else DEFAULT_MAC_BOOT_VOLUME)
        rest = parts
    return ["Browser", "Explorer", volume] + rest + [p.stem]


def same_name(a: str, b: str) -> bool:
    """Igual que sameName de ApiBrowserProbe.qml: sin mayúsculas o con una "s" final de diferencia."""
    x, y = a.lower(), b.lower()
    return x == y or x + "s" == y or y + "s" == x


def same_path(a: List[str], b: List[str]) -> bool:
    return len(a) == len(b) and all(same_name(x, y) for x, y in zip(a, b))


class Toggle:
    """Qué hace cada pulsación: ir a las sugerencias (guardando dónde estaba) o volver. La playlist se
    reconoce por su camino o, si Traktor la muestra por otro, por sus dos últimos nombres (carpeta y
    playlist): ese camino nuevo pasa a ser el destino y se avisa a on_learn para guardarlo."""

    def __init__(self, target: List[str], on_learn: Optional[Callable[[List[str]], None]] = None):
        self.target = list(target)
        self.saved: Optional[Tuple[List[str], int]] = None
        self.retry: Optional[Tuple[str, List[str], int]] = None
        self.on_learn = on_learn

    def failed(self, action: str, path: List[str], index: int) -> None:
        """No llegó: la pulsación siguiente repite lo mismo (el árbol quedó a medio camino, no es un
        lugar del DJ que haya que guardar)."""
        self.retry = (action, path, index)

    def at_target(self, cur: List[str]) -> bool:
        if same_path(cur, self.target):
            return True
        tail = self.target[-2:]
        if len(cur) > len(tail) and same_path(cur[-len(tail):], tail):
            self.target = list(cur)
            if self.on_learn is not None:
                self.on_learn(list(cur))
            return True
        return False

    def decide(self, path: str, index: int) -> Tuple[str, List[str], int]:
        cur = [p for p in path.split(" | ") if p]
        if self.retry is not None:
            retry, self.retry = self.retry, None
            if not (retry[0] == "sugerencias" and self.at_target(cur)):
                return retry if retry[0] == "volver" else ("sugerencias", self.target, 0)
            # el DJ llegó a mano a las sugerencias después de un intento fallido: sigue como siempre
        if self.at_target(cur):
            if self.saved is not None:
                saved, self.saved = self.saved, None
                return "volver", saved[0], saved[1]
            return "sugerencias", self.target, 0          # sin lugar guardado: N nuevas
        self.saved = (cur, index)
        return "sugerencias", self.target, 0


# ---------------------------------------------------------------- ajustes guardados

def load_settings(path: Path = SETTINGS_FILE) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(data: dict, path: Path = SETTINGS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def button_from_settings(data: dict) -> Button:
    b = data.get("button")
    try:
        return Button(str(b["kind"]), int(b["channel"]), int(b["number"]))
    except (TypeError, KeyError, ValueError):
        return DEFAULT_BUTTON


def learn_button(events: "queue.SimpleQueue", stop: threading.Event) -> Optional[Button]:
    print("[INFO] Aprieta el botón del controlador que va a abrir las sugerencias...")
    while not stop.is_set():
        try:
            packed, _ = events.get(timeout=0.2)
        except queue.Empty:
            continue
        b = Button.from_event(*decode(packed))
        if b is not None:
            print(f"[OK] Botón: {b.name()}")
            return b
    return None


class Suggestions:
    """Escribe cada ronda de prueba en el .nml (las entradas copiadas de la colección). Los temas salen de
    los que tienen cues propios (se nota al cargarlos que Traktor usa su entrada); si son menos de n,
    como en una colección recién armada, de todos los que tienen el archivo en su lugar."""

    def __init__(self, collection: Path, out_dir: Path, n: int):
        self.tracks = collection_tracks(collection)
        self.pool = "temas con cues propios"
        if len(self.tracks) < n:
            self.tracks = collection_tracks(collection, own_cues_only=False)
            self.pool = "temas de la colección"
        if len(self.tracks) < n:
            raise ValueError(f"La colección {collection} tiene {len(self.tracks)} temas con el archivo en su "
                             f"lugar y hacen falta {n}: agregar música a Traktor o usar --n más chico")
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
    ap = argparse.ArgumentParser(description="Sugerencias en vivo en Traktor con un botón del controlador "
                                             "(Windows y macOS; guía: docs/SUGERENCIAS_EN_VIVO.md)")
    ap.add_argument("--midi", default=None,
                    help=f"Parte del nombre de la entrada MIDI del controlador (por defecto la guardada o {DEFAULT_MIDI})")
    ap.add_argument("--aprender", action="store_true",
                    help=f"Elegir otro botón apretándolo (por defecto {DEFAULT_BUTTON.name()}); nota o CC")
    ap.add_argument("--n", type=int, default=None, help="Temas por lista (por defecto el guardado o 10)")
    ap.add_argument("--dir", default=None, help="Carpeta del .nml de sugerencias (por defecto la guardada o "
                                                f"{default_live_dir()})")
    ap.add_argument("--backend", choices=BACKENDS, default=None,
                    help=f"Motor MIDI (por defecto {default_backend()} en este sistema)")
    ap.add_argument("--listar", action="store_true", help="Mostrar las entradas MIDI y salir")
    ap.add_argument("--collection", default=None, help="collection.nml (por defecto la de Traktor)")
    ap.add_argument("--port", type=int, default=8080,
                    help="Puerto del receptor; tiene que ser el de ApiClient.js y ApiBrowserProbe.qml (8080)")
    ap.add_argument("--ajustes", default=str(SETTINGS_FILE), help=argparse.SUPPRESS)
    args = ap.parse_args()

    settings_path = Path(args.ajustes)
    settings = load_settings(settings_path)
    if args.midi:
        settings["midi"] = args.midi
    if args.n:
        settings["n"] = args.n
    if args.dir:
        settings["dir"] = args.dir
    if args.backend:
        settings["backend"] = args.backend

    backend = settings.get("backend") or default_backend()
    try:
        names = list_inputs(backend)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 1
    if args.listar:
        print(f"[INFO] Entradas MIDI ({backend}):")
        for i, name in enumerate(names):
            print(f"  {i}: {name}")
        return 0

    save_settings(settings, settings_path)
    stop = threading.Event()
    previous = signal.signal(signal.SIGINT, lambda *_: stop.set())
    try:
        return run_session(settings_path, stop, port=args.port, collection_arg=args.collection,
                           learn=args.aprender)
    finally:
        signal.signal(signal.SIGINT, previous)


def console_say(level: str, text: str) -> None:
    print(f"[{LEVELS.get(level, level.upper())}] {text}", flush=True)


class MidiWatcher:
    """Mantiene abierta la entrada del controlador: la busca por una palabra de su nombre, la abre y, si
    cambia la lista de entradas (se desenchufó y se volvió a enchufar, o apareció más tarde), la vuelve a
    abrir. Avisa una sola vez por cada problema distinto."""

    def __init__(self, text: str, backend: str, say: Say, lister=list_inputs, opener=open_input):
        self.text, self.backend, self.say = text, backend, say
        self.lister, self.opener = lister, opener
        self.names: Optional[List[str]] = None
        self.input = None
        self.name = ""
        self._last_problem = ""

    def _problem(self, level: str, text: str) -> None:
        if text != self._last_problem:
            self._last_problem = text
            self.say(level, text)

    def poll(self) -> None:
        try:
            names = self.lister(self.backend)
        except RuntimeError as exc:                     # p. ej. falta python-rtmidi
            self._problem("error", str(exc))
            return
        if self.input is not None and names == self.names:
            return
        changed = self.names is not None and names != self.names
        self.names = names
        if self.input is not None:
            self.input.close()
            self.input = None
            self.say("aviso", "Cambiaron las entradas MIDI: vuelvo a abrir el controlador")
        idx = find_input(names, self.text)
        if idx is None:
            self._problem("aviso", f"No veo el controlador «{self.text}» entre las entradas MIDI {names}: "
                                   "lo sigo buscando (¿enchufado?; --midi elige otro)")
            return
        inp = self.opener(idx, self.backend)
        error = inp.open()
        if error:
            self._problem("error", f"No pude abrir {names[idx]}: {error}")
            return
        self.input, self.name = inp, names[idx]
        if self._last_problem or changed:
            self.say("ok", f"Controlador abierto: {self.name}")
        self._last_problem = ""

    def get(self, timeout: float):
        if self.input is None:
            time.sleep(timeout)
            return None
        try:
            return self.input.events.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        if self.input is not None:
            self.input.close()
            self.input = None


def run_session(settings_path: Path, stop: threading.Event, say: Say = console_say, port: int = 8080,
                collection_arg: Optional[str] = None, learn: bool = False) -> int:
    """Una sesión de sugerencias en vivo hasta que stop se activa: la ventana la corta con Ctrl+C y el
    servicio automático (sugerencias_servicio.py) cuando se cierra Traktor. 0 si terminó bien."""
    settings = load_settings(settings_path)
    backend = settings.get("backend") or default_backend()
    collection = Path(collection_arg) if collection_arg else find_collection()
    if collection is None or not collection.exists():
        say("error", "No encontré collection.nml de Traktor (Documentos/Native Instruments/Traktor*); "
                     "indicarlo con --collection")
        return 1
    out_dir = Path(settings.get("dir") or default_live_dir())
    try:
        sugg = Suggestions(collection, out_dir, int(settings.get("n") or 10))
        if not sugg.nml.exists():
            sugg.next()                                     # para que la playlist exista en el Explorer
    except (ValueError, OSError) as exc:
        say("error", str(exc))
        return 1
    say("info", f"Colección: {collection} ({len(sugg.tracks)} {sugg.pool} para las listas de prueba)")
    d2 = inst.default_d2()
    if d2 is None:
        say("aviso", "No encontré Traktor instalado en el lugar de siempre: si está en otro, instalar la sonda "
                     "con instalar_api_client.py --sonda --d2 <carpeta qml/CSI/D2>")
    elif not inst.is_ready(d2):
        installer = Path(__file__).resolve().parent / "instalar_api_client.py"
        say("aviso", f"A Traktor le falta la sonda o está desactualizada. Cerrar Traktor y correr una vez:\n"
                     f'         "{sys.executable}" "{installer}" --sonda')
    learned: Dict[str, List[str]] = settings.get("destinos") or {}
    target = learned.get(str(sugg.nml)) or explorer_path(sugg.nml)

    state = te.State(show_browser=False)
    log_path = te.DEFAULT_LOG_DIR / f"sugerencias_vivo_{dt.datetime.now():%Y%m%d_%H%M%S}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        servers = te.open_servers(port, te.make_handler(state, log_path))
    except OSError as exc:
        if te.port_in_use(exc):
            say("error", f"El puerto {port} ya lo usa otro programa: ¿ya corren las sugerencias (en otra ventana "
                         "o en el inicio automático) o traktor_estado.py?")
            return 1
        raise

    serve_thread = threading.Thread(target=te.serve, args=(servers, stop), daemon=True)
    serve_thread.start()
    midi = MidiWatcher(settings.get("midi") or DEFAULT_MIDI, backend, say)
    try:
        midi.poll()
        if learn:
            if midi.input is None:
                say("error", "Para aprender el botón el controlador tiene que estar enchufado")
                return 1
            button = learn_button(midi.input.events, stop)
            if button is None:
                return 0
            settings["button"] = asdict(button)
            save_settings(settings, settings_path)
        else:
            button = button_from_settings(settings)

        def remember(path: List[str]) -> None:
            settings.setdefault("destinos", {})[str(sugg.nml)] = path
            save_settings(settings, settings_path)
            say("ok", f"Aprendí el camino hasta las sugerencias: {' › '.join(path[1:])}")

        say("ok", f"Listo. Controlador: {midi.name or '(todavía no aparece)'}. Botón: {button.name()}. "
                  f"Sugerencias en {sugg.nml}.")
        toggle = Toggle(target, on_learn=remember)
        last, last_poll = 0.0, time.monotonic()
        while not stop.is_set():
            if time.monotonic() - last_poll > MIDI_POLL_S:
                midi.poll()
                last_poll = time.monotonic()
            event = midi.get(0.1)
            if event is None:
                continue
            packed, t = event
            if not button.pressed(*decode(packed)):
                continue
            if t - last < DEBOUNCE_S:
                continue
            last = t
            press(state, toggle, sugg, log_path, say)
    finally:
        stop.set()
        serve_thread.join(timeout=3)
        midi.close()
    say("info", "Fin de la sesión de sugerencias.")
    return 0


def press(state: te.State, toggle: Toggle, sugg: Suggestions, log_path: Optional[Path] = None,
          say: Say = console_say) -> None:
    """Una pulsación del botón. Anota en el registro de la sesión qué hizo, si llegó, cuánto tardó, desde
    dónde y, en una ida, los temas sugeridos (para ver después cuáles se cargaron en un deck). Lo normal
    se imprime; los problemas van también por say (el servicio los muestra como notificación)."""
    t0 = time.perf_counter()
    rid = state.enqueue("report")
    rep = state.wait_result(rid, 2.0)
    if rep is None:
        te.log_event(log_path, "boton", "sin_respuesta", {})
        say("aviso", "La pantalla del D2 no responde: ¿Traktor abierto, con el Kontrol D2 agregado en el "
                     "Controller Manager y la sonda instalada (instalar_api_client.py --sonda)?")
        return
    action, path, index = toggle.decide(rep.get("path", ""), int(rep.get("currentIndex", -1)))
    event = {"desde": rep.get("path", ""), "hacia": path}
    if action == "sugerencias":
        chosen = sugg.next()
        event.update({"lista": sugg.round, "temas": [str(t.path) for t in chosen]})
        print(f"[{dt.datetime.now():%H:%M:%S}] Sugerencias (lista {sugg.round}): "
              + "; ".join(f"{t.artist} - {t.title}" for t in chosen[:3]) + " ...")
    gid = state.enqueue("goto", {"path": path, "index": index})
    res = state.wait_result(gid, 12.0)
    ms = (time.perf_counter() - t0) * 1000
    now = f"{dt.datetime.now():%H:%M:%S}"
    ok = res is not None and res.get("reason") == "gotoDone"
    event.update({"llego": ok, "ms": round(ms), "error": "" if ok else (res or {}).get("error", "sin respuesta")})
    te.log_event(log_path, "boton", action, event)
    if ok:
        where = " › ".join(path[1:])
        print(f"[{now}] {'En las sugerencias' if action == 'sugerencias' else 'De vuelta en ' + where}"
              f" ({ms:.0f} ms)")
        return
    toggle.failed(action, path, index)
    why = "el navegador del D2 no contestó a tiempo" if res is None else res.get("error")
    text = f"[{now}] No pude llegar ({why}). Apretando otra vez lo reintento."
    if action == "sugerencias":
        text += (f" Si Traktor muestra la carpeta por otro camino: navega a mano una vez hasta la playlist "
                 f"«{NML_NAME}» y aprieta el botón; aprendo ese camino.")
    say("aviso", text)


if __name__ == "__main__":
    sys.exit(main())
