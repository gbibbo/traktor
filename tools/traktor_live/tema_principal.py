"""
PURPOSE: El tema principal que suena en Traktor y los temas ya tocados en la sesión, con las reglas de
         Gabriel (docs/DECISIONS.md, 2026-11-06, puntos 6 y 7). Tracker es lógica pura: recibe con su
         instante los eventos del receptor (deckLoaded, updateDeck, updateChannel, updateMasterClock) y
         los faders de la DJM, sin leer archivos ni MIDI, para probarlo reproduciendo registros.
         - MASTER = un deck (A a D): el tema principal es el cargado en ese deck, suene o no.
         - MASTER = master tempo (o todavía desconocido): entre los decks al aire (en play, con tema y
           fader > 0), si alguno tiene el fader >= 90 % cuentan solo esos. Gana la sección main (30 a
           70 % del tema) sobre la intro (< 30 %), y la intro sobre el outro (>= 70 %). En empate sigue el
           actual; si el actual no está entre los mejores, gana el de fader más alto y, a igual fader, el
           que suena hace más. Si el actual sigue al aire, el cambio tiene que sostenerse HOLD_S; si salió
           del aire (pausa o fader a 0), cambia enseguida. Sin nadie al aire sigue el que estaba. Un
           cambio de MASTER recalcula enseguida.
         - Fader de cada deck: con mezcla externa, el de la DJM (CC 17 a 20 en el canal MIDI 1 = decks A a
           D); con la interna, el volumen del canal de Traktor (updateChannel onAirLevel). Sin dato todavía
           (la DJM manda un fader solo cuando se mueve): arriba.
         - Tocado: al menos PLAYED_S en play con el fader >= PLAYED_LEVEL. Cargar no alcanza, y preescuchar
           con el fader abajo tampoco.
         Live junta el Tracker con el registro de la sesión y los avisos, bajo un candado: lo usan el hilo
         del receptor HTTP y el de la DJM. mixer_mode() lee en Traktor Settings.tsi si la mezcla es
         externa o interna (Routing.Mixer.External), porque la API de Traktor no lo dice.
CHANGELOG:
  - 2026-11-06: Creación (etapa de recomendación, unidad 1: tema principal y registro).
"""
from __future__ import annotations

import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.traktor_live import traktor_estado as te  # noqa: E402
from tools.traktor_live.midi_monitor import decode  # noqa: E402

DECKS = "ABCD"
INTRO_END, OUTRO_START = 0.30, 0.70           # intro < 30 % del tema <= main < 70 % <= outro
SECTION_RANK = {"main": 2, "intro": 1, "outro": 0}
STRONG_LEVEL = 0.90                           # "un tema principal seguramente esté por encima del 90 %"
HOLD_S = 3.0                                  # cuánto se sostiene un cambio con el actual todavía al aire
PLAYED_S = 30.0                               # tocado: segundos en play con el fader >= PLAYED_LEVEL
PLAYED_LEVEL = 0.50
EXTRAPOLATE_MAX_S = 3.0                       # la posición llega cada 1 s mientras suena; entre medio se estima
STEP_MAX_S = 10.0                             # tope del tiempo al aire que se suma de una vez
DEFAULT_FADERS: Dict[str, Tuple[int, int]] = {"A": (1, 17), "B": (1, 18), "C": (1, 19), "D": (1, 20)}
SETTINGS_TSI = "Traktor Settings.tsi"
MIXER_KEY = "Routing.Mixer.External"          # 1 = mezcla externa, 0 = interna
Change = Tuple[str, dict]                     # ("principal" o "tocado", datos para el registro)


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class Deck:
    path: str = ""
    title: str = ""
    artist: str = ""
    length: float = 0.0                       # segundos del tema
    elapsed: float = 0.0                      # segundos del tema en el último aviso
    t_elapsed: float = 0.0                    # instante de ese aviso
    playing: bool = False
    tempo: float = 1.0                        # 1.03 = 3 % más rápido
    bpm: float = 0.0                          # BPM del tema (sin el tempo)
    key: str = ""                             # tonalidad resultante de Traktor
    started: float = 0.0                      # instante del último play

    def position(self, now: float) -> Optional[float]:
        """Fracción del tema (0 a 1), estimada desde el último aviso si está sonando."""
        if self.length <= 0:
            return None
        e = self.elapsed
        if self.playing:
            e += min(max(now - self.t_elapsed, 0.0), EXTRAPOLATE_MAX_S) * (self.tempo or 1.0)
        return e / self.length

    def section(self, now: float) -> str:
        pos = self.position(now)
        if pos is None:
            return "main"                     # largo desconocido: no se lo castiga
        return "intro" if pos < INTRO_END else ("main" if pos < OUTRO_START else "outro")


@dataclass(frozen=True)
class Principal:
    deck: str
    path: str
    title: str
    artist: str


class Tracker:
    """Estado de los cuatro decks, el MASTER y los faders; decide el tema principal y lo tocado."""

    def __init__(self, mixer: Optional[str] = None, faders: Optional[Dict[str, Tuple[int, int]]] = None):
        self.mixer = mixer                    # "externa", "interna" o None (sin saber)
        self.faders = {d: (int(v[0]), int(v[1])) for d, v in (faders or DEFAULT_FADERS).items()}
        self._by_cc = {v: d for d, v in self.faders.items()}
        self.decks: Dict[str, Deck] = {d: Deck() for d in DECKS}
        self.master: Optional[str] = None
        self.master_known = False
        self.channel_level: Dict[str, float] = {}
        self.djm_level: Dict[str, float] = {}
        self.principal: Optional[Principal] = None
        self._pending: Optional[Tuple[str, str, float]] = None
        self.on_air_s: Dict[str, float] = defaultdict(float)
        self.played: List[str] = []
        self._played_set = set()
        self._last: Optional[float] = None

    # ---------------------------------------------------------------- entradas

    def fader_deck(self, channel: int, cc: int) -> Optional[str]:
        return self._by_cc.get((channel, cc))

    def level(self, deck: str) -> float:
        if self.mixer == "externa":
            v = self.djm_level.get(deck)
        elif self.mixer == "interna":
            v = self.channel_level.get(deck)
        else:                                 # sin saber el modo: la DJM si mandó algo
            v = self.djm_level.get(deck, self.channel_level.get(deck))
        return 1.0 if v is None else v

    def on_event(self, kind: str, target: str, data: dict, now: float) -> List[Change]:
        out = self._advance(now)
        force = False
        if kind == "deckLoaded" and target in self.decks:
            playing = bool(data.get("isPlaying"))
            self.decks[target] = Deck(
                path=str(data.get("filePath") or ""), title=str(data.get("title") or ""),
                artist=str(data.get("artist") or ""), length=_num(data.get("trackLength")),
                elapsed=_num(data.get("elapsedTime")), t_elapsed=now, playing=playing,
                tempo=_num(data.get("tempo"), 1.0) or 1.0, bpm=_num(data.get("bpm")),
                key=str(data.get("resultingKey") or ""), started=now if playing else 0.0)
        elif kind == "updateDeck" and target in self.decks:
            d = self.decks[target]
            if "elapsedTime" in data:
                d.elapsed, d.t_elapsed = _num(data["elapsedTime"]), now
            if "isPlaying" in data:
                playing = bool(data["isPlaying"])
                if playing and not d.playing:
                    d.started = now
                d.playing = playing
            if "tempo" in data:
                d.tempo = _num(data["tempo"], 1.0) or 1.0
            if "resultingKey" in data:
                d.key = str(data["resultingKey"] or "")
        elif kind == "updateChannel":
            deck = DECKS[int(target) - 1] if target in ("1", "2", "3", "4") else None
            if deck is None or "onAirLevel" not in data:
                return out
            self.channel_level[deck] = _num(data["onAirLevel"])
        elif kind == "updateMasterClock":
            master = data.get("deck") if data.get("deck") in self.decks else None
            force = not self.master_known or master != self.master
            self.master, self.master_known = master, True
        else:
            return out
        return out + self._evaluate(now, force)

    def on_fader(self, channel: int, cc: int, value: int, now: float) -> List[Change]:
        deck = self.fader_deck(channel, cc)
        if deck is None:
            return []
        out = self._advance(now)
        self.djm_level[deck] = max(0, min(127, int(value))) / 127.0
        return out + self._evaluate(now, False)

    def set_mixer(self, mixer: Optional[str], now: float) -> List[Change]:
        out = self._advance(now)
        self.mixer = mixer
        return out + self._evaluate(now, True)

    # ---------------------------------------------------------------- reglas

    def _advance(self, now: float) -> List[Change]:
        """Suma el tiempo al aire desde el último evento y avisa los temas que pasan a tocados."""
        out: List[Change] = []
        if self._last is not None:
            step = min(max(now - self._last, 0.0), STEP_MAX_S)
            for name in DECKS:
                d = self.decks[name]
                if step <= 0 or not (d.playing and d.path) or self.level(name) < PLAYED_LEVEL:
                    continue
                self.on_air_s[d.path] += step
                if d.path not in self._played_set and self.on_air_s[d.path] >= PLAYED_S:
                    self._played_set.add(d.path)
                    self.played.append(d.path)
                    out.append(("tocado", {"deck": name, "path": d.path, "title": d.title, "artist": d.artist,
                                           "segundos": round(self.on_air_s[d.path], 1)}))
        self._last = now if self._last is None else max(self._last, now)
        return out

    def _current_deck(self) -> Optional[str]:
        """Deck que todavía tiene cargado el tema principal."""
        p = self.principal
        return p.deck if p is not None and self.decks[p.deck].path == p.path else None

    def _evaluate(self, now: float, force: bool) -> List[Change]:
        if self.master_known and self.master is not None:          # modo simple
            self._pending = None
            if not self.decks[self.master].path:
                return []
            return self._set(self.master, now, "deck", f"deck {self.master} es el MASTER", [])
        on_air = [x for x in DECKS if self.decks[x].playing and self.decks[x].path and self.level(x) > 0]
        if not on_air:
            self._pending = None
            return []
        pool = [x for x in on_air if self.level(x) >= STRONG_LEVEL] or on_air
        rank = {x: SECTION_RANK[self.decks[x].section(now)] for x in pool}
        best = max(rank.values())
        top = [x for x in pool if rank[x] == best]
        cur = self._current_deck()
        if cur in top:
            self._pending = None
            return []
        winner = max(top, key=lambda x: (self.level(x), -self.decks[x].started))
        if cur is None or cur not in on_air:
            why = "el anterior salió del aire" if self.principal is not None else "primer tema al aire"
            if len(on_air) == 1:
                why = "único al aire" if self.principal is None else "único al aire; el anterior salió"
            return self._set(winner, now, "tempo", why, on_air)
        why = (f"{self.decks[winner].section(now)} gana a {self.decks[cur].section(now)}"
               if rank.get(cur, -1) < best and cur in pool else "fader por encima del 90 % y el anterior no")
        if force:
            return self._set(winner, now, "tempo", why, on_air)
        key = (winner, self.decks[winner].path)
        if self._pending is None or self._pending[:2] != key:
            self._pending = (winner, self.decks[winner].path, now)
            return []
        if now - self._pending[2] >= HOLD_S:
            return self._set(winner, now, "tempo", why, on_air)
        return []

    def _set(self, deck: str, now: float, mode: str, why: str, on_air: List[str]) -> List[Change]:
        d = self.decks[deck]
        new = Principal(deck, d.path, d.title, d.artist)
        self._pending = None
        if new == self.principal:
            return []
        prev, self.principal = self.principal, new
        pos = d.position(now)
        return [("principal", {
            "deck": deck, "path": d.path, "title": d.title, "artist": d.artist, "modo": mode, "motivo": why,
            "seccion": d.section(now), "posicion": None if pos is None else round(pos, 3),
            "fader": round(self.level(deck), 3), "mezcla": self.mixer,
            "al_aire": {x: {"seccion": self.decks[x].section(now), "fader": round(self.level(x), 3)} for x in on_air},
            "anterior": None if prev is None else {"deck": prev.deck, "path": prev.path},
        })]


def faders_from(data) -> Dict[str, Tuple[int, int]]:
    """Faders guardados en sugerencias_vivo.json ({"A": [canal, cc], ...}) o los de la DJM-750MK2."""
    try:
        out = {str(d): (int(v[0]), int(v[1])) for d, v in data.items() if str(d) in tuple(DECKS)}
    except (AttributeError, TypeError, ValueError, IndexError):
        out = {}
    return out or dict(DEFAULT_FADERS)


# ---------------------------------------------------------------- modo de mezcla (Traktor Settings.tsi)

def settings_tsi(collection: Optional[Path]) -> Optional[Path]:
    """Traktor Settings.tsi está en la misma carpeta que collection.nml."""
    return None if collection is None else Path(collection).parent / SETTINGS_TSI


def mixer_mode(tsi: Optional[Path]) -> Optional[str]:
    """"externa", "interna" o None si no se puede leer."""
    if tsi is None:
        return None
    try:
        root = ET.parse(tsi).getroot()
    except (OSError, ET.ParseError):
        return None
    for e in root.iter("Entry"):
        if e.get("Name") == MIXER_KEY:
            return "externa" if str(e.get("Value", "")).strip() == "1" else "interna"
    return None


class MixerWatch:
    """Vuelve a leer el modo de mezcla cuando Traktor reescribe sus ajustes (al abrir y al cerrar)."""

    def __init__(self, tsi: Optional[Path]):
        self.tsi = tsi
        self._mtime: Optional[float] = None
        self.mode: Optional[str] = None

    def poll(self) -> bool:
        """True si el modo cambió desde la lectura anterior."""
        try:
            mtime = self.tsi.stat().st_mtime if self.tsi is not None else None
        except OSError:
            mtime = None
        if mtime is None or mtime == self._mtime:
            return False
        self._mtime = mtime
        mode = mixer_mode(self.tsi)
        changed = mode != self.mode
        self.mode = mode
        return changed


# ---------------------------------------------------------------- en vivo

def describe(data: dict) -> str:
    """Una línea para la ventana y servicio.log."""
    who = f"{data.get('artist') or ''} - {data.get('title') or ''}".strip(" -") or data.get("path", "?")
    if data.get("modo") == "deck":
        return f"Tema principal: {who} (deck {data['deck']}, es el MASTER)"
    pos = data.get("posicion")
    where = f"{data.get('seccion')}" + ("" if pos is None else f" {100 * pos:.0f} %")
    return f"Tema principal: {who} (deck {data['deck']}, {where}, fader {100 * data.get('fader', 0):.0f} %; {data.get('motivo')})"


class Live:
    """El Tracker con candado, el registro de la sesión y los avisos. http() lo llama el receptor por
    cada evento de Traktor y midi() el hilo de la DJM por cada mensaje; on_principal recibe cada cambio
    (para calcular las sugerencias cuando cambia el tema principal, no al apretar el botón)."""

    def __init__(self, tracker: Tracker, log_path: Optional[Path], say: Callable[[str, str], None],
                 clock: Callable[[], float] = time.monotonic):
        self.tracker, self.log_path, self.say, self.clock = tracker, log_path, say, clock
        self.lock = threading.Lock()
        self.on_principal: List[Callable[[dict], None]] = []

    def http(self, kind: str, target: str, data: dict) -> None:
        with self.lock:
            changes = self.tracker.on_event(kind, target, data, self.clock())
        self._report(changes)

    def midi(self, packed: int) -> None:
        kind, ch, d1, d2 = decode(packed)
        if kind != "cc":
            return
        deck = self.tracker.fader_deck(ch, d1)
        if deck is None:
            return
        with self.lock:
            moved = self.tracker.djm_level.get(deck) != d2 / 127.0
            changes = self.tracker.on_fader(ch, d1, d2, self.clock())
        if moved:
            te.log_event(self.log_path, "fader", deck, {"canal": ch, "cc": d1, "valor": d2})
        self._report(changes)

    def set_mixer(self, mixer: Optional[str]) -> None:
        with self.lock:
            changes = self.tracker.set_mixer(mixer, self.clock())
        te.log_event(self.log_path, "mezcla", "", {"modo": mixer})
        self.say("info", f"Mezcla {mixer}: los faders salen de "
                         + {"externa": "la DJM", "interna": "Traktor"}.get(mixer or "", "la DJM o de Traktor (no pude leer el modo)"))
        self._report(changes)

    def _report(self, changes: List[Change]) -> None:
        for kind, data in changes:
            te.log_event(self.log_path, kind, data.get("deck", ""), data)
            if kind == "principal":
                self.say("info", describe(data))
                for fn in self.on_principal:
                    fn(data)
            elif kind == "tocado":
                who = f"{data.get('artist') or ''} - {data.get('title') or ''}".strip(" -")
                self.say("info", f"Tocado: {who} (deck {data['deck']}, {data['segundos']:.0f} s al aire)")
