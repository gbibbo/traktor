"""
PURPOSE: Las sugerencias reales en vivo (DECISIONS 2026-11-06, puntos 7 y 8): los N temas de la biblioteca
         más parecidos por CLAP al tema principal, con el BPM a ±20 del que suena y tonalidad compatible
         transponiendo hasta 2 semitonos (docs/HARMONIC_COMPATIBILITY.md, corte 0,60), sin el tema
         principal ni lo ya tocado en la sesión. Solo biblioteca estándar: lee el índice que arma
         indice_recomendacion.py y la colección de Traktor, de donde salen la tonalidad y el BPM que
         analizó Traktor (mandan sobre los de los tags) y la entrada que se copia al .nml.
         - Se sugiere solo la copia del paquete (<unidad>:\\TRAKTOR ML) que Traktor tiene en su colección;
           la raíz es la que más temas del paquete tiene en la colección.
         - El tema principal se reconoce por su ruta en el paquete o por la original (Música/...).
         - Un candidato sin tonalidad conocida (ni de Traktor ni de los tags) no pasa: la regla pide
           compatibilidad. Si no se conoce la del tema principal, la tonalidad no filtra.
         - Si el tema principal no está en el índice o no tiene embedding, quedan los compatibles por BPM
           y tonalidad, del BPM más cercano al más lejano (modo "sin_clap").
         - Se calcula cuando cambia el tema principal o un tema pasa a tocado (Live), no al apretar.
         Orden (DECISIONS 2026-11-06, punto 9): cada candidato que pasa los filtros recibe un puntaje de 0
         a 1 por componente y el final es el promedio pesado de los que se conocen:
           clap    parecido coseno (recortado a 0-1);
           bpm     1 con el mismo BPM que suena, 0 a ±20 (lineal);
           clave   la regla armónica: 1 la misma tonalidad original, menos si hay que mover el pitch;
           genero  1 el mismo género (Beatport; si falta, el del clasificador), 0 otro;
           tiv     reservado para el vector TIV (todavía no existe).
         Los pesos van en "pesos" de sugerencias_vivo.json; por defecto solo CLAP (Gabriel, 11-06: por
         ahora solo parecido CLAP). Cada lista anota los componentes de sus temas para ajustar los pesos.
CHANGELOG:
  - 2026-11-06: Creación (etapa de recomendación, unidad 2: sugerencias por CLAP).
  - 2026-11-06: Puntaje como promedio pesado de componentes (clap, bpm, clave, genero, tiv) y género en
                el índice.
"""
from __future__ import annotations

import base64
import json
import operator
import re
import sys
import threading
import time
import unicodedata
import xml.etree.ElementTree as ET
from array import array
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common import harmonic  # noqa: E402
from src.v4.common.traktor_nml import entry_path  # noqa: E402
from tools.traktor_live import traktor_estado as te  # noqa: E402
from tools.traktor_live.sugerencias_itunes import Track  # noqa: E402

DEFAULT_INDEX = te.DEFAULT_LOG_DIR / "indice_recomendacion.json"
DEFAULT_N = 30
BPM_RANGE = 20.0
KEY_MIN = 0.60                        # corte aprobado (HARMONIC_COMPATIBILITY.md)
COMPONENTS = ("clap", "bpm", "clave", "genero", "tiv")
DEFAULT_WEIGHTS: Dict[str, float] = {"clap": 1.0, "bpm": 0.0, "clave": 0.0, "genero": 0.0, "tiv": 0.0}
MAJOR_CAMELOT = [8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6, 1]     # índice = semitono desde Do
MINOR_CAMELOT = [5, 12, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10]
_KEY_TEXT = re.compile(r"^\s*(\d{1,2})\s*([dmABab])\b")


def traktor_key(value) -> str:
    """MUSICAL_KEY VALUE de collection.nml (0-11 mayores desde Do, 12-23 menores desde Do menor) -> Camelot."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return "?"
    if 0 <= v < 12:
        return f"{MAJOR_CAMELOT[v]}B"
    if 12 <= v < 24:
        return f"{MINOR_CAMELOT[v - 12]}A"
    return "?"


def deck_key(text) -> str:
    """Tonalidad que manda la API ("12m", "6d" en Open Key; también Camelot o nombres) -> Camelot."""
    if not text:
        return "?"
    m = _KEY_TEXT.match(str(text))
    if m and 1 <= int(m.group(1)) <= 12:
        n, mode = int(m.group(1)), m.group(2)
        if mode in "dm":                                 # Open Key: 1d = Do mayor = 8B, 1m = La menor = 8A
            return f"{(n + 6) % 12 + 1}{'A' if mode == 'm' else 'B'}"
        return f"{n}{mode.upper()}"
    return harmonic.to_camelot(str(text))


def norm(path: str) -> str:
    """Ruta comparable entre Windows y macOS: NFC, "/" y minúsculas."""
    return unicodedata.normalize("NFC", str(path)).replace("\\", "/").lower()


def suffixes(path: str) -> Iterable[str]:
    """Del final más largo al más corto: "c:/a/b/x.mp3" -> "c:/a/b/x.mp3", "a/b/x.mp3", "b/x.mp3", "x.mp3"."""
    parts = [p for p in norm(path).split("/") if p]
    for i in range(len(parts)):
        yield "/".join(parts[i:])


@dataclass
class Item:
    uid: str
    rel: str                          # archivo dentro del paquete (_organizacion.csv)
    original: str                     # carpeta y nombre originales dentro de Música/
    artist: str
    title: str
    bpm: Optional[float]
    key: str                          # Camelot o "?"
    vec: Optional[List[float]]        # embedding normalizado
    genre: Optional[str] = None       # género de Beatport (o del clasificador)
    bpm_src: str = "tags"
    key_src: str = "tags"
    entry: Optional[ET.Element] = None
    path: Optional[Path] = None

    def track(self) -> Track:
        e = self.entry
        return Track(path=self.path, title=e.get("TITLE", self.title) if e is not None else self.title,
                     artist=e.get("ARTIST", self.artist) if e is not None else self.artist, seconds=0.0,
                     bpm="?" if self.bpm is None else f"{self.bpm:.0f}", key=self.key)


def load_index(path: Path) -> Tuple[dict, List[Item]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = []
    for t in data.get("temas", []):
        vec = None
        if t.get("emb"):
            a = array("f")
            a.frombytes(base64.b64decode(t["emb"]))
            if sys.byteorder == "big":
                a.byteswap()
            vec = a.tolist()
        bpm = t.get("bpm")
        items.append(Item(uid=t["uid"], rel=t.get("archivo") or "", original=t.get("original") or "",
                          artist=t.get("artista") or "", title=t.get("titulo") or "",
                          bpm=float(bpm) if bpm else None, key=harmonic.to_camelot(t.get("key")), vec=vec,
                          genre=t.get("genero") or None))
    return data, items


def weights_from(data) -> Dict[str, float]:
    """Pesos guardados ({"clap": 1, "bpm": 0.5, ...}) sobre los de por defecto; los desconocidos o
    negativos se ignoran."""
    out = dict(DEFAULT_WEIGHTS)
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                if k in out and float(v) >= 0:
                    out[k] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def weighted(components: Dict[str, Optional[float]], weights: Dict[str, float]) -> Optional[float]:
    """Promedio pesado de los componentes conocidos con peso > 0 (None si no queda ninguno)."""
    num = den = 0.0
    for k, w in weights.items():
        v = components.get(k)
        if w > 0 and v is not None:
            num += w * v
            den += w
    return num / den if den > 0 else None


def package_root(items: List[Item], paths: Iterable[str]) -> Optional[str]:
    """Raíz (normalizada) del paquete en la colección: donde más temas del índice aparecen."""
    rels = {norm(it.rel) for it in items if it.rel}
    counts: Counter = Counter()
    for p in paths:
        parts = p.split("/")
        for i in range(1, len(parts)):
            if "/".join(parts[i:]) in rels:
                counts["/".join(parts[:i])] += 1
                break
    return counts.most_common(1)[0][0] if counts else None


def dot(a: List[float], b: List[float]) -> float:
    return sum(map(operator.mul, a, b))


class Recommender:
    """Candidatos = temas del índice cuya copia del paquete está en la colección de Traktor. update()
    calcula la lista para un tema principal; snapshot() la entrega al botón."""

    def __init__(self, items: List[Item], entries: Dict[str, ET.Element], n: int = DEFAULT_N,
                 weights: Optional[Dict[str, float]] = None):
        self.n = n
        self.weights = weights_from(weights)
        by_norm = {norm(k): e for k, e in entries.items()}
        self.root = package_root(items, by_norm.keys())
        self.by_suffix: Dict[str, Item] = {}
        for it in items:
            for key in (norm(it.rel), norm(it.original)):
                if key:
                    self.by_suffix.setdefault(key, it)
        self.candidates: List[Item] = []
        for it in items:
            e = by_norm.get(f"{self.root}/{norm(it.rel)}") if self.root else None
            if e is None:
                continue
            it.entry, it.path = e, entry_path(e)
            m = e.find("MUSICAL_KEY")
            k = traktor_key(m.get("VALUE")) if m is not None else "?"
            if k != "?":
                it.key, it.key_src = k, "traktor"
            tempo = e.find("TEMPO")
            try:
                bpm = float(tempo.get("BPM")) if tempo is not None and tempo.get("BPM") else 0.0
            except ValueError:
                bpm = 0.0
            if bpm > 0:
                it.bpm, it.bpm_src = bpm, "traktor"
            self.candidates.append(it)
        self.lock = threading.Lock()
        self._tracks: List[Track] = []
        self._info: dict = {}
        self._principal: Optional[dict] = None

    def summary(self) -> str:
        c = self.candidates
        src = Counter(it.key_src if it.key != "?" else "sin" for it in c)
        return (f"{len(c)} temas sugeribles (paquete {self.root or '?'} en la colección de Traktor); tonalidad "
                f"de Traktor {src.get('traktor', 0)}, de tags {src.get('tags', 0)}, sin tonalidad {src.get('sin', 0)}")

    def identify(self, path: str) -> Optional[Item]:
        for s in suffixes(path):
            it = self.by_suffix.get(s)
            if it is not None:
                return it
        return None

    def compute(self, principal: dict, played: Iterable[str] = ()) -> Tuple[List[Track], dict]:
        t0 = time.perf_counter()
        main = self.identify(principal.get("path") or "")
        tempo = float(principal.get("tempo") or 1.0) or 1.0
        base = float(principal.get("bpm") or 0.0)
        if base <= 0 and main is not None and main.bpm:
            base = main.bpm
        bpm0 = base * tempo if base > 0 else None
        key0 = deck_key(principal.get("key"))
        if key0 == "?" and main is not None:
            key0 = main.key
        excluded = {it.uid for it in (self.identify(p) for p in played) if it is not None}
        if main is not None:
            excluded.add(main.uid)
        use_clap = main is not None and main.vec is not None
        weights = dict(self.weights)
        if not use_clap and sum(w for k, w in weights.items() if k != "clap") == 0:
            weights = {"bpm": 1.0}                      # sin CLAP y sin otros pesos: el BPM más cercano
        rows = []
        no_key = 0
        for it in self.candidates:
            if it.uid in excluded:
                continue
            if bpm0 is not None and (it.bpm is None or abs(it.bpm - bpm0) > BPM_RANGE):
                continue
            score, shift, _rel = harmonic.best_key_shift(key0, it.key)
            if key0 != "?":
                if it.key == "?":
                    no_key += 1
                    continue
                if score < KEY_MIN:
                    continue
            if use_clap and it.vec is None:
                continue
            sim = dot(main.vec, it.vec) if use_clap else None
            comps = {
                "clap": None if sim is None else min(1.0, max(0.0, sim)),
                "bpm": None if bpm0 is None or it.bpm is None else max(0.0, 1.0 - abs(it.bpm - bpm0) / BPM_RANGE),
                "clave": score if key0 != "?" and it.key != "?" else None,
                "genero": (1.0 if it.genre == main.genre else 0.0)
                if main is not None and main.genre and it.genre else None,
                "tiv": None,
            }
            final = weighted(comps, weights)
            rows.append((it, sim, shift, comps, -1.0 if final is None else final))
        rows.sort(key=lambda r: -r[4])
        top = rows[: self.n]
        info = {"modo": "clap" if use_clap else "sin_clap",
                "principal": {"path": principal.get("path"), "uid": main.uid if main else None,
                              "bpm": None if bpm0 is None else round(bpm0, 2), "clave": key0,
                              "genero": main.genre if main else None},
                "pesos": weights, "candidatos": len(rows), "sin_tonalidad": no_key, "excluidos": len(excluded),
                "uids": [r[0].uid for r in top],
                "puntaje": [round(r[4], 4) for r in top],
                "componentes": {k: [None if r[3][k] is None else round(r[3][k], 4) for r in top]
                                for k in COMPONENTS if any(r[3][k] is not None for r in top)},
                "parecido": [None if r[1] is None else round(r[1], 4) for r in top],
                "semitonos": [r[2] for r in top],
                "ms": round((time.perf_counter() - t0) * 1000, 1)}
        return [r[0].track() for r in top], info

    def update(self, principal: dict, played: Iterable[str] = ()) -> dict:
        tracks, info = self.compute(principal, list(played))
        with self.lock:
            self._principal, self._tracks, self._info = dict(principal), tracks, info
        return info

    def refresh(self, played: Iterable[str] = ()) -> Optional[dict]:
        """Recalcula para el mismo tema principal (un tema pasó a tocado)."""
        with self.lock:
            principal = self._principal
        return None if principal is None else self.update(principal, played)

    def snapshot(self) -> Tuple[List[Track], dict]:
        with self.lock:
            return list(self._tracks), dict(self._info)


def load_recommender(entries: Dict[str, ET.Element], index_path: Path = DEFAULT_INDEX,
                     n: int = DEFAULT_N, weights: Optional[Dict[str, float]] = None) -> Tuple[Optional[Recommender], str]:
    """(Recommender, texto para el aviso). None si no hay índice o no se puede leer."""
    if not Path(index_path).exists():
        return None, (f"No hay índice de recomendación ({index_path}): las sugerencias son al azar. Armarlo una "
                      f"vez con el Python del pipeline: python tools/traktor_live/indice_recomendacion.py")
    try:
        data, items = load_index(Path(index_path))
    except (OSError, ValueError, KeyError) as exc:
        return None, f"No pude leer el índice {index_path}: {type(exc).__name__}: {exc}"
    rec = Recommender(items, entries, n, weights)
    if not rec.candidates:
        return None, (f"Ningún tema del índice está en la colección de Traktor: las sugerencias son al azar "
                      f"(¿Traktor tiene la carpeta del paquete en su colección?)")
    stale = ""
    package_csv = Path(str(data.get("paquete") or "")) / "_organizacion.csv"
    try:
        if data.get("paquete") and package_csv.exists() and \
                package_csv.stat().st_mtime > time.mktime(time.strptime(data["creado"], "%Y-%m-%dT%H:%M:%S")):
            stale = " ÍNDICE VIEJO: la biblioteca cambió después; volver a correr indice_recomendacion.py."
    except (OSError, ValueError, KeyError):
        pass
    pesos = ", ".join(f"{k} {w:g}" for k, w in rec.weights.items() if w > 0)
    return rec, f"Recomendación por {data.get('rep', '?')}: {rec.summary()}. Pesos: {pesos}.{stale}"
