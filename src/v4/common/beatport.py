"""
PURPOSE: Beatport como fuente de metadatos (género, sello, fecha de lanzamiento, remixers) sin la
         API oficial: lee el JSON que la búsqueda pública de beatport.com embebe en la página
         (__NEXT_DATA__). Tres piezas:
           - BeatportClient.search(q): resultados de /search/tracks resumidos, con caché en disco por
             consulta (una consulta nunca se repite) y una pausa mínima entre pedidos.
           - match(query, candidatos): decide el nivel del match
               A = ISRC exacto;
               B = misma versión: título parecido, al menos un artista en común y la misma mezcla
                   sin contar extended/original/radio/edit (el corte puede ser otro);
               C = solo otras versiones (remix de otra persona): no hereda el género;
               D = nada.
             Entre candidatos del mismo nivel prefiere sello igual, fecha igual, duración igual y el
             lanzamiento más antiguo (el original antes que los compilados).
           - proposed_tags(...): convención de Beatport para los tags. Artist = artistas sin los
             remixers; Remixers = nombre de la mezcla si es un remix ("Adana Twins Remix"),
             "Extended Mix" si el archivo es la Extended, si no "Original Mix"; Label, Genre y
             Released (fecha AAAA-MM-DD).
CHANGELOG:
  - 2026-09-29: Creación inicial.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Sequence

SEARCH_URL = "https://www.beatport.com/search/tracks"
TRACK_URL = "https://www.beatport.com/track/{slug}/{track_id}"
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
MAX_RESULTS = 60           # Beatport ordena por relevancia; más abajo casi nunca está el tema
TITLE_MIN_SIM = 0.85
SAME_CUT_TOLERANCE_S = 3.0

_NEXT_DATA_RE = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)
_VERSION_HINT = re.compile(
    r"\b(remix|rmx|mix|edit|dub|version|rework|bootleg|vip|extended|original|instrumental|refix|"
    r"reprise|re-?edit|flip|remaster(?:ed)?|a+c+ap+el+a|a\s+cap+el+a)\b", re.I)
_FEAT = re.compile(r"\s*[\(\[]\s*(?:feat|ft|featuring)\b\.?[^\)\]]*[\)\]]|\s+(?:feat|ft|featuring)\b\.?\s+.*$", re.I)
# Palabras de una mezcla que no cambian la producción (corte, remasterización)
_NEUTRAL_MIX_WORDS = {"extended", "original", "mix", "version", "radio", "edit", "club", "remaster",
                      "remastered", "full", "length", "album", "single", "the", "12", "7", "inch"}
_LABEL_SUFFIXES = {"records", "record", "recordings", "recording", "recs", "rec", "music", "label",
                   "productions", "production", "ltd", "inc", "digital", "audio"}
_GENERIC_ARTIST_WORDS = {"feat", "with", "band", "music", "project", "orchestra", "remix", "presents",
                         "collective", "sound", "sounds", "crew", "brothers", "sisters", "house"}
_EXTENDED = re.compile(r"\bextended\b", re.I)
_CUT_WORDS = re.compile(r"\b(extended|radio|club)\b\s*", re.I)
_JUNK = [re.compile(p, re.I) for p in (
    r"^\s*\d{1,3}\s*[.\-_)]\s+",          # "09. " número de pista
    r"[\(\[]\s*\d{1,2}[AB]\s*[\)\]]",     # "(6A)" Camelot de Mixed In Key
    r"\s*-\s*from YouTube\s*$",
    r"\[[^\]]*\]",                        # "[DA09]" catálogo, "[free download]"
    r"\bfree\s+download\b",
    r"[\(\[]\s*(?:official\s+)?(?:music\s+)?(?:video|audio|lyric\s+video|lyrics|visuali[sz]er|clip)\s*[\)\]]",
    r"^\s*premiere\s*:\s*",               # "Premiere: Josh Baker - ..."
)]


class BeatportBlocked(RuntimeError):
    """Beatport respondió 403/429 de forma sostenida: cortar la corrida sin martillar."""


# ---------------------------------------------------------------------------
# Normalización (puro, testeable)
# ---------------------------------------------------------------------------

_TRANSLIT = str.maketrans({"ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ß": "ss",
                           "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "þ": "th", "Þ": "TH", "ð": "d"})


def norm(s: Optional[str]) -> str:
    """Minúsculas ASCII sin puntuación: 'Shlømo' == 'Shlomo', 'Kröcher' == 'Krocher'."""
    s = unicodedata.normalize("NFKD", (s or "").translate(_TRANSLIT)).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def strip_feat(s: Optional[str]) -> str:
    return _FEAT.sub("", s or "").strip()


def clean_text(s: Optional[str]) -> str:
    s = (s or "").replace(" | ", " - ")
    for rx in _JUNK:
        s = rx.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip(" -_")


def is_version_text(s: Optional[str]) -> bool:
    """¿Nombra una versión ('Remix', 'Extended Mix', 'Dub', 'Rework')?"""
    return bool(_VERSION_HINT.search(s or ""))


def split_title_mix(title: Optional[str]) -> tuple[str, str]:
    """'Roar (Adana Twins Remix)' / 'Love's Theme - Myd Remix' -> (título, mezcla).
    Un paréntesis sin palabras de versión es parte del título: 'The Drums (Din Daa Daa)'."""
    t = strip_feat(clean_text(title))
    m = re.match(r"^(.*?)\s*[\(\[]([^\(\)\[\]]+)[\)\]]\s*(?:(?:19|20)\d\d)?\s*$", t)  # "(... mix) 2011"
    if m and m.group(1).strip() and _VERSION_HINT.search(m.group(2)):
        return m.group(1).strip(), m.group(2).strip()
    m = re.match(r"^(.*?)\s+-\s+(.+)$", t)
    if m and _VERSION_HINT.search(m.group(2)):
        return m.group(1).strip(), m.group(2).strip()
    return t, ""


def mix_core(mix: Optional[str]) -> frozenset:
    """Lo que identifica la producción: 'Myd Extended Remix' y 'Myd Remix' -> {myd, remix};
    'Original Mix', 'Extended Mix' y 'Radio Edit' -> vacío."""
    words = {"remix" if w == "rmx" else w for w in norm(mix).split()} - _NEUTRAL_MIX_WORDS
    return frozenset(w for w in words if not re.fullmatch(r"(19|20)\d\d", w))


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(strip_feat(a)), norm(strip_feat(b))).ratio()


def artist_names_in(names: Sequence[str], text: str) -> List[str]:
    hay = f" {norm(text)} "
    return [n for n in names if norm(n) and f" {norm(n)} " in hay]


def split_local_artists(s: Optional[str]) -> List[str]:
    parts = re.split(r"\s*(?:,|;|/|\bfeat\b\.?|\bft\b\.?|\bfeaturing\b|\bvs\b\.?|\bx\b)\s*",
                     strip_feat(s or ""), flags=re.I)
    return [p.strip() for p in parts if p and p.strip()]


# ---------------------------------------------------------------------------
# Resultados de Beatport
# ---------------------------------------------------------------------------

def parse_search_html(html: str) -> List[dict]:
    """Lista cruda de temas de la página de búsqueda (vacía si la estructura no aparece)."""
    m = _NEXT_DATA_RE.search(html)
    if not m:
        return []
    data = json.loads(m.group(1))
    for q in ((data.get("props") or {}).get("pageProps") or {}).get("dehydratedState", {}).get("queries", []):
        payload = (q.get("state") or {}).get("data")
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            return [x for x in payload["data"] if isinstance(x, dict) and "track_id" in x]
    return []


def summarize(raw: dict) -> dict:
    arts = raw.get("artists") or []
    genres = [g.get("genre_name") for g in raw.get("genre") or [] if g.get("genre_name")]
    length = raw.get("length")
    title = (raw.get("track_name") or "").strip()
    return {
        "track_id": int(raw["track_id"]),
        "title": title,
        "mix": (raw.get("mix_name") or "").strip(),
        "artists": [a["artist_name"] for a in arts if a.get("artist_type_name") == "Artist"],
        "remixers": [a["artist_name"] for a in arts if a.get("artist_type_name") == "Remixer"],
        "label": ((raw.get("label") or {}).get("label_name") or "").strip() or None,
        "release_date": (raw.get("release_date") or "")[:10] or None,
        "release_name": (raw.get("release") or {}).get("release_name"),
        "genre": genres[0] if genres else None,
        "isrc": (raw.get("isrc") or "").strip().upper() or None,
        "bpm": raw.get("bpm"),
        "key": raw.get("key_name"),
        "length_s": round(length / 1000.0, 1) if isinstance(length, (int, float)) else None,
        "is_dj_edit": bool(raw.get("is_dj_edit")),
        "is_ugc_remix": bool(raw.get("is_ugc_remix")),
        "url": TRACK_URL.format(slug=(norm(title).replace(" ", "-") or "track"), track_id=raw["track_id"]),
    }


class BeatportClient:
    """Búsqueda de temas con caché en disco. offline=True solo lee la caché."""

    def __init__(self, cache_dir: Path, min_interval: float = 1.2, offline: bool = False):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self.offline = offline
        self._last = 0.0
        self._session = None
        self.n_requests = 0

    def _path(self, q: str) -> Path:
        return self.cache_dir / f"{hashlib.sha1(q.encode('utf-8')).hexdigest()}.json"

    def search(self, q: str) -> List[dict]:
        q = re.sub(r"\s+", " ", q or "").strip()
        if not q:
            return []
        path = self._path(q)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["results"]
        if self.offline:
            return []
        html = self._get(q)
        results = [summarize(r) for r in parse_search_html(html)[:MAX_RESULTS]]
        path.write_text(json.dumps({"query": q, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                    "results": results}, ensure_ascii=False), encoding="utf-8")
        return results

    def _get(self, q: str) -> str:
        import requests
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})
        backoff = 10.0
        for _ in range(5):
            wait = self._last + self.min_interval + random.uniform(0, 0.4) - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.n_requests += 1
            try:
                r = self._session.get(SEARCH_URL, params={"q": q}, timeout=30)
            except requests.RequestException:
                time.sleep(backoff)
                backoff *= 2
                continue
            if r.status_code in (403, 429, 503):
                time.sleep(backoff)
                backoff *= 2
                continue
            r.raise_for_status()
            return r.text
        raise BeatportBlocked(f"Beatport no respondió bien tras 5 intentos (q={q!r})")


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

@dataclass
class TrackQuery:
    artists: str                      # como está en el archivo: "Peer Kusiv, Lenny"
    title: str                        # título base, sin la mezcla
    mix: str = ""                     # "Adana Twins Remix", "Original Mix", "" si no se sabe
    isrcs: tuple = ()
    label: Optional[str] = None
    date: Optional[str] = None        # "2018-11-12" o "2018"
    duration_s: Optional[float] = None

    def search_queries(self) -> List[str]:
        """ISRC primero; después artista + título; por último con la mezcla, para traer remixes."""
        first = (split_local_artists(self.artists) or [""])[0]
        core = " ".join(sorted(mix_core(self.mix)))
        qs = [i for i in self.isrcs if i]
        qs += [f"{strip_feat(self.artists)} {self.title}", f"{first} {self.title} {core}".strip(),
               f"{first} {self.title}"]
        return list(dict.fromkeys(re.sub(r"\s+", " ", q).strip() for q in qs if q and q.strip()))


def effective_mix(c: dict) -> str:
    """Mezcla real de un candidato: Beatport a veces pone el remix en el título
    ('Klimax (Patrice Baumel Remix)' + 'Extended Mix')."""
    _, in_title = split_title_mix(c["title"])
    return f"{in_title} {c['mix']}".strip() if in_title else c["mix"]


def _artist_score(q: TrackQuery, c: dict) -> float:
    bp = list(c["artists"]) + list(c["remixers"])
    local = split_local_artists(q.artists)
    if not bp or not local:
        return 0.0
    context = f"{q.artists} {q.mix}"  # sin el título: "The Light" no es el artista de "The Light"
    bp_cover = len(artist_names_in(bp, context)) / len(bp)
    local_cover = len(artist_names_in(local, " ".join(bp))) / len(local)
    return (bp_cover + local_cover) / 2


def _artist_token_overlap(q: TrackQuery, c: dict) -> bool:
    """Alias del mismo artista ('Darco (IL)' / 'DARCO 09'): comparten una palabra distintiva."""
    def toks(s: str) -> set:
        return {w for w in norm(s).split() if len(w) >= 4 and w not in _GENERIC_ARTIST_WORDS}
    return bool(toks(q.artists) & toks(" ".join(list(c["artists"]) + list(c["remixers"]))))


def norm_label(s: Optional[str]) -> str:
    """'Subliminal Records' ~ 'Subliminal'; 'Full Time Records' ~ 'Fulltime Production'."""
    return "".join(w for w in norm(s).split() if w not in _LABEL_SUFFIXES)


def _label_match(q: TrackQuery, c: dict) -> bool:
    return bool(q.label and c["label"] and norm_label(q.label) and norm_label(q.label) == norm_label(c["label"]))


def _date_match(q: TrackQuery, c: dict) -> bool:
    if not q.date or not c["release_date"]:
        return False
    d = str(q.date).strip()[:10]
    return c["release_date"].startswith(d) if len(d) >= 7 else c["release_date"][:4] == d[:4]


def _same_cut(q: TrackQuery, c: dict) -> bool:
    return (q.duration_s is not None and c["length_s"] is not None
            and abs(float(q.duration_s) - float(c["length_s"])) <= SAME_CUT_TOLERANCE_S)


def _confirmed(q: TrackQuery, c: dict) -> bool:
    return _label_match(q, c) or _date_match(q, c) or _same_cut(q, c)


def _artist_set(c: dict) -> frozenset:
    return frozenset(norm(n) for n in list(c["artists"]) + list(c["remixers"]))


@dataclass
class MatchResult:
    level: str                                   # A | B | C | D
    best: Optional[dict] = None
    n_candidates: int = 0
    n_pool: int = 0
    same_version_genres: List[str] = field(default_factory=list)  # mismos artistas, título y mezcla
    hint_genres: List[str] = field(default_factory=list)          # géneros de otras versiones (nivel C)
    label_match: bool = False
    date_match: bool = False
    same_cut: bool = False
    title_sim: float = 0.0
    artist_score: float = 0.0                    # 0 si el artista solo coincide como alias
    original_genre: Optional[str] = None         # género del lanzamiento más antiguo de esa versión

    @property
    def genre_conflict(self) -> bool:
        return len(self.same_version_genres) > 1


def match(q: TrackQuery, candidates: Sequence[dict]) -> MatchResult:
    uniq = {c["track_id"]: c for c in candidates}.values()
    scored = []
    for c in uniq:
        if c["is_dj_edit"] or c["is_ugc_remix"]:
            continue
        base, _ = split_title_mix(c["title"])
        scored.append((c, title_similarity(q.title, base), _artist_score(q, c)))
    res = MatchResult(level="D", n_candidates=len(uniq))

    isrcs = {i.strip().upper() for i in q.isrcs if i}
    exact = [(c, t, a) for c, t, a in scored if c["isrc"] and c["isrc"] in isrcs and t >= 0.6]
    # Artista por nombre completo; por alias solo con título casi idéntico y sello/fecha/duración
    pool = [(c, t, a) for c, t, a in scored if t >= TITLE_MIN_SIM and (
        a > 0 or (t >= 0.95 and _artist_token_overlap(q, c) and _confirmed(q, c)))]
    res.n_pool = len(pool)
    target = mix_core(q.mix)
    same = [(c, t, a) for c, t, a in pool if mix_core(effective_mix(c)) == target]

    def rank(item):
        c, t, a = item
        earliest_first = -int((c["release_date"] or "9999-99-99").replace("-", ""))
        return (_label_match(q, c), _date_match(q, c), _same_cut(q, c), round(a, 2), round(t, 2), earliest_first)

    if exact:
        group, res.level = exact, "A"
    elif same:
        group, res.level = same, "B"
    elif pool:
        res.level = "C"
        res.hint_genres = sorted({c["genre"] for c, _, _ in pool if c["genre"]})
        return res
    else:
        return res

    best, t, a = max(group, key=rank)
    res.best, res.title_sim, res.artist_score = best, t, a
    same_artists = [c for c, _, _ in group if c["genre"] and _artist_set(c) == _artist_set(best)]
    res.same_version_genres = sorted({c["genre"] for c in same_artists})
    # Beatport asigna el género por lanzamiento: el mismo tema en un compilado puede tener otro
    if same_artists:
        res.original_genre = min(same_artists, key=lambda c: c["release_date"] or "9999")["genre"]
    res.label_match = _label_match(q, best)
    res.date_match = _date_match(q, best)
    res.same_cut = _same_cut(q, best)
    return res


# ---------------------------------------------------------------------------
# Tags propuestos (convención de Beatport)
# ---------------------------------------------------------------------------

def remixers_tag(mix: str, same_cut: bool, local_mix: str = "") -> str:
    """Remix -> nombre de la mezcla ('Adana Twins Remix'); si el archivo es otro corte que el de
    Beatport, sin extended/radio/club ('Myd Extended Remix' -> 'Myd Remix').
    No remix -> 'Extended Mix' si el archivo es la Extended (mismo corte que una Extended de
    Beatport, o su tag actual lo dice), si no 'Original Mix'."""
    if mix_core(mix):
        return mix if same_cut else re.sub(r"\s+", " ", _CUT_WORDS.sub("", mix)).strip()
    if same_cut and _EXTENDED.search(mix or ""):
        return mix
    if not same_cut and _EXTENDED.search(local_mix or ""):
        return "Extended Mix"
    return "Original Mix"


def proposed_tags(q: TrackQuery, res: MatchResult) -> Dict[str, Optional[str]]:
    """Tags según Beatport para un match A/B. Artist sin los remixers, en el orden del archivo."""
    c = res.best
    if c is None:
        return {}
    mix = effective_mix(c)
    remixers = remixers_tag(mix, res.same_cut or res.level == "A", q.mix)
    remixer_names = {norm(r) for r in c["remixers"]}
    is_remix = bool(mix_core(mix))

    def is_remixer(name: str) -> bool:  # tipado como Remixer, o nombrado en la mezcla de un remix
        return norm(name) in remixer_names or (is_remix and f" {norm(name)} " in f" {norm(mix)} ")

    artists = [n for n in c["artists"] if not is_remixer(n)] or list(c["artists"])  # nunca vacío
    local_order = norm(q.artists)
    artists.sort(key=lambda n: local_order.find(norm(n)) if norm(n) in local_order else 10_000)
    base, _ = split_title_mix(c["title"])
    return {
        "artist": ", ".join(artists),
        "title": base,
        "remixers": remixers,
        "label": c["label"],
        "genre": c["genre"],
        "released": c["release_date"],
    }
