"""
PURPOSE: Temas repetidos: la regla que decidió Gabriel (DECISIONS 2026-09-29, informe
         docs/reports/temas_repetidos_2026-09-29.md).
           - Qué copia se queda (choose_keeper): siempre la MP3 de 320 kbps; si no hay, la de mejor
             calidad (sin pérdida antes que comprimida; entre comprimidas, más kbps). Un empate de
             calidad no se resuelve solo: se pregunta.
           - Decisiones (artifacts/v4/datasets/<dataset>/duplicate_decisions.json): qué copias salen, a
             favor de cuál, por qué capa y quién decidió; también los pares que Gabriel dijo que son
             distintos, para no volver a preguntar. Las copias descartadas no entran a ninguna
             organización (organize.Library las excluye).
           - Detección (find_candidates), por capas:
               sonido:   coseno CLAP >= 0.98 y duración a <= 5 s -> casi idéntico (se resuelve solo si la
                         calidad decide);
               parecido: coseno entre 0.97 y 0.98, duración a <= 5 s -> se pregunta (Abotha, 0.979, era
                         el mismo; dos tramos de un set continuo, 0.972, no);
               nombre: mismo artista, título y mezcla, duración a <= 2 s, coseno >= 0.90 -> se pregunta;
               corte:  mismo artista, título y mezcla con otra duración (edit contra original), coseno
                       >= 0.90 -> es duplicado (regla d) pero se pregunta cuál queda.
             El hash exacto de audio ya lo resuelve Phase 0 (duplicates.csv).
CHANGELOG:
  - 2026-09-29: Creación inicial.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

DECISIONS_FILE = "duplicate_decisions.json"
SOUND_COS, SOUND_DUR = 0.98, 5.0
LIKE_COS = 0.97
NAME_COS, NAME_DUR = 0.90, 2.0
MP3_TOP_KBPS = 315          # "MP3 320": CBR 320 (mutagen informa 320; margen por redondeo)
LOSSLESS = {"wav", "flac", "aif", "aiff"}

_VERSION = re.compile(r"\b(mix|remix|rmx|edit|dub|version|rework|remaster(ed)?|vip|bootleg|instrumental|variation)\b",
                      re.IGNORECASE)
_BRACKETS = re.compile(r"[(\[]([^()\[\]]+)[)\]]")
_PLAIN_MIX = re.compile(r"\b(original|extended|radio|club|main|album|edit|mix|version|remaster(ed)?|\d{4})\b")


# ---------------------------------------------------------------------------
# Nombres
# ---------------------------------------------------------------------------

def bracket_mix(*texts: str) -> str:
    """Primer paréntesis o corchete que nombra una versión ('Adana Twins Remix Two'), buscando en orden."""
    for text in texts:
        found = [m.strip() for m in _BRACKETS.findall(text or "") if _VERSION.search(m)]
        if found:
            return found[0]
    return ""


def norm_text(s: Optional[str]) -> str:
    """Minúsculas, sin acentos, sin 'feat. …' ni puntuación: 'Iñaky García' -> 'inaky garcia'."""
    s = "".join(c for c in unicodedata.normalize("NFKD", str(s or "")) if not unicodedata.combining(c)).lower()
    s = re.sub(r"\b(feat|ft|featuring)\b\.?.*$", "", s)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def base_title(title: Optional[str]) -> str:
    """Título sin paréntesis ni 'Artista - ' delante (hay tags de título con 'Artista - Título')."""
    t = _BRACKETS.sub(" ", str(title or ""))
    if " - " in t:
        t = t.split(" - ", 1)[1]
    return norm_text(t)


def norm_mix(mix: Optional[str]) -> str:
    """La mezcla sin las palabras de corte o edición: 'Extended Mix', 'Original Mix' y '' quedan iguales."""
    return " ".join(_PLAIN_MIX.sub(" ", norm_text(mix)).split())


# ---------------------------------------------------------------------------
# Calidad y copia que se queda
# ---------------------------------------------------------------------------

def read_quality(path) -> Dict:
    """Formato, kbps y si es sin pérdida. kbps de un sin pérdida = bits * frecuencia * canales."""
    path = Path(path)
    fmt = path.suffix.lower().lstrip(".")
    out = {"fmt": fmt, "kbps": 0, "lossless": fmt in LOSSLESS}
    try:
        import mutagen
        info = mutagen.File(str(path)).info
        if out["lossless"]:
            bits = getattr(info, "bits_per_sample", 16) or 16
            out["kbps"] = int(bits * info.sample_rate * getattr(info, "channels", 2) / 1000)
        else:
            out["kbps"] = int(round(getattr(info, "bitrate", 0) / 1000))
            out["lossless"] = getattr(info, "codec", "") == "alac"
    except Exception:  # noqa: BLE001 (un archivo ilegible queda último)
        pass
    return out


def quality_rank(q: Dict) -> Tuple[int, int, int]:
    """Mayor = mejor: MP3 de 320 primero (regla de Gabriel); después sin pérdida; después kbps."""
    top = int(q["fmt"] == "mp3" and q["kbps"] >= MP3_TOP_KBPS)
    return top, int(bool(q["lossless"])), int(q["kbps"])


def quality_label(q: Dict) -> str:
    return f"{q['fmt'].upper()} {'sin pérdida' if q['lossless'] else str(q['kbps']) + 'k'}"


def choose_keeper(qualities: Sequence[Dict]) -> Optional[int]:
    """Índice de la copia que se queda, o None si las mejores empatan en calidad (se pregunta)."""
    ranks = [quality_rank(q) for q in qualities]
    best = max(ranks)
    winners = [i for i, r in enumerate(ranks) if r == best]
    return winners[0] if len(winners) == 1 else None


# ---------------------------------------------------------------------------
# Decisiones
# ---------------------------------------------------------------------------

def load_decisions(artifacts: Path) -> List[Dict]:
    p = Path(artifacts) / DECISIONS_FILE
    return json.loads(p.read_text(encoding="utf-8")).get("decisions", []) if p.exists() else []


def save_decisions(artifacts: Path, decisions: List[Dict]) -> None:
    p = Path(artifacts) / DECISIONS_FILE
    p.write_text(json.dumps({"decisions": decisions}, indent=2, ensure_ascii=False), encoding="utf-8")


def add_decision(artifacts: Path, verdict: str, tracks: List[Dict], keep: Optional[str] = None, layer: str = "",
                 by: str = "gabriel", note: str = "") -> Dict:
    """verdict 'mismo' (keep = track_uid que se queda; el resto sale) o 'distintos'. tracks: [{track_uid, rel_path}].
    Una decisión nueva sobre los mismos temas reemplaza a la anterior."""
    if verdict not in ("mismo", "distintos"):
        raise ValueError(f"veredicto inválido: {verdict}")
    uids = {t["track_uid"] for t in tracks}
    if verdict == "mismo" and keep not in uids:
        raise ValueError("La copia que se queda tiene que ser uno de los temas del grupo")
    entry = {"verdict": verdict, "tracks": [{"track_uid": t["track_uid"], "rel_path": t["rel_path"]} for t in tracks],
             "keep": keep if verdict == "mismo" else None, "layer": layer, "by": by,
             "date": dt.date.today().isoformat(), "note": note}
    decisions = [d for d in load_decisions(artifacts) if not uids & {t["track_uid"] for t in d["tracks"]}]
    decisions.append(entry)
    save_decisions(artifacts, decisions)
    return entry


def dropped(artifacts: Path) -> Dict[str, str]:
    """{track_uid descartado: track_uid que se queda} de las decisiones 'mismo'."""
    out = {}
    for d in load_decisions(artifacts):
        if d["verdict"] == "mismo":
            for t in d["tracks"]:
                if t["track_uid"] != d["keep"]:
                    out[t["track_uid"]] = d["keep"]
    return out


def decided_pairs(decisions: Iterable[Dict]) -> Set[frozenset]:
    """Pares ya decididos (en cualquier sentido): no se vuelven a preguntar."""
    out = set()
    for d in decisions:
        u = [t["track_uid"] for t in d["tracks"]]
        out |= {frozenset((a, b)) for i, a in enumerate(u) for b in u[i + 1:]}
    return out


# ---------------------------------------------------------------------------
# Detección
# ---------------------------------------------------------------------------

def find_candidates(uids: Sequence[str], emb: np.ndarray, dur: Sequence[float], artist: Sequence[str],
                    title: Sequence[str], mix: Sequence[str], skip: Set[frozenset] = frozenset(),
                    focus: Optional[Set[str]] = None) -> List[Dict]:
    """Grupos de posibles repetidos: [{layer, members (índices), sim}]. layer del grupo = la más dudosa de
    sus pares (sonido < nombre < parecido < corte). focus: solo grupos que tocan alguno de esos uids (música nueva).
    skip: pares ya decididos."""
    X = np.asarray(emb, dtype=np.float64)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    S = X @ X.T
    dur = np.asarray(dur, dtype=float)
    key = [f"{norm_text(a)}|{base_title(t)}|{norm_mix(m)}" for a, t, m in zip(artist, title, mix)]
    named = [bool(base_title(t)) for t in title]
    n = len(uids)
    order = {"sonido": 0, "nombre": 1, "parecido": 2, "corte": 3}
    edges: Dict[Tuple[int, int], str] = {}
    for i in range(n):
        for j in range(i + 1, n):
            if frozenset((uids[i], uids[j])) in skip:
                continue
            if focus is not None and uids[i] not in focus and uids[j] not in focus:
                continue
            dd, c = abs(dur[i] - dur[j]), S[i, j]
            if c >= SOUND_COS and dd <= SOUND_DUR:
                edges[(i, j)] = "sonido"
            elif named[i] and key[i] == key[j] and c >= NAME_COS:
                edges[(i, j)] = "nombre" if dd <= NAME_DUR else "corte"
            elif c >= LIKE_COS and dd <= SOUND_DUR:
                edges[(i, j)] = "parecido"
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j in edges:
        parent[find(i)] = find(j)
    groups: Dict[int, List[int]] = {}
    for i, j in edges:
        groups.setdefault(find(i), [])
    for i in range(n):
        if find(i) in groups:
            groups[find(i)].append(i)
    out = []
    for members in groups.values():
        pairs = [(a, b) for a in members for b in members if a < b and (a, b) in edges]
        layer = max((edges[p] for p in pairs), key=order.get)
        out.append({"layer": layer, "members": members, "sim": round(float(max(S[a, b] for a, b in pairs)), 3),
                    "dur_diff": round(float(max(abs(dur[a] - dur[b]) for a in members for b in members)), 1)})
    return sorted(out, key=lambda g: (order[g["layer"]], -g["sim"]))


def resolve(group: Dict, qualities: Sequence[Dict]) -> Tuple[Optional[int], bool]:
    """(copia que se queda o None, se resuelve sola). Solo 'sonido' con calidad distinta se resuelve sola."""
    keep = choose_keeper(qualities)
    return keep, group["layer"] == "sonido" and keep is not None
