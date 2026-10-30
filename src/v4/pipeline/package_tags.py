"""
PURPOSE: Etiquetas y carátula de los temas de un paquete portátil (usb_package.py) para que Rekordbox,
         Traktor y el explorador de archivos muestren bien cada tema. Trabaja sobre las copias del
         paquete (pendrive o carpeta en disco), nunca sobre la biblioteca original. Tres pasos:
           - lookup: busca cada tema de _organizacion.csv en Beatport (beatport.py: misma caché y
             mismas reglas de match que beatport_lookup.py) y, si Beatport no lo confirma, en Deezer
             (solo carátula y álbum: artista, título y mezcla verificados). -> lookup.csv
           - plan: valor nuevo de cada campo al lado del actual -> plan.csv + summary.json, y baja
             las carátulas (600x600, JPEG limpio) a covers/. No toca el audio.
               * Title siempre del archivo, limpio (número de pista, "(9A)", catálogo, "Artista - "
                 repetido); el de Beatport queda de referencia (trae errores de tipeo).
               * Beatport confirmado (A, o B con sello, fecha o duración): Artist, Remixers, Label y
                 Released con la convención de DECISIONS 2026-09-29; Album = lanzamiento si el
                 archivo no tiene (o tiene el sello como álbum).
               * Sin Beatport confirmado: artista limpio (vacío -> nombre de archivo); la mezcla que
                 solo está en el nombre de archivo pasa a Remixers; álbum de Deezer si falta.
               * Genre no cambia salvo que esté vacío. El comentario (tonalidad, energía, Vocal) no
                 se toca.
               * Carátula solo si falta o está rota (no se decodifica, o WEBP): Beatport confirmado,
                 Deezer verificado, o Beatport B sin confirmar (mismo artista, título y mezcla; se
                 cuenta aparte). Una "image/jpg" válida queda con el tipo "image/jpeg".
             Title = "Título (Mezcla)" sin "Original Mix" (Gabriel, 2026-10-30); --title-mix separate
             deja la mezcla solo en Remixers.
           - write: aplica plan.csv a una o varias copias del paquete (--target). Por archivo:
             respaldo JSON de lo que cambia (y de la carátula vieja), escritura con la versión ID3
             del archivo, relectura y control de que el audio sin tags no cambió. WAV no se escribe
             (Traktor no lee sus tags; el texto llega por el XML/NML).
CHANGELOG:
  - 2026-10-30: Creación inicial (pendrive para Rekordbox con controladora: metadata y carátulas).
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.beatport import (  # noqa: E402
    BeatportBlocked, BeatportClient, clean_text, is_version_text, mix_core, norm, split_local_artists,
    split_title_mix, strip_feat,
)
from src.v4.common.config_loader import load_config  # noqa: E402
from src.v4.common.path_resolver import resolve_artifacts_root  # noqa: E402
from src.v4.common.tags import audio_payload_hash, read_release_tags, read_tags  # noqa: E402
from src.v4.pipeline.beatport_lookup import _confident, build_queries, lookup, record  # noqa: E402

FIELDS = ("title", "artist", "remixers", "album", "label", "released", "genre")
ID3_FRAMES = {"title": "TIT2", "artist": "TPE1", "remixers": "TPE4", "album": "TALB", "label": "TPUB",
              "released": "TDRC", "genre": "TCON"}
VORBIS_KEYS = {"title": "title", "artist": "artist", "remixers": "remixer", "album": "album", "label": "label",
               "released": "date", "genre": "genre"}
WRITABLE = (".mp3", ".flac", ".aif", ".aiff")
COVER_PX = 600
DEEZER_SEARCH = "https://api.deezer.com/search"
_EXTRA_JUNK = [re.compile(p, re.I) for p in (
    r"\bvk\.com\S*", r"\bwww\.\S+", r"\S+\.(?:com|net|org|ru)\b",
)]


def _s(v) -> str:
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()


# ---------------------------------------------------------------------------
# Estado de la carátula de un archivo
# ---------------------------------------------------------------------------

def _pictures(audio) -> list:
    pics = list(getattr(audio, "pictures", None) or [])              # FLAC
    tags = getattr(audio, "tags", None)
    if tags is not None and hasattr(tags, "getall"):                  # ID3
        pics += tags.getall("APIC")
    return pics


def image_format(data: bytes) -> str:
    """'JPEG', 'PNG', 'WEBP'... o 'BAD' si no se puede decodificar."""
    from PIL import Image
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            return im.format or "BAD"
    except Exception:  # noqa: BLE001
        return "BAD"


def cover_state(path: Path) -> str:
    """ok | mime (JPEG válido con tipo 'image/jpg' u otro) | none | bad (rota o formato no soportado)."""
    import mutagen
    try:
        audio = mutagen.File(str(path))
    except Exception:  # noqa: BLE001
        return "none"
    pics = _pictures(audio) if audio is not None else []
    if not pics:
        return "none"
    states = []
    for p in pics:
        fmt = image_format(bytes(p.data)) if p.data else "BAD"
        if fmt == "JPEG":
            states.append("ok" if p.mime == "image/jpeg" else "mime")
        elif fmt == "PNG":
            states.append("ok" if p.mime == "image/png" else "mime")
        else:
            states.append("bad")
    if "bad" in states:
        return "bad"
    return "mime" if "mime" in states else "ok"


# ---------------------------------------------------------------------------
# Limpieza local de artista y título (puro, testeable)
# ---------------------------------------------------------------------------

def _clean(s: str) -> str:
    s = clean_text(s)
    for rx in _EXTRA_JUNK:
        s = rx.sub(" ", s)
    s = re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", s))       # "( Original Mix )" -> "(Original Mix)"
    return re.sub(r"\s+", " ", s).strip(" -_")


def _amp(artist: str) -> str:
    """'Nu  Jo Ke': el '&' quitado del nombre de archivo deja dos espacios entre artistas. Si los
    nombres ya van separados por comas ('Louie Vega, The Martinez  Brothers'), solo es un espacio de más."""
    return re.sub(r"(?<=\S) {2,}(?=\S)", " " if "," in (artist or "") else " & ", artist or "")


def _amp_in_mix(title: str) -> str:
    """'(Maceo Plex  Shall Ocin Remix)' -> '(Maceo Plex & Shall Ocin Remix)': solo entre paréntesis."""
    return re.sub(r"[\(\[][^\(\)\[\]]*[\)\]]",
                  lambda m: re.sub(r"(?<=\w) {2,}(?=\w)", " & ", m.group(0)), title or "")


_LOWER_VERSION = re.compile(r"\b(remix|mix|edit|dub|version|rework|extended|original|radio|club|instrumental|"
                            r"vip|bootleg|refix)\b")


def tidy_mix(mix: str) -> str:
    """'original mix' -> 'Original Mix', 'Tripswitch remix' -> 'Tripswitch Remix' (solo las palabras de
    versión escritas en minúscula; 'Cosmjn EDIT' queda igual)."""
    return _LOWER_VERSION.sub(lambda m: m.group(1).capitalize(), mix or "")


def is_junk(value: str) -> bool:
    """El valor trae algo que la limpieza saca: '(4A) Circulation', '06. E-Mov', 'Kolombo  Malikk'."""
    return bool(value) and _clean(_amp(value)) != re.sub(r"\s+", " ", value).strip()


def _split(text: str) -> Tuple[str, str]:
    head, sep, tail = (text or "").partition(" - ")
    if sep:
        return _clean(_amp(head)), _clean(tail)
    return "", _clean(text)


_ARTIST_SEP = re.compile(r"\s*(?:,|;|/|&|\band\b|\bfeat\b\.?|\bft\b\.?|\bvs\b\.?|\bx\b)\s*", re.I)


def _same_artist(a: str, b: str) -> bool:
    """Mismo artista: igual normalizado, un nombre en común, o casi igual ('Sbastien Lger')."""
    if not norm(a) or not norm(b):
        return False
    if norm(a) == norm(b) or SequenceMatcher(None, norm(a), norm(b)).ratio() >= 0.8:
        return True
    names = lambda s: {norm(x) for x in _ARTIST_SEP.split(s) if norm(x)}  # noqa: E731
    return bool(names(a) & names(b))


def split_mix(title: str) -> Tuple[str, str]:
    """'Tema (Mezcla)' / 'Tema - Mezcla' -> (tema, mezcla); conserva 'feat.' en el título."""
    m = re.match(r"^(.*?)\s*[\(\[]([^\(\)\[\]]+)[\)\]]\s*$", title or "")
    if m and m.group(1).strip() and is_version_text(m.group(2)):
        return m.group(1).strip(), m.group(2).strip()
    m = re.match(r"^(.*?)\s+-\s+(.+)$", title or "")
    if m and m.group(1).strip() and is_version_text(m.group(2)):
        return m.group(1).strip(), m.group(2).strip()
    return (title or "").strip(), ""


def local_fields(tag_artist: str, tag_title: str, tag_remixer: str, filename: str) -> Dict[str, str]:
    """Artista, título base y mezcla desde los tags actuales, con el nombre de archivo de respaldo.
    Saca número de pista, '(9A)', catálogo y 'Artista - ' repetido en el título; la mezcla que solo
    está en el nombre de archivo ('Heliopolis' / '... (Silinder Remix).mp3') se recupera."""
    f_artist, f_title = _split(_amp_in_mix(Path(filename).stem))
    artist = _clean(_amp(tag_artist or ""))
    tag_title = _amp_in_mix(tag_title or "")
    title = _clean(tag_title)
    t_artist, t_title = _split(tag_title)
    if t_artist and t_title and (not artist or _same_artist(t_artist, artist) or _same_artist(t_artist, f_artist)):
        artist = artist or t_artist
        title = t_title
    artist = artist or f_artist
    title = title or f_title
    base, mix = split_mix(title)
    if not mix:
        f_base, f_mix = split_mix(f_title)
        if f_mix and SequenceMatcher(None, norm(f_base), norm(base)).ratio() >= 0.85:
            mix = f_mix
    return {"artist": artist, "title": base, "mix": tidy_mix(mix)}


def title_with_mix(base: str, mix: str) -> str:
    """'Children' + 'Extended Mix' -> 'Children (Extended Mix)'; 'Original Mix' no se agrega."""
    if not mix or norm(mix) in ("original mix", "original"):
        return base
    if norm(mix) and norm(mix) in norm(base):
        return base
    return f"{base} ({mix})"


# ---------------------------------------------------------------------------
# Deezer (respaldo para carátula y álbum)
# ---------------------------------------------------------------------------

class DeezerClient:
    """Búsqueda pública de Deezer con caché en disco por consulta."""

    def __init__(self, cache_dir: Path, min_interval: float = 0.25, offline: bool = False):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval, self.offline = min_interval, offline
        self._last, self._session, self.n_requests = 0.0, None, 0

    def search(self, q: str) -> List[dict]:
        q = re.sub(r"\s+", " ", q or "").strip()
        if not q:
            return []
        path = self.cache_dir / f"{hashlib.sha1(q.encode('utf-8')).hexdigest()}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["results"]
        if self.offline:
            return []
        import requests
        if self._session is None:
            self._session = requests.Session()
        data = None
        for attempt in range(4):
            wait = self._last + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.n_requests += 1
            try:
                r = self._session.get(DEEZER_SEARCH, params={"q": q}, timeout=20)
                data = r.json()
            except Exception:  # noqa: BLE001
                time.sleep(2 * (attempt + 1))
                continue
            if isinstance(data, dict) and "error" in data:     # cuota: esperar y repetir
                time.sleep(5 * (attempt + 1))
                data = None
                continue
            break
        if data is None:
            return []
        results = [{"id": x.get("id"), "title": x.get("title") or "", "title_short": x.get("title_short") or "",
                    "title_version": x.get("title_version") or "", "duration": x.get("duration"),
                    "artist": (x.get("artist") or {}).get("name") or "",
                    "album": (x.get("album") or {}).get("title") or "",
                    "cover": (x.get("album") or {}).get("cover_xl") or ""}
                   for x in (data.get("data") or [])[:25]]
        path.write_text(json.dumps({"query": q, "results": results}, ensure_ascii=False), encoding="utf-8")
        return results


def deezer_pick(artist: str, title: str, mix: str, duration_s: Optional[float],
                candidates: List[dict]) -> Optional[dict]:
    """Candidato de Deezer del mismo tema: artista del archivo, título parecido (>= 0.85) y la misma
    mezcla (sin contar extended/original/radio) o la misma duración (+-3 s). El de duración más cercana."""
    local = split_local_artists(artist) or [artist]
    ok = []
    for c in candidates:
        if not c.get("cover"):
            continue
        if not any(norm(a) and (norm(a) == norm(c["artist"]) or f" {norm(c['artist'])} " in f" {norm(a)} ")
                   for a in local):
            continue
        base_c, mix_c = split_title_mix(c["title"])
        if SequenceMatcher(None, norm(strip_feat(title)), norm(strip_feat(base_c))).ratio() < 0.85:
            continue
        same_dur = duration_s is not None and c.get("duration") and abs(float(c["duration"]) - duration_s) <= 3
        if mix_core(mix) != mix_core(mix_c) and not same_dur:
            continue
        ok.append((not same_dur, abs(float(c.get("duration") or 0) - (duration_s or 0)), c))
    return min(ok, key=lambda x: (x[0], x[1]))[2] if ok else None


# ---------------------------------------------------------------------------
# Paquete: lista de temas
# ---------------------------------------------------------------------------

def read_package(package: Path) -> pd.DataFrame:
    """Una fila por tema de _organizacion.csv (track_uid único), con la ruta del archivo en el paquete."""
    df = pd.read_csv(package / "_organizacion.csv", encoding="utf-8-sig", dtype=str).fillna("")
    df = df.drop_duplicates("track_uid").reset_index(drop=True)
    df["path"] = [str(package / a) for a in df["archivo"]]
    return df


def current_tags(path: Path) -> Dict[str, str]:
    t, r = read_tags(path), read_release_tags(path)
    out = {"tag_artist": _s(t["tag_artist"]), "tag_title": _s(t["tag_title"]), "tag_album": _s(t["tag_album"]),
           "tag_genre": _s(t["tag_genre"]), "tag_label": _s(t["tag_label"]), "tag_remixer": _s(r["tag_remixer"]),
           "tag_isrc": _s(r["tag_isrc"]), "tag_date": _s(r["tag_date"])}
    out["cover_state"] = cover_state(path)
    return out


# ---------------------------------------------------------------------------
# lookup
# ---------------------------------------------------------------------------

def run_lookup(package: Path, out_dir: Path, artifacts_root: Path, folder: Optional[str] = None,
               limit: Optional[int] = None, offline: bool = False, min_interval: float = 1.2) -> int:
    pkg = read_package(package)
    if folder:
        pkg = pkg[pkg["archivo"].str.contains(folder, case=False, regex=False)]
    if limit:
        pkg = pkg.head(limit)
    bp = BeatportClient(artifacts_root.parent / "beatport_cache", min_interval=min_interval, offline=offline)
    dz = DeezerClient(artifacts_root.parent / "deezer_cache", offline=offline)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "lookup.csv"
    done: Dict[str, dict] = {}
    if out_csv.exists():   # reanudable
        for r in pd.read_csv(out_csv, encoding="utf-8-sig", dtype=str).fillna("").to_dict("records"):
            done[r["track_uid"]] = r
    print(f"[lookup] {len(pkg)} temas ({len(done)} ya hechos) -> {out_csv}", flush=True)
    rows: List[dict] = []
    t0, status = time.monotonic(), 0
    try:
        for i, p in enumerate(pkg.itertuples(), 1):
            if p.track_uid in done:
                rows.append(done[p.track_uid])
                continue
            cur = current_tags(Path(p.path))
            dur = float(p.duracion_s) if p.duracion_s else None
            row = pd.Series({"track_uid": p.track_uid, "rel_path": p.archivo, "filename": p.nombre_original,
                             "duration_s": dur, **cur})
            rel = {"tag_remixer": cur["tag_remixer"], "tag_isrc": cur["tag_isrc"], "tag_date": cur["tag_date"]}
            variants = build_queries(row, rel)
            if variants:
                res, q, used = lookup(bp, variants)
            else:
                from src.v4.common.beatport import MatchResult, TrackQuery
                res, q, used = MatchResult(level="D"), TrackQuery(artists="", title=""), []
            rec = record(row, rel, q, res, used)
            b = res.best or {}
            rec.update({"archivo": p.archivo, "nombre_original": p.nombre_original, "duration_s": dur,
                        "cur_album": cur["tag_album"], "cover_state": cur["cover_state"],
                        "bp_release_name": b.get("release_name"), "bp_image": b.get("release_image")})
            loc = local_fields(cur["tag_artist"], cur["tag_title"], cur["tag_remixer"], p.nombre_original)
            rec.update({"loc_artist": loc["artist"], "loc_title": loc["title"], "loc_mix": loc["mix"]})
            if not _confident(res) and loc["artist"] and loc["title"]:
                first = (split_local_artists(loc["artist"]) or [loc["artist"]])[0]
                cands = dz.search(f"{first} {loc['title']}")
                pick = deezer_pick(loc["artist"], loc["title"], loc["mix"], dur, cands)
                if pick is None and loc["mix"]:
                    cands = dz.search(f"{first} {loc['title']} {' '.join(sorted(mix_core(loc['mix'])))}")
                    pick = deezer_pick(loc["artist"], loc["title"], loc["mix"], dur, cands)
                if pick:
                    rec.update({"dz_id": pick["id"], "dz_title": pick["title"], "dz_artist": pick["artist"],
                                "dz_album": pick["album"], "dz_cover": pick["cover"], "dz_duration": pick["duration"]})
            rows.append(rec)
            if i % 25 == 0 or i == len(pkg):
                el = time.monotonic() - t0
                lv = pd.Series([str(r.get("level")) for r in rows]).value_counts().to_dict()
                print(f"[lookup] {i}/{len(pkg)} beatport={bp.n_requests} deezer={dz.n_requests} "
                      f"{el / 60:.1f} min niveles={lv}", flush=True)
            if i % 50 == 0:
                pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    except BeatportBlocked as exc:
        print(f"[ERROR] {exc}. Guardo lo hecho ({len(rows)}); la caché permite retomar.")
        status = 2
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    return status


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def plan_row(r: dict, title_mix: str = "in-title") -> Dict[str, str]:
    """Valores nuevos de una fila de lookup.csv. '' = no se toca. cover_url / cover_source para la carátula."""
    confirmed = _truthy(r.get("confirmed")) and r.get("level") in ("A", "B")
    new = {f: "" for f in FIELDS}
    cur = {"title": _s(r.get("cur_title")), "artist": _s(r.get("cur_artist")), "remixers": _s(r.get("cur_remixers")),
           "album": _s(r.get("cur_album")), "label": _s(r.get("cur_label")), "released": _s(r.get("cur_released")),
           "genre": _s(r.get("cur_genre"))}
    if confirmed:
        # Título del archivo (limpio): el de Beatport queda como referencia (DECISIONS 2026-09-29).
        # Si el archivo nombra otra mezcla que Beatport ('Kofi (CIOZ Remix)' con el ISRC del original),
        # manda el archivo. Artist, Label y Released de Beatport solo donde falta o hay basura: un
        # valor válido no se pisa (Beatport puede devolver otra edición: un compilado, otro sello).
        base, loc_mix, bp_mix = _s(r.get("loc_title")), _s(r.get("loc_mix")), _s(r.get("new_remixers"))
        mix = bp_mix if not loc_mix or mix_core(loc_mix) == mix_core(bp_mix) else loc_mix
        if not cur["remixers"]:
            new["remixers"] = mix
        if not cur["artist"] or is_junk(cur["artist"]):
            new["artist"] = _s(r.get("new_artist")) or _s(r.get("loc_artist"))
        if not cur["label"]:
            new["label"] = _s(r.get("new_label"))
        if not cur["released"]:
            new["released"] = _s(r.get("new_released"))
        in_rmx = mix_core(new["remixers"] or cur["remixers"]) == mix_core(loc_mix)
        if title_mix == "in-title":
            new["title"] = title_with_mix(base, loc_mix or mix)
        else:
            new["title"] = base if not loc_mix or in_rmx else title_with_mix(base, loc_mix)
        rel_name = _s(r.get("bp_release_name"))
        if rel_name and (not cur["album"] or norm(cur["album"]) == norm(cur["label"])
                         or norm(cur["album"]) == norm(_s(r.get("new_label")))):
            new["album"] = rel_name
        if not cur["genre"] and _s(r.get("new_genre")):
            new["genre"] = _s(r.get("new_genre"))
        source = f"beatport_{r.get('level')}"
    else:
        mix, base = _s(r.get("loc_mix")), _s(r.get("loc_title"))
        if mix and not cur["remixers"]:
            new["remixers"] = mix
        # Separado: la mezcla sale del título solo si queda en Remixers (si Remixers dice otra cosa, se conserva)
        mix_kept = not mix or new["remixers"] == mix or mix_core(cur["remixers"]) == mix_core(mix)
        separate = title_mix == "separate" and mix_kept and norm(mix) not in ("original mix", "original")
        new.update(artist=_s(r.get("loc_artist")), title=base if separate or not mix else title_with_mix(base, mix))
        if _s(r.get("dz_album")) and not cur["album"]:
            new["album"] = _s(r.get("dz_album"))
        source = "deezer" if _s(r.get("dz_cover")) else "local"
    # Solo lo que cambia de verdad
    for f in FIELDS:
        if not new[f]:
            continue
        same = cur[f][:10] == new[f][:10] if f == "released" else cur[f] == new[f]
        if same:
            new[f] = ""
    cover_url, cover_source = "", ""
    state = _s(r.get("cover_state"))
    if state in ("none", "bad"):
        img = _s(r.get("bp_image"))
        if confirmed and img:
            cover_url, cover_source = img.replace("{w}", str(COVER_PX)).replace("{h}", str(COVER_PX)), "beatport"
        elif _s(r.get("dz_cover")):
            cover_url, cover_source = _s(r.get("dz_cover")), "deezer"
        elif r.get("level") == "B" and img:
            # mismo artista, título y mezcla, sin sello/fecha/duración iguales: otra edición del tema
            cover_url = img.replace("{w}", str(COVER_PX)).replace("{h}", str(COVER_PX))
            cover_source = "beatport_B_sin_confirmar"
    return {**{f"new_{f}": new[f] for f in FIELDS}, "source": source, "cover_url": cover_url,
            "cover_source": cover_source}


def fetch_cover(url: str, cache_dir: Path) -> Optional[Path]:
    """Baja y normaliza una carátula: JPEG RGB de hasta 600x600 sin metadatos. None si falla."""
    import requests
    from PIL import Image
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{hashlib.sha1(url.encode('utf-8')).hexdigest()}.jpg"
    if out.exists():
        return out
    if not url.startswith("https://"):
        return None
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "traktor-ml/1.0"})
            if r.status_code != 200 or len(r.content) > 15 * 1024 * 1024:
                return None
            with Image.open(io.BytesIO(r.content)) as im:
                if im.format not in ("JPEG", "PNG", "WEBP") or im.width * im.height > 40_000_000:
                    return None
                im.load()
                rgb = im.convert("RGB")
            rgb.thumbnail((COVER_PX, COVER_PX))
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=90, optimize=True)
            out.write_bytes(buf.getvalue())
            return out
        except Exception:  # noqa: BLE001
            time.sleep(1 + attempt)
    return None


def run_plan(out_dir: Path, artifacts_root: Path, title_mix: str = "in-title") -> int:
    lk = pd.read_csv(out_dir / "lookup.csv", encoding="utf-8-sig", dtype=str).fillna("")
    rows = []
    covers = artifacts_root.parent / "cover_cache"
    for i, r in enumerate(lk.to_dict("records"), 1):
        # la limpieza local se recalcula acá: cambiar sus reglas no obliga a repetir las búsquedas
        loc = local_fields(r.get("cur_artist", ""), r.get("cur_title", ""), "", r.get("nombre_original", ""))
        r.update(loc_artist=loc["artist"], loc_title=loc["title"], loc_mix=loc["mix"])
        p = plan_row(r, title_mix)
        cover_file = ""
        if p["cover_url"]:
            f = fetch_cover(p["cover_url"], covers)
            cover_file = str(f) if f else ""
            if not f:
                p["cover_source"] = f"falló:{p['cover_source']}"
        rows.append({"track_uid": r["track_uid"], "archivo": r["archivo"], "level": r["level"],
                     "confirmed": r["confirmed"], "cover_state": r["cover_state"],
                     **{f"cur_{f}": _s(r.get(f"cur_{f}")) for f in FIELDS}, **p, "cover_file": cover_file,
                     "bp_url": r.get("bp_url", ""), "dz_title": r.get("dz_title", ""),
                     "dz_artist": r.get("dz_artist", "")})
        if i % 100 == 0:
            print(f"[plan] {i}/{len(lk)}", flush=True)
    plan = pd.DataFrame(rows)
    plan.to_csv(out_dir / "plan.csv", index=False, encoding="utf-8-sig")
    summary = summarize_plan(plan)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def summarize_plan(plan: pd.DataFrame) -> dict:
    n = len(plan)
    out = {"n_tracks": n, "levels": plan["level"].value_counts().to_dict(),
           "confirmed": int(plan["confirmed"].map(_truthy).sum()),
           "source": plan["source"].value_counts().to_dict(),
           "cover_state_before": plan["cover_state"].value_counts().to_dict(),
           "cover_new": plan["cover_source"].replace("", "ninguna").value_counts().to_dict()}
    out["changes"] = {f: int((plan[f"new_{f}"].fillna("") != "").sum()) for f in FIELDS}
    out["fill_empty"] = {f: int(((plan[f"new_{f}"].fillna("") != "") & (plan[f"cur_{f}"].fillna("") == "")).sum())
                         for f in FIELDS}
    has_cover = plan["cover_state"].isin(["ok", "mime"]) | (plan["cover_file"].fillna("") != "")
    out["with_cover_after"] = int(has_cover.sum())
    out["files_touched"] = int(((plan[[f"new_{f}" for f in FIELDS]].fillna("") != "").any(axis=1)
                                | (plan["cover_file"].fillna("") != "") | (plan["cover_state"] == "mime")).sum())
    return out


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------

def audio_hash(path: Path) -> str:
    """Hash del audio para controlar la escritura. En AIFF, el de los bloques COMM y SSND: el
    audio_payload_hash del repo incluye ahí los tags (es el track_uid; DECISIONS 2026-09-29, punto 7)."""
    if Path(path).suffix.lower() not in (".aif", ".aiff"):
        return audio_payload_hash(path)
    import struct
    h = hashlib.sha256()
    with open(path, "rb") as f:
        head = f.read(12)
        if head[:4] != b"FORM":
            return audio_payload_hash(path)
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            cid, size = hdr[:4], struct.unpack(">I", hdr[4:])[0]
            if cid in (b"COMM", b"SSND"):
                h.update(cid)
                remaining = size
                while remaining > 0:
                    chunk = f.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    h.update(chunk)
                    remaining -= len(chunk)
                f.seek(size & 1, 1)
            else:
                f.seek(size + (size & 1), 1)
    return h.hexdigest()


def _apic(data: bytes):
    from mutagen.id3 import APIC
    return APIC(encoding=0, mime="image/jpeg", type=3, desc="", data=data)


def write_file(path: Path, changes: Dict[str, str], cover: Optional[bytes], fix_mime: bool) -> dict:
    """Escribe los campos de changes (texto no vacío) y la carátula. Devuelve el respaldo de lo que había."""
    import mutagen
    suffix = path.suffix.lower()
    backup: dict = {"fields": {}, "pictures": None}   # pictures: None = la carátula no se toca
    if suffix == ".flac":
        from mutagen.flac import FLAC, Picture
        audio = FLAC(str(path))
        if audio.tags is None:
            audio.add_tags()
        for f, v in changes.items():
            k = VORBIS_KEYS[f]
            backup["fields"][f] = list(audio.tags.get(k, []))
            audio.tags[k] = [v]
        if cover is not None or fix_mime:
            backup["pictures"] = [{"mime": p.mime, "type": p.type, "desc": p.desc,
                                   "data": base64.b64encode(p.data).decode()} for p in audio.pictures]
        if cover is not None:
            audio.clear_pictures()
            pic = Picture()
            pic.type, pic.mime, pic.desc, pic.data = 3, "image/jpeg", "", cover
            audio.add_picture(pic)
        elif fix_mime:
            for p in audio.pictures:
                fmt = image_format(p.data)
                p.mime = {"JPEG": "image/jpeg", "PNG": "image/png"}.get(fmt, p.mime)
        audio.save()
        return backup
    from mutagen import id3
    audio = mutagen.File(str(path))
    if audio is None:
        raise ValueError(f"no se pudo abrir {path}")
    if audio.tags is None:
        audio.add_tags()
    tags = audio.tags
    version = tags.version[1] if getattr(tags, "version", None) and tags.version[1] in (3, 4) else 3
    for f, v in changes.items():
        fid = ID3_FRAMES[f]
        backup["fields"][f] = [str(x) for fr in tags.getall(fid) for x in fr.text]
        tags.delall(fid)
        tags.add(getattr(id3, fid)(encoding=3, text=[v]))
    if cover is not None or fix_mime:
        backup["pictures"] = [{"mime": p.mime, "type": p.type, "desc": p.desc,
                               "data": base64.b64encode(p.data).decode()} for p in tags.getall("APIC")]
    if cover is not None:
        tags.delall("APIC")
        tags.add(_apic(cover))
    elif fix_mime:
        for p in tags.getall("APIC"):
            fmt = image_format(p.data)
            p.mime = {"JPEG": "image/jpeg", "PNG": "image/png"}.get(fmt, p.mime)
    audio.save(v2_version=version)
    return backup


def run_write(plan_csv: Path, targets: List[Path], backup_root: Path, folder: Optional[str] = None,
              dry_run: bool = False) -> int:
    plan = pd.read_csv(plan_csv, encoding="utf-8-sig", dtype=str).fillna("")
    if folder:
        plan = plan[plan["archivo"].str.contains(folder, case=False, regex=False)]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    status = 0
    for target in targets:
        bdir = backup_root / f"{re.sub(r'[^A-Za-z0-9]+', '_', str(target)).strip('_')}_{stamp}"
        log: List[dict] = []
        backups: Dict[str, dict] = {}
        n_ok = n_err = n_same = 0
        for i, r in enumerate(plan.to_dict("records"), 1):
            path = target / r["archivo"]
            rec = {"track_uid": r["track_uid"], "archivo": r["archivo"]}
            try:
                if not path.exists():
                    raise FileNotFoundError("no existe")
                if path.suffix.lower() not in WRITABLE:
                    rec["status"] = f"formato {path.suffix.lower()} sin escribir"
                    log.append(rec)
                    continue
                cur = current_tags(path)
                cur_vals = {"title": cur["tag_title"], "artist": cur["tag_artist"], "remixers": cur["tag_remixer"],
                            "album": cur["tag_album"], "label": cur["tag_label"], "released": cur["tag_date"],
                            "genre": cur["tag_genre"]}
                changes = {}
                for f in FIELDS:
                    v = r[f"new_{f}"]
                    if v and not (cur_vals[f][:10] == v[:10] if f == "released" else cur_vals[f] == v):
                        changes[f] = v
                cover = None
                if r["cover_file"] and cur["cover_state"] in ("none", "bad"):
                    cover = Path(r["cover_file"]).read_bytes()
                fix_mime = cur["cover_state"] == "mime" and cover is None
                rec["changes"] = ", ".join(list(changes) + (["cover"] if cover else []) + (["mime"] if fix_mime else []))
                if not changes and cover is None and not fix_mime:
                    rec["status"] = "sin cambios"
                    n_same += 1
                    log.append(rec)
                    continue
                if dry_run:
                    rec["status"] = "simulado"
                    log.append(rec)
                    continue
                before = audio_hash(path)
                backups[r["track_uid"]] = {"archivo": r["archivo"], **write_file(path, changes, cover, fix_mime)}
                after = audio_hash(path)
                if before != after:
                    raise RuntimeError("el audio cambió al escribir los tags")
                rec["status"] = "ok"
                n_ok += 1
            except Exception as exc:  # noqa: BLE001
                rec["status"] = f"error: {type(exc).__name__}: {exc}"
                n_err += 1
                status = 1
            log.append(rec)
            if i % 100 == 0 or i == len(plan):
                print(f"[write] {target}: {i}/{len(plan)} ok={n_ok} sin cambios={n_same} errores={n_err}", flush=True)
                if backups and not dry_run:
                    bdir.mkdir(parents=True, exist_ok=True)
                    (bdir / "backup.json").write_text(json.dumps(backups, ensure_ascii=False), encoding="utf-8")
        bdir.mkdir(parents=True, exist_ok=True)
        if backups:
            (bdir / "backup.json").write_text(json.dumps(backups, ensure_ascii=False), encoding="utf-8")
        pd.DataFrame(log).to_csv(bdir / "write_log.csv", index=False, encoding="utf-8-sig")
        print(f"[write] {target}: ok={n_ok} sin cambios={n_same} errores={n_err}; respaldo y log en {bdir}")
    return status


def run_revert(backup_json: Path, target: Path) -> int:
    """Devuelve los campos y carátulas guardados en backup.json."""
    data = json.loads(backup_json.read_text(encoding="utf-8"))
    import mutagen
    from mutagen import id3
    n = 0
    for uid, b in data.items():
        path = target / b["archivo"]
        if path.suffix.lower() == ".flac":
            from mutagen.flac import FLAC, Picture
            audio = FLAC(str(path))
            for f, vals in b["fields"].items():
                k = VORBIS_KEYS[f]
                if vals:
                    audio.tags[k] = vals
                elif k in audio.tags:
                    del audio.tags[k]
            if b.get("pictures") is not None:     # None = la carátula no se tocó
                audio.clear_pictures()
                for p in b["pictures"]:
                    pic = Picture()
                    pic.type, pic.mime, pic.desc, pic.data = p["type"], p["mime"], p["desc"], base64.b64decode(p["data"])
                    audio.add_picture(pic)
            audio.save()
        else:
            audio = mutagen.File(str(path))
            tags = audio.tags
            version = tags.version[1] if tags.version[1] in (3, 4) else 3
            for f, vals in b["fields"].items():
                fid = ID3_FRAMES[f]
                tags.delall(fid)
                if vals:
                    tags.add(getattr(id3, fid)(encoding=3, text=vals))
            if b.get("pictures") is not None:     # None = la carátula no se tocó
                tags.delall("APIC")
                for p in b["pictures"]:
                    tags.add(id3.APIC(encoding=3, mime=p["mime"], type=p["type"], desc=p["desc"],
                                      data=base64.b64decode(p["data"])))
            audio.save(v2_version=version)
        n += 1
    print(f"[revert] {n} archivos restaurados en {target}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Etiquetas y carátulas de los temas de un paquete portátil")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("lookup", "plan", "write", "revert"):
        p = sub.add_parser(name)
        p.add_argument("--out-dir", default=None, help="Carpeta de lookup.csv / plan.csv "
                       "(por defecto artifacts/v4/package_tags/<paquete>, junto a beatport_cache)")
        p.add_argument("--config", default=None)
        if name == "lookup":
            p.add_argument("--package", required=True, help="Carpeta del paquete (la que tiene _organizacion.csv)")
            p.add_argument("--folder", default=None)
            p.add_argument("--limit", type=int, default=None)
            p.add_argument("--offline", action="store_true")
            p.add_argument("--min-interval", type=float, default=1.2)
            p.add_argument("--low-priority", action="store_true")
        if name == "plan":
            p.add_argument("--package", required=True)
            p.add_argument("--title-mix", choices=("in-title", "separate"), default="in-title")
        if name == "write":
            p.add_argument("--package", required=True, help="Paquete del que salió el plan")
            p.add_argument("--target", action="append", required=True, help="Copia del paquete a escribir (repetible)")
            p.add_argument("--folder", default=None)
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--low-priority", action="store_true")
        if name == "revert":
            p.add_argument("--package", required=True)
            p.add_argument("--backup", required=True)
            p.add_argument("--target", required=True)
    args = ap.parse_args()
    if getattr(args, "low_priority", False):
        from src.v4.pipeline.extract_representations import lower_priority
        lower_priority()
    config = load_config(Path(args.config) if args.config else None)
    artifacts_root = resolve_artifacts_root(config)
    package = Path(args.package)
    out_dir = Path(args.out_dir) if args.out_dir else artifacts_root.parent / "package_tags" / re.sub(
        r"[^A-Za-z0-9]+", "_", package.name).strip("_")
    if args.cmd == "lookup":
        return run_lookup(package, out_dir, artifacts_root, args.folder, args.limit, args.offline, args.min_interval)
    if args.cmd == "plan":
        return run_plan(out_dir, artifacts_root, args.title_mix)
    if args.cmd == "write":
        return run_write(out_dir / "plan.csv", [Path(t) for t in args.target], out_dir / "backups",
                         args.folder, args.dry_run)
    return run_revert(Path(args.backup), Path(args.target))


if __name__ == "__main__":
    sys.exit(main())
