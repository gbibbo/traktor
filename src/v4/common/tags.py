"""
PURPOSE: Tags de audio para V4 (mutagen), para correr sin Essentia en Windows.
         - read_tags: artista, título, álbum, género, sello, BPM, tonalidad y la energía que
           Mixed In Key escribe en el comentario ("8A - Energy 6").
         - audio_payload_hash: SHA256 del audio sin los bloques de tags (ID3v2/ID3v1/APEv2 en
           MP3, bloques de metadata en FLAC), para que editar un tag no cambie el track_uid ni
           invalide las cachés de embeddings.
         - with_token / write_comment: marca idempotente en el comentario (etiqueta "Vocal"),
           que Rekordbox y Traktor muestran en la columna Comments.
         - read_release_tags: mezcla (Remixer), ISRC, fecha e ID de Spotify, para buscar en Beatport.
         - read_cover: carátula incluida en el archivo o imagen de su carpeta.
CHANGELOG:
  - 2026-09-27: Creación inicial (MVP de biblioteca completa en Windows).
  - 2026-09-29: read_release_tags (búsqueda en Beatport), sin cambiar read_tags ni el catálogo.
  - 2026-09-29: read_cover (carátula del archivo o de la carpeta) para la columna de la app.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.v4.common.harmonic import to_camelot

_MIK_RE = re.compile(r"^\s*(\d{1,2}[AB])(?:\s*/\s*\d{1,2}[AB])?\s*-\s*Energy\s*(\d{1,2})", re.IGNORECASE)
_ENERGY_RE = re.compile(r"\bEnergy\s*(\d{1,2})\b", re.IGNORECASE)
# Comentarios técnicos de iTunes que no son el comentario visible
_SKIP_COMM_DESC = {"itunnorm", "itunsmpb", "itunpgap", "itunes_cddb_ids"}

WRITABLE_SUFFIXES = (".mp3", ".aif", ".aiff", ".flac")


# ---------------------------------------------------------------------------
# Parsing puro (testeable sin archivos)
# ---------------------------------------------------------------------------

def parse_bpm(value) -> Optional[float]:
    """'128', '127.98', '1280' (x10) -> float en [40, 300]; None si no es interpretable."""
    if value is None:
        return None
    try:
        bpm = float(str(value).strip().replace(",", "."))
    except ValueError:
        return None
    if 300 < bpm < 3000 and 40 <= bpm / 10 <= 300:
        bpm = bpm / 10
    return round(bpm, 2) if 40 <= bpm <= 300 else None


def parse_mik_comment(comment: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    """'8A - Energy 6' -> ('8A', 6). '10B/7A - Energy 6' -> ('10B', 6). Otros -> (None, energía o None)."""
    if not comment:
        return None, None
    m = _MIK_RE.match(comment)
    if m:
        cam = to_camelot(m.group(1))
        return (cam if cam != "?" else None), int(m.group(2))
    e = _ENERGY_RE.search(comment)
    return None, (int(e.group(1)) if e else None)


def with_token(comment: Optional[str], token: str, sep: str = " - ") -> str:
    """Agrega token al final del comentario si no está ya como palabra. Idempotente."""
    base = (comment or "").strip()
    if re.search(rf"(?<![\w]){re.escape(token)}(?![\w])", base, re.IGNORECASE):
        return base
    return f"{base}{sep}{token}" if base else token


# ---------------------------------------------------------------------------
# Lectura
# ---------------------------------------------------------------------------

def _id3_text(tags, frame_id: str) -> Optional[str]:
    frame = tags.get(frame_id)
    if frame is None or not getattr(frame, "text", None):
        return None
    return str(frame.text[0]).strip() or None


def _id3_comment(tags) -> Optional[str]:
    best = None
    for key in tags.keys():
        if not key.startswith("COMM"):
            continue
        frame = tags[key]
        if str(getattr(frame, "desc", "")).lower() in _SKIP_COMM_DESC:
            continue
        text = str(frame.text[0]).strip() if frame.text else ""
        if getattr(frame, "desc", "") == "":
            return text or None
        best = best or (text or None)
    return best


def _vorbis(tags, *names) -> Optional[str]:
    for n in names:
        vals = tags.get(n)
        if vals:
            return str(vals[0]).strip() or None
    return None


def _mp4(tags, *names) -> Optional[str]:
    for n in names:
        vals = tags.get(n)
        if vals:
            v = vals[0]
            if isinstance(v, bytes):
                v = v.decode("utf-8", "ignore")
            return str(v).strip() or None
    return None


def read_tags(path: Path) -> Dict[str, object]:
    """Tags normalizados con prefijo tag_ más mik_camelot / mik_energy. Nunca lanza."""
    out: Dict[str, object] = {k: None for k in (
        "tag_artist", "tag_title", "tag_album", "tag_genre", "tag_label",
        "tag_bpm", "tag_key", "tag_comment", "tag_camelot", "mik_energy")}
    try:
        import mutagen
        audio = mutagen.File(str(path))
    except Exception:  # noqa: BLE001
        return out
    tags = getattr(audio, "tags", None) if audio is not None else None
    if tags is None:
        return out

    cls = type(tags).__name__
    if cls == "ID3" or hasattr(tags, "getall"):
        out.update(tag_artist=_id3_text(tags, "TPE1"), tag_title=_id3_text(tags, "TIT2"),
                   tag_album=_id3_text(tags, "TALB"), tag_genre=_id3_text(tags, "TCON"),
                   tag_label=_id3_text(tags, "TPUB"), tag_bpm=_id3_text(tags, "TBPM"),
                   tag_key=_id3_text(tags, "TKEY"), tag_comment=_id3_comment(tags))
    elif cls == "MP4Tags":
        out.update(tag_artist=_mp4(tags, "\xa9ART"), tag_title=_mp4(tags, "\xa9nam"),
                   tag_album=_mp4(tags, "\xa9alb"), tag_genre=_mp4(tags, "\xa9gen"),
                   tag_label=_mp4(tags, "----:com.apple.iTunes:LABEL"),
                   tag_bpm=_mp4(tags, "tmpo"), tag_key=_mp4(tags, "----:com.apple.iTunes:initialkey"),
                   tag_comment=_mp4(tags, "\xa9cmt"))
    else:  # Vorbis (FLAC/OGG)
        out.update(tag_artist=_vorbis(tags, "artist"), tag_title=_vorbis(tags, "title"),
                   tag_album=_vorbis(tags, "album"), tag_genre=_vorbis(tags, "genre"),
                   tag_label=_vorbis(tags, "label", "organization", "publisher"),
                   tag_bpm=_vorbis(tags, "bpm"), tag_key=_vorbis(tags, "initialkey", "key"),
                   tag_comment=_vorbis(tags, "comment", "description"))

    out["tag_bpm"] = parse_bpm(out["tag_bpm"])
    mik_cam, energy = parse_mik_comment(out["tag_comment"])
    key_cam = to_camelot(out["tag_key"]) if out["tag_key"] else "?"
    out["tag_camelot"] = key_cam if key_cam != "?" else mik_cam
    out["mik_energy"] = energy
    return out


def read_release_tags(path: Path) -> Dict[str, Optional[str]]:
    """Tags para buscar el tema en Beatport: mezcla (Remixer/TPE4), ISRC, fecha e ID de Spotify
    (TXXX que escribe el descargador de playlists). Nunca lanza."""
    out: Dict[str, Optional[str]] = {"tag_remixer": None, "tag_isrc": None, "tag_date": None,
                                     "tag_spotify_id": None}
    try:
        import mutagen
        audio = mutagen.File(str(path))
    except Exception:  # noqa: BLE001
        return out
    tags = getattr(audio, "tags", None) if audio is not None else None
    if tags is None:
        return out
    cls = type(tags).__name__
    if cls == "ID3" or hasattr(tags, "getall"):
        out.update(tag_remixer=_id3_text(tags, "TPE4"), tag_isrc=_id3_text(tags, "TSRC"),
                   tag_date=_id3_text(tags, "TDRC") or _id3_text(tags, "TDRL") or _id3_text(tags, "TYER"))
        for frame in tags.getall("TXXX"):
            if str(getattr(frame, "desc", "")).lower() == "spotify track id" and frame.text:
                out["tag_spotify_id"] = str(frame.text[0]).strip() or None
    elif cls == "MP4Tags":
        out.update(tag_remixer=_mp4(tags, "----:com.apple.iTunes:REMIXER"),
                   tag_isrc=_mp4(tags, "----:com.apple.iTunes:ISRC"), tag_date=_mp4(tags, "\xa9day"))
    else:  # Vorbis (FLAC/OGG)
        out.update(tag_remixer=_vorbis(tags, "remixer", "mixartist"), tag_isrc=_vorbis(tags, "isrc"),
                   tag_date=_vorbis(tags, "date", "year"))
    return out


_FOLDER_IMAGES = ("cover", "folder", "front", "albumart")


def _image_mime(data: bytes, fallback: str = "image/jpeg") -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    return fallback or "image/jpeg"


def read_cover(path: Path) -> Optional[Tuple[bytes, str]]:
    """Carátula del tema: la imagen incluida en el archivo (ID3 APIC, con preferencia la de portada;
    imágenes de FLAC; covr de MP4) o, si no tiene, cover/folder/front.jpg|png de su carpeta.
    Devuelve (bytes, tipo MIME) o None. Nunca lanza."""
    path = Path(path)
    try:
        import mutagen
        audio = mutagen.File(str(path))
        tags = getattr(audio, "tags", None) if audio is not None else None
        pics = list(getattr(audio, "pictures", None) or [])          # FLAC
        if tags is not None and hasattr(tags, "getall"):              # ID3 (MP3, WAV, AIFF)
            pics += tags.getall("APIC")
        pics.sort(key=lambda p: getattr(p, "type", 0) != 3)           # portada primero
        for p in pics:
            if p.data:
                return bytes(p.data), _image_mime(p.data, getattr(p, "mime", ""))
        covr = tags.get("covr") if tags is not None and type(tags).__name__ == "MP4Tags" else None
        if covr:
            return bytes(covr[0]), _image_mime(bytes(covr[0]))
    except Exception:  # noqa: BLE001
        pass
    for f in sorted(path.parent.iterdir()) if path.parent.is_dir() else []:
        if f.stem.lower() in _FOLDER_IMAGES and f.suffix.lower() in (".jpg", ".jpeg", ".png") and f.is_file():
            data = f.read_bytes()
            return data, _image_mime(data)
    return None


# ---------------------------------------------------------------------------
# Identidad del audio sin tags
# ---------------------------------------------------------------------------

def _syncsafe(b: bytes) -> int:
    return (b[0] << 21) | (b[1] << 14) | (b[2] << 7) | b[3]


def payload_range(path: Path) -> Tuple[int, int]:
    """[inicio, fin) de los bytes de audio, excluyendo tags conocidos. Formatos no soportados: archivo entero."""
    path = Path(path)
    size = path.stat().st_size
    start, end = 0, size
    suffix = path.suffix.lower()
    with open(path, "rb") as f:
        # ID3v2 al principio (MP3, y a veces FLAC); puede haber más de uno encadenado
        while True:
            f.seek(start)
            head = f.read(10)
            if len(head) == 10 and head[:3] == b"ID3" and all(x < 0x80 for x in head[6:10]):
                footer = 10 if head[5] & 0x10 else 0
                start += 10 + _syncsafe(head[6:10]) + footer
            else:
                break
        if suffix == ".flac":
            f.seek(start)
            if f.read(4) == b"fLaC":
                pos = start + 4
                while True:
                    f.seek(pos)
                    hdr = f.read(4)
                    if len(hdr) < 4:
                        break
                    pos += 4 + int.from_bytes(hdr[1:4], "big")
                    if hdr[0] & 0x80:
                        break
                start = pos
        elif suffix == ".mp3":
            # ID3v1 (128 bytes) y APEv2 al final, en cualquier orden habitual
            for _ in range(3):
                if end - 128 >= start:
                    f.seek(end - 128)
                    if f.read(3) == b"TAG":
                        end -= 128
                        continue
                if end - 32 >= start:
                    f.seek(end - 32)
                    foot = f.read(32)
                    if foot[:8] == b"APETAGEX":
                        tag_size = int.from_bytes(foot[12:16], "little")
                        has_header = bool(int.from_bytes(foot[20:24], "little") & 0x80000000)
                        end -= tag_size + (32 if has_header else 0)
                        continue
                break
    return start, max(start, end)


def audio_payload_hash(path: Path) -> str:
    """SHA256 (64 hex) de los bytes de audio sin tags. Estable ante ediciones de tags."""
    start, end = payload_range(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(start)
        remaining = end - start
        while remaining > 0:
            chunk = f.read(min(1 << 20, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Escritura
# ---------------------------------------------------------------------------

def write_comment(path: Path, text: str) -> None:
    """Reemplaza el comentario visible (ID3 COMM sin descripción, o Vorbis COMMENT)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in WRITABLE_SUFFIXES:
        raise ValueError(f"formato no soportado para escribir comentario: {suffix}")
    import mutagen
    if suffix == ".flac":
        from mutagen.flac import FLAC
        audio = FLAC(str(path))
        audio["comment"] = [text]
        audio.save()
        return
    from mutagen.id3 import COMM
    audio = mutagen.File(str(path))
    if audio is None:
        raise ValueError(f"no se pudo abrir {path}")
    if audio.tags is None:
        audio.add_tags()
    tags = audio.tags
    version = tags.version[1] if getattr(tags, "version", None) and tags.version[1] in (3, 4) else 3
    lang = "eng"
    for key in list(tags.keys()):
        if key.startswith("COMM") and getattr(tags[key], "desc", None) == "":
            lang = tags[key].lang or lang
            del tags[key]
    tags.add(COMM(encoding=3, lang=lang, desc="", text=[text]))
    audio.save(v2_version=version)
