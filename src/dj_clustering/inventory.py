"""
PURPOSE: Library scanning, audio probing, metadata extraction, and row assembly
         for the DJ clustering inventory pipeline. All functions degrade gracefully
         on missing backends or missing metadata; they never raise on per-file errors.

CHANGELOG:
  D1.2 - Initial implementation.
"""

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Runtime backend capability flags (set once at import time)
# ---------------------------------------------------------------------------
_SOUNDFILE = importlib.util.find_spec("soundfile") is not None
_MUTAGEN = importlib.util.find_spec("mutagen") is not None
_TORCHAUDIO = importlib.util.find_spec("torchaudio") is not None

ALLOWED_EXTENSIONS: frozenset = frozenset(
    {".wav", ".flac", ".mp3", ".m4a", ".aiff", ".aif", ".ogg"}
)


# ---------------------------------------------------------------------------
# File scanning
# ---------------------------------------------------------------------------

def scan_audio_files(
    library_root: Path,
    recursive: bool = True,
    ignore_hidden: bool = True,
) -> List[Path]:
    """
    Return a sorted list of audio files that pass the extension whitelist.
    Non-audio files (e.g. PNG, JPG) are silently excluded.
    """
    pattern = "**/*" if recursive else "*"
    results = []
    for path in library_root.glob(pattern):
        if not path.is_file():
            continue
        if ignore_hidden:
            try:
                rel = path.relative_to(library_root)
            except ValueError:
                rel = path
            if any(part.startswith(".") for part in rel.parts):
                continue
        if path.suffix.lower() in ALLOWED_EXTENSIONS:
            results.append(path)
    return sorted(results)


def count_all_files(library_root: Path, recursive: bool = True) -> int:
    """Count all files (any type) in the library root."""
    pattern = "**/*" if recursive else "*"
    return sum(1 for p in library_root.glob(pattern) if p.is_file())


# ---------------------------------------------------------------------------
# Decode / duration probing
# ---------------------------------------------------------------------------

def try_decode(filepath: Path) -> Tuple[str, Optional[float], Optional[str]]:
    """
    Try available backends to determine audio readability and duration.

    Returns (decode_status, duration_seconds, sanitized_decode_error).

    decode_status values:
      "ok"            — soundfile succeeded
      "ok_probe_only" — soundfile failed, ffprobe succeeded
      "ok_torchaudio" — both above failed, torchaudio succeeded
      "failed"        — all backends failed

    decode_error is sanitized: no private file paths included.
    """
    last_err: Optional[str] = None

    # --- Backend 1: soundfile ---
    if _SOUNDFILE:
        try:
            import soundfile as sf  # type: ignore
            info = sf.info(str(filepath))
            return "ok", float(info.duration), None
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {str(exc)[:80]}"

    # --- Backend 2: ffprobe ---
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin:
        try:
            proc = subprocess.run(
                [ffprobe_bin, "-v", "quiet", "-print_format", "json",
                 "-show_format", str(filepath)],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode == 0:
                data = json.loads(proc.stdout)
                dur = float(data.get("format", {}).get("duration", 0) or 0)
                if dur > 0:
                    return "ok_probe_only", dur, None
        except Exception:
            pass

    # --- Backend 3: torchaudio ---
    if _TORCHAUDIO:
        try:
            import torchaudio  # type: ignore
            info = torchaudio.info(str(filepath))
            dur = info.num_frames / info.sample_rate
            return "ok_torchaudio", float(dur), None
        except Exception:
            pass

    return "failed", None, last_err


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def _parse_filename_tags(stem: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract (artist, title) from 'Artist - Title' filename pattern."""
    if " - " in stem:
        parts = stem.split(" - ", 1)
        return parts[0].strip() or None, parts[1].strip() or None
    return None, None


def _quality_flag(fields: Dict[str, Any]) -> str:
    """Compute metadata_quality_flags from the populated tag fields."""
    core = [fields.get("artist"), fields.get("title"),
            fields.get("album"), fields.get("genre")]
    filled = sum(bool(v) for v in core)
    if filled == 4:
        return "full_tags"
    if filled > 0:
        return "partial_tags"
    return "no_tags"


def _fill_easyid3(filepath: Path, fields: Dict[str, Any]) -> bool:
    """Try EasyID3 for MP3/WAV/AIFF. Returns True if tags were loaded."""
    try:
        from mutagen.easyid3 import EasyID3  # type: ignore
        easy = EasyID3(str(filepath))

        def _first(key: str) -> Optional[str]:
            vals = easy.get(key)
            return str(vals[0]) if vals else None

        fields["artist"] = _first("artist")
        fields["title"] = _first("title")
        fields["album"] = _first("album")
        fields["genre"] = _first("genre")
        fields["year"] = _first("date")
        fields["key"] = _first("initialkey")
        bpm_str = _first("bpm")
        if bpm_str:
            try:
                fields["bpm"] = float(bpm_str)
            except (ValueError, TypeError):
                pass
        return True
    except Exception:
        return False


def _fill_label_id3(filepath: Path, fields: Dict[str, Any]) -> None:
    """Try to extract TPUB (label/publisher) via full ID3 object."""
    try:
        from mutagen.id3 import ID3  # type: ignore
        full = ID3(str(filepath))
        tpub = full.get("TPUB")
        if tpub:
            text = tpub.text[0] if hasattr(tpub, "text") and tpub.text else str(tpub)
            fields["label"] = str(text) if text else None
    except Exception:
        pass


def _fill_generic_mutagen(filepath: Path, fields: Dict[str, Any]) -> bool:
    """Try mutagen.File for non-ID3 formats (FLAC, OGG). Returns True if tags loaded."""
    try:
        from mutagen import File as MutFile  # type: ignore
        mf = MutFile(str(filepath))
        if mf is None or mf.tags is None:
            return False

        def _vorbis(key: str) -> Optional[str]:
            for k in (key, key.upper()):
                v = mf.tags.get(k)
                if v:
                    return str(v[0])
            return None

        fields["artist"] = _vorbis("artist")
        fields["title"] = _vorbis("title")
        fields["album"] = _vorbis("album")
        fields["genre"] = _vorbis("genre")
        fields["label"] = _vorbis("publisher") or _vorbis("label")
        fields["year"] = _vorbis("date") or _vorbis("year")
        fields["key"] = _vorbis("key") or _vorbis("initialkey")
        bpm_str = _vorbis("bpm")
        if bpm_str:
            try:
                fields["bpm"] = float(bpm_str)
            except (ValueError, TypeError):
                pass
        return True
    except Exception:
        return False


def extract_metadata(filepath: Path) -> Dict[str, Any]:
    """
    Extract tag metadata for one audio file. Degrades gracefully.

    Returns a dict with: artist, title, album, genre, label, year, bpm, key,
    and metadata_quality_flags.
    """
    fields: Dict[str, Any] = {
        "artist": None, "title": None, "album": None, "genre": None,
        "label": None, "year": None, "bpm": None, "key": None,
    }

    if _MUTAGEN:
        ext = filepath.suffix.lower()
        loaded = False

        if ext in (".mp3", ".wav", ".aiff", ".aif", ".m4a"):
            loaded = _fill_easyid3(filepath, fields)
            if loaded:
                _fill_label_id3(filepath, fields)
        elif ext in (".flac", ".ogg"):
            loaded = _fill_generic_mutagen(filepath, fields)

        if loaded:
            fields["metadata_quality_flags"] = _quality_flag(fields)
        else:
            # mutagen present but couldn't read tags — treat as no_tags
            fields["metadata_quality_flags"] = "no_tags"
        return fields

    # mutagen unavailable — fall back to filename parsing
    artist, title = _parse_filename_tags(filepath.stem)
    if artist is not None:
        fields["artist"] = artist
        fields["title"] = title
        fields["metadata_quality_flags"] = "filename_parsed"
    else:
        fields["metadata_quality_flags"] = "no_tag_library"
    return fields
