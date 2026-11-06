"""
PURPOSE: The recording queue of a download: the tracks that have to be recorded from Spotify (not on
         Soulseek, nothing Soulseek gave passed the verification, or the search failed). The downloader
         appends each one as soon as it knows; the orchestrator records them while the search goes on, and
         the app shows them. One JSON object per line, append only: a reader never meets a half-replaced
         file, and a last line still being written (no newline yet) is skipped until it is complete.

         {"type": "track", "playlist_position", "duration_ms", "duration_seconds", "duration_hms",
          "spotify": {...}, "origin": "missing" | "failed" | "search_error", "reason": str}
         {"type": "end", "status": "finished" | "search_failed", "error": str | None}

         The track lines carry what playlist_loopback_recorder.py needs in its manifest. A track is in the
         queue once (the first line wins); the end line closes the queue only when it is the last line.
CHANGELOG:
  - 2026-11-06: Creación: búsqueda y descarga tema a tema, grabación en cuanto aparece el primer faltante.
                Se lee en bytes: una línea cortada en medio de una letra con tilde no rompe la lectura.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

QUEUE_NAME = "recording_queue.jsonl"
END_FINISHED = "finished"            # el descargador recorrió toda la playlist
END_SEARCH_FAILED = "search_failed"  # la búsqueda en Soulseek dejó de funcionar: se cortó antes de terminar


@dataclass
class QueueState:
    tracks: list[dict[str, Any]] = field(default_factory=list)
    end: dict[str, Any] | None = None

    @property
    def closed(self) -> bool:
        return self.end is not None

    def ids(self) -> list[str]:
        return [track_id(t) for t in self.tracks]


def track_id(entry: dict[str, Any]) -> str:
    return str((entry.get("spotify") or {}).get("spotify_id") or "")


def read_queue(path: Path) -> QueueState:
    """What the queue holds now. A missing file is an empty, open queue."""
    try:
        data = path.read_bytes()
    except OSError:
        return QueueState()
    state = QueueState()
    seen: set[str] = set()
    complete = data.split(b"\n")[:-1]  # lo que sigue al último salto de línea se está escribiendo
    for raw in complete:
        if not raw.strip():
            continue
        try:
            item = json.loads(raw.decode("utf-8"))
        except ValueError:  # también UnicodeDecodeError
            continue
        if not isinstance(item, dict):
            continue
        if item.get("type") == "end":
            state.end = item
            continue
        if item.get("type") != "track" or not isinstance(item.get("spotify"), dict):
            continue
        state.end = None  # algo se agregó después de un cierre: la cola sigue abierta
        tid = track_id(item)
        if not tid or tid in seen:
            continue
        seen.add(tid)
        state.tracks.append(item)
    return state


def _append(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        handle.flush()


def append_track(path: Path, entry: dict[str, Any], *, origin: str, reason: str) -> None:
    _append(path, {"type": "track", **entry, "origin": origin, "reason": reason, "queued_at": time.time()})


def close_queue(path: Path, status: str, error: str | None = None) -> None:
    _append(path, {"type": "end", "status": status, "error": error, "at": time.time()})
