"""
PURPOSE: Stable audio identity — SHA256 content hashing, deterministic track_id
         generation, exact duplicate detection, and canonical track selection.
         All identity logic is pure-function; no I/O beyond the hash read.

CHANGELOG:
  D1.2 - Initial implementation.
"""

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

_CHUNK = 65536


def compute_audio_content_hash(filepath: Path) -> Optional[str]:
    """Stream SHA256 of raw file bytes. Returns None only on I/O error."""
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as fh:
            while chunk := fh.read(_CHUNK):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def select_canonical(group: List[Path]) -> Path:
    """Return the lexicographically-first path in a duplicate group."""
    return min(group)


def assign_track_ids(files: List[Path]) -> Dict[Path, dict]:
    """
    Compute all identity fields for a list of audio file paths.

    Canonical representative per hash group = lexicographically-first path.
    Canonical track_id = sha256hex[:16]   (no suffix)
    Duplicates        = sha256hex[:16]_d01, _d02, ... (1-indexed, sorted order)

    Returns a mapping path → dict with keys:
      audio_content_hash, track_id, canonical_track_id,
      is_canonical, is_exact_duplicate, duplicate_of_track_id
    """
    # Step 1: compute hashes
    hashes: Dict[Path, Optional[str]] = {f: compute_audio_content_hash(f) for f in files}

    # Step 2: group by hash (None-hash files each get their own singleton group)
    hash_groups: Dict[str, List[Path]] = defaultdict(list)
    nohash: List[Path] = []

    for path, h in hashes.items():
        if h is None:
            nohash.append(path)
        else:
            hash_groups[h].append(path)

    # Step 3: sort each group for determinism
    for h in hash_groups:
        hash_groups[h].sort()
    nohash.sort()

    result: Dict[Path, dict] = {}

    # Step 4: assign track_ids for hashed files
    for h, group in hash_groups.items():
        base = h[:16]
        canonical_tid = base  # no suffix for canonical

        for i, path in enumerate(group):
            is_canonical = i == 0
            is_dup = i > 0

            if is_canonical:
                tid = canonical_tid
                dup_of = None
            else:
                tid = f"{base}_d{i:02d}"
                dup_of = canonical_tid

            result[path] = {
                "audio_content_hash": h,
                "track_id": tid,
                "canonical_track_id": canonical_tid,
                "is_canonical": is_canonical,
                "is_exact_duplicate": is_dup,
                "duplicate_of_track_id": dup_of,
            }

    # Step 5: handle unreadable files (no hash) — each is its own singleton
    for i, path in enumerate(nohash):
        tid = f"nohash_{i:04d}"
        result[path] = {
            "audio_content_hash": None,
            "track_id": tid,
            "canonical_track_id": tid,
            "is_canonical": True,
            "is_exact_duplicate": False,
            "duplicate_of_track_id": None,
        }

    return result
