"""
PURPOSE: Triplet question generation for DJ clustering manual comparison queue.
         Provides KNN index construction from pair-feature cosine similarities,
         V4 playlist cluster-map parsing (with Windows-path basename extraction
         and EXTINF metadata fallback), genre-based fallback cluster mapping,
         nearest-neighbor and cluster-boundary triplet sampling (deterministic
         under a caller-supplied seed), de-duplication, and final question
         DataFrame assembly with the 14 plan-required columns.

CHANGELOG:
  D3.2 - Initial implementation.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SIM_COL = "mert_full_last_layer_mean__cosine_similarity"
DEFAULT_AVAIL_COL = "mert_full_last_layer_mean__available"
DEFAULT_KNN_K = 50
DEFAULT_SEED = 42
V4_COVERAGE_THRESHOLD = 0.70

# Required columns for the output question DataFrame.
QUESTION_COLUMNS: Tuple[str, ...] = (
    "question_id",
    "anchor_track_id",
    "candidate_b_track_id",
    "candidate_c_track_id",
    "anchor_artist",
    "anchor_title",
    "candidate_b_artist",
    "candidate_b_title",
    "candidate_c_artist",
    "candidate_c_title",
    "anchor_file_path",
    "candidate_b_file_path",
    "candidate_c_file_path",
    "selection_source",
)

# ---------------------------------------------------------------------------
# KNN index
# ---------------------------------------------------------------------------


def build_knn_index(
    pair_df: pd.DataFrame,
    track_ids: List[str],
    sim_col: str = DEFAULT_SIM_COL,
    avail_col: str = DEFAULT_AVAIL_COL,
    k: int = DEFAULT_KNN_K,
) -> Dict[str, List[Tuple[float, str]]]:
    """Build per-track KNN lookup from pairwise cosine similarities.

    Only pairs where ``avail_col`` is True are included. Each track's entry is
    sorted by similarity descending and capped at ``k`` neighbors.

    Returns {track_id: [(sim, neighbor_track_id), ...]} for every track_id in
    ``track_ids``. Tracks with no available pairs have an empty list.
    """
    avail = pair_df[pair_df[avail_col].astype(bool)].copy()

    df_a = avail[["track_id_a", "track_id_b", sim_col]].rename(
        columns={"track_id_a": "query", "track_id_b": "neighbor"}
    )
    df_b = avail[["track_id_b", "track_id_a", sim_col]].rename(
        columns={"track_id_b": "query", "track_id_a": "neighbor"}
    )
    all_edges = pd.concat([df_a, df_b], ignore_index=True)

    tid_set = set(track_ids)
    all_edges = all_edges[all_edges["query"].isin(tid_set)]
    all_edges = all_edges.sort_values(["query", sim_col], ascending=[True, False])

    result: Dict[str, List[Tuple[float, str]]] = {tid: [] for tid in track_ids}
    for query, group in all_edges.groupby("query", sort=True):
        if query in result:
            top = group.head(k)
            result[query] = list(
                zip(top[sim_col].tolist(), top["neighbor"].tolist())
            )

    return result


# ---------------------------------------------------------------------------
# V4 playlist cluster map
# ---------------------------------------------------------------------------


def _normalize_str(s: str) -> str:
    """Case-fold and strip accents for fuzzy string matching."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _windows_basename(path_line: str) -> str:
    """Extract the basename from a Windows or POSIX absolute path string."""
    normalized = path_line.replace("\\", "/")
    return normalized.split("/")[-1]


def parse_v4_cluster_map(
    v4_dir: Path,
    inventory_df: pd.DataFrame,
    coverage_threshold: float = V4_COVERAGE_THRESHOLD,
) -> Tuple[Dict[str, str], float, str]:
    """Parse V4 L1 cluster groups from m3u playlist files.

    Searches ``v4_dir`` recursively for ``*.m3u`` files. Each file's L1 group
    label is derived from its immediate parent directory name (the component up
    to and including the second underscore, e.g. ``L1_A``).

    Mapping strategy (tried in order):
      1. Normalized filename (basename of Windows or POSIX path).
      2. EXTINF display-name matched against normalized ``file_name`` in
         inventory (covers cases where only the display string is reliable).

    Returns:
      cluster_map:  {track_id -> L1_group_label}
      coverage:     fraction of playlist entries mapped to a canonical track_id
      method:       'normalized_filename' or 'normalized_filename+extinf_fallback'

    Does NOT raise if coverage is below ``coverage_threshold``; callers decide
    the fallback policy.  Playlist entry paths and filenames are never logged
    or written to committed outputs.
    """
    v4_dir = Path(v4_dir)
    m3u_files = sorted(v4_dir.rglob("*.m3u"))

    if not m3u_files:
        return {}, 0.0, "no_m3u_files"

    # Build fast lookups from inventory (canonical tracks only).
    fname_lower_to_tid: Dict[str, str] = {}
    display_name_to_tid: Dict[str, str] = {}
    for _, row in inventory_df.iterrows():
        fn = str(row.get("file_name", "") or "")
        if fn:
            fname_lower_to_tid[fn.lower()] = str(row["track_id"])
        # EXTINF display-name fallback: normalized file_name without extension.
        stem = Path(fn).stem if fn else ""
        if stem:
            display_name_to_tid[_normalize_str(stem)] = str(row["track_id"])

    cluster_map: Dict[str, str] = {}
    total_entries = 0
    mapped_via_filename = 0
    mapped_via_extinf = 0
    extinf_used = False

    for m3u_path in m3u_files:
        parent_dir = m3u_path.parent.name
        parts = parent_dir.split("_")
        l1_group = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else parent_dir

        current_extinf_display: Optional[str] = None
        try:
            content = m3u_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#EXTM3U"):
                continue
            if line.startswith("#EXTINF"):
                comma_idx = line.find(",")
                if comma_idx >= 0:
                    current_extinf_display = line[comma_idx + 1 :].strip()
                continue
            if line.startswith("#"):
                continue

            total_entries += 1
            basename = _windows_basename(line)
            basename_lower = basename.lower()
            extinf_text = current_extinf_display
            current_extinf_display = None

            if basename_lower in fname_lower_to_tid:
                tid = fname_lower_to_tid[basename_lower]
                cluster_map[tid] = l1_group
                mapped_via_filename += 1
                continue

            # EXTINF fallback: match normalized display text against normalized
            # file stem from inventory (avoids using private metadata columns
            # as matching keys while still recovering entries whose filenames
            # differ from the Windows path basename).
            if extinf_text:
                extinf_norm = _normalize_str(extinf_text)
                if extinf_norm in display_name_to_tid:
                    tid = display_name_to_tid[extinf_norm]
                    cluster_map[tid] = l1_group
                    mapped_via_extinf += 1
                    extinf_used = True
                    continue

    coverage = (mapped_via_filename + mapped_via_extinf) / total_entries if total_entries > 0 else 0.0
    method = "normalized_filename+extinf_fallback" if extinf_used else "normalized_filename"
    return cluster_map, coverage, method


# ---------------------------------------------------------------------------
# Genre-based fallback cluster map
# ---------------------------------------------------------------------------


def genre_cluster_map(inventory_df: pd.DataFrame) -> Dict[str, str]:
    """Fallback cluster assignment using the genre metadata field.

    Tracks with missing or empty genre are assigned to the group 'unknown'.
    """
    result: Dict[str, str] = {}
    for _, row in inventory_df.iterrows():
        genre = str(row.get("genre", "") or "").strip() or "unknown"
        result[str(row["track_id"])] = genre
    return result


# ---------------------------------------------------------------------------
# Triplet sampling helpers
# ---------------------------------------------------------------------------


def _hash_lookup(inventory_df: pd.DataFrame) -> Dict[str, str]:
    return dict(zip(inventory_df["track_id"].astype(str), inventory_df["audio_content_hash"].astype(str)))


def _has_hash_collision(anchor: str, b: str, c: str, hashes: Dict[str, str]) -> bool:
    ha, hb, hc = hashes.get(anchor), hashes.get(b), hashes.get(c)
    if ha and hb and hc:
        return ha == hb or ha == hc or hb == hc
    return False


# ---------------------------------------------------------------------------
# Nearest-neighbor triplet sampling
# ---------------------------------------------------------------------------


def sample_nn_triplets(
    knn_index: Dict[str, List[Tuple[float, str]]],
    inventory_df: pd.DataFrame,
    n: int = 20,
    seed: int = DEFAULT_SEED,
) -> List[Dict]:
    """Sample ``n`` nearest-neighbor triplets deterministically.

    Anchor is drawn uniformly at random from canonical tracks that have ≥2 KNN
    neighbors. Candidates B and C are sampled without replacement from the
    anchor's top-50 KNN list. ``selection_source`` is ``'mert_full_knn'``.
    """
    rng = np.random.default_rng(seed)
    hashes = _hash_lookup(inventory_df)

    canonical_ids = list(inventory_df["track_id"].astype(str))
    eligible = [tid for tid in canonical_ids if len(knn_index.get(tid, [])) >= 2]
    if not eligible:
        return []

    triplets: List[Dict] = []
    seen: set = set()
    max_attempts = max(n * 100, 2000)

    for _ in range(max_attempts):
        if len(triplets) >= n:
            break
        anchor = eligible[int(rng.integers(0, len(eligible)))]
        neighbors = knn_index[anchor]
        if len(neighbors) < 2:
            continue
        idxs = rng.choice(len(neighbors), size=2, replace=False).tolist()
        b = neighbors[idxs[0]][1]
        c = neighbors[idxs[1]][1]
        if b == c or b == anchor or c == anchor:
            continue
        key = (anchor, frozenset([b, c]))
        if key in seen:
            continue
        if _has_hash_collision(anchor, b, c, hashes):
            continue
        seen.add(key)
        triplets.append(
            {
                "anchor_track_id": anchor,
                "candidate_b_track_id": b,
                "candidate_c_track_id": c,
                "selection_source": "mert_full_knn",
            }
        )

    return triplets


# ---------------------------------------------------------------------------
# Cluster-boundary triplet sampling
# ---------------------------------------------------------------------------


def sample_boundary_triplets(
    knn_index: Dict[str, List[Tuple[float, str]]],
    cluster_map: Dict[str, str],
    inventory_df: pd.DataFrame,
    n: int = 20,
    seed: int = DEFAULT_SEED,
    selection_source: str = "v4_boundary",
) -> List[Dict]:
    """Sample ``n`` cluster-boundary triplets deterministically.

    For each eligible anchor (a track in ``cluster_map`` with ≥1 same-cluster
    AND ≥1 cross-cluster neighbor in ``knn_index``):
      - candidate B = closest same-cluster KNN neighbor
      - candidate C = closest cross-cluster KNN neighbor

    Eligible anchors are shuffled under ``seed`` before iteration to ensure
    diversity. ``selection_source`` defaults to ``'v4_boundary'`` but can be
    set to ``'genre_boundary'`` by the caller when using the genre fallback.
    """
    rng = np.random.default_rng(seed)
    hashes = _hash_lookup(inventory_df)

    eligible: List[Dict] = []
    for tid, cluster in cluster_map.items():
        neighbors = knn_index.get(tid, [])
        same = [(s, ntid) for s, ntid in neighbors if cluster_map.get(ntid) == cluster]
        cross = [(s, ntid) for s, ntid in neighbors if cluster_map.get(ntid) not in (None, cluster)]
        if same and cross:
            eligible.append({"anchor": tid, "best_same": same[0][1], "best_cross": cross[0][1]})

    if not eligible:
        return []

    shuffled_indices = rng.permutation(len(eligible)).tolist()
    eligible_shuffled = [eligible[i] for i in shuffled_indices]

    triplets: List[Dict] = []
    seen: set = set()

    for item in eligible_shuffled:
        if len(triplets) >= n:
            break
        anchor = item["anchor"]
        b = item["best_same"]
        c = item["best_cross"]
        if b == c or b == anchor or c == anchor:
            continue
        key = (anchor, frozenset([b, c]))
        if key in seen:
            continue
        if _has_hash_collision(anchor, b, c, hashes):
            continue
        seen.add(key)
        triplets.append(
            {
                "anchor_track_id": anchor,
                "candidate_b_track_id": b,
                "candidate_c_track_id": c,
                "selection_source": selection_source,
            }
        )

    return triplets


# ---------------------------------------------------------------------------
# De-duplication
# ---------------------------------------------------------------------------


def deduplicate_triplets(
    triplets: List[Dict],
    inventory_df: pd.DataFrame,
) -> List[Dict]:
    """Remove duplicate (anchor, {B, C}) tuples and audio_content_hash collisions.

    Preserves the first occurrence of each unique (anchor, frozenset({B, C}))
    key. Removes any triplet whose three tracks share an audio_content_hash.
    """
    hashes = _hash_lookup(inventory_df)
    seen: set = set()
    result: List[Dict] = []
    for t in triplets:
        a = t["anchor_track_id"]
        b = t["candidate_b_track_id"]
        c = t["candidate_c_track_id"]
        key = (a, frozenset([b, c]))
        if key in seen:
            continue
        if _has_hash_collision(a, b, c, hashes):
            continue
        seen.add(key)
        result.append(t)
    return result


# ---------------------------------------------------------------------------
# Question DataFrame assembly
# ---------------------------------------------------------------------------


def assemble_question_df(
    triplets: List[Dict],
    inventory_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the final question DataFrame with all 14 plan-required columns.

    Assigns ``question_id`` values Q001, Q002, … in order. Metadata columns
    (artist, title, file_path) are populated from ``inventory_df``.
    """
    meta = (
        inventory_df.set_index("track_id")[["artist", "title", "file_path"]]
        .to_dict("index")
    )

    rows = []
    for i, t in enumerate(triplets, start=1):
        anchor = t["anchor_track_id"]
        b = t["candidate_b_track_id"]
        c = t["candidate_c_track_id"]
        m_a = meta.get(anchor, {})
        m_b = meta.get(b, {})
        m_c = meta.get(c, {})
        rows.append(
            {
                "question_id": f"Q{i:03d}",
                "anchor_track_id": anchor,
                "candidate_b_track_id": b,
                "candidate_c_track_id": c,
                "anchor_artist": m_a.get("artist", ""),
                "anchor_title": m_a.get("title", ""),
                "candidate_b_artist": m_b.get("artist", ""),
                "candidate_b_title": m_b.get("title", ""),
                "candidate_c_artist": m_c.get("artist", ""),
                "candidate_c_title": m_c.get("title", ""),
                "anchor_file_path": m_a.get("file_path", ""),
                "candidate_b_file_path": m_b.get("file_path", ""),
                "candidate_c_file_path": m_c.get("file_path", ""),
                "selection_source": t["selection_source"],
            }
        )

    if not rows:
        return pd.DataFrame(columns=list(QUESTION_COLUMNS))
    return pd.DataFrame(rows, columns=list(QUESTION_COLUMNS))
