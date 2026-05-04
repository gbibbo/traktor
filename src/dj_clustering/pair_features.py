"""
PURPOSE: Build pairwise features for DJ clustering. Provides deterministic
         enumeration of unordered canonical pairs, vectorized cosine pairwise
         features for every (MERT source, aggregation) combination, raw-BPM
         tempo-equivalence pairwise features, and metadata-similarity features
         over the D1.3-validated sanity fields. All outputs are aligned with a
         caller-provided sorted list of canonical track_ids; tracks lacking a
         feature stay in the universe and produce ``available = False`` for
         that feature only.

CHANGELOG:
  D3.1 - Initial implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PAIR_ID_SEPARATOR = "__"

MERT_SOURCES: Tuple[str, ...] = ("mert_full", "mert_perc", "mert_concat")
MERT_AGGREGATIONS: Tuple[str, ...] = (
    "last_layer_mean",
    "last_4_layers_mean",
    "layer_7_mean",
)


def enumerate_canonical_pairs(
    track_ids: Sequence[str],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Return deterministic upper-triangle pair enumeration.

    track_ids may be unsorted; the function sorts ascending and yields all
    unordered pairs (a, b) with ``a < b``. Returns:
      pair_ids:     list[str], length P = N*(N-1)/2
      idx_a:        np.ndarray[int64], indices into the sorted track list
      idx_b:        np.ndarray[int64], same; idx_a[k] < idx_b[k]
    """
    sorted_ids = sorted(set(track_ids))
    if len(sorted_ids) != len(list(track_ids)):
        raise ValueError("enumerate_canonical_pairs requires unique track_ids")
    n = len(sorted_ids)
    if n < 2:
        return [], np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    idx_a, idx_b = np.triu_indices(n, k=1)
    idx_a = idx_a.astype(np.int64)
    idx_b = idx_b.astype(np.int64)
    pair_ids: List[str] = [
        f"{sorted_ids[a]}{PAIR_ID_SEPARATOR}{sorted_ids[b]}"
        for a, b in zip(idx_a.tolist(), idx_b.tolist())
    ]
    return pair_ids, idx_a, idx_b


def align_embedding_matrix(
    embeddings: np.ndarray,
    src_track_ids: Sequence[str],
    all_track_ids: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Project an (M, D) embedding array onto the (N, D) canonical universe.

    Rows of tracks present in ``src_track_ids`` are copied verbatim. Rows for
    tracks absent from ``src_track_ids`` are set to NaN and ``available=False``.
    The output row order matches ``all_track_ids`` exactly (caller's sort).
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
    if embeddings.shape[0] != len(src_track_ids):
        raise ValueError(
            "embeddings rows do not match src_track_ids length: "
            f"{embeddings.shape[0]} vs {len(src_track_ids)}"
        )
    src_lookup: Dict[str, int] = {tid: i for i, tid in enumerate(src_track_ids)}
    n = len(all_track_ids)
    d = embeddings.shape[1]
    aligned = np.full((n, d), np.nan, dtype=np.float32)
    available = np.zeros(n, dtype=bool)
    for i, tid in enumerate(all_track_ids):
        j = src_lookup.get(tid)
        if j is not None:
            aligned[i] = embeddings[j].astype(np.float32, copy=False)
            available[i] = True
    return aligned, available


def compute_cosine_pair_features(
    aligned_embeddings: np.ndarray,
    available: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Cosine pair features over the canonical pair universe.

    Inputs:
      aligned_embeddings: (N, D) array, NaN rows for missing tracks. Embeddings
        must already be L2-normalized (the D2.2 storage contract).
      available: (N,) bool, marks which canonical tracks have the embedding.
      idx_a, idx_b: pair-level indices from ``enumerate_canonical_pairs``.

    Outputs (each shape (P,)):
      cosine_similarity   : float32, dot product, NaN where unavailable.
      cosine_distance     : float32, 1 - similarity, NaN where unavailable.
      cosine_embedding_01 : float32, (similarity + 1) / 2, in [0, 1] where
                            available; NaN otherwise.
      available           : bool, True iff both endpoints have the embedding.
    """
    n = aligned_embeddings.shape[0]
    if available.shape != (n,):
        raise ValueError("available shape mismatch")

    avail_pair = available[idx_a] & available[idx_b]

    # Vectorized dot products. NaN rows naturally propagate to NaN sims.
    sim = np.einsum("ij,ij->i", aligned_embeddings[idx_a], aligned_embeddings[idx_b])
    sim = sim.astype(np.float32, copy=False)
    # Force NaN where unavailable (defensive; should already be NaN).
    sim = np.where(avail_pair, sim, np.float32(np.nan))

    # Numerical clamp to handle minor float drift slightly outside [-1, 1]
    # for L2-normalized vectors.
    valid_mask = avail_pair
    if valid_mask.any():
        clamped = np.clip(sim[valid_mask], -1.0, 1.0)
        sim_out = sim.copy()
        sim_out[valid_mask] = clamped
        sim = sim_out

    distance = np.where(avail_pair, np.float32(1.0) - sim, np.float32(np.nan)).astype(
        np.float32, copy=False
    )
    cos01 = np.where(
        avail_pair, (sim + np.float32(1.0)) * np.float32(0.5), np.float32(np.nan)
    ).astype(np.float32, copy=False)

    return {
        "cosine_similarity": sim,
        "cosine_distance": distance,
        "cosine_embedding_01": cos01,
        "available": avail_pair,
    }


def _bpm_array_for_canonical(
    bpm_lookup: Mapping[str, float],
    all_track_ids: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (bpm_array shape (N,), available bool (N,)). NaN where missing."""
    n = len(all_track_ids)
    bpm = np.full(n, np.nan, dtype=np.float64)
    avail = np.zeros(n, dtype=bool)
    for i, tid in enumerate(all_track_ids):
        v = bpm_lookup.get(tid)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            bpm[i] = float(v)
            avail[i] = True
    return bpm, avail


def compute_bpm_pair_features(
    bpm_lookup: Mapping[str, float],
    all_track_ids: Sequence[str],
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    tolerance_bpm: float,
) -> Dict[str, np.ndarray]:
    """Pairwise BPM features under 0.5x / 1x / 2x tempo equivalence.

    All operations use raw BPM (real BPM units) supplied via ``bpm_lookup``.
    The D2.2 normalized BPM artifact must NOT be passed here.

    Outputs (each shape (P,)):
      bpm_raw_a, bpm_raw_b : float32, NaN where unavailable.
      bpm_abs_diff         : float32, |a - b|, NaN where unavailable.
      bpm_eff_abs_diff     : float32, smallest |a' - b'| under tempo equivalence.
      bpm_similarity       : float32, 1 - clip(eff/tol, 0, 1), in [0, 1] where
                             available; NaN otherwise.
      bpm_available        : bool, True iff both endpoints have raw BPM.
    """
    if tolerance_bpm <= 0:
        raise ValueError("tolerance_bpm must be positive")

    bpm_arr, avail = _bpm_array_for_canonical(bpm_lookup, all_track_ids)
    a = bpm_arr[idx_a]
    b = bpm_arr[idx_b]
    avail_pair = avail[idx_a] & avail[idx_b]

    abs_diff_1 = np.abs(a - b)
    abs_diff_2 = np.abs(a - 2.0 * b)
    abs_diff_3 = np.abs(a - 0.5 * b)
    abs_diff_4 = np.abs(2.0 * a - b)
    abs_diff_5 = np.abs(0.5 * a - b)

    eff = np.minimum.reduce(
        [abs_diff_1, abs_diff_2, abs_diff_3, abs_diff_4, abs_diff_5]
    )

    sim = 1.0 - np.clip(eff / float(tolerance_bpm), 0.0, 1.0)

    nan = np.float32(np.nan)
    out = {
        "bpm_raw_a": np.where(avail_pair, a, np.nan).astype(np.float32, copy=False),
        "bpm_raw_b": np.where(avail_pair, b, np.nan).astype(np.float32, copy=False),
        "bpm_abs_diff": np.where(avail_pair, abs_diff_1, np.nan).astype(
            np.float32, copy=False
        ),
        "bpm_eff_abs_diff": np.where(avail_pair, eff, np.nan).astype(
            np.float32, copy=False
        ),
        "bpm_similarity": np.where(avail_pair, sim, np.nan).astype(
            np.float32, copy=False
        ),
        "bpm_available": avail_pair,
    }
    # Defensive: ensure missing flagged values are NaN, not 0.
    for key in ("bpm_raw_a", "bpm_raw_b", "bpm_abs_diff", "bpm_eff_abs_diff", "bpm_similarity"):
        out[key] = np.where(avail_pair, out[key], nan).astype(np.float32, copy=False)
    return out


def _normalize_metadata_value(value: object) -> Optional[str]:
    """Lower-case, strip; treat empty/NaN as missing."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    s = str(value).strip().lower()
    if not s or s in {"nan", "none", "null"}:
        return None
    return s


def compute_metadata_pair_features(
    inventory: pd.DataFrame,
    fields: Sequence[str],
    all_track_ids: Sequence[str],
    idx_a: np.ndarray,
    idx_b: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Pairwise metadata similarity over D1.3-validated sanity fields.

    For each field f in ``fields`` we compare normalized values; per pair the
    similarity is the mean of binary matches over fields available on BOTH
    sides. If no fields are available on both sides, similarity = 0 and
    ``metadata_similarity_available = False``.

    Inputs:
      inventory: pandas DataFrame indexed by track_id (or with a track_id col).
      fields: ordered list of field names to use (e.g. ['genre','artist','label']).
      all_track_ids: canonical track id order.
      idx_a, idx_b: pair indices.

    Outputs:
      metadata_similarity              : float32 in [0, 1].
      metadata_similarity_available    : bool.
      metadata_<field>_available       : bool per field.
    """
    if "track_id" in inventory.columns:
        inv = inventory.set_index("track_id", drop=False)
    else:
        inv = inventory

    n = len(all_track_ids)
    field_present: Dict[str, np.ndarray] = {}
    field_value: Dict[str, List[Optional[str]]] = {}
    for f in fields:
        if f not in inv.columns:
            present = np.zeros(n, dtype=bool)
            values: List[Optional[str]] = [None] * n
        else:
            values = []
            present_arr = np.zeros(n, dtype=bool)
            for i, tid in enumerate(all_track_ids):
                if tid in inv.index:
                    raw = inv.at[tid, f] if f in inv.columns else None
                    norm = _normalize_metadata_value(raw)
                else:
                    norm = None
                values.append(norm)
                if norm is not None:
                    present_arr[i] = True
            present = present_arr
        field_present[f] = present
        field_value[f] = values

    p = idx_a.shape[0]
    match_sum = np.zeros(p, dtype=np.int32)
    avail_count = np.zeros(p, dtype=np.int32)

    out: Dict[str, np.ndarray] = {}
    for f in fields:
        present = field_present[f]
        avail_pair = present[idx_a] & present[idx_b]
        out[f"metadata_{f}_available"] = avail_pair
        # Build a match vector only where both sides available.
        if avail_pair.any():
            values = field_value[f]
            a_vals = [values[i] for i in idx_a[avail_pair].tolist()]
            b_vals = [values[i] for i in idx_b[avail_pair].tolist()]
            matches = np.array(
                [1 if a == b else 0 for a, b in zip(a_vals, b_vals)], dtype=np.int32
            )
            match_local = np.zeros(p, dtype=np.int32)
            match_local[avail_pair] = matches
            match_sum += match_local
        avail_count += avail_pair.astype(np.int32)

    has_any = avail_count > 0
    sim = np.zeros(p, dtype=np.float32)
    np.divide(
        match_sum,
        np.where(has_any, avail_count, 1),
        out=sim,
        where=has_any,
        casting="unsafe",
    )
    sim = np.where(has_any, sim, np.float32(0.0)).astype(np.float32, copy=False)

    out["metadata_similarity"] = sim
    out["metadata_similarity_available"] = has_any
    return out


def assemble_pair_dataframe(
    pair_ids: Sequence[str],
    track_ids_a: Sequence[str],
    track_ids_b: Sequence[str],
    cosine_blocks: Mapping[Tuple[str, str], Mapping[str, np.ndarray]],
    bpm_block: Mapping[str, np.ndarray],
    metadata_block: Mapping[str, np.ndarray],
    bpm_norm_abs_diff: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Build the final pair_features DataFrame in deterministic column order."""
    cols: Dict[str, np.ndarray] = {}
    cols["pair_id"] = np.asarray(pair_ids, dtype=object)
    cols["pair_index"] = np.arange(len(pair_ids), dtype=np.int64)
    cols["track_id_a"] = np.asarray(track_ids_a, dtype=object)
    cols["track_id_b"] = np.asarray(track_ids_b, dtype=object)

    for source in MERT_SOURCES:
        for agg in MERT_AGGREGATIONS:
            block = cosine_blocks.get((source, agg))
            if block is None:
                continue
            prefix = f"{source}_{agg}__"
            cols[prefix + "cosine_similarity"] = block["cosine_similarity"]
            cols[prefix + "cosine_distance"] = block["cosine_distance"]
            cols[prefix + "cosine_embedding_01"] = block["cosine_embedding_01"]
            cols[prefix + "available"] = block["available"]

    for key in (
        "bpm_raw_a",
        "bpm_raw_b",
        "bpm_abs_diff",
        "bpm_eff_abs_diff",
        "bpm_similarity",
        "bpm_available",
    ):
        cols[key] = bpm_block[key]
    if bpm_norm_abs_diff is not None:
        cols["bpm_norm_abs_diff"] = bpm_norm_abs_diff.astype(np.float32, copy=False)

    for key in metadata_block.keys():
        cols[key] = metadata_block[key]

    return pd.DataFrame(cols)


def cosine_default_column(
    default_source: str, default_aggregation: str
) -> str:
    """Return the cosine_embedding_01 column name for the default embedding."""
    return f"{default_source}_{default_aggregation}__cosine_embedding_01"


def cosine_default_available_column(
    default_source: str, default_aggregation: str
) -> str:
    return f"{default_source}_{default_aggregation}__available"


def load_track_ids(path: Path) -> List[str]:
    """Read a track_ids.txt artifact file; one ID per line."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]
