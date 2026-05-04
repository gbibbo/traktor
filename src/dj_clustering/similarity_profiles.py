"""
PURPOSE: Committed similarity profile registry for the DJ clustering Phase 3
         pair feature table. Each profile combines normalized component
         features (default cosine_embedding_01, bpm_similarity, key_compat,
         metadata_similarity) into a [0, 1] similarity and a [0, 1] distance.
         Profiles whose required components are unavailable are recorded with
         an explicit ``omission_reason`` and emit no columns.

CHANGELOG:
  D3.1 - Initial implementation. Committed profiles: audio_only,
         audio_plus_metadata_light. Profiles requiring key_compat are omitted
         because the dataset has 0% key coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ProfileSpec:
    """Static specification of a committed similarity profile.

    weights: ordered mapping from component name -> weight; weights must sum
        to 1.0 across the components actually present.
    required_components: components whose availability is mandatory; if any
        is unavailable the profile is omitted.
    """

    name: str
    weights: Dict[str, float]
    required_components: Tuple[str, ...]


# All four plan profiles. Components reference symbolic names; the executor
# maps them to concrete columns.
ALL_PROFILES: Dict[str, ProfileSpec] = {
    "audio_only": ProfileSpec(
        name="audio_only",
        weights={"cosine_embedding_01": 1.00},
        required_components=("cosine_embedding_01",),
    ),
    "audio_plus_bpm_key_light": ProfileSpec(
        name="audio_plus_bpm_key_light",
        weights={
            "cosine_embedding_01": 0.90,
            "bpm_similarity": 0.05,
            "key_compat": 0.05,
        },
        required_components=("cosine_embedding_01", "bpm_similarity", "key_compat"),
    ),
    "audio_plus_metadata_light": ProfileSpec(
        name="audio_plus_metadata_light",
        weights={
            "cosine_embedding_01": 0.85,
            "metadata_similarity": 0.15,
        },
        required_components=("cosine_embedding_01", "metadata_similarity"),
    ),
    "audio_plus_bpm_key_metadata_light": ProfileSpec(
        name="audio_plus_bpm_key_metadata_light",
        weights={
            "cosine_embedding_01": 0.80,
            "bpm_similarity": 0.05,
            "key_compat": 0.05,
            "metadata_similarity": 0.10,
        },
        required_components=(
            "cosine_embedding_01",
            "bpm_similarity",
            "key_compat",
            "metadata_similarity",
        ),
    ),
}


@dataclass
class ProfileRegistryEntry:
    name: str
    status: str  # "committed" or "omitted"
    omission_reason: Optional[str] = None
    similarity_column: Optional[str] = None
    distance_column: Optional[str] = None
    available_column: Optional[str] = None


def build_profile_registry(
    available_components: Mapping[str, bool],
) -> List[ProfileRegistryEntry]:
    """Decide which profiles are committed vs omitted given component flags.

    available_components: mapping of component name -> bool (e.g.
        {"cosine_embedding_01": True, "bpm_similarity": True, "key_compat":
        False, "metadata_similarity": True}).

    Returns a deterministic list of ProfileRegistryEntry objects, one per
    profile in ALL_PROFILES. Omission reason for a missing component is
    "<component>_unavailable".
    """
    entries: List[ProfileRegistryEntry] = []
    for name, spec in ALL_PROFILES.items():
        missing = [
            c for c in spec.required_components if not available_components.get(c, False)
        ]
        if missing:
            reason = "_unavailable, ".join(missing) + "_unavailable"
            entries.append(
                ProfileRegistryEntry(
                    name=name,
                    status="omitted",
                    omission_reason=reason,
                )
            )
        else:
            entries.append(
                ProfileRegistryEntry(
                    name=name,
                    status="committed",
                    similarity_column=f"profile_{name}_similarity",
                    distance_column=f"profile_{name}_distance",
                    available_column=f"profile_{name}_available",
                )
            )
    return entries


def compute_profile_columns(
    df: pd.DataFrame,
    profile: ProfileSpec,
    component_columns: Mapping[str, str],
    component_available_columns: Mapping[str, str],
) -> Dict[str, np.ndarray]:
    """Compute similarity, distance, and availability arrays for a profile.

    component_columns maps each profile component name to a column in df
    holding its [0, 1]-scaled value. component_available_columns maps each to
    a boolean availability column. All components must be present in df.

    The output similarity is the weighted sum (weights re-normalized over the
    components present in df, which by contract is all of them); per-pair the
    profile is unavailable if any required component is unavailable for that
    pair.
    """
    n = len(df)
    weights = profile.weights
    if not np.isclose(sum(weights.values()), 1.0, atol=1e-9):
        raise ValueError(
            f"profile {profile.name}: weights must sum to 1.0, got {sum(weights.values())}"
        )

    avail = np.ones(n, dtype=bool)
    for comp in profile.required_components:
        col = component_available_columns[comp]
        avail = avail & df[col].to_numpy(dtype=bool, copy=False)

    sim = np.zeros(n, dtype=np.float64)
    for comp, w in weights.items():
        col = component_columns[comp]
        vals = df[col].to_numpy(dtype=np.float64, copy=False)
        # Where component is unavailable, value is NaN; mask handled at the end.
        sim = sim + np.where(np.isnan(vals), 0.0, vals) * w

    sim_out = np.where(avail, sim, np.nan).astype(np.float32, copy=False)
    dist_out = np.where(avail, 1.0 - sim, np.nan).astype(np.float32, copy=False)

    # Numerical clamp to handle minor float drift outside [0, 1].
    if avail.any():
        clamp = np.clip(sim_out[avail], 0.0, 1.0)
        sim_out2 = sim_out.copy()
        sim_out2[avail] = clamp
        dist_out2 = dist_out.copy()
        dist_out2[avail] = (1.0 - clamp).astype(np.float32, copy=False)
        sim_out = sim_out2
        dist_out = dist_out2

    return {
        f"profile_{profile.name}_similarity": sim_out,
        f"profile_{profile.name}_distance": dist_out,
        f"profile_{profile.name}_available": avail,
    }


def attach_committed_profiles(
    df: pd.DataFrame,
    registry: List[ProfileRegistryEntry],
    component_columns: Mapping[str, str],
    component_available_columns: Mapping[str, str],
) -> pd.DataFrame:
    """Attach all committed profiles' columns to df in registry order."""
    out = df
    for entry in registry:
        if entry.status != "committed":
            continue
        spec = ALL_PROFILES[entry.name]
        cols = compute_profile_columns(
            out, spec, component_columns, component_available_columns
        )
        for k, v in cols.items():
            out[k] = v
    return out
