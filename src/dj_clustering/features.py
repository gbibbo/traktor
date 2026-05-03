"""
PURPOSE: Feature extraction orchestration for the DJ clustering pipeline.
         Handles per-track MERT full/perc/concat extraction, BPM feature,
         fallback policy, and manifest assembly.

CHANGELOG:
  D2.2 - Initial implementation.
"""

from __future__ import annotations

import math
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.dj_clustering.hpss import hpss_available, hpss_percussive
from src.dj_clustering.mert import (
    aggregate_segments,
    extract_segment_embeddings,
    l2_normalize,
)
from src.dj_clustering.segments import compute_segments

# Required columns in the feature manifest
FEATURE_MANIFEST_COLUMNS = [
    "track_id",
    "feature_source",
    "aggregation",
    "embedding_path",
    "embedding_dim",
    "status",
    "failure_reason",
    "num_segments_used",
]


class FeatureExtractor:
    """
    Orchestrates feature extraction for canonical decoded tracks.

    Parameters
    ----------
    config : dict
        Parsed features.yaml as a Python dict.
    device : str
        Torch device string ('cuda' or 'cpu').
    """

    def __init__(self, config: Dict, device: str) -> None:
        self.config = config
        self.device = device

        audio_cfg = config["audio"]
        self.target_sr: int = audio_cfg["sample_rate"]

        mert_cfg = config["mert"]
        self.aggregation_choices: List[str] = mert_cfg["aggregation_choices"]

        self._hpss_available: bool = _check_hpss_available()

        perc_cfg = config["percussion"]
        # Check if any Demucs stems exist (optional higher-quality source)
        stems_path = Path(config["artifact_root"]) / perc_cfg.get("stems_path", "")
        self._demucs_stems_exist: bool = (
            stems_path.exists() and perc_cfg.get("demucs_if_stems_exist", False)
        )

    def is_hpss_available(self) -> bool:
        return self._hpss_available

    def extract_track(
        self,
        row: pd.Series,
        model,
        processor,
    ) -> Dict[str, Any]:
        """
        Extract all features for one track.

        Returns a result dict with keys:
          track_id, segments_used, mert_full, mert_perc, mert_concat,
          hpss_failed, hpss_failure_reason
        Each embedding value is a dict {aggregation: np.ndarray} or None on failure.
        """
        track_id = str(row["track_id"])
        file_path = Path(str(row["file_path"]))
        duration = float(row["duration_seconds"])

        result: Dict[str, Any] = {
            "track_id": track_id,
            "segments_used": 0,
            "mert_full": None,
            "mert_perc": None,
            "mert_concat": None,
            "hpss_failed": False,
            "hpss_failure_reason": None,
            "mert_full_failure_reason": None,
        }

        segments = compute_segments(track_id, duration)
        result["segments_used"] = len(segments)

        # Load full audio once; slice per segment
        try:
            full_audio = _load_audio(file_path, self.target_sr)
        except Exception as exc:
            reason = f"audio_load_error: {type(exc).__name__}: {str(exc)[:80]}"
            result["mert_full_failure_reason"] = reason
            return result

        # MERT full-mix extraction
        try:
            seg_full_embs = []
            for seg in segments:
                seg_audio = _slice_audio(full_audio, seg["start_sec"], seg["end_sec"], self.target_sr)
                seg_emb = extract_segment_embeddings(
                    seg_audio, self.target_sr, model, processor,
                    self.aggregation_choices, self.device,
                )
                seg_full_embs.append(seg_emb)
            result["mert_full"] = aggregate_segments(seg_full_embs, self.aggregation_choices)
        except Exception as exc:
            result["mert_full_failure_reason"] = (
                f"mert_full_error: {type(exc).__name__}: {str(exc)[:80]}"
            )
            return result

        # MERT percussive extraction (only if HPSS available)
        if self._hpss_available:
            try:
                seg_perc_embs = []
                for seg in segments:
                    seg_audio = _slice_audio(
                        full_audio, seg["start_sec"], seg["end_sec"], self.target_sr
                    )
                    perc_audio = hpss_percussive(seg_audio, self.target_sr)
                    seg_emb = extract_segment_embeddings(
                        perc_audio, self.target_sr, model, processor,
                        self.aggregation_choices, self.device,
                    )
                    seg_perc_embs.append(seg_emb)
                result["mert_perc"] = aggregate_segments(
                    seg_perc_embs, self.aggregation_choices
                )
            except Exception as exc:
                result["hpss_failed"] = True
                result["hpss_failure_reason"] = (
                    f"{type(exc).__name__}: {str(exc)[:80]}"
                )

        # mert_concat: requires both full and perc
        if result["mert_full"] is not None and result["mert_perc"] is not None:
            try:
                result["mert_concat"] = {
                    agg: self.compute_mert_concat(
                        result["mert_full"][agg], result["mert_perc"][agg]
                    )
                    for agg in self.aggregation_choices
                }
            except Exception as exc:
                result["mert_concat"] = None

        return result

    def compute_mert_concat(
        self, full_emb: np.ndarray, perc_emb: np.ndarray
    ) -> np.ndarray:
        """
        Concatenate L2-normalized full and perc embeddings, then L2-normalize.
        Policy: l2_normalize_each_then_concat_then_l2_normalize.
        """
        return l2_normalize(
            np.concatenate([l2_normalize(full_emb), l2_normalize(perc_emb)], axis=0)
        )

    def compute_bpm_feature(
        self, inventory_df: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract and normalize BPM as a scalar feature.

        Returns (array shape (N, 1), track_ids list).
        Null BPM values are imputed with the median of non-null values.
        """
        canonical = inventory_df[
            inventory_df["is_canonical"] & (inventory_df["decode_status"] != "failed")
        ].copy()

        bpms = canonical["bpm"].astype(float)
        median_bpm = float(bpms.dropna().median())
        bpms_filled = bpms.fillna(median_bpm)

        bpm_min = float(bpms_filled.min())
        bpm_max = float(bpms_filled.max())
        denom = bpm_max - bpm_min if bpm_max != bpm_min else 1.0
        normalized = ((bpms_filled - bpm_min) / denom).values.astype(np.float32)

        track_ids = canonical["track_id"].astype(str).tolist()
        return normalized.reshape(-1, 1), track_ids

    def build_feature_manifest(
        self,
        results: List[Dict],
        artifact_root: Path,
    ) -> pd.DataFrame:
        """
        Build the feature manifest DataFrame from per-track extraction results.

        One row per (track_id, feature_source, aggregation) combination.
        """
        rows = []

        for r in results:
            tid = r["track_id"]
            n_segs = r.get("segments_used", 0)

            # mert_full
            for agg in self.aggregation_choices:
                if r["mert_full"] is not None:
                    emb = r["mert_full"][agg]
                    rel_path = str(
                        Path("features/mert_full") / agg / "embeddings.npy"
                    )
                    rows.append(_manifest_row(
                        tid, "mert_full", agg, rel_path,
                        int(emb.shape[0]), "ok", None, n_segs,
                    ))
                else:
                    rows.append(_manifest_row(
                        tid, "mert_full", agg, None, 0, "failed",
                        r.get("mert_full_failure_reason"), n_segs,
                    ))

            # mert_perc
            for agg in self.aggregation_choices:
                if not self._hpss_available:
                    rows.append(_manifest_row(
                        tid, "mert_perc", agg, None, 0, "omitted",
                        "hpss_globally_unavailable", n_segs,
                    ))
                elif r["mert_perc"] is not None:
                    emb = r["mert_perc"][agg]
                    rel_path = str(
                        Path("features/mert_perc") / agg / "embeddings.npy"
                    )
                    rows.append(_manifest_row(
                        tid, "mert_perc", agg, rel_path,
                        int(emb.shape[0]), "ok", None, n_segs,
                    ))
                else:
                    rows.append(_manifest_row(
                        tid, "mert_perc", agg, None, 0, "omitted",
                        r.get("hpss_failure_reason", "hpss_per_track_failure"), n_segs,
                    ))

            # mert_concat
            for agg in self.aggregation_choices:
                if r["mert_concat"] is not None:
                    emb = r["mert_concat"][agg]
                    rel_path = str(
                        Path("features/mert_concat") / agg / "embeddings.npy"
                    )
                    rows.append(_manifest_row(
                        tid, "mert_concat", agg, rel_path,
                        int(emb.shape[0]), "ok", None, n_segs,
                    ))
                else:
                    reason = "mert_perc_unavailable" if r["mert_full"] is not None else "mert_full_failed"
                    rows.append(_manifest_row(
                        tid, "mert_concat", agg, None, 0, "omitted", reason, n_segs,
                    ))

        return pd.DataFrame(rows, columns=FEATURE_MANIFEST_COLUMNS)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_hpss_available() -> bool:
    """Return True if HPSS can be instantiated without error."""
    try:
        return hpss_available()
    except Exception:
        return False


def _load_audio(filepath: Path, target_sr: int) -> np.ndarray:
    """Load full mono audio resampled to target_sr using soundfile."""
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly

    audio, file_sr = sf.read(str(filepath), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # stereo → mono

    if file_sr != target_sr:
        g = gcd(target_sr, file_sr)
        audio = resample_poly(audio, target_sr // g, file_sr // g)

    return audio.astype(np.float32)


def _slice_audio(
    audio: np.ndarray,
    start_sec: float,
    end_sec: float,
    sr: int,
) -> np.ndarray:
    """Slice audio array to [start_sec, end_sec]. Returns available samples if short."""
    start = int(start_sec * sr)
    end = int(end_sec * sr)
    return audio[start:end]


def _manifest_row(
    track_id: str,
    feature_source: str,
    aggregation: str,
    embedding_path: Optional[str],
    embedding_dim: int,
    status: str,
    failure_reason: Optional[str],
    num_segments_used: int,
) -> Dict:
    return {
        "track_id": track_id,
        "feature_source": feature_source,
        "aggregation": aggregation,
        "embedding_path": embedding_path,
        "embedding_dim": embedding_dim,
        "status": status,
        "failure_reason": failure_reason,
        "num_segments_used": num_segments_used,
    }
