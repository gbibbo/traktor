"""
PURPOSE: MERT model loading, forward pass, and layer aggregation for the DJ clustering
         feature extraction pipeline. Designed to run inside Apptainer on a100 GPU nodes.

CHANGELOG:
  D2.2 - Initial implementation.
"""

from __future__ import annotations

import importlib.util
from typing import Dict, List, Optional, Tuple

import numpy as np

_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
_TORCH = importlib.util.find_spec("torch") is not None

# Layer index mapping for MERT-v1-330M.
# The model outputs hidden_states as a tuple of (n_layers + 1) tensors:
#   index 0 = CNN feature extractor output (input to transformer)
#   indices 1..24 = transformer layer outputs
_AGGREGATION_LAYER_MAP = {
    "last_layer_mean": -1,         # hidden_states[-1]
    "layer_7_mean": 7,             # hidden_states[7]
}
_LAST_4_LAYERS_AGG = "last_4_layers_mean"  # handled separately


def mert_available() -> bool:
    """Return True if torch and transformers are importable."""
    return _TRANSFORMERS and _TORCH


def load_mert(
    model_name: str,
    device: str,
    trust_remote_code: bool = True,
) -> Tuple:
    """
    Load MERT model and processor. Returns (model, processor).

    Raises ImportError if transformers or torch are missing.
    Raises RuntimeError on model load failure.
    """
    if not _TRANSFORMERS:
        raise ImportError("transformers is required for MERT but is not installed")
    if not _TORCH:
        raise ImportError("torch is required for MERT but is not installed")

    import torch
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    model = model.to(device)
    model.eval()
    return model, processor


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a 1-D or 2-D array. Zero vectors are returned unchanged (no NaN)."""
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        return v / norm if norm > 0.0 else v
    # 2-D: normalize each row
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1.0)
    return v / norms


def _aggregate_hidden_states(
    hidden_states: Tuple,
    aggregation: str,
) -> np.ndarray:
    """
    Aggregate MERT hidden states for one forward pass into a 1-D embedding.

    hidden_states : tuple of tensors, each shape (1, T, D)
    Returns shape (D,).
    """
    import torch

    if aggregation == _LAST_4_LAYERS_AGG:
        # Mean of last 4 layers, then mean over time
        layers = torch.stack(list(hidden_states[-4:]), dim=0)  # (4, 1, T, D)
        emb = layers.mean(dim=0).squeeze(0).mean(dim=0)        # (D,)
    elif aggregation in _AGGREGATION_LAYER_MAP:
        idx = _AGGREGATION_LAYER_MAP[aggregation]
        emb = hidden_states[idx].squeeze(0).mean(dim=0)        # (D,)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation!r}")

    return emb.detach().cpu().float().numpy()


def extract_segment_embeddings(
    audio_array: np.ndarray,
    sr: int,
    model,
    processor,
    aggregation_choices: List[str],
    device: str,
) -> Dict[str, np.ndarray]:
    """
    Run one MERT forward pass on a single audio segment.

    Returns a dict mapping aggregation name → 1-D float32 embedding (D,).
    The processor handles resampling; audio should already be at target_sr (24 kHz).
    """
    import torch

    inputs = processor(
        audio_array,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs["input_values"].to(device)

    with torch.no_grad():
        outputs = model(input_values, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # tuple of (1, T, D)

    result: Dict[str, np.ndarray] = {}
    for agg in aggregation_choices:
        result[agg] = _aggregate_hidden_states(hidden_states, agg)

    return result


def aggregate_segments(
    seg_embeddings: List[Dict[str, np.ndarray]],
    aggregation_choices: List[str],
) -> Dict[str, np.ndarray]:
    """
    Aggregate per-segment embeddings into a track-level embedding.

    Policy: l2_normalize each segment embedding → mean → l2_normalize result.
    Returns dict mapping aggregation name → 1-D float32 embedding (D,).
    """
    if not seg_embeddings:
        raise ValueError("seg_embeddings must be non-empty")

    result: Dict[str, np.ndarray] = {}
    for agg in aggregation_choices:
        vecs = np.stack(
            [l2_normalize(e[agg]) for e in seg_embeddings], axis=0
        )  # (n_segs, D)
        track_emb = l2_normalize(vecs.mean(axis=0))  # (D,)
        result[agg] = track_emb

    return result
