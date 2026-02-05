"""
PURPOSE: Embedding extraction utilities using Essentia TensorFlow models.
         Supports discogs-effnet-bs64 (1280-dim) and genre_discogs400 classifier.

CHANGELOG:
    2025-02-04: Extracted from V1 extract_embeddings.py for V2 pipeline.
"""
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np

import essentia.standard as es


# Constants
EFFNET_EMBEDDING_DIM = 1280
GENRE_TOP_N = 3  # Number of top genres to return


def extract_effnet_embedding(
    audio: np.ndarray,
    model_path: Path
) -> np.ndarray:
    """
    Extract embedding using discogs-effnet-bs64.

    The model outputs activations from the penultimate layer.
    Output key: "PartitionedCall:1" gives embeddings.

    Args:
        audio: Audio signal at 16kHz (mono, float32)
        model_path: Path to .pb model file

    Returns:
        np.ndarray: Shape (1280,) embedding vector (time-averaged)
    """
    model = es.TensorflowPredictEffnetDiscogs(
        graphFilename=str(model_path),
        output="PartitionedCall:1"
    )
    embeddings = model(audio)
    # Model returns (N, 1280) where N depends on audio length
    # Average pool across time dimension
    embedding = np.mean(embeddings, axis=0)
    return embedding


def extract_genre_predictions(
    audio: np.ndarray,
    effnet_model_path: Path,
    genre_model_path: Path,
    top_n: int = GENRE_TOP_N
) -> Tuple[List[str], List[float]]:
    """
    Extract genre predictions using genre_discogs400 classifier.

    This is a two-stage process:
    1. Extract embeddings with effnet
    2. Classify using genre_discogs400 head

    Args:
        audio: Audio signal at 16kHz (mono, float32)
        effnet_model_path: Path to effnet .pb model
        genre_model_path: Path to genre classifier .pb model
        top_n: Number of top genres to return

    Returns:
        Tuple of (genre_names, confidences) lists
    """
    import json

    # Load genre metadata for label names
    genre_json_path = genre_model_path.with_suffix('.json')
    if genre_json_path.exists():
        with open(genre_json_path, 'r') as f:
            genre_metadata = json.load(f)
            genre_labels = genre_metadata.get('classes', [])
    else:
        genre_labels = [f"genre_{i}" for i in range(400)]

    # Extract embeddings first
    effnet_model = es.TensorflowPredictEffnetDiscogs(
        graphFilename=str(effnet_model_path),
        output="PartitionedCall:1"
    )
    embeddings = effnet_model(audio)

    # Average embeddings over time
    avg_embedding = np.mean(embeddings, axis=0, keepdims=True).astype(np.float32)

    # Apply genre classifier using correct input/output nodes
    # genre_discogs400 expects embeddings as input, not raw audio
    genre_model = es.TensorflowPredict2D(
        graphFilename=str(genre_model_path),
        input="serving_default_model_Placeholder",
        output="PartitionedCall:0"
    )

    predictions = genre_model(avg_embedding)

    # Get top-N predictions
    if predictions.ndim > 1:
        predictions = predictions.squeeze()

    top_indices = np.argsort(predictions)[::-1][:top_n]
    top_genres = [genre_labels[i] if i < len(genre_labels) else f"genre_{i}" for i in top_indices]
    top_confidences = [float(predictions[i]) for i in top_indices]

    return top_genres, top_confidences


def load_effnet_model(model_path: Path) -> es.TensorflowPredictEffnetDiscogs:
    """
    Load and return effnet model for reuse.

    Args:
        model_path: Path to .pb model file

    Returns:
        Loaded Essentia model object
    """
    return es.TensorflowPredictEffnetDiscogs(
        graphFilename=str(model_path),
        output="PartitionedCall:1"
    )


def batch_extract_embeddings(
    model: es.TensorflowPredictEffnetDiscogs,
    audio: np.ndarray
) -> np.ndarray:
    """
    Extract embedding using pre-loaded model.

    Args:
        model: Pre-loaded effnet model
        audio: Audio signal at 16kHz (mono, float32)

    Returns:
        np.ndarray: Shape (1280,) embedding vector (time-averaged)
    """
    embeddings = model(audio)
    embedding = np.mean(embeddings, axis=0)
    return embedding
