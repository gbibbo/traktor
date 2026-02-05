"""
PURPOSE: Extract audio embeddings using Essentia TensorFlow models.
         Supports discogs-effnet-bs64 (1280-dim) and discogs-maest-30s-pw (768-dim).
         Selects 20 random tracks (seed=42) for Phase 1 validation.

CHANGELOG:
    2025-02-03: Initial implementation for Phase 1 validation.
"""
from pathlib import Path
from typing import List, Tuple, Optional
import argparse
import json
import random
import sys

import numpy as np
from tqdm import tqdm

# Essentia imports
import essentia.standard as es


# Constants
SAMPLE_RATE = 16000  # Required by both models
EFFNET_EMBEDDING_DIM = 1280
MAEST_EMBEDDING_DIM = 768
RANDOM_SEED = 42
DEFAULT_N_TRACKS = 20


def get_audio_files(audio_dir: Path, extensions: Tuple[str, ...] = (".mp3", ".wav", ".flac")) -> List[Path]:
    """
    Get all audio files from directory.

    Args:
        audio_dir: Directory to search
        extensions: Tuple of valid audio extensions

    Returns:
        List of audio file paths
    """
    audio_files = []
    for ext in extensions:
        audio_files.extend(audio_dir.glob(f"*{ext}"))
        audio_files.extend(audio_dir.glob(f"*{ext.upper()}"))
    return sorted(audio_files)


def select_random_tracks(audio_files: List[Path], n_tracks: int, seed: int = RANDOM_SEED) -> List[Path]:
    """
    Select n random tracks from list with fixed seed for reproducibility.

    Args:
        audio_files: List of all audio files
        n_tracks: Number of tracks to select
        seed: Random seed for reproducibility

    Returns:
        List of selected audio file paths
    """
    random.seed(seed)
    if n_tracks >= len(audio_files):
        return audio_files
    return random.sample(audio_files, n_tracks)


def load_audio(audio_path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Load audio file and resample to target sample rate.

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate (default 16kHz)

    Returns:
        Audio signal as numpy array
    """
    loader = es.MonoLoader(filename=str(audio_path), sampleRate=sample_rate, resampleQuality=4)
    audio = loader()
    return audio


def extract_effnet_embedding(audio: np.ndarray, model_path: Path) -> np.ndarray:
    """
    Extract embedding using discogs-effnet-bs64.

    The model outputs activations from the penultimate layer.
    Output key: "PartitionedCall:1" gives embeddings of shape (1280,)

    Args:
        audio: Audio signal at 16kHz
        model_path: Path to .pb model file

    Returns:
        np.ndarray: Shape (1280,) embedding vector
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


def extract_maest_embedding(audio: np.ndarray, model_path: Path) -> np.ndarray:
    """
    Extract embedding using discogs-maest-30s-pw.

    MAEST processes 30-second chunks. For longer audio, we take the first 30s.
    Output key for embeddings (CLS token): varies by version.

    Args:
        audio: Audio signal at 16kHz
        model_path: Path to .pb model file

    Returns:
        np.ndarray: Shape (768,) embedding vector
    """
    # MAEST expects 30 seconds of audio at 16kHz = 480000 samples
    target_length = 30 * SAMPLE_RATE

    if len(audio) > target_length:
        # Take first 30 seconds
        audio = audio[:target_length]
    elif len(audio) < target_length:
        # Pad with zeros
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')

    model = es.TensorflowPredictMAEST(
        graphFilename=str(model_path),
        output="PartitionedCall:7"  # CLS token embeddings
    )
    embedding = model(audio)
    # Should return (768,) directly or (1, 768)
    if embedding.ndim > 1:
        embedding = embedding.squeeze()
    return embedding


def process_tracks(
    audio_files: List[Path],
    model_dir: Path,
    models: List[str] = ["effnet", "maest"],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """
    Process all audio files and extract embeddings.

    Args:
        audio_files: List of audio file paths to process
        model_dir: Directory containing model files
        models: List of models to use ("effnet", "maest", or both)

    Returns:
        effnet_embeddings: Shape (N, 1280) or None
        maest_embeddings: Shape (N, 768) or None
        filenames: List of processed filenames
    """
    # Find model files
    effnet_model = None
    maest_model = None

    if "effnet" in models:
        effnet_files = list(model_dir.glob("*effnet*.pb"))
        if effnet_files:
            effnet_model = effnet_files[0]
            print(f"[INFO] EffNet model: {effnet_model}")
        else:
            print("[ERROR] EffNet model not found!")
            if "effnet" in models:
                models.remove("effnet")

    if "maest" in models:
        maest_files = list(model_dir.glob("*maest*.pb"))
        if maest_files:
            maest_model = maest_files[0]
            print(f"[INFO] MAEST model: {maest_model}")
        else:
            print("[ERROR] MAEST model not found!")
            if "maest" in models:
                models.remove("maest")

    if not models:
        raise RuntimeError("No models available for extraction!")

    # Initialize storage
    effnet_embeddings = [] if "effnet" in models else None
    maest_embeddings = [] if "maest" in models else None
    filenames = []
    failed_files = []

    print(f"\n[INFO] Processing {len(audio_files)} tracks...")

    for audio_path in tqdm(audio_files, desc="Extracting embeddings"):
        try:
            # Load audio
            audio = load_audio(audio_path)

            # Extract EffNet embedding
            if "effnet" in models:
                emb_effnet = extract_effnet_embedding(audio, effnet_model)
                effnet_embeddings.append(emb_effnet)

            # Extract MAEST embedding
            if "maest" in models:
                emb_maest = extract_maest_embedding(audio, maest_model)
                maest_embeddings.append(emb_maest)

            filenames.append(audio_path.name)

        except Exception as e:
            print(f"\n[WARN] Failed to process {audio_path.name}: {e}")
            failed_files.append(audio_path.name)
            continue

    # Convert to numpy arrays
    if effnet_embeddings:
        effnet_embeddings = np.stack(effnet_embeddings, axis=0)
        print(f"[INFO] EffNet embeddings shape: {effnet_embeddings.shape}")
    else:
        effnet_embeddings = None

    if maest_embeddings:
        maest_embeddings = np.stack(maest_embeddings, axis=0)
        print(f"[INFO] MAEST embeddings shape: {maest_embeddings.shape}")
    else:
        maest_embeddings = None

    if failed_files:
        print(f"\n[WARN] Failed to process {len(failed_files)} files:")
        for f in failed_files[:5]:
            print(f"  - {f}")
        if len(failed_files) > 5:
            print(f"  ... and {len(failed_files) - 5} more")

    return effnet_embeddings, maest_embeddings, filenames


def save_results(
    output_dir: Path,
    effnet_embeddings: Optional[np.ndarray],
    maest_embeddings: Optional[np.ndarray],
    filenames: List[str],
) -> None:
    """
    Save embeddings and manifest to output directory.

    Args:
        output_dir: Directory to save results
        effnet_embeddings: EffNet embeddings array or None
        maest_embeddings: MAEST embeddings array or None
        filenames: List of processed filenames
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    if effnet_embeddings is not None:
        effnet_path = output_dir / "embeddings_effnet.npy"
        np.save(effnet_path, effnet_embeddings)
        print(f"[SAVED] {effnet_path}")

    if maest_embeddings is not None:
        maest_path = output_dir / "embeddings_maest.npy"
        np.save(maest_path, maest_embeddings)
        print(f"[SAVED] {maest_path}")

    # Save manifest with filenames
    manifest = {
        "filenames": filenames,
        "n_tracks": len(filenames),
        "models": {
            "effnet": {
                "dim": EFFNET_EMBEDDING_DIM,
                "file": "embeddings_effnet.npy" if effnet_embeddings is not None else None
            },
            "maest": {
                "dim": MAEST_EMBEDDING_DIM,
                "file": "embeddings_maest.npy" if maest_embeddings is not None else None
            }
        }
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[SAVED] {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract audio embeddings using Essentia TensorFlow models"
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Directory containing audio files"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/essentia"),
        help="Directory containing Essentia model files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save embeddings"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["effnet", "maest"],
        choices=["effnet", "maest"],
        help="Models to use for extraction"
    )
    parser.add_argument(
        "--n-tracks",
        type=int,
        default=DEFAULT_N_TRACKS,
        help=f"Number of tracks to process (default: {DEFAULT_N_TRACKS}, use -1 for all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for track selection (default: {RANDOM_SEED})"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("TRAKTOR ML - Audio Embedding Extractor")
    print("=" * 60)
    print(f"Audio directory: {args.audio_dir}")
    print(f"Model directory: {args.model_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Models: {args.models}")
    print(f"N tracks: {args.n_tracks}")
    print(f"Seed: {args.seed}")
    print()

    # Get audio files
    audio_files = get_audio_files(args.audio_dir)
    print(f"[INFO] Found {len(audio_files)} audio files in {args.audio_dir}")

    if not audio_files:
        print("[ERROR] No audio files found!")
        return 1

    # Select random subset
    if args.n_tracks > 0 and args.n_tracks < len(audio_files):
        audio_files = select_random_tracks(audio_files, args.n_tracks, args.seed)
        print(f"[INFO] Selected {len(audio_files)} random tracks (seed={args.seed})")

    # Process tracks
    effnet_emb, maest_emb, filenames = process_tracks(
        audio_files,
        args.model_dir,
        args.models
    )

    if not filenames:
        print("[ERROR] No tracks were processed successfully!")
        return 1

    # Save results
    save_results(args.output_dir, effnet_emb, maest_emb, filenames)

    print("\n" + "=" * 60)
    print("[SUCCESS] Embedding extraction complete!")
    print(f"  Processed: {len(filenames)} tracks")
    if effnet_emb is not None:
        print(f"  EffNet: {effnet_emb.shape}")
    if maest_emb is not None:
        print(f"  MAEST: {maest_emb.shape}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
