"""
PURPOSE: Phase 1 of V2 pipeline - Extract drum and fulltrack embeddings.
         Uses Demucs for stem separation and Essentia EffNet for embeddings.
         Processes everything in memory (no intermediate WAV files).

CHANGELOG:
    2025-02-04: Initial implementation for V2 drum-first hierarchy pipeline.
"""
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
import argparse
import json
import sys

import numpy as np
from tqdm import tqdm
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.common.audio_utils import get_audio_files, validate_audio_file
from scripts.common.demucs_utils import load_demucs_model, process_track_stems
from scripts.common.embedding_utils import extract_effnet_embedding, load_effnet_model, batch_extract_embeddings


# Constants
EFFNET_EMBEDDING_DIM = 1280


def process_single_track(
    audio_path: Path,
    demucs_model: torch.nn.Module,
    demucs_sr: int,
    effnet_model,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a single track: separate stems and extract embeddings.

    All processing is done in memory - no intermediate files.

    Args:
        audio_path: Path to audio file
        demucs_model: Loaded Demucs model
        demucs_sr: Demucs sample rate
        effnet_model: Loaded Essentia effnet model
        device: Processing device

    Returns:
        Tuple of (drum_embedding, fulltrack_embedding)
        Each is a 1D numpy array of shape (1280,)
    """
    # Get drum and full audio at 16kHz (in memory)
    drums_16k, full_16k = process_track_stems(
        audio_path, demucs_model, demucs_sr, device=device
    )

    # Extract embeddings
    drum_embedding = batch_extract_embeddings(effnet_model, drums_16k)
    full_embedding = batch_extract_embeddings(effnet_model, full_16k)

    return drum_embedding, full_embedding


def process_all_tracks(
    audio_files: List[Path],
    demucs_model: torch.nn.Module,
    demucs_sr: int,
    effnet_model_path: Path,
    device: str = "cuda",
    checkpoint_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Process all audio files and extract embeddings.

    Args:
        audio_files: List of audio file paths
        demucs_model: Loaded Demucs model
        demucs_sr: Demucs sample rate
        effnet_model_path: Path to effnet model file
        device: Processing device
        checkpoint_path: Optional path for checkpointing progress

    Returns:
        drum_embeddings: Shape (N, 1280)
        fulltrack_embeddings: Shape (N, 1280)
        filenames: List of processed filenames
    """
    # Load effnet model
    effnet_model = load_effnet_model(effnet_model_path)

    # Initialize storage
    drum_embeddings = []
    fulltrack_embeddings = []
    filenames = []
    failed_files = []

    # Load checkpoint if exists
    start_idx = 0
    if checkpoint_path and checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
            start_idx = checkpoint.get('last_processed_idx', 0) + 1
            filenames = checkpoint.get('filenames', [])
            if checkpoint.get('drum_embeddings_file'):
                drum_embeddings = np.load(checkpoint['drum_embeddings_file']).tolist()
                fulltrack_embeddings = np.load(checkpoint['fulltrack_embeddings_file']).tolist()
            print(f"[RESUME] Resuming from index {start_idx}/{len(audio_files)}")

    print(f"\n[INFO] Processing {len(audio_files)} tracks (starting from {start_idx})...")

    for idx in tqdm(range(start_idx, len(audio_files)), desc="Extracting embeddings"):
        audio_path = audio_files[idx]

        try:
            # Validate file
            if not validate_audio_file(audio_path):
                print(f"\n[WARN] Invalid file: {audio_path.name}")
                failed_files.append(audio_path.name)
                continue

            # Process track
            drum_emb, full_emb = process_single_track(
                audio_path, demucs_model, demucs_sr, effnet_model, device
            )

            drum_embeddings.append(drum_emb)
            fulltrack_embeddings.append(full_emb)
            filenames.append(audio_path.name)

            # Checkpoint every 10 tracks
            if checkpoint_path and (idx + 1) % 10 == 0:
                save_checkpoint(
                    checkpoint_path,
                    idx,
                    filenames,
                    np.stack(drum_embeddings),
                    np.stack(fulltrack_embeddings)
                )

        except Exception as e:
            print(f"\n[ERROR] Failed to process {audio_path.name}: {e}")
            failed_files.append(audio_path.name)
            continue

    # Convert to numpy arrays
    drum_embeddings = np.stack(drum_embeddings, axis=0)
    fulltrack_embeddings = np.stack(fulltrack_embeddings, axis=0)

    print(f"\n[INFO] Processed {len(filenames)} tracks successfully")
    print(f"[INFO] Drum embeddings shape: {drum_embeddings.shape}")
    print(f"[INFO] Fulltrack embeddings shape: {fulltrack_embeddings.shape}")

    if failed_files:
        print(f"\n[WARN] Failed to process {len(failed_files)} files:")
        for f in failed_files[:5]:
            print(f"  - {f}")
        if len(failed_files) > 5:
            print(f"  ... and {len(failed_files) - 5} more")

    return drum_embeddings, fulltrack_embeddings, filenames


def save_checkpoint(
    checkpoint_path: Path,
    last_idx: int,
    filenames: List[str],
    drum_embeddings: np.ndarray,
    fulltrack_embeddings: np.ndarray
):
    """Save progress checkpoint."""
    output_dir = checkpoint_path.parent
    drum_file = output_dir / "checkpoint_drum_embeddings.npy"
    full_file = output_dir / "checkpoint_fulltrack_embeddings.npy"

    np.save(drum_file, drum_embeddings)
    np.save(full_file, fulltrack_embeddings)

    checkpoint = {
        'last_processed_idx': last_idx,
        'filenames': filenames,
        'drum_embeddings_file': str(drum_file),
        'fulltrack_embeddings_file': str(full_file),
    }

    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f)


def save_results(
    output_dir: Path,
    drum_embeddings: np.ndarray,
    fulltrack_embeddings: np.ndarray,
    filenames: List[str],
    demucs_model_name: str = "htdemucs",
    effnet_model_name: str = "discogs-effnet-bs64-1",
) -> None:
    """
    Save embeddings and manifest to output directory.

    Args:
        output_dir: Directory to save results
        drum_embeddings: Drum embeddings array
        fulltrack_embeddings: Fulltrack embeddings array
        filenames: List of processed filenames
        demucs_model_name: Name of Demucs model used
        effnet_model_name: Name of effnet model used
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    drum_path = output_dir / "drum_embeddings.npy"
    np.save(drum_path, drum_embeddings)
    print(f"[SAVED] {drum_path}")

    fulltrack_path = output_dir / "fulltrack_embeddings.npy"
    np.save(fulltrack_path, fulltrack_embeddings)
    print(f"[SAVED] {fulltrack_path}")

    # Save manifest
    manifest = {
        "version": "2.0",
        "pipeline": "drum_first_hierarchy",
        "created": datetime.now().isoformat(),
        "tracks": filenames,
        "n_tracks": len(filenames),
        "embeddings": {
            "drum": {
                "file": "drum_embeddings.npy",
                "dim": EFFNET_EMBEDDING_DIM,
                "model": effnet_model_name,
                "source": "demucs_drums_stem"
            },
            "fulltrack": {
                "file": "fulltrack_embeddings.npy",
                "dim": EFFNET_EMBEDDING_DIM,
                "model": effnet_model_name,
                "source": "original_mix"
            }
        },
        "demucs_model": demucs_model_name,
    }

    manifest_path = output_dir / "manifest_v2.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[SAVED] {manifest_path}")

    # Clean up checkpoint files if they exist
    checkpoint_files = [
        output_dir / "checkpoint.json",
        output_dir / "checkpoint_drum_embeddings.npy",
        output_dir / "checkpoint_fulltrack_embeddings.npy",
    ]
    for f in checkpoint_files:
        if f.exists():
            f.unlink()
            print(f"[CLEANUP] Removed checkpoint: {f.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Extract drum and fulltrack embeddings using Demucs + Essentia"
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Directory containing audio files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save embeddings"
    )
    parser.add_argument(
        "--effnet-model",
        type=Path,
        default=Path("models/essentia/discogs-effnet-bs64-1.pb"),
        help="Path to Essentia EffNet model"
    )
    parser.add_argument(
        "--demucs-model",
        type=str,
        default="htdemucs",
        help="Demucs model name (default: htdemucs)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for processing (default: cuda)"
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Enable checkpointing for resume capability"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("TRAKTOR ML V2 - Phase 1: Embedding Extraction")
    print("=" * 60)
    print(f"Audio directory: {args.audio_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"EffNet model: {args.effnet_model}")
    print(f"Demucs model: {args.demucs_model}")
    print(f"Device: {args.device}")
    print()

    # Check CUDA
    if args.device == "cuda":
        if torch.cuda.is_available():
            print(f"[INFO] CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"[INFO] CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("[WARN] CUDA not available, falling back to CPU")
            args.device = "cpu"

    # Get audio files
    audio_files = get_audio_files(args.audio_dir)
    print(f"[INFO] Found {len(audio_files)} audio files")

    if not audio_files:
        print("[ERROR] No audio files found!")
        return 1

    # Verify effnet model
    if not args.effnet_model.exists():
        print(f"[ERROR] EffNet model not found: {args.effnet_model}")
        print("[HINT] Run: python scripts/hpc/process/v2/download_models.py --all")
        return 1

    # Load Demucs model
    print(f"\n[INFO] Loading Demucs model '{args.demucs_model}'...")
    try:
        demucs_model, demucs_sr = load_demucs_model(args.demucs_model, args.device)
        print(f"[INFO] Demucs loaded successfully (sr={demucs_sr})")
    except Exception as e:
        print(f"[ERROR] Failed to load Demucs: {e}")
        print("[HINT] Install with: pip install demucs")
        return 1

    # Setup checkpoint
    checkpoint_path = None
    if args.checkpoint:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = args.output_dir / "checkpoint.json"

    # Process all tracks
    drum_emb, full_emb, filenames = process_all_tracks(
        audio_files,
        demucs_model,
        demucs_sr,
        args.effnet_model,
        args.device,
        checkpoint_path,
    )

    if not filenames:
        print("[ERROR] No tracks were processed successfully!")
        return 1

    # Save results
    save_results(
        args.output_dir,
        drum_emb,
        full_emb,
        filenames,
        args.demucs_model,
        args.effnet_model.stem,
    )

    print("\n" + "=" * 60)
    print("[SUCCESS] Phase 1 complete!")
    print(f"  Processed: {len(filenames)} tracks")
    print(f"  Drum embeddings: {drum_emb.shape}")
    print(f"  Fulltrack embeddings: {full_emb.shape}")
    print(f"  Output: {args.output_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
