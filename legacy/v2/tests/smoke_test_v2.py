#!/usr/bin/env python3
"""
PURPOSE: Smoke test for V2 pipeline integration.
         Validates that Demucs and Essentia work together correctly.
         Uses a single track with optional duration limit for fast CPU testing.

CHANGELOG:
    2025-02-05: Initial creation for V2 integration testing.

USAGE:
    # Quick CPU test (30 seconds of audio)
    python tests/smoke_test_v2.py --device cpu --max-duration 30

    # Full track on GPU
    python tests/smoke_test_v2.py --device cuda

    # Specific audio file
    python tests/smoke_test_v2.py --audio-file path/to/track.mp3 --device cpu
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_audio_clipped(audio_path: Path, max_duration_sec: float = None) -> tuple:
    """
    Load audio file with optional duration limit.
    Uses soundfile for better HPC compatibility (no FFmpeg required).

    Args:
        audio_path: Path to audio file
        max_duration_sec: Maximum duration in seconds (None = full track)

    Returns:
        Tuple of (waveform as torch.Tensor, sample_rate)
    """
    import soundfile as sf

    # Load audio with soundfile
    audio, sr = sf.read(str(audio_path), dtype='float32')

    # Convert to torch tensor with shape (channels, samples)
    if audio.ndim == 1:
        # Mono: add channel dimension
        waveform = torch.from_numpy(audio).unsqueeze(0)
    else:
        # Stereo/multi: transpose to (channels, samples)
        waveform = torch.from_numpy(audio.T)

    # Clip to max duration if specified
    if max_duration_sec is not None:
        num_frames = int(max_duration_sec * sr)
        if waveform.shape[1] > num_frames:
            waveform = waveform[:, :num_frames]

    return waveform, sr


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for V2 pipeline (Demucs + Essentia integration)"
    )
    parser.add_argument(
        "--audio-file",
        type=Path,
        default=None,
        help="Path to audio file (default: first file in test_20)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cuda", "cpu"],
        help="Device for processing (default: cpu for quick testing)"
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Max audio duration in seconds (default: 30s for fast testing)"
    )
    parser.add_argument(
        "--effnet-model",
        type=Path,
        default=PROJECT_ROOT / "models/essentia/discogs-effnet-bs64-1.pb",
        help="Path to Essentia EffNet model"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("TRAKTOR ML V2 - SMOKE TEST")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Max duration: {args.max_duration}s")
    print()

    # 1. Find audio file
    if args.audio_file is None:
        test_dir = PROJECT_ROOT / "data/raw_audio/test_20"
        audio_files = list(test_dir.glob("*.mp3")) + list(test_dir.glob("*.wav"))
        if not audio_files:
            print("[ERROR] No audio files found in test_20")
            return 1
        audio_path = audio_files[0]
    else:
        audio_path = args.audio_file
        if not audio_path.exists():
            print(f"[ERROR] Audio file not found: {audio_path}")
            return 1

    print(f"[1/5] Audio file: {audio_path.name}")

    # 2. Check EffNet model
    if not args.effnet_model.exists():
        print(f"[ERROR] EffNet model not found: {args.effnet_model}")
        print("[HINT] Run: python scripts/hpc/process/v2/download_models.py")
        return 1
    print(f"[2/5] EffNet model: {args.effnet_model.name}")

    # 3. Load Demucs
    print(f"[3/5] Loading Demucs model (htdemucs)...")
    try:
        from scripts.common.demucs_utils import load_demucs_model
        demucs_model, demucs_sr = load_demucs_model("htdemucs", args.device)
        print(f"      Demucs loaded (sr={demucs_sr})")
    except ImportError as e:
        print(f"[ERROR] Failed to import Demucs: {e}")
        print("[HINT] Install with: pip install --user demucs")
        return 1
    except Exception as e:
        print(f"[ERROR] Failed to load Demucs model: {e}")
        return 1

    # 4. Load EffNet
    print(f"[4/5] Loading EffNet model...")
    try:
        from scripts.common.embedding_utils import load_effnet_model
        effnet_model = load_effnet_model(args.effnet_model)
        print(f"      EffNet loaded")
    except Exception as e:
        print(f"[ERROR] Failed to load EffNet model: {e}")
        return 1

    # 5. Process track
    print(f"[5/5] Processing track (this may take a moment on CPU)...")
    try:
        from scripts.common.demucs_utils import (
            load_audio_for_demucs,
            separate_stems,
            get_drum_stem,
            stem_to_mono_numpy
        )
        from scripts.common.embedding_utils import batch_extract_embeddings

        # Load audio (clipped for speed)
        waveform, sr = load_audio_clipped(audio_path, args.max_duration)
        duration = waveform.shape[1] / sr
        print(f"      Loaded {duration:.1f}s of audio")

        # Resample to Demucs sample rate if needed
        if sr != demucs_sr:
            waveform = torchaudio.functional.resample(waveform, sr, demucs_sr)

        # Ensure stereo
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]

        # Separate stems
        print("      Separating stems with Demucs...")
        stems = separate_stems(demucs_model, waveform, device=args.device)
        print(f"      Stems separated: {list(stems.keys())}")

        # Get drums
        drums = get_drum_stem(stems)
        drums_16k = stem_to_mono_numpy(drums, target_sr=16000, source_sr=demucs_sr)
        print(f"      Drums audio shape: {drums_16k.shape}")

        # Get full mix
        full_mono = waveform.mean(dim=0, keepdim=True)
        full_16k = stem_to_mono_numpy(full_mono, target_sr=16000, source_sr=demucs_sr)
        print(f"      Full audio shape: {full_16k.shape}")

        # Extract embeddings
        print("      Extracting embeddings...")
        drum_embedding = batch_extract_embeddings(effnet_model, drums_16k)
        full_embedding = batch_extract_embeddings(effnet_model, full_16k)

    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 6. Validate results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Check shapes
    print(f"Drum embedding shape:     {drum_embedding.shape}")
    print(f"Fulltrack embedding shape: {full_embedding.shape}")

    expected_shape = (1280,)
    shape_ok = drum_embedding.shape == expected_shape and full_embedding.shape == expected_shape

    if shape_ok:
        print(f"[PASS] Shapes are correct: {expected_shape}")
    else:
        print(f"[FAIL] Expected shape {expected_shape}")
        return 1

    # Check non-zero
    drum_nonzero = not np.allclose(drum_embedding, 0)
    full_nonzero = not np.allclose(full_embedding, 0)

    print(f"Drum embedding non-zero:  {drum_nonzero}")
    print(f"Full embedding non-zero:  {full_nonzero}")

    if drum_nonzero and full_nonzero:
        print("[PASS] Embeddings are non-zero")
    else:
        print("[FAIL] Embeddings are all zeros!")
        return 1

    # Check reasonable values
    drum_stats = f"min={drum_embedding.min():.4f}, max={drum_embedding.max():.4f}, mean={drum_embedding.mean():.4f}"
    full_stats = f"min={full_embedding.min():.4f}, max={full_embedding.max():.4f}, mean={full_embedding.mean():.4f}"

    print(f"Drum stats:  {drum_stats}")
    print(f"Full stats:  {full_stats}")

    # Check correlation (drums and full should be somewhat different)
    correlation = np.corrcoef(drum_embedding, full_embedding)[0, 1]
    print(f"Correlation drum/full: {correlation:.4f}")

    if correlation < 0.99:
        print("[PASS] Embeddings are distinct (correlation < 0.99)")
    else:
        print("[WARN] Embeddings are nearly identical - check Demucs separation")

    print()
    print("=" * 70)
    print("[SUCCESS] Smoke test passed!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Submit the full Phase 1 job:")
    print("     ./slurm/tools/on_submit.sh sbatch slurm/jobs/v2/v2_phase1.job")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
