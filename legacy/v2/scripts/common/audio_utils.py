"""
PURPOSE: Audio loading and format conversion utilities for TRAKTOR ML V2.
         Supports loading via Essentia and torchaudio, with format conversion
         between PyTorch tensors and numpy arrays.

CHANGELOG:
    2025-02-04: Extracted from V1 extract_embeddings.py for V2 pipeline.
"""
from pathlib import Path
from typing import List, Tuple, Union
import numpy as np

# Essentia imports
import essentia.standard as es


# Constants
ESSENTIA_SAMPLE_RATE = 16000  # Required by Essentia models
DEMUCS_SAMPLE_RATE = 44100    # Demucs default sample rate


def get_audio_files(
    audio_dir: Path,
    extensions: Tuple[str, ...] = (".mp3", ".wav", ".flac")
) -> List[Path]:
    """
    Get all audio files from directory.

    Args:
        audio_dir: Directory to search
        extensions: Tuple of valid audio extensions

    Returns:
        List of audio file paths sorted alphabetically
    """
    audio_files = []
    for ext in extensions:
        audio_files.extend(audio_dir.glob(f"*{ext}"))
        audio_files.extend(audio_dir.glob(f"*{ext.upper()}"))
    return sorted(audio_files)


def load_audio_essentia(
    audio_path: Path,
    sample_rate: int = ESSENTIA_SAMPLE_RATE
) -> np.ndarray:
    """
    Load audio file using Essentia MonoLoader.

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate (default 16kHz for Essentia models)

    Returns:
        Audio signal as numpy array (mono, float32)
    """
    loader = es.MonoLoader(
        filename=str(audio_path),
        sampleRate=sample_rate,
        resampleQuality=4
    )
    audio = loader()
    return audio


def load_audio_torch(audio_path: Path) -> Tuple["torch.Tensor", int]:
    """
    Load audio file using torchaudio.

    Args:
        audio_path: Path to audio file

    Returns:
        Tuple of (waveform tensor, sample_rate)
        waveform shape: (channels, samples)
    """
    import torchaudio
    waveform, sr = torchaudio.load(str(audio_path))
    return waveform, sr


def torch_to_essentia(
    waveform: "torch.Tensor",
    source_sr: int,
    target_sr: int = ESSENTIA_SAMPLE_RATE
) -> np.ndarray:
    """
    Convert PyTorch waveform tensor to Essentia-compatible numpy array.

    Handles:
    - Stereo to mono conversion
    - Resampling to target sample rate
    - Tensor to numpy conversion

    Args:
        waveform: PyTorch tensor of shape (channels, samples) or (samples,)
        source_sr: Source sample rate
        target_sr: Target sample rate (default 16kHz)

    Returns:
        Mono audio as numpy array (float32)
    """
    import torch
    import torchaudio.functional as F

    # Ensure 2D: (channels, samples)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Convert to mono by averaging channels
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if source_sr != target_sr:
        waveform = F.resample(waveform, source_sr, target_sr)

    # Convert to numpy (mono)
    audio = waveform.squeeze(0).cpu().numpy().astype(np.float32)

    return audio


def validate_audio_file(audio_path: Path) -> bool:
    """
    Check if audio file exists and is readable.

    Args:
        audio_path: Path to audio file

    Returns:
        True if file is valid, False otherwise
    """
    if not audio_path.exists():
        return False
    if not audio_path.is_file():
        return False
    if audio_path.stat().st_size == 0:
        return False
    return True
