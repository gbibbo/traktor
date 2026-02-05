"""
PURPOSE: Demucs stem separation utilities for TRAKTOR ML V2.
         Provides in-memory stem separation without writing intermediate WAV files.

CHANGELOG:
    2025-02-04: Initial implementation for V2 drum-first hierarchy pipeline.
"""
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

import torch
import torchaudio.transforms as T


# Constants
DEMUCS_SAMPLE_RATE = 44100
DEMUCS_SOURCES = ["drums", "bass", "other", "vocals"]


def load_demucs_model(
    model_name: str = "htdemucs",
    device: str = "cuda"
) -> Tuple["torch.nn.Module", int]:
    """
    Load Demucs model for stem separation.

    Args:
        model_name: Model name ('htdemucs', 'htdemucs_ft', etc.)
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        Tuple of (model, sample_rate)
    """
    from demucs.pretrained import get_model

    model = get_model(model_name)
    model.to(device)
    model.eval()

    return model, model.samplerate


def load_audio_for_demucs(
    audio_path: Path,
    target_sr: int = DEMUCS_SAMPLE_RATE
) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and prepare for Demucs processing.
    Uses soundfile for HPC compatibility (no FFmpeg required).

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (default 44100 for Demucs)

    Returns:
        Tuple of (waveform tensor, sample_rate)
        waveform shape: (channels, samples)
    """
    import soundfile as sf

    # Load with soundfile
    audio, sr = sf.read(str(audio_path), dtype='float32')

    # Convert to torch tensor
    if audio.ndim == 1:
        # Mono: shape (samples,) -> (1, samples)
        waveform = torch.from_numpy(audio).unsqueeze(0)
    else:
        # Stereo/multi: shape (samples, channels) -> (channels, samples)
        waveform = torch.from_numpy(audio.T)

    # Resample if needed
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        waveform = resampler(waveform)

    # Ensure stereo (Demucs expects 2 channels)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2, :]

    return waveform, target_sr


def separate_stems(
    model: "torch.nn.Module",
    waveform: torch.Tensor,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Separate audio into stems using Demucs (in memory).

    Args:
        model: Loaded Demucs model
        waveform: Input waveform tensor (channels, samples)
        device: Device for processing

    Returns:
        Dict mapping stem name to tensor: {'drums': tensor, 'bass': tensor, ...}
        Each tensor has shape (channels, samples)
    """
    from demucs.apply import apply_model

    # Add batch dimension: (1, channels, samples)
    waveform = waveform.to(device)
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)

    # Separate
    with torch.no_grad():
        sources = apply_model(model, waveform, device=device)[0]
        # sources shape: (n_sources, channels, samples)

    # Map to dict
    stems = {}
    for i, source_name in enumerate(model.sources):
        stems[source_name] = sources[i].cpu()

    return stems


def get_drum_stem(
    stems: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    Extract drum stem from separated sources.

    Args:
        stems: Dict from separate_stems()

    Returns:
        Drum stem tensor (channels, samples)
    """
    return stems["drums"]


def stem_to_mono_numpy(
    stem: torch.Tensor,
    target_sr: int = 16000,
    source_sr: int = DEMUCS_SAMPLE_RATE
) -> np.ndarray:
    """
    Convert stem tensor to mono numpy array at target sample rate.

    Args:
        stem: Stem tensor (channels, samples)
        target_sr: Target sample rate (default 16kHz for Essentia)
        source_sr: Source sample rate (default 44100 from Demucs)

    Returns:
        Mono audio as numpy array (float32)
    """
    # Convert to mono
    if stem.dim() == 2 and stem.shape[0] > 1:
        stem = stem.mean(dim=0, keepdim=True)
    elif stem.dim() == 1:
        stem = stem.unsqueeze(0)

    # Resample if needed
    if source_sr != target_sr:
        resampler = T.Resample(source_sr, target_sr)
        stem = resampler(stem)

    # Convert to numpy
    audio = stem.squeeze(0).numpy().astype(np.float32)

    return audio


def process_track_stems(
    audio_path: Path,
    model: "torch.nn.Module",
    model_sr: int,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a single track: load, separate, return drum and full audio.

    This is the main function for phase1_extraction.py that processes
    audio completely in memory without writing intermediate files.

    Args:
        audio_path: Path to audio file
        model: Loaded Demucs model
        model_sr: Model sample rate
        device: Device for processing

    Returns:
        Tuple of (drums_audio_16k, full_audio_16k)
        Both as mono numpy arrays at 16kHz for Essentia
    """
    # Load audio
    waveform, sr = load_audio_for_demucs(audio_path, target_sr=model_sr)

    # Separate stems
    stems = separate_stems(model, waveform, device=device)

    # Get drum stem as 16kHz mono numpy
    drums = get_drum_stem(stems)
    drums_16k = stem_to_mono_numpy(drums, target_sr=16000, source_sr=model_sr)

    # Get full mix as 16kHz mono numpy
    full_mono = waveform.mean(dim=0, keepdim=True)  # Stereo to mono
    full_16k = stem_to_mono_numpy(full_mono, target_sr=16000, source_sr=model_sr)

    # Clear GPU memory
    del stems, waveform
    if device == "cuda":
        torch.cuda.empty_cache()

    return drums_16k, full_16k
