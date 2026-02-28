"""
PURPOSE: Demucs stem separation utilities for TRAKTOR ML V4.
         Adaptado de legacy/v2/scripts/common/demucs_utils.py.
         Cambio clave V4: target_sr default = MERT_SAMPLE_RATE (24000), NO 16000.
         Demucs opera internamente a 44.1kHz. El resample a 24kHz se aplica DESPUÉS
         de la separación, al convertir stems a numpy.
CHANGELOG:
  - 2026-02-28: Adaptado de V2. Cambiado target_sr de 16000 a MERT_SAMPLE_RATE (24000).
"""
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torchaudio.transforms as T

from src.v4.config import DEMUCS_SAMPLE_RATE, DEMUCS_MODEL_NAME, MERT_SAMPLE_RATE


DEMUCS_SOURCES = ["drums", "bass", "other", "vocals"]


def load_demucs_model(
    model_name: str = DEMUCS_MODEL_NAME,
    device: str = "cuda",
) -> Tuple["torch.nn.Module", int]:
    """
    Cargar modelo Demucs.

    Returns: (model, sample_rate) donde sample_rate es el SR nativo del modelo (44100).
    """
    from demucs.pretrained import get_model

    model = get_model(model_name)
    model.to(device)
    model.eval()
    return model, model.samplerate


def load_audio_for_demucs(
    audio_path: Path,
    target_sr: int = DEMUCS_SAMPLE_RATE,
) -> Tuple[torch.Tensor, int]:
    """
    Cargar audio y preparar para Demucs (usa soundfile, no FFmpeg).

    Returns: (waveform tensor shape (2, samples), target_sr)
             Siempre stereo — Demucs requiere 2 canales.
    """
    import soundfile as sf

    audio, sr = sf.read(str(audio_path), dtype="float32")

    if audio.ndim == 1:
        waveform = torch.from_numpy(audio).unsqueeze(0)
    else:
        waveform = torch.from_numpy(audio.T)  # (channels, samples)

    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        waveform = resampler(waveform)

    # Asegurar stereo
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2, :]

    return waveform, target_sr


def separate_stems(
    model: "torch.nn.Module",
    waveform: torch.Tensor,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Separar audio en stems con Demucs (en memoria, sin escribir WAVs).

    Returns: Dict {'drums': tensor, 'bass': tensor, 'other': tensor, 'vocals': tensor}
             Cada tensor tiene shape (channels, samples).
    """
    from demucs.apply import apply_model

    waveform = waveform.to(device)
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)  # (1, channels, samples)

    with torch.no_grad():
        sources = apply_model(model, waveform, device=device)[0]
        # sources shape: (n_sources, channels, samples)

    stems = {name: sources[i].cpu() for i, name in enumerate(model.sources)}
    return stems


def stem_to_mono_numpy(
    stem: torch.Tensor,
    target_sr: int = MERT_SAMPLE_RATE,
    source_sr: int = DEMUCS_SAMPLE_RATE,
) -> np.ndarray:
    """
    Convertir stem tensor a numpy mono a target_sr.

    Flujo: stem 44.1kHz stereo → mono → resample a target_sr (24kHz para MERT) → numpy.

    Args:
        stem: Tensor (channels, samples)
        target_sr: SR objetivo. Default MERT_SAMPLE_RATE = 24000. NO usar 16000 en V4.
        source_sr: SR del stem (Demucs output = 44100).

    Returns: numpy array float32 mono.
    """
    if stem.dim() == 2 and stem.shape[0] > 1:
        stem = stem.mean(dim=0, keepdim=True)
    elif stem.dim() == 1:
        stem = stem.unsqueeze(0)

    if source_sr != target_sr:
        resampler = T.Resample(source_sr, target_sr)
        stem = resampler(stem)

    return stem.squeeze(0).numpy().astype(np.float32)


def process_track_stems(
    audio_path: Path,
    model: "torch.nn.Module",
    model_sr: int,
    device: str = "cuda",
    target_sr: int = MERT_SAMPLE_RATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Procesar un track completo: cargar → separar → retornar drums y full mix.

    Flujo:
      1. Cargar a 44.1kHz (Demucs nativo).
      2. Demucs separa en drums/bass/other/vocals a 44.1kHz.
      3. Stem drums: resample a target_sr (24kHz) → mono numpy.
      4. Full mix: stereo a mono → resample a target_sr → numpy.

    Args:
        target_sr: SR para la salida. Default MERT_SAMPLE_RATE = 24000. NO usar 16000.

    Returns: (drums_audio, full_audio) — ambos mono numpy float32 a target_sr.
    """
    waveform, sr = load_audio_for_demucs(audio_path, target_sr=model_sr)

    stems = separate_stems(model, waveform, device=device)

    drums_out = stem_to_mono_numpy(stems["drums"], target_sr=target_sr, source_sr=model_sr)

    full_mono = waveform.mean(dim=0, keepdim=True)
    full_out = stem_to_mono_numpy(full_mono, target_sr=target_sr, source_sr=model_sr)

    del stems, waveform
    if device == "cuda":
        torch.cuda.empty_cache()

    return drums_out, full_out
