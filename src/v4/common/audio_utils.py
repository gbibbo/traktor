"""
PURPOSE: Audio loading, validation and DJ-oriented segmentation for TRAKTOR ML V4.
         Adaptado de legacy/v2/scripts/common/audio_utils.py.
         Cambios V4: SRs importados de config.py (NO hardcoded 16000), carga principal
         via torchaudio (soundfile fallback), añadida get_dj_segments con modo
         beat-aware + fallback porcentual.
CHANGELOG:
  - 2026-02-28: Adaptado de V2. Eliminado Essentia 16kHz. Añadido get_dj_segments.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.v4.config import ESSENTIA_SAMPLE_RATE, SEGMENT_DURATION_S, N_INTRO_SEGMENTS, N_MID_SEGMENTS, N_OUTRO_SEGMENTS, SEGMENT_DURATION_BARS


def get_audio_files(
    audio_dir: Path,
    extensions: Tuple[str, ...] = (".mp3", ".wav", ".flac", ".aiff", ".aif", ".m4a"),
) -> List[Path]:
    """
    Listar archivos de audio en un directorio.

    Returns: Lista ordenada de paths de audio.
    """
    audio_dir = Path(audio_dir)
    audio_files = []
    for ext in extensions:
        audio_files.extend(audio_dir.glob(f"*{ext}"))
        audio_files.extend(audio_dir.glob(f"*{ext.upper()}"))
    return sorted(set(audio_files))


def load_audio_torch(
    audio_path: Path,
    target_sr: int = ESSENTIA_SAMPLE_RATE,
) -> Tuple["torch.Tensor", int]:
    """
    Cargar audio con soundfile (más robusto que torchaudio en este entorno).
    Convierte a mono, resamplea a target_sr.

    Returns: (waveform tensor shape (1, samples), target_sr)
    """
    import soundfile as sf
    import torch
    import torchaudio.transforms as T

    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    # data shape: (samples, channels)
    waveform = torch.from_numpy(data.T)  # (channels, samples)

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform, target_sr


def torch_to_essentia(
    waveform: "torch.Tensor",
    source_sr: int,
    target_sr: int = ESSENTIA_SAMPLE_RATE,
) -> np.ndarray:
    """
    Convertir tensor PyTorch a numpy array mono a target_sr (para Essentia).

    Args:
        waveform: Tensor (channels, samples) o (samples,)
        source_sr: Sample rate del waveform de entrada
        target_sr: Sample rate objetivo (default: ESSENTIA_SAMPLE_RATE = 44100Hz)

    Returns: numpy array float32 mono
    """
    import torchaudio.functional as F

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if source_sr != target_sr:
        waveform = F.resample(waveform, source_sr, target_sr)

    return waveform.squeeze(0).cpu().numpy().astype(np.float32)


def validate_audio_file(audio_path: Path) -> bool:
    """
    Verificar que el archivo existe, no está vacío, y soundfile puede leer su info.
    """
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        return False
    if audio_path.stat().st_size == 0:
        return False
    try:
        import soundfile as sf
        sf.info(str(audio_path))
        return True
    except Exception:
        return False


def get_dj_segments(
    audio: np.ndarray,
    sr: int,
    segment_duration_s: float = SEGMENT_DURATION_S,
    n_intro: int = N_INTRO_SEGMENTS,
    n_mid: int = N_MID_SEGMENTS,
    n_outro: int = N_OUTRO_SEGMENTS,
    beat_ticks: Optional[np.ndarray] = None,
    bpm: Optional[float] = None,
    bars_per_segment: int = SEGMENT_DURATION_BARS,
    beat_conf_threshold: float = 0.5,
    beat_confidence: Optional[float] = None,
) -> List[np.ndarray]:
    """
    Extraer segmentos representativos de un track para embedding.

    Modo beat-aware (si beat_ticks, bpm proporcionados y beat_confidence >= threshold):
      - Barra = 4 beats. Segmento = bars_per_segment barras.
      - Intro: primeros bars_per_segment barras.
      - Mid: barras centrales (distribución equiespaciada).
      - Outro: últimas bars_per_segment barras.
      - Fallback a modo porcentual si beat_ticks < 32 o confianza baja.

    Modo fallback porcentual:
      - Intro: zona 0%-15%. Mid: zona 35%-65%. Outro: zona 85%-100%.
      - Segmentos equiespaciados de segment_duration_s dentro de cada zona.

    Returns: Lista de n_intro+n_mid+n_outro arrays numpy, cada uno de exactamente
             int(segment_duration_s * sr) muestras (pad con ceros si es necesario).
    """
    seg_len = int(segment_duration_s * sr)
    total_samples = len(audio)

    use_beat_mode = False
    if (
        beat_ticks is not None
        and bpm is not None
        and len(beat_ticks) >= 32
        and (beat_confidence is None or beat_confidence >= beat_conf_threshold)
    ):
        use_beat_mode = True

    def _extract_at(start_sample: int) -> np.ndarray:
        end = start_sample + seg_len
        if end <= total_samples:
            return audio[start_sample:end].copy()
        # Pad con ceros
        chunk = audio[start_sample:min(end, total_samples)].copy()
        pad = np.zeros(seg_len - len(chunk), dtype=audio.dtype)
        return np.concatenate([chunk, pad])

    if use_beat_mode:
        beats_per_bar = 4
        samples_per_beat = beat_ticks[1:] - beat_ticks[:-1]
        median_beat_samples = int(np.median(samples_per_beat))
        bar_samples = beats_per_bar * median_beat_samples
        n_bars = total_samples // bar_samples

        intro_starts = [i * bar_samples for i in range(n_intro)]
        outro_starts = [max(0, (n_bars - (n_outro - i)) * bar_samples) for i in range(n_outro)]
        mid_center = n_bars // 2
        mid_half = n_mid // 2
        mid_starts = [max(0, (mid_center - mid_half + i) * bar_samples) for i in range(n_mid)]

        all_starts = intro_starts + mid_starts + outro_starts
    else:
        # Fallback porcentual
        def _zone_starts(zone_start_pct: float, zone_end_pct: float, n: int) -> List[int]:
            zone_start = int(zone_start_pct * total_samples)
            zone_end = int(zone_end_pct * total_samples)
            zone_len = zone_end - zone_start
            if n == 1:
                return [zone_start + zone_len // 2 - seg_len // 2]
            step = (zone_len - seg_len) // max(1, n - 1)
            return [zone_start + i * step for i in range(n)]

        intro_starts = _zone_starts(0.0, 0.15, n_intro)
        mid_starts = _zone_starts(0.35, 0.65, n_mid)
        outro_starts = _zone_starts(0.85, 1.0, n_outro)
        all_starts = intro_starts + mid_starts + outro_starts

    # Clamp starts
    all_starts = [max(0, min(s, total_samples - 1)) for s in all_starts]
    return [_extract_at(s) for s in all_starts]
