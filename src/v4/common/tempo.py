"""
PURPOSE: Estimación de BPM sin Essentia (numpy/scipy) para los temas sin BPM en los tags: envolvente
         de onsets por flujo espectral (log-magnitud, 100 cuadros/s) y autocorrelación con peine de
         múltiplos del período, sumada sobre los segmentos DJ. Pensado para house/techno en 4/4:
         busca en [bpm_min, bpm_max] (por defecto 95-150).
CHANGELOG:
  - 2026-09-27: Creación inicial (fallback de phase1_tags --estimate-missing).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

FPS = 100.0


def onset_envelope(audio: np.ndarray, sr: int, n_fft: int = 2048) -> np.ndarray:
    """Flujo espectral positivo sobre log-magnitud, a FPS cuadros por segundo, sin tendencia local."""
    from scipy.signal import stft
    hop = int(round(sr / FPS))
    _, _, spec = stft(audio.astype(np.float32), fs=sr, nperseg=n_fft, noverlap=n_fft - hop,
                      boundary=None, padded=False)
    mag = np.log1p(100.0 * np.abs(spec))
    flux = np.maximum(0.0, np.diff(mag, axis=1)).sum(axis=0)
    if flux.size == 0:
        return flux
    k = int(FPS)  # quitar la media local de ~1 s
    local = np.convolve(flux, np.ones(k) / k, mode="same")
    return np.maximum(0.0, flux - local)


def _autocorr(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    n = int(2 ** np.ceil(np.log2(2 * len(x))))
    f = np.fft.rfft(x, n)
    ac = np.fft.irfft(f * np.conj(f), n)[:len(x)]
    return ac / ac[0] if ac[0] > 0 else ac


def tempo_curve(env: np.ndarray, bpms: np.ndarray, n_mult: int = 4) -> np.ndarray:
    """Puntaje por BPM candidato: autocorrelación interpolada en 1..n_mult períodos."""
    ac = _autocorr(env)
    lags = np.arange(len(ac))
    score = np.zeros(len(bpms))
    for m in range(1, n_mult + 1):
        lag = m * 60.0 * FPS / bpms
        valid = lag < len(ac) - 1
        score[valid] += np.interp(lag[valid], lags, ac) / m
    return score


def estimate_bpm(segments: List[np.ndarray], sr: int, bpm_min: float = 95.0,
                 bpm_max: float = 150.0) -> Optional[float]:
    """BPM estimado (resolución 0.05) sumando la curva de cada segmento; None si no hay señal."""
    bpms = np.arange(bpm_min, bpm_max + 1e-9, 0.05)
    total = np.zeros(len(bpms))
    for seg in segments:
        env = onset_envelope(seg, sr)
        if env.size < 4 * FPS or not np.any(env > 0):
            continue
        total += tempo_curve(env, bpms)
    if not np.any(total):
        return None
    return round(float(bpms[int(np.argmax(total))]), 2)
