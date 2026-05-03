"""
PURPOSE: scipy/numpy HPSS (Harmonic-Percussive Source Separation) to extract the
         percussive component of an audio signal. No librosa dependency.

CHANGELOG:
  D2.2 - Initial implementation using scipy.ndimage.median_filter and scipy.fft.
"""

from __future__ import annotations

import importlib.util
from typing import Optional

import math

import numpy as np

# Guard: scipy is required
_SCIPY = importlib.util.find_spec("scipy") is not None


def hpss_available() -> bool:
    """Return True if the scipy/numpy HPSS backend can be used."""
    return _SCIPY


def hpss_percussive(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    harmonic_filter_size: int = 17,
    percussive_filter_size: int = 17,
    power: float = 2.0,
    margin: float = 1.0,
) -> np.ndarray:
    """
    Extract the percussive component of a mono audio signal via median-filter HPSS.

    Parameters
    ----------
    y : np.ndarray, shape (N,)
        Mono float32/float64 audio.
    sr : int
        Sample rate (informational; not used in computation).
    n_fft : int
        FFT size.
    hop_length : int
        Hop between successive frames.
    win_length : int or None
        Window length; defaults to n_fft.
    harmonic_filter_size : int
        Median filter length along the time axis (harmonic emphasis).
    percussive_filter_size : int
        Median filter length along the frequency axis (percussive emphasis).
    power : float
        Wiener exponent.
    margin : float
        Separation margin (>1 widens the gap between H and P masks).

    Returns
    -------
    np.ndarray, shape (N,)
        Percussive component, same length as input.

    Raises
    ------
    ImportError
        If scipy is not installed.
    RuntimeError
        If the input has fewer samples than n_fft.
    """
    if not _SCIPY:
        raise ImportError("scipy is required for HPSS but is not installed")

    from scipy.fft import rfft, irfft
    from scipy.ndimage import median_filter
    from scipy.signal.windows import hann

    if win_length is None:
        win_length = n_fft

    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 1:
        raise ValueError(f"Expected mono audio (1-D), got shape {y.shape}")
    if len(y) < n_fft:
        raise RuntimeError(
            f"Audio too short for HPSS: {len(y)} samples, need at least {n_fft}"
        )

    window = hann(win_length, sym=False).astype(np.float32)

    # Use ceil to ensure the last few samples are covered by at least one frame
    n_frames = 1 + math.ceil(max(0, len(y) - n_fft) / hop_length)
    pad_length = (n_frames - 1) * hop_length + n_fft
    if pad_length > len(y):
        y_padded = np.pad(y, (0, pad_length - len(y)))
    else:
        y_padded = y

    # Build frames: shape (n_fft, n_frames)
    frames = np.lib.stride_tricks.as_strided(
        y_padded,
        shape=(n_fft, n_frames),
        strides=(y_padded.strides[0], y_padded.strides[0] * hop_length),
    ).copy()  # copy to avoid negative strides on some platforms

    # Apply window to each frame
    windowed = frames[:win_length] * window[:, np.newaxis]

    # STFT: shape (n_fft//2 + 1, n_frames)
    stft = rfft(windowed, n=n_fft, axis=0)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # Median filtering
    # Harmonic: filter along time axis (axis=1)
    harmonic = median_filter(magnitude, size=(1, harmonic_filter_size))
    # Percussive: filter along frequency axis (axis=0)
    percussive = median_filter(magnitude, size=(percussive_filter_size, 1))

    # Wiener masks
    eps = np.finfo(np.float32).eps
    harmonic_mask = (harmonic * margin) ** power / (
        (harmonic * margin) ** power + percussive ** power + eps
    )
    percussive_mask = (percussive * margin) ** power / (
        (percussive * margin) ** power + harmonic ** power + eps
    )
    # Masks sum to ≤1 when margin=1; > 1 possible when margin > 1 (soft gain)
    _ = harmonic_mask  # computed but only percussive_mask is used

    # Apply percussive mask to STFT
    perc_stft = (magnitude * percussive_mask) * np.exp(1j * phase)

    # iSTFT via overlap-add
    n_bins = stft.shape[0]
    output_length = (n_frames - 1) * hop_length + n_fft
    y_perc = np.zeros(output_length, dtype=np.float32)
    norm = np.zeros(output_length, dtype=np.float32)

    # Reconstruct each frame
    reconstructed = irfft(perc_stft, n=n_fft, axis=0)  # shape (n_fft, n_frames)
    windowed_out = reconstructed[:win_length] * window[:, np.newaxis]

    for i in range(n_frames):
        start = i * hop_length
        y_perc[start : start + win_length] += windowed_out[:, i]
        norm[start : start + win_length] += window ** 2

    # Normalize overlap-add
    norm = np.where(norm > eps, norm, 1.0)
    y_perc /= norm

    # Trim to original length
    return y_perc[: len(y)]
