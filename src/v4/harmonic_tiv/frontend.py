"""
PURPOSE: Del audio al espectro por semitono que alimenta los TIV, en CPU y sin Essentia (corre en
         Windows y macOS). Sigue los pasos del FEM#10 de la tesis de Gabriel (HPSS de librosa con
         kernel (13, 31) -> NNLSChroma de Essentia con useNNLS=False -> media de los cromas) pero deja
         cada paso como parámetro, y guarda un caché por tema para probar variantes sin re-analizar:
           1. Decodificación con ffmpeg a mono (22050 Hz por defecto: misma resolución en Hz y en
              segundos que la tesis a 44100 con ventanas del doble de muestras).
           2. HPSS opcional (Fitzgerald 2010, máscara suave como librosa.decompose.hpss).
           3. STFT larga (8192 a 22050 Hz = 2,7 Hz por bin) -> espectro log-frecuencia con 3 bins por
              semitono, de A0 (27,5 Hz) a G#7: 84 semitonos (como LogSpectrum de NNLS-Chroma).
           4. Afinación global (fase del período de 3 bins, Mauch & Dixon 2010) y corrección.
           5. Blanqueo espectral opcional (resta la media móvil de una octava y divide por el desvío).
           6. Espectro por semitono: bin central afinado (useNNLS=False) o NNLS aproximado por
              actualizaciones multiplicativas con plantillas armónicas (spectralShape 0,7).
           7. Promedio en bloques de `block_s` segundos (el caché), con la energía RMS de cada bloque.
         Los perfiles de croma (agudos sin el bajo, bajo, completo) se aplican después sobre el caché.
         Aproximación documentada: no es bit a bit Essentia; las ventanas de los perfiles imitan las de
         NNLS-Chroma pero no son sus tablas.
CHANGELOG:
  - 2026-11-06: Creación (rama harmonic-tiv).
"""
from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

A0_HZ = 27.5
N_SEMI = 84               # A0 .. G#7
BPS = 3                   # bins por semitono del espectro log-frecuencia
N_LOG = N_SEMI * BPS      # 252 bins; el central de cada semitono es el índice 3*s + 1


@dataclass(frozen=True)
class FrontendConfig:
    sr: int = 22050
    n_fft: int = 8192
    hop: int = 1024
    hpss_n_fft: int = 1024        # equivale a 2048 a 44100 (default de librosa)
    hpss_hop: int = 256
    hpss_kernel_time: int = 13    # mediana a lo largo del tiempo (componente armónica)
    hpss_kernel_freq: int = 31    # mediana a lo largo de la frecuencia (componente percusiva)
    hpss_power: float = 2.0
    whitening: float = 1.0        # exponente del desvío (1.0 = default de NNLSChroma)
    whiten_window_bins: int = 37  # una octava de bins log (impar)
    nnls_shape: float = 0.7
    nnls_harmonics: int = 20
    nnls_iters: int = 150
    block_s: float = 0.5
    chunk_frames: int = 512

    def as_dict(self) -> Dict:
        return asdict(self)


# --------------------------------------------------------------------------- audio

def load_mono(path: Path, sr: int) -> np.ndarray:
    """Audio mono float32 a `sr` con ffmpeg (en PATH). Lee MP3/FLAC/WAV/M4A iguales en Windows y Mac."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def _stft(x: np.ndarray, n_fft: int, hop: int):
    import torch
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    window = torch.hann_window(n_fft)
    return torch.stft(xt, n_fft=n_fft, hop_length=hop, window=window, center=True, return_complex=True), window


def hpss_harmonic(x: np.ndarray, cfg: FrontendConfig, segment_s: float = 60.0, pad_s: float = 2.0) -> np.ndarray:
    """Componente armónica por HPSS con máscara suave (como librosa.decompose.hpss + istft), por tramos
    de `segment_s` con `pad_s` de margen a cada lado: la memoria queda acotada (el tema entero de una
    vez pedía ~1,8 GB por proceso en temas de 15 min) y el resultado es el mismo, porque la mediana
    temporal abarca 13 frames (0,15 s) y el margen se descarta."""
    # Tramos y márgenes en múltiplos del salto: así los frames coinciden con los del tema entero.
    hop = cfg.hpss_hop
    seg = max(1, int(segment_s * cfg.sr) // hop) * hop
    pad = max(1, int(pad_s * cfg.sr) // hop) * hop
    if len(x) <= seg + 2 * pad:
        return _hpss_block(x, cfg)
    out = np.empty(len(x), dtype=np.float32)
    for a in range(0, len(x), seg):
        b = min(len(x), a + seg)
        lo, hi = max(0, a - pad), min(len(x), b + pad)
        y = _hpss_block(x[lo:hi], cfg)
        out[a:b] = y[a - lo: a - lo + (b - a)]
    return out


def _hpss_block(x: np.ndarray, cfg: FrontendConfig) -> np.ndarray:
    import torch
    from scipy.ndimage import median_filter
    spec, window = _stft(x, cfg.hpss_n_fft, cfg.hpss_hop)
    mag = spec.abs().numpy()
    harm = median_filter(mag, size=(1, cfg.hpss_kernel_time), mode="reflect")
    perc = median_filter(mag, size=(cfg.hpss_kernel_freq, 1), mode="reflect")
    p = cfg.hpss_power
    hp, pp = harm ** p, perc ** p
    mask = hp / np.maximum(hp + pp, np.finfo(np.float32).tiny)
    y = torch.istft(spec * torch.from_numpy(mask.astype(np.float32)), n_fft=cfg.hpss_n_fft,
                    hop_length=cfg.hpss_hop, window=window, length=len(x))
    return y.numpy().astype(np.float32)


# --------------------------------------------------------------------------- log-frecuencia

def log_bin_freqs() -> np.ndarray:
    """Frecuencia central de los N_LOG bins: el bin 3*s + 1 cae exacto en el semitono s (A0 = 0)."""
    b = np.arange(N_LOG)
    return A0_HZ * 2.0 ** ((b - 1) / (12.0 * BPS))


def logfreq_matrix(sr: int, n_fft: int) -> np.ndarray:
    """Matriz (N_LOG, n_fft/2+1) que lleva magnitudes lineales a log-frecuencia: núcleo triangular en
    frecuencia, de ancho el mayor entre la separación de bins log y la de bins FFT (interpola abajo,
    promedia arriba)."""
    f_lin = np.arange(n_fft // 2 + 1) * sr / n_fft
    df = sr / n_fft
    fc = log_bin_freqs()
    width = np.maximum(fc * (2.0 ** (1.0 / (12 * BPS)) - 1.0), df)
    m = np.clip(1.0 - np.abs(f_lin[None, :] - fc[:, None]) / width[:, None], 0.0, None)
    m /= np.maximum(m.sum(axis=1, keepdims=True), 1e-12)
    return m.astype(np.float32)


def logfreq_spectrogram(x: np.ndarray, cfg: FrontendConfig, matrix: Optional[np.ndarray] = None) -> np.ndarray:
    """(N_LOG, frames) por tramos de `chunk_frames` (memoria acotada aunque el tema dure 15 minutos)."""
    import torch
    if matrix is None:
        matrix = logfreq_matrix(cfg.sr, cfg.n_fft)
    mt = torch.from_numpy(matrix)
    window = torch.hann_window(cfg.n_fft)
    pad = cfg.n_fft // 2
    xp = np.pad(x.astype(np.float32), (pad, pad))
    n_frames = 1 + max(0, (len(xp) - cfg.n_fft) // cfg.hop)
    out = np.empty((N_LOG, n_frames), dtype=np.float32)
    span = cfg.chunk_frames
    for f0 in range(0, n_frames, span):
        f1 = min(n_frames, f0 + span)
        seg = torch.from_numpy(xp[f0 * cfg.hop: (f1 - 1) * cfg.hop + cfg.n_fft])
        spec = torch.stft(seg, n_fft=cfg.n_fft, hop_length=cfg.hop, window=window, center=False,
                          return_complex=True).abs()
        out[:, f0:f1] = (mt @ spec[:, : f1 - f0]).numpy()
    return out


def estimate_tuning(logspec: np.ndarray) -> float:
    """Desafinación global en semitonos, en [-0.5, 0.5): fase de la componente de período 3 bins del
    espectro log sumado en el tiempo (Mauch & Dixon 2010). +0.2 = el tema está 20 cents arriba."""
    s = logspec.sum(axis=1)
    b = np.arange(N_LOG)
    z = np.sum(s * np.exp(-2j * np.pi * (b - 1) / BPS))
    # Un tono desafinado d semitonos cae en el bin 3s + 1 + 3d: fase -2*pi*d.
    return float(-np.angle(z) / (2 * np.pi)) if np.abs(z) > 0 else 0.0


def shift_bins(logspec: np.ndarray, shift: float) -> np.ndarray:
    """Corre el espectro log `shift` bins hacia abajo (interpolación lineal): un tema afinado +1 bin
    queda con sus notas en los bins centrales."""
    if abs(shift) < 1e-9:
        return logspec
    b = np.arange(N_LOG, dtype=np.float64)
    src = b + shift
    i0 = np.clip(np.floor(src).astype(int), 0, N_LOG - 1)
    i1 = np.clip(i0 + 1, 0, N_LOG - 1)
    w = (src - np.floor(src))[:, None]
    out = (1 - w) * logspec[i0] + w * logspec[i1]
    out[(src < 0) | (src > N_LOG - 1)] = 0.0
    return out.astype(np.float32)


def whiten(logspec: np.ndarray, window_bins: int, power: float) -> np.ndarray:
    """Resta la media móvil (una octava) y divide por el desvío móvil^power; lo negativo queda en 0."""
    if power <= 0:
        return logspec
    from scipy.ndimage import uniform_filter1d
    mean = uniform_filter1d(logspec, size=window_bins, axis=0, mode="nearest")
    sq = uniform_filter1d(logspec ** 2, size=window_bins, axis=0, mode="nearest")
    std = np.sqrt(np.maximum(sq - mean ** 2, 0.0))
    # Piso relativo al pico del frame: sin él, las zonas casi vacías (desvío ~0) explotan.
    floor = 0.01 * logspec.max(axis=0, keepdims=True) + 1e-9
    out = (logspec - mean) / np.maximum(std, floor) ** power
    out[logspec <= mean] = 0.0
    return out.astype(np.float32)


def semitone_linear(logspec: np.ndarray) -> np.ndarray:
    """Espectro por semitono sin NNLS: el bin central (ya afinado) de cada semitono (84, frames)."""
    return logspec[1::BPS, :]


def nnls_dictionary(shape: float, n_harm: int) -> np.ndarray:
    """Plantillas (N_LOG, N_SEMI): armónicos h = 1..n_harm de cada nota con amplitud shape^(h-1)."""
    d = np.zeros((N_LOG, N_SEMI), dtype=np.float64)
    b = np.arange(N_LOG)
    for s in range(N_SEMI):
        for h in range(1, n_harm + 1):
            pos = BPS * s + 1 + 12 * BPS * np.log2(h)
            if pos > N_LOG:
                break
            d[:, s] += shape ** (h - 1) * np.clip(1.0 - np.abs(b - pos), 0.0, None)
    d /= np.maximum(np.linalg.norm(d, axis=0, keepdims=True), 1e-12)
    return d.astype(np.float32)


def semitone_nnls(logspec: np.ndarray, dictionary: np.ndarray, iters: int) -> np.ndarray:
    """NNLS aproximado para todos los frames a la vez (actualizaciones multiplicativas de Lee & Seung
    con diccionario fijo; convergen a la solución de mínimos cuadrados no negativos)."""
    d = dictionary.astype(np.float64)
    y = np.maximum(logspec.astype(np.float64), 0.0)
    dty = d.T @ y
    dtd = d.T @ d
    x = np.maximum(dty, 1e-9)
    for _ in range(iters):
        x *= dty / np.maximum(dtd @ x, 1e-12)
    return x.astype(np.float32)


# --------------------------------------------------------------------------- caché por tema

def block_pool(frames: np.ndarray, hop_s: float, block_s: float):
    """Promedio por bloques de `block_s` s a lo largo del último eje; devuelve (bloques, n_frames)."""
    per = max(1, int(round(block_s / hop_s)))
    n = frames.shape[-1]
    nb = int(np.ceil(n / per))
    pad = nb * per - n
    padded = np.pad(frames, [(0, 0)] * (frames.ndim - 1) + [(0, pad)])
    sums = padded.reshape(frames.shape[:-1] + (nb, per)).sum(axis=-1)
    counts = np.full(nb, per)
    counts[-1] = per - pad
    return sums / counts, counts


def frame_rms(x: np.ndarray, cfg: FrontendConfig, n_frames: int) -> np.ndarray:
    """RMS de la señal en las mismas ventanas (centradas) que la STFT larga."""
    pad = cfg.n_fft // 2
    xp = np.pad(x.astype(np.float64), (pad, pad))
    csum = np.concatenate([[0.0], np.cumsum(xp * xp)])
    starts = np.arange(n_frames) * cfg.hop
    ends = np.minimum(starts + cfg.n_fft, len(xp))
    return np.sqrt(np.maximum(csum[ends] - csum[starts], 0.0) / np.maximum(ends - starts, 1)).astype(np.float32)


def analyze_signal(x: np.ndarray, cfg: FrontendConfig, variants=("raw_lin", "white_lin", "white_nnls"),
                   matrix: Optional[np.ndarray] = None, dictionary: Optional[np.ndarray] = None) -> Dict:
    """Espectros por semitono en bloques para una señal ya decodificada. Devuelve un dict con
    'tuning' (semitonos), 'counts' (frames por bloque), 'rms' (por bloque) y cada variante (84, B)."""
    logspec = logfreq_spectrogram(x, cfg, matrix)
    tuning = estimate_tuning(logspec)
    tuned = shift_bins(logspec, tuning * BPS)
    out: Dict = {"tuning": tuning}
    hop_s = cfg.hop / cfg.sr
    rms = frame_rms(x, cfg, logspec.shape[1])
    out["rms"], out["counts"] = block_pool(rms[None, :] ** 2, hop_s, cfg.block_s)
    out["rms"] = np.sqrt(out["rms"][0]).astype(np.float32)
    white = None
    for v in variants:
        if v == "raw_lin":
            sem = semitone_linear(tuned)
        elif v in ("white_lin", "white_nnls"):
            if white is None:
                white = whiten(tuned, cfg.whiten_window_bins, cfg.whitening)
            if v == "white_lin":
                sem = semitone_linear(white)
            else:
                if dictionary is None:
                    dictionary = nnls_dictionary(cfg.nnls_shape, cfg.nnls_harmonics)
                sem = semitone_nnls(white, dictionary, cfg.nnls_iters)
        else:
            raise ValueError(f"variante desconocida: {v}")
        out[v], _ = block_pool(sem, hop_s, cfg.block_s)
        out[v] = out[v].astype(np.float16)
    return out


# --------------------------------------------------------------------------- cromas desde el caché

def _cos_ramp(n: int, a: float, b: float) -> np.ndarray:
    """0 antes de a, 1 después de b, rampa coseno en el medio (índices de semitono)."""
    s = np.arange(n, dtype=np.float64)
    t = np.clip((s - a) / max(b - a, 1e-9), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * t)


def chroma_profile(name: str) -> np.ndarray:
    """Pesos por semitono (84,) para sumar a croma. 'treble' imita el croma de NNLS-Chroma (sin el
    registro grave), 'bass' su croma de bajo, 'full' todo el rango, 'mid' sin graves ni agudos
    extremos. Semitono 0 = A0 (27,5 Hz), 12 = A1 (55 Hz), 24 = A2 (110 Hz), 36 = A3 (220 Hz)."""
    if name == "treble":
        return _cos_ramp(N_SEMI, 14, 26) * (1 - _cos_ramp(N_SEMI, 70, 84))
    if name == "bass":
        s = np.arange(N_SEMI)
        return np.where(s <= 38, np.sin(np.pi * s / 38.0) ** 2, 0.0)
    if name == "full":
        return np.ones(N_SEMI)
    if name == "mid":
        return _cos_ramp(N_SEMI, 22, 30) * (1 - _cos_ramp(N_SEMI, 62, 74))
    raise ValueError(f"perfil desconocido: {name}")


def semitone_to_chroma(sem: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """(84, B) -> (B, 12) con índice 0 = C (el semitono 0 es A, clase 9)."""
    w = np.asarray(sem, dtype=np.float64) * profile[:, None]
    chroma = np.zeros((12, w.shape[1]))
    for s in range(N_SEMI):
        chroma[(s + 9) % 12] += w[s]
    return chroma.T
