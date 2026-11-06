"""
PURPOSE: Tonal Interval Vectors (TIV; Bernardes et al. 2016, JNMR 45:281) y la compatibilidad armónica
         "small-scale" de TIV.lib (Ramires et al. 2020), en numpy puro y vectorizados: un tema contra
         toda la colección y en las 12 transposiciones de una sola vez.
         Reimplementa lo que usó la tesis de Gabriel (github.com/gbibbo/harmonic_mix, tivlib/tiv.py):
           T(k) = w(k) * DFT(c)(k) / DFT(c)(0),  k = 1..6,  w = (3, 8, 11.5, 15, 14.5, 7.5)
           hc_tivlib(t, c) = ||t - c|| / (2 ||w||)  *  (1 - ||(t + c) / 2|| / ||w||)   (0 = lo más compatible)
         Transponer c(n) -> c(n - p) (p semitonos hacia arriba) multiplica T(k) por exp(-2*pi*j*k*p/12).
         Convención de cromas: índice 0 = C, 1 = C#, ..., 11 = B.
         Nota: el coeficiente k = 6 (y la energía, k = 0) de la DFT de 12 valores reales es real por
         definición; que su parte imaginaria sea siempre 0 no es un bug de TIV.lib (la tesis, 6.2, lo
         reportó como bug).
CHANGELOG:
  - 2026-11-06: Creación (rama harmonic-tiv, investigación de compatibilidad armónica).
"""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

WEIGHTS = np.array([3.0, 8.0, 11.5, 15.0, 14.5, 7.5])
W_NORM = float(np.linalg.norm(WEIGHTS))
_K = np.arange(1, 7)


def tiv_from_chroma(chroma: np.ndarray) -> np.ndarray:
    """TIV (complejo, último eje de largo 6) de cromas con el último eje de largo 12 (índice 0 = C).
    Croma de energía nula -> TIV nulo (igual que TIV.lib.from_pcp)."""
    c = np.asarray(chroma, dtype=np.float64)
    if c.shape[-1] != 12:
        raise ValueError(f"el croma debe tener 12 clases en el último eje, tiene {c.shape[-1]}")
    fft = np.fft.rfft(c, n=12, axis=-1)
    energy = fft[..., :1].real
    safe = np.where(energy > 0, energy, 1.0)
    vec = fft[..., 1:7] / safe * WEIGHTS
    return np.where(energy > 0, vec, 0.0 + 0.0j)


def transpose(tiv: np.ndarray, semitones) -> np.ndarray:
    """TIV del croma transpuesto `semitones` hacia arriba (np.roll(croma, semitones)). Acepta un entero
    o un arreglo de enteros que se agrega como eje nuevo antes del eje de los 6 coeficientes."""
    s = np.asarray(semitones)
    phase = np.exp(-2j * np.pi * np.multiply.outer(s, _K) / 12.0)       # (..., 6)
    if s.ndim == 0:
        return tiv * phase
    return tiv[..., None, :] * phase


def tonal_strength(tiv: np.ndarray) -> np.ndarray:
    """||T|| / ||w|| en [0, 1]: 0 = croma plano (sin centro tonal), 1 = una sola clase de altura."""
    return np.linalg.norm(tiv, axis=-1) / W_NORM


def hc_tivlib(t: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Small-scale compatibility de TIV.lib (0 = lo más compatible, 1 = lo menos). Broadcast en los
    ejes previos al de los 6 coeficientes."""
    relatedness = np.linalg.norm(t - c, axis=-1) / (2.0 * W_NORM)
    dissonance = 1.0 - np.linalg.norm((t + c) / 2.0, axis=-1) / W_NORM
    return relatedness * dissonance


def hc_distance(t: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Solo la parte de 'relatedness' de TIV.lib: distancia euclídea normalizada (0 = mismo TIV)."""
    return np.linalg.norm(t - c, axis=-1) / (2.0 * W_NORM)


def hc_mix_dissonance(t: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Solo la disonancia del TIV de la mezcla (pesos iguales): 1 - ||(t + c)/2|| / ||w||."""
    return 1.0 - np.linalg.norm((t + c) / 2.0, axis=-1) / W_NORM


def coherence_loss(t: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Cuánta claridad tonal se pierde al mezclar: (||t|| + ||c||)/2 - ||(t + c)/2||, sobre ||w||.
    Es >= 0 (desigualdad triangular); 0 si los dos TIV apuntan igual o si uno es nulo (un tema sin
    centro tonal es neutro, no 'compatible con todo' ni 'incompatible'); crece cuando ambos son tonales
    y chocan. Hipótesis H6/H7 del protocolo."""
    nt = np.linalg.norm(t, axis=-1)
    nc = np.linalg.norm(c, axis=-1)
    return ((nt + nc) / 2.0 - np.linalg.norm((t + c) / 2.0, axis=-1)) / W_NORM


METRICS = {"tivlib": hc_tivlib, "distance": hc_distance, "mix_dissonance": hc_mix_dissonance,
           "coherence_loss": coherence_loss}


def hc_matrix(targets: np.ndarray, candidates: np.ndarray, shifts: Iterable[int] = range(12),
              metric=hc_tivlib) -> np.ndarray:
    """Matriz (N, M, S) de `metric` entre N TIV objetivo y M candidatos transpuestos por cada shift."""
    t = np.asarray(targets)[:, None, None, :]                                    # (N, 1, 1, 6)
    c = transpose(np.asarray(candidates), np.asarray(list(shifts)))[None, ...]   # (1, M, S, 6)
    return metric(t, c)


def best_shift(t: np.ndarray, c: np.ndarray, max_shift: int = 6,
               shift_penalty: float = 0.0) -> Tuple[int, float]:
    """(semitonos a transponer el candidato, hc) que minimiza hc_tivlib + penalización·|s| en
    [-max_shift, max_shift] (con max_shift = 6 son 13 valores: -6 y +6 dan el mismo TIV). Ante empate
    gana la transposición más chica, como en src/v4/common/harmonic.py."""
    shifts = sorted(range(-max_shift, max_shift + 1), key=abs)
    vals = hc_tivlib(np.asarray(t)[None, :], transpose(np.asarray(c), np.asarray(shifts)))
    cost = vals + shift_penalty * np.abs(np.asarray(shifts))
    i = int(np.argmin(cost))
    return shifts[i], float(vals[i])


def tivlib_best_shift(t: np.ndarray, c: np.ndarray) -> Tuple[int, float]:
    """Exactamente lo que hacía TIV.lib.get_max_compatibility: argmin sobre los shifts 0..11 (gana el
    primero ante empate) y los mayores que 5 pasan a negativos (rango -6..5)."""
    vals = hc_tivlib(np.asarray(t)[None, :], transpose(np.asarray(c), np.arange(12)))
    i = int(np.argmin(vals))
    return (i - 12 if i > 5 else i), float(vals[i])


def thesis_percent(hc: np.ndarray) -> np.ndarray:
    """Porcentaje que mostraba la interfaz de la tesis: 100(1 - hc), reescalado de [70, 100] a [0, 100]
    (ecuaciones 4.1 y 4.2). Monótono: no cambia ningún ranking."""
    return (100.0 * (1.0 - np.asarray(hc)) - 70.0) * 100.0 / 30.0
