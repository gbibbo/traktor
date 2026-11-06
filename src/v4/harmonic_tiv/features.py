"""
PURPOSE: Del caché por tema (tools/harmonic_tiv/extract_semitones.py) a un TIV por tema según una
         configuración explícita, para comparar variantes de a una variable por vez:
           preproc   'orig' | 'hpss'                       (separación armónico/percusivo)
           spec      'raw_lin' | 'white_lin' | 'white_nnls' (blanqueo y NNLS del espectro)
           profile   'treble' | 'full' | 'bass' | 'mid'     (qué registro entra al croma)
           section   'whole' | 'center30' | 'center50' | 'intro25' | 'outro25' | 'core'
           weighting 'energy' (media cruda: la tesis) | 'norm' (cada bloque pesa igual) |
                     'salience' (cada bloque normalizado, pesado por su claridad tonal)
         FEM#10 de la tesis = hpss / white_lin / treble / whole / energy (el código publicado usaba
         center30). Los perfiles y el NNLS son aproximaciones de NNLS-Chroma (ver frontend.py).
CHANGELOG:
  - 2026-11-06: Creación (rama harmonic-tiv).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

from src.v4.harmonic_tiv import frontend as fe
from src.v4.harmonic_tiv import tiv as tv


@dataclass(frozen=True)
class TivConfig:
    preproc: str = "hpss"
    spec: str = "white_lin"
    profile: str = "treble"
    section: str = "whole"
    weighting: str = "energy"
    silence_db: float = -45.0     # bloques más de 45 dB bajo el pico del tema no cuentan (salvo 'energy')

    def name(self) -> str:
        return f"{self.preproc}/{self.spec}/{self.profile}/{self.section}/{self.weighting}"

    def as_dict(self) -> Dict:
        return asdict(self)

    def with_(self, **kw) -> "TivConfig":
        return replace(self, **kw)


THESIS_FEM10 = TivConfig()
THESIS_CODE = TivConfig(section="center30")


def load_cache(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def section_mask(n_blocks: int, section: str, rms: Optional[np.ndarray] = None, block_s: float = 0.5) -> np.ndarray:
    """Bloques de la sección: 'whole', 'centerNN' / 'introNN' / 'outroNN' (porcentaje de la duración),
    'firstNs' / 'lastNs' (segundos, p. ej. 'last120s' = los últimos 2 minutos) o 'core' (por energía)."""
    m = np.zeros(n_blocks, dtype=bool)
    timed = re.fullmatch(r"(first|last)(\d+)s", section)
    if timed:
        nb = max(1, min(n_blocks, int(round(int(timed.group(2)) / block_s))))
        if timed.group(1) == "first":
            m[:nb] = True
        else:
            m[n_blocks - nb:] = True
        return m
    if section == "whole":
        m[:] = True
    elif section.startswith("center"):
        frac = int(section[6:]) / 100.0
        a = int(round(n_blocks * (0.5 - frac / 2)))
        b = int(round(n_blocks * (0.5 + frac / 2)))
        m[a:max(b, a + 1)] = True
    elif section.startswith("intro"):
        m[: max(1, int(round(n_blocks * int(section[5:]) / 100.0)))] = True
    elif section.startswith("outro"):
        m[n_blocks - max(1, int(round(n_blocks * int(section[5:]) / 100.0))):] = True
    elif section == "core":
        # 60 % central por energía: descarta intro y outro de baja energía
        if rms is None:
            raise ValueError("'core' necesita rms")
        e = np.convolve(rms.astype(np.float64) ** 2, np.ones(16) / 16, mode="same")
        thr = np.quantile(e, 0.4)
        m = e >= thr
    else:
        raise ValueError(f"sección desconocida: {section}")
    return m


def block_chroma(cache: Dict[str, np.ndarray], cfg: TivConfig) -> np.ndarray:
    """Croma por bloque (B, 12) del tema según cfg (sin agregar)."""
    sem = cache[f"{cfg.preproc}__{cfg.spec}"].astype(np.float64)
    return fe.semitone_to_chroma(sem, fe.chroma_profile(cfg.profile))


def aggregate_chroma(cache: Dict[str, np.ndarray], cfg: TivConfig, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Croma (12,) agregado de los bloques de la sección (o de `mask`, si se pasa)."""
    ch = block_chroma(cache, cfg)
    counts = cache[f"{cfg.preproc}__counts"].astype(np.float64)
    rms = cache["orig__rms"].astype(np.float64)
    if mask is None:
        mask = section_mask(len(ch), cfg.section, rms)
    if cfg.weighting == "energy":
        w = counts * mask
        return (ch * w[:, None]).sum(axis=0) / max(w.sum(), 1e-12)
    peak = max(rms.max(), 1e-12)
    loud = 20 * np.log10(np.maximum(rms, 1e-12) / peak) > cfg.silence_db
    tot = ch.sum(axis=1)
    valid = mask & loud & (tot > 0)
    if not valid.any():
        valid = mask & (tot > 0)
    if not valid.any():
        return np.zeros(12)
    norm = ch[valid] / tot[valid, None]
    if cfg.weighting == "norm":
        w = counts[valid]
    elif cfg.weighting == "salience":
        w = counts[valid] * tv.tonal_strength(tv.tiv_from_chroma(norm))
    else:
        raise ValueError(f"ponderación desconocida: {cfg.weighting}")
    return (norm * w[:, None]).sum(axis=0) / max(w.sum(), 1e-12)


def track_tiv(cache: Dict[str, np.ndarray], cfg: TivConfig) -> np.ndarray:
    return tv.tiv_from_chroma(aggregate_chroma(cache, cfg))


def fragment_tivs(cache: Dict[str, np.ndarray], cfg: TivConfig, frag_s: float = 30.0, block_s: float = 0.5,
                  min_loud: float = 0.5) -> np.ndarray:
    """TIV de fragmentos consecutivos sin solapar de `frag_s` segundos, con la misma configuración que
    el tema (salvo la sección). Un fragmento con menos de `min_loud` de bloques sonando queda como NaN
    (se descarta en el objetivo autosupervisado). Devuelve (F, 6) complejo."""
    rms = cache["orig__rms"].astype(np.float64)
    n = len(rms)
    w = max(1, int(round(frag_s / block_s)))
    loud = 20 * np.log10(np.maximum(rms, 1e-12) / max(rms.max(), 1e-12)) > cfg.silence_db
    out = []
    for a in range(0, n - w + 1, w):
        mask = np.zeros(n, dtype=bool)
        mask[a:a + w] = True
        if loud[a:a + w].mean() < min_loud:
            out.append(np.full(6, np.nan + 0j))
            continue
        out.append(tv.tiv_from_chroma(aggregate_chroma(cache, cfg, mask=mask)))
    return np.array(out).reshape(-1, 6)


def track_features(cache: Dict[str, np.ndarray], cfg: TivConfig, frag_s: float = 30.0,
                   mix_section_s: int = 120) -> Dict[str, np.ndarray]:
    """Todo lo que la evaluación usa de un tema en una sola pasada por su caché: TIV de la sección de
    cfg, de los primeros y los últimos `mix_section_s` segundos, y de los fragmentos consecutivos."""
    return {"tiv": track_tiv(cache, cfg),
            "first": track_tiv(cache, cfg.with_(section=f"first{mix_section_s}s")),
            "last": track_tiv(cache, cfg.with_(section=f"last{mix_section_s}s")),
            "frags": fragment_tivs(cache, cfg, frag_s)}


def collection_tivs(cache_dir: Path, ids: Iterable[str], cfg: TivConfig) -> Dict[str, np.ndarray]:
    """TIV de cada id con caché; los que no lo tienen quedan fuera (el llamador lo informa)."""
    out = {}
    for i in ids:
        p = Path(cache_dir) / f"{i}.npz"
        if p.exists():
            out[i] = track_tiv(load_cache(p), cfg)
    return out


# --------------------------------------------------------------------------- tonalidad desde el croma

_KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_TEMP_MAJOR = np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0])
_TEMP_MINOR = np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0])
PROFILES = {"krumhansl": (_KK_MAJOR, _KK_MINOR), "temperley": (_TEMP_MAJOR, _TEMP_MINOR)}
_MINOR_CAMELOT = [5, 12, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10]
_MAJOR_CAMELOT = [8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6, 1]


def key_from_chroma(chroma: np.ndarray, profile: str = "temperley", minor_bias: float = 0.0) -> str:
    """Tonalidad (Camelot) por correlación con perfiles rotados. `minor_bias` suma a las menores."""
    major, minor = PROFILES[profile]
    c = np.asarray(chroma, dtype=np.float64)
    if c.sum() <= 0:
        return "?"
    best, best_r = "?", -np.inf
    for tonic in range(12):
        for mode, prof in (("B", major), ("A", minor)):
            r = np.corrcoef(c, np.roll(prof, tonic))[0, 1] + (minor_bias if mode == "A" else 0.0)
            if r > best_r:
                best_r = r
                best = f"{(_MINOR_CAMELOT if mode == 'A' else _MAJOR_CAMELOT)[tonic]}{mode}"
    return best
