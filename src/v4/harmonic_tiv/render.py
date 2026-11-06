"""
PURPOSE: Arma transiciones de DJ cortas para la prueba de escucha a ciegas (etapa C del protocolo,
         docs/plans/armonia_tiv_protocolo.md). Las dos mezclas de un ítem comparten exactamente el
         mismo tramo del tema principal y la misma plantilla; solo cambia el candidato.
         Plantilla (en beats del tema principal):
           [0, lead)                principal solo
           [lead, lead + overlap)   los dos: el candidato entra en 4 beats sin graves; a mitad del
                                    solapamiento se cambian los graves (corte a 180 Hz, como el EQ del
                                    DJ); el principal sale en los últimos 4 beats
           [.., + tail)             candidato solo
         Tempo: el candidato se lleva al BPM del principal con rubberband (filtro de ffmpeg), sin
         cambiar la altura (key lock). Fase de beat: grilla de Traktor si existe; si no, la fase que
         maximiza la envolvente de ataques graves (bombo) en 60 s alrededor del tramo. Tramos: la ventana más tonal del caché (segunda
         mitad del principal, primera mitad del candidato), para que la armonía se oiga.
CHANGELOG:
  - 2026-11-06: Creación (rama harmonic-tiv).
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.v4.harmonic_tiv import frontend as fe
from src.v4.harmonic_tiv import tiv as tv

SR = 44100
# La fase estimada del audio cae 45 ms después de la grilla de Traktor en 20 de 25 temas medidos el
# 2026-11-06 (dispersión ±8 ms; los otros 5 difieren ~1/4 de beat). Se corrige para mezclar temas con y
# sin grilla en la misma convención.
AUDIO_TO_TRAKTOR_S = -0.045
KICK_CONF_MIN = 1.5      # confianza mínima de la fase de bombo para autoalinear dentro de la mezcla


@dataclass(frozen=True)
class Template:
    lead_beats: int = 8
    overlap_beats: int = 32
    tail_beats: int = 4
    fade_beats: int = 4
    bass_cut_hz: float = 180.0
    rms_dbfs: float = -18.0


def read_excerpt(path: Path, start_s: float, dur_s: float, sr: int = SR, tempo: float = 1.0) -> np.ndarray:
    """Tramo mono float32 desde `start_s` (segundos del archivo) de `dur_s` segundos *de salida*;
    con tempo != 1 se estira con rubberband sin cambiar la altura (lee dur_s·tempo del original)."""
    src_dur = dur_s * tempo + 0.5
    af = f"rubberband=tempo={tempo:.6f}:pitchq=quality" if abs(tempo - 1.0) > 1e-6 else "anull"
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{max(0.0, start_s):.4f}", "-t", f"{src_dur:.4f}",
           "-i", str(path), "-af", af, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.float32).copy()
    n = int(round(dur_s * sr))
    return np.pad(x, (0, max(0, n - len(x))))[:n]


def onset_envelope(x: np.ndarray, sr: int = SR, hop: int = 220, n_fft: int = 1024, max_hz: float = 130.0) -> np.ndarray:
    """Subida de la energía log por debajo de `max_hz` (bombo), con frames de 5 ms. Solo graves: con todo
    el espectro, el hi-hat a contratiempo gana y la fase cae medio beat corrida (medido el 2026-11-06)."""
    import torch
    spec = torch.stft(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), n_fft=n_fft, hop_length=hop,
                      window=torch.hann_window(n_fft), return_complex=True).abs().numpy()
    band = (spec[1: max(2, int(max_hz * n_fft / sr) + 1)] ** 2).sum(axis=0)
    loge = np.log1p(1e3 * band)
    return np.concatenate([[0.0], np.maximum(np.diff(loge), 0)])


def beat_phase_conf(x: np.ndarray, bpm: float, sr: int = SR, hop: int = 220):
    """(segundos del primer beat desde el inicio de x, confianza): fase que maximiza la envolvente del
    bombo sumada sobre todos los beats; confianza = máximo / media sobre las fases (1 = sin bombo)."""
    env = onset_envelope(x, sr, hop)
    fps = sr / hop
    period = 60.0 / bpm * fps
    n = int((len(env) - period) // period)
    if n < 2:
        return 0.0, 1.0
    phases = np.arange(0, period, 0.25)
    idx = np.clip(np.round(phases[:, None] + period * np.arange(n)[None, :]).astype(int), 0, len(env) - 1)
    sums = env[idx].sum(axis=1)
    return float(phases[int(np.argmax(sums))] / fps), float(sums.max() / max(sums.mean(), 1e-12))


def beat_phase(x: np.ndarray, bpm: float, sr: int = SR, hop: int = 220) -> float:
    return beat_phase_conf(x, bpm, sr, hop)[0]


def tonal_window(cache: Dict[str, np.ndarray], start_frac: float, end_frac: float, window_s: float,
                 block_s: float = 0.5, cfg_spec: str = "white_lin", preproc: str = "hpss") -> float:
    """Inicio (s) de la ventana de `window_s` con más claridad tonal (croma de agudos normalizado, por
    bloque, pesado por la energía) dentro de [start_frac, end_frac] de la duración del tema."""
    sem = cache[f"{preproc}__{cfg_spec}"].astype(np.float64)
    ch = fe.semitone_to_chroma(sem, fe.chroma_profile("treble"))
    tot = ch.sum(axis=1)
    strength = np.where(tot > 0, tv.tonal_strength(tv.tiv_from_chroma(ch / np.maximum(tot[:, None], 1e-12))), 0)
    rms = cache["orig__rms"].astype(np.float64)
    loud = np.clip(20 * np.log10(np.maximum(rms, 1e-9) / max(rms.max(), 1e-9)) + 30, 0, 30) / 30
    val = strength * loud
    nb, w = len(val), max(1, int(round(window_s / block_s)))
    a, b = int(nb * start_frac), max(int(nb * start_frac) + 1, int(nb * end_frac) - w)
    if b <= a:
        return max(0.0, a * block_s)
    csum = np.concatenate([[0.0], np.cumsum(val)])
    means = (csum[a + w:b + w] - csum[a:b]) / w
    return float((a + int(np.argmax(means))) * block_s)


def _butter(x: np.ndarray, cutoff: float, kind: str, sr: int = SR) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, cutoff, btype=kind, fs=sr, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def _rms_norm(x: np.ndarray, dbfs: float) -> np.ndarray:
    r = np.sqrt(np.mean(x.astype(np.float64) ** 2))
    return (x * (10 ** (dbfs / 20) / max(r, 1e-9))).astype(np.float32)


def mix_transition(target: np.ndarray, cand: np.ndarray, beat_s: float, t: Template, sr: int = SR) -> np.ndarray:
    """Mezcla según la plantilla. `target` y `cand` ya empiezan en un beat y están al mismo tempo:
    target dura lead + overlap beats; cand dura overlap + tail beats."""
    spb = beat_s * sr
    n_lead, n_ov, n_tail = int(round(t.lead_beats * spb)), int(round(t.overlap_beats * spb)), int(round(t.tail_beats * spb))
    n_fade = int(round(t.fade_beats * spb))
    total = n_lead + n_ov + n_tail
    target = _rms_norm(np.pad(target, (0, max(0, n_lead + n_ov - len(target))))[: n_lead + n_ov], t.rms_dbfs)
    cand = _rms_norm(np.pad(cand, (0, max(0, n_ov + n_tail - len(cand))))[: n_ov + n_tail], t.rms_dbfs)
    out = np.zeros(total, dtype=np.float32)
    # principal: graves enteros hasta la mitad del solapamiento, después sin graves; sale al final
    half = n_lead + n_ov // 2
    t_low, t_high = _butter(target, t.bass_cut_hz, "lowpass", sr), _butter(target, t.bass_cut_hz, "highpass", sr)
    g_t = np.ones(n_lead + n_ov, dtype=np.float32)
    g_t[n_lead + n_ov - n_fade:] = np.linspace(1, 0, n_fade, dtype=np.float32)
    low_on = np.zeros(n_lead + n_ov, dtype=np.float32)
    low_on[:half] = 1.0
    out[: n_lead + n_ov] += g_t * (t_high + low_on * t_low)
    # candidato: entra sin graves; graves desde la mitad del solapamiento
    c_low, c_high = _butter(cand, t.bass_cut_hz, "lowpass", sr), _butter(cand, t.bass_cut_hz, "highpass", sr)
    g_c = np.ones(n_ov + n_tail, dtype=np.float32)
    g_c[:n_fade] = np.linspace(0, 1, n_fade, dtype=np.float32)
    c_low_on = np.zeros(n_ov + n_tail, dtype=np.float32)
    c_low_on[n_ov // 2:] = 1.0
    out[n_lead:] += g_c * (c_high + c_low_on * c_low)
    return out


def write_mp3(x: np.ndarray, path: Path, sr: int = SR, bitrate: str = "192k") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
           "-c:a", "libmp3lame", "-b:a", bitrate, str(path)]
    subprocess.run(cmd, input=np.ascontiguousarray(x, dtype=np.float32).tobytes(), check=True, capture_output=True)


@dataclass
class TrackInfo:
    uid: str
    path: str
    bpm: float
    cache: Dict[str, np.ndarray]
    grid_start_s: Optional[float] = None     # primer beat de la grilla de Traktor (s)
    grid_bpm: Optional[float] = None


def snap_to_grid(t_s: float, info: TrackInfo) -> Optional[float]:
    if info.grid_start_s is None or not info.grid_bpm:
        return None
    beat = 60.0 / info.grid_bpm
    k = np.round((t_s - info.grid_start_s) / beat)
    return float(info.grid_start_s + k * beat)


def snap_by_audio(t_s: float, info: TrackInfo, bpm: float, context_s: float = 90.0) -> float:
    """Beat más cercano a t_s estimando la fase sobre `context_s` segundos alrededor (con bombo): la
    ventana más tonal suele ser un break sin bombo, donde la fase no se puede medir (2026-11-06)."""
    a = max(0.0, t_s - context_s / 2)
    probe = read_excerpt(Path(info.path), a, context_s)
    first = a + beat_phase(probe, bpm) + AUDIO_TO_TRAKTOR_S
    beat = 60.0 / bpm
    return float(first + np.round((t_s - first) / beat) * beat)


def render_pair(target: TrackInfo, cand_a: TrackInfo, cand_b: TrackInfo, out_a: Path, out_b: Path,
                t: Template = Template()) -> Dict:
    """Dos mezclas que comparten el tramo del principal. Devuelve los parámetros usados."""
    bpm_t = target.grid_bpm or target.bpm
    beat_s = 60.0 / bpm_t
    t_len = (t.lead_beats + t.overlap_beats) * beat_s
    t0 = tonal_window(target.cache, 0.40, 0.92, t_len)
    snapped = snap_to_grid(t0, target)
    if snapped is None:
        snapped = snap_by_audio(t0, target, bpm_t)
    tgt = read_excerpt(Path(target.path), snapped, t_len)
    mixes, params = [], {"target_start_s": round(snapped, 3), "bpm": round(bpm_t, 3)}
    for name, c, out in (("a", cand_a, out_a), ("b", cand_b, out_b)):
        bpm_c = c.grid_bpm or c.bpm
        # medio o doble tempo: se lleva al múltiplo más cercano
        ratio = min((bpm_t / (bpm_c * m) for m in (0.5, 1.0, 2.0)), key=lambda r: abs(np.log(r)))
        c_len = (t.overlap_beats + t.tail_beats) * beat_s
        c0 = tonal_window(c.cache, 0.08, 0.60, c_len * ratio)
        cs = snap_to_grid(c0, c)
        if cs is None:
            cs = snap_by_audio(c0, c, bpm_c)
        cx = read_excerpt(Path(c.path), cs, c_len + beat_s, tempo=ratio)
        # Autoalineación: si los dos tramos tienen bombo, se mide la fase de cada uno y se corre el
        # candidato para que sus beats caigan sobre los del principal (corrige grillas o fases malas).
        pt_, ct_ = beat_phase_conf(tgt, bpm_t)
        pc_, cc_ = beat_phase_conf(cx[: int(c_len * SR)], bpm_t)
        shift = 0.0
        if ct_ >= KICK_CONF_MIN and cc_ >= KICK_CONF_MIN:
            shift = ((pc_ - pt_ + beat_s / 2) % beat_s) - beat_s / 2
        k0 = int(round(max(0.0, shift) * SR))
        cx = cx[k0:] if shift >= 0 else np.concatenate([np.zeros(int(round(-shift * SR)), np.float32), cx])
        cx = cx[: int(round(c_len * SR))]
        mixes.append(mix_transition(tgt, cx, beat_s, t))
        params[f"cand_{name}_align_ms"] = round(shift * 1000, 1)
        params[f"cand_{name}_kick_conf"] = round(min(ct_, cc_), 2)
        params[f"cand_{name}_start_s"] = round(cs, 3)
        params[f"cand_{name}_tempo"] = round(ratio, 5)
    peak = max(float(np.abs(m).max()) for m in mixes)
    gain = 10 ** (-1.0 / 20) / max(peak, 1e-9)            # misma ganancia para las dos: -1 dBFS de pico
    for m, out in zip(mixes, (out_a, out_b)):
        write_mp3(m * gain, out)
    params["gain_db"] = round(20 * np.log10(gain), 2)
    return params
