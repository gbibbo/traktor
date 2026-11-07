"""
PURPOSE: Beat phase of a track against one constant grid, window by window (diagnostic, not pipeline).

A track made at a fixed tempo keeps its kicks at the same phase of a constant grid all along. If the
recording lost audio, the phase jumps (beats come earlier); if silence or repeated audio was inserted, the
phase jumps the other way. A steady ramp means the grid period is wrong (fitted out here).

Usage: python beat_phase.py AUDIO [--start S] [--end S] [--bpm-min 110 --bpm-max 160] [--csv OUT]
CHANGELOG:
- 2026-11-07: First version, to audit the Spotify recordings (download_JIJIJI) that Traktor gridded badly.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt

SR = 22050
HOP = 32          # 1.45 ms per envelope frame


def decode(path: str, start: float | None, end: float | None) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.4f}"]
    if end is not None:
        cmd += ["-to", f"{end:.4f}"] if start is None else ["-t", f"{end - start:.4f}"]
    cmd += ["-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<f4").astype(np.float64)


def kick_envelope(x: np.ndarray) -> np.ndarray:
    """Onset strength of the kick: 35-150 Hz energy, smoothed to ~25 Hz, positive log slope."""
    sos = butter(4, [35, 150], btype="band", fs=SR, output="sos")
    power = sosfiltfilt(sos, x) ** 2
    smooth = sosfiltfilt(butter(2, 25, btype="low", fs=SR, output="sos"), power)
    n = len(smooth) // HOP
    energy = np.maximum(smooth[: n * HOP].reshape(n, HOP).mean(axis=1), 1e-10)
    log_e = np.log(energy)
    flux = np.maximum(0.0, np.diff(log_e, prepend=log_e[0]))
    return flux


def best_period(env: np.ndarray, fps: float, bpm_min: float, bpm_max: float) -> float:
    """BPM whose comb (a constant grid over the whole track) catches the most onset energy."""
    t = np.arange(len(env)) / fps
    w = env - env.mean()

    def strength(bpm_grid: np.ndarray) -> np.ndarray:
        out = np.empty(len(bpm_grid))
        for i, b in enumerate(bpm_grid):
            out[i] = abs(np.sum(w * np.exp(-2j * np.pi * t * b / 60.0)))
        return out

    coarse = np.arange(bpm_min, bpm_max, 0.05)
    s = strength(coarse)
    b0 = coarse[int(np.argmax(s))]
    fine = np.arange(b0 - 0.1, b0 + 0.1, 0.0005)
    return float(fine[int(np.argmax(strength(fine)))])


def phase_track(env: np.ndarray, fps: float, bpm: float, win_s: float, hop_s: float):
    period = 60.0 / bpm
    t = np.arange(len(env)) / fps
    w = env - env.mean()
    z_all = w * np.exp(-2j * np.pi * t / period)
    win, hop = int(win_s * fps), int(hop_s * fps)
    rows = []
    for a in range(0, len(env) - win + 1, hop):
        z = z_all[a:a + win].sum()
        mag = abs((env[a:a + win] * np.exp(-2j * np.pi * t[a:a + win] / period)).sum()) / (env[a:a + win].sum() + 1e-12)
        # phase -> offset in ms of the beats relative to the grid (positive = beats later)
        offset_ms = -np.angle(z) / (2 * np.pi) * period * 1000.0
        rows.append(((a + win / 2) / fps, offset_ms, mag))
    return np.array(rows), period


def unwrap_ms(offsets: np.ndarray, period_ms: float) -> np.ndarray:
    return np.unwrap(offsets / period_ms * 2 * np.pi) / (2 * np.pi) * period_ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--bpm-min", type=float, default=110.0)
    ap.add_argument("--bpm-max", type=float, default=160.0)
    ap.add_argument("--bpm", type=float, default=None, help="Grid BPM to use instead of the best fit")
    ap.add_argument("--win", type=float, default=6.0)
    ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--min-strength", type=float, default=0.25)
    ap.add_argument("--jump-ms", type=float, default=8.0)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    x = decode(args.audio, args.start, args.end)
    fps = SR / HOP
    env = kick_envelope(x)
    bpm = args.bpm or best_period(env, fps, args.bpm_min, args.bpm_max)
    rows, period = phase_track(env, fps, bpm, args.win, args.hop)
    good = rows[:, 2] >= args.min_strength
    period_ms = period * 1000.0
    off = rows[:, 1].copy()
    # unwrap only over the windows where the grid is clear; others keep the last clear value
    clear_t, clear_off = rows[good, 0], unwrap_ms(off[good], period_ms)
    print(f"file      : {args.audio}")
    print(f"duration  : {len(x) / SR:.2f} s   grid BPM: {bpm:.4f}   windows clear: {good.sum()}/{len(rows)}")
    if good.sum() < 3:
        print("not enough clear windows")
        return 1
    # linear fit = residual tempo error; the rest = jumps
    slope, icpt = np.polyfit(clear_t, clear_off, 1)
    resid = clear_off - (slope * clear_t + icpt)
    print(f"phase span: {clear_off.max() - clear_off.min():.1f} ms   linear drift: {slope * 60:.2f} ms/min"
          f"   residual sd: {resid.std():.2f} ms   max |resid|: {np.abs(resid).max():.1f} ms")
    d = np.diff(clear_off)
    dt = np.diff(clear_t)
    jumps = [(clear_t[i], d[i]) for i in range(len(d)) if abs(d[i]) >= args.jump_ms and dt[i] <= 3 * args.hop]
    print(f"jumps >= {args.jump_ms} ms between neighbouring clear windows: {len(jumps)}"
          f"  (sum {sum(j for _, j in jumps):+.1f} ms)")
    for t, j in jumps[:40]:
        print(f"   t={t + (args.start or 0):8.2f} s   {j:+7.1f} ms")
    print(f"first-window offset: {clear_off[0]:.1f} ms   last: {clear_off[-1]:.1f} ms")
    if args.csv:
        np.savetxt(args.csv, np.column_stack([rows, good]), delimiter=",",
                   header="t_s,offset_ms,strength,clear", comments="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
