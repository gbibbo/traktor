"""
PURPOSE: Do the losses the capture logged explain the grid of a recorded track? (diagnostic)

Grid coherence = how much of the kick onset energy one constant grid catches (1 = all of it on the grid), at the
best BPM. Computed on the file's own time axis and on the axis corrected with the audio the capture lost before
each instant (capture_timing.csv: wall clock minus audio received). If the losses are the cause, the corrected
axis is clearly more coherent; if they are not, correcting changes little or makes it worse.

Usage: python explain.py AUDIO TIMING_CSV OFFSET_S [--segment N] [--bpm-min 120 --bpm-max 160]
CHANGELOG:
- 2026-11-07: First version.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beat_phase as bp  # noqa: E402
from staircase import capture_deficit  # noqa: E402


def coherence(env: np.ndarray, t: np.ndarray, bpm_grid: np.ndarray) -> tuple[float, float]:
    w = env - env.mean()
    norm = np.abs(w).sum()
    best = (0.0, 0.0)
    for b in bpm_grid:
        c = abs(np.sum(w * np.exp(-2j * np.pi * t * b / 60.0))) / norm
        if c > best[0]:
            best = (c, b)
    return best


def main() -> int:
    argv = sys.argv[1:]
    def opt(name, default):
        if name in argv:
            i = argv.index(name); v = argv[i + 1]; del argv[i:i + 2]; return type(default)(v)
        return default
    segment = opt("--segment", -1); bmin = opt("--bpm-min", 120.0); bmax = opt("--bpm-max", 160.0)
    audio, timing, offset = argv[0], Path(argv[1]), float(argv[2])
    x = bp.decode(audio, None, None)
    fps = bp.SR / bp.HOP
    env = bp.kick_envelope(x)
    t = np.arange(len(env)) / fps
    if str(timing) == "-":
        lost = np.zeros_like(t)
    else:
        stream, deficit, _ = capture_deficit(timing, segment)
        # median of 7 rows (7 s): the queue backlog makes each row +-50 ms of noise, a loss stays
        smooth = np.array([np.median(deficit[max(0, i - 3):i + 4]) for i in range(len(deficit))])
        lost = np.interp(t + offset, stream, smooth) - np.interp(offset, stream, smooth)   # ms lost before t
        lost = np.maximum.accumulate(np.maximum(lost, 0))   # losses only add up
    b0 = bp.best_period(env, fps, bmin, bmax)
    grid = np.arange(b0 - 0.6, b0 + 0.6, 0.002)
    raw = coherence(env, t, grid)
    fixed = coherence(env, t + lost / 1000.0, grid)
    print(f"{Path(audio).name}")
    print(f"   capture lost over the file: {lost[-1]:.0f} ms")
    print(f"   grid coherence as recorded:      {raw[0]:.3f} at {raw[1]:.3f} BPM")
    print(f"   grid coherence, losses put back: {fixed[0]:.3f} at {fixed[1]:.3f} BPM")
    return 0


if __name__ == "__main__":
    sys.exit(main())
