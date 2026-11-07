"""
PURPOSE: Beat phase of one recorded track against a fixed integer-BPM grid, in 15 s bins, next to the audio the
         capture lost in the same bins (capture_timing.csv). Diagnostic for the Spotify recordings.
Usage: python staircase.py WAV START END BPM [TIMING_CSV] [--segment N]
CHANGELOG:
- 2026-11-07: First version.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beat_phase as bp  # noqa: E402


def capture_deficit(timing_csv: Path, segment: int):
    rows = list(csv.DictReader(timing_csv.open(encoding="utf-8")))
    segs, cur = [], []
    for r in rows:
        if cur and float(r["stream_s"]) < float(cur[-1]["stream_s"]):
            segs.append(cur); cur = []
        cur.append(r)
    segs.append(cur)
    seg = segs[segment]
    wall = np.array([float(r["wall_unix"]) for r in seg]); stream = np.array([float(r["stream_s"]) for r in seg])
    gap = np.array([float(r["max_loop_gap_ms"]) for r in seg])
    deficit = ((wall - wall[1]) - (stream - stream[1])) * 1000.0
    return stream, deficit, gap


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    segment = int(sys.argv[sys.argv.index("--segment") + 1]) if "--segment" in sys.argv else -1
    if "--segment" in sys.argv:
        args.remove(str(segment))
    offset = float(sys.argv[sys.argv.index("--offset") + 1]) if "--offset" in sys.argv else 0.0
    if "--offset" in sys.argv:
        args.remove(sys.argv[sys.argv.index("--offset") + 1])
    wav, start, end, bpm = args[0], float(args[1]), float(args[2]), float(args[3])
    timing = Path(args[4]) if len(args) > 4 else None
    x = bp.decode(wav, start, end)
    fps = bp.SR / bp.HOP
    env = bp.kick_envelope(x)
    rows, period = bp.phase_track(env, fps, bpm, 6.0, 1.0)
    ok = rows[:, 2] >= 0.5
    P = period * 1000
    u = bp.unwrap_ms(rows[ok, 1], P)
    tt = rows[ok, 0] + start
    if timing:
        stream, deficit, gap = capture_deficit(timing, segment)
        stream = stream - offset  # audio time of this file
    print(f"{Path(wav).name}  [{start:.2f}-{end:.2f}]  grid {bpm} BPM  clear windows {ok.sum()}/{len(rows)}")
    print(f"{'bin (s in WAV)':>16s} {'beat phase':>11s} {'n':>3s}   {'capture lost':>12s} {'max pause':>9s}")
    first = None
    for b0 in np.arange(start, end, 15.0):
        m = (tt >= b0) & (tt < b0 + 15)
        lost = pause = ""
        if timing is not None:
            sel = (stream >= b0) & (stream < b0 + 15)
            if sel.any():
                d0, d1 = np.interp([b0, min(b0 + 15, end)], stream, deficit)
                lost = f"{d1 - d0:+8.0f} ms"
                pause = f"{gap[sel].max():6.0f} ms"
        ph = f"{np.median(u[m]):8.1f} ms" if m.sum() >= 3 else "         -"
        if m.sum() >= 3 and first is None:
            first = np.median(u[m])
        print(f"{b0:7.1f}-{b0 + 15:<7.1f} {ph:>11s} {int(m.sum()):3d}   {lost:>12s} {pause:>9s}")
    if first is not None:
        last_bins = [np.median(u[(tt >= b0) & (tt < b0 + 15)]) for b0 in np.arange(start, end, 15.0)
                     if ((tt >= b0) & (tt < b0 + 15)).sum() >= 3]
        print(f"phase change first -> last bin: {last_bins[-1] - first:+.0f} ms (negative = beats came early)")
    if timing is not None:
        inside = (stream >= start) & (stream <= end)
        d0, d1 = np.interp([start, end], stream, deficit)
        print(f"capture lost over the track: {d1 - d0:+.0f} ms "
              f"(wall minus audio received; +-50 ms of queue noise)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
