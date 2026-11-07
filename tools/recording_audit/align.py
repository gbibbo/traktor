"""
PURPOSE: Align a Spotify recording against the original file (Soulseek) second by second (diagnostic).

The offset (where each stretch of the recording is found in the original, minus where it sits in the recording)
stays constant while nothing is lost; each time the capture lost audio it grows by that much. Prints the offset
steps and their total.
Usage: python align.py RECORDING ORIGINAL [--win 2.0] [--hop 1.0]
CHANGELOG:
- 2026-11-07: First version.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np

SR = 8000


def decode(path: str) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<f4").astype(np.float64)


def ncc_best(seg: np.ndarray, ref: np.ndarray) -> tuple[int, float]:
    """Lag of seg inside ref with the best normalized correlation, and that correlation."""
    n = len(seg)
    size = 1 << int(np.ceil(np.log2(len(ref) + n)))
    seg0 = seg - seg.mean()
    corr = np.fft.irfft(np.fft.rfft(ref, size) * np.conj(np.fft.rfft(seg0, size)), size)[: len(ref) - n + 1]
    c2 = np.concatenate([[0.0], np.cumsum(ref ** 2)])
    c1 = np.concatenate([[0.0], np.cumsum(ref)])
    var = (c2[n:] - c2[:-n]) - (c1[n:] - c1[:-n]) ** 2 / n
    corr = corr / (np.sqrt(np.maximum(var, 1e-12)) * np.sqrt(np.sum(seg0 ** 2)) + 1e-12)
    i = int(np.argmax(corr))
    return i, float(corr[i])


def main() -> int:
    args = sys.argv[1:]
    win = float(args[args.index("--win") + 1]) if "--win" in args else 2.0
    hop = float(args[args.index("--hop") + 1]) if "--hop" in args else 1.0
    rec, ref = decode(args[0]), decode(args[1])
    # global start: the first 20 s of the recording inside the first 60 s of the original
    lag0, c0 = ncc_best(rec[5 * SR:25 * SR], ref[: 70 * SR])
    offset = lag0 / SR - 5.0
    print(f"recording {len(rec) / SR:.2f} s, original {len(ref) / SR:.2f} s; start offset {offset * 1000:+.1f} ms "
          f"(corr {c0:.2f})")
    rows = []
    W, H, S = int(win * SR), int(hop * SR), int(0.5 * SR)
    for a in range(0, len(rec) - W, H):
        lo = int(a + offset * SR) - S
        if lo < 0 or lo + W + 2 * S > len(ref):
            continue
        lag, c = ncc_best(rec[a:a + W], ref[lo:lo + W + 2 * S])
        off = (lo + lag - a) / SR
        rows.append((a / SR, off, c))
        if c > 0.8:
            offset = off
    rows = np.array(rows)
    good = rows[:, 2] > 0.8
    print(f"windows aligned with corr > 0.8: {good.sum()}/{len(rows)}")
    t, off = rows[good, 0], rows[good, 1] * 1000
    print(f"offset at start {off[:5].mean():+.1f} ms, at end {off[-5:].mean():+.1f} ms -> "
          f"total {off[-5:].mean() - off[:5].mean():+.1f} ms (positive = the recording lost that much)")
    steps = np.diff(off)
    big = np.flatnonzero(np.abs(steps) >= 5)
    print(f"steps >= 5 ms: {len(big)}, sum {steps[big].sum():+.1f} ms")
    for i in big[:40]:
        print(f"   at {t[i + 1]:7.1f} s in the recording: {steps[i]:+7.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
