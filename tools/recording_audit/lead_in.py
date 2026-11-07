"""
PURPOSE: Lead before the first sound and the first kick of audio files, next to Traktor's AutoGrid start.
         Usage: python lead_in.py FILE_OR_FOLDER... (TRAKTOR_NML=<collection.nml> to read another collection)
CHANGELOG:
- 2026-11-07: First version (diagnostic of the Spotify recordings' grid,
              docs/reports/grabaciones_spotify_grilla_2026-11-07.md).
"""
from __future__ import annotations

import os
import sys
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

SR = 44100
NML = Path(os.environ.get("TRAKTOR_NML")
           or Path.home() / "Documents" / "Native Instruments" / "Traktor 4.4.1" / "collection.nml")


def decode(path: Path, seconds: float = 20.0) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-t", str(seconds), "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<f4").astype(np.float64)


def first_above(x: np.ndarray, dbfs: float) -> float | None:
    idx = np.flatnonzero(np.abs(x) >= 10 ** (dbfs / 20))
    return idx[0] / SR * 1000 if len(idx) else None


def first_kick_ms(x: np.ndarray) -> float | None:
    """Start of the first strong low-band (35-150 Hz) attack: where the smoothed energy first reaches 10 % of
    the typical kick peak of the first 20 s, walked back to where its rise began."""
    sos = butter(4, [35, 150], btype="band", fs=SR, output="sos")
    p = sosfiltfilt(sos, x) ** 2
    env = sosfiltfilt(butter(2, 60, btype="low", fs=SR, output="sos"), p)
    env = np.maximum(env, 0)
    peak = np.percentile(env, 99.5)
    if peak <= 0:
        return None
    above = np.flatnonzero(env >= 0.1 * peak)
    if not len(above):
        return None
    i = above[0]
    floor = 0.01 * peak
    while i > 0 and env[i - 1] > floor and env[i - 1] <= env[i]:
        i -= 1
    return i / SR * 1000


def traktor_grid() -> dict[str, tuple[float | None, float | None]]:
    out = {}
    if not NML.exists():
        return out
    for e in ET.parse(NML).getroot().iter("ENTRY"):
        loc = e.find("LOCATION")
        if loc is None:
            continue
        starts = [float(c.get("START")) for c in e.findall("CUE_V2") if c.get("TYPE") == "4"]
        tempo = e.find("TEMPO")
        out[(loc.get("FILE") or "").lower()] = (starts[0] if starts else None,
                                                float(tempo.get("BPM")) if tempo is not None and tempo.get("BPM") else None)
    return out


def main() -> int:
    grid = traktor_grid()
    print(f"{'file':42s} {'-60dB':>7s} {'-40dB':>7s} {'kick':>7s} {'Traktor':>8s} {'BPM':>8s}")
    for arg in sys.argv[1:]:
        for f in sorted(Path(arg).glob("*.mp3")) if Path(arg).is_dir() else [Path(arg)]:
            x = decode(f)
            g = grid.get(f.name.lower(), (None, None))
            fmt = lambda v: f"{v:7.1f}" if v is not None else "      -"
            print(f"{f.name[:42]:42s} {fmt(first_above(x, -60))} {fmt(first_above(x, -40))} {fmt(first_kick_ms(x))} "
                  f"{(f'{g[0]:8.1f}' if g[0] is not None else '       -')} {(f'{g[1]:8.3f}' if g[1] else '       -')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
