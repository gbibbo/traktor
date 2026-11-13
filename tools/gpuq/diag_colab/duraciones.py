"""How long each Colab attempt lasted, from the dispatchers' `docker logs`.

PURPOSE: pairs every "colab: starting <job> attempt N" with the next "colab: <job> -> <outcome>" and prints the
duration of each attempt, then a summary per period (before and after --corte) for GPU attempts. Input: one file
per dispatcher, saved on the Pi with
    docker logs gpuq 2>&1 | grep '] colab: ' > gpuq.txt
    docker logs erclm-colab 2>&1 | grep '] colab: ' > erclm.txt
CHANGELOG: 2026-11-13 created for docs/reports/colab_corte_13min_2026-11-13.md.
"""
from __future__ import annotations

import argparse
import re
import statistics
from datetime import datetime
from pathlib import Path

LINE = re.compile(r"\[(\S+ \S+)\] colab: (?:starting (\S+) attempt (\d+)|(\S+) -> (\S+))")


def attempts(text: str) -> list[dict]:
    """Attempts in one dispatcher's log, in order."""
    open_, rows = {}, []
    for line in text.splitlines():
        m = LINE.search(line)
        if not m:
            continue
        t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        if m.group(2):
            open_[m.group(2)] = (t, int(m.group(3)))
        elif m.group(4) in open_:
            t0, n = open_.pop(m.group(4))
            rows.append({"job": m.group(4), "n": n, "start": t0, "min": (t - t0).total_seconds() / 60,
                         "outcome": m.group(5)})
    return rows


def summary(rows: list[dict]) -> str:
    out = []
    for outcome in sorted({r["outcome"] for r in rows}):
        d = sorted(r["min"] for r in rows if r["outcome"] == outcome)
        out.append(f"{outcome} {len(d)} (median {statistics.median(d):.1f} min, max {d[-1]:.1f}, "
                   f"over 15 min {sum(x > 15 for x in d)})")
    return "; ".join(out) or "none"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="+", type=Path, help="one grep of docker logs per dispatcher")
    ap.add_argument("--cpu", default="", help="comma-separated ids of CPU-only jobs (left out of the summary)")
    ap.add_argument("--corte", default="2026-11-02 17:00", help="UTC time that splits the summary")
    a = ap.parse_args(argv)
    cpu = {j for j in a.cpu.split(",") if j}
    cut = datetime.strptime(a.corte, "%Y-%m-%d %H:%M")
    rows = []
    for path in a.logs:
        for r in attempts(path.read_text(encoding="utf-8", errors="replace")):
            rows.append(r | {"src": path.stem, "gpu": r["job"] not in cpu})
    rows.sort(key=lambda r: r["start"])
    for r in rows:
        print(f"{r['src']:8} {r['start']:%Y-%m-%d %H:%M} {'GPU' if r['gpu'] else 'CPU'} {r['job']:34} "
              f"#{r['n']:<3} {r['min']:6.1f} min  {r['outcome']}")
    gpu = [r for r in rows if r["gpu"]]
    print(f"\nGPU before {a.corte} UTC: " + summary([r for r in gpu if r["start"] < cut]))
    print(f"GPU from {a.corte} UTC: " + summary([r for r in gpu if r["start"] >= cut]))


if __name__ == "__main__":
    main()
