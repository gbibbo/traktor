"""How long each Colab or Kaggle attempt lasted, from the dispatchers' `docker logs`, and the week's Kaggle hours.

PURPOSE: pairs every "<lane>: starting <job> attempt N" with the next "<lane>: <job> -> <outcome>" and prints the
duration of each attempt, then a summary per period (before and after --corte) for GPU attempts. With
--horas-desde it also adds up the lane's GPU hours since that time, counting the attempts still running: that is
the check of the Kaggle quota (about 30 GPU hours a week for all of Gabriel's projects, AGENTS.md «Colab first»).
Input: one file per dispatcher, saved on the Pi with
    docker logs gpuq 2>&1 | grep -E '] (colab|kaggle): ' > gpuq.txt
    docker logs erclm-colab 2>&1 | grep -E '] (colab|kaggle): ' > erclm.txt
Kaggle hours this week:
    python tools/gpuq/diag_colab/duraciones.py gpuq.txt erclm.txt --lane kaggle --horas-desde "2026-11-08 00:00"
CHANGELOG: 2026-11-13 created for docs/reports/colab_corte_13min_2026-11-13.md. Same day: --lane and --horas-desde
for the Kaggle quota.
"""
from __future__ import annotations

import argparse
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

LINE = re.compile(r"\[(\S+ \S+)\] (colab|kaggle): (?:starting (\S+) attempt (\d+)|(\S+) -> (\S+))")


def attempts(text: str, lane: str = "colab", now: datetime | None = None) -> list[dict]:
    """Attempts of one lane in one dispatcher's log, in order. With `now`, attempts still open count as running."""
    open_, rows = {}, []
    for line in text.splitlines():
        m = LINE.search(line)
        if not m or m.group(2) != lane:
            continue
        t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        if m.group(3):
            open_[m.group(3)] = (t, int(m.group(4)))
        elif m.group(5) in open_:
            t0, n = open_.pop(m.group(5))
            rows.append({"job": m.group(5), "n": n, "start": t0, "min": (t - t0).total_seconds() / 60,
                         "outcome": m.group(6)})
    if now is not None:
        for job, (t0, n) in open_.items():
            rows.append({"job": job, "n": n, "start": t0, "min": (now - t0).total_seconds() / 60,
                         "outcome": "running"})
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
    ap.add_argument("--lane", choices=("colab", "kaggle"), default="colab")
    ap.add_argument("--cpu", default="", help="comma-separated ids of CPU-only jobs (left out of the summary)")
    ap.add_argument("--corte", default="2026-11-02 17:00", help="UTC time that splits the summary")
    ap.add_argument("--horas-desde", help="UTC time: add up the lane's GPU hours from then, running attempts included")
    ap.add_argument("--ahora", help="UTC time taken as now for running attempts (default: the clock)")
    a = ap.parse_args(argv)
    cpu = {j for j in a.cpu.split(",") if j}
    cut = datetime.strptime(a.corte, "%Y-%m-%d %H:%M")
    now = (datetime.strptime(a.ahora, "%Y-%m-%d %H:%M") if a.ahora
           else datetime.now(timezone.utc).replace(tzinfo=None))
    rows = []
    for path in a.logs:
        for r in attempts(path.read_text(encoding="utf-8", errors="replace"), a.lane, now):
            rows.append(r | {"src": path.stem, "gpu": r["job"] not in cpu})
    rows.sort(key=lambda r: r["start"])
    for r in rows:
        print(f"{r['src']:8} {r['start']:%Y-%m-%d %H:%M} {'GPU' if r['gpu'] else 'CPU'} {r['job']:34} "
              f"#{r['n']:<3} {r['min']:6.1f} min  {r['outcome']}")
    gpu = [r for r in rows if r["gpu"]]
    print(f"\nGPU before {a.corte} UTC: " + summary([r for r in gpu if r["start"] < cut]))
    print(f"GPU from {a.corte} UTC: " + summary([r for r in gpu if r["start"] >= cut]))
    if a.horas_desde:
        since = datetime.strptime(a.horas_desde, "%Y-%m-%d %H:%M")
        hours = sum(r["min"] for r in gpu if r["start"] >= since) / 60
        running = sum(1 for r in gpu if r["start"] >= since and r["outcome"] == "running")
        print(f"{a.lane} GPU hours since {a.horas_desde} UTC: {hours:.1f} ({running} attempt(s) still running)")


if __name__ == "__main__":
    main()
