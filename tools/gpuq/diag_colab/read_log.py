"""Reads the end of the diagnosis log on the VM.

PURPOSE: after E1, shows whether the VM is still alive and how far the work got.
CHANGELOG: 2026-11-13 created (docs/reports/colab_corte_13min_2026-11-13.md).
"""
up = float(open("/proc/uptime").read().split()[0])
try:
    lines = open("/content/diag.log").read().strip().splitlines()
except Exception as e:
    lines = [f"no log: {e}"]
print(f"READLOG uptime {up/60:.1f} min | last: {lines[-1]} | lines {len(lines)}", flush=True)
