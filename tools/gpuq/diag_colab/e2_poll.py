"""E2 poll: a long `colab exec` that stays connected up to 10 min, then reports like the E0 poll.

PURPOSE: the candidate fix for gpuq (docs/reports/colab_corte_13min_2026-11-13.md). The job stays detached, as
in E0, but the kernel is busy and the connection open almost all the time: this cell prints a line every 30 s
for HOLD_S seconds, returns early when the job has written "E0 DONE", and run_diag.sh starts the next one at
once.
CHANGELOG: 2026-11-13 created.
"""
import os
import subprocess
import time

HOLD_S = 600
t0 = time.time()


def last_line():
    try:
        return open("/content/diag.log").read().strip().splitlines()[-1]
    except Exception as e:  # noqa: BLE001 - the report must come back whatever happens
        return f"no log: {e}"


while time.time() - t0 < HOLD_S and "DONE" not in last_line():
    time.sleep(30)
    print(f"E2 hold {(time.time() - t0) / 60:4.1f} min | {last_line()}", flush=True)
up = float(open("/proc/uptime").read().split()[0])
q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                   capture_output=True, text=True).stdout.strip()
print(f"E2POLL uptime {up / 60:.1f} min | {last_line()} | gpu {q} | pid {os.getpid()}", flush=True)
