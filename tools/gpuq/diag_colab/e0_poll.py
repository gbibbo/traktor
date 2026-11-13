"""E0 poll: one short `colab exec` per minute, like the gpuq poll.

PURPOSE: prints VM uptime, the last line of /content/diag.log and GPU use.
CHANGELOG: 2026-11-13 created (docs/reports/colab_corte_13min_2026-11-13.md).
"""
import subprocess
up = float(open("/proc/uptime").read().split()[0])
try:
    last = open("/content/diag.log").read().strip().splitlines()[-1]
except Exception as e:
    last = f"no log: {e}"
q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                   capture_output=True, text=True).stdout.strip()
print(f"E0POLL uptime {up/60:.1f} min | {last} | gpu {q}", flush=True)
