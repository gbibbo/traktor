"""E1 of the Colab cut diagnosis: GPU work inside one long `colab exec`, connection open all the time.

PURPOSE: piped to `colab exec` by run_diag.sh; 35 min of T4 matmuls, one line every 30 s.
CHANGELOG: 2026-11-13 created (docs/reports/colab_corte_13min_2026-11-13.md).
"""
import time, torch
t0 = time.time()
up = float(open("/proc/uptime").read().split()[0])
print(f"E1 start, vm uptime {up/60:.1f} min, cuda {torch.cuda.is_available()}", flush=True)
a = torch.randn(4096, 4096, device="cuda")
n = 0
with open("/content/diag.log", "a") as f:
    while time.time() - t0 < 35 * 60:
        s = time.time()
        while time.time() - s < 30:
            b = a @ a
            torch.cuda.synchronize()
            n += 1
        msg = f"E1 {(time.time() - t0) / 60:5.1f} min, {n} matmuls"
        print(msg, flush=True)
        f.write(msg + "\n")
        f.flush()
print("E1 DONE", flush=True)
