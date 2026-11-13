"""E0 of the Colab cut diagnosis (control): the same GPU work, detached from the kernel as gpuq does.

PURPOSE: piped to `colab exec` by run_diag.sh; starts e0_loop.py with Popen(start_new_session=True).
CHANGELOG: 2026-11-13 created (docs/reports/colab_corte_13min_2026-11-13.md). Same day: LOOP is a raw
string; the first E0 wrote a loop with a syntax error, so its job never ran.
"""
import subprocess, sys
LOOP = r'''
import time, torch
t0 = time.time()
a = torch.randn(4096, 4096, device="cuda")
n = 0
with open("/content/diag.log", "a") as f:
    f.write(f"E0 loop start cuda={torch.cuda.is_available()}\n")
    f.flush()
    while time.time() - t0 < 35 * 60:
        s = time.time()
        while time.time() - s < 30:
            b = a @ a
            torch.cuda.synchronize()
            n += 1
        f.write(f"E0 {(time.time() - t0) / 60:5.1f} min, {n} matmuls\n")
        f.flush()
    f.write("E0 DONE\n")
'''
open("/content/e0_loop.py", "w").write(LOOP)
p = subprocess.Popen([sys.executable, "/content/e0_loop.py"], stdin=subprocess.DEVNULL,
                     stdout=open("/content/e0_loop.out", "w"), stderr=subprocess.STDOUT, start_new_session=True)
up = float(open("/proc/uptime").read().split()[0])
print(f"E0 started pid {p.pid}, vm uptime {up/60:.1f} min", flush=True)
