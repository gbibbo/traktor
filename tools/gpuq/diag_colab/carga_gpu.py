"""E3 of the Colab cut diagnosis: the synthetic GPU work of E0, sent as an ordinary gpuq job.

PURPOSE: holds the work fixed (T4 matmuls, no Drive, no downloads) and changes only the path: gpuq's whole
start (credentials upload, rclone, checkout of this repo, detached job, one poll a minute). If it is cut at
about 13 min, something in gpuq's start triggers the cut; if it lives, something the real jobs do (Drive and
Hugging Face traffic, CPU decoding) does. Queue it only with Gabriel's OK and with the Colab lane free:
    python tools/gpuq/cola.py add traktor-diag-e3-1 --cmd "python tools/gpuq/diag_colab/carga_gpu.py --min 35" \
        --gpu T4 --platform colab --no-setup --timeout-h 1
CHANGELOG: 2026-11-13 created (docs/reports/colab_corte_13min_2026-11-13.md).
"""
import argparse
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min", type=float, default=35, help="minutes of GPU work")
    a = ap.parse_args()
    import torch  # on the VM; not needed to read --help here

    t0 = time.time()
    up = float(open("/proc/uptime").read().split()[0])
    print(f"E3 start, vm uptime {up / 60:.1f} min, cuda {torch.cuda.is_available()}", flush=True)
    x = torch.randn(4096, 4096, device="cuda")
    n = 0
    while time.time() - t0 < a.min * 60:
        s = time.time()
        while time.time() - s < 30:
            x @ x
            torch.cuda.synchronize()
            n += 1
        print(f"E3 {(time.time() - t0) / 60:5.1f} min, {n} matmuls", flush=True)
    print("E3 DONE", flush=True)


if __name__ == "__main__":
    main()
