"""E3 and E4 of the Colab cut diagnosis: the synthetic GPU work of E0, sent as an ordinary gpuq job.

PURPOSE: holds the work fixed (T4 matmuls) and adds one thing at a time. E3, with no options, changes only
the path: gpuq's whole start (credentials upload, rclone, checkout of this repo, detached job, one poll a
minute). If E3 is cut at about 13 min, something in gpuq's start triggers the cut; if it lives, something
the real jobs do does. GPU work alone (E0, E1) and CPU work alone (ERC-LM, October) do not, so E4 adds the
real jobs' traffic: --drive N downloads N real Beatport shard tars of DJSets v1 (about 550 MB each, read
only), one every 3 min, and puts a small partial on the Drive every 3 min under diag_colab/<run>/, like A0;
--hf downloads the two models A0 loads (CLAP and MAEST) at the start. Queue only with Gabriel's OK (given
on 2026-11-13 for the Colab lane):
    python tools/gpuq/cola.py add traktor-diag-e3-1 --cmd "python tools/gpuq/diag_colab/carga_gpu.py --min 35" \
        --gpu T4 --platform colab --no-setup --timeout-h 1
    ... --cmd "python tools/gpuq/diag_colab/carga_gpu.py --min 20 --drive 6 --run e4-drive" ...
    ... --cmd "python tools/gpuq/diag_colab/carga_gpu.py --min 20 --hf --run e4-hf" ...
A test that lives 20 min while real jobs die at 12 to 14 says that piece is not the trigger.
CHANGELOG: 2026-11-13 created (docs/reports/colab_corte_13min_2026-11-13.md). Same day: --drive, --hf, --run
for E4; then --sin-bajadas and --sin-subidas for E5, which split the Drive traffic of E4 (the cut came with it):
    ... --cmd "python tools/gpuq/diag_colab/carga_gpu.py --min 20 --drive 6 --sin-bajadas --run e5-subidas" ...
    ... --cmd "python tools/gpuq/diag_colab/carga_gpu.py --min 20 --drive 6 --sin-subidas --run e5-bajadas" ...
    ... --cmd "RCLONE_MULTI_THREAD_STREAMS=0 python tools/gpuq/diag_colab/carga_gpu.py --min 20 --drive 6
            --sin-subidas --run e5-1hilo" ...
"""
import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SHARDS = "djsets_v1/audio/beatport"
MODELS = ("laion/clap-htsat-unfused", "mtg-upf/discogs-maest-30s-pw-129e")
EVERY_S = 180


def remote():
    sys.path.insert(0, str(ROOT))
    from src.v4.common.remote_storage import open_remote
    return open_remote()


class Traffic:
    """The Drive traffic of a real A0 job: a shard tar down and a small partial up, every EVERY_S seconds."""

    def __init__(self, n_tars: int, run: str, down: bool = True, up: bool = True):
        self.r = remote()
        self.tars = sorted(t for t in self.r.ls(SHARDS) if t.endswith(".tar"))[:n_tars] if down else []
        self.run, self.next, self.done_tars, self.puts, self.up = run, 0.0, 0, 0, up
        self.tmp = Path(tempfile.mkdtemp(prefix="e4_"))

    def tick(self, elapsed_min: float):
        if time.time() < self.next:
            return
        self.next = time.time() + EVERY_S
        if self.done_tars < len(self.tars):
            local = self.tmp / "shard.tar"
            t = time.time()
            self.r.get(f"{SHARDS}/{self.tars[self.done_tars]}", local)
            mb = local.stat().st_size / 2**20
            local.unlink()
            self.done_tars += 1
            print(f"E4 drive: {self.tars[self.done_tars - 1]} down, {mb:.0f} MB in {time.time() - t:.0f} s", flush=True)
        if not self.up:
            return
        part = self.tmp / "parcial.txt"
        part.write_text(f"{elapsed_min:.1f} min\n", encoding="utf-8")
        self.r.put(part, f"diag_colab/{self.run}/parcial_{self.puts:03d}.txt")
        self.puts += 1
        print(f"E4 drive: partial {self.puts} up", flush=True)


def download_models():
    from huggingface_hub import snapshot_download
    for name in MODELS:
        t = time.time()
        path = snapshot_download(name, token=os.environ.get("HF_TOKEN") or None)
        mb = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 2**20
        print(f"E4 hf: {name}, {mb:.0f} MB in {time.time() - t:.0f} s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min", type=float, default=35, help="minutes of GPU work")
    ap.add_argument("--drive", type=int, default=0, help="real shard tars to download, one every 3 min")
    ap.add_argument("--hf", action="store_true", help="download CLAP and MAEST from Hugging Face at the start")
    ap.add_argument("--sin-bajadas", action="store_true", help="with --drive: only the partials, no tar downloads")
    ap.add_argument("--sin-subidas", action="store_true", help="with --drive: only the tar downloads, no partials")
    ap.add_argument("--run", default="e3", help="name of the run, for the partials on the Drive")
    a = ap.parse_args()
    import torch  # on the VM; not needed to read --help here

    t0 = time.time()
    up = float(open("/proc/uptime").read().split()[0])
    tag = "E4" if (a.drive or a.hf) else "E3"
    print(f"{tag} start, vm uptime {up / 60:.1f} min, cuda {torch.cuda.is_available()}, drive {a.drive}, "
          f"hf {a.hf}", flush=True)
    if a.hf:
        download_models()
    traffic = Traffic(a.drive, a.run, down=not a.sin_bajadas, up=not a.sin_subidas) if a.drive else None
    if traffic:
        print(f"E4 drive: {len(traffic.tars)} tars to get, partials {'on' if traffic.up else 'off'}, "
              f"RCLONE_MULTI_THREAD_STREAMS={os.environ.get('RCLONE_MULTI_THREAD_STREAMS', 'default')}", flush=True)
    x = torch.randn(4096, 4096, device="cuda")
    n = 0
    while time.time() - t0 < a.min * 60:
        if traffic:
            traffic.tick((time.time() - t0) / 60)
        s = time.time()
        while time.time() - s < 30:
            x @ x
            torch.cuda.synchronize()
            n += 1
        print(f"{tag} {(time.time() - t0) / 60:5.1f} min, {n} matmuls", flush=True)
    print(f"{tag} DONE", flush=True)


if __name__ == "__main__":
    main()
