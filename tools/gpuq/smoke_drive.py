"""
PURPOSE: Prueba de punta a punta de gpuq con el Drive de TRAKTOR ML. La PC sube una entrada con un número
         al azar; un trabajo de gpuq la lee desde una VM con GPU, hace una cuenta en la GPU y escribe el
         resultado en el Drive; la PC lo baja y comprueba que la VM devolvió el mismo número.
           python tools/gpuq/smoke_drive.py prepare --run <nombre>      (en la PC)
           python tools/gpuq/smoke_drive.py run --run <nombre>          (lo corre gpuq en la VM)
           python tools/gpuq/smoke_drive.py check --run <nombre>        (en la PC)
         Archivos: <raíz remota>/smoke/<nombre>/input.json y result.json (raíz: remote_storage.py).
CHANGELOG:
  - 2026-11-09: primera versión (puesta en marcha de gpuq en este repo).
"""
from __future__ import annotations

import argparse
import os
import platform
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from src.v4.common.remote_storage import open_remote, remote_root  # noqa: E402


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _paths(run: str) -> tuple[str, str]:
    return f"smoke/{run}/input.json", f"smoke/{run}/result.json"


def machine_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "gpuq_platform": os.environ.get("GPUQ_PLATFORM"), "job": os.environ.get("GPUQ_JOB_ID"),
            "attempt": os.environ.get("GPUQ_ATTEMPT"), "code_sha": os.environ.get("GPUQ_CODE_SHA")}
    if shutil.which("nvidia-smi"):
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True)
        info["nvidia_smi"] = r.stdout.strip()
    return info


def gpu_matmul(n: int = 4096, seed: int = 0) -> dict:
    """A seeded matrix product on the GPU when there is one. Returns device, timing and a checksum."""
    try:
        import torch
    except ImportError:
        return {"torch": None, "cuda": False}
    out = {"torch": torch.__version__, "cuda": bool(torch.cuda.is_available())}
    dev = "cuda" if out["cuda"] else "cpu"
    if out["cuda"]:
        out["device_name"] = torch.cuda.get_device_name(0)
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(n, n, generator=g).to(dev)
    b = torch.randn(n, n, generator=g).to(dev)
    (a @ b).sum().item()  # warm-up
    if out["cuda"]:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    c = a @ b
    if out["cuda"]:
        torch.cuda.synchronize()
    out.update({"device": dev, "n": n, "matmul_ms": round((time.perf_counter() - t0) * 1000, 2),
                "checksum": round(float(c.double().abs().mean().item()), 6)})
    return out


def cmd_prepare(args) -> int:
    remote = open_remote()
    rel_in, rel_out = _paths(args.run)
    if remote.exists(rel_out):
        print(f"ya hay un resultado en {rel_out}; usá otro --run")
        return 2
    payload = {"run": args.run, "nonce": secrets.token_hex(16), "created_at": _now()}
    info = remote.put_json(payload, rel_in)
    print(f"subido {remote_root()}/{rel_in} ({info['size']} bytes, md5 {info['md5']}), nonce {payload['nonce']}")
    return 0


def cmd_run(args) -> int:
    print("máquina:", machine_info(), flush=True)
    remote = open_remote()
    root = remote_root()
    rel_in, rel_out = _paths(args.run)
    print(f"raíz remota {root}", flush=True)
    inp = remote.get_json(rel_in)
    if inp is None:
        print(f"FALLA: no encuentro {root}/{rel_in}; corré antes 'prepare' en la PC", flush=True)
        return 2
    print(f"leído {rel_in}: nonce {inp['nonce']}", flush=True)
    calc = gpu_matmul(n=args.n)
    print("cuenta:", calc, flush=True)
    if args.require_gpu and not calc.get("cuda"):
        print("FALLA: no hay CUDA en esta máquina", flush=True)
        return 3
    result = {"run": args.run, "nonce_echo": inp["nonce"], "input_created_at": inp.get("created_at"),
              "machine": machine_info(), "gpu": calc, "finished_at": _now()}
    info = remote.put_json(result, rel_out)
    back = remote.get_json(rel_out)
    if not back or back.get("nonce_echo") != inp["nonce"]:
        print("FALLA: el resultado releído del Drive no coincide", flush=True)
        return 4
    print(f"escrito y releído {root}/{rel_out} ({info['size']} bytes, md5 {info['md5']})", flush=True)
    return 0


def cmd_check(args) -> int:
    remote = open_remote()
    rel_in, rel_out = _paths(args.run)
    inp, res = remote.get_json(rel_in), remote.get_json(rel_out)
    if inp is None or res is None:
        print(f"falta {'la entrada' if inp is None else 'el resultado'} en {remote_root()}/smoke/{args.run}")
        return 2
    ok_nonce = res.get("nonce_echo") == inp.get("nonce")
    gpu = res.get("gpu") or {}
    print(f"nonce devuelto por la VM: {'coincide' if ok_nonce else 'NO coincide'}")
    print(f"máquina: {res.get('machine', {}).get('gpuq_platform')} · {res.get('machine', {}).get('nvidia_smi')}")
    print(f"GPU: cuda={gpu.get('cuda')} {gpu.get('device_name', '')} · torch {gpu.get('torch')} · "
          f"matmul {gpu.get('n')}x{gpu.get('n')} en {gpu.get('matmul_ms')} ms")
    print(f"código {res.get('machine', {}).get('code_sha')} · terminado {res.get('finished_at')}")
    return 0 if ok_nonce and (gpu.get("cuda") or not args.require_gpu) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("CHANGELOG")[0], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("action", choices=("prepare", "run", "check"))
    ap.add_argument("--run", required=True, help="nombre de la prueba (carpeta smoke/<nombre>)")
    ap.add_argument("--n", type=int, default=4096, help="tamaño de las matrices de la cuenta")
    ap.add_argument("--allow-cpu", dest="require_gpu", action="store_false", help="no fallar sin CUDA (tests)")
    args = ap.parse_args(argv)
    return {"prepare": cmd_prepare, "run": cmd_run, "check": cmd_check}[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
