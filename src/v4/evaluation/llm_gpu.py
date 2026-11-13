"""
PURPOSE: El LLM del teléfono, el mismo modelo que usa el Worker en Workers AI (Gemma 4 26B A4B, pesos abiertos,
         Apache 2.0), corrido con llama.cpp en una GPU de gpuq, para medir instrucciones con el banco del teléfono sin
         gastar neuronas (Gabriel, 2026-11-13). Contesta una lista de pedidos compatibles con OpenAI (la arma
         cloudflare/banco_llm.mjs con las mismas instrucciones del Worker) y deja las respuestas en el remoto:
           <remoto>/<salida>/respuestas.jsonl   una línea por pedido: id, content, finish_reason, usage, ms
           <remoto>/<salida>/meta.json          versión de llama.cpp, modelo y cuantización, GPU, tiempos
         Trabajo de gpuq (AGENTS.md, «Compute»; docs/GPUQ.md): se retoma (salta los ids que ya están en el remoto) y sube
         un parcial cada 20 respuestas o 5 minutos. llama.cpp viene compilado de su página de versiones (fijada) y el
         modelo, en GGUF, de ggml-org en Hugging Face; nada se compila en la máquina.
         Validación en CPU (paso 2 del procedimiento): --backend cpu con un modelo chico (ggml-org/gemma-3-270m-it-GGUF)
         y TRAKTOR_REMOTE apuntando a una carpeta local, en Linux (WSL).
           python src/v4/evaluation/llm_gpu.py correr --pedidos llm_banco/pedidos/<n>.jsonl --salida llm_banco/<corrida>
                  [--cuantizacion Q8_0] [--max N] [--backend cuda|cpu] [--repo ggml-org/gemma-4-26B-A4B-it-GGUF]
CHANGELOG:
  - 2026-11-13: Creación (línea B, L7: las instrucciones del LLM se miden en el cluster).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)

LLAMA_TAG = "b12363"                         # versión fijada de llama.cpp (2026-11-13)
LLAMA_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{nombre}"
PAQUETES = {"cuda": ["llama-{tag}-bin-ubuntu-cuda-12.8-x64.tar.gz", "cudart-llama-{tag}-bin-ubuntu-cuda-12.8-x64.tar.gz"],
            "cpu": ["llama-{tag}-bin-ubuntu-x64.tar.gz"]}
REPO_MODELO = "ggml-org/gemma-4-26B-A4B-it-GGUF"
PUERTO = 8080
CTX = 8192
PARCIAL_CADA = 20
PARCIAL_S = 300.0


def log(texto: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {texto}", flush=True)


def bajar(url: str, destino: Path) -> Path:
    if destino.exists() and destino.stat().st_size > 0:
        return destino
    tmp = destino.with_suffix(destino.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "traktor-ml-llm-gpu"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 22)
    tmp.replace(destino)
    return destino


def preparar_llama(scratch: Path, backend: str, tag: str = LLAMA_TAG) -> Path:
    """Baja y desempaqueta llama.cpp ya compilado; devuelve la ruta de llama-server."""
    base = scratch / f"llama-{tag}-{backend}"
    base.mkdir(parents=True, exist_ok=True)
    for patron in PAQUETES[backend]:
        nombre = patron.format(tag=tag)
        tgz = bajar(LLAMA_URL.format(tag=tag, nombre=nombre), scratch / nombre)
        marca = base / (nombre + ".ok")
        if not marca.exists():
            with tarfile.open(tgz) as t:
                t.extractall(base)
            marca.write_text("ok")
    servidores = [p for p in base.rglob("llama-server") if p.is_file()]
    if not servidores:
        raise RuntimeError(f"no encontré llama-server en {base}")
    servidor = servidores[0]
    servidor.chmod(0o755)
    return servidor


def bajar_modelo(repo: str, cuantizacion: str, scratch: Path) -> Path:
    from huggingface_hub import HfApi, hf_hub_download
    archivos = [f for f in HfApi().list_repo_files(repo) if f.endswith(".gguf") and cuantizacion in f
                and not f.split("/")[-1].startswith(("mtp-", "dflash-", "mmproj-"))]
    if not archivos:
        raise RuntimeError(f"{repo} no tiene un GGUF {cuantizacion}")
    rutas = [Path(hf_hub_download(repo, f, cache_dir=str(scratch / "hf"))) for f in sorted(archivos)]
    return rutas[0]                                 # si viene partido, llama.cpp lee las otras partes solo


def esperar_servidor(proc: subprocess.Popen, timeout_s: float = 1200.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server terminó con código {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PUERTO}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(3)
    raise RuntimeError("llama-server no quedó listo a tiempo")


def preguntar(pedido: dict) -> dict:
    cuerpo = {k: pedido[k] for k in ("messages", "max_tokens", "temperature", "response_format", "chat_template_kwargs")
              if k in pedido}
    req = urllib.request.Request(f"http://127.0.0.1:{PUERTO}/v1/chat/completions",
                                 data=json.dumps(cuerpo).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        datos = json.loads(r.read().decode("utf-8"))
    ms = round((time.perf_counter() - t0) * 1000)
    eleccion = (datos.get("choices") or [{}])[0]
    return {"id": pedido["id"], "content": (eleccion.get("message") or {}).get("content"),
            "finish_reason": eleccion.get("finish_reason"), "usage": datos.get("usage"), "ms": ms}


def gpu_info() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout.strip().replace("\n", " | ")
    except (OSError, subprocess.SubprocessError):
        return "sin GPU"


def correr(args: argparse.Namespace) -> int:
    from src.v4.common.remote_storage import open_remote
    remote = open_remote()
    scratch = Path(os.environ.get("GPUQ_SCRATCH") or tempfile.gettempdir()) / "llm_gpu"
    scratch.mkdir(parents=True, exist_ok=True)
    t_inicio = time.time()

    pedidos_local = scratch / "pedidos.jsonl"
    remote.get(args.pedidos, str(pedidos_local))
    pedidos = [json.loads(l) for l in pedidos_local.read_text(encoding="utf-8").splitlines() if l.strip()]
    respuestas: Dict[str, dict] = {}
    salida_local = scratch / "respuestas.jsonl"
    if "respuestas.jsonl" in set(remote.ls(args.salida)):
        remote.get(f"{args.salida}/respuestas.jsonl", str(salida_local))
        for l in salida_local.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                respuestas[r["id"]] = r
    pendientes = [p for p in pedidos if p["id"] not in respuestas]
    if args.max is not None:
        pendientes = pendientes[: max(0, args.max - len(respuestas))]
    log(f"{len(pedidos)} pedidos, {len(respuestas)} ya contestados, {len(pendientes)} por hacer; GPU: {gpu_info()}")
    if not pendientes:
        return 0

    t0 = time.time()
    servidor = preparar_llama(scratch, args.backend)
    t_llama = time.time() - t0
    t0 = time.time()
    modelo = bajar_modelo(args.repo, args.cuantizacion, scratch)
    t_modelo = time.time() - t0
    log(f"llama.cpp {LLAMA_TAG} listo en {t_llama:.0f} s; modelo {modelo.name} ({modelo.stat().st_size / 1e9:.1f} GB) "
        f"en {t_modelo:.0f} s")
    env = dict(os.environ)
    libs = sorted({str(p.parent) for p in servidor.parent.parent.rglob("*.so*")} | {str(servidor.parent)})
    env["LD_LIBRARY_PATH"] = ":".join(libs + [env.get("LD_LIBRARY_PATH", "")])
    cmd = [str(servidor), "-m", str(modelo), "-c", str(CTX), "--host", "127.0.0.1", "--port", str(PUERTO), "--jinja",
           "-np", "1"]
    if args.backend == "cuda":
        cmd += ["-ngl", "999", "--split-mode", "layer"]
    registro = open(scratch / "llama-server.log", "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=registro, stderr=subprocess.STDOUT, env=env)
    try:
        t0 = time.time()
        esperar_servidor(proc)
        t_carga = time.time() - t0
        log(f"servidor listo en {t_carga:.0f} s")

        def subir() -> None:
            salida_local.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in respuestas.values()),
                                    encoding="utf-8")
            remote.put(str(salida_local), f"{args.salida}/respuestas.jsonl")

        ultimo = time.time()
        hechos = 0
        for p in pendientes:
            try:
                r = preguntar(p)
            except (urllib.error.URLError, OSError, ValueError) as exc:
                r = {"id": p["id"], "content": None, "error": f"{type(exc).__name__}: {exc}"[:300]}
            respuestas[p["id"]] = r
            hechos += 1
            tok = (r.get("usage") or {}).get("completion_tokens")
            log(f"  {p['id']}: {r.get('ms', '?')} ms, {tok} tokens de salida")
            if hechos % PARCIAL_CADA == 0 or time.time() - ultimo > PARCIAL_S:
                subir()
                ultimo = time.time()
        subir()
        ms = [r["ms"] for r in respuestas.values() if r.get("ms")]
        meta = {"llama_cpp": LLAMA_TAG, "backend": args.backend, "repo": args.repo, "modelo": modelo.name,
                "cuantizacion": args.cuantizacion, "gpu": gpu_info(), "maquina": platform.node(), "ctx": CTX,
                "pedidos": len(pedidos), "contestados": len(respuestas),
                "errores": sum(1 for r in respuestas.values() if r.get("error")),
                "segundos": {"llama": round(t_llama), "modelo": round(t_modelo), "carga": round(t_carga),
                             "total": round(time.time() - t_inicio)},
                "ms_por_pedido_mediana": sorted(ms)[len(ms) // 2] if ms else None,
                "commit": os.environ.get("GPUQ_REF") or os.environ.get("GITHUB_SHA")}
        meta_local = scratch / "meta.json"
        meta_local.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        remote.put(str(meta_local), f"{args.salida}/meta.json")
        log(f"[OK] {len(respuestas)} respuestas en {args.salida}; {json.dumps(meta['segundos'])}; "
            f"mediana {meta['ms_por_pedido_mediana']} ms por pedido")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        registro.close()
        if proc.returncode not in (0, None, -15) or not respuestas:
            print((scratch / "llama-server.log").read_text(encoding="utf-8", errors="replace")[-3000:], flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="El LLM del teléfono en una GPU de gpuq (llama.cpp)")
    sub = ap.add_subparsers(dest="accion", required=True)
    c = sub.add_parser("correr")
    c.add_argument("--pedidos", required=True, help="ruta en el remoto del JSONL de pedidos")
    c.add_argument("--salida", required=True, help="carpeta en el remoto para respuestas.jsonl y meta.json")
    c.add_argument("--cuantizacion", default="Q8_0")
    c.add_argument("--repo", default=REPO_MODELO)
    c.add_argument("--backend", choices=("cuda", "cpu"), default="cuda")
    c.add_argument("--max", type=int, default=None, help="contestar como mucho N pedidos en total (trabajo corto)")
    c.set_defaults(func=correr)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
