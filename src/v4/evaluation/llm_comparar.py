"""
PURPOSE: ¿Las conclusiones medidas con el LLM en gpuq valen para el LLM de producción? Corre los mismos pedidos por
         Workers AI (el Gemma del Worker, con `wrangler dev` y un Worker de prueba que nunca se publica: gasta neuronas)
         y compara sus respuestas con las de gpuq (src/v4/evaluation/llm_gpu.py) con el umbral fijado antes de medir en
         config/telefono/llm_transferencia_umbral.json:
           - cuántas respuestas pasan el validador del Worker en cada servidor (cloudflare/validar_respuestas.mjs);
           - si coinciden en lo que «no se oye»;
           - cuánto se parecen los 30 temas que cada uno trae de la biblioteca (coseno CLAP del promedio de sus textos,
             como la lista de hoy).
           python src/v4/evaluation/llm_comparar.py cloudflare --pedidos <pedidos.jsonl> --salida <respuestas_cf.jsonl>
           python src/v4/evaluation/llm_comparar.py comparar --pedidos <p.jsonl> --a <respuestas_cf.jsonl> --b <respuestas_gpu.jsonl>
CHANGELOG:
  - 2026-11-13: Creación (línea B, L7: Gemma en gpuq frente a Gemma en Workers AI).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)

WRANGLER = "wrangler@4.159.0"
UMBRAL = REPO_ROOT / "config" / "telefono" / "llm_transferencia_umbral.json"


def leer_jsonl(path: Path) -> List[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _puerto() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def cmd_cloudflare(args: argparse.Namespace) -> int:
    """Los pedidos por Workers AI con wrangler dev (sesión de wrangler de Gabriel; no se publica nada)."""
    npx = shutil.which("npx")
    if npx is None:
        print("[ERROR] Hace falta Node (npx)")
        return 1
    pedidos = leer_jsonl(args.pedidos)
    salida = Path(args.salida)
    hechas: Dict[str, dict] = {r["id"]: r for r in leer_jsonl(salida)} if salida.exists() else {}
    pendientes = [p for p in pedidos if p["id"] not in hechas]
    print(f"[INFO] {len(pedidos)} pedidos, {len(hechas)} ya contestados, {len(pendientes)} por hacer", flush=True)
    if not pendientes:
        return 0
    work = Path(tempfile.mkdtemp(prefix="llm_cf_"))
    config = {"name": "traktor-banco-llm-dev", "main": (REPO_ROOT / "cloudflare" / "banco_llm_worker.js").as_posix(),
              "compatibility_date": "2026-11-09", "ai": {"binding": "AI"}}
    (work / "wrangler.jsonc").write_text(json.dumps(config), encoding="utf-8")
    puerto = _puerto()
    log = open(work / "wrangler.log", "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen([npx, "--yes", WRANGLER, "dev", "--ip", "127.0.0.1", "--port", str(puerto), "--config",
                             str(work / "wrangler.jsonc"), "--show-interactive-dev-session=false"],
                            cwd=work, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                            env={**os.environ, "WRANGLER_SEND_METRICS": "false", "NO_COLOR": "1"})
    url = f"http://127.0.0.1:{puerto}/run"
    try:
        for _ in range(240):
            try:
                urllib.request.urlopen(urllib.request.Request(url, data=b"{}", method="GET"), timeout=2)
            except urllib.error.HTTPError:
                break                                   # responde (404 a GET): está arriba
            except OSError:
                time.sleep(0.5)
        else:
            print("[ERROR] wrangler dev no arrancó")
            return 1
        for p in pendientes:
            cuerpo = {k: p[k] for k in ("messages", "max_tokens", "temperature", "response_format", "chat_template_kwargs")}
            req = urllib.request.Request(url, data=json.dumps(cuerpo).encode("utf-8"), method="POST",
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                datos = json.loads(r.read().decode("utf-8"))
            hechas[p["id"]] = {"id": p["id"], **datos}
            print(f"  {p['id']}: {datos.get('ms')} ms {'(error)' if datos.get('error') else ''}", flush=True)
            salida.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in hechas.values()), encoding="utf-8")
        usos = [x.get("usage") or {} for x in hechas.values()]
        entrada = sum(u.get("prompt_tokens") or 0 for u in usos)
        salida_t = sum(u.get("completion_tokens") or 0 for u in usos)
        neuronas = entrada * 9091 / 1e6 + salida_t * 27273 / 1e6     # precios de Workers AI del 2026-11-09
        print(f"[OK] {len(hechas)} respuestas; tokens de entrada {entrada}, de salida {salida_t}; unas {neuronas:.0f} neuronas")
        return 0
    finally:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
        else:
            proc.terminate()
        log.close()


def validar(respuestas: Path) -> Dict[str, Optional[dict]]:
    out = subprocess.run([shutil.which("node") or "node", str(REPO_ROOT / "cloudflare" / "validar_respuestas.mjs"), str(respuestas)],
                         capture_output=True, text=True, encoding="utf-8", check=True).stdout
    return {r["id"]: r["valido"] for r in (json.loads(l) for l in out.splitlines() if l.strip())}


def cmd_comparar(args: argparse.Namespace) -> int:
    import numpy as np
    from tools.traktor_live import conceptos_agente as ca
    umbral = json.loads(UMBRAL.read_text(encoding="utf-8"))
    ids = [p["id"] for p in leer_jsonl(args.pedidos)]
    va, vb = validar(Path(args.a)), validar(Path(args.b))
    comunes = [i for i in ids if i in va and i in vb]
    tasa_a = sum(va[i] is not None for i in comunes) / len(comunes)
    tasa_b = sum(vb[i] is not None for i in comunes) / len(comunes)
    ambos = [i for i in comunes if va[i] and vb[i]]
    nso = sum({w.lower() for w in va[i]["no_se_oye"]} == {w.lower() for w in vb[i]["no_se_oye"]} for i in ambos) / max(1, len(ambos))
    print(f"[INFO] {len(comunes)} pedidos en los dos; válidas: A {tasa_a:.2f}, B {tasa_b:.2f}; «no se oye» igual en {nso:.2f}")
    sol: List[float] = []
    if not args.sin_clap and ambos:
        bib = ca.cargar_biblioteca(say=ca.console_say)
        codificar = ca.CodificadorTexto()
        for i in ambos:
            listas = []
            for v in (va[i], vb[i]):
                q = np.asarray(codificar([t["texto"] for t in v["textos"]]), dtype=np.float64).mean(axis=0)
                q = q / np.linalg.norm(q)
                listas.append(set(np.argsort(-(np.asarray(bib.matrix, dtype=np.float64) @ q), kind="stable")[:30].tolist()))
            sol.append(len(listas[0] & listas[1]) / 30)
            print(f"  {i}: {sol[-1]:.2f}  A: {va[i]['textos'][0]['texto'][:60]!r}  B: {vb[i]['textos'][0]['texto'][:60]!r}")
        print(f"[INFO] solapamiento de los 30 primeros: mediana {median(sol):.2f}, mínimo {min(sol):.2f}")
    problemas = []
    if abs(tasa_a - tasa_b) > umbral["valido_diferencia_max"]:
        problemas.append(f"válidas difieren en {abs(tasa_a - tasa_b):.2f}")
    if nso < umbral["no_se_oye_acuerdo_min"]:
        problemas.append(f"«no se oye» coincide en {nso:.2f}")
    if sol and median(sol) < umbral["solapamiento_top30_mediana_min"]:
        problemas.append(f"solapamiento mediano {median(sol):.2f}")
    print("[CUMPLE] se itera en gpuq" if not problemas else "[NO CUMPLE] " + "; ".join(problemas))
    return 0 if not problemas else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Gemma en Workers AI frente a Gemma en gpuq")
    sub = ap.add_subparsers(dest="accion", required=True)
    c = sub.add_parser("cloudflare")
    c.add_argument("--pedidos", type=Path, required=True)
    c.add_argument("--salida", type=Path, required=True)
    c.set_defaults(func=cmd_cloudflare)
    k = sub.add_parser("comparar")
    k.add_argument("--pedidos", type=Path, required=True)
    k.add_argument("--a", type=Path, required=True, help="respuestas de Workers AI")
    k.add_argument("--b", type=Path, required=True, help="respuestas de gpuq")
    k.add_argument("--sin-clap", action="store_true")
    k.set_defaults(func=cmd_comparar)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
