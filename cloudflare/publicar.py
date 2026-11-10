"""
PURPOSE: Publica «Traktor Concept AI» y conecta este ordenador, en un solo paso que corre Gabriel (publicar en producción
         es decisión suya): despliega cloudflare/concept-ai-worker.js con wrangler, deja un DEVICE_TOKEN nuevo en Cloudflare
         y el mismo en artifacts/v4/traktor_live/conceptos_agente.json (fuera de Git; nunca se imprime), comprueba la
         conexión sin consumir trabajos de la cola y activa el agente con Traktor (conceptos_agente.py instalar).
         Antes: npx wrangler@4.159.0 login, una vez. Si algo sale mal: npx wrangler@4.159.0 rollback (desde cloudflare/).
         Uso, desde la raíz del repo y con el Python del pipeline:  python cloudflare/publicar.py
CHANGELOG:
  - 2026-11-09: Creación inicial.
"""
from __future__ import annotations

import json
import secrets
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from tools.traktor_live import conceptos_agente as ca  # noqa: E402

CF = REPO_ROOT / "cloudflare"
WRANGLER = "wrangler@4.159.0"


def wrangler(*args: str, entrada: str | None = None) -> subprocess.CompletedProcess:
    npx = shutil.which("npx")
    if npx is None:
        sys.exit("[ERROR] Hace falta Node.js (npx)")
    return subprocess.run([npx, "--yes", WRANGLER, *args], cwd=CF, input=entrada, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=300)


def paso(texto: str) -> None:
    print(f"\n== {texto}", flush=True)


def main() -> int:
    paso("Cuenta de Cloudflare")
    who = wrangler("whoami")
    if "You are logged in" not in who.stdout:
        print("[ERROR] Falta el login: npx wrangler@4.159.0 login")
        return 1
    print("[OK] " + next(l.strip() for l in who.stdout.splitlines() if "logged in" in l))

    paso("Publicar el Worker")
    dep = wrangler("deploy")
    if dep.returncode != 0:
        print(dep.stdout[-2000:], dep.stderr[-2000:])
        print("[ERROR] No se publicó. Nada cambió en Cloudflare.")
        return 1
    for linea in dep.stdout.splitlines():
        if any(k in linea for k in ("Uploaded", "Deployed", "workers.dev", "Version ID")):
            print("   " + linea.strip())

    paso("Token del ordenador")
    previo = ca.load_ajustes()
    token = previo.token if previo else secrets.token_urlsafe(32)
    sec = wrangler("secret", "put", "DEVICE_TOKEN", entrada=token)
    if sec.returncode != 0:
        print(sec.stdout[-1500:], sec.stderr[-1500:])
        print("[ERROR] No se pudo guardar el token en Cloudflare")
        return 1
    ca.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    ca.CONFIG_FILE.write_text(json.dumps({"url": ca.DEFAULT_URL, "device_token": token}, indent=2), encoding="utf-8")
    print(f"[OK] {'Mismo token de antes' if previo else 'Token nuevo'} en Cloudflare y en {ca.CONFIG_FILE} (fuera de Git)")

    paso("Comprobar la conexión (sin tocar la cola)")
    client = ca.WorkerClient(ca.DEFAULT_URL, token, timeout=20)
    for intento in range(10):                     # el secreto tarda unos segundos en llegar
        try:
            if client.claim(str(uuid.uuid4())) is None:
                print("[OK] El Worker acepta a este ordenador")
                break
        except ca.NoAutorizado:
            time.sleep(3)
        except (OSError, urllib.error.URLError) as exc:
            print(f"   reintento ({type(exc).__name__})")
            time.sleep(3)
    else:
        print("[ERROR] El Worker no acepta el token todavía: volver a correr este script en un minuto")
        return 1

    paso("Agente con Traktor")
    inst = subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "traktor_live" / "conceptos_agente.py"), "instalar"],
                          cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(inst.stdout.strip() or inst.stderr.strip())
    if inst.returncode != 0:
        return 1

    print(f"\nListo. En el teléfono: {ca.DEFAULT_URL}/  (con Traktor abierto en este ordenador)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
