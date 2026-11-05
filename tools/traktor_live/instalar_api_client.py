"""
PURPOSE: Instala, revisa o desinstala traktor-api-client (tools/traktor_live/traktor_qml/) en la carpeta
         de programa de Traktor: copia Api/ a Resources64/qml/CSI/D2/ y agrega dos líneas al D2.qml de
         la versión instalada (import "./Api" y ApiModule {}), en vez de reemplazarlo por el D2.qml del
         mod, que es de otra versión. Antes guarda una copia de la carpeta D2 en
         artifacts/v4/traktor_backups/. Con Traktor abierto no hace nada. Si hace falta permiso de
         administrador, se vuelve a abrir pidiéndolo (Windows muestra el aviso de UAC).
CHANGELOG:
  - 2026-11-04: Creación inicial (Traktor 4.4.1).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from src.v4.common.traktor_nml import traktor_running  # noqa: E402

VENDOR_API = Path(__file__).resolve().parent / "traktor_qml" / "Api"
BACKUP_ROOT = REPO_ROOT / "artifacts" / "v4" / "traktor_backups"
LOG_PATH = BACKUP_ROOT / "instalar_api_client.log"
IMPORT_LINE = 'import "./Api"'
MODULE_LINE = "  ApiModule {}"
ANCHOR_IMPORT = 'import "../Common/Settings"'


def default_d2() -> Path:
    base = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    return base / "Native Instruments" / "Traktor Pro 4" / "Resources64" / "qml" / "CSI" / "D2"


def patch_d2(text: str) -> str:
    """D2.qml con el módulo Api: el import después de los imports de Traktor y ApiModule {} al
    principio del Mapping, como en el D2.qml del mod. Si ya está, lo devuelve igual."""
    if "ApiModule" in text:
        return text
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(nl)
    imports = [i for i, ln in enumerate(lines) if ln.strip().startswith("import ")]
    anchor = next((i for i, ln in enumerate(lines) if ln.strip() == ANCHOR_IMPORT), imports[-1] if imports else None)
    mapping = next((i for i, ln in enumerate(lines) if ln.strip() == "Mapping"), None)
    if anchor is None or mapping is None:
        raise ValueError("D2.qml no tiene la forma esperada (imports y 'Mapping'): no lo toco")
    brace = next((i for i in range(mapping + 1, len(lines)) if lines[i].strip() == "{"), None)
    if brace is None:
        raise ValueError("D2.qml: no encontré la llave de 'Mapping': no lo toco")
    lines[brace + 1:brace + 1] = [MODULE_LINE, ""]
    lines[anchor + 1:anchor + 1] = [IMPORT_LINE]
    return nl.join(lines)


def unpatch_d2(text: str) -> str:
    """Quita lo que agrega patch_d2."""
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(nl)
    out: List[str] = []
    skip_blank = False
    for ln in lines:
        if ln == IMPORT_LINE:
            continue
        if ln == MODULE_LINE:
            skip_blank = True
            continue
        if skip_blank and ln == "":
            skip_blank = False
            continue
        skip_blank = False
        out.append(ln)
    return nl.join(out)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def log(msg: str) -> None:
    print(msg)
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")


def writable(folder: Path) -> bool:
    probe = folder / ".traktor_ml_write_test"
    try:
        probe.write_bytes(b"")
        probe.unlink()
        return True
    except OSError:
        return False


def relaunch_as_admin(argv: List[str]) -> bool:
    """Vuelve a abrir este script como administrador (aviso de UAC). True si Windows lo lanzó."""
    if sys.platform != "win32":
        return False
    import ctypes
    params = subprocess.list2cmdline([str(Path(__file__).resolve())] + argv + ["--pause"])
    rc = ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, params, str(REPO_ROOT), 1)
    return rc > 32


def api_files() -> List[Path]:
    return sorted(p for p in VENDOR_API.iterdir() if p.is_file())


def status(d2: Path) -> bool:
    qml = d2 / "D2.qml"
    patched = qml.exists() and "ApiModule" in qml.read_text(encoding="utf-8-sig")
    same = [(d2 / "Api" / p.name).exists() and sha256(d2 / "Api" / p.name) == sha256(p) for p in api_files()]
    print(f"[INFO] {qml}: {'con ApiModule' if patched else 'sin ApiModule'} (sha256 {sha256(qml)[:12]})")
    print(f"[INFO] {d2 / 'Api'}: {sum(same)} de {len(same)} archivos iguales a los del repo")
    return patched and all(same)


def install(d2: Path) -> None:
    qml = d2 / "D2.qml"
    text = qml.read_bytes().decode("utf-8")
    new = patch_d2(text)
    if new == text and status(d2):
        log(f"[OK] Ya estaba instalado en {d2}: no cambié nada")
        return
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = BACKUP_ROOT / f"qml_CSI_D2_{stamp}"
    shutil.copytree(d2, backup)
    log(f"[OK] Copia de {d2} en {backup} (D2.qml sha256 {sha256(qml)})")
    (d2 / "Api").mkdir(exist_ok=True)
    for p in api_files():
        shutil.copy2(p, d2 / "Api" / p.name)
    if new != text:
        tmp = qml.with_name("D2.qml.tmp")
        tmp.write_bytes(new.encode("utf-8"))
        os.replace(tmp, qml)
    ok = status(d2)
    log(f"[{'OK' if ok else 'ERROR'}] Instalado en {d2}" + ("" if ok else ": revisar"))


def uninstall(d2: Path) -> None:
    qml = d2 / "D2.qml"
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = BACKUP_ROOT / f"qml_CSI_D2_{stamp}_antes_de_desinstalar"
    shutil.copytree(d2, backup)
    text = qml.read_bytes().decode("utf-8")
    new = unpatch_d2(text)
    if new != text:
        tmp = qml.with_name("D2.qml.tmp")
        tmp.write_bytes(new.encode("utf-8"))
        os.replace(tmp, qml)
    for p in api_files():
        (d2 / "Api" / p.name).unlink(missing_ok=True)
    if (d2 / "Api").exists() and not any((d2 / "Api").iterdir()):
        (d2 / "Api").rmdir()
    log(f"[OK] Desinstalado de {d2} (copia previa en {backup}; D2.qml sha256 {sha256(qml)})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Instala traktor-api-client en Traktor")
    ap.add_argument("--d2", default=None, help="Carpeta qml/CSI/D2 de Traktor (por defecto la de Program Files)")
    ap.add_argument("--status", action="store_true", help="Solo mostrar si está instalado")
    ap.add_argument("--uninstall", action="store_true", help="Quitar el mod y dejar el D2.qml original")
    ap.add_argument("--pause", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    d2 = Path(args.d2) if args.d2 else default_d2()
    try:
        if not (d2 / "D2.qml").exists():
            print(f"[ERROR] No existe {d2 / 'D2.qml'}")
            return 1
        if args.status:
            return 0 if status(d2) else 2
        if traktor_running():
            print("[ERROR] Traktor está abierto: cerrarlo y volver a correr esto")
            return 1
        if not writable(d2):
            if args.pause:
                log(f"[ERROR] Sin permiso de escritura en {d2} aun como administrador")
                return 1
            argv = [a for a in sys.argv[1:]]
            if relaunch_as_admin(argv):
                print("[INFO] Hace falta permiso de administrador: Windows va a preguntar. "
                      f"El resultado queda en {LOG_PATH}")
                return 0
            print(f"[ERROR] Sin permiso para escribir en {d2}: Windows no dio el permiso de administrador "
                  "(aviso de UAC rechazado o sin respuesta). No cambié nada; volver a correrlo y aceptar el aviso")
            return 1
        uninstall(d2) if args.uninstall else install(d2)
        return 0
    except Exception as exc:  # en la ventana de administrador, que se lea el error antes de cerrarse
        log(f"[ERROR] {type(exc).__name__}: {exc}")
        return 1
    finally:
        if args.pause:
            input("Enter para cerrar...")


if __name__ == "__main__":
    sys.exit(main())
