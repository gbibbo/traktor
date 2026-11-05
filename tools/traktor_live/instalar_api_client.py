"""
PURPOSE: Instala, revisa o desinstala traktor-api-client (tools/traktor_live/traktor_qml/) en la carpeta
         de programa de Traktor: copia Api/ a qml/CSI/D2/ y agrega dos líneas al D2.qml de la versión
         instalada (import "./Api" y ApiModule {}), en vez de reemplazarlo por el D2.qml del mod, que es
         de otra versión. Antes guarda una copia de la carpeta D2 en artifacts/v4/traktor_backups/. Con
         Traktor abierto no hace nada. Si hace falta permiso de administrador, se vuelve a correr
         pidiéndolo: Windows muestra el aviso de UAC y macOS pide la contraseña.
         Dónde busca Traktor: Windows, %ProgramFiles%\\Native Instruments\\Traktor*\\Resources64\\qml;
         macOS, /Applications/Native Instruments/Traktor*/Traktor*.app/Contents/Resources/qml.
CHANGELOG:
  - 2026-11-04: Creación inicial (Traktor 4.4.1).
  - 2026-11-04: --sonda agrega ApiBrowserProbe.qml (traktor_qml/Screens/) al final de
                Screens/S8/Views/Screen.qml, la pantalla del D2, para ver si su navegador existe con un D2
                virtual y sigue al árbol de Traktor; --quitar-sonda y --uninstall la sacan.
  - 2026-11-05: macOS (carpeta dentro de la app y permiso con osascript; las copias vuelven a ser del
                usuario) y cualquier Traktor instalado (Pro 3 o 4): si hay varios, el más nuevo o --d2.
  - 2026-11-05: is_ready(): chequeo sin imprimir, para que sugerencias_vivo.py avise si falta la sonda.
                Los QML se comparan sin importar CRLF/LF: desde un clon nuevo de Windows (CRLF), la sonda
                instalada desde otra copia (LF) parecía desactualizada. Respaldos en carpetas únicas
                (dos en el mismo segundo fallaban).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from src.v4.common.traktor_nml import traktor_running  # noqa: E402

VENDOR_API = Path(__file__).resolve().parent / "traktor_qml" / "Api"
VENDOR_PROBE = Path(__file__).resolve().parent / "traktor_qml" / "Screens" / "ApiBrowserProbe.qml"
BACKUP_ROOT = REPO_ROOT / "artifacts" / "v4" / "traktor_backups"
LOG_PATH = BACKUP_ROOT / "instalar_api_client.log"
IMPORT_LINE = 'import "./Api"'
MODULE_LINE = "  ApiModule {}"
ANCHOR_IMPORT = 'import "../Common/Settings"'
PROBE_LINE = '  ApiBrowserProbe { screenSide: screen.isLeftScreen ? "left" : "right" }'
MAC_APPS = Path("/Applications/Native Instruments")
OWNER: Optional[Tuple[int, int]] = None      # macOS: dueño de las copias cuando esto corre como root


def qml_dirs(platform: Optional[str] = None, root: Optional[Path] = None) -> List[Path]:
    """Carpetas qml de cada Traktor instalado que tenga CSI/D2/D2.qml, de la versión más nueva a la más
    vieja. root reemplaza Program Files o /Applications/Native Instruments (para los tests)."""
    system = platform or sys.platform
    if system == "win32":
        base = root or Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Native Instruments"
        found = list(base.glob("Traktor*/Resources64/qml"))
    elif system == "darwin":
        base = root or MAC_APPS
        found = list(base.glob("Traktor*/Traktor*.app/Contents/Resources/qml"))
    else:
        return []

    def version(p: Path) -> List[int]:             # "Traktor Pro 4" -> [4]
        return [int(n) for n in re.findall(r"\d+", p.relative_to(base).parts[0])]

    found = [p for p in found if (p / "CSI" / "D2" / "D2.qml").exists()]
    return sorted(found, key=version, reverse=True)


def default_d2(platform: Optional[str] = None, root: Optional[Path] = None) -> Optional[Path]:
    dirs = qml_dirs(platform, root)
    return dirs[0] / "CSI" / "D2" if dirs else None


def give_back(path: Path) -> None:
    """macOS como root: lo que escribe este script en el repo vuelve a ser del usuario."""
    if OWNER is None or not hasattr(os, "chown") or not path.exists():
        return
    for p in [path] + (list(path.rglob("*")) if path.is_dir() else []):
        os.chown(p, *OWNER)


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


def patch_screen(text: str) -> str:
    """Screen.qml con la sonda como último hijo del Item raíz (antes de su llave final)."""
    if "ApiBrowserProbe" in text:
        return text
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(nl)
    closing = [i for i, ln in enumerate(lines) if ln.rstrip() == "}"]
    if not closing:
        raise ValueError("Screen.qml no tiene la forma esperada: no lo toco")
    lines.insert(closing[-1], PROBE_LINE)
    return nl.join(lines)


def unpatch_screen(text: str) -> str:
    nl = "\r\n" if "\r\n" in text else "\n"
    return nl.join(ln for ln in text.split(nl) if ln != PROBE_LINE)


def screen_qml(d2: Path) -> Path:
    """Pantalla del D2 (la misma vista que el S8): qml/Screens/S8/Views/Screen.qml."""
    return d2.parent.parent / "Screens" / "S8" / "Views" / "Screen.qml"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def backup_dir(name: str) -> Path:
    """Carpeta de respaldo que todavía no existe: <name>, o <name>_2, _3... (dos respaldos en el mismo
    segundo chocaban)."""
    path, n = BACKUP_ROOT / name, 1
    while path.exists():
        n += 1
        path = BACKUP_ROOT / f"{name}_{n}"
    return path


def same_text(a: Path, b: Path) -> bool:
    """Mismo contenido sin importar los finales de línea: un clon en Windows trae CRLF y uno en macOS
    LF, y la sonda instalada desde uno no tiene que parecer otra vista desde el otro."""
    return a.read_bytes().replace(b"\r\n", b"\n") == b.read_bytes().replace(b"\r\n", b"\n")


def log(msg: str) -> None:
    print(msg, flush=True)
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")
    give_back(LOG_PATH)


def writable(folder: Path) -> bool:
    probe = folder / ".traktor_ml_write_test"
    try:
        probe.write_bytes(b"")
        probe.unlink()
        return True
    except OSError:
        return False


def applescript_string(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def mac_admin_command(argv: List[str], uid: int, gid: int) -> List[str]:
    """osascript que vuelve a correr este script como administrador (macOS pide la contraseña)."""
    import shlex
    cmd = shlex.join([sys.executable, str(Path(__file__).resolve())] + argv + ["--owner", f"{uid}:{gid}"])
    return ["osascript", "-e", f"do shell script {applescript_string(cmd)} with administrator privileges"]


def relaunch_as_admin(argv: List[str]) -> bool:
    """Vuelve a correr este script como administrador. Windows: aviso de UAC en una ventana nueva (True
    si Windows la lanzó). macOS: contraseña con osascript, en esta misma ventana (True si terminó bien)."""
    if sys.platform == "win32":
        import ctypes
        params = subprocess.list2cmdline([str(Path(__file__).resolve())] + argv + ["--pause"])
        rc = ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, params, str(REPO_ROOT), 1)
        return rc > 32
    if sys.platform == "darwin":
        res = subprocess.run(mac_admin_command(argv, os.getuid(), os.getgid()), capture_output=True, text=True)
        print(res.stdout.rstrip())
        if res.returncode != 0 and res.stderr.strip():
            print(res.stderr.rstrip())
        return res.returncode == 0
    return False


def api_files() -> List[Path]:
    return sorted(p for p in VENDOR_API.iterdir() if p.is_file())


def status(d2: Path) -> bool:
    qml = d2 / "D2.qml"
    patched = qml.exists() and "ApiModule" in qml.read_text(encoding="utf-8-sig")
    same = [(d2 / "Api" / p.name).exists() and same_text(d2 / "Api" / p.name, p) for p in api_files()]
    print(f"[INFO] {qml}: {'con ApiModule' if patched else 'sin ApiModule'} (sha256 {sha256(qml)[:12]})")
    print(f"[INFO] {d2 / 'Api'}: {sum(same)} de {len(same)} archivos iguales a los del repo")
    return patched and all(same)


def probe_status(d2: Path) -> bool:
    sq = screen_qml(d2)
    patched = sq.exists() and "ApiBrowserProbe" in sq.read_text(encoding="utf-8-sig")
    probe = sq.parent / VENDOR_PROBE.name
    same = probe.exists() and same_text(probe, VENDOR_PROBE)
    print(f"[INFO] {sq}: {'con la sonda' if patched else 'sin la sonda'}; {probe.name} "
          f"{'igual al del repo' if same else 'ausente o distinto'}")
    return patched and same


def is_ready(d2: Path) -> bool:
    """¿Están el módulo y la sonda instalados e iguales a los del repo? Sin imprimir (para
    sugerencias_vivo.py, que avisa si hay que correr esto)."""
    sq = screen_qml(d2)
    probe = sq.parent / VENDOR_PROBE.name
    try:
        return ("ApiModule" in (d2 / "D2.qml").read_text(encoding="utf-8-sig")
                and all((d2 / "Api" / p.name).exists() and same_text(d2 / "Api" / p.name, p)
                        for p in api_files())
                and "ApiBrowserProbe" in sq.read_text(encoding="utf-8-sig")
                and probe.exists() and same_text(probe, VENDOR_PROBE))
    except OSError:
        return False


def _replace_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(text.encode("utf-8"))
    os.replace(tmp, path)


def install_probe(d2: Path) -> None:
    sq = screen_qml(d2)
    text = sq.read_bytes().decode("utf-8")
    new = patch_screen(text)
    if new == text and probe_status(d2):
        log(f"[OK] La sonda ya estaba en {sq}: no cambié nada")
        return
    backup = backup_dir(f"qml_Screens_S8_Views_{dt.datetime.now():%Y%m%d_%H%M%S}")
    backup.mkdir(parents=True)
    shutil.copy2(sq, backup / sq.name)
    give_back(backup)
    log(f"[OK] Copia de {sq} en {backup} (sha256 {sha256(sq)})")
    shutil.copy2(VENDOR_PROBE, sq.parent / VENDOR_PROBE.name)
    if new != text:
        _replace_text(sq, new)
    ok = probe_status(d2)
    log(f"[{'OK' if ok else 'ERROR'}] Sonda instalada en {sq.parent}" + ("" if ok else ": revisar"))


def uninstall_probe(d2: Path) -> None:
    sq = screen_qml(d2)
    probe = sq.parent / VENDOR_PROBE.name
    text = sq.read_bytes().decode("utf-8")
    new = unpatch_screen(text)
    if new == text and not probe.exists():
        return
    backup = backup_dir(f"qml_Screens_S8_Views_{dt.datetime.now():%Y%m%d_%H%M%S}_antes_de_quitar_sonda")
    backup.mkdir(parents=True)
    shutil.copy2(sq, backup / sq.name)
    give_back(backup)
    if new != text:
        _replace_text(sq, new)
    probe.unlink(missing_ok=True)
    log(f"[OK] Sonda quitada de {sq.parent} (copia previa en {backup}; Screen.qml sha256 {sha256(sq)})")


def install(d2: Path) -> None:
    qml = d2 / "D2.qml"
    text = qml.read_bytes().decode("utf-8")
    new = patch_d2(text)
    if new == text and status(d2):
        log(f"[OK] Ya estaba instalado en {d2}: no cambié nada")
        return
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = backup_dir(f"qml_CSI_D2_{stamp}")
    shutil.copytree(d2, backup)
    give_back(backup)
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
    backup = backup_dir(f"qml_CSI_D2_{stamp}_antes_de_desinstalar")
    shutil.copytree(d2, backup)
    give_back(backup)
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
    global OWNER
    ap = argparse.ArgumentParser(description="Instala traktor-api-client en Traktor (Windows y macOS)")
    ap.add_argument("--d2", default=None,
                    help="Carpeta qml/CSI/D2 de Traktor (por defecto la del Traktor más nuevo instalado)")
    ap.add_argument("--status", action="store_true", help="Solo mostrar si está instalado")
    ap.add_argument("--uninstall", action="store_true", help="Quitar el mod y la sonda y dejar los QML originales")
    ap.add_argument("--sonda", action="store_true", help="Instalar también la sonda del navegador del D2")
    ap.add_argument("--quitar-sonda", action="store_true", help="Quitar solo la sonda")
    ap.add_argument("--pause", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--owner", default=None, help=argparse.SUPPRESS)        # uid:gid, macOS como root
    args = ap.parse_args()
    if args.owner:
        uid, gid = (int(x) for x in args.owner.split(":"))
        OWNER = (uid, gid)
    d2 = Path(args.d2) if args.d2 else default_d2()
    try:
        if d2 is None:
            print("[ERROR] No encontré Traktor instalado (busqué en "
                  f"{'Program Files' if sys.platform == 'win32' else MAC_APPS}). Indicar la carpeta con --d2")
            return 1
        if not (d2 / "D2.qml").exists():
            print(f"[ERROR] No existe {d2 / 'D2.qml'}")
            return 1
        print(f"[INFO] Traktor: {d2}")
        if args.status:
            ok = status(d2)
            probe_status(d2)
            return 0 if ok else 2
        if traktor_running():
            log("[ERROR] Traktor está abierto: no cambié nada. Cerrarlo y volver a correr esto")
            return 1
        if not (writable(d2) and writable(screen_qml(d2).parent)):
            if args.pause or args.owner:
                log(f"[ERROR] Sin permiso de escritura en {d2} aun como administrador")
                return 1
            argv = [a for a in sys.argv[1:]]
            if sys.platform == "darwin":
                print("[INFO] Hace falta permiso de administrador: macOS va a pedir la contraseña")
            if relaunch_as_admin(argv):
                if sys.platform == "win32":
                    log("[INFO] Pedí permiso de administrador: el resultado sale en la ventana nueva y en "
                        f"{LOG_PATH}")
                return 0
            log("[ERROR] El sistema no dio el permiso de administrador (aviso rechazado o sin respuesta): "
                "no cambié nada. Volver a correrlo y aceptar el aviso")
            return 1
        if args.uninstall:
            uninstall_probe(d2)
            uninstall(d2)
        elif args.quitar_sonda:
            uninstall_probe(d2)
        else:
            install(d2)
            if args.sonda:
                install_probe(d2)
        return 0
    except Exception as exc:  # en la ventana de administrador, que se lea el error antes de cerrarse
        log(f"[ERROR] {type(exc).__name__}: {exc}")
        return 1
    finally:
        if args.pause:
            input("Enter para cerrar...")


if __name__ == "__main__":
    sys.exit(main())
