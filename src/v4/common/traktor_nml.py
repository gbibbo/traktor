"""
PURPOSE: Lectura de la colección de Traktor (collection.nml) sin dependencias fuera de la biblioteca
         estándar: dónde está, si Traktor está abierto, la ruta de cada entrada y si tiene cues propios.
         La usan traktor_collection.py (que escribe la colección) y las herramientas en vivo de
         tools/traktor_live/, que tienen que correr también en un .venv mínimo.
CHANGELOG:
  - 2026-11-04: Creación: find_collection, traktor_running, entry_key, entry_path y has_own_cues salen
                de src/v4/pipeline/traktor_collection.py, que importaba pandas por usb_package.
  - 2026-11-05: macOS: traktor_running mira los procesos con ps, y entry_path arma las rutas de Mac (en
                el .nml, VOLUME es el nombre del disco: el de arranque es "/", los otros /Volumes/<nombre>).
  - 2026-11-06: En Windows, traktor_running lee los procesos por la API (sin tasklist, que desde el
                servicio invisible de las sugerencias abría una consola cada vez).
"""
from __future__ import annotations

import functools
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

DRIVE_RE = re.compile(r"[A-Za-z]:")
DEFAULT_MAC_BOOT_VOLUME = "Macintosh HD"


def find_collection() -> Optional[Path]:
    """collection.nml más reciente en Documentos/Native Instruments/Traktor* (igual en Windows y Mac)."""
    base = Path(os.path.expanduser("~")) / "Documents" / "Native Instruments"
    found = sorted((d / "collection.nml" for d in base.glob("Traktor*") if (d / "collection.nml").exists()),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return found[0] if found else None


def _windows_process_names() -> list:
    """Nombres de los ejecutables en marcha, por la API de Windows (CreateToolhelp32Snapshot): sin abrir
    un proceso aparte, que desde un servicio invisible haría parpadear una consola."""
    import ctypes
    from ctypes import wintypes

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                    ("szExeFile", wintypes.WCHAR * 260)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)                  # TH32CS_SNAPPROCESS
    if snap in (None, wintypes.HANDLE(-1).value):
        return []
    names = []
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        ok = k32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            names.append(entry.szExeFile)
            ok = k32.Process32NextW(snap, ctypes.byref(entry))
    finally:
        k32.CloseHandle(snap)
    return names


def traktor_running() -> bool:
    """¿Hay un Traktor abierto? Windows: la lista de procesos de la API; macOS: nombres de proceso de ps
    (sin la línea de comandos, que en este repo dice "traktor" en cada script)."""
    if sys.platform == "win32":
        return any(n.lower().startswith("traktor") for n in _windows_process_names())
    if sys.platform == "darwin":
        out = subprocess.run(["ps", "-axco", "comm="], capture_output=True, text=True).stdout
        return any(line.strip().startswith("Traktor") for line in out.splitlines())
    return False


@functools.lru_cache(maxsize=1)
def mac_boot_volume() -> str:
    """Nombre del disco de arranque del Mac ("Macintosh HD" salvo que se haya renombrado): la entrada
    de /Volumes que apunta a "/"."""
    try:
        for d in Path("/Volumes").iterdir():
            if os.path.realpath(d) == "/":
                return d.name
    except OSError:
        pass
    return DEFAULT_MAC_BOOT_VOLUME


def entry_key(e: ET.Element) -> str:
    loc = e.find("LOCATION")
    return loc.get("VOLUME", "") + loc.get("DIR", "") + loc.get("FILE", "")


def location_str(volume: str, dir_: str, file: str, boot_volume: Optional[str] = None) -> str:
    """Ruta de un LOCATION de Traktor. Windows: VOLUME "C:" y DIR "/:Users/:x/:" -> C:\\Users\\x\\<file>.
    Mac: VOLUME es el nombre del disco; el de arranque cuelga de "/", los demás de /Volumes."""
    if DRIVE_RE.fullmatch(volume):
        return volume + dir_.replace("/:", "\\") + file
    rel = dir_.replace("/:", "/") + file
    if not rel.startswith("/"):
        rel = "/" + rel
    if not volume:
        return rel
    boot = boot_volume if boot_volume is not None else (mac_boot_volume() if sys.platform == "darwin"
                                                        else DEFAULT_MAC_BOOT_VOLUME)
    return rel if volume == boot else f"/Volumes/{volume}{rel}"


def entry_path(e: ET.Element) -> Path:
    loc = e.find("LOCATION")
    return Path(location_str(loc.get("VOLUME", ""), loc.get("DIR", ""), loc.get("FILE", "")))


def has_own_cues(e: ET.Element) -> bool:
    """¿Tiene cues o loops puestos por el usuario? Traktor crea solos la grilla (TYPE 4), un marcador
    de carga (TYPE 3) y un hotcue en el primer beat de la grilla ("AutoGrid" o "n.n."): no cuentan."""
    cues = e.findall("CUE_V2")
    grid = [float(c.get("START", "0")) for c in cues if c.get("TYPE") == "4"]
    return any(c.get("TYPE") in ("0", "1", "2", "5")
               and not (grid and abs(float(c.get("START", "0")) - grid[0]) < 0.01) for c in cues)
