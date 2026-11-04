"""
PURPOSE: Lectura de la colección de Traktor (collection.nml) sin dependencias fuera de la biblioteca
         estándar: dónde está, si Traktor está abierto, la ruta de cada entrada y si tiene cues propios.
         La usan traktor_collection.py (que escribe la colección) y las herramientas en vivo de
         tools/traktor_live/, que tienen que correr también en un .venv mínimo.
CHANGELOG:
  - 2026-11-04: Creación: find_collection, traktor_running, entry_key, entry_path y has_own_cues salen
                de src/v4/pipeline/traktor_collection.py, que importaba pandas por usb_package.
"""
from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional


def find_collection() -> Optional[Path]:
    """collection.nml más reciente en Documentos/Native Instruments/Traktor*."""
    base = Path(os.path.expanduser("~")) / "Documents" / "Native Instruments"
    found = sorted((d / "collection.nml" for d in base.glob("Traktor*") if (d / "collection.nml").exists()),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return found[0] if found else None


def traktor_running() -> bool:
    if sys.platform != "win32":
        return False
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq Traktor*"], capture_output=True, text=True).stdout
    return "Traktor" in out


def entry_key(e: ET.Element) -> str:
    loc = e.find("LOCATION")
    return loc.get("VOLUME", "") + loc.get("DIR", "") + loc.get("FILE", "")


def entry_path(e: ET.Element) -> Path:
    loc = e.find("LOCATION")
    return Path(loc.get("VOLUME", "") + loc.get("DIR", "").replace("/:", "\\") + loc.get("FILE", ""))


def has_own_cues(e: ET.Element) -> bool:
    """¿Tiene cues o loops puestos por el usuario? Traktor crea solos la grilla (TYPE 4), un marcador
    de carga (TYPE 3) y un hotcue en el primer beat de la grilla ("AutoGrid" o "n.n."): no cuentan."""
    cues = e.findall("CUE_V2")
    grid = [float(c.get("START", "0")) for c in cues if c.get("TYPE") == "4"]
    return any(c.get("TYPE") in ("0", "1", "2", "5")
               and not (grid and abs(float(c.get("START", "0")) - grid[0]) < 0.01) for c in cues)
