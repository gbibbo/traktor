"""
PURPOSE: Caché de lo leído en los tags de cada archivo (read_tags, read_release_tags), guardada por ruta con el
         tamaño y la fecha de modificación del archivo: un archivo que no cambió no se vuelve a abrir. Escribir un
         tag cambia la fecha, así que lo editado (Beatport, Vocal, Traktor) se vuelve a leer. Pensada para los
         pasos que recorren toda la biblioteca en cada agregado: rehacer las playlists del pendrive
         (usb_package.refresh_from_org) y buscar temas repetidos (dedupe.candidates). Leer los tags de los 1760
         temas del pendrive llevaba unos 2 minutos por pasada (medido el 2026-11-07 sobre 150 temas).
CHANGELOG:
  - 2026-11-07: Creación.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict

FILE_NAME = "tag_cache.json"


def _jsonable(value: Any) -> Any:
    """NaN no es JSON: queda vacío (None), que es como read_tags informa un dato que falta."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


class TagCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.dirty = False
        try:
            self.data: Dict[str, Dict[str, Any]] = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}

    def read(self, file: Path, what: str, reader: Callable[[Path], Any]) -> Any:
        """reader(file), o lo guardado si el archivo tiene el mismo tamaño y la misma fecha."""
        try:
            st = Path(file).stat()
        except OSError:  # no está: el lector dice lo que corresponda (vacío), sin guardar nada
            return reader(Path(file))
        key = str(Path(file))
        entry = self.data.get(key)
        if entry is None or entry.get("size") != st.st_size or entry.get("mtime_ns") != st.st_mtime_ns:
            entry = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
            self.data[key] = entry
        if what not in entry:
            entry[what] = _jsonable(reader(Path(file)))
            self.dirty = True
        value = entry[what]
        return dict(value) if isinstance(value, dict) else value

    def save(self) -> None:
        if not self.dirty:
            return
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)
        self.dirty = False

