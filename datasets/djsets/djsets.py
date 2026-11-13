"""
PURPOSE: Lector del dataset DJSets (sets de 1001Tracklists + audio de sus temas). Un solo archivo, solo biblioteca
         estándar de Python 3.8+, para copiarlo junto al dataset y usarlo sin este repositorio:
           ds = DJSets("ruta/a/djsets_v1")
           ds.sets(), ds.set_tracks("14y24gtk"), ds.tracks(), ds.track("25p3fc35")
           ds.audio_bytes("25p3fc35")  -> bytes del MP3/M4A (de la carpeta desempaquetada o del .tar)
           ds.pairs()                  -> temas consecutivos de cada set (los dos con audio, si se pide)
         Línea de comandos:
           python djsets.py info     <raíz>
           python djsets.py validate <raíz> [--full]   (md5 de metadatos y shards; --full: cada audio)
           python djsets.py unpack   <raíz> [--source beatport|youtube]  (audio_files/, salta lo ya hecho)
         El formato (carpetas, columnas, de dónde sale cada dato) está en el README.md del dataset.
CHANGELOG:
  - 2026-11-13: Creación (v1).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tarfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

__version__ = "1.0"

METADATA = "metadata"
FILES = {"sets": "sets.csv", "set_tracks": "set_tracks.csv", "tracks": "tracks.csv", "shards": "shards.csv",
         "split": "split_by_dj.csv"}
CHECKSUMS = "checksums.md5"
INFO = "dataset_info.json"
BLOQUE = 1 << 20


def _leer_csv(ruta: Path) -> List[Dict[str, str]]:
    with Path(ruta).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def md5_archivo(ruta: Path) -> str:
    h = hashlib.md5()
    with Path(ruta).open("rb") as handle:
        for bloque in iter(lambda: handle.read(BLOQUE), b""):
            h.update(bloque)
    return h.hexdigest()


class DJSets:
    """Acceso al dataset. Las filas son dicts de texto (como en el CSV); los vacíos son ""."""

    def __init__(self, root) -> None:
        self.root = Path(root)
        if not (self.root / METADATA / FILES["tracks"]).exists():
            raise FileNotFoundError(f"{self.root} no parece un dataset DJSets: falta {METADATA}/{FILES['tracks']}")
        self._cache: Dict[str, List[Dict[str, str]]] = {}
        self._por_id: Optional[Dict[str, Dict[str, str]]] = None
        self._miembros: Dict[str, Dict[str, tarfile.TarInfo]] = {}

    # ------------------------------------------------------------ tablas

    def _tabla(self, nombre: str) -> List[Dict[str, str]]:
        if nombre not in self._cache:
            self._cache[nombre] = _leer_csv(self.root / METADATA / FILES[nombre])
        return self._cache[nombre]

    def info(self) -> dict:
        return json.loads((self.root / INFO).read_text(encoding="utf-8"))

    def sets(self) -> List[Dict[str, str]]:
        return self._tabla("sets")

    def tracks(self, with_audio: Optional[bool] = None) -> List[Dict[str, str]]:
        filas = self._tabla("tracks")
        if with_audio is None:
            return filas
        return [f for f in filas if (f["has_audio"] == "1") == with_audio]

    def track(self, track_id: str) -> Dict[str, str]:
        if self._por_id is None:
            self._por_id = {f["track_id"]: f for f in self._tabla("tracks")}
        return self._por_id[track_id]

    def set_tracks(self, set_id: Optional[str] = None) -> List[Dict[str, str]]:
        filas = self._tabla("set_tracks")
        return filas if set_id is None else [f for f in filas if f["set_id"] == set_id]

    def shards(self) -> List[Dict[str, str]]:
        return self._tabla("shards")

    def split(self) -> Dict[str, str]:
        """set_id -> train / validation / test (propuesta por DJ, ver README)."""
        return {f["set_id"]: f["split"] for f in self._tabla("split")}

    def pairs(self, with_audio: bool = False) -> List[Tuple[str, str, str]]:
        """(set_id, tema, tema siguiente) para filas consecutivas del set, las dos con track_id. Una fila tocada encima
        de la anterior («w/», with_previous=1) no forma par ni corta la cadena: se salta. Una fila sin track_id (sin
        identificar, «ID - ID») sí corta: no se sabe qué sonó entre los dos."""
        con_audio = {f["track_id"] for f in self.tracks(with_audio=True)} if with_audio else None
        salida, anterior, set_actual = [], None, None
        for fila in self._tabla("set_tracks"):
            if fila["set_id"] != set_actual:
                set_actual, anterior = fila["set_id"], None
            if fila["with_previous"] == "1":
                continue
            actual = fila["track_id"] or None
            if anterior and actual and (con_audio is None or (anterior in con_audio and actual in con_audio)):
                salida.append((set_actual, anterior, actual))
            anterior = actual
        return salida

    # ------------------------------------------------------------ audio

    def audio_path(self, track_id: str) -> Optional[Path]:
        """Archivo de audio desempaquetado (audio_files/...), si existe."""
        fila = self.track(track_id)
        if fila["has_audio"] != "1":
            return None
        ruta = self.root / fila["audio_file"]
        return ruta if ruta.exists() else None

    def _miembro(self, shard: str, miembro: str) -> tarfile.TarInfo:
        if shard not in self._miembros:
            with tarfile.open(self.root / shard) as paquete:
                self._miembros[shard] = {m.name: m for m in paquete.getmembers()}
        return self._miembros[shard][miembro]

    def audio_bytes(self, track_id: str) -> bytes:
        fila = self.track(track_id)
        if fila["has_audio"] != "1":
            raise KeyError(f"{track_id} no tiene audio ({fila['audio_status']})")
        ruta = self.audio_path(track_id)
        if ruta is not None:
            return ruta.read_bytes()
        info = self._miembro(fila["audio_shard"], fila["audio_member"])
        with (self.root / fila["audio_shard"]).open("rb") as handle:
            handle.seek(info.offset_data)
            return handle.read(info.size)

    def iter_audio(self, source: Optional[str] = None, skip_missing: bool = False) -> Iterator[Tuple[Dict[str, str], bytes]]:
        """(fila del tema, bytes) leyendo cada shard de corrido: lo más rápido para recorrer todo. Un audio compartido
        por varios track_id se entrega una vez por track_id. skip_missing=True salta los shards que no están (copia
        parcial) en vez de fallar."""
        por_shard: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
        for fila in self.tracks(with_audio=True):
            if source and fila["audio_source"] != source:
                continue
            por_shard.setdefault(fila["audio_shard"], {}).setdefault(fila["audio_member"], []).append(fila)
        for shard in sorted(por_shard):
            if skip_missing and not (self.root / shard).exists():
                continue
            with tarfile.open(self.root / shard) as paquete:
                for info in paquete:
                    filas = por_shard[shard].get(info.name)
                    if not filas:
                        continue
                    datos = paquete.extractfile(info).read()
                    for fila in filas:
                        yield fila, datos

    def unpack(self, source: Optional[str] = None, skip_missing: bool = False, log=print) -> int:
        """Escribe cada audio en su audio_file (audio_files/<fuente>/<NN>/<archivo>). Salta los que ya están con el
        tamaño correcto; escribe a .part y renombra, así una corrida cortada se retoma. Devuelve cuántos escribió."""
        escritos, vistos = 0, set()
        for fila, datos in self.iter_audio(source, skip_missing):
            destino = self.root / fila["audio_file"]
            if destino in vistos:
                continue
            vistos.add(destino)
            if destino.exists() and destino.stat().st_size == len(datos):
                continue
            destino.parent.mkdir(parents=True, exist_ok=True)
            parcial = destino.with_name(destino.name + ".part")
            parcial.write_bytes(datos)
            parcial.replace(destino)
            escritos += 1
            if escritos % 1000 == 0:
                log(f"  {escritos} escritos")
        return escritos

    # ------------------------------------------------------------ validación

    def validate(self, full: bool = False, log=print) -> List[str]:
        """Problemas encontrados (lista vacía = todo bien). Siempre: cada archivo de checksums.md5 existe y su md5
        coincide, y cada tema con audio apunta a un shard listado. full=True: además lee cada audio y compara su md5
        con audio_md5 (unos 62 GB de lectura)."""
        problemas = []
        lista = self.root / CHECKSUMS
        for linea in lista.read_text(encoding="utf-8").splitlines():
            if not linea.strip():
                continue
            esperado, nombre = linea.split(None, 1)
            nombre = nombre.lstrip("*")
            ruta = self.root / nombre
            if not ruta.exists():
                problemas.append(f"falta {nombre}")
            elif md5_archivo(ruta) != esperado:
                problemas.append(f"md5 distinto: {nombre}")
        shards = {f["shard"] for f in self.shards()}
        for fila in self.tracks(with_audio=True):
            if fila["audio_shard"] not in shards:
                problemas.append(f"{fila['track_id']}: shard desconocido {fila['audio_shard']}")
        if full:
            vistos = 0
            for fila, datos in self.iter_audio():
                vistos += 1
                if hashlib.md5(datos).hexdigest() != fila["audio_md5"]:
                    problemas.append(f"{fila['track_id']}: md5 del audio distinto")
            esperados = len(self.tracks(with_audio=True))
            if vistos != esperados:
                problemas.append(f"audio leído de {vistos} temas, se esperaban {esperados}")
        return problemas


# ---------------------------------------------------------------- línea de comandos


def main(argv: Optional[List[str]] = None) -> int:
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="Dataset DJSets: info, validación y desempaquetado")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for nombre in ("info", "validate", "unpack"):
        p = sub.add_parser(nombre)
        p.add_argument("root", type=Path)
        if nombre == "validate":
            p.add_argument("--full", action="store_true", help="leer y verificar cada audio")
        if nombre == "unpack":
            p.add_argument("--source", choices=("beatport", "youtube"))
            p.add_argument("--skip-missing", action="store_true", help="saltar los shards que no estén")
    args = parser.parse_args(argv)
    ds = DJSets(args.root)
    if args.cmd == "info":
        print(json.dumps(ds.info(), ensure_ascii=False, indent=1))
        return 0
    if args.cmd == "validate":
        problemas = ds.validate(full=args.full)
        for p in problemas[:50]:
            print(p)
        print("OK" if not problemas else f"{len(problemas)} problemas")
        return 0 if not problemas else 1
    print(f"escritos {ds.unpack(args.source, args.skip_missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
