"""
PURPOSE: D6, el audio de los temas de los sets: previews de Beatport de los temas de D5 (1001Tracklists), guardados
         en el Drive de Gabriel en erclm:traktor_ml/beatport_previews/<corrida>/, en tandas de hasta 500 previews
         por .tar (en el Drive cada archivo cuesta una llamada y la tanda subida sirve de marca de «hecho").
           elegir   -> <root>/beatport_previews/<corrida>/temas.csv: temas identificados distintos, con botón de
                       compra, de los sets bajados por tracklists_1001.py, repartidos entre los sets
           enlaces  -> beatport_id_1001: el enlace del botón de compra en 1001Tracklists (el mismo pedido que hace
                       el clic, uno por tema, 8 s entre pedidos, con la sesión de Gabriel y parada al primer
                       rechazo)
           buscar   -> beatport_id_busqueda y nivel: búsqueda por nombre en Beatport con match() de beatport.py
                       (A = ISRC, B = misma versión, C = otra versión, D = nada); el piloto la compara con enlaces
           bajar    -> por tanda: página del tema en Beatport (sample_url y metadatos), el MP3 del preview, un .tar
                       con manifest.json y la subida con open_remote().put (atómica, verificada por tamaño y MD5).
                       Salta las tandas que ya están en el remoto, así se retoma si se corta. Los MP3 pasan por una
                       carpeta temporal en D:, nunca en C:; con TRAKTOR_REMOTE=<carpeta local> no toca el Drive.
           resumen  -> acuerdo entre enlaces y búsqueda, peso por preview, tiempo por tema y respuestas de Beatport
         Reglas: docs/GPUQ.md y AGENTS.md («Compute»). Nada fuera de erclm:traktor_ml/. La corrida grande y el
         primer trabajo de gpuq de este tipo esperan el OK de Gabriel; también su decisión sobre el ai-train=no
         que declara Beatport.
CHANGELOG:
  - 2026-11-09: Creación (piloto de 100 temas).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import re
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from src.v4.common import beatport as bp  # noqa: E402

DEFAULT_ROOT = REPO_ROOT / "artifacts" / "v4" / "dj_sets"
CARPETA_REMOTA = "beatport_previews"
TANDA = 500
PAUSA_1001_MS = 8000
PAUSA_BEATPORT_S = 1.2
MIN_BYTES_MP3 = 20_000
COLUMNAS = ["clave", "nombre", "artista", "titulo", "id_numerico", "view_item", "set_id", "beatport_id_1001", "fuente_1001",
            "beatport_id_busqueda", "nivel_busqueda", "pedidos_busqueda"]


# ---------------------------------------------------------------- tabla de temas


def leer_tabla(ruta: Path) -> List[Dict[str, str]]:
    with Path(ruta).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def escribir_tabla(ruta: Path, filas: Sequence[Dict[str, str]]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(".tmp")
    with temporal.open("w", encoding="utf-8-sig", newline="") as handle:
        escritor = csv.DictWriter(handle, fieldnames=COLUMNAS, extrasaction="ignore")
        escritor.writeheader()
        escritor.writerows(filas)
    temporal.replace(ruta)


def separar_nombre(nombre: str) -> tuple:
    """'Artista & Otro - Título (Remix)' -> ('Artista & Otro', 'Título (Remix)')."""
    if " - " not in (nombre or ""):
        return "", (nombre or "").strip()
    artista, titulo = nombre.split(" - ", 1)
    return artista.strip(), titulo.strip()


def view_item_de(html: str) -> str:
    """Número del tracklist que el sitio manda en cada pedido de enlaces (viewItem)."""
    encontrado = re.search(r"viewItem:\s*'?(\d+)", html or "")
    return encontrado.group(1) if encontrado else ""


def elegir(sets: Sequence[Dict], n: int) -> List[Dict[str, str]]:
    """Temas identificados con botón de compra, sin repetir, tomados de a uno por set y por vuelta."""
    colas = []
    for dato in sets:
        cola = []
        for tema in dato["temas"]:
            if tema["junto_al_anterior"] or not tema["identificado"] or "affiliate" not in tema["fuentes"]:
                continue
            if not tema["id_numerico"] or not tema["id_texto"]:
                continue
            nombre = tema["nombre"] or tema["visible"]
            artista, titulo = separar_nombre(nombre)
            if not artista or not titulo:
                continue
            cola.append({"clave": tema["id_texto"], "nombre": nombre, "artista": artista, "titulo": titulo,
                         "id_numerico": tema["id_numerico"], "view_item": dato.get("view_item", ""), "set_id": dato["id"]})
        colas.append(cola)
    elegidos, vistos = [], set()
    while len(elegidos) < n and any(colas):
        for cola in colas:
            while cola:
                fila = cola.pop(0)
                if fila["clave"] not in vistos:
                    vistos.add(fila["clave"])
                    elegidos.append(fila)
                    break
            if len(elegidos) >= n:
                break
    return elegidos


# ---------------------------------------------------------------- 1001Tracklists


def url_enlace(fila: Dict[str, str]) -> str:
    """El mismo pedido que hace el sitio al pulsar el botón de compra de un tema."""
    return (f"/ajax/get_medialink.php?idObject=5&idItem={fila['id_numerico']}&idSource=affiliate&viewSource=1"
            f"&viewItem={fila['view_item']}&autoPlayOff=true&insertAfter=true")


def leer_enlace(respuesta: Dict) -> tuple:
    """(id de Beatport, fuente) de la respuesta de get_medialink.php; fuente '1' es Beatport."""
    datos = (respuesta or {}).get("data") or []
    if not (respuesta or {}).get("success") or not datos:
        return "", "sin_enlace"
    primero = datos[0]
    fuente = str(primero.get("source") or "")
    jugador = str(primero.get("playerId") or "")
    return (jugador if fuente == "1" and jugador.isdigit() else ""), (fuente or "sin_fuente")


PEDIR = """async (url) => {
  const r = await fetch(url, {credentials: 'same-origin', headers: {'X-Requested-With': 'XMLHttpRequest'}});
  return {status: r.status, text: await r.text()};
}"""


# ---------------------------------------------------------------- Beatport


def consulta_de(fila: Dict[str, str]) -> bp.TrackQuery:
    base, mezcla = bp.split_title_mix(fila["titulo"])
    return bp.TrackQuery(artists=fila["artista"], title=base, mix=mezcla)


def buscar_en_beatport(cliente: bp.BeatportClient, fila: Dict[str, str]) -> tuple:
    """(id, nivel, pedidos): prueba las consultas de TrackQuery hasta un nivel A o B."""
    consulta = consulta_de(fila)
    antes, candidatos, mejor = cliente.n_requests, [], bp.MatchResult(level="D")
    for texto in consulta.search_queries():
        candidatos.extend(cliente.search(texto))
        mejor = bp.match(consulta, candidatos)
        if mejor.level in ("A", "B"):
            break
    identificador = str(mejor.best["track_id"]) if mejor.best else ""
    return identificador, mejor.level, cliente.n_requests - antes


def es_mp3(contenido: bytes) -> bool:
    return len(contenido) >= MIN_BYTES_MP3 and (contenido[:3] == b"ID3" or (contenido[0] == 0xFF and contenido[1] & 0xE0 == 0xE0))


def beatport_id(fila: Dict[str, str]) -> str:
    """El enlace de 1001Tracklists manda; si no hay, la búsqueda solo si es la misma versión (A o B)."""
    if fila.get("beatport_id_1001"):
        return fila["beatport_id_1001"]
    if fila.get("nivel_busqueda") in ("A", "B"):
        return fila.get("beatport_id_busqueda", "")
    return ""


def nombre_tanda(k: int) -> str:
    return f"tanda_{k:05d}"


def tandas_pendientes(n_filas: int, hechas: Iterable[str], tam: int = TANDA) -> List[int]:
    hechas = set(hechas)
    return [k for k in range((n_filas + tam - 1) // tam) if f"{nombre_tanda(k)}.tar" not in hechas]


def armar_tanda(filas: Sequence[Dict[str, str]], detalles: Callable[[int], Optional[dict]], bajar_bytes: Callable[[str], bytes],
                carpeta: Path, nombre: str) -> tuple:
    """Baja los previews de una tanda a carpeta/ y arma <nombre>.tar con manifest.json. (ruta del tar, manifiesto)."""
    carpeta.mkdir(parents=True, exist_ok=True)
    entradas, archivos = [], []
    for fila in filas:
        entrada = {"clave_1001": fila["clave"], "nombre_1001": fila["nombre"], "beatport_id": beatport_id(fila)}
        if not entrada["beatport_id"]:
            entradas.append(dict(entrada, estado="sin_id"))
            continue
        t0 = time.monotonic()
        try:
            datos = detalles(int(entrada["beatport_id"]))
        except bp.BeatportBlocked:
            raise
        except Exception as exc:  # una página rota no tumba la tanda
            entradas.append(dict(entrada, estado=f"pagina_error: {type(exc).__name__}"))
            continue
        if not datos or not datos.get("sample_url"):
            entradas.append(dict(entrada, estado="sin_preview", **({"detalles": datos} if datos else {})))
            continue
        t1 = time.monotonic()
        contenido = bajar_bytes(datos["sample_url"])
        t2 = time.monotonic()
        if not es_mp3(contenido):
            entradas.append(dict(entrada, estado="mp3_invalido", bytes=len(contenido), detalles=datos))
            continue
        archivo = carpeta / f"{entrada['beatport_id']}.mp3"
        archivo.write_bytes(contenido)
        archivos.append(archivo)
        entradas.append(dict(entrada, estado="ok", archivo=archivo.name, bytes=len(contenido),
                             md5=hashlib.md5(contenido).hexdigest(), segundos_pagina=round(t1 - t0, 2),
                             segundos_mp3=round(t2 - t1, 2), detalles=datos))
    manifiesto = {"tanda": nombre, "creado": time.strftime("%Y-%m-%dT%H:%M:%S"), "temas": entradas,
                  "ok": sum(e["estado"] == "ok" for e in entradas)}
    (carpeta / "manifest.json").write_text(json.dumps(manifiesto, ensure_ascii=False, indent=1), encoding="utf-8")
    tar = carpeta.parent / f"{nombre}.tar"
    with tarfile.open(tar, "w") as paquete:
        paquete.add(carpeta / "manifest.json", arcname="manifest.json")
        for archivo in archivos:
            paquete.add(archivo, arcname=archivo.name)
    return tar, manifiesto


class Descargador:
    """MP3 de los previews con la misma pausa que el cliente de Beatport; dos rechazos seguidos cortan la corrida."""

    def __init__(self, pausa_s: float = PAUSA_BEATPORT_S):
        import requests

        self.sesion = requests.Session()
        self.sesion.headers.update({"User-Agent": bp.USER_AGENT})
        self.pausa_s, self._ultimo, self.estados = pausa_s, 0.0, Counter()

    def __call__(self, url: str) -> bytes:
        for intento in range(2):
            espera = self._ultimo + self.pausa_s + random.uniform(0, 0.4) - time.monotonic()
            if espera > 0:
                time.sleep(espera)
            self._ultimo = time.monotonic()
            respuesta = self.sesion.get(url, timeout=60)
            self.estados[respuesta.status_code] += 1
            if respuesta.status_code == 200:
                return respuesta.content
            if respuesta.status_code in (403, 429) and intento == 0:
                time.sleep(15)
                continue
            if respuesta.status_code in (403, 429):
                raise bp.BeatportBlocked(f"el preview respondió {respuesta.status_code} dos veces")
            return b""
        return b""


# ---------------------------------------------------------------- comandos


def base_corrida(args) -> Path:
    return args.root / "beatport_previews" / args.corrida


def cmd_elegir(args) -> int:
    carpeta = args.root / "1001tracklists" / "sets"
    sets = []
    for ruta in sorted(carpeta.glob("*.json")):
        dato = json.loads(ruta.read_text(encoding="utf-8"))
        html = ruta.with_suffix(".html.gz")
        if html.exists() and not dato.get("view_item"):
            dato["view_item"] = view_item_de(gzip.open(html, "rt", encoding="utf-8").read())
        sets.append(dato)
    filas = elegir(sets, args.n)
    escribir_tabla(base_corrida(args) / "temas.csv", filas)
    print(f"{len(sets)} sets, {len(filas)} temas elegidos -> {base_corrida(args) / 'temas.csv'}; sin viewItem: {sum(not f['view_item'] for f in filas)}")
    return 0


def cmd_enlaces(args) -> int:
    from tools.dj_sets.tracklists_1001 import DEFAULT_PERFIL, Detencion, navegador

    ruta = base_corrida(args) / "temas.csv"
    filas = leer_tabla(ruta)
    pendientes = [f for f in filas if not f.get("fuente_1001")]
    print(f"{len(filas)} temas, {len(pendientes)} sin enlace pedido; unos {len(pendientes) * 8.5 / 60:.0f} min", flush=True)
    if not pendientes:
        return 0
    try:
        with navegador(args.perfil or DEFAULT_PERFIL, args.esperar_login) as (pagina, _estado):
            for indice, fila in enumerate(pendientes, 1):
                pagina.wait_for_timeout(PAUSA_1001_MS)
                respuesta = pagina.evaluate(PEDIR, url_enlace(fila))
                if respuesta["status"] != 200:
                    print(f"PARO en {fila['clave']}: HTTP {respuesta['status']}. No sigo ni reintento.", flush=True)
                    return 2
                try:
                    identificador, fuente = leer_enlace(json.loads(respuesta["text"]))
                except ValueError:
                    print(f"PARO en {fila['clave']}: la respuesta no es JSON (¿captcha?). No sigo.", flush=True)
                    return 2
                fila["beatport_id_1001"], fila["fuente_1001"] = identificador, fuente
                escribir_tabla(ruta, filas)
                print(f"[{indice}/{len(pendientes)}] {fila['nombre'][:60]}: {identificador or fuente}", flush=True)
    except Detencion as exc:
        print(f"PARO: {exc}", flush=True)
        return 2
    return 0


def cmd_buscar(args) -> int:
    ruta = base_corrida(args) / "temas.csv"
    filas = leer_tabla(ruta)
    responde, motivo = bp.available()
    if not responde:
        print(f"PARO: Beatport {motivo}.", flush=True)
        return 2
    cliente = bp.BeatportClient(args.root / "beatport_previews" / "cache", min_interval=PAUSA_BEATPORT_S)
    for indice, fila in enumerate(filas, 1):
        if fila.get("nivel_busqueda"):
            continue
        try:
            identificador, nivel, pedidos = buscar_en_beatport(cliente, fila)
        except bp.BeatportBlocked as exc:
            print(f"PARO: {exc}", flush=True)
            return 2
        fila["beatport_id_busqueda"], fila["nivel_busqueda"], fila["pedidos_busqueda"] = identificador, nivel, str(pedidos)
        escribir_tabla(ruta, filas)
        print(f"[{indice}/{len(filas)}] {fila['nombre'][:60]}: {nivel} {identificador}", flush=True)
    return 0


def cmd_bajar(args) -> int:
    from src.v4.common.remote_storage import open_remote, remote_root

    filas = leer_tabla(base_corrida(args) / "temas.csv")
    remoto = open_remote()
    carpeta_remota = f"{CARPETA_REMOTA}/{args.corrida}"
    hechas = set(remoto.ls(carpeta_remota))  # {} si la carpeta no existe; exists() es solo para archivos
    pendientes = tandas_pendientes(len(filas), hechas, args.tanda)
    print(f"remoto {remote_root()}/{carpeta_remota}: {len(filas)} temas, tandas de {args.tanda}, pendientes {pendientes}", flush=True)
    responde, motivo = bp.available()
    if not responde:
        print(f"PARO: Beatport {motivo}.", flush=True)
        return 2
    cliente = bp.BeatportClient(args.root / "beatport_previews" / "cache", min_interval=PAUSA_BEATPORT_S)
    descargador = Descargador()
    for k in pendientes:
        nombre = nombre_tanda(k)
        temporal = Path(args.temporal) / args.corrida / nombre
        t0 = time.monotonic()
        try:
            tar, manifiesto = armar_tanda(filas[k * args.tanda:(k + 1) * args.tanda], cliente.track_details, descargador, temporal, nombre)
        except bp.BeatportBlocked as exc:
            print(f"PARO en {nombre}: {exc}. Lo ya subido queda.", flush=True)
            return 2
        manifiesto["segundos_tanda"] = round(time.monotonic() - t0, 1)
        manifiesto["estados_http_mp3"] = dict(descargador.estados)
        remoto.put(str(tar), f"{carpeta_remota}/{nombre}.tar")
        remoto.put_json(manifiesto, f"{carpeta_remota}/{nombre}.json")
        copia = base_corrida(args) / f"{nombre}.json"
        copia.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=1), encoding="utf-8")
        for archivo in sorted(temporal.glob("*")):
            archivo.unlink()
        temporal.rmdir()
        tamano = tar.stat().st_size
        tar.unlink()
        print(f"{nombre}: {manifiesto['ok']} previews, {tamano / 1e6:.1f} MB, {manifiesto['segundos_tanda']:.0f} s; subida verificada", flush=True)
    return 0


def cmd_resumen(args) -> int:
    filas = leer_tabla(base_corrida(args) / "temas.csv")
    ambos = [f for f in filas if f.get("beatport_id_1001") and f.get("nivel_busqueda") in ("A", "B")]
    iguales = sum(f["beatport_id_1001"] == f["beatport_id_busqueda"] for f in ambos)
    print(f"temas {len(filas)} | enlace 1001: {sum(bool(f.get('beatport_id_1001')) for f in filas)} | fuentes {dict(Counter(f.get('fuente_1001', '') for f in filas))}")
    print(f"búsqueda por nombre: niveles {dict(Counter(f.get('nivel_busqueda', '') for f in filas))}; "
          f"con los dos (A/B): {len(ambos)}, mismo id: {iguales}; pedidos {sum(int(f.get('pedidos_busqueda') or 0) for f in filas)}")
    for ruta in sorted(base_corrida(args).glob("tanda_*.json")):
        manifiesto = json.loads(ruta.read_text(encoding="utf-8"))
        ok = [e for e in manifiesto["temas"] if e["estado"] == "ok"]
        pesos = sorted(e["bytes"] for e in ok)
        print(f"{ruta.stem}: estados {dict(Counter(e['estado'] for e in manifiesto['temas']))}; HTTP mp3 {manifiesto.get('estados_http_mp3')}; "
              f"{manifiesto.get('segundos_tanda')} s en total")
        if pesos:
            print(f"   peso: mediana {pesos[len(pesos) // 2] / 1e6:.2f} MB, total {sum(pesos) / 1e6:.1f} MB; "
                  f"página {sum(e['segundos_pagina'] for e in ok) / len(ok):.2f} s y mp3 {sum(e['segundos_mp3'] for e in ok) / len(ok):.2f} s por tema")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="carpeta de datos de sets (en este PC, en D:)")
    parser.add_argument("--corrida", default="piloto", help="nombre de la corrida: carpeta local y remota")
    comandos = parser.add_subparsers(dest="comando", required=True)
    p = comandos.add_parser("elegir")
    p.add_argument("--n", type=int, default=100)
    p.set_defaults(func=cmd_elegir)
    p = comandos.add_parser("enlaces")
    p.add_argument("--perfil", type=Path, default=None, help="perfil del navegador con la sesión (por defecto, el de tracklists_1001)")
    p.add_argument("--esperar-login", type=int, default=0)
    p.set_defaults(func=cmd_enlaces)
    p = comandos.add_parser("buscar")
    p.set_defaults(func=cmd_buscar)
    p = comandos.add_parser("bajar")
    p.add_argument("--tanda", type=int, default=TANDA)
    p.add_argument("--temporal", required=True, help="carpeta temporal para los MP3 y el .tar (en D:, no en C:)")
    p.set_defaults(func=cmd_bajar)
    p = comandos.add_parser("resumen")
    p.set_defaults(func=cmd_resumen)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
