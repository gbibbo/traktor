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
         Corrida grande (Gabriel, 2026-11-09): buscar primero en Beatport y usar el enlace de 1001Tracklists si el
         tema no salió con la búsqueda; la búsqueda y la bajada corren en gpuq (Kaggle, CPU) con `nube`:
           elegir --todos --n 1000000 ; subir-temas            (en esta PC)
           python tools/gpuq/cola.py add <id> --cmd "python tools/dj_sets/beatport_previews.py --corrida <c> nube" --gpu CPU
CHANGELOG:
  - 2026-11-09: Creación (piloto de 100 temas).
  - 2026-11-09: búsqueda primero y enlace de 1001Tracklists de respaldo; subir-temas y nube para gpuq; elegir --todos.
  - 2026-11-10: la búsqueda usa match(soft_mix=True) solo si ninguna consulta trajo la versión exacta (columna
                mezcla_suave); «aka» en el artista; rescate (los temas que otras corridas no resolvieron) y
                enlaces --solo-sin-id. Kaggle no sirve para esto: Beatport le pide la verificación de Cloudflare.
  - 2026-11-10: una conexión cortada al bajar un MP3 se reintenta (15 s, 1 min, 3 min) y después para la corrida limpia
                en vez de tumbarla con una excepción (lote_04, 13:00); enlaces --n (pedidos por corrida).
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
COLUMNAS = ["clave", "nombre", "artista", "titulo", "con_boton", "id_numerico", "view_item", "set_id", "beatport_id_1001", "fuente_1001",
            "beatport_id_busqueda", "nivel_busqueda", "pedidos_busqueda", "mezcla_suave"]
ESTADOS_RESUELTOS = ("ok", "repetido")


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


def elegir(sets: Sequence[Dict], n: int, todos: bool = False) -> List[Dict[str, str]]:
    """Temas identificados sin repetir, tomados de a uno por set y por vuelta. Por defecto solo los que tienen botón de
    compra; con todos=True también los otros (la búsqueda en Beatport puede encontrarlos; el respaldo de 1001Tracklists,
    no)."""
    colas = []
    for dato in sets:
        cola = []
        for tema in dato["temas"]:
            boton = "affiliate" in tema["fuentes"]
            if tema["junto_al_anterior"] or not tema["identificado"] or not (boton or todos):
                continue
            nombre = tema["nombre"] or tema["visible"]
            artista, titulo = separar_nombre(nombre)
            if not artista or not titulo or (boton and not tema["id_numerico"]):
                continue
            clave = tema["id_texto"] or "nombre:" + bp.norm(nombre)
            cola.append({"clave": clave, "nombre": nombre, "artista": artista, "titulo": titulo, "con_boton": "si" if boton else "",
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


def sin_alias(artista: str) -> str:
    """'Ignacio aka Steve Rachmad' -> 'Ignacio, Steve Rachmad' (Beatport publica uno de los dos nombres). No toca
    nombres que llevan la palabra: 'AKA AKA & Thalstroem'."""
    partes = [p.strip() for p in re.split(r"\s+aka\s+", artista or "", flags=re.I)]

    def es_nombre(parte: str) -> bool:
        palabras = bp.norm(parte).split()
        return bool(palabras) and palabras[0] != "aka" and palabras[-1] != "aka"

    if len(partes) < 2 or not all(es_nombre(p) for p in partes):
        return artista
    return ", ".join(partes)


def consulta_de(fila: Dict[str, str], alias: bool = False) -> bp.TrackQuery:
    base, mezcla = bp.split_title_mix(fila["titulo"])
    return bp.TrackQuery(artists=sin_alias(fila["artista"]) if alias else fila["artista"], title=base, mix=mezcla)


def _orden(resultado: bp.MatchResult) -> tuple:
    """A o B estricto, después B por la mezcla flexible, después C, después D."""
    return ({"A": 3, "B": 2, "C": 1}.get(resultado.level, 0), not resultado.soft_mix)


def buscar_en_beatport(cliente: bp.BeatportClient, fila: Dict[str, str]) -> tuple:
    """(id, nivel, pedidos): prueba las consultas de TrackQuery hasta un nivel A o B estricto. La comparación flexible
    de la mezcla (match soft_mix) vale solo si ninguna consulta trajo la versión exacta; queda en fila['mezcla_suave'].
    Si el artista trae un alias ('X aka Y'), primero con el nombre como viene y después con los dos nombres separados."""
    consultas = [consulta_de(fila)]
    if sin_alias(fila["artista"]) != fila["artista"]:
        consultas.append(consulta_de(fila, alias=True))
    antes, mejor = cliente.n_requests, bp.MatchResult(level="D")
    for consulta in consultas:
        candidatos = []
        for texto in consulta.search_queries():
            candidatos.extend(cliente.search(texto))
            resultado = bp.match(consulta, candidatos, soft_mix=True)
            if _orden(resultado) >= _orden(mejor):  # con los candidatos juntados, el último resultado del mismo nivel
                mejor = resultado
            if mejor.level in ("A", "B") and not mejor.soft_mix:
                break
        if mejor.level in ("A", "B") and not mejor.soft_mix:
            break
    fila["mezcla_suave"] = "si" if mejor.soft_mix else ""
    identificador = str(mejor.best["track_id"]) if mejor.best else ""
    return identificador, mejor.level, cliente.n_requests - antes


def es_mp3(contenido: bytes) -> bool:
    return len(contenido) >= MIN_BYTES_MP3 and (contenido[:3] == b"ID3" or (contenido[0] == 0xFF and contenido[1] & 0xE0 == 0xE0))


def ids_candidatos(fila: Dict[str, str]) -> List[str]:
    """Primero la búsqueda en Beatport si dio la misma versión (A o B); después el enlace de 1001Tracklists
    (Gabriel, 2026-11-09: buscar primero y usar el enlace del tracklist si el tema no salió con la búsqueda)."""
    ids = []
    if fila.get("nivel_busqueda") in ("A", "B") and fila.get("beatport_id_busqueda"):
        ids.append(fila["beatport_id_busqueda"])
    if fila.get("beatport_id_1001") and fila["beatport_id_1001"] not in ids:
        ids.append(fila["beatport_id_1001"])
    return ids


def beatport_id(fila: Dict[str, str]) -> str:
    ids = ids_candidatos(fila)
    return ids[0] if ids else ""


def nombre_tanda(k: int) -> str:
    return f"tanda_{k:05d}"


def tandas_pendientes(n_filas: int, hechas: Iterable[str], tam: int = TANDA) -> List[int]:
    hechas = set(hechas)
    return [k for k in range((n_filas + tam - 1) // tam) if f"{nombre_tanda(k)}.tar" not in hechas]


def armar_tanda(filas: Sequence[Dict[str, str]], detalles: Callable[[int], Optional[dict]], bajar_bytes: Callable[[str], bytes],
                carpeta: Path, nombre: str, buscar: Optional[Callable[[Dict[str, str]], tuple]] = None) -> tuple:
    """Baja los previews de una tanda a carpeta/ y arma <nombre>.tar con manifest.json. (ruta del tar, manifiesto).
    Con buscar, busca en Beatport los temas que todavía no se buscaron. Prueba los ids de ids_candidatos en orden:
    si la página no existe o no tiene preview, pasa al siguiente."""
    carpeta.mkdir(parents=True, exist_ok=True)
    entradas, archivos, en_tanda = [], [], {}
    for fila in filas:
        if buscar and not fila.get("nivel_busqueda"):
            fila["beatport_id_busqueda"], fila["nivel_busqueda"], _ = buscar(fila)  # BeatportBlocked corta la corrida
        base = {"clave_1001": fila["clave"], "nombre_1001": fila["nombre"], "nivel_busqueda": fila.get("nivel_busqueda", ""),
                "beatport_id_busqueda": fila.get("beatport_id_busqueda", ""), "beatport_id_1001": fila.get("beatport_id_1001", ""),
                "mezcla_suave": fila.get("mezcla_suave", "")}
        ids = ids_candidatos(fila)
        if not ids:
            entradas.append(dict(base, estado="sin_id"))
            continue
        estado, extra = "sin_preview", {}
        for identificador in ids:
            via = "busqueda" if identificador == base["beatport_id_busqueda"] and base["nivel_busqueda"] in ("A", "B") else "1001"
            if identificador in en_tanda:
                estado, extra = "repetido", {"beatport_id": identificador, "via": via, "igual_a": en_tanda[identificador]}
                break
            t0 = time.monotonic()
            try:
                datos = detalles(int(identificador))
            except bp.BeatportBlocked:
                raise
            except Exception as exc:  # una página rota no tumba la tanda
                estado, extra = f"pagina_error: {type(exc).__name__}", {"beatport_id": identificador}
                continue
            if not datos or not datos.get("sample_url"):
                estado, extra = "sin_preview", {"beatport_id": identificador, **({"detalles": datos} if datos else {})}
                continue
            t1 = time.monotonic()
            contenido = bajar_bytes(datos["sample_url"])
            t2 = time.monotonic()
            if not es_mp3(contenido):
                estado, extra = "mp3_invalido", {"beatport_id": identificador, "bytes": len(contenido), "detalles": datos}
                continue
            archivo = carpeta / f"{identificador}.mp3"
            archivo.write_bytes(contenido)
            archivos.append(archivo)
            en_tanda[identificador] = fila["clave"]
            estado, extra = "ok", {"beatport_id": identificador, "via": via, "archivo": archivo.name, "bytes": len(contenido),
                                   "md5": hashlib.md5(contenido).hexdigest(), "segundos_pagina": round(t1 - t0, 2),
                                   "segundos_mp3": round(t2 - t1, 2), "detalles": datos}
            break
        entradas.append(dict(base, estado=estado, **extra))
    manifiesto = {"tanda": nombre, "creado": time.strftime("%Y-%m-%dT%H:%M:%S"), "temas": entradas,
                  "ok": sum(e["estado"] == "ok" for e in entradas)}
    (carpeta / "manifest.json").write_text(json.dumps(manifiesto, ensure_ascii=False, indent=1), encoding="utf-8")
    tar = carpeta.parent / f"{nombre}.tar"
    with tarfile.open(tar, "w") as paquete:
        paquete.add(carpeta / "manifest.json", arcname="manifest.json")
        for archivo in archivos:
            paquete.add(archivo, arcname=archivo.name)
    return tar, manifiesto


ESPERAS_CORTE_S = (15, 60, 180)  # conexión cortada: reintentos antes de parar la corrida (que después se retoma)


class Descargador:
    """MP3 de los previews con la misma pausa que el cliente de Beatport; dos rechazos seguidos cortan la corrida. Una
    conexión cortada se reintenta tras ESPERAS_CORTE_S; si sigue, la corrida para limpia (BeatportBlocked)."""

    def __init__(self, pausa_s: float = PAUSA_BEATPORT_S, sesion=None):
        if sesion is None:
            import requests

            sesion = requests.Session()
            sesion.headers.update({"User-Agent": bp.USER_AGENT})
        self.sesion = sesion
        self.pausa_s, self._ultimo, self.estados = pausa_s, 0.0, Counter()

    def __call__(self, url: str) -> bytes:
        import requests

        rechazos = cortes = 0
        while True:
            espera = self._ultimo + self.pausa_s + random.uniform(0, 0.4) - time.monotonic()
            if espera > 0:
                time.sleep(espera)
            self._ultimo = time.monotonic()
            try:
                respuesta = self.sesion.get(url, timeout=60)
            except requests.RequestException as exc:  # 2026-11-10: «Remote end closed connection» tumbó lote_04
                self.estados[type(exc).__name__] += 1
                if cortes == len(ESPERAS_CORTE_S):
                    raise bp.BeatportBlocked(f"el preview no respondió {cortes + 1} veces seguidas ({type(exc).__name__})") from exc
                time.sleep(ESPERAS_CORTE_S[cortes])
                cortes += 1
                continue
            self.estados[respuesta.status_code] += 1
            if respuesta.status_code == 200:
                return respuesta.content
            if respuesta.status_code in (403, 429):
                if rechazos:
                    raise bp.BeatportBlocked(f"el preview respondió {respuesta.status_code} dos veces")
                rechazos += 1
                time.sleep(15)
                continue
            return b""


# ---------------------------------------------------------------- comandos


def base_corrida(args) -> Path:
    return args.root / "beatport_previews" / args.corrida


def excluidas_de(args) -> set:
    """Claves de los temas de otras corridas (--excluir), para que una corrida nueva no repita temas."""
    claves = set()
    for corrida in args.excluir or []:
        claves |= {f["clave"] for f in leer_tabla(args.root / "beatport_previews" / corrida / "temas.csv")}
    return claves


def cmd_elegir(args) -> int:
    carpeta = args.root / "1001tracklists" / "sets"
    sets = []
    for ruta in sorted(carpeta.glob("*.json")):
        dato = json.loads(ruta.read_text(encoding="utf-8"))
        html = ruta.with_suffix(".html.gz")
        if html.exists() and not dato.get("view_item"):
            dato["view_item"] = view_item_de(gzip.open(html, "rt", encoding="utf-8").read())
        sets.append(dato)
    filas = elegir(sets, args.n + len(excluidas_de(args)), args.todos)
    excluidas = excluidas_de(args)
    filas = [f for f in filas if f["clave"] not in excluidas][:args.n]
    destino = base_corrida(args) / "temas.csv"
    if not filas:
        print(f"Nada nuevo para {args.corrida}: no se escribe la lista.")
        return 3
    if destino.exists():
        print(f"PARO: {destino} ya existe; las tandas se cuentan por fila, así que una lista nueva va en otra corrida.")
        return 2
    escribir_tabla(destino, filas)
    print(f"{len(sets)} sets, {len(filas)} temas elegidos -> {base_corrida(args) / 'temas.csv'}; sin viewItem: {sum(not f['view_item'] for f in filas)}")
    return 0


def filas_sin_resolver(base: Path) -> List[Dict[str, str]]:
    """Las filas de temas.csv de una corrida que en sus manifiestos locales no quedaron ok ni repetidas (sin id, sin
    preview, error de página), con la búsqueda borrada para que se repita con el código de ahora."""
    tabla = {f["clave"]: f for f in leer_tabla(base / "temas.csv")}
    filas = []
    for ruta in sorted(base.glob("tanda_*.json")):
        for entrada in json.loads(ruta.read_text(encoding="utf-8"))["temas"]:
            if entrada["estado"] not in ESTADOS_RESUELTOS:
                filas.append(dict(tabla[entrada["clave_1001"]], beatport_id_busqueda="", nivel_busqueda="",
                                  pedidos_busqueda="", mezcla_suave=""))
    return filas


def cmd_rescate(args) -> int:
    """Una corrida nueva con los temas que otras corridas no resolvieron: se buscan de nuevo (la caché ahorra los pedidos
    repetidos), se pide el enlace de 1001Tracklists de los que tienen botón (enlaces --solo-sin-id) y se bajan."""
    excluidas, vistas, filas = excluidas_de(args), set(), []
    for corrida in args.de:
        base = args.root / "beatport_previews" / corrida
        n_temas = len(leer_tabla(base / "temas.csv"))
        n_tandas = len(list(base.glob("tanda_*.json")))
        nuevas = [f for f in filas_sin_resolver(base) if f["clave"] not in excluidas and f["clave"] not in vistas]
        vistas |= {f["clave"] for f in nuevas}
        filas += nuevas
        print(f"{corrida}: {n_tandas} de {(n_temas + TANDA - 1) // TANDA} tandas con manifiesto local, {len(nuevas)} sin resolver", flush=True)
    destino = base_corrida(args) / "temas.csv"
    if not filas:
        print(f"Nada sin resolver para {args.corrida}.")
        return 3
    if destino.exists():
        print(f"PARO: {destino} ya existe; las tandas se cuentan por fila, así que una lista nueva va en otra corrida.")
        return 2
    escribir_tabla(destino, filas)
    print(f"{len(filas)} temas -> {destino}", flush=True)
    return 0


def pendientes_de_enlace(filas: Sequence[Dict[str, str]], solo_sin_id: bool) -> List[Dict[str, str]]:
    """Filas a las que falta pedir el enlace. Con solo_sin_id, solo las que tienen botón de compra (las otras no
    tienen enlace que dar) y que la búsqueda no resolvió (Gabriel: primero la búsqueda)."""
    return [f for f in filas if not f.get("fuente_1001") and (not solo_sin_id or (
        f.get("con_boton") == "si" and f.get("nivel_busqueda") not in ("A", "B")))]


def cmd_enlaces(args) -> int:
    from tools.dj_sets.tracklists_1001 import DEFAULT_PERFIL, Detencion, navegador

    ruta = base_corrida(args) / "temas.csv"
    filas = leer_tabla(ruta)
    pendientes = pendientes_de_enlace(filas, args.solo_sin_id)
    print(f"{len(filas)} temas, {len(pendientes)} sin enlace pedido", flush=True)
    pendientes = pendientes[:args.n] if args.n else pendientes
    print(f"pido {len(pendientes)}; unos {len(pendientes) * 8.5 / 60:.0f} min", flush=True)
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


def procesar_tandas(filas: List[Dict[str, str]], remoto, carpeta_remota: str, temporal: Path, cache: Path, tam: int,
                    copias: Optional[Path] = None) -> int:
    """Las tandas que faltan en el remoto: buscar, bajar, empaquetar y subir. 0 si terminó, 2 si Beatport cortó."""
    from src.v4.common.remote_storage import remote_root

    hechas = set(remoto.ls(carpeta_remota))  # {} si la carpeta no existe; exists() es solo para archivos
    pendientes = tandas_pendientes(len(filas), hechas, tam)
    print(f"remoto {remote_root()}/{carpeta_remota}: {len(filas)} temas, tandas de {tam}, faltan {len(pendientes)}: {pendientes[:12]}", flush=True)
    if not pendientes:
        return 0
    responde, motivo = bp.available()
    if not responde:
        print(f"PARO: Beatport {motivo}.", flush=True)
        return 2
    cliente = bp.BeatportClient(cache, min_interval=PAUSA_BEATPORT_S)
    descargador = Descargador()

    def buscar(fila: Dict[str, str]) -> tuple:
        return buscar_en_beatport(cliente, fila)

    for k in pendientes:
        nombre = nombre_tanda(k)
        carpeta = Path(temporal) / nombre
        t0 = time.monotonic()
        try:
            tar, manifiesto = armar_tanda(filas[k * tam:(k + 1) * tam], cliente.track_details, descargador, carpeta, nombre, buscar)
        except bp.BeatportBlocked as exc:
            print(f"PARO en {nombre}: {exc}. Lo ya subido queda.", flush=True)
            return 2
        manifiesto["segundos_tanda"] = round(time.monotonic() - t0, 1)
        manifiesto["estados_http_mp3"] = dict(descargador.estados)
        manifiesto["pedidos_beatport"] = cliente.n_requests
        remoto.put(str(tar), f"{carpeta_remota}/{nombre}.tar")
        remoto.put_json(manifiesto, f"{carpeta_remota}/{nombre}.json")
        if copias:
            Path(copias).mkdir(parents=True, exist_ok=True)
            (Path(copias) / f"{nombre}.json").write_text(json.dumps(manifiesto, ensure_ascii=False, indent=1), encoding="utf-8")
        for archivo in sorted(carpeta.glob("*")):
            archivo.unlink()
        carpeta.rmdir()
        tamano = tar.stat().st_size
        tar.unlink()
        print(f"{nombre}: {manifiesto['ok']} de {len(manifiesto['temas'])} previews, {tamano / 1e6:.1f} MB, "
              f"{manifiesto['segundos_tanda']:.0f} s; subida verificada", flush=True)
    return 0


def cmd_bajar(args) -> int:
    from src.v4.common.remote_storage import open_remote

    return procesar_tandas(leer_tabla(base_corrida(args) / "temas.csv"), open_remote(), f"{CARPETA_REMOTA}/{args.corrida}",
                           Path(args.temporal) / args.corrida, args.root / "beatport_previews" / "cache", args.tanda, base_corrida(args))


def cmd_subir_temas(args) -> int:
    """Sube la lista de temas de una corrida para la VM. No reemplaza una lista distinta: las tandas se cuentan por fila."""
    from src.v4.common.remote_storage import import_gpuq_storage, open_remote

    local = base_corrida(args) / "temas.csv"
    rel = f"{CARPETA_REMOTA}/{args.corrida}/temas.csv"
    remoto = open_remote()
    actual = remoto.stat(rel)
    propio = import_gpuq_storage().md5_file(local)
    if actual and actual.get("md5") and actual["md5"] != propio:
        print(f"PARO: {rel} ya está en el remoto con otro contenido; una lista nueva va en otra corrida.", flush=True)
        return 2
    if not actual:
        remoto.put(str(local), rel)
    print(f"{rel}: {len(leer_tabla(local))} temas, md5 {propio}", flush=True)
    return 0


def cmd_nube(args) -> int:
    """Lo que corre en la VM de gpuq: la lista sale del remoto y la carpeta temporal es de la VM."""
    import tempfile

    from src.v4.common.remote_storage import open_remote

    remoto = open_remote()
    carpeta_remota = f"{CARPETA_REMOTA}/{args.corrida}"
    trabajo = Path(args.temporal) if args.temporal else Path(tempfile.gettempdir()) / "beatport_previews" / args.corrida
    trabajo.mkdir(parents=True, exist_ok=True)
    remoto.get(f"{carpeta_remota}/temas.csv", trabajo / "temas.csv")
    return procesar_tandas(leer_tabla(trabajo / "temas.csv"), remoto, carpeta_remota, trabajo / "tandas", trabajo / "cache", args.tanda)


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
    p.add_argument("--todos", action="store_true", help="también los temas identificados sin botón de compra")
    p.add_argument("--excluir", action="append", help="corrida anterior cuyos temas no se repiten (se puede repetir)")
    p.set_defaults(func=cmd_elegir)
    p = comandos.add_parser("subir-temas", help="sube temas.csv de la corrida al remoto, para la VM")
    p.set_defaults(func=cmd_subir_temas)
    p = comandos.add_parser("nube", help="en la VM de gpuq: lista desde el remoto, buscar, bajar y subir por tandas")
    p.add_argument("--tanda", type=int, default=TANDA)
    p.add_argument("--temporal", default="", help="carpeta temporal (por defecto, la de la VM)")
    p.set_defaults(func=cmd_nube)
    p = comandos.add_parser("rescate", help="corrida nueva con los temas que otras corridas no resolvieron")
    p.add_argument("--de", action="append", required=True, help="corrida cuyos temas sin resolver se toman (se puede repetir)")
    p.add_argument("--excluir", action="append", help="corrida (otro rescate) cuyos temas no se repiten")
    p.set_defaults(func=cmd_rescate)
    p = comandos.add_parser("enlaces")
    p.add_argument("--perfil", type=Path, default=None, help="perfil del navegador con la sesión (por defecto, el de tracklists_1001)")
    p.add_argument("--esperar-login", type=int, default=0)
    p.add_argument("--solo-sin-id", action="store_true", help="solo los temas con botón de compra que la búsqueda no encontró")
    p.add_argument("--n", type=int, default=0, help="como máximo N pedidos en esta corrida (0 = todos); el resto queda pendiente")
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
