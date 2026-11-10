"""
PURPOSE: D6, el audio de los temas de los sets que Beatport no da (Gabriel, 2026-11-10: autoriza YouTube para los que
         faltan; DECISIONS 2026-11-07 p. 12 ya lo autorizaba para esta aplicación). Toma los temas que una o más
         corridas de beatport_previews.py no resolvieron (sin id, sin preview), los busca en YouTube con yt-dlp, elige
         el video con la misma versión y baja el audio completo, sin recodificar (m4a si hay). Guarda en el Drive de
         Gabriel, erclm:traktor_ml/youtube_temas/<corrida>/, en tandas de hasta 200 temas por .tar con manifest.json,
         como beatport_previews.py; salta las tandas que ya están, así se retoma si se corta.
           elegir  -> <root>/youtube_temas/<corrida>/temas.csv con los temas sin resolver de las corridas --de
                      (--solo-sin-boton: solo los que no tienen botón de compra en 1001Tracklists, que no tienen
                      enlace que esperar)
           bajar   -> buscar, elegir, bajar, empaquetar y subir por tandas
           probar  -> sin subir nada: el elegido para temas que Beatport ya identificó, con su duración, para medir
                      cuántas veces YouTube da la misma grabación (misma duración ±3 s)
         Cómo elige (elegir_video, sin red): de los primeros resultados de la búsqueda, descarta los que nombran algo
         que el tema no nombra (live, set, @, played by, snippet, slowed…), los de menos de 90 s o más de 20 min, los
         de otro título (similitud < 0,85, como match() de Beatport), los que no nombran a ningún artista del tema, y
         los de otra versión (la mezcla se compara como en Beatport: estricta y, si no, soft_mix_core; un tema sin
         mezcla solo acepta original, extended, radio o edit). Entre los que quedan prefiere el canal «- Topic» (el
         audio oficial que YouTube Music arma con lo que mandan los sellos) y la mezcla estricta.
         Si YouTube pide iniciar sesión para confirmar que no es un robot, la corrida para: no se usa la cuenta de
         Gabriel ni se esquiva la verificación.
         Nada fuera de erclm:traktor_ml/. El audio pasa por una carpeta temporal en D:, nunca en C:.
CHANGELOG:
  - 2026-11-10: Creación. La restricción de edad no para la corrida (se prueba el siguiente video); artista numérico
                («747 - »), sin marca de país y sin espacios; «Remix by X»; alias en la mezcla; corte largo antes que Radio Edit.
                bajar --pausa (YouTube pidió la verificación anti-robots tras unos 200 temas a 2 s).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from src.v4.common import beatport as bp  # noqa: E402
from tools.dj_sets import beatport_previews as bpv  # noqa: E402

DEFAULT_ROOT = bpv.DEFAULT_ROOT
CARPETA_REMOTA = "youtube_temas"
N_RESULTADOS = 8
TANDA = 200                    # unos 7 MB por tema: 500 pasarían los 4 GB por archivo de FAT32 (el pendrive D:)
MIN_S, MAX_S = 90, 20 * 60
PAUSA_S = 2.0
FORMATO = "bestaudio[ext=m4a]/bestaudio"
# Lo que delata un video que no es el tema publicado (si el tema mismo no lo nombra).
PROHIBIDAS = re.compile(r"\b(live|set|played\s+by|full\s+album|megamix|mixtape|podcast|boiler\s+room|snippet|preview|teaser|"
                        r"cover|karaoke|reaction|tutorial|slowed|sped\s+up|nightcore|8d|hour|loop|mashup|bootleg|"
                        r"unreleased|vinyl\s+rip)\b|@", re.I)
SEPARADORES = re.compile(r"\s+[-–—|]\s+")
CORTE = re.compile(r"\b(radio|edit|short|single)\b", re.I)
VIDEO = re.compile(r"\b(official\s+(?:music\s+)?video|music\s+video|video\s+clip|visuali[sz]er|lyric)", re.I)


class YouTubeBloqueado(RuntimeError):
    """YouTube pide iniciar sesión o corta: la corrida para (no se usa ninguna cuenta ni se esquiva)."""


def ytdlp(*args: str, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "yt_dlp", *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


def revisar_bloqueo(stderr: str) -> None:
    """Para la corrida si YouTube pide confirmar que no es un robot o limita (429). Un video con restricción de edad
    («Sign in to confirm your age») no es un bloqueo: ese video no se baja y se prueba el siguiente."""
    if re.search(r"confirm you.re not a bot|HTTP Error 429|too many requests", stderr or "", re.I):
        raise YouTubeBloqueado((stderr or "").strip().splitlines()[-1][:200])


# ---------------------------------------------------------------- elegir el video (puro)


def partes_del_video(entrada: Dict) -> tuple:
    """(artista, título con la mezcla) de un resultado de YouTube. En un canal «X - Topic» el artista es X y el título
    del video es el del tema; si no, «Artista - Título» del título del video, o el canal como artista."""
    canal = (entrada.get("channel") or entrada.get("uploader") or "").strip()
    crudo = (entrada.get("title") or "").strip()
    if canal.endswith(" - Topic"):
        return canal[: -len(" - Topic")], bp.clean_text(crudo)
    # Partir antes de limpiar: clean_text toma «747 - » por un número de pista y se lleva al artista.
    partes = SEPARADORES.split(crudo, maxsplit=1)
    if len(partes) == 2:
        return bp.clean_text(partes[0]), bp.clean_text(partes[1])
    return canal, bp.clean_text(crudo)


def _artistas(artista: str) -> List[str]:
    """Los nombres del artista de 1001Tracklists, sueltos y juntos, y sin la marca de país: 'VLTRA (IT)' -> 'VLTRA'."""
    nombres = bp.split_local_artists(bpv.sin_alias(artista))
    todos = set()
    for n in nombres:
        for x in [n] + [y for y in re.split(r"\s*&\s*|\s+and\s+", n) if y]:
            todos.add(x)
            todos.add(re.sub(r"\s*\([^)]*\)", "", x).strip())
    return [n for n in todos if len(bp.norm(n)) >= 2]


def _compacto(texto: str) -> str:
    return bp.norm(texto).replace(" ", "")


def _nucleo(mezcla: str) -> frozenset:
    return bp.soft_mix_core(mezcla) - {"by"}


def misma_mezcla(mezcla_q: str, mezcla_v: str) -> tuple:
    """(misma versión, estricta). Estricta como match() de Beatport; si no, soft_mix_core (posesivo, «Mix» por
    «Remix») sin «by» («Reorganized by X» == «X Reorganized»), y un alias en la mezcla del tema: «Sterac aka Steve
    Rachmad Remix» acepta «Sterac Remix». Siempre que a la mezcla le quede un nombre además de «remix»."""
    if bp.mix_core(mezcla_q) == bp.mix_core(mezcla_v):
        return True, True
    nq, nv = _nucleo(mezcla_q), _nucleo(mezcla_v)
    if nq - {"remix"} and nq == nv:
        return True, False
    if "aka" in nq and nv - {"remix"} and nv <= nq:
        return True, False
    return False, False


def evaluar(artista: str, titulo_1001: str, entrada: Dict) -> tuple:
    """(acepta, motivo, puntaje) de un resultado de YouTube para un tema de 1001Tracklists."""
    nombre_1001 = f"{artista} - {titulo_1001}"
    crudo = entrada.get("title") or ""
    propias = {m.group(0).lower() for m in PROHIBIDAS.finditer(nombre_1001)}
    ajenas = [m.group(0).lower() for m in PROHIBIDAS.finditer(crudo) if m.group(0).lower() not in propias]
    if ajenas:
        return False, f"nombra {ajenas[0]}", ()
    duracion = entrada.get("duration")
    if not duracion or not MIN_S <= float(duracion) <= MAX_S:
        return False, "duración", ()
    base_q, mezcla_q = bp.split_title_mix(titulo_1001)
    yt_artista, yt_titulo = partes_del_video(entrada)
    orientaciones = [(yt_artista, yt_titulo)]
    canal = entrada.get("channel") or ""
    if SEPARADORES.search(crudo) and not canal.endswith(" - Topic"):
        orientaciones.append((yt_titulo, yt_artista))  # algunos canales ponen «Título - Artista»
    motivo = "título"
    nombres = _artistas(artista)
    for yt_artista, yt_titulo in orientaciones:
        base_v, mezcla_v = bp.split_title_mix(yt_titulo)
        similitud = bp.title_similarity(base_q, base_v)
        if similitud < bp.TITLE_MIN_SIM:
            continue
        donde = f" {bp.norm(yt_artista)} {bp.norm(canal)} "
        compactos = {_compacto(yt_artista), _compacto(canal.replace(" - Topic", ""))}
        if not any(f" {bp.norm(a)} " in donde or _compacto(a) in compactos for a in nombres):  # «K.O.L.O.» == «Kolo»
            motivo = "artista"
            continue
        igual, estricta = misma_mezcla(mezcla_q, mezcla_v)
        if not igual:
            motivo = "versión"
            continue
        # Entre los aceptados: el corte largo antes que un «Radio Edit» que el tema no pide, después el canal «- Topic»,
        # el audio antes que el videoclip (que puede traer otro corte o sonidos) y la mezcla estricta.
        corte = not (CORTE.search(mezcla_v) and not CORTE.search(mezcla_q))
        topic = canal.endswith(" - Topic")
        audio = not VIDEO.search(crudo)
        return True, "ok", (corte, topic, audio, estricta, round(similitud, 2))
    return False, motivo, ()


def elegir_videos(artista: str, titulo_1001: str, entradas: Sequence[Dict]) -> tuple:
    """(entradas aceptadas, de la mejor a la peor; motivos de descarte contados)."""
    motivos, aceptadas = Counter(), []
    for orden, entrada in enumerate(entradas):
        acepta, motivo, puntaje = evaluar(artista, titulo_1001, entrada)
        motivos[motivo] += 1
        if acepta:
            aceptadas.append((puntaje, -orden, entrada))
    aceptadas.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [x[2] for x in aceptadas], dict(motivos)


def elegir_video(artista: str, titulo_1001: str, entradas: Sequence[Dict]) -> tuple:
    """(entrada elegida o None, motivos de descarte contados)."""
    aceptadas, motivos = elegir_videos(artista, titulo_1001, entradas)
    return (aceptadas[0] if aceptadas else None), motivos


# ---------------------------------------------------------------- red


def consulta_youtube(fila: Dict[str, str]) -> str:
    return f"{bpv.sin_alias(fila['artista']).replace(',', '')} - {fila['titulo']}"


def buscar(texto: str, n: int = N_RESULTADOS, intentos: int = 2) -> List[Dict]:
    """Los primeros n resultados; si la búsqueda vuelve vacía o falla, otro intento a los 10 s (en la prueba del
    2026-11-10, tres temas que YouTube tiene volvieron sin resultados)."""
    for intento in range(intentos):
        if intento:
            time.sleep(10)
        try:
            proceso = ytdlp("--flat-playlist", "-J", f"ytsearch{n}:{texto}", timeout=120)
        except subprocess.TimeoutExpired:
            continue
        revisar_bloqueo(proceso.stderr)
        try:
            entradas = (json.loads(proceso.stdout).get("entries") or []) if proceso.returncode == 0 else []
        except ValueError:
            entradas = []
        if entradas:
            return [{k: e.get(k) for k in ("id", "title", "channel", "uploader", "duration")} for e in entradas]
    return []


def bajar_audio(video_id: str, carpeta: Path) -> Optional[Dict]:
    carpeta.mkdir(parents=True, exist_ok=True)
    try:
        proceso = ytdlp("-f", FORMATO, "--no-playlist", "--no-part", "-o", str(carpeta / "%(id)s.%(ext)s"), "--print-json", "-q",
                        f"https://www.youtube.com/watch?v={video_id}")
    except subprocess.TimeoutExpired:
        return None
    revisar_bloqueo(proceso.stderr)
    if proceso.returncode != 0 or not proceso.stdout.strip():
        return None
    try:
        datos = json.loads(proceso.stdout.strip().splitlines()[-1])
    except ValueError:
        return None
    archivo = carpeta / f"{video_id}.{datos.get('ext')}"
    if not archivo.exists():
        return None
    return {"archivo": archivo, "ext": datos.get("ext"), "abr": datos.get("abr"), "acodec": datos.get("acodec"),
            "asr": datos.get("asr"), "duracion_s": datos.get("duration")}


# ---------------------------------------------------------------- tandas


def armar_tanda(filas: Sequence[Dict[str, str]], buscar_fn: Callable[[str], List[Dict]],
                bajar_fn: Callable[[str, Path], Optional[Dict]], carpeta: Path, nombre: str, pausa_s: float = PAUSA_S) -> tuple:
    """Busca, elige y baja el audio de cada tema; arma <nombre>.tar con manifest.json. (ruta del tar, manifiesto)."""
    carpeta.mkdir(parents=True, exist_ok=True)
    entradas, archivos, en_tanda = [], [], {}
    for fila in filas:
        base = {"clave_1001": fila["clave"], "nombre_1001": fila["nombre"], "consulta": consulta_youtube(fila)}
        resultados = buscar_fn(base["consulta"])
        aceptadas, motivos = elegir_videos(fila["artista"], fila["titulo"], resultados)
        base["descartes"] = motivos
        base["resultados"] = resultados
        if not aceptadas:
            entradas.append(dict(base, estado="sin_version" if resultados else "sin_resultados"))
            time.sleep(pausa_s * random.uniform(0.8, 1.2))
            continue
        if aceptadas[0]["id"] in en_tanda:
            entradas.append(dict(base, estado="repetido", video_id=aceptadas[0]["id"], igual_a=en_tanda[aceptadas[0]["id"]]))
            continue
        elegido, audio, fallidos = None, None, []
        for candidato in aceptadas[:3]:  # restricción de edad, video quitado: el siguiente de la misma versión
            time.sleep(pausa_s * random.uniform(0.8, 1.2))
            audio = bajar_fn(candidato["id"], carpeta)
            if audio:
                elegido = candidato
                break
            fallidos.append(candidato["id"])
        if not audio:
            entradas.append(dict(base, estado="error_bajada", video_id=fallidos[0], fallidos=fallidos))
            continue
        base["fallidos"] = fallidos
        contenido = audio["archivo"].read_bytes()
        archivos.append(audio["archivo"])
        en_tanda[elegido["id"]] = fila["clave"]
        entradas.append(dict(base, estado="ok", video_id=elegido["id"], titulo_youtube=elegido.get("title"),
                             canal=elegido.get("channel"), topic=(elegido.get("channel") or "").endswith(" - Topic"),
                             archivo=audio["archivo"].name, bytes=len(contenido), md5=hashlib.md5(contenido).hexdigest(),
                             **{k: audio[k] for k in ("ext", "abr", "acodec", "asr", "duracion_s")}))
        time.sleep(pausa_s * random.uniform(0.8, 1.2))
    manifiesto = {"tanda": nombre, "creado": time.strftime("%Y-%m-%dT%H:%M:%S"), "temas": entradas,
                  "ok": sum(e["estado"] == "ok" for e in entradas)}
    (carpeta / "manifest.json").write_text(json.dumps(manifiesto, ensure_ascii=False, indent=1), encoding="utf-8")
    tar = carpeta.parent / f"{nombre}.tar"
    with tarfile.open(tar, "w") as paquete:
        paquete.add(carpeta / "manifest.json", arcname="manifest.json")
        for archivo in archivos:
            paquete.add(archivo, arcname=archivo.name)
    return tar, manifiesto


def base_corrida(args) -> Path:
    return args.root / CARPETA_REMOTA / args.corrida


def cmd_elegir(args) -> int:
    excluidas = set()
    for corrida in args.excluir or []:
        excluidas |= {f["clave"] for f in bpv.leer_tabla(args.root / CARPETA_REMOTA / corrida / "temas.csv")}
    filas, vistas = [], set()
    for corrida in args.de:
        base = args.root / "beatport_previews" / corrida
        nuevas = [f for f in bpv.filas_sin_resolver(base) if f["clave"] not in excluidas and f["clave"] not in vistas
                  and (not args.solo_sin_boton or f.get("con_boton") != "si")]
        vistas |= {f["clave"] for f in nuevas}
        filas += nuevas
        print(f"{corrida}: {len(nuevas)} temas sin resolver", flush=True)
    destino = base_corrida(args) / "temas.csv"
    if not filas:
        print(f"Nada para {args.corrida}.")
        return 3
    if destino.exists():
        print(f"PARO: {destino} ya existe; las tandas se cuentan por fila, así que una lista nueva va en otra corrida.")
        return 2
    bpv.escribir_tabla(destino, filas)
    print(f"{len(filas)} temas -> {destino}", flush=True)
    return 0


def cmd_bajar(args) -> int:
    from src.v4.common.remote_storage import open_remote, remote_root

    filas = bpv.leer_tabla(base_corrida(args) / "temas.csv")
    remoto = open_remote()
    carpeta_remota = f"{CARPETA_REMOTA}/{args.corrida}"
    pendientes = bpv.tandas_pendientes(len(filas), set(remoto.ls(carpeta_remota)), args.tanda)
    print(f"remoto {remote_root()}/{carpeta_remota}: {len(filas)} temas, tandas de {args.tanda}, faltan {len(pendientes)}: {pendientes[:12]}", flush=True)
    for k in pendientes:
        nombre = bpv.nombre_tanda(k)
        carpeta = Path(args.temporal) / args.corrida / nombre
        t0 = time.monotonic()
        try:
            tar, manifiesto = armar_tanda(filas[k * args.tanda:(k + 1) * args.tanda], buscar, bajar_audio, carpeta, nombre,
                                          pausa_s=args.pausa)
        except YouTubeBloqueado as exc:
            print(f"PARO en {nombre}: YouTube {exc}. Lo ya subido queda.", flush=True)
            return 2
        manifiesto["segundos_tanda"] = round(time.monotonic() - t0, 1)
        remoto.put(str(tar), f"{carpeta_remota}/{nombre}.tar")
        remoto.put_json(manifiesto, f"{carpeta_remota}/{nombre}.json")
        (base_corrida(args) / f"{nombre}.json").write_text(json.dumps(manifiesto, ensure_ascii=False, indent=1), encoding="utf-8")
        for archivo in sorted(carpeta.glob("*")):
            archivo.unlink()
        carpeta.rmdir()
        tamano = tar.stat().st_size
        tar.unlink()
        estados = Counter(e["estado"] for e in manifiesto["temas"])
        print(f"{nombre}: {manifiesto['ok']} de {len(manifiesto['temas'])} temas, {dict(estados)}, {tamano / 1e6:.1f} MB, "
              f"{manifiesto['segundos_tanda']:.0f} s; subida verificada", flush=True)
    return 0


def cmd_probar(args) -> int:
    """Temas que Beatport ya identificó (estado ok en las corridas --de): el video que elegiría YouTube y su duración
    frente a la de Beatport. No baja audio."""
    import glob

    muestra = []
    for corrida in args.de:
        base = args.root / "beatport_previews" / corrida
        tabla = {f["clave"]: f for f in bpv.leer_tabla(base / "temas.csv")}
        for ruta in sorted(glob.glob(str(base / "tanda_*.json"))):
            for e in json.loads(Path(ruta).read_text(encoding="utf-8"))["temas"]:
                if e["estado"] == "ok" and (e.get("detalles") or {}).get("track_length_ms"):
                    muestra.append((tabla[e["clave_1001"]], e))
    random.Random(args.semilla).shuffle(muestra)
    filas = []
    for fila, e in muestra[:args.n]:
        resultados = buscar(consulta_youtube(fila))
        elegido, motivos = elegir_video(fila["artista"], fila["titulo"], resultados)
        largo = e["detalles"]["track_length_ms"] / 1000
        d = elegido.get("duration") if elegido else None
        filas.append({"nombre_1001": fila["nombre"], "beatport": f"{', '.join(e['detalles']['artists'])} - {e['detalles']['track_name']} ({e['detalles']['mix_name']})",
                      "largo_beatport_s": largo, "elegido": elegido, "diferencia_s": None if d is None else round(float(d) - largo, 1),
                      "descartes": motivos})
        print(f"{fila['nombre'][:60]:60s} | {((elegido or {}).get('title') or '-')[:50]:50s} | "
              f"{((elegido or {}).get('channel') or '')[:20]:20s} | {filas[-1]['diferencia_s']}", flush=True)
        time.sleep(PAUSA_S * random.uniform(0.8, 1.2))
    con = [f for f in filas if f["elegido"]]
    iguales = [f for f in con if abs(f["diferencia_s"]) <= 3]
    print(f"\n{len(filas)} temas: YouTube eligió {len(con)}; misma duración que Beatport (±3 s): {len(iguales)}; "
          f"otra duración: {len(con) - len(iguales)}", flush=True)
    if args.salida:
        Path(args.salida).write_text(json.dumps(filas, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="carpeta de datos de sets (en este PC, en D:)")
    parser.add_argument("--corrida", default="prueba", help="nombre de la corrida: carpeta local y remota")
    comandos = parser.add_subparsers(dest="comando", required=True)
    p = comandos.add_parser("elegir")
    p.add_argument("--de", action="append", required=True, help="corrida de beatport_previews cuyos temas sin resolver se toman")
    p.add_argument("--excluir", action="append", help="corrida de youtube_temas cuyos temas no se repiten")
    p.add_argument("--solo-sin-boton", action="store_true", help="solo los que no tienen botón de compra en 1001Tracklists")
    p.set_defaults(func=cmd_elegir)
    p = comandos.add_parser("bajar")
    p.add_argument("--tanda", type=int, default=TANDA)
    p.add_argument("--temporal", required=True, help="carpeta temporal para el audio y el .tar (en D:, no en C:)")
    p.add_argument("--pausa", type=float, default=PAUSA_S, help="segundos entre pedidos a YouTube (el 2026-11-10 pidió la verificación anti-robots tras unos 200 temas a 2 s)")
    p.set_defaults(func=cmd_bajar)
    p = comandos.add_parser("probar")
    p.add_argument("--de", action="append", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--semilla", type=int, default=20261110)
    p.add_argument("--salida", default="")
    p.set_defaults(func=cmd_probar)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
