"""
PURPOSE: D5: elige y baja sets de 1001Tracklists con la cuenta de Gabriel (plan de recomendación, sección 6;
         DECISIONS 2026-11-08 punto 1 y 2026-11-09). Las reglas y las semillas están en
         config/dj_sets/tracklists_1001.json. Los datos van a <root>/1001tracklists/, fuera de Git.
           buscar     -> una búsqueda filtrada por DJ, término o evento: busquedas/<consulta>.json (y .html.gz)
           candidatos -> cura las filas sin abrir ningún set: candidatos.csv (con columna veto) y descartados.csv
           bajar      -> abre las páginas de los candidatos en orden: sets/<id>.json (y .html.gz)
           medir      -> calidad de lo bajado: medidas.csv (tiempos, pares consecutivos, afinidad, enlaces)
         Dos usos, según lo que pidió Gabriel: los sets de 3 h o más enseñan la trayectoria de la noche; para
         saber qué temas van juntos alcanzan los temas consecutivos de sets de cualquier largo. Los sets
         comerciales (para Gabriel, el EDM del tipo David Guetta y Martin Garrix) se descartan; descartados.csv
         dice por qué. Ante la duda, adentro: una etiqueta del sitio no saca a un DJ de las semillas.
         Reglas con el sitio: navegador visible con la sesión iniciada a mano por Gabriel (la contraseña no
         pasa por acá; el perfil queda fuera del repo), 8 s entre pedidos (Crawl-delay), sin reintentos y parada
         en la primera respuesta anormal o captcha. No se hace scroll infinito, que es donde el sitio pide
         captcha. Los enlaces por tema (/ajax/get_medialink.php) no se piden acá.
CHANGELOG:
  - 2026-11-09: Creación.
  - 2026-11-09: «Comercial» según Gabriel: sin listas de dudosos ni de eventos; las semillas no salen por etiqueta.
  - 2026-11-09: bajar --uso pares toma solo los de pares y --por-dj da un piloto variado; medir cuenta temas y pares en
    la biblioteca con la normalización de zona_sets.py.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import html as htmllib
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)

BASE = "https://www.1001tracklists.com"
PAUSA_MS = 8000
ASENTAR_MS = 2500
TIMEOUT_MS = 60000
MAX_VACIAS = 12
DEFAULT_ROOT = REPO_ROOT / "artifacts" / "v4" / "dj_sets"
DEFAULT_PERFIL = Path.home() / ".traktor_ml" / "1001tracklists_perfil"
DEFAULT_CONFIG = REPO_ROOT / "config" / "dj_sets" / "tracklists_1001.json"
DEFAULT_SPOTIFY = REPO_ROOT / "config" / "dj_sets" / "artistas_spotify.txt"

MIN_MINUTOS = 60
MIN_MINUTOS_TRAYECTORIA = 180
MIN_ID_TRAYECTORIA = 0.60
MIN_ID_PARES = 0.70
MIN_PARES_ESPERADOS = 8
TEMAS_HORA = (7.0, 26.0)
TOPE_POR_DJ = 15
TOPE_POR_PROGRAMA = 30

COLUMNAS_CANDIDATOS = [
    "veto", "uso", "puntaje", "titulo", "fecha", "minutos", "total", "identificados", "pct_id", "temas_hora",
    "generos", "grabacion", "dj", "afinidad_dj", "tipo", "b2b", "grupo_noche", "consultas", "id", "ruta",
]
COLUMNAS_DESCARTADOS = ["motivo", "titulo", "fecha", "minutos", "total", "identificados", "generos", "consultas", "id"]


class Detencion(RuntimeError):
    """El sitio respondió algo anormal: se para sin reintentar."""


def norm(texto: str) -> str:
    """Sin acentos ni mayúsculas; deja & para nombres como Pig&Dan."""
    texto = unicodedata.normalize("NFKD", texto or "")
    texto = "".join(c for c in texto if not unicodedata.combining(c)).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9& ]+", " ", texto)).strip()


def slug(texto: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", norm(texto)).strip("-") or "x"


def limpiar(fragmento: str) -> str:
    texto = htmllib.unescape(re.sub(r"<[^>]+>", " ", fragmento or ""))
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", texto)).strip()


# ---------------------------------------------------------------- filas de los listados


def minutos_de_texto(texto: str) -> int:
    """'6h ', '2h 1m', '45m', '1d 2h' -> minutos."""
    total = 0
    for valor, unidad, factor in (("d", "d", 1440), ("h", "h", 60), ("m", "m", 1)):
        encontrado = re.search(r"(\d+)\s*" + unidad + r"\b", texto or "")
        if encontrado:
            total += int(encontrado.group(1)) * factor
    return total


def identificados_de_texto(texto: str) -> Tuple[int, int]:
    """'90/105' -> (90, 105); 'all /73' -> (73, 73); '00/15' -> (0, 15)."""
    encontrado = re.search(r"(all|\d+)\s*/\s*(\d+)", texto or "", re.I)
    if not encontrado:
        return 0, 0
    total = int(encontrado.group(2))
    return (total if encontrado.group(1).lower() == "all" else int(encontrado.group(1))), total


def _campo(bloque: str, titulo: str) -> str:
    encontrado = re.search(r'title="' + re.escape(titulo) + r'"[^>]*>(.*?)</div>', bloque, re.S)
    return limpiar(encontrado.group(1)) if encontrado else ""


def parsear_lista(raw: str) -> List[Dict]:
    """Las filas de una página de resultados: todo lo que se sabe sin abrir el set."""
    filas = []
    for bloque in re.split(r'(?=<div class="bItm action oItm)', raw)[1:]:
        bloque = bloque[:6000]
        enlace = re.search(r'class="bTitle"[^>]*>\s*<a href="([^"]+)"[^>]*>(.*?)</a>', bloque, re.S)
        identificador = re.search(r'data-id="([^"]+)"', bloque)
        if not enlace or not identificador:
            continue
        identificados, total = identificados_de_texto(_campo(bloque, "IDed tracks / total tracks"))
        vistas = re.search(r'title="tracklist views" data-count="(\d+)"', bloque)
        filas.append({
            "id": identificador.group(1),
            "ruta": htmllib.unescape(enlace.group(1)),
            "titulo": limpiar(enlace.group(2)),
            "fecha": _campo(bloque, "tracklist date"),
            "minutos": minutos_de_texto(_campo(bloque, "play time")),
            "identificados": identificados,
            "total": total,
            "vistas": int(vistas.group(1)) if vistas else 0,
            "generos": [g.strip() for g in _campo(bloque, "musicstyle(s)").split(",") if g.strip()],
            "iconos": sorted(set(re.findall(r'<i title="([^"]+)" class="fa', bloque))),
        })
    return filas


def resultados_declarados(raw: str) -> Optional[str]:
    plano = re.sub(r"<[^>]+>", " ", raw)
    encontrado = re.search(r'search result for\s+(?:"|&quot;)[^"&]*(?:"|&quot;)\s+([\d.,]+k?)', plano)
    return encontrado.group(1) if encontrado else None


# ---------------------------------------------------------------- reglas


def cargar_config(ruta: Path) -> Dict:
    config = json.loads(Path(ruta).read_text(encoding="utf-8"))
    config["_stream"] = re.compile(config["stream"])
    config["_dj_comerciales"] = {norm(d) for d in config["djs_comerciales"]}
    config["_semillas"] = {norm(d) for d in config["djs"]}
    return config


def partes_titulo(titulo: str) -> Tuple[str, str, str]:
    """(dj, resto, tipo). '@' es un evento real; 'X - Programa 12' es radio o podcast."""
    if " @ " in titulo:
        dj, resto = titulo.split(" @ ", 1)
        return dj.strip(), resto.strip(), "evento"
    if " - " in titulo:
        dj, resto = titulo.split(" - ", 1)
        return dj.strip(), resto.strip(), "programa"
    return "", titulo.strip(), "otro"


def djs_de(parte: str) -> List[str]:
    """Separa B2B: 'A & B', 'A b2b B', 'A + More'. No parte 'Pig&Dan'."""
    partes = re.split(r"\s+(?:&|\+|b2b|vs\.?|x|and)\s+|,\s*", parte or "", flags=re.I)
    return [p.strip() for p in partes if p.strip() and norm(p) != "more"]


def contiene_nombre(nombre: str, parte_dj: str) -> bool:
    """¿Toca este DJ? Nombre exacto entre los DJs del título; 'Anna' no es 'Anna Lunoe'.

    Un nombre de dos palabras o más admite un agregado ('Carl Cox presents X').
    """
    buscado = norm(nombre)
    for dj in djs_de(parte_dj):
        actual = norm(dj)
        if actual == buscado or (" " in buscado and actual.startswith(buscado + " ")):
            return True
    return False


def motivo_comercial(fila: Dict, config: Dict) -> Optional[str]:
    """Motivo por el que el set se considera comercial o fuera de estilo; None si pasa."""
    dj_parte, _, _ = partes_titulo(fila["titulo"])
    nombres = {norm(d) for d in djs_de(dj_parte)}
    if nombres & config["_dj_comerciales"]:
        return "comercial_dj"
    # Las semillas son el gusto de Gabriel: una etiqueta del sitio no las saca (Digweed figura como Mainstage en Sziget).
    if nombres & config["_semillas"]:
        return None
    generos = set(fila["generos"])
    if generos & set(config["generos_comerciales"]):
        return "comercial_genero"
    if generos & set(config["generos_fuera_de_estilo"]):
        return "fuera_de_estilo"
    if not generos & set(config["generos_aceptados"]):
        return "sin_genero_aceptado"
    return None


def evaluar(fila: Dict, config: Dict) -> Dict:
    """Agrega a la fila uso ('', trayectoria, pares, trayectoria+pares) y motivo de descarte."""
    dj_parte, resto, tipo = partes_titulo(fila["titulo"])
    salida = dict(fila, dj=dj_parte, tipo=tipo, resto=resto, uso="", motivo="")
    salida["b2b"] = len(djs_de(dj_parte)) > 1
    salida["varios"] = bool(re.search(r"\+\s*more\b", dj_parte, re.I))
    salida["es_stream"] = bool(config["_stream"].search(norm(fila["titulo"])))
    horas = fila["minutos"] / 60
    salida["temas_hora"] = round(fila["total"] / horas, 1) if horas else 0.0
    razon = fila["identificados"] / fila["total"] if fila["total"] else 0.0
    salida["pct_id"] = round(100 * razon)

    motivo = motivo_comercial(fila, config)
    if not motivo:
        if fila["minutos"] <= 0 or fila["total"] <= 0:
            motivo = "sin_duracion"
        elif fila["minutos"] < MIN_MINUTOS:
            motivo = "corto"
        elif not TEMAS_HORA[0] <= salida["temas_hora"] <= TEMAS_HORA[1]:
            motivo = "densidad"
    if motivo:
        salida["motivo"] = motivo
        return salida

    trayectoria = (tipo == "evento" and not salida["es_stream"] and not salida["varios"]
                   and fila["minutos"] >= MIN_MINUTOS_TRAYECTORIA and razon >= MIN_ID_TRAYECTORIA)
    pares = razon >= MIN_ID_PARES and fila["total"] * razon * razon >= MIN_PARES_ESPERADOS
    salida["uso"] = "+".join(n for n, activo in (("trayectoria", trayectoria), ("pares", pares)) if activo)
    if not salida["uso"]:
        salida["motivo"] = "pocos_identificados"
    return salida


def tiene_grabacion(fila: Dict) -> bool:
    return any(re.search(r"youtube|soundcloud|mixcloud|premium audio|apple music|audio link", i) for i in fila["iconos"])


def pesos_dj(biblioteca: Optional[Path], spotify: Optional[Path]) -> Dict[str, float]:
    """Afinidad de cada DJ con Gabriel: temas suyos en la biblioteca y si lo escucha en Spotify (0 a 1)."""
    conteo: Counter = Counter()
    if biblioteca and Path(biblioteca).exists():
        with Path(biblioteca).open(encoding="utf-8-sig", newline="") as handle:
            for fila in csv.DictReader(handle):
                for artista in re.split(r"\s*(?:&|,|\+|/|\bfeat\.?\b|\bft\.?\b|\bvs\.?\b)\s*", fila.get("artista") or ""):
                    if norm(artista):
                        conteo[norm(artista)] += 1
    tope = max(conteo.values()) if conteo else 1
    pesos = {a: 0.7 * math.log2(1 + n) / math.log2(1 + tope) for a, n in conteo.items()}
    if spotify and Path(spotify).exists():
        for linea in Path(spotify).read_text(encoding="utf-8").splitlines():
            if norm(linea):
                pesos[norm(linea)] = pesos.get(norm(linea), 0.0) + 0.3
    return pesos


def puntaje(fila: Dict, pesos: Dict[str, float]) -> float:
    """Orden dentro de cada uso; no decide qué entra. Identificación, afinidad, grabación y época."""
    razon = fila["identificados"] / fila["total"] if fila["total"] else 0.0
    afinidad = max((pesos.get(norm(d), 0.0) for d in djs_de(fila["dj"])), default=0.0)
    anio = int(fila["fecha"][:4]) if re.match(r"\d{4}", fila["fecha"] or "") else 2012
    epoca = max(0.0, min(1.0, (anio - 2012) / 14))
    return round(2 * razon + 2 * afinidad + (1.0 if tiene_grabacion(fila) else 0.0) + 0.5 * epoca, 3)


def programa_de(fila: Dict) -> str:
    """'Transitions 1149' y 'Transitions 1150' son el mismo programa."""
    return norm(re.sub(r"\d+", " ", fila.get("resto", ""))) if fila["tipo"] == "programa" else ""


def grupo_noche(fila: Dict) -> str:
    """Las noches publicadas en partes comparten DJ y título sin el número de parte."""
    titulo = re.sub(r"\b(part|pt|hour|cd|disc)\s*\d+\b", " ", norm(fila["titulo"]))
    if titulo == norm(fila["titulo"]):
        return ""
    return slug(titulo)


def unir_filas(archivos: Iterable[Path]) -> Tuple[List[Dict], Counter]:
    """Junta las búsquedas, una fila por set; recuerda qué consultas lo encontraron."""
    por_id: Dict[str, Dict] = {}
    otros = Counter()
    for ruta in sorted(archivos):
        dato = json.loads(ruta.read_text(encoding="utf-8"))
        consulta = dato["consulta"]
        for fila in dato["filas"]:
            dj_parte, _, _ = partes_titulo(fila["titulo"])
            if consulta["tipo"] == "dj" and not contiene_nombre(consulta["dj"], dj_parte):
                otros[consulta["id"]] += 1
                continue
            existente = por_id.setdefault(fila["id"], dict(fila, consultas=[]))
            existente["consultas"].append(consulta["id"])
    return list(por_id.values()), otros


def seleccionar(filas: List[Dict], pesos: Dict[str, float]) -> Tuple[List[Dict], List[Dict]]:
    """Ordena por uso y puntaje, y aplica los topes por DJ y por programa."""
    aceptados, descartados = [], []
    for fila in filas:
        fila["puntaje"] = puntaje(fila, pesos)
    orden = sorted(filas, key=lambda f: ("trayectoria" not in f["uso"], -f["puntaje"], f["id"]))
    por_dj: Counter = Counter()
    por_programa: Counter = Counter()
    for fila in orden:
        if not fila["uso"]:
            descartados.append(fila)
            continue
        dj = norm((djs_de(fila["dj"]) or [fila["dj"]])[0])
        programa = programa_de(fila)
        if por_dj[dj] >= TOPE_POR_DJ:
            descartados.append(dict(fila, motivo="tope_dj"))
        elif programa and por_programa[programa] >= TOPE_POR_PROGRAMA:
            descartados.append(dict(fila, motivo="tope_programa"))
        else:
            por_dj[dj] += 1
            if programa:
                por_programa[programa] += 1
            aceptados.append(fila)
    return aceptados, descartados


# ---------------------------------------------------------------- páginas de set


def texto_visible(bloque: str) -> str:
    """El nombre tal como se ve en la fila; los span van anidados, así que se cuentan hasta cerrar el primero."""
    apertura = re.search(r'<span class="trackValue[^"]*"[^>]*>', bloque)
    if not apertura:
        return ""
    profundidad, posicion = 1, apertura.end()
    for marca in re.finditer(r"<span[\s>]|</span>", bloque[posicion:]):
        profundidad += 1 if marca.group(0) != "</span>" else -1
        if profundidad == 0:
            return limpiar(bloque[posicion:posicion + marca.start()])
    return ""


def parsear_set(raw: str) -> Dict:
    """Una página de set: lo mismo que da el botón de exportar y además id, sello, enlaces y grabación."""
    titulo = re.search(r"<title>(.*?)</title>", raw, re.S)
    canon = re.search(r'<link rel="canonical" href="([^"]+)"', raw)
    temas = []
    for bloque in re.split(r'(?=<div id="tlp_\d+" class="[^"]*tlpItem)', raw)[1:]:
        bloque = bloque[:14000]
        cabecera = re.match(r'<div id="tlp_(\d+)" class="([^"]*)"([^>]*)>', bloque)
        atributo = lambda n: (re.search(n + r'="([^"]*)"', cabecera.group(3)) or [None, ""])[1]  # noqa: E731
        meta = lambda n: htmllib.unescape((re.search(r'itemprop="' + n + r'" content="([^"]*)"', bloque) or [None, ""])[1])  # noqa: E731
        numero = re.search(r'_tracknumber_value"[^>]*>\s*([^<]*?)\s*<', bloque)
        cue = re.search(r'_cue_seconds"[^>]*value="(\d+)"', bloque)
        edicion = re.search(r'class="italic trackEditData[^"]*"[^>]*>(.*?)</span>', bloque, re.S)
        duracion = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$", meta("duration"))
        id_num = re.search(r'class="iRow grow mediaRow" data-trackid="(\d+)"', bloque)
        temas.append({
            "tlp": cabecera.group(1),
            "numero": numero.group(1) if numero else "",
            "junto_al_anterior": "con" in cabecera.group(2).split(),
            "identificado": atributo("data-isided") == "true",
            "segundo": int(cue.group(1)) if cue else None,
            "nombre": unicodedata.normalize("NFC", meta("name")),
            "visible": texto_visible(bloque),
            "edicion": limpiar(edicion.group(1)) if edicion else "",
            "artista": unicodedata.normalize("NFC", meta("byArtist")),
            "sello": limpiar(meta("publisher")),
            "duracion_s": (int(duracion.group(1) or 0) * 3600 + int(duracion.group(2) or 0) * 60
                           + int(duracion.group(3) or 0)) if duracion else None,
            "genero": meta("genre"),
            "ruta_tema": meta("url"),
            "id_texto": atributo("data-trackid"),
            "id_numerico": id_num.group(1) if id_num else "",
            "fuentes": sorted(set(re.findall(r"idSource: '([^']+)'", bloque))),
        })
    grabaciones = []
    for encontrado in re.finditer(r'data-idmedia="(\d+)" data-idsource="(\d+)" class="([^"]*)"', raw):
        tramo = raw[encontrado.end():encontrado.end() + 4000]
        iframe = re.search(r'<iframe[^>]+src="([^"]+)"', tramo)
        largo = re.search(r'itemprop="duration" content="PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?"', tramo)
        externo = re.search(r"mediaItem(\S+)", encontrado.group(3))
        grabaciones.append({
            "id_medio": encontrado.group(1),
            "fuente": encontrado.group(2),
            "id_externo": externo.group(1) if externo else "",
            "iframe": htmllib.unescape(iframe.group(1)) if iframe else "",
            "duracion_s": (int(largo.group(1) or 0) * 3600 + int(largo.group(2) or 0) * 60 + int(largo.group(3) or 0)) if largo else None,
        })
    return {
        "titulo": limpiar(titulo.group(1)) if titulo else "",
        "url": canon.group(1) if canon else "",
        "temas": temas,
        "grabaciones": grabaciones,
    }


def clave_tema(artista: str, titulo: str) -> Optional[Tuple[str, str]]:
    """(primer artista, título con la versión) normalizados igual que el cruce de D3 con la biblioteca (zona_sets.py)."""
    from src.v4.evaluation.zona_sets import estricto, primer_artista

    clave = (estricto(primer_artista(artista)), estricto(titulo))
    return clave if all(clave) else None


def clave_de_nombre(nombre: str) -> Optional[Tuple[str, str]]:
    if " - " not in (nombre or ""):
        return None
    artista, titulo = nombre.split(" - ", 1)
    return clave_tema(artista, titulo)


def medir_set(dato: Dict, artistas_biblioteca: set, generos_comerciales: set, claves_biblioteca: Optional[set] = None) -> Dict:
    """Calidad de un set bajado. Los pares ignoran las filas tocadas encima de otra. Con claves_biblioteca, cuenta además
    los temas y pares que están en la biblioteca (lo que necesita E13/E14 de la auditoría del 2026-11-09)."""
    temas = [t for t in dato["temas"] if not t["junto_al_anterior"]]
    n = len(temas)
    claves = [clave_de_nombre(t["nombre"] or t["visible"]) for t in temas] if claves_biblioteca else [None] * n
    en_biblioteca = [c is not None and c in claves_biblioteca for c in claves]
    distintos = len({c for c, dentro in zip(claves, en_biblioteca) if dentro})
    identificados = sum(t["identificado"] for t in temas)
    con_tiempo = sum(1 for i, t in enumerate(temas) if t["segundo"] is not None and (t["segundo"] > 0 or i == 0))
    pares = sum(a["identificado"] and b["identificado"] for a, b in zip(temas, temas[1:]))
    tramo = mejor = 0
    for tema in temas:
        tramo = tramo + 1 if tema["identificado"] else 0
        mejor = max(mejor, tramo)
    nombrados = [t for t in temas if " - " in (t["nombre"] or t["visible"])]
    afines = sum(1 for t in nombrados if {norm(a) for a in re.split(r"\s*(?:&|,|\+|/)\s*", (t["nombre"] or t["visible"]).split(" - ", 1)[0])} & artistas_biblioteca)
    con_genero = [t for t in temas if t["genero"]]
    return {
        "titulo": dato["titulo"],
        "temas": n,
        "pct_id": round(100 * identificados / max(1, n)),
        "con_tiempo": con_tiempo,
        "pct_tiempo": round(100 * con_tiempo / max(1, n)),
        "pares_consecutivos": pares,
        "pct_pares": round(100 * pares / max(1, n - 1)),
        "tramo_mas_largo": mejor,
        "pct_artista_en_biblioteca": round(100 * afines / max(1, len(nombrados))),
        "temas_en_biblioteca": sum(en_biblioteca),
        "pares_consecutivos_en_biblioteca": sum(a and b for a, b in zip(en_biblioteca, en_biblioteca[1:])),
        "pares_mismo_set_en_biblioteca": distintos * (distintos - 1) // 2,
        "pct_genero_comercial": round(100 * sum(t["genero"] in generos_comerciales for t in con_genero) / max(1, len(con_genero))),
        "pct_beatport": round(100 * sum("affiliate" in t["fuentes"] for t in temas if t["identificado"]) / max(1, identificados)),
        "pct_youtube": round(100 * sum("13" in t["fuentes"] for t in temas if t["identificado"]) / max(1, identificados)),
        "grabaciones": len(dato["grabaciones"]),
        "fuentes_grabacion": " ".join(sorted({g["fuente"] for g in dato["grabaciones"]})),
    }


# ---------------------------------------------------------------- consultas


def generar_consultas(config: Dict) -> List[Dict]:
    """DJ: 60 min o más, los más largos primero. El buscador pide 5 letras o más."""
    consultas = []
    for dj in config["djs"]:
        texto = dj if len(dj) >= 5 else dj + " @"
        consultas.append({"id": "dj__" + slug(dj), "tipo": "dj", "dj": dj, "texto": texto, "duracion": 60, "Type": "0", "orderby": "duration"})
    for termino in config["terminos"]:
        for ventana in termino.get("ventanas") or [{}]:
            sufijo = "__" + ventana["desde"][:4] if ventana.get("desde") else ""
            consultas.append({
                "id": "termino__" + slug(termino["texto"]) + sufijo, "tipo": "termino", "texto": termino["texto"],
                "duracion": termino["duracion"], "Type": termino["Type"], "orderby": "duration",
                "desde": ventana.get("desde", ""), "hasta": ventana.get("hasta", ""),
            })
    for evento in config["eventos"]:
        consultas.append({"id": "evento__" + slug(evento), "tipo": "evento", "texto": evento, "duracion": 120, "Type": "0", "orderby": "duration"})
    return consultas


def campos_formulario(consulta: Dict) -> Dict[str, str]:
    """Los mismos campos que manda el formulario del sitio al aplicar sus filtros."""
    campos = {
        "main_search": consulta["texto"], "search_selection": "9", "filterObject": "9",
        "duration": str(consulta["duracion"]), "Type": consulta["Type"], "orderby": consulta["orderby"],
    }
    if consulta.get("desde"):
        campos["startDate"] = consulta["desde"]
    if consulta.get("hasta"):
        campos["endDate"] = consulta["hasta"]
    return campos


ENVIAR_FORMULARIO = """(campos) => {
  const f = document.createElement('form');
  f.method = 'post'; f.action = '/search/result.php'; f.acceptCharset = 'utf-8';
  for (const [k, v] of Object.entries(campos)) {
    const i = document.createElement('input');
    i.type = 'hidden'; i.name = k; i.value = v; f.appendChild(i);
  }
  document.body.appendChild(f); f.submit();
}"""


# ---------------------------------------------------------------- navegador


@contextlib.contextmanager
def navegador(perfil: Path, esperar_login: int):
    from playwright.sync_api import sync_playwright

    perfil.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        contexto = playwright.chromium.launch_persistent_context(user_data_dir=str(perfil), headless=False, args=["--mute-audio"])
        try:
            pagina = contexto.new_page()
            estado: Dict[str, int] = {}
            pagina.on("response", lambda r: estado.__setitem__("valor", r.status) if r.request.is_navigation_request() else None)
            pagina.goto(BASE + "/", wait_until="domcontentloaded", timeout=TIMEOUT_MS)
            pagina.wait_for_timeout(ASENTAR_MS)
            if not pagina.locator('[title="sign out"]').count():
                if esperar_login <= 0:
                    raise Detencion("sin sesión: correr con --esperar-login 300 e iniciarla a mano en la ventana")
                print(f"Iniciá sesión en la ventana; espero {esperar_login} s.", flush=True)
                pagina.wait_for_selector('[title="sign out"]', timeout=esperar_login * 1000)
            yield pagina, estado
        finally:
            contexto.close()


def problema_de(pagina, estado: Dict[str, int]) -> Optional[str]:
    if estado.get("valor") not in (200, 304):
        return f"estado HTTP {estado.get('valor')}"
    if pagina.locator("#iScrollCaptcha, #captcha, iframe[src*='challenges.cloudflare']").count():
        return "captcha"
    if not pagina.locator('[title="sign out"]').count():
        return "se cerró la sesión"
    return None


def _gz(ruta: Path, texto: str) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(ruta, "wt", encoding="utf-8", newline="") as handle:
        handle.write(texto)


def _json(ruta: Path, dato) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(dato, ensure_ascii=False, indent=1), encoding="utf-8", newline="")


# ---------------------------------------------------------------- comandos


def cmd_buscar(args) -> int:
    consultas = generar_consultas(cargar_config(args.config))
    destino = args.root / "1001tracklists" / "busquedas"
    if args.solo:
        consultas = [c for c in consultas if any(s in c["id"] for s in args.solo)]
    pendientes = [c for c in consultas if not (destino / f"{c['id']}.json").exists()]
    if args.limite:
        pendientes = pendientes[:args.limite]
    print(f"{len(consultas)} consultas, {len(pendientes)} pendientes; unos {len(pendientes) * 11 / 60:.0f} min", flush=True)
    if args.dry_run or not pendientes:
        for consulta in pendientes:
            print(consulta["id"], campos_formulario(consulta))
        return 0
    vacias = 0
    with navegador(args.perfil, args.esperar_login) as (pagina, estado):
        for indice, consulta in enumerate(pendientes, 1):
            pagina.wait_for_timeout(PAUSA_MS)
            with pagina.expect_navigation(wait_until="domcontentloaded", timeout=TIMEOUT_MS):
                pagina.evaluate(ENVIAR_FORMULARIO, campos_formulario(consulta))
            pagina.wait_for_timeout(ASENTAR_MS)
            problema = problema_de(pagina, estado)
            if problema:
                print(f"PARO en {consulta['id']}: {problema}. No sigo ni reintento.", flush=True)
                return 2
            raw = pagina.content()
            filas = parsear_lista(raw)
            duracion = re.findall(r'<input[^>]*id="duration"[^>]*value="([^"]*)"', raw)
            _gz(destino / f"{consulta['id']}.html.gz", raw)
            _json(destino / f"{consulta['id']}.json", {"consulta": consulta, "resultados": resultados_declarados(raw), "duracion_en_formulario": duracion[:1], "filas": filas})
            minutos = [f["minutos"] for f in filas]
            print(f"[{indice}/{len(pendientes)}] {consulta['id']}: filas={len(filas)} declarados={resultados_declarados(raw)} filtro={duracion[:1]} min={min(minutos) if minutos else '-'} max={max(minutos) if minutos else '-'}", flush=True)
            vacias = vacias + 1 if not filas else 0
            if vacias >= MAX_VACIAS:
                print(f"PARO: {vacias} búsquedas seguidas sin filas; puede ser un bloqueo. No sigo.", flush=True)
                return 2
    return 0


def cmd_candidatos(args) -> int:
    config = cargar_config(args.config)
    base = args.root / "1001tracklists"
    filas, otros = unir_filas((base / "busquedas").glob("*.json"))
    evaluadas = [evaluar(f, config) for f in filas]
    aceptados, descartados = seleccionar(evaluadas, pesos_dj(args.biblioteca, args.spotify))
    pesos = pesos_dj(args.biblioteca, args.spotify)
    previos = {}
    salida = base / "candidatos.csv"
    if salida.exists():
        with salida.open(encoding="utf-8-sig", newline="") as handle:
            previos = {f["id"]: f.get("veto", "") for f in csv.DictReader(handle)}
    grupos = Counter(grupo_noche(f) for f in aceptados)
    with salida.open("w", encoding="utf-8-sig", newline="") as handle:
        escritor = csv.DictWriter(handle, fieldnames=COLUMNAS_CANDIDATOS, extrasaction="ignore")
        escritor.writeheader()
        for fila in aceptados:
            grupo = grupo_noche(fila)
            escritor.writerow(dict(
                fila, veto=previos.get(fila["id"], ""), generos=", ".join(fila["generos"]),
                grabacion="si" if tiene_grabacion(fila) else "",
                afinidad_dj=round(max((pesos.get(norm(d), 0.0) for d in djs_de(fila["dj"])), default=0.0), 2),
                b2b="si" if fila["b2b"] else "", grupo_noche=grupo if grupos[grupo] > 1 else "",
                consultas=" ".join(fila["consultas"]),
            ))
    with (base / "descartados.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        escritor = csv.DictWriter(handle, fieldnames=COLUMNAS_DESCARTADOS, extrasaction="ignore")
        escritor.writeheader()
        for fila in sorted(descartados, key=lambda f: (f["motivo"], f["titulo"])):
            escritor.writerow(dict(fila, generos=", ".join(fila["generos"]), consultas=" ".join(fila["consultas"])))
    usos = Counter(f["uso"] for f in aceptados)
    motivos = Counter(f["motivo"] for f in descartados)
    resumen = {
        "sets_unicos": len(evaluadas), "candidatos": len(aceptados), "por_uso": dict(usos),
        "minutos_trayectoria": sum(f["minutos"] for f in aceptados if "trayectoria" in f["uso"]),
        "con_grabacion": sum(tiene_grabacion(f) for f in aceptados),
        "descartados_por_motivo": dict(motivos.most_common()),
        "filas_de_otro_dj": sum(otros.values()),
        "djs_distintos": len({norm((djs_de(f["dj"]) or [f["dj"]])[0]) for f in aceptados}),
    }
    _json(base / "resumen_candidatos.json", resumen)
    print(json.dumps(resumen, ensure_ascii=False, indent=1))
    return 0


def sirve_para(fila: Dict, uso: Optional[str]) -> bool:
    """'pares' toma solo los sets que no son de trayectoria (los de trayectoria también dan pares)."""
    return not uso or (uso == "pares" and fila["uso"] == "pares") or (uso == "trayectoria" and "trayectoria" in fila["uso"])


def cmd_bajar(args) -> int:
    base = args.root / "1001tracklists"
    with (base / "candidatos.csv").open(encoding="utf-8-sig", newline="") as handle:
        tabla = list(csv.DictReader(handle))
    pendientes = [f for f in tabla if not f["veto"].strip() and sirve_para(f, args.uso) and not (base / "sets" / f"{f['id']}.json").exists()]
    if args.por_dj:
        vistos: Counter = Counter()
        elegidos = []
        for fila in pendientes:
            dj = norm((djs_de(fila["dj"]) or [fila["dj"]])[0])
            if vistos[dj] < args.por_dj:
                vistos[dj] += 1
                elegidos.append(fila)
        pendientes = elegidos
    pendientes = pendientes[:args.n]
    print(f"{len(tabla)} candidatos, bajo {len(pendientes)}; unos {len(pendientes) * 11 / 60:.0f} min", flush=True)
    if args.dry_run or not pendientes:
        for fila in pendientes:
            print(fila["uso"], fila["titulo"])
        return 0
    with navegador(args.perfil, args.esperar_login) as (pagina, estado):
        for indice, fila in enumerate(pendientes, 1):
            pagina.wait_for_timeout(PAUSA_MS)
            pagina.goto(BASE + fila["ruta"], wait_until="domcontentloaded", timeout=TIMEOUT_MS)
            pagina.wait_for_timeout(ASENTAR_MS)
            problema = problema_de(pagina, estado)
            if problema:
                print(f"PARO en {fila['id']}: {problema}. No sigo ni reintento.", flush=True)
                return 2
            raw = pagina.content()
            dato = parsear_set(raw)
            if not dato["temas"]:
                print(f"PARO en {fila['id']}: la página no trae temas. No sigo.", flush=True)
                return 2
            _gz(base / "sets" / f"{fila['id']}.html.gz", raw)
            _json(base / "sets" / f"{fila['id']}.json", dict(dato, id=fila["id"], uso=fila["uso"], fecha=fila["fecha"], minutos=int(fila["minutos"])))
            print(f"[{indice}/{len(pendientes)}] {fila['id']} {dato['titulo'][:70]}: {len(dato['temas'])} temas", flush=True)
    return 0


def cmd_medir(args) -> int:
    config = cargar_config(args.config)
    base = args.root / "1001tracklists"
    artistas, claves = set(), set()
    with Path(args.biblioteca).open(encoding="utf-8-sig", newline="") as handle:
        for fila in csv.DictReader(handle):
            artistas |= {norm(a) for a in re.split(r"\s*(?:&|,|\+|/|\bfeat\.?\b|\bft\.?\b|\bvs\.?\b)\s*", fila.get("artista") or "") if len(norm(a)) > 2}
            clave = clave_tema(fila.get("artista") or "", fila.get("titulo") or "")
            if clave:
                claves.add(clave)
    medidas = []
    for ruta in sorted((base / "sets").glob("*.json")):
        dato = json.loads(ruta.read_text(encoding="utf-8"))
        medidas.append(dict(medir_set(dato, artistas, set(config["generos_comerciales"]), claves), id=dato["id"], uso=dato["uso"], minutos=dato["minutos"]))
    if not medidas:
        print("No hay sets bajados.")
        return 1
    with (base / "medidas.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        escritor = csv.DictWriter(handle, fieldnames=list(medidas[0]))
        escritor.writeheader()
        escritor.writerows(medidas)
    for medida in medidas:
        print(f"{medida['uso']:18s} {medida['titulo'][:48]:48s} n={medida['temas']:3d} id={medida['pct_id']:3d}% t={medida['pct_tiempo']:3d}% pares={medida['pct_pares']:3d}% tramo={medida['tramo_mas_largo']:3d} afin={medida['pct_artista_en_biblioteca']:3d}% com={medida['pct_genero_comercial']:3d}% bp={medida['pct_beatport']:3d}% grab={medida['fuentes_grabacion']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="carpeta de datos de sets, fuera de Git")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--perfil", type=Path, default=DEFAULT_PERFIL, help="perfil del navegador con la sesión; fuera del repo")
    parser.add_argument("--esperar-login", type=int, default=0, help="segundos para iniciar sesión a mano si hace falta")
    comandos = parser.add_subparsers(dest="comando", required=True)
    buscar = comandos.add_parser("buscar")
    buscar.add_argument("--solo", action="append", help="solo las consultas cuyo id contiene este texto")
    buscar.add_argument("--limite", type=int, default=40, help="tope de búsquedas por corrida (el 2026-11-09 el sitio cortó la conexión tras 80 seguidas); 0, sin tope")
    buscar.add_argument("--dry-run", action="store_true")
    buscar.set_defaults(func=cmd_buscar)
    candidatos = comandos.add_parser("candidatos")
    candidatos.add_argument("--biblioteca", type=Path, help="_organizacion.csv del paquete, para la afinidad de cada DJ")
    candidatos.add_argument("--spotify", type=Path, default=DEFAULT_SPOTIFY)
    candidatos.set_defaults(func=cmd_candidatos)
    bajar = comandos.add_parser("bajar")
    bajar.add_argument("--n", type=int, default=50, help="tope de páginas por corrida")
    bajar.add_argument("--uso", choices=["trayectoria", "pares"], help="trayectoria: los de 3 h o más; pares: los que son solo de pares")
    bajar.add_argument("--por-dj", type=int, default=0, help="tope de sets por DJ en esta corrida (un piloto variado: 1)")
    bajar.add_argument("--dry-run", action="store_true")
    bajar.set_defaults(func=cmd_bajar)
    medir = comandos.add_parser("medir")
    medir.add_argument("--biblioteca", type=Path, required=True)
    medir.set_defaults(func=cmd_medir)
    args = parser.parse_args()
    try:
        return args.func(args)
    except Detencion as exc:
        print(f"PARO: {exc}", flush=True)
        return 2
    except Exception as exc:
        # Conexión cortada o timeout del navegador: el sitio puede estar rechazando, así que tampoco se reintenta.
        if not type(exc).__module__.startswith("playwright"):
            raise
        print(f"PARO: error de conexión ({str(exc).splitlines()[0][:120]}). No sigo ni reintento; lo ya guardado queda.", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
