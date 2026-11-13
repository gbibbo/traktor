"""
PURPOSE: El ejecutor del formulario del teléfono (plan de recomendación v2, sección 3; ideas L2 y L8): recibe un formulario
         ya validado (formulario.py) y la biblioteca, filtra, puntúa, corta y dice qué hizo. Es siempre igual: el LLM y los
         presets solo llenan casillas.
         Orden de los pasos, cada uno con cuántos temas quedan:
           1. biblioteca;
           2. referencia: «lo que suena» (del registro del vivo) o un tema nombrado (coincidencia aproximada sobre títulos
              y artistas; si hay dudas, devuelve candidatos y ninguna lista). Con referencia, el puntaje y los filtros
              son los de Parecidos, tal como están por defecto en el vivo (recomendador.py, sugerencias_vivo.json): BPM
              ±20, tonalidad compatible, sin lo ya tocado. Sin referencia, se saca lo ya tocado esta noche;
           3. géneros: excluir; incluir duro (solo esos) o blando (esos primero, y entran los vecinos de la taxonomía con
              parecido >= vecino_minimo; los temas sin género quedan fuera);
           4. BPM (rango, o relativo a la referencia); 5. tonalidad; 6. voz; 7. año, sello y artista.
         Puntaje: con referencia, el de Parecidos; con texto, el coseno CLAP del promedio de las frases (la lista de hoy);
         con los dos, el promedio de sus percentiles (regla nueva, a escuchar); sin ninguno, lo más típico del grupo que
         pasó los filtros (coseno con su centro). Después se corta en «largo» y, si se pidió, se ordena por BPM.
         El momento de la noche ordena y no filtra (tabla medida en DJSets). Lo que todavía no existe (after, «como tal
         DJ», perfiles de pesos, orden por momento) se ignora con un aviso.
CHANGELOG:
  - 2026-11-13: Creación inicial (línea B, unidad B1).
  - 2026-11-13: Sin «tonalidad libre»: el filtro de tonalidad del vivo es regla de Gabriel y queda («ignora la
                tonalidad» baja su peso; valores por defecto de la auditoría, DECISIONS 2026-11-13 punto 9).
  - 2026-11-13: El momento ordena y no filtra (DECISIONS 2026-11-13, punto 10; tabla config/telefono/momentos.json):
                promedio de percentiles entre el puntaje de siempre y el del momento (cuánto más típico es el género
                en ese momento que en la noche; con referencia, también la cercanía al BPM que suena más el
                corrimiento medido). After no tiene datos y se ignora con aviso.
"""
from __future__ import annotations

import copy
import difflib
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v4.common import harmonic  # noqa: E402
from tools.traktor_live import formulario as fm  # noqa: E402
from tools.traktor_live import recomendador as rc  # noqa: E402

TEMA_CLARO = 0.75            # coincidencia de un tema nombrado: puntaje mínimo del primero
TEMA_MARGEN = 0.10           # y ventaja sobre el segundo; si no, el teléfono muestra candidatos
TEMA_MINIMO = 0.40           # por debajo, no es candidato
MAX_CANDIDATOS = 5
RUIDO = re.compile(r"\b(original|extended|radio|club|mix|edit|remaster(ed)?|feat|ft|vs)\b")
VACIAS = {"de", "del", "la", "el", "los", "las", "y", "a", "un", "una", "como", "tipo", "tema", "algo", "parecido",
          "the", "by", "of"}                      # palabras del pedido que no nombran el tema


@dataclass
class Sesion:
    """Lo que registra el vivo: el tema principal (deck, path, bpm, tempo, key, title, artist) y lo ya tocado."""
    principal: Optional[dict] = None
    tocados: List[str] = field(default_factory=list)


@dataclass
class Resultado:
    indices: List[int]                       # en bib.items, en el orden de la lista
    puntajes: List[float]
    pasos: List[dict]                        # [{"paso": texto, "quedan": n}]
    ignorado: List[dict]                     # [{"casilla": ..., "motivo": ...}]
    referencia: Optional[dict] = None        # {"titulo", "artista"} del tema de referencia
    candidatos: List[dict] = field(default_factory=list)   # tema nombrado con dudas: [{"uid", "titulo", "artista"}]


# ---------------------------------------------------------------- tema nombrado (L8)

def normalizar(texto: str) -> str:
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode("ascii").lower()
    t = re.sub(r"\.(mp3|wav|flac|aiff?|m4a)$", "", t)
    t = re.sub(r"^\s*\d+[.)]?\s+", "", t)                 # «02. Artista - Tema»
    t = re.sub(r"[\(\[]\s*(1[0-2]|[1-9])[ab]\s*[\)\]]", " ", t)   # «(9A)»
    t = RUIDO.sub(" ", re.sub(r"[^a-z0-9]+", " ", t))
    return re.sub(r"\s+", " ", t).strip()


def _cubre(q: str, palabras: Sequence[str]) -> float:
    if q in palabras:
        return 1.0
    best = max((difflib.SequenceMatcher(None, q, w).ratio() for w in palabras), default=0.0)
    return best if best >= 0.85 else 0.0                 # una letra de más o de menos («fischer»)


def puntaje_tema(busca: str, it) -> float:
    """Cuánto del texto buscado está en artista + título + nombre del archivo (0 a 1)."""
    q = [w for w in normalizar(busca).split() if w not in VACIAS] or normalizar(busca).split()
    if not q:
        return 0.0
    nombre = Path(str(it.rel).replace("\\", "/")).name
    lado = normalizar(f"{it.artist} {it.title} {nombre}")
    palabras = lado.split()
    cobertura = sum(_cubre(w, palabras) for w in q) / len(q)
    parecido = difflib.SequenceMatcher(None, " ".join(q), normalizar(f"{it.artist} {it.title}")).ratio()
    return 0.85 * cobertura + 0.15 * parecido


def resolver_tema(busca: str, items: Sequence) -> Tuple[Optional[int], List[Tuple[float, int]]]:
    """(índice del tema si es claro, candidatos ordenados [(puntaje, índice)])."""
    ranking = sorted(((puntaje_tema(busca, it), i) for i, it in enumerate(items)), key=lambda x: (-x[0], x[1]))
    ranking = [(s, i) for s, i in ranking if s >= TEMA_MINIMO][:MAX_CANDIDATOS]
    if ranking and ranking[0][0] >= TEMA_CLARO and (len(ranking) == 1 or ranking[0][0] - ranking[1][0] >= TEMA_MARGEN):
        return ranking[0][1], ranking
    return None, ranking


# ---------------------------------------------------------------- utilidades

def _titulo(it) -> Tuple[str, str]:
    e = getattr(it, "entry", None)
    title = (e.get("TITLE") if e is not None else "") or it.title or Path(str(it.rel).replace("\\", "/")).stem
    artist = (e.get("ARTIST") if e is not None else "") or it.artist or ""
    return title[:160], artist[:120]


def _percentiles(values: Sequence[float]) -> List[float]:
    return [1.0 if p is None else p for p in rc.percentiles(list(values))]


def _contiene(nombre: str, texto: Optional[str]) -> bool:
    """Un sello o un artista pedido, dentro del campo (sin tildes ni mayúsculas, palabras enteras)."""
    if not texto:
        return False
    n, t = normalizar(nombre), normalizar(texto)
    return bool(n) and re.search(rf"(^|\s){re.escape(n)}(\s|$)", t) is not None


def _pendiente(casilla: str) -> dict:
    return {"casilla": casilla, "motivo": fm.contrato()["pendiente"][casilla]}


# ---------------------------------------------------------------- el ejecutor

def ejecutar(form: dict, bib, sesion: Optional[Sesion] = None,
             codificar: Optional[Callable[[Sequence[str]], object]] = None) -> Resultado:
    """Ejecuta un formulario validado sobre la biblioteca (Biblioteca de conceptos_agente: items, matrix, vocal, rec)."""
    import numpy as np
    c = fm.contrato()
    items = bib.items
    vivos = list(range(len(items)))
    pasos = [{"paso": "Biblioteca", "quedan": len(vivos)}]
    ignorado: List[dict] = []
    f = form["filtros"]
    otros = f["otros"]
    sesion = sesion or Sesion()
    rec = getattr(bib, "rec", None)
    tocados = list(sesion.tocados) if otros["excluir_tocados"] else []

    def paso(texto: str, keep: Callable[[int], bool]) -> None:
        nonlocal vivos
        vivos = [i for i in vivos if keep(i)]
        pasos.append({"paso": texto[:120], "quedan": len(vivos)})

    # 2. referencia
    ref = form["referencia"]
    principal: Optional[dict] = None
    ref_txt: Optional[dict] = None
    if ref["tipo"] == "sonando":
        if sesion.principal and sesion.principal.get("path"):
            principal = sesion.principal
            ref_txt = {"titulo": str(principal.get("title") or "")[:160], "artista": str(principal.get("artist") or "")[:120]}
        else:
            ignorado.append({"casilla": "referencia", "motivo": "No sé qué está sonando: el vivo no registró ningún tema "
                                                                "en esta sesión (¿está abierto Traktor con las sugerencias?)."})
    elif ref["tipo"] == "tema":
        elegido = next((i for i, it in enumerate(items) if it.uid == ref.get("uid")), None) if ref.get("uid") else None
        if elegido is None and ref.get("busca"):
            elegido, ranking = resolver_tema(ref["busca"], items)
            if elegido is None:
                if not ranking:
                    ignorado.append({"casilla": "referencia", "motivo": f"No encontré «{ref['busca']}» en la biblioteca."})
                else:
                    cand = [{"uid": items[i].uid, "titulo": _titulo(items[i])[0], "artista": _titulo(items[i])[1]}
                            for _, i in ranking]
                    return Resultado([], [], pasos, ignorado, None, cand)
        if elegido is not None:
            it = items[elegido]
            principal = {"path": str(it.path or it.rel), "bpm": it.bpm, "tempo": 1.0, "key": it.key}
            titulo, artista = _titulo(it)
            ref_txt = {"titulo": titulo, "artista": artista}
    elif ref["tipo"] == "dj":
        ignorado.append(_pendiente("referencia.dj"))

    ref_score: Dict[str, float] = {}
    ref_rank: Dict[str, int] = {}
    bpm0: Optional[float] = None
    key0 = "?"
    if principal is not None:
        if rec is None:
            ignorado.append({"casilla": "referencia", "motivo": "Sin la colección de Traktor no hay Parecidos."})
        else:
            r = copy.copy(rec)                                        # mismo recomendador, sin tocar el del vivo
            r.n = len(r.candidates)
            r.options = dict(rec.options)
            if isinstance(f["tonalidad"], dict):                   # una tonalidad dada reemplaza a la de lo que suena
                r.options["filtro_clave"] = False
            _, info = r.compute(principal, tocados)
            ref_score = dict(zip(info["uids"], info["puntaje"]))
            ref_rank.update({u: k for k, u in enumerate(info["uids"])})    # desempata como Parecidos (puntaje redondeado)
            bpm0 = info["principal"]["bpm"]
            key0 = info["principal"]["clave"] or "?"
            filtra = r.options["filtro_clave"] and key0 != "?"
            nombre = ref_txt["titulo"] if ref_txt and ref_txt["titulo"] else "la referencia"
            paso(f"Parecidos a «{nombre}» (BPM ±{rc.BPM_RANGE:g}{', tonalidad compatible' if filtra else ''}"
                 f"{', sin lo ya tocado' if tocados else ''})", lambda i: items[i].uid in ref_score)
    elif tocados and rec is not None:
        fuera = {it.uid for it in (rec.identify(p) for p in tocados) if it is not None}
        if fuera:
            paso("Sin lo ya tocado esta noche", lambda i: items[i].uid not in fuera)

    # 3. géneros
    g = f["generos"]
    nivel: Dict[int, int] = {}
    if g and g["excluir"]:
        excl = set(g["excluir"])
        paso("Sin " + ", ".join(g["excluir"]), lambda i: items[i].genre not in excl)
    if g and g["incluir"]:
        incl = set(g["incluir"])
        if g["modo"] == "duro":
            paso(", ".join(g["incluir"]), lambda i: items[i].genre in incl)
        else:
            tax = getattr(rec, "taxonomy", None) if rec is not None else None
            minimo = c["generos"]["vecino_minimo"]
            vecinos = set()
            if tax is None:
                ignorado.append({"casilla": "generos", "motivo": "Sin la taxonomía de géneros: entra solo el género pedido."})
            else:
                presentes = {items[i].genre for i in vivos if items[i].genre} - incl
                vecinos = {v for v in presentes if max((tax.compare((a, None), (v, None)) or 0.0) for a in incl) >= minimo}
            for i in vivos:
                nivel[i] = 0 if items[i].genre in incl else 1
            texto = ", ".join(g["incluir"]) + (f" primero; vecinos: {', '.join(sorted(vecinos))}" if vecinos else "")
            paso(texto, lambda i: items[i].genre in incl or items[i].genre in vecinos)

    # 4. BPM
    b = f["bpm"]
    if b and "rango" in b:
        lo, hi = b["rango"]
        paso(fm._bpm_texto(b), lambda i: items[i].bpm is not None and (lo is None or items[i].bpm >= lo)
             and (hi is None or items[i].bpm <= hi))
    elif b and "relativo" in b:
        if bpm0 is None:
            ignorado.append({"casilla": "filtros.bpm", "motivo": "«Más rápido o más lento» necesita un tema de referencia con BPM."})
        else:
            d0, d1 = c["bpm"]["relativo"][b["relativo"]]
            lo, hi = bpm0 + d0, bpm0 + d1
            paso(f"BPM {lo:.0f} a {hi:.0f} ({fm.RELATIVO[b['relativo']]} que {bpm0:.0f})",
                 lambda i: items[i].bpm is not None and lo <= items[i].bpm <= hi)

    # 5. tonalidad
    t = f["tonalidad"]
    if isinstance(t, dict):
        k = t["clave"]
        paso(f"Tonalidad compatible con {k}", lambda i: items[i].key != "?" and harmonic.best_key_shift(k, items[i].key)[0] >= rc.KEY_MIN)
    elif t in ("compatible", "igual"):
        if key0 == "?":
            ignorado.append({"casilla": "filtros.tonalidad", "motivo": "No hay tema de referencia con tonalidad conocida."})
        elif t == "igual":
            paso(f"Misma tonalidad ({key0})", lambda i: items[i].key == key0)
        elif not (rec is not None and rec.options.get("filtro_clave")):
            paso(f"Tonalidad compatible con {key0}", lambda i: items[i].key != "?" and harmonic.best_key_shift(key0, items[i].key)[0] >= rc.KEY_MIN)

    # 6. voz
    if f["voz"] in ("con", "sin"):
        quiere = f["voz"] == "con"
        paso("Con voz" if quiere else "Sin voces", lambda i: bib.vocal.get(items[i].uid) is quiere)

    # 7. año, sello, artista
    if otros["anio"]:
        lo, hi = otros["anio"]
        anios = {i: fm.anio_de(items[i]) for i in vivos}
        paso(f"Año {lo or '…'} a {hi or '…'}", lambda i: anios.get(i) is not None and (lo is None or anios[i] >= lo)
             and (hi is None or anios[i] <= hi))
    if otros["sellos"]:
        paso("Sello " + ", ".join(otros["sellos"]), lambda i: any(_contiene(s, fm.sello_de(items[i])) for s in otros["sellos"]))
    if otros["artistas"]:
        paso("De " + ", ".join(otros["artistas"]), lambda i: any(_contiene(a, items[i].artist) for a in otros["artistas"]))
    if otros["artistas_excluir"]:
        paso("Nada de " + ", ".join(otros["artistas_excluir"]),
             lambda i: not any(_contiene(a, items[i].artist) for a in otros["artistas_excluir"]))

    # momento: ordena, no filtra (DECISIONS 2026-11-13, punto 10)
    momento = None
    if form["momento"]:
        tabla = fm.momentos()
        entrada = (tabla.get("momentos") or {}).get(form["momento"]) or {}
        noche = tabla.get("generos_noche") or {}
        if entrada.get("bpm") is None or not entrada.get("generos") or not noche:
            ignorado.append(_pendiente("momento"))
        else:
            delta = entrada.get("bpm_delta")
            # con referencia, lo que suena corrido según lo medido; sin referencia, la mediana medida de ese momento
            objetivo = bpm0 + float(delta) if bpm0 is not None and delta is not None else entrada.get("bpm_mediana")
            momento = {"lift": {g: s / noche[g] for g, s in entrada["generos"].items() if noche.get(g)},
                       "objetivo": objetivo}

    # lo que todavía no existe
    if form["pesos"]["perfil"] != "por_defecto" or form["pesos"]["ajustes"]:
        ignorado.append(_pendiente("pesos.perfil"))
    if form["orden"] == "momento":
        ignorado.append(_pendiente("orden.momento"))

    # puntaje
    if not vivos:
        return Resultado([], [], pasos, ignorado, ref_txt)
    matrix = np.asarray(bib.matrix, dtype=np.float64)
    puntos: Optional[List[float]] = None
    if ref_score:
        puntos = [ref_score[items[i].uid] for i in vivos]
    if form["texto"]:
        if codificar is None:
            raise ValueError("Hace falta el codificador de texto para la casilla texto")
        q = np.asarray(codificar(list(form["texto"])), dtype=np.float64).mean(axis=0)
        q = q / np.linalg.norm(q)
        txt = (matrix[vivos] @ q).tolist()
        if puntos is None:
            puntos = txt
        else:
            puntos = [(a + b) / 2 for a, b in zip(_percentiles(puntos), _percentiles(txt))]
    if momento is not None:
        # géneros más típicos de ese momento que de la noche entera; con referencia, también el BPM cerca de lo que
        # suena corrido según lo medido (la apertura más lenta, el cierre más rápido)
        pm = _percentiles([momento["lift"].get(items[i].genre, 0.0) for i in vivos])
        if momento["objetivo"] is not None:
            obj = momento["objetivo"]
            cerca = _percentiles([0.0 if items[i].bpm is None else max(0.0, 1.0 - abs(items[i].bpm - obj) / rc.BPM_RANGE)
                                  for i in vivos])
            pm = [(2 * a + b) / 3 for a, b in zip(pm, cerca)]     # el género pesa el doble: el BPM se corre poco
        puntos = pm if puntos is None else [(a + b) / 2 for a, b in zip(_percentiles(puntos), pm)]
    if puntos is None:
        if fm.es_vacio(form):
            ignorado.append({"casilla": "formulario", "motivo": "El pedido no tiene nada que buscar."})
            return Resultado([], [], pasos, ignorado, ref_txt)
        centro = matrix[vivos].mean(axis=0)
        centro = centro / (np.linalg.norm(centro) or 1.0)
        puntos = (matrix[vivos] @ centro).tolist()
    orden = sorted(range(len(vivos)), key=lambda j: (nivel.get(vivos[j], 0), -puntos[j],
                                                    ref_rank.get(items[vivos[j]].uid, j), j))
    elegidos = orden[: form["largo"]]
    if form["orden"] in ("bpm_sube", "bpm_baja"):
        sube = form["orden"] == "bpm_sube"
        elegidos.sort(key=lambda j: (items[vivos[j]].bpm is None, (items[vivos[j]].bpm or 0) * (1 if sube else -1)))
    return Resultado([vivos[j] for j in elegidos], [float(min(1.0, max(-1.0, puntos[j]))) for j in elegidos],
                     pasos, ignorado, ref_txt)
