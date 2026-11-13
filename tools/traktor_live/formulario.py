"""
PURPOSE: El formulario de consulta del teléfono (plan de recomendación v2, sección 3.2; ideas L1, L3 y L4). El LLM y los
         presets llenan casillas de listas cerradas; el ordenador ejecuta el formulario siempre igual (ejecutor.py).
         Este módulo es el contrato del lado del ordenador:
         - validar(): normaliza un formulario y descarta, con un aviso por cada uno, los valores que no están en sus
           listas (config/telefono/formulario.json, que también lee el Worker);
         - componer(): junta presets elegidos en el teléfono (config/telefono/presets.json) en un formulario;
         - describir(): el formulario en palabras, para el teléfono;
         - catalogo(): lo que hay en la biblioteca, solo en números (géneros con su conteo, BPM, cobertura de cada campo),
           que es lo único que va al Worker. Ningún título, artista, sello ni ruta.
         Línea de comandos: python tools/traktor_live/formulario.py catalogo   (imprime el catálogo de la biblioteca)
CHANGELOG:
  - 2026-11-13: Creación inicial (línea B, unidad B0). La tonalidad ya no tiene «libre»: el filtro del vivo queda.
  - 2026-11-13: momentos(): la tabla de momentos de config/telefono/momentos.json, para el ejecutor.
"""
from __future__ import annotations

import copy
import json
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)

CONTRATO_PATH = REPO_ROOT / "config" / "telefono" / "formulario.json"
PRESETS_PATH = REPO_ROOT / "config" / "telefono" / "presets.json"
MOMENTOS_PATH = REPO_ROOT / "config" / "telefono" / "momentos.json"
VERSION = 1
CAMELOT = re.compile(r"^(1[0-2]|[1-9])[AB]$")
CASILLAS = ("version", "filtros", "referencia", "pesos", "momento", "texto", "orden", "largo")
FILTROS = ("bpm", "generos", "tonalidad", "voz", "otros")
OTROS = ("anio", "sellos", "artistas", "artistas_excluir", "excluir_tocados")


class FormularioInvalido(ValueError):
    """Un formulario con valores fuera de sus listas, en modo estricto."""

    def __init__(self, problemas: Sequence[str]):
        super().__init__("; ".join(problemas))
        self.problemas = list(problemas)


@lru_cache(maxsize=4)
def _leer(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def contrato(path: Path = CONTRATO_PATH) -> dict:
    return _leer(str(path))


def presets(path: Path = PRESETS_PATH) -> dict:
    return _leer(str(path))


def momentos(path: Path = MOMENTOS_PATH) -> dict:
    """La tabla de momentos medida en DJSets (tools/dj_sets/momentos_noche.py). Vacía si falta."""
    try:
        return _leer(str(path))
    except (OSError, ValueError):
        return {}


def vacio() -> dict:
    """El formulario sin nada pedido: todas las casillas en su valor por defecto."""
    c = contrato()
    return {"version": VERSION,
            "filtros": {"bpm": None, "generos": None, "tonalidad": None, "voz": "da_igual",
                        "otros": {"anio": None, "sellos": [], "artistas": [], "artistas_excluir": [],
                                  "excluir_tocados": True}},
            "referencia": {"tipo": "ninguna"},
            "pesos": {"perfil": "por_defecto", "ajustes": []},
            "momento": None, "texto": [], "orden": "puntaje", "largo": c["largo"]["defecto"]}


# ---------------------------------------------------------------- validar

def _numero(x) -> Optional[float]:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or x != x:
        return None
    return float(x)


def _nombres(raw, c: dict, casilla: str, problemas: List[str]) -> List[str]:
    """Lista de nombres libres (sellos, artistas): los resuelve el ordenador, nunca el LLM."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        problemas.append(f"{casilla}: no es una lista")
        return []
    out = []
    for x in raw:
        t = x.strip() if isinstance(x, str) else ""
        if 1 <= len(t) <= c["otros"]["max_caracteres"] and t not in out:
            out.append(t)
        else:
            problemas.append(f"{casilla}: valor descartado")
    if len(out) > c["otros"]["max_nombres"]:
        problemas.append(f"{casilla}: más de {c['otros']['max_nombres']} nombres")
    return out[: c["otros"]["max_nombres"]]


def _bpm(raw, c: dict, problemas: List[str]) -> Optional[dict]:
    if raw is None:
        return None
    if isinstance(raw, dict) and set(raw) == {"relativo"}:
        if raw["relativo"] in c["bpm"]["relativo"]:
            return {"relativo": raw["relativo"]}
        problemas.append(f"bpm.relativo fuera de lista: {raw['relativo']!r}")
        return None
    if isinstance(raw, dict) and set(raw) == {"rango"} and isinstance(raw["rango"], list) and len(raw["rango"]) == 2:
        lo_lim, hi_lim = c["bpm"]["limites"]
        lo, hi = raw["rango"]
        lo = None if lo is None else _numero(lo)
        hi = None if hi is None else _numero(hi)
        ok = (raw["rango"][0] is None or lo is not None) and (raw["rango"][1] is None or hi is not None)
        ok = ok and (lo is not None or hi is not None)
        ok = ok and all(v is None or lo_lim <= v <= hi_lim for v in (lo, hi))
        ok = ok and (lo is None or hi is None or lo <= hi)
        if ok:
            return {"rango": [lo, hi]}
    problemas.append("bpm: ni un rango válido ni un valor relativo de la lista")
    return None


def _generos(raw, c: dict, validos: Optional[Iterable[str]], problemas: List[str]) -> Optional[dict]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        problemas.append("generos: no es un objeto")
        return None
    validos = None if validos is None else set(validos)
    listas = {}
    for k in ("incluir", "excluir"):
        vals = raw.get(k) or []
        if not isinstance(vals, list):
            problemas.append(f"generos.{k}: no es una lista")
            vals = []
        limpio = []
        for g in vals:
            if isinstance(g, str) and (validos is None or g in validos) and g not in limpio:
                limpio.append(g)
            else:
                problemas.append(f"generos.{k}: {g!r} no está en el catálogo")
        if len(limpio) > c["generos"]["max"]:
            problemas.append(f"generos.{k}: más de {c['generos']['max']}")
        listas[k] = limpio[: c["generos"]["max"]]
    modo = raw.get("modo", "blando")
    if modo not in c["generos"]["modos"]:
        problemas.append(f"generos.modo fuera de lista: {modo!r}")
        modo = "blando"
    extra = set(raw) - {"incluir", "excluir", "modo"}
    if extra:
        problemas.append(f"generos: claves desconocidas {sorted(extra)}")
    if not listas["incluir"] and not listas["excluir"]:
        return None
    return {"incluir": listas["incluir"], "excluir": listas["excluir"], "modo": modo}


def _tonalidad(raw, c: dict, problemas: List[str]):
    if raw is None or raw in c["tonalidad"]:
        return raw
    if isinstance(raw, dict) and set(raw) == {"clave"} and isinstance(raw["clave"], str) \
            and CAMELOT.match(raw["clave"].strip().upper()):
        return {"clave": raw["clave"].strip().upper()}
    problemas.append(f"tonalidad fuera de lista: {raw!r}")
    return None


def _referencia(raw, c: dict, problemas: List[str]) -> dict:
    if raw is None:
        return {"tipo": "ninguna"}
    if not isinstance(raw, dict) or raw.get("tipo") not in c["referencia"]:
        problemas.append(f"referencia fuera de lista: {raw!r}")
        return {"tipo": "ninguna"}
    tipo = raw["tipo"]
    if tipo in ("ninguna", "sonando"):
        return {"tipo": tipo}
    out = {"tipo": tipo}
    busca = raw.get("busca")
    if isinstance(busca, str) and 1 <= len(busca.strip()) <= c["otros"]["max_caracteres"]:
        out["busca"] = busca.strip()
    uid = raw.get("uid")
    if tipo == "tema" and isinstance(uid, str) and re.fullmatch(r"[0-9A-Za-z_-]{1,80}", uid):
        out["uid"] = uid
    if "busca" not in out and "uid" not in out:
        problemas.append(f"referencia.{tipo}: falta qué buscar")
        return {"tipo": "ninguna"}
    return out


def _pesos(raw, c: dict, problemas: List[str]) -> dict:
    out = {"perfil": "por_defecto", "ajustes": []}
    if raw is None:
        return out
    if not isinstance(raw, dict):
        problemas.append("pesos: no es un objeto")
        return out
    p = c["pesos"]
    if raw.get("perfil", "por_defecto") in p["perfiles"]:
        out["perfil"] = raw.get("perfil", "por_defecto")
    else:
        problemas.append(f"pesos.perfil fuera de lista: {raw.get('perfil')!r}")
    for a in raw.get("ajustes") or []:
        if isinstance(a, dict) and set(a) == {"factor", "nivel"} and a["factor"] in p["factores"] \
                and a["nivel"] in p["niveles"] and all(x["factor"] != a["factor"] for x in out["ajustes"]):
            out["ajustes"].append({"factor": a["factor"], "nivel": a["nivel"]})
        else:
            problemas.append(f"pesos.ajustes: descartado {a!r}")       # nunca un número
    if len(out["ajustes"]) > p["max_ajustes"]:
        problemas.append(f"pesos.ajustes: más de {p['max_ajustes']}")
        out["ajustes"] = out["ajustes"][: p["max_ajustes"]]
    return out


def _texto(raw, c: dict, problemas: List[str]) -> List[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        problemas.append("texto: no es una lista")
        return []
    out = []
    for t in raw:
        s = t.strip() if isinstance(t, str) else ""
        if 3 <= len(s) <= c["texto"]["max_caracteres"] and re.fullmatch(r"[\x20-\x7e]+", s) and s not in out:
            out.append(s)
        else:
            problemas.append("texto: frase descartada (inglés, ASCII, de 3 a 180 caracteres)")
    if len(out) > c["texto"]["max"]:
        problemas.append(f"texto: más de {c['texto']['max']} frases")
    return out[: c["texto"]["max"]]


def validar(raw, generos_validos: Optional[Iterable[str]] = None, estricto: bool = False) -> Tuple[dict, List[str]]:
    """(formulario normalizado, problemas). Todo valor fuera de su lista se descarta y deja un problema; en modo estricto,
    cualquier problema es FormularioInvalido. generos_validos: los géneros del catálogo (None: no se comprueban)."""
    c = contrato()
    problemas: List[str] = []
    form = vacio()
    if not isinstance(raw, dict):
        problemas.append("el formulario no es un objeto")
        raw = {}
    if raw.get("version", VERSION) != VERSION:
        problemas.append(f"versión desconocida: {raw.get('version')!r}")
        raw = {}
    for k in set(raw) - set(CASILLAS):
        problemas.append(f"casilla desconocida: {k}")
    filtros = raw.get("filtros") or {}
    if not isinstance(filtros, dict):
        problemas.append("filtros: no es un objeto")
        filtros = {}
    for k in set(filtros) - set(FILTROS):
        problemas.append(f"filtro desconocido: {k}")
    f = form["filtros"]
    f["bpm"] = _bpm(filtros.get("bpm"), c, problemas)
    f["generos"] = _generos(filtros.get("generos"), c, generos_validos, problemas)
    f["tonalidad"] = _tonalidad(filtros.get("tonalidad"), c, problemas)
    voz = filtros.get("voz", "da_igual")
    if voz in c["voz"]:
        f["voz"] = voz
    else:
        problemas.append(f"voz fuera de lista: {voz!r}")
    otros = filtros.get("otros") or {}
    if not isinstance(otros, dict):
        problemas.append("otros: no es un objeto")
        otros = {}
    for k in set(otros) - set(OTROS):
        problemas.append(f"otros: clave desconocida {k}")
    anio = otros.get("anio")
    if anio is not None:
        lo_lim, hi_lim = c["otros"]["anio"]
        ok = isinstance(anio, list) and len(anio) == 2 and any(v is not None for v in anio) and all(
            v is None or (isinstance(v, int) and not isinstance(v, bool) and lo_lim <= v <= hi_lim) for v in anio) \
            and (anio[0] is None or anio[1] is None or anio[0] <= anio[1])
        if ok:
            f["otros"]["anio"] = list(anio)
        else:
            problemas.append(f"otros.anio inválido: {anio!r}")
    for k in ("sellos", "artistas", "artistas_excluir"):
        f["otros"][k] = _nombres(otros.get(k), c, f"otros.{k}", problemas)
    if "excluir_tocados" in otros:
        if isinstance(otros["excluir_tocados"], bool):
            f["otros"]["excluir_tocados"] = otros["excluir_tocados"]
        else:
            problemas.append("otros.excluir_tocados: no es verdadero o falso")
    form["referencia"] = _referencia(raw.get("referencia"), c, problemas)
    form["pesos"] = _pesos(raw.get("pesos"), c, problemas)
    momento = raw.get("momento")
    if momento is None or momento in c["momentos"]:
        form["momento"] = momento
    else:
        problemas.append(f"momento fuera de lista: {momento!r}")
    form["texto"] = _texto(raw.get("texto"), c, problemas)
    orden = raw.get("orden", "puntaje")
    if orden in c["orden"]:
        form["orden"] = orden
    else:
        problemas.append(f"orden fuera de lista: {orden!r}")
    largo = raw.get("largo", c["largo"]["defecto"])
    if isinstance(largo, int) and not isinstance(largo, bool) and 1 <= largo <= c["largo"]["max"]:
        form["largo"] = largo
    else:
        problemas.append(f"largo fuera de rango: {largo!r}")
    if estricto and problemas:
        raise FormularioInvalido(problemas)
    return form, problemas


def es_vacio(form: dict) -> bool:
    """Nada pedido: ni filtros, ni referencia, ni texto, ni momento."""
    v = vacio()
    return all(form.get(k) == v[k] for k in ("filtros", "referencia", "momento", "texto"))


# ---------------------------------------------------------------- presets

def _fundir(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _fundir(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def componer(ids: Sequence[str], generos: Sequence[str] = (), tabla: Optional[dict] = None) -> Tuple[dict, List[str], List[str]]:
    """Los presets elegidos en el teléfono, juntos: (formulario parcial, etiquetas, problemas). En los grupos «bpm» y
    «momento» vale uno solo (gana el último). Un género elegido como preset filtra duro."""
    tabla = tabla or presets()
    por_id = {p["id"]: p for p in tabla["presets"]}
    form: dict = {}
    etiquetas: List[str] = []
    problemas: List[str] = []
    grupos: Dict[str, str] = {}
    for i in ids:
        p = por_id.get(i)
        if p is None:
            problemas.append(f"preset desconocido: {i!r}")
            continue
        if p["grupo"] in ("bpm", "momento") and p["grupo"] in grupos:
            problemas.append(f"dos presets del grupo {p['grupo']}: queda {i}")
            etiquetas = [e for e in etiquetas if e != grupos[p["grupo"]]]
        grupos[p["grupo"]] = p["etiqueta"]
        form = _fundir(form, p["formulario"])
        etiquetas.append(p["etiqueta"])
    gen = [g for g in dict.fromkeys(generos) if isinstance(g, str) and g.strip()]
    if gen:
        form = _fundir(form, {"filtros": {"generos": {"incluir": gen, "modo": "duro"}}})
        etiquetas.extend(gen)
    return form, etiquetas, problemas


# ---------------------------------------------------------------- en palabras

RELATIVO = {"mas_lento": "más lento", "igual": "mismo tempo", "mas_rapido": "más rápido"}


def _bpm_texto(b: dict) -> str:
    if "relativo" in b:
        return f"BPM {RELATIVO[b['relativo']]} que la referencia"
    lo, hi = b["rango"]
    if lo is not None and hi is not None:
        return f"BPM {lo:g} a {hi:g}"
    return f"BPM desde {lo:g}" if lo is not None else f"BPM hasta {hi:g}"


def describir(form: dict) -> List[str]:
    """El formulario en frases cortas, en el orden en que se aplica. El texto en inglés no entra: el teléfono lo muestra
    aparte («Busqué»)."""
    out: List[str] = []
    ref = form["referencia"]
    if ref["tipo"] == "sonando":
        out.append("Parecidos a lo que suena")
    elif ref["tipo"] == "tema":
        out.append(f"Parecidos a «{ref.get('busca') or 'el tema elegido'}»")
    elif ref["tipo"] == "dj":
        out.append(f"Como un set de {ref.get('busca')}")
    f = form["filtros"]
    g = f["generos"]
    if g and g["incluir"]:
        out.append(", ".join(g["incluir"]) + (" (y géneros vecinos)" if g["modo"] == "blando" else ""))
    if g and g["excluir"]:
        out.append("Sin " + ", ".join(g["excluir"]))
    if f["bpm"]:
        out.append(_bpm_texto(f["bpm"]))
    t = f["tonalidad"]
    if t == "compatible":
        out.append("Tonalidad compatible")
    elif t == "igual":
        out.append("Misma tonalidad")
    elif isinstance(t, dict):
        out.append(f"Tonalidad compatible con {t['clave']}")
    if f["voz"] == "sin":
        out.append("Sin voces")
    elif f["voz"] == "con":
        out.append("Con voz")
    o = f["otros"]
    if o["anio"]:
        lo, hi = o["anio"]
        out.append(f"De {lo} a {hi}" if lo and hi else (f"Desde {lo}" if lo else f"Hasta {hi}"))
    if o["sellos"]:
        out.append("Sello " + ", ".join(o["sellos"]))
    if o["artistas"]:
        out.append("De " + ", ".join(o["artistas"]))
    if o["artistas_excluir"]:
        out.append("Nada de " + ", ".join(o["artistas_excluir"]))
    if not o["excluir_tocados"]:
        out.append("Puede repetir lo ya tocado")
    if form["momento"]:
        out.append(f"Momento: {form['momento']} (ordena, no filtra)")
    if form["orden"] == "bpm_sube":
        out.append("Ordenados por BPM, de menor a mayor")
    elif form["orden"] == "bpm_baja":
        out.append("Ordenados por BPM, de mayor a menor")
    if form["largo"] != contrato()["largo"]["defecto"]:
        out.append(f"{form['largo']} temas")
    return out


# ---------------------------------------------------------------- catálogo

def anio_de(it) -> Optional[int]:
    """Año de lanzamiento de la colección de Traktor (RELEASE_DATE «2019/5/3»)."""
    info = it.entry.find("INFO") if getattr(it, "entry", None) is not None else None
    m = re.match(r"\s*(\d{4})", (info.get("RELEASE_DATE") or "") if info is not None else "")
    return int(m.group(1)) if m else None


def sello_de(it) -> Optional[str]:
    info = it.entry.find("INFO") if getattr(it, "entry", None) is not None else None
    label = (info.get("LABEL") or "").strip() if info is not None else ""
    return label or None


def catalogo(items: Sequence, vocal: Dict[str, bool], creado: Optional[str] = None) -> dict:
    """Lo que hay en la biblioteca, en agregados. Es lo único de la biblioteca que va al Worker: los géneros (etiquetas
    de Beatport) con su conteo y números; ni títulos, ni artistas, ni sellos, ni rutas."""
    import statistics
    n = len(items)
    generos = Counter(it.genre for it in items if it.genre)
    bpms = sorted(float(it.bpm) for it in items if it.bpm)
    anios = [a for a in (anio_de(it) for it in items) if a]
    voces = [vocal[it.uid] for it in items if it.uid in vocal]

    def q(p: float) -> Optional[float]:
        return None if not bpms else round(bpms[min(len(bpms) - 1, int(p * (len(bpms) - 1) + 0.5))], 1)

    return {
        "version": 1,
        "creado": creado,
        "temas": n,
        "generos": [{"nombre": g, "temas": c} for g, c in sorted(generos.items(), key=lambda x: (-x[1], x[0]))],
        "sin_genero": n - sum(generos.values()),
        "bpm": None if not bpms else {"min": round(bpms[0], 1), "p10": q(0.1), "mediana": round(statistics.median(bpms), 1),
                                      "p90": q(0.9), "max": round(bpms[-1], 1)},
        "voz": {"con": sum(voces), "sin": len(voces) - sum(voces)},
        "anio": None if not anios else {"min": min(anios), "max": max(anios)},
        "campos": {"genero": sum(generos.values()), "bpm": len(bpms),
                   "tonalidad": sum(1 for it in items if it.key and it.key != "?"),
                   "voz": len(voces), "anio": len(anios),
                   "sello": sum(1 for it in items if sello_de(it)),
                   "artista": sum(1 for it in items if (it.artist or "").strip())},
        "pendientes": sorted(contrato()["pendiente"]),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Contrato del formulario del teléfono")
    ap.add_argument("accion", choices=("catalogo",))
    ap.parse_args(argv)
    from tools.traktor_live import conceptos_agente as ca
    bib = ca.cargar_biblioteca(say=ca.console_say)
    print(json.dumps(catalogo(bib.items, bib.vocal), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
