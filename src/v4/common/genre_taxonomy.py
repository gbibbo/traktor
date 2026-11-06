"""
PURPOSE: Similitud entre géneros de Beatport a partir de una taxonomía propia (DECISIONS 2026-11-06,
         punto 10). La taxonomía la construye una red de agentes y se guarda como JSON:
           familias   [{id, nombre, padre ("" en la raíz), descripcion}]: árbol por encima del género;
           generos    [{beatport, familia, nota}]: cada género de Beatport en una familia;
           relaciones [{a, b, similitud, motivo, fuentes}]: vínculos explícitos entre géneros o
                      subgéneros ("Género" o "Género > Subgénero"), que pueden subir la similitud;
           regla      {mismo_genero_otro_subgenero, misma_familia, familia_hermana, sin_relacion}.
         similarity(a, b), con a y b como (género, subgénero o None):
           - el mismo género: 1 si el subgénero es el mismo o alguno falta; si no, mismo_genero_otro_subgenero;
           - si no, por el árbol: la misma familia, familias con el mismo padre (hermanas) o nada;
           - y el máximo con las relaciones explícitas que apliquen (de género o de subgénero).
         compare(a, b) es lo que usa el recomendador: antes de comparar, "como_genero" lleva un subgénero a
         un género ("DJ Edits > Hip-Hop" -> "Hip-Hop"); un género de "sin_genero" (DJ Tools / Acapellas,
         que no es un estilo) o un género que falta da None, y el componente no cuenta.
         validate() devuelve los problemas de forma (géneros faltantes o repetidos, familias o nombres
         desconocidos, valores fuera de 0-1, regla no decreciente). Solo biblioteca estándar.
         La taxonomía en uso es config/genre_taxonomy/taxonomia.json (DEFAULT_PATH).
CHANGELOG:
  - 2026-11-06: Creación.
  - 2026-11-06: compare(), "sin_genero" y "como_genero" (propuestas del editor de la red de agentes) y
                DEFAULT_PATH.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

Pair = Tuple[str, Optional[str]]
DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "genre_taxonomy" / "taxonomia.json"
RULE_KEYS = ("mismo_genero_otro_subgenero", "misma_familia", "familia_hermana", "sin_relacion")


def split_name(name: str) -> Pair:
    """"Género > Subgénero" -> ("Género", "Subgénero"); "Género" -> ("Género", None)."""
    if ">" in name:
        g, s = name.split(">", 1)
        return g.strip(), s.strip() or None
    return name.strip(), None


class Taxonomy:
    def __init__(self, data: dict):
        self.data = data
        self.parent = {f["id"]: (f.get("padre") or "") for f in data.get("familias", [])}
        self.family = {g["beatport"]: g["familia"] for g in data.get("generos", [])}
        self.rule = {k: float(data.get("regla", {}).get(k, 0.0)) for k in RULE_KEYS}
        self.links: Dict[Tuple[Pair, Pair], float] = {}
        for r in data.get("relaciones", []):
            a, b, v = split_name(r["a"]), split_name(r["b"]), float(r["similitud"])
            for key in ((a, b), (b, a)):
                self.links[key] = max(v, self.links.get(key, 0.0))
        self.none = set(data.get("sin_genero", []))
        self.alias = {split_name(k): v for k, v in data.get("como_genero", {}).items()}

    @classmethod
    def load(cls, path: Path) -> "Taxonomy":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def _tree(self, ga: str, gb: str) -> float:
        fa, fb = self.family.get(ga), self.family.get(gb)
        if fa is None or fb is None:
            return 0.0
        if fa == fb:
            return self.rule["misma_familia"]
        pa, pb = self.parent.get(fa, ""), self.parent.get(fb, "")
        if pa and pa == pb:
            return self.rule["familia_hermana"]
        return self.rule["sin_relacion"]

    def similarity(self, a: Pair, b: Pair) -> float:
        (ga, sa), (gb, sb) = a, b
        if ga == gb:
            if sa is None or sb is None or sa == sb:
                return 1.0
            base = self.rule["mismo_genero_otro_subgenero"]
        else:
            base = self._tree(ga, gb)
        cands = [base]
        for x in {(ga, sa), (ga, None)}:              # una relación de subgénero vale solo para ese subgénero
            for y in {(gb, sb), (gb, None)}:
                v = self.links.get((x, y))
                if v is not None:
                    cands.append(v)
        return max(cands)

    def canonical(self, genre: Optional[str], sub: Optional[str] = None) -> Optional[Pair]:
        """(género, subgénero) tal como se compara; None si no hay género o no cuenta como estilo."""
        if not genre or genre in self.none:
            return None
        g = self.alias.get((genre, sub))
        return (g, None) if g else (genre, sub)

    def compare(self, a: Pair, b: Pair) -> Optional[float]:
        """similarity() de dos temas, o None si a alguno le falta el género (el componente no cuenta)."""
        ca, cb = self.canonical(*a), self.canonical(*b)
        return None if ca is None or cb is None else self.similarity(ca, cb)

    def matrix(self, genres: Iterable[str]) -> Dict[str, Dict[str, float]]:
        gs = list(genres)
        return {a: {b: round(self.similarity((a, None), (b, None)), 4) for b in gs} for a in gs}


def validate(data: dict, beatport: Dict[str, List[str]]) -> List[str]:
    """Problemas de forma de una taxonomía; beatport = {género: [subgéneros]} de Beatport."""
    out: List[str] = []
    fams = [f.get("id") for f in data.get("familias", [])]
    fam_set = set(fams)
    if len(fams) != len(fam_set):
        out.append("hay ids de familia repetidos")
    for f in data.get("familias", []):
        if f.get("padre") and f["padre"] not in fam_set:
            out.append(f"la familia {f.get('id')} tiene un padre desconocido: {f['padre']}")
    seen: Dict[str, int] = {}
    for g in data.get("generos", []):
        seen[g.get("beatport")] = seen.get(g.get("beatport"), 0) + 1
        if g.get("beatport") not in beatport:
            out.append(f"género que no es de Beatport: {g.get('beatport')}")
        if g.get("familia") not in fam_set:
            out.append(f"{g.get('beatport')}: familia desconocida {g.get('familia')}")
    for name in beatport:
        if seen.get(name, 0) == 0:
            out.append(f"falta el género de Beatport: {name}")
        elif seen[name] > 1:
            out.append(f"género repetido: {name}")
    for r in data.get("relaciones", []):
        for side in ("a", "b"):
            g, s = split_name(str(r.get(side, "")))
            if g not in beatport or (s is not None and s not in beatport[g]):
                out.append(f"relación con un nombre que no es de Beatport: {r.get(side)}")
        try:
            v = float(r.get("similitud"))
            if not 0.0 <= v <= 1.0:
                out.append(f"similitud fuera de 0-1: {r.get('a')} / {r.get('b')} = {v}")
        except (TypeError, ValueError):
            out.append(f"similitud no numérica: {r.get('a')} / {r.get('b')}")
    for name in data.get("sin_genero", []):
        if name not in beatport:
            out.append(f"sin_genero con un nombre que no es de Beatport: {name}")
    for src, dst in data.get("como_genero", {}).items():
        g, sub = split_name(src)
        if g not in beatport or sub is None or sub not in beatport[g] or dst not in beatport:
            out.append(f"como_genero con un nombre que no es de Beatport: {src} -> {dst}")
    rule = data.get("regla", {})
    try:
        vals = [float(rule[k]) for k in RULE_KEYS]
        if any(not 0.0 <= v <= 1.0 for v in vals):
            out.append("la regla tiene valores fuera de 0-1")
        if any(vals[i] < vals[i + 1] for i in range(len(vals) - 1)):
            out.append("la regla tiene que decrecer: mismo_genero_otro_subgenero >= misma_familia >= "
                       "familia_hermana >= sin_relacion")
    except (KeyError, TypeError, ValueError):
        out.append(f"a la regla le faltan valores: {', '.join(RULE_KEYS)}")
    return out
