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
         validate() devuelve los problemas de forma (géneros faltantes o repetidos, familias o nombres
         desconocidos, valores fuera de 0-1, regla no decreciente). Solo biblioteca estándar.
CHANGELOG:
  - 2026-11-06: Creación.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

Pair = Tuple[str, Optional[str]]
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
