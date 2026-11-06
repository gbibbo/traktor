"""
PURPOSE: Tests de src/v4/common/genre_taxonomy.py: similitud por el árbol (mismo género, misma familia,
         familias hermanas, nada), relaciones explícitas de género y de subgénero, subgénero faltante,
         matriz simétrica y validación de forma.
CHANGELOG:
  - 2026-11-06: Creación.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.genre_taxonomy import Taxonomy, split_name, validate  # noqa: E402

BEATPORT = {"Techno (Peak Time / Driving)": ["Driving", "Peak Time", "Psy-Techno"],
            "Techno (Raw / Deep / Hypnotic)": ["Raw"], "Tech House": ["Latin Tech"], "House": ["Acid"],
            "Psy-Trance": ["Full-On"], "Hip-Hop": []}
DATA = {
    "familias": [{"id": "edm", "nombre": "EDM", "padre": "", "descripcion": ""},
                 {"id": "techno", "nombre": "Techno", "padre": "edm", "descripcion": ""},
                 {"id": "house", "nombre": "House", "padre": "edm", "descripcion": ""},
                 {"id": "trance", "nombre": "Trance", "padre": "edm", "descripcion": ""},
                 {"id": "open", "nombre": "Open format", "padre": "", "descripcion": ""}],
    "generos": [{"beatport": "Techno (Peak Time / Driving)", "familia": "techno", "nota": ""},
                {"beatport": "Techno (Raw / Deep / Hypnotic)", "familia": "techno", "nota": ""},
                {"beatport": "Tech House", "familia": "house", "nota": ""},
                {"beatport": "House", "familia": "house", "nota": ""},
                {"beatport": "Psy-Trance", "familia": "trance", "nota": ""},
                {"beatport": "Hip-Hop", "familia": "open", "nota": ""}],
    "relaciones": [{"a": "Tech House", "b": "Techno (Peak Time / Driving)", "similitud": 0.7, "motivo": "", "fuentes": []},
                   {"a": "Techno (Peak Time / Driving) > Psy-Techno", "b": "Psy-Trance", "similitud": 0.8,
                    "motivo": "", "fuentes": []}],
    "regla": {"mismo_genero_otro_subgenero": 0.9, "misma_familia": 0.6, "familia_hermana": 0.3, "sin_relacion": 0.0},
}
T = ("Techno (Peak Time / Driving)", None)


def test_tree_levels_and_subgenres():
    t = Taxonomy(DATA)
    assert t.similarity(T, T) == 1.0
    assert t.similarity(("Techno (Peak Time / Driving)", "Driving"), ("Techno (Peak Time / Driving)", "Peak Time")) == 0.9
    assert t.similarity(("Techno (Peak Time / Driving)", "Driving"), T) == 1.0          # subgénero faltante
    assert t.similarity(T, ("Techno (Raw / Deep / Hypnotic)", None)) == 0.6              # misma familia
    assert t.similarity(("House", None), ("Psy-Trance", None)) == 0.3                    # familias hermanas
    assert t.similarity(("House", None), ("Hip-Hop", None)) == 0.0                       # nada en común
    assert t.similarity(("Desconocido", None), ("House", None)) == 0.0


def test_explicit_relations_raise_it_and_subgenre_links_apply_only_to_the_subgenre():
    t = Taxonomy(DATA)
    assert t.similarity(("Tech House", "Latin Tech"), T) == 0.7 == t.similarity(T, ("Tech House", None))
    assert t.similarity(("Techno (Peak Time / Driving)", "Psy-Techno"), ("Psy-Trance", "Full-On")) == 0.8
    assert t.similarity(("Techno (Peak Time / Driving)", "Driving"), ("Psy-Trance", None)) == 0.3
    assert t.similarity(T, ("Psy-Trance", None)) == 0.3
    m = t.matrix(["House", "Tech House", "Psy-Trance"])
    assert all(m[a][b] == m[b][a] for a in m for b in m) and m["House"]["Tech House"] == 0.6


def test_split_name():
    assert split_name("Techno (Peak Time / Driving) > Psy-Techno") == ("Techno (Peak Time / Driving)", "Psy-Techno")
    assert split_name(" House ") == ("House", None)


def test_validate_reports_shape_problems():
    assert validate(DATA, BEATPORT) == []
    bad = {**DATA, "generos": DATA["generos"][:-1] + [{"beatport": "House", "familia": "nada", "nota": ""}],
           "relaciones": [{"a": "House > Bounce", "b": "X", "similitud": 1.5, "motivo": "", "fuentes": []}],
           "regla": {"mismo_genero_otro_subgenero": 0.5, "misma_familia": 0.6, "familia_hermana": 0.3, "sin_relacion": 0}}
    problems = validate(bad, BEATPORT)
    text = " | ".join(problems)
    for expected in ("falta el género de Beatport: Hip-Hop", "género repetido: House", "familia desconocida nada",
                     "House > Bounce", "similitud fuera de 0-1", "la regla tiene que decrecer"):
        assert expected in text, expected
