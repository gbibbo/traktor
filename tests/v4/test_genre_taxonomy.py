"""
PURPOSE: Tests de src/v4/common/genre_taxonomy.py: similitud por el árbol (mismo género, misma familia,
         familias hermanas, nada), relaciones explícitas de género y de subgénero, subgénero faltante,
         matriz simétrica y validación de forma. compare() con "sin_genero" y "como_genero", y la taxonomía
         en uso (config/genre_taxonomy/taxonomia.json) validada contra los géneros de Beatport.
CHANGELOG:
  - 2026-11-06: Creación.
  - 2026-11-06: compare(), alias y la taxonomía en uso.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.genre_taxonomy import DEFAULT_PATH, Taxonomy, split_name, validate  # noqa: E402

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


def test_compare_skips_unknown_and_tool_genres_and_maps_aliases():
    data = {**DATA, "sin_genero": ["Hip-Hop"], "como_genero": {"House > Acid": "Tech House"}}
    t = Taxonomy(data)
    assert t.compare(("House", None), ("Tech House", None)) == 0.6
    assert t.compare(("House", "Acid"), ("Tech House", None)) == 1.0               # se compara como Tech House
    assert t.compare(("Hip-Hop", None), ("House", None)) is None                   # no es un estilo: no cuenta
    assert t.compare((None, None), ("House", None)) is None
    assert t.compare(("Desconocido", None), ("House", None)) == 0.0
    assert validate(data, BEATPORT) == []
    bad = {**data, "sin_genero": ["Nada"], "como_genero": {"House > Bounce": "Tech House", "House > Acid": "X"}}
    text = " | ".join(validate(bad, BEATPORT))
    assert "sin_genero con un nombre que no es de Beatport: Nada" in text
    assert "House > Bounce -> Tech House" in text and "House > Acid -> X" in text


def test_the_taxonomy_in_use_is_valid_and_keeps_the_main_relations():
    beatport = json.loads((DEFAULT_PATH.parent / "beatport_generos.json").read_text(encoding="utf-8"))["generos"]
    data = json.loads(DEFAULT_PATH.read_text(encoding="utf-8"))
    assert len(beatport) == 47 and validate(data, beatport) == []
    t = Taxonomy(data)
    th, pt, raw = ("Tech House", None), ("Techno (Peak Time / Driving)", None), ("Techno (Raw / Deep / Hypnotic)", None)
    assert t.compare(th, th) == 1.0
    assert t.compare(th, ("House", None)) > t.compare(th, pt) > t.compare(th, ("Psy-Trance", None))
    assert t.compare(pt, raw) > t.compare(pt, ("Deep House", None))
    assert t.compare(("DJ Tools / Acapellas", "Acapellas"), th) is None
    assert t.compare(("DJ Edits", "Hip-Hop"), ("Hip-Hop", None)) == 1.0
    gs = list(beatport)
    m = t.matrix(gs)
    assert all(0.0 <= m[a][b] <= 1.0 and m[a][b] == m[b][a] for a in gs for b in gs)
