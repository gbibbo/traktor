"""
PURPOSE: Tests del contrato del formulario del teléfono (tools/traktor_live/formulario.py, config/telefono/): valida los
         ejemplos, descarta todo lo que está fuera de sus listas (y en modo estricto lo rechaza), junta presets, describe
         el formulario en palabras y arma el catálogo sin nombres de temas, artistas, sellos ni rutas.
CHANGELOG:
  - 2026-11-13: Creación inicial (línea B, unidad B0).
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.traktor_live import formulario as fm  # noqa: E402
from tools.traktor_live import recomendador as rc  # noqa: E402

GENEROS = ["Tech House", "Techno (Peak Time / Driving)", "Melodic House & Techno", "Deep House", "House"]

EJEMPLOS = [
    # «tech house a 125 sin voces para el pico» (por reglas, sin LLM)
    {"version": 1, "filtros": {"generos": {"incluir": ["Tech House"], "modo": "blando"}, "bpm": {"rango": [123, 127]},
                               "voz": "sin"}, "momento": "pico"},
    # «como lo que suena pero más rápido y con trompetas»
    {"referencia": {"tipo": "sonando"}, "filtros": {"bpm": {"relativo": "mas_rapido"}, "tonalidad": "compatible",
                                                    "otros": {"excluir_tocados": True}},
     "texto": ["driving house groove with bright trumpets and brass stabs"]},
    # «música de cumpleaños con cornetas»: solo texto, como hoy
    {"texto": ["festive house with bright trumpets and a bouncy cheerful groove",
               "joyful festive music with loud brass instruments and hand claps"]},
    # «arriba de 130», «algo en 8A», «clásicos de los 90», «nada de fisher», «solo 10, por bpm»
    {"filtros": {"bpm": {"rango": [130, None]}, "tonalidad": {"clave": "8a"},
                 "otros": {"anio": [None, 1999], "artistas_excluir": ["FISHER"]}}, "largo": 10, "orden": "bpm_sube"},
    # «como ese tema de artbat», «que pese más el género»
    {"referencia": {"tipo": "tema", "busca": "artbat"}, "pesos": {"perfil": "genero", "ajustes": [{"factor": "bpm", "nivel": "menos"}]}},
]


@pytest.mark.parametrize("raw", EJEMPLOS)
def test_examples_validate_without_problems(raw):
    form, problemas = fm.validar(raw, GENEROS, estricto=True)
    assert problemas == [] and form["version"] == 1
    assert json.loads(json.dumps(form)) == form                       # serializable tal cual


def test_defaults_fill_every_slot():
    form, problemas = fm.validar({})
    assert problemas == [] and form == fm.vacio() and fm.es_vacio(form)
    assert form["largo"] == 30 and form["filtros"]["voz"] == "da_igual" and form["referencia"] == {"tipo": "ninguna"}


def test_normalizes_values():
    form, _ = fm.validar(EJEMPLOS[3], GENEROS)
    assert form["filtros"]["tonalidad"] == {"clave": "8A"} and form["filtros"]["bpm"] == {"rango": [130.0, None]}


@pytest.mark.parametrize("raw,casilla", [
    ({"filtros": {"generos": {"incluir": ["Bossa Nova"]}}}, "generos"),
    ({"filtros": {"generos": {"incluir": ["Tech House"], "modo": "medio"}}}, "generos.modo"),
    ({"filtros": {"bpm": {"rango": [130, 120]}}}, "bpm"),
    ({"filtros": {"bpm": {"rango": [10, 20]}}}, "bpm"),
    ({"filtros": {"bpm": {"relativo": "mucho_mas_rapido"}}}, "bpm.relativo"),
    ({"filtros": {"bpm": 126}}, "bpm"),
    ({"filtros": {"voz": "poca"}}, "voz"),
    ({"filtros": {"tonalidad": {"clave": "13A"}}}, "tonalidad"),
    ({"filtros": {"tonalidad": "armonica"}}, "tonalidad"),
    ({"filtros": {"energia": 7}}, "filtro desconocido"),
    ({"referencia": {"tipo": "artista", "busca": "x"}}, "referencia"),
    ({"referencia": {"tipo": "tema"}}, "referencia.tema"),
    ({"pesos": {"perfil": "bailable"}}, "pesos.perfil"),
    ({"pesos": {"ajustes": [{"factor": "bpm", "nivel": 0.7}]}}, "pesos.ajustes"),     # nunca un número
    ({"pesos": {"ajustes": [{"factor": "clap", "nivel": "mas"}]}}, "pesos.ajustes"),
    ({"momento": "madrugada"}, "momento"),
    ({"texto": ["música oscura con trompetas"]}, "texto"),                               # inglés, ASCII
    ({"texto": ["a", "b", "c", "d"]}, "texto"),
    ({"orden": "azar"}, "orden"),
    ({"largo": 50}, "largo"),
    ({"version": 2}, "versión"),
    ({"bpm": [120, 125]}, "casilla desconocida"),
])
def test_out_of_list_values_are_dropped_and_rejected_in_strict_mode(raw, casilla):
    form, problemas = fm.validar(raw, GENEROS)
    assert any(casilla in p for p in problemas), problemas
    assert fm.validar(form, GENEROS, estricto=True)[1] == []         # lo que queda siempre es válido
    with pytest.raises(fm.FormularioInvalido):
        fm.validar(raw, GENEROS, estricto=True)


def test_presets_compose_into_a_valid_form():
    form, etiquetas, problemas = fm.componer(["mas_rapido", "sin_voces"], ["Tech House", "House"])
    assert problemas == [] and etiquetas == ["Más rápido", "Sin voces", "Tech House", "House"]
    v, p = fm.validar(form, GENEROS, estricto=True)
    assert v["referencia"] == {"tipo": "sonando"} and v["filtros"]["bpm"] == {"relativo": "mas_rapido"}
    assert v["filtros"]["voz"] == "sin" and v["filtros"]["generos"] == {"incluir": ["Tech House", "House"], "excluir": [], "modo": "duro"}


def test_presets_one_per_exclusive_group_and_unknown_ids():
    form, etiquetas, problemas = fm.componer(["mas_lento", "mas_rapido", "nada"])
    assert form["filtros"]["bpm"] == {"relativo": "mas_rapido"} and etiquetas == ["Más rápido"] and len(problemas) == 2


def test_every_preset_is_a_valid_form_and_momentos_wait():
    for p in fm.presets()["presets"]:
        fm.validar(p["formulario"], estricto=True)
        assert p.get("pendiente", False) == (p["grupo"] == "momento")


def test_describe_in_words():
    form, _ = fm.validar(EJEMPLOS[0], GENEROS)
    assert fm.describir(form) == ["Tech House (y géneros vecinos)", "BPM 123 a 127", "Sin voces", "Momento: pico"]
    form, _ = fm.validar(EJEMPLOS[3], GENEROS)
    assert fm.describir(form) == ["BPM desde 130", "Tonalidad compatible con 8A", "Hasta 1999", "Nada de FISHER",
                                  "Ordenados por BPM, de menor a mayor", "10 temas"]


def _item(uid, genre, bpm, key="8A", label="", date="", artist="Artista Secreto"):
    e = ET.fromstring(f'<ENTRY TITLE="Tema {uid}"><INFO LABEL="{label}" RELEASE_DATE="{date}"></INFO></ENTRY>')
    it = rc.Item(uid=uid, rel=f"Musica\\ruta privada\\{uid}.mp3", original="", artist=artist, title=f"Tema {uid}",
                 bpm=bpm, key=key, vec=[1.0], genre=genre)
    it.entry = e
    return it


def test_catalog_is_aggregates_only():
    items = [_item("a", "Tech House", 124.0, label="Sello Secreto", date="2019/5/3"), _item("b", "Tech House", 126.0),
             _item("c", "House", 122.0, key="?"), _item("d", None, 70.0)]
    cat = fm.catalogo(items, {"a": True, "b": False, "c": True})
    assert cat["temas"] == 4 and cat["sin_genero"] == 1
    assert cat["generos"] == [{"nombre": "Tech House", "temas": 2}, {"nombre": "House", "temas": 1}]
    assert cat["bpm"]["min"] == 70.0 and cat["bpm"]["max"] == 126.0 and cat["voz"] == {"con": 2, "sin": 1}
    assert cat["anio"] == {"min": 2019, "max": 2019}
    assert cat["campos"] == {"genero": 3, "bpm": 4, "tonalidad": 3, "voz": 3, "anio": 1, "sello": 1, "artista": 4}
    texto = json.dumps(cat, ensure_ascii=False)
    assert "Secreto" not in texto and "privada" not in texto and "Tema" not in texto
    assert len(texto) < 4000
