"""
PURPOSE: Tests del banco de regresión del teléfono (tools/traktor_live/banco_telefono.py, config/telefono/): cada pedido
         del banco tiene un formulario esperado válido; un intérprete que entiende todo bien saca 1 en cada casilla; uno
         que inventa un género o se equivoca en una casilla se detecta, y el umbral se aplica como está escrito.
CHANGELOG:
  - 2026-11-13: Creación inicial (línea B, B3).
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.traktor_live import banco_telefono as bt  # noqa: E402
from tools.traktor_live import formulario as fm  # noqa: E402

BANCO = bt.cargar()
BEATPORT = {"Tech House", "Techno (Peak Time / Driving)", "Techno (Raw / Deep / Hypnotic)", "Melodic House & Techno",
            "Deep House", "House", "Indie Dance", "Afro House", "Minimal / Deep Tech", "Electronica", "Progressive House",
            "Organic House", "Psy-Trance", "Rock"}


def perfecto(pedido, hay_set):
    e = bt.esperado(next(x for x in BANCO if x["pedido"] == pedido and x["set"] == hay_set))
    form = dict(e["formulario"])
    form["texto"] = ["some sound words"] if e["texto"] else []
    return {"formulario": form, "salida": e["salida"], "no_se_oye": ["algo"] if e["no_se_oye"] else []}


def test_bank_has_150_to_200_requests_with_valid_expected_forms():
    assert 150 <= len(BANCO) <= 200 and len({x["id"] for x in BANCO}) == len(BANCO)
    assert len({(x["pedido"], x["set"]) for x in BANCO}) == len(BANCO)
    for x in BANCO:
        e = bt.esperado(x)
        fm.validar(e["formulario"], BEATPORT, estricto=True)
        assert e["salida"] in bt.SALIDAS and set(x["espera"]) <= {
            "generos", "bpm", "tonalidad", "voz", "otros", "referencia", "pesos", "momento", "orden", "largo",
            "texto", "salida", "no_se_oye"}
    assert sum(x["origen"] == "gabriel" for x in BANCO) == 3


def test_a_perfect_interpreter_scores_one_everywhere():
    r = bt.evaluar(perfecto, BANCO)
    assert set(r["exactitud"].values()) == {1.0} and r["generos_inventados"] == [] and bt.cumple(r, "completo") == []


def test_invented_genres_and_wrong_slots_are_caught():
    def inventa(pedido, hay_set):
        p = perfecto(pedido, hay_set)
        if pedido == "algo oscuro":
            p["formulario"]["filtros"]["generos"] = {"incluir": ["Techno (Raw / Deep / Hypnotic)"], "modo": "blando"}
        if pedido == "126 bpm":
            p["formulario"]["filtros"]["bpm"] = {"rango": [123, 129]}
        return p
    r = bt.evaluar(inventa, BANCO)
    assert r["generos_inventados"] == [next(x["id"] for x in BANCO if x["pedido"] == "algo oscuro")]
    assert r["fallas"]["bpm"] == ["s9"] and any("inventados" in p for p in bt.cumple(r, "completo"))


def test_rules_are_judged_only_on_what_they_fill():
    def solo_bpm(pedido, hay_set):
        p = perfecto(pedido, hay_set)
        return {"formulario": {"filtros": {"bpm": p["formulario"]["filtros"]["bpm"]}}, "salida": "lista", "no_se_oye": []}
    r = bt.evaluar(solo_bpm, BANCO)
    assert r["precision"]["bpm"] == 1.0 and r["precision"]["generos"] is None and bt.cumple(r, "reglas") == []


def test_named_track_matches_by_words_and_the_baseline_is_recorded():
    x = next(x for x in BANCO if x["id"] == "s27")
    pred = {"formulario": {"referencia": {"tipo": "tema", "busca": "Losing It - FISHER"}}}
    assert bt.comparar(pred, x)["referencia"]
    assert not bt.comparar({"formulario": {"referencia": {"tipo": "tema", "busca": "losing"}}}, x)["referencia"]
    hoy = bt.evaluar(bt.interprete_hoy, BANCO)
    assert hoy["exactitud"]["generos"] < 0.9 and bt.cumple(hoy, "completo")
    umbral = json.loads(bt.UMBRAL_PATH.read_text(encoding="utf-8"))
    assert umbral["reglas"]["precision_minima"] == 0.97 and umbral["completo"]["exactitud_minima"] == 0.90
