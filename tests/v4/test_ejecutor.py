"""
PURPOSE: Tests del ejecutor del formulario del teléfono (tools/traktor_live/ejecutor.py) con una biblioteca armada a mano
         y el recomendador de verdad: cada filtro y su conteo, el género duro y el blando (con los vecinos de la
         taxonomía), la referencia «lo que suena» y un tema nombrado (con candidatos si hay dudas), lo que se ignora con
         aviso y el orden. Dos pruebas con la biblioteca real (se saltan sin el índice): un formulario con solo texto da
         la lista de siempre, y uno con solo «lo que suena», la de Parecidos.
CHANGELOG:
  - 2026-11-13: Creación inicial (línea B, unidad B1).
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.traktor_live import conceptos_agente as ca  # noqa: E402
from tools.traktor_live import ejecutor as ej  # noqa: E402
from tools.traktor_live import formulario as fm  # noqa: E402
from tools.traktor_live import recomendador as rc  # noqa: E402

# uid: (vector, BPM, tonalidad, género, voz, artista, título, sello, fecha)
TEMAS = {
    "main": ((1, 0, 0), 124.0, "8A", "Tech House", False, "Uno", "Principal", "", ""),
    "a": ((0.95, 0.31, 0), 125.0, "8A", "Tech House", False, "FISHER", "Losing It (Extended Mix)", "Catch & Release", "2018/7/1"),
    "b": ((0.6, 0.8, 0), 126.0, "9A", "House", True, "ARTBAT", "Aquarius", "Afterlife Records", "2019/1/1"),
    "c": ((0.9, 0, 0.43), 150.0, "8A", "Tech House", True, "Dos", "Rápido", "", "1998/1/1"),
    "d": ((0.8, 0, 0.6), 123.0, "2B", "Minimal / Deep Tech", False, "Tres", "Choca", "", ""),
    "e": ((0, 1, 0), 120.0, "8A", "Deep House", True, "Matador, ARTBAT", "Visitor", "Afterlife", "2020/1/1"),
    "f": ((0, 0, 1), 128.0, "8A", None, False, "Cuatro", "Sin género", "", ""),
    "g": ((0.99, 0.1, 0), 124.0, "8A", "Techno (Peak Time / Driving)", False, "Cinco", "Ya tocado", "", ""),
    "h": ((-1, 0, 0), 132.0, "7A", "Melodic House & Techno", True, "Seis", "Lejos", "", ""),
}
RAIZ = "C:\\TRAKTOR ML\\Musica\\P1\\"


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return (v / np.linalg.norm(v)).tolist()


def biblioteca(ajustes=None):
    from src.v4.common.genre_taxonomy import DEFAULT_PATH, Taxonomy
    items, entries = [], {}
    for uid, (vec, bpm, key, genre, _v, artist, title, label, date) in TEMAS.items():
        it = rc.Item(uid=uid, rel=f"Musica\\P1\\{uid}.mp3", original=f"Viejo\\{uid}.mp3", artist=artist, title=title,
                     bpm=bpm, key=key, vec=unit(vec), genre=genre)
        e = ET.Element("ENTRY", TITLE=title, ARTIST=artist)
        ET.SubElement(e, "LOCATION", VOLUME="C:", DIR="/:TRAKTOR ML/:Musica/:P1/:", FILE=f"{uid}.mp3")
        ET.SubElement(e, "INFO", LABEL=label, RELEASE_DATE=date)
        entries[f"{RAIZ}{uid}.mp3"] = e
        items.append(it)
    rec = rc.Recommender(items, entries, 30, (ajustes or {}).get("pesos"), Taxonomy.load(DEFAULT_PATH), (ajustes or {}).get("opciones"))
    assert len(rec.candidates) == len(TEMAS)
    return ca.Biblioteca(items=rec.candidates, matrix=np.array([it.vec for it in rec.candidates], dtype=np.float32),
                         vocal={u: t[4] for u, t in TEMAS.items()}, nml_version="20", rec=rec)


def codificar(frases):
    return np.array([unit((1, 0, 0))] * len(frases))


SONANDO = ej.Sesion({"path": RAIZ + "main.mp3", "bpm": 124.0, "tempo": 1.0, "key": "8A", "title": "Principal", "artist": "Uno"},
                    [RAIZ + "g.mp3"])


def correr(raw, sesion=None, bib=None):
    form, problemas = fm.validar(raw, [t[3] for t in TEMAS.values() if t[3]], estricto=True)
    bib = bib or biblioteca()
    res = ej.ejecutar(form, bib, sesion, codificar)
    return [bib.items[i].uid for i in res.indices], res


def quedan(res):
    return [p["quedan"] for p in res.pasos]


# ---------------------------------------------------------------- lo de hoy

def test_text_only_is_todays_list():
    uids, res = correr({"texto": ["festive house"]})
    bib = biblioteca()
    sims = bib.matrix @ np.asarray(unit((1, 0, 0)), dtype=np.float32)
    assert uids == [bib.items[i].uid for i in np.argsort(-sims, kind="stable")]
    assert quedan(res) == [9] and res.ignorado == []


def test_playing_track_only_is_todays_parecidos():
    uids, res = correr({"referencia": {"tipo": "sonando"}}, SONANDO)
    bib = biblioteca()
    _, info = bib.rec.compute(SONANDO.principal, SONANDO.tocados)
    assert uids == info["uids"] and uids[0] == "a"
    assert not {"main", "c", "d", "g"} & set(uids)      # el que suena, c por BPM, d por tonalidad, g ya tocado
    assert res.referencia == {"titulo": "Principal", "artista": "Uno"} and quedan(res) == [9, len(uids)]
    assert "Parecidos a «Principal»" in res.pasos[1]["paso"]


def test_parecidos_uses_the_live_settings():
    sin_clave = biblioteca({"opciones": {"filtro_clave": False}})
    uids, _ = correr({"referencia": {"tipo": "sonando"}}, SONANDO, sin_clave)
    assert "d" in uids


# ---------------------------------------------------------------- filtros

def test_hard_genre_keeps_only_that_genre():
    uids, res = correr({"filtros": {"generos": {"incluir": ["Tech House"], "modo": "duro"}}, "texto": ["x y z"]})
    assert set(uids) == {"main", "a", "c"} and res.pasos[1] == {"paso": "Tech House", "quedan": 3}


def test_soft_genre_puts_the_genre_first_then_its_neighbors():
    uids, res = correr({"filtros": {"generos": {"incluir": ["Tech House"], "modo": "blando"}}, "texto": ["x y z"]})
    assert uids[:3] == ["main", "a", "c"] and set(uids[3:]) == {"b", "d"}       # House 0,85 y Minimal 0,8; Deep House 0,55 no
    assert "vecinos: House, Minimal / Deep Tech" in res.pasos[1]["paso"] and "f" not in uids


def test_excluded_genre_and_artist():
    uids, _ = correr({"filtros": {"generos": {"excluir": ["Tech House"]}, "otros": {"artistas_excluir": ["artbat"]}},
                      "texto": ["x y z"]})
    assert set(uids) == {"d", "f", "g", "h"}


def test_bpm_range_and_key_and_voice():
    uids, res = correr({"filtros": {"bpm": {"rango": [123, 128]}, "tonalidad": {"clave": "8A"}, "voz": "sin"}, "texto": ["x y z"]})
    assert set(uids) == {"main", "a", "f", "g"} and quedan(res) == [9, 6, 5, 4]


def test_relative_bpm_follows_the_reference():
    uids, res = correr({"referencia": {"tipo": "sonando"}, "filtros": {"bpm": {"relativo": "mas_lento"}}}, SONANDO)
    assert uids == ["e"] and res.pasos[-1]["paso"].startswith("BPM 116 a 122")


def test_relative_bpm_without_reference_is_ignored_with_a_reason():
    _, res = correr({"filtros": {"bpm": {"relativo": "mas_rapido"}}, "texto": ["x y z"]})
    assert [i["casilla"] for i in res.ignorado] == ["filtros.bpm"]


def test_the_live_key_filter_stays_unless_a_key_is_given():
    uids, _ = correr({"referencia": {"tipo": "sonando"}, "pesos": {"ajustes": [{"factor": "tonalidad", "nivel": "menos"}]}}, SONANDO)
    assert "d" not in uids                                   # «ignora la tonalidad» no saca el filtro del vivo
    uids, _ = correr({"referencia": {"tipo": "sonando"}, "filtros": {"tonalidad": {"clave": "2B"}}}, SONANDO)
    assert uids[0] == "d" and "a" not in uids                # una tonalidad pedida reemplaza a la de lo que suena


def test_same_key_and_played_tracks_can_come_back():
    uids, _ = correr({"referencia": {"tipo": "sonando"}, "filtros": {"tonalidad": "igual", "otros": {"excluir_tocados": False}}}, SONANDO)
    assert set(uids) == {"a", "e", "f", "g"}


def test_year_label_and_artist():
    assert correr({"filtros": {"otros": {"anio": [2019, None]}}, "texto": ["x y z"]})[0] == ["b", "e"]
    assert set(correr({"filtros": {"otros": {"sellos": ["afterlife"]}}, "texto": ["x y z"]})[0]) == {"b", "e"}
    assert correr({"filtros": {"otros": {"artistas": ["Fisher"]}}, "texto": ["x y z"]})[0] == ["a"]


def test_played_tracks_leave_without_reference():
    uids, res = correr({"texto": ["x y z"]}, SONANDO)
    assert "g" not in uids and res.pasos[1] == {"paso": "Sin lo ya tocado esta noche", "quedan": 8}


# ---------------------------------------------------------------- referencia nombrada, avisos, orden

def test_named_track_is_resolved_on_the_computer():
    uids, res = correr({"referencia": {"tipo": "tema", "busca": "algo como losing it de fisher"}})
    assert res.referencia["titulo"] == "Losing It (Extended Mix)" and "a" not in uids and uids


def test_ambiguous_track_returns_candidates_and_no_list():
    uids, res = correr({"referencia": {"tipo": "tema", "busca": "artbat"}})
    assert uids == [] and {c["titulo"] for c in res.candidatos} == {"Aquarius", "Visitor"}
    uid = next(c["uid"] for c in res.candidatos if c["titulo"] == "Visitor")
    uids, res = correr({"referencia": {"tipo": "tema", "uid": uid}})
    assert res.referencia["titulo"] == "Visitor" and "e" not in uids


def test_unknown_track_and_pending_slots_are_ignored_with_a_reason():
    _, res = correr({"referencia": {"tipo": "tema", "busca": "bossa nova brasilera"}, "momento": "after",
                     "pesos": {"perfil": "genero"}, "orden": "momento", "texto": ["x y z"]})
    assert [i["casilla"] for i in res.ignorado] == ["referencia", "momento", "pesos.perfil", "orden.momento"]
    _, res = correr({"referencia": {"tipo": "dj", "busca": "Adriatique"}, "texto": ["x y z"]})
    assert res.ignorado[0]["casilla"] == "referencia.dj"
    _, res = correr({"referencia": {"tipo": "sonando"}, "texto": ["x y z"]}, None)
    assert res.ignorado[0]["casilla"] == "referencia" and "sonando" in res.ignorado[0]["motivo"]


def test_empty_form_and_nothing_left():
    uids, res = correr({})
    assert uids == [] and res.ignorado[-1]["casilla"] == "formulario"
    uids, res = correr({"filtros": {"bpm": {"rango": [200, 210]}}})
    assert uids == [] and quedan(res) == [9, 0]


def test_filters_alone_order_by_the_most_typical_and_cut():
    uids, _ = correr({"filtros": {"generos": {"incluir": ["Tech House"], "modo": "duro"}}, "largo": 2})
    assert len(uids) == 2 and "c" not in uids              # c es el menos parecido al centro de los tres


def test_order_by_bpm_after_cutting():
    uids, _ = correr({"texto": ["x y z"], "largo": 4, "orden": "bpm_baja"})
    bpm = {u: t[1] for u, t in TEMAS.items()}
    assert [bpm[u] for u in uids] == sorted((bpm[u] for u in uids), reverse=True) and len(uids) == 4


def test_reference_and_text_mix_by_percentiles():
    uids, res = correr({"referencia": {"tipo": "sonando"}, "texto": ["x y z"]}, SONANDO)
    solo_ref, _ = correr({"referencia": {"tipo": "sonando"}}, SONANDO)
    assert set(uids) == set(solo_ref) and all(0 <= s <= 1 for s in res.puntajes)


# ---------------------------------------------------------------- biblioteca real

def _real():
    if not rc.DEFAULT_INDEX.exists():
        pytest.skip("sin el índice de recomendación en esta máquina")
    bib = ca.cargar_biblioteca()
    if bib.rec is None:
        pytest.skip("sin la colección de Traktor")
    return bib


def test_real_library_text_only_matches_the_old_search():
    bib = _real()
    q = np.random.default_rng(20261113).normal(size=bib.matrix.shape[1])
    q = q / np.linalg.norm(q)
    res = ej.ejecutar(fm.validar({"texto": ["any text"]})[0], bib, None, lambda f: np.array([q]))
    sims = np.asarray(bib.matrix @ q, dtype=np.float64)
    assert res.indices == [int(i) for i in np.argsort(-sims, kind="stable")[:30]]


def test_real_library_playing_track_only_matches_parecidos():
    bib = _real()
    it = bib.items[len(bib.items) // 3]
    principal = {"path": str(it.path), "bpm": it.bpm, "tempo": 1.0, "key": it.key, "title": it.title, "artist": it.artist}
    tocados = [str(bib.items[j].path) for j in range(5)]
    res = ej.ejecutar(fm.validar({"referencia": {"tipo": "sonando"}})[0], bib, ej.Sesion(principal, tocados))
    _, info = bib.rec.compute(principal, tocados)
    assert [bib.items[i].uid for i in res.indices] == info["uids"]


# ---------------------------------------------------------------- momento de la noche (ordena, no filtra)

TABLA = {"generos_noche": {"Tech House": 0.2, "House": 0.2, "Deep House": 0.1, "Techno (Peak Time / Driving)": 0.1,
                           "Minimal / Deep Tech": 0.1, "Melodic House & Techno": 0.1},
         "momentos": {"apertura": {"bpm": [120, 126], "bpm_mediana": 123, "bpm_delta": -4.0, "generos": {"Deep House": 0.3, "Tech House": 0.1}},
                      "pico": {"bpm": [123, 128], "bpm_mediana": 125, "bpm_delta": 1.0, "generos": {"Techno (Peak Time / Driving)": 0.3, "House": 0.1}},
                      "after": {"bpm": None, "generos": {}}}}


@pytest.fixture
def tabla(monkeypatch):
    monkeypatch.setattr(fm, "momentos", lambda: TABLA)


def test_moment_alone_orders_by_the_genres_typical_of_that_moment(tabla):
    uids, res = correr({"momento": "pico"})
    assert uids[0] == "g" and len(uids) == 9 and res.ignorado == []      # Techno ×3; no saca a nadie
    assert correr({"momento": "apertura"})[0][0] == "e"                    # Deep House ×3
    uids, _ = correr({"momento": "pico", "filtros": {"generos": {"incluir": ["Tech House"], "modo": "duro"}}})
    assert uids == ["a", "main", "c"]                                      # mismo género: decide el BPM cerca de 125


def test_moment_with_reference_moves_toward_the_measured_bpm(tabla):
    apertura, _ = correr({"referencia": {"tipo": "sonando"}, "momento": "apertura"}, SONANDO)
    solo, _ = correr({"referencia": {"tipo": "sonando"}}, SONANDO)
    assert set(apertura) == set(solo) and apertura.index("e") < solo.index("e")   # e: Deep House, 120 BPM (124 − 4)


def test_moment_without_data_is_ignored_with_a_reason(tabla):
    uids, res = correr({"momento": "after", "texto": ["x y z"]})
    assert [i["casilla"] for i in res.ignorado] == ["momento"] and uids == correr({"texto": ["x y z"]})[0]
