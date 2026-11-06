"""
PURPOSE: Tests de las sugerencias reales (tools/traktor_live/recomendador.py e indice_recomendacion.py) con
         una colección de Traktor y un índice armados a mano: orden por parecido CLAP, BPM ±20 del que
         suena, tonalidad compatible hasta 2 semitonos (la de Traktor manda sobre la de los tags; sin
         tonalidad no pasa), fuera el tema principal y lo tocado, reconocimiento por la ruta original de
         Música/, raíz del paquete, modo sin CLAP, conversión de tonalidades, el índice que arma
         build_index y la lista que escribe Suggestions con el recomendador.
CHANGELOG:
  - 2026-11-06: Creación (etapa de recomendación, unidad 2).
"""
import base64
import json
import math
import sys
import xml.etree.ElementTree as ET
from array import array
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.traktor_live import indice_recomendacion as ir  # noqa: E402
from tools.traktor_live import recomendador as rc  # noqa: E402
from tools.traktor_live import sugerencias_vivo as sv  # noqa: E402
from tools.traktor_live.sugerencias_explorer import entries_by_path  # noqa: E402

# uid: (vector, BPM de tags, tonalidad de tags, BPM de Traktor, MUSICAL_KEY de Traktor, en la colección)
TRACKS = {
    "main": ((1.0, 0.0), 120, "8A", None, None, True),
    "a": ((0.95, 0.31), 125, "8A", None, None, True),      # el más parecido que pasa
    "b": ((0.6, 0.8), 128, "9A", None, None, True),        # vecina: pasa, menos parecido
    "c": ((0.99, 0.1), 150, "8A", None, None, True),       # BPM a 26 del que suena: no
    "d": ((0.98, 0.2), 124, "2B", None, None, True),       # tonalidad incompatible: no
    "e": ((0.97, 0.24), 124, None, None, None, True),      # sin tonalidad: no
    "f": ((0.9, 0.43), 160, "2B", 123.0, 21, True),        # tags malos, Traktor 123 BPM y 8A: pasa
    "g": ((0.99, 0.05), 124, "8A", None, None, True),      # ya tocado: no
    "h": ((0.999, 0.01), 124, "8A", None, None, False),    # no está en la colección de Traktor: no
}
GENRES = {"main": "Tech House", "a": "Techno", "b": "Tech House", "f": "Tech House", "g": "Techno"}


def unit_b64(v):
    n = math.sqrt(sum(x * x for x in v))
    return base64.b64encode(array("f", [x / n for x in v]).tobytes()).decode("ascii")


def make_library(tmp_path):
    """Paquete <tmp>/TRAKTOR ML/Musica/P1 con archivos vacíos, su collection.nml y el índice."""
    pkg = tmp_path / "TRAKTOR ML"
    folder = pkg / "Musica" / "P1"
    folder.mkdir(parents=True)
    root = ET.Element("NML", VERSION="20")
    coll = ET.SubElement(root, "COLLECTION")
    temas = []
    for uid, (vec, bpm, key, tbpm, tkey, in_coll) in TRACKS.items():
        f = folder / f"{uid}.mp3"
        f.write_bytes(b"")
        temas.append({"uid": uid, "archivo": f"Musica\\P1\\{uid}.mp3", "original": f"Viejo\\{uid}.mp3",
                      "artista": "X", "titulo": uid, "bpm": bpm, "key": key, "emb": unit_b64(vec),
                      "genero": GENRES.get(uid)})
        if not in_coll:
            continue
        e = ET.SubElement(coll, "ENTRY", TITLE=uid, ARTIST="X")
        p = f.resolve()
        ET.SubElement(e, "LOCATION", VOLUME=p.drive if p.drive else "", FILE=p.name,
                      DIR="/:" + "/:".join(p.parent.parts[1:]) + "/:")
        if tbpm:
            ET.SubElement(e, "TEMPO", BPM=f"{tbpm:.6f}")
        if tkey is not None:
            ET.SubElement(e, "MUSICAL_KEY", VALUE=str(tkey))
    collection = tmp_path / "collection.nml"
    ET.ElementTree(root).write(collection, encoding="utf-8", xml_declaration=True)
    index = tmp_path / "indice.json"
    index.write_text(json.dumps({"version": 1, "rep": "clap_full", "creado": "2026-11-06T17:00:00",
                                 "temas": temas}), encoding="utf-8")
    return collection, index


def principal(**kw):
    p = {"deck": "B", "path": "C:\\Música\\Viejo\\main.mp3", "bpm": 120.0, "tempo": 1.0333, "key": "1m"}
    p.update(kw)
    return p


def test_traktor_and_open_key_texts_become_camelot():
    assert [rc.deck_key(k) for k in ("12m", "6m", "1d", "8A", "11b", "Am", "", None, "13m")] == \
        ["7A", "1A", "8B", "8A", "11B", "8A", "?", "?", "?"]
    assert [rc.traktor_key(v) for v in (0, 21, 14, 7, 23, 24, "x")] == ["8B", "8A", "7A", "9B", "10A", "?", "?"]


def test_most_similar_by_clap_within_20_bpm_and_compatible_key(tmp_path):
    collection, index = make_library(tmp_path)
    rec, text = rc.load_recommender(entries_by_path(collection), index, n=30)
    assert rec is not None and "8 temas sugeribles" in text and "de Traktor 1" in text
    tracks, info = rec.compute(principal(), played=[str(tmp_path / "TRAKTOR ML" / "Musica" / "P1" / "g.mp3")])
    assert [t.title for t in tracks] == ["a", "f", "b"]
    assert info["modo"] == "clap" and info["principal"]["uid"] == "main" and info["principal"]["clave"] == "8A"
    assert info["principal"]["bpm"] == 124.0 and info["sin_tonalidad"] == 1 and info["excluidos"] == 2
    assert info["semitonos"] == [0, 0, 0] and info["parecido"][0] > info["parecido"][1] > info["parecido"][2]
    assert tracks[0].path.name == "a.mp3"


def test_the_list_is_cut_at_n_and_played_tracks_are_found_by_their_original_path(tmp_path):
    collection, index = make_library(tmp_path)
    rec, _ = rc.load_recommender(entries_by_path(collection), index, n=2)
    tracks, _ = rec.compute(principal(), played=["/Users/x/Música/Viejo/a.mp3"])
    assert [t.title for t in tracks] == ["g", "f"]


def test_unknown_deck_key_falls_back_to_the_index_and_an_unknown_track_uses_bpm_and_key(tmp_path):
    collection, index = make_library(tmp_path)
    rec, _ = rc.load_recommender(entries_by_path(collection), index)
    assert [t.title for t in rec.compute(principal(key=""))[0]] == ["g", "a", "f", "b"]
    tracks, info = rec.compute({"path": "C:\\otro\\desconocido.mp3", "bpm": 124, "tempo": 1.0, "key": "8A"})
    assert info["modo"] == "sin_clap" and info["principal"]["uid"] is None
    assert [t.title for t in tracks] == ["g", "a", "f", "main", "b"]          # del BPM más cercano al más lejano


def test_update_snapshot_and_refresh_keep_the_last_main_track(tmp_path):
    collection, index = make_library(tmp_path)
    rec, _ = rc.load_recommender(entries_by_path(collection), index)
    assert rec.snapshot() == ([], {}) and rec.refresh([]) is None
    rec.update(principal(), [])
    assert [t.title for t in rec.snapshot()[0]] == ["g", "a", "f", "b"]
    rec.refresh([str(tmp_path / "TRAKTOR ML" / "Musica" / "P1" / "g.mp3")])
    assert [t.title for t in rec.snapshot()[0]] == ["a", "f", "b"]


def test_package_root_is_where_most_index_tracks_are():
    items = [rc.Item(uid=str(i), rel=f"Musica\\P\\{i}.mp3", original="", artist="", title="", bpm=None,
                     key="?", vec=None) for i in range(3)]
    paths = ["c:/traktor ml/musica/p/0.mp3", "c:/traktor ml/musica/p/1.mp3", "d:/traktor ml/musica/p/2.mp3",
             "c:/música/otro.mp3"]
    assert rc.package_root(items, paths) == "c:/traktor ml"
    assert rc.package_root(items, ["c:/música/otro.mp3"]) is None


def test_missing_index_or_no_track_in_traktor_says_why(tmp_path):
    collection, index = make_library(tmp_path)
    rec, text = rc.load_recommender(entries_by_path(collection), tmp_path / "no.json")
    assert rec is None and "indice_recomendacion.py" in text
    rec, text = rc.load_recommender({}, index)
    assert rec is None and "colección de Traktor" in text


def test_build_index_normalizes_embeddings_and_keeps_tags():
    rows = [{"track_uid": "u1", "archivo": "Musica\\P\\a.mp3", "carpeta_original": "Viejo", "nombre_original": "a.mp3",
             "artista": "A", "titulo": "T", "carpeta": "C", "playlist": "P"},
            {"track_uid": "u2", "archivo": "Musica\\P\\b.mp3", "carpeta_original": "", "nombre_original": "b.mp3"}]
    index = ir.build_index(rows, ["u2", "u1"], [[0.0, 2.0], [3.0, 4.0]], {"u1": {"bpm": 124.0, "key": "8A"}}, "clap_full",
                           {"u1": {"genero": "Tech House", "fuente": "beatport"}})
    t1, t2 = index["temas"]
    vec = array("f")
    vec.frombytes(base64.b64decode(t1["emb"]))
    assert [round(x, 4) for x in vec] == [0.6, 0.8] and t1["original"] == "Viejo\\a.mp3"
    assert t1["bpm"] == 124.0 and t1["key"] == "8A" and t2["bpm"] is None and t2["original"] == "b.mp3"
    assert index["dim"] == 2 and index["rep"] == "clap_full" and index["version"] == 2
    assert (t1["genero"], t1["genero_fuente"], t2["genero"]) == ("Tech House", "beatport", None)


def test_suggestions_write_the_recommended_list_and_fall_back_to_random(tmp_path):
    collection, index = make_library(tmp_path)
    sugg = sv.Suggestions(collection, tmp_path / "sug", 3)
    first = sugg.next()
    assert sugg.last_mode == "azar" and len(first) == 3
    rec, _ = rc.load_recommender(sugg.by_path, index)
    sugg.recommender = rec
    assert len(sugg.next()) == 3 and sugg.last_mode == "azar"          # todavía no hay tema principal
    rec.update(principal(), [])
    chosen = sugg.next()
    assert [t.title for t in chosen] == ["g", "a", "f", "b"] and sugg.last_mode == "clap"
    nml = ET.parse(sugg.nml).getroot()
    assert [e.get("TITLE") for e in nml.find("COLLECTION")] == ["g", "a", "f", "b"]
    assert sugg.last_info["uids"] == ["g", "a", "f", "b"]


def test_the_session_recalculates_on_main_track_pitch_and_played(tmp_path):
    from tools.traktor_live import tema_principal as tp
    calls = []

    class FakeRecommender:
        def update(self, p, played):
            calls.append(("update", p["deck"], round(p["bpm"] * p["tempo"]), p["key"]))
            return {"modo": "clap"}

        def refresh(self, played):
            calls.append(("refresh", list(played)))
            return {"modo": "clap"}

    log, now = tmp_path / "sesion.jsonl", [0.0]
    live = tp.Live(tp.Tracker(mixer="externa"), log, lambda level, text: None, clock=lambda: now[0])
    on_deck = sv.hook_recommender(live, FakeRecommender(), log)

    def event(kind, target, data):
        live.http(kind, target, data)
        on_deck(kind, target, data)

    event("updateMasterClock", "", {"deck": "A"})
    event("deckLoaded", "A", {"filePath": "a.mp3", "trackLength": 300, "bpm": 120, "tempo": 1.0, "resultingKey": "1m"})
    event("updateDeck", "A", {"tempo": 1.002})                 # 120,2 BPM: mismo redondeo, no recalcula
    event("updateDeck", "A", {"tempo": 1.05})                  # 126 BPM
    event("updateDeck", "A", {"resultingKey": "2m"})           # el pitch cambió la tonalidad
    event("updateDeck", "B", {"tempo": 1.1})                   # otro deck: no
    event("updateDeck", "A", {"isPlaying": True, "elapsedTime": 0})
    for t in range(1, 32):
        now[0] = float(t)
        event("updateDeck", "A", {"elapsedTime": t})
    assert calls == [("update", "A", 120, "1m"), ("update", "A", 126, "1m"), ("update", "A", 126, "2m"),
                     ("refresh", ["a.mp3"])]
    kinds = [json.loads(line)["kind"] for line in log.read_text(encoding="utf-8").splitlines()]
    assert kinds.count("recomendacion") == 4 and kinds.count("principal") == 1 and kinds.count("tocado") == 1


def test_weights_combine_the_components_and_default_to_clap_only():
    assert rc.weights_from(None) == {"clap": 1.0, "bpm": 0.0, "clave": 0.0, "genero": 0.0, "tiv": 0.0}
    assert rc.weights_from({"bpm": "0.5", "x": 3, "clave": -1})["bpm"] == 0.5
    assert rc.weights_from({"clave": -1})["clave"] == 0.0
    assert rc.weighted({"clap": 0.8, "bpm": 0.5, "tiv": None}, {"clap": 1, "bpm": 3, "tiv": 5}) == (0.8 + 1.5) / 4
    assert rc.weighted({"clap": None}, {"clap": 1}) is None


def test_bpm_key_and_genre_weights_change_the_order(tmp_path):
    collection, index = make_library(tmp_path)
    entries = entries_by_path(collection)

    def order(weights):
        rec, text = rc.load_recommender(entries, index, weights=weights)
        tracks, info = rec.compute(principal())
        return [t.title for t in tracks], info, text

    titles, info, text = order(None)
    assert titles == ["g", "a", "f", "b"] and "Pesos: clap 1." in text
    assert set(info["componentes"]) == {"clap", "bpm", "clave", "genero"}         # tiv todavía no existe
    assert [round(x, 2) for x in info["componentes"]["bpm"]] == [1.0, 0.95, 0.95, 0.8]       # suena a 123,996
    assert info["principal"]["genero"] == "Tech House"
    assert order({"clap": 0, "bpm": 1})[0] == ["g", "f", "a", "b"]           # 124, 123, 125, 128 BPM (suena a 123,996)
    assert order({"clap": 0, "clave": 1})[0] == ["a", "f", "g", "b"]         # 8A antes que la vecina 9A
    assert order({"clap": 0, "genero": 1})[0] == ["b", "f", "a", "g"]        # Tech House como el principal
    titles, info, _ = order({"clap": 1, "genero": 1})
    assert titles[:2] == ["f", "b"] and info["puntaje"][0] > info["puntaje"][2]


def test_genre_comes_from_beatport_or_a_confident_classifier():
    assert ir.genre_of("Tech House", "House", "beatport_B", None) == {"genero": "Tech House", "fuente": "beatport"}
    assert ir.genre_of(None, "Deep House", "classifier", 0.61) == {"genero": "Deep House", "fuente": "clasificador"}
    assert ir.genre_of(None, "Deep House", "classifier", 0.59) is None
    assert ir.genre_of(None, "Jazz (Discogs)", "gate_discogs", 0.9) is None
    assert ir.genre_of(float("nan"), None, "classifier", None) is None
