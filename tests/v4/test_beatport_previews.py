"""
PURPOSE: Previews de Beatport de los temas de los sets (tools/dj_sets/beatport_previews.py): elección repartida entre
         sets, lectura del enlace de 1001Tracklists y de la página de un tema de Beatport, tandas pendientes, y una
         tanda armada y subida a un remoto local (sin red: las descargas son funciones de prueba).
CHANGELOG:
  - 2026-11-09: Creación.
"""
import json
import tarfile

import pytest

from src.v4.common import beatport as bp
from tools.dj_sets import beatport_previews as bpv


def tema(texto, nombre, ided=True, fuentes=("13", "affiliate"), con=False, numerico="1"):
    return {"id_texto": texto, "nombre": nombre, "visible": nombre, "identificado": ided, "fuentes": list(fuentes),
            "junto_al_anterior": con, "id_numerico": numerico}


def test_elegir_reparte_entre_sets_y_no_repite():
    s1 = {"id": "s1", "view_item": "10", "temas": [tema("a", "A - Uno"), tema("b", "B - Dos"), tema("c", "C - Tres")]}
    s2 = {"id": "s2", "view_item": "20", "temas": [tema("a", "A - Uno"), tema("d", "D - Cuatro"),
                                                   tema("x", "X - Sin compra", fuentes=("13",)), tema("y", "ID - ID", ided=False)]}
    filas = bpv.elegir([s1, s2], 4)
    assert [f["clave"] for f in filas] == ["a", "d", "b", "c"]
    assert filas[1]["view_item"] == "20" and filas[1]["artista"] == "D" and filas[1]["titulo"] == "Cuatro"


def test_view_item_y_url():
    html = "new MediaViewer(this, null, {idObject: 5, idItem: 1181899, idSource: '13', viewSource: 1, viewItem: 419922 }"
    assert bpv.view_item_de(html) == "419922"
    url = bpv.url_enlace({"id_numerico": "1122250", "view_item": "683948"})
    assert "idItem=1122250" in url and "idSource=affiliate" in url and "viewItem=683948" in url


def test_leer_enlace():
    # Respuesta real del 2026-11-08 (recortada): el botón de compra lleva a Beatport (source 1).
    beatport = {"success": True, "data": [{"type": "audio", "playerId": "21777813", "duration": "271", "source": "1"}]}
    assert bpv.leer_enlace(beatport) == ("21777813", "1")
    assert bpv.leer_enlace({"success": True, "data": [{"playerId": "abc", "source": "36"}]}) == ("", "36")
    assert bpv.leer_enlace({"success": False}) == ("", "sin_enlace")


def test_parse_track_details():
    datos = {"track_name": "Move It 2 The Drum", "mix_name": "Extended Mix", "artists": [{"name": "Maddix"}],
             "sample_url": "https://geo-samples.beatport.com/track/x.LOFI.mp3", "sample_start_ms": 108408,
             "sample_end_ms": 228408, "bpm": 147, "key": "G Major", "isrc": "NL8RL2558543",
             "genre": {"name": "Techno (Peak Time / Driving)"}, "label": {"name": "EXTATIC"},
             "release": {"name": "Move It", "release_date": "2025-06-13"}}
    html = ('<script id="__NEXT_DATA__" type="application/json">'
            + json.dumps({"props": {"pageProps": {"dehydratedState": {"queries": [
                {"queryKey": ["track-details-21777813"], "state": {"data": datos}}]}}}}) + "</script>")
    d = bp.parse_track_details(html, 21777813)
    assert d["sample_url"].endswith(".LOFI.mp3") and d["sample_end_ms"] - d["sample_start_ms"] == 120000
    assert (d["artists"], d["key"], d["genre"], d["label"], d["release_date"]) == (["Maddix"], "G Major", "Techno (Peak Time / Driving)", "EXTATIC", "2025-06-13")
    assert bp.parse_track_details(html, 1) is None


def test_beatport_id_prefiere_1001_y_solo_misma_version():
    assert bpv.beatport_id({"beatport_id_1001": "1", "beatport_id_busqueda": "2", "nivel_busqueda": "B"}) == "1"
    assert bpv.beatport_id({"beatport_id_1001": "", "beatport_id_busqueda": "2", "nivel_busqueda": "B"}) == "2"
    assert bpv.beatport_id({"beatport_id_1001": "", "beatport_id_busqueda": "2", "nivel_busqueda": "C"}) == ""


def test_tandas_pendientes():
    assert bpv.tandas_pendientes(1001, [], 500) == [0, 1, 2]
    assert bpv.tandas_pendientes(1001, ["tanda_00000.tar", "tanda_00000.json"], 500) == [1, 2]
    assert bpv.tandas_pendientes(100, ["tanda_00000.tar"], 500) == []


def test_es_mp3():
    assert bpv.es_mp3(b"ID3" + b"\x00" * 30000)
    assert bpv.es_mp3(b"\xff\xfb" + b"\x00" * 30000)
    assert not bpv.es_mp3(b"<html>" + b"\x00" * 30000)
    assert not bpv.es_mp3(b"ID3")


def test_armar_tanda_y_subir_a_un_remoto_local(tmp_path, monkeypatch):
    remote_storage = pytest.importorskip("src.v4.common.remote_storage")
    try:
        storage = remote_storage.import_gpuq_storage()
    except ImportError:
        pytest.skip("gpuq no está en esta máquina")
    filas = [{"clave": "a", "nombre": "A - Uno", "beatport_id_1001": "11"},
             {"clave": "b", "nombre": "B - Dos", "beatport_id_1001": "", "beatport_id_busqueda": "22", "nivel_busqueda": "C"},
             {"clave": "c", "nombre": "C - Tres", "beatport_id_1001": "33"}]
    detalles = {11: {"sample_url": "u11", "bpm": 124}, 33: {"sample_url": None}}
    tar, manifiesto = bpv.armar_tanda(filas, detalles.get, lambda url: b"ID3" + b"\x01" * 25000, tmp_path / "t" / "tanda_00000", "tanda_00000")
    assert [e["estado"] for e in manifiesto["temas"]] == ["ok", "sin_id", "sin_preview"]
    with tarfile.open(tar) as paquete:
        assert sorted(paquete.getnames()) == ["11.mp3", "manifest.json"]
    remoto = storage.LocalRemote(tmp_path / "remoto")
    remoto.put(str(tar), "beatport_previews/piloto/tanda_00000.tar")
    assert bpv.tandas_pendientes(len(filas), remoto.ls("beatport_previews/piloto")) == []
