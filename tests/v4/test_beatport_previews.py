"""
PURPOSE: Previews de Beatport de los temas de los sets (tools/dj_sets/beatport_previews.py): elección repartida entre
         sets, lectura del enlace de 1001Tracklists y de la página de un tema de Beatport, tandas pendientes, y una
         tanda armada y subida a un remoto local (sin red: las descargas son funciones de prueba).
CHANGELOG:
  - 2026-11-09: Creación.
  - 2026-11-10: alias en el artista, la búsqueda no se queda con la comparación flexible si una consulta posterior
                trae la versión exacta, rescate de los temas sin resolver y enlaces solo para los sin id con botón.
  - 2026-11-10: el descargador de MP3 reintenta una conexión cortada y después para limpio.
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


def test_primero_la_busqueda_y_despues_el_enlace_de_1001():
    # Gabriel, 2026-11-09: buscar primero en Beatport; el enlace del tracklist si el tema no salió con la búsqueda.
    assert bpv.ids_candidatos({"beatport_id_1001": "1", "beatport_id_busqueda": "2", "nivel_busqueda": "B"}) == ["2", "1"]
    assert bpv.ids_candidatos({"beatport_id_1001": "1", "beatport_id_busqueda": "2", "nivel_busqueda": "C"}) == ["1"]
    assert bpv.ids_candidatos({"beatport_id_1001": "2", "beatport_id_busqueda": "2", "nivel_busqueda": "A"}) == ["2"]
    assert bpv.beatport_id({"beatport_id_1001": "", "beatport_id_busqueda": "2", "nivel_busqueda": "C"}) == ""


def test_armar_tanda_busca_y_usa_el_respaldo(tmp_path):
    filas = [{"clave": "a", "nombre": "A - Uno", "beatport_id_1001": "55"},          # la búsqueda da 44, sin preview: va el 55
             {"clave": "b", "nombre": "B - Dos", "beatport_id_1001": ""},            # la búsqueda da 66
             {"clave": "c", "nombre": "C - Tres", "beatport_id_1001": "66"}]         # mismo tema de Beatport que b
    resultados = {"a": ("44", "B", 1), "b": ("66", "A", 1), "c": ("", "D", 3)}
    detalles = {44: {"sample_url": None}, 55: {"sample_url": "u55"}, 66: {"sample_url": "u66"}}
    _, manifiesto = bpv.armar_tanda(filas, detalles.get, lambda url: b"ID3" + b"\x01" * 25000, tmp_path / "t", "tanda_00000",
                                    buscar=lambda fila: resultados[fila["clave"]])
    temas = manifiesto["temas"]
    assert [(e["estado"], e.get("beatport_id"), e.get("via")) for e in temas] == [("ok", "55", "1001"), ("ok", "66", "busqueda"), ("repetido", "66", "1001")]
    assert temas[0]["nivel_busqueda"] == "B" and filas[0]["beatport_id_busqueda"] == "44"


def test_elegir_todos_incluye_los_sin_boton():
    s = {"id": "s", "view_item": "1", "temas": [tema("a", "A - Uno"), tema("b", "B - Dos", fuentes=("13",), numerico="")]}
    assert [f["clave"] for f in bpv.elegir([s], 10)] == ["a"]
    filas = bpv.elegir([s], 10, todos=True)
    assert [(f["clave"], f["con_boton"]) for f in filas] == [("a", "si"), ("b", "")]


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


def test_sin_alias():
    assert bpv.sin_alias("Ignacio aka Steve Rachmad") == "Ignacio, Steve Rachmad"
    assert bpv.sin_alias("Inkswel Aka The Snaglepuss ft. Elf Tranzporter") == "Inkswel, The Snaglepuss ft. Elf Tranzporter"
    # Nombres que llevan la palabra no se tocan.
    assert bpv.sin_alias("AKA AKA") == "AKA AKA"
    assert bpv.sin_alias("AKA AKA & Thalstroem") == "AKA AKA & Thalstroem"
    assert bpv.sin_alias("Kaka") == "Kaka"


class ClienteFalso:
    def __init__(self, respuestas):
        self.respuestas, self.n_requests, self.pedidas = respuestas, 0, []

    def search(self, texto):
        self.n_requests += 1
        self.pedidas.append(texto)
        return self.respuestas.get(len(self.pedidas)) or self.respuestas.get(texto, [])


def cand(track_id, title, mix, artists, remixers=()):
    return {"track_id": track_id, "title": title, "mix": mix, "artists": list(artists), "remixers": list(remixers),
            "label": "L", "release_date": "2020-01-01", "genre": "House", "isrc": None, "length_s": 400.0,
            "is_dj_edit": False, "is_ugc_remix": False}


def test_buscar_sigue_si_solo_hay_coincidencia_flexible():
    fila = {"artista": "Bonobo", "titulo": "Sleepwalker (Iorie Lost Dream Remix)"}
    # La primera consulta trae solo la escritura con posesivo; la segunda, la exacta: gana la exacta.
    cliente = ClienteFalso({1: [cand(1, "Sleepwalker", "Iorie's Lost Dream Remix", ["Bonobo"])],
                            2: [cand(2, "Sleepwalker", "Iorie Lost Dream Remix", ["Bonobo"])]})
    assert bpv.buscar_en_beatport(cliente, fila) == ("2", "B", 2) and fila["mezcla_suave"] == ""
    # Si ninguna consulta trae la exacta, queda la flexible, marcada, después de todas las consultas.
    cliente = ClienteFalso({1: [cand(1, "Sleepwalker", "Iorie's Lost Dream Remix", ["Bonobo"])]})
    identificador, nivel, pedidos = bpv.buscar_en_beatport(cliente, fila)
    todas = bpv.consulta_de(fila).search_queries()
    assert (identificador, nivel, fila["mezcla_suave"]) == ("1", "B", "si") and pedidos == len(cliente.pedidas) == len(todas) == 2


def test_rescate_toma_lo_no_resuelto_y_borra_la_busqueda(tmp_path):
    base = tmp_path / "lote_01"
    bpv.escribir_tabla(base / "temas.csv", [
        {"clave": k, "nombre": f"A - {k}", "artista": "A", "titulo": k, "con_boton": "si", "nivel_busqueda": "C",
         "beatport_id_busqueda": "9"} for k in "abcd"])
    temas = [{"clave_1001": "a", "estado": "ok"}, {"clave_1001": "b", "estado": "sin_id"},
             {"clave_1001": "c", "estado": "repetido"}, {"clave_1001": "d", "estado": "sin_preview"}]
    (base / "tanda_00000.json").write_text(json.dumps({"temas": temas}), encoding="utf-8")
    filas = bpv.filas_sin_resolver(base)
    assert [f["clave"] for f in filas] == ["b", "d"]
    assert all(f["nivel_busqueda"] == "" and f["beatport_id_busqueda"] == "" for f in filas)


def test_enlaces_solo_sin_id_con_boton():
    filas = [{"clave": "a", "con_boton": "si", "nivel_busqueda": "B"},
             {"clave": "b", "con_boton": "si", "nivel_busqueda": "C"},
             {"clave": "c", "con_boton": "", "nivel_busqueda": "D"},
             {"clave": "d", "con_boton": "si", "nivel_busqueda": "D", "fuente_1001": "1"}]
    assert [f["clave"] for f in bpv.pendientes_de_enlace(filas, True)] == ["b"]
    assert [f["clave"] for f in bpv.pendientes_de_enlace(filas, False)] == ["a", "b", "c"]


class SesionFalsa:
    def __init__(self, respuestas):
        self.respuestas, self.pedidos = list(respuestas), 0

    def get(self, url, timeout=None):
        self.pedidos += 1
        respuesta = self.respuestas.pop(0)
        if isinstance(respuesta, Exception):
            raise respuesta
        return respuesta


class RespuestaFalsa:
    def __init__(self, status_code, content=b""):
        self.status_code, self.content = status_code, content


def test_descargador_reintenta_la_conexion_cortada_y_despues_para_limpio(monkeypatch):
    requests = pytest.importorskip("requests")
    monkeypatch.setattr(bpv.time, "sleep", lambda s: None)
    # 2026-11-10: «Remote end closed connection without response» tumbó lote_04 con una excepción.
    corte = requests.exceptions.ConnectionError("Remote end closed connection without response")
    sesion = SesionFalsa([corte, corte, RespuestaFalsa(200, b"ID3mp3")])
    descargador = bpv.Descargador(pausa_s=0, sesion=sesion)
    assert descargador("u") == b"ID3mp3" and sesion.pedidos == 3 and descargador.estados["ConnectionError"] == 2
    # 2026-11-11: el manifiesto ordena las claves (put_json): todas de texto, aunque haya cortes y códigos HTTP.
    assert sorted(descargador.estados) == ["200", "ConnectionError"] and json.dumps(descargador.estados, sort_keys=True)
    sesion = SesionFalsa([corte] * (len(bpv.ESPERAS_CORTE_S) + 1))
    with pytest.raises(bp.BeatportBlocked):
        bpv.Descargador(pausa_s=0, sesion=sesion)("u")
    assert sesion.pedidos == len(bpv.ESPERAS_CORTE_S) + 1
    # Dos rechazos seguidos paran; uno solo se reintenta; otro código da vacío.
    with pytest.raises(bp.BeatportBlocked):
        bpv.Descargador(pausa_s=0, sesion=SesionFalsa([RespuestaFalsa(403), RespuestaFalsa(429)]))("u")
    assert bpv.Descargador(pausa_s=0, sesion=SesionFalsa([RespuestaFalsa(429), RespuestaFalsa(200, b"x")]))("u") == b"x"
    assert bpv.Descargador(pausa_s=0, sesion=SesionFalsa([RespuestaFalsa(404)]))("u") == b""


def test_buscar_prueba_el_alias_solo_si_el_nombre_como_viene_no_alcanza():
    fila = {"artista": "Ignacio aka Steve Rachmad", "titulo": "Organa"}
    ignacio = cand(7, "Organa", "Original Mix", ["Ignacio"])
    # El nombre como viene no encuentra; con los nombres separados, sí.
    cliente = ClienteFalso({"Ignacio Organa": [ignacio]})
    assert bpv.buscar_en_beatport(cliente, fila)[:2] == ("7", "B")
    assert cliente.pedidas[0].startswith("Ignacio aka Steve Rachmad") and "Ignacio, Steve Rachmad Organa" in cliente.pedidas
    # Si el nombre como viene ya encuentra la versión, no se pide nada más.
    rogue = cand(8, "Electric Safari", "Original Mix", ["Rogue D"])
    cliente = ClienteFalso({1: [rogue]})
    assert bpv.buscar_en_beatport(cliente, {"artista": "Memoryman aka Uovo & Rogue D", "titulo": "Electric Safari"})[:2] == ("8", "B")
    assert len(cliente.pedidas) == 1

