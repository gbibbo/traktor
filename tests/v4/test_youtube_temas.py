"""
PURPOSE: Elección del video de YouTube para los temas de los sets que Beatport no da (tools/dj_sets/youtube_temas.py),
         sin red: canal «- Topic», video de un set, otra versión, otro artista, posesivo en la mezcla, duraciones y
         una tanda armada con funciones de prueba.
CHANGELOG:
  - 2026-11-10: Creación.
"""
import tarfile

from tools.dj_sets import youtube_temas as yt


def video(titulo, canal, duracion=400, id_="v"):
    return {"id": id_, "title": titulo, "channel": canal, "uploader": canal, "duration": duracion}


def test_partes_del_video():
    assert yt.partes_del_video(video("Organa", "Ignacio - Topic")) == ("Ignacio", "Organa")
    assert yt.partes_del_video(video("Ignacio – Organa [MM038]", "the29nov films")) == ("Ignacio", "Organa")
    assert yt.partes_del_video(video("Organa", "Algún Sello")) == ("Algún Sello", "Organa")


def test_elige_el_audio_oficial_y_descarta_el_video_de_un_set():
    # Resultados reales de «Ignacio Organa» (2026-11-10).
    resultados = [video("ignacio - ORGANA", "TheAmazingJokerMan", 460, "a"), video("Organa", "Ignacio - Topic", 458, "b"),
                  video("Ignacio - Organa [played by Joris Voorn]", "Dance TV", 335, "c")]
    elegido, motivos = yt.elegir_video("Ignacio aka Steve Rachmad", "Organa", resultados)
    assert elegido["id"] == "b" and motivos == {"ok": 2, "nombra played by": 1}


def test_descarta_otra_version_otro_artista_y_duraciones_raras():
    assert not yt.evaluar("Ignacio", "Organa", video("Ignacio - Organa (Joris Voorn Remix)", "x"))[0]
    assert yt.evaluar("Bonobo", "Sleepwalker (Iorie Lost Dream Remix)", video("Bonobo - Sleepwalker", "x"))[1] == "versión"
    assert yt.evaluar("Ignacio", "Organa", video("Otro Artista - Organa", "x"))[1] == "artista"
    assert yt.evaluar("Ignacio", "Organa", video("Ignacio - Organa", "x", duracion=3600))[1] == "duración"
    assert yt.evaluar("Ignacio", "Organa", video("Ignacio - Organa", "x", duracion=60))[1] == "duración"
    assert yt.evaluar("Ignacio", "Organa", video("Ignacio - Organic", "x"))[1] == "título"


def test_acepta_posesivo_extended_y_lo_que_el_tema_mismo_nombra():
    assert yt.evaluar("Bonobo", "Sleepwalker (Iorie Lost Dream Remix)", video("Sleepwalker (Iorie's Lost Dream Remix)", "Bonobo - Topic"))[0]
    assert yt.evaluar("Ignacio", "Organa", video("Ignacio - Organa (Extended Mix)", "x"))[0]
    # «Live» es parte del tema: no descarta.
    assert yt.evaluar("Air Liquide", "This Is Not A Mind Trip (Live At Love Parade)",
                      video("Air Liquide - This Is Not A Mind Trip (Live At Love Parade)", "x"))[0]
    # Un remix sin nombre no es la versión original.
    assert not yt.evaluar("Waldeck", "Just Close Your Eyes", video("Waldeck - Just Close Your Eyes (Remix)", "x"))[0]


def test_armar_tanda_con_funciones_de_prueba(tmp_path):
    filas = [{"clave": "a", "nombre": "Ignacio - Organa", "artista": "Ignacio", "titulo": "Organa"},
             {"clave": "b", "nombre": "Nadie - Nada", "artista": "Nadie", "titulo": "Nada"}]
    resultados = {"Ignacio - Organa": [video("Organa", "Ignacio - Topic", 458, "b1")], "Nadie - Nada": [video("Otro - Cosa", "y")]}

    def bajar(video_id, carpeta):
        carpeta.mkdir(parents=True, exist_ok=True)
        archivo = carpeta / f"{video_id}.m4a"
        archivo.write_bytes(b"\x00" * 1000)
        return {"archivo": archivo, "ext": "m4a", "abr": 129.5, "acodec": "mp4a.40.2", "asr": 44100, "duracion_s": 458}

    tar, manifiesto = yt.armar_tanda(filas, resultados.get, bajar, tmp_path / "t", "tanda_00000", pausa_s=0)
    assert [e["estado"] for e in manifiesto["temas"]] == ["ok", "sin_version"]
    assert manifiesto["temas"][0]["topic"] and manifiesto["temas"][0]["video_id"] == "b1"
    with tarfile.open(tar) as paquete:
        assert sorted(paquete.getnames()) == ["b1.m4a", "manifest.json"]


def test_titulo_y_artista_al_reves_y_audio_antes_que_videoclip():
    assert yt.evaluar("Olivier Giacomotto", "Meteoroid", video("Meteoroid - Olivier Giacomotto", "x"))[0]
    resultados = [video("Booka Shade - Love Inc (Official Music Video)", "Armada Music TV", 260, "v"),
                  video("Booka Shade - Love Inc", "Booka Shade", 240, "a")]
    assert yt.elegir_video("Booka Shade", "Love Inc", resultados)[0]["id"] == "a"
    # Un videoclip solo, si es la misma versión, se acepta.
    assert yt.elegir_video("Booka Shade", "Love Inc", resultados[:1])[0]["id"] == "v"


def test_restriccion_de_edad_no_para_la_corrida_y_se_prueba_el_siguiente(tmp_path):
    import pytest

    # 2026-11-10: «Sign in to confirm your age» paró la corrida como si fuera el control anti-robots.
    yt.revisar_bloqueo("ERROR: [youtube] H9-1ZjuywFQ: Sign in to confirm your age. This video may be inappropriate")
    with pytest.raises(yt.YouTubeBloqueado):
        yt.revisar_bloqueo("ERROR: [youtube] x: Sign in to confirm you're not a bot")
    filas = [{"clave": "a", "nombre": "Ignacio - Organa", "artista": "Ignacio", "titulo": "Organa"}]
    resultados = {"Ignacio - Organa": [video("Organa", "Ignacio - Topic", 458, "edad"), video("Ignacio - Organa", "Otro", 460, "bien")]}

    def bajar(video_id, carpeta):
        if video_id == "edad":
            return None
        carpeta.mkdir(parents=True, exist_ok=True)
        archivo = carpeta / f"{video_id}.m4a"
        archivo.write_bytes(b"\x00" * 1000)
        return {"archivo": archivo, "ext": "m4a", "abr": 129.5, "acodec": "mp4a.40.2", "asr": 44100, "duracion_s": 460}

    _, manifiesto = yt.armar_tanda(filas, resultados.get, bajar, tmp_path / "t", "tanda_00000", pausa_s=0)
    tema = manifiesto["temas"][0]
    assert (tema["estado"], tema["video_id"], tema["fallidos"]) == ("ok", "bien", ["edad"])


def test_descartes_de_la_primera_tanda_que_eran_aciertos():
    # yt_01 tanda_00000 (2026-11-10): estos se descartaron y eran la misma versión.
    assert yt.evaluar("747", "Aurora Centralis (nthng Saiyan Saga)", video("747 - Aurora Centralis (nthng’s Saiyan Saga) ", "HATE"))[0]
    assert yt.evaluar("VLTRA (IT)", "2 The Bit", video("VLTRA - 2 The Bit", "Uniqueness - House"))[0]
    assert yt.evaluar("Kolo", "Track One", video("Track One", "K.O.L.O. - Topic"))[0]
    assert yt.evaluar("Josh Wink", "Counter Clock 319 (Chris Liebing Reorganized)",
                      video("Josh Wink - Counter Clock 319 (Reorganized by Chris Liebing)", "Repiet"))[0]
    assert yt.evaluar("Christian Borée", "I Against I (Sterac aka Steve Rachmad Remix)",
                      video("Christian Borée - I Against I (Sterac Remix)", "Shrinkk"))[0]
    # El alias no abre la puerta a otro remixer.
    assert not yt.evaluar("Christian Borée", "I Against I (Sterac aka Steve Rachmad Remix)",
                          video("Christian Borée - I Against I (Other Remix)", "x"))[0]


def test_prefiere_el_corte_largo_al_radio_edit():
    resultados = [video("Timecheck (Radio Edit)", "Iteration X - Topic", 105, "radio"),
                  video("Iteration X - Timecheck", "Algún Canal", 420, "largo")]
    assert yt.elegir_video("Iteration X", "Timecheck", resultados)[0]["id"] == "largo"
    # Si el tema es un edit, el edit no pierde puntos.
    assert yt.elegir_video("Das Koolies", "Som Bom Magnifico (Edit)", [video("Som Bom Magnifico (Edit)", "Das Koolies - Topic", 177, "e")])[0]["id"] == "e"

