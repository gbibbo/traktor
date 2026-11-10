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

