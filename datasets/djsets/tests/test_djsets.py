"""
PURPOSE: Tests del dataset DJSets: build_djsets.armar/escribir sobre un D5/D6 en miniatura (dos sets, una tanda de
         Beatport y una de YouTube) y el lector djsets.py sobre el resultado (validate, audio desde el .tar, pares,
         unpack que se retoma).
CHANGELOG:
  - 2026-11-13: Creación.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI))

import build_djsets as b  # noqa: E402
import djsets  # noqa: E402

MP3 = {"100": b"ID3" + b"a" * 3000, "200": b"ID3" + b"b" * 4000}
M4A = b"\x00\x00\x00\x18ftypM4A " + b"c" * 5000


def tema(id_texto, nombre, junto=False, segundo=None, identificado=True):
    return {"numero": "", "junto_al_anterior": junto, "identificado": identificado, "segundo": segundo, "nombre": nombre,
            "visible": nombre, "edicion": "", "artista": nombre.split(" - ")[0], "sello": "Sello", "duracion_s": 400,
            "genero": "Techno", "id_texto": id_texto, "id_numerico": "1" if id_texto else "", "fuentes": ["affiliate"]}


def tanda(remoto: Path, carpeta: str, corrida: str, entradas, archivos):
    d = remoto / carpeta / corrida
    d.mkdir(parents=True, exist_ok=True)
    manifiesto = {"tanda": "tanda_00000", "creado": "2026-11-10T00:00:00", "temas": entradas,
                  "ok": sum(e["estado"] == "ok" for e in entradas)}
    (d / "tanda_00000.json").write_text(json.dumps(manifiesto, sort_keys=True), encoding="utf-8")
    tmp = d / "tmp"
    tmp.mkdir()
    (tmp / "manifest.json").write_text(json.dumps(manifiesto), encoding="utf-8")
    with tarfile.open(d / "tanda_00000.tar", "w") as paquete:
        paquete.add(tmp / "manifest.json", arcname="manifest.json")
        for nombre, datos in archivos.items():
            (tmp / nombre).write_bytes(datos)
            paquete.add(tmp / nombre, arcname=nombre)
    shutil.rmtree(tmp)


def listado(remoto: Path, carpeta: str):
    base = remoto / carpeta
    return {p.relative_to(base).as_posix(): {"Path": p.relative_to(base).as_posix(), "Size": p.stat().st_size,
                                             "Hashes": {"md5": djsets.md5_archivo(p)}}
            for p in base.rglob("*") if p.is_file()}


def armar_mini(tmp_path: Path) -> Path:
    root, remoto, out = tmp_path / "root", tmp_path / "remoto", tmp_path / "djsets_v1"
    sets = {"aaa": [tema("t1", "Art1 - Song1", segundo=0), tema("t2", "Art2 - Song2", segundo=300),
                    tema("t4", "Art4 - Song4", junto=True), tema("", "ID - ID", identificado=False),
                    tema("t3", "Art3 - Song3", segundo=700)],
            "bbb": [tema("t2", "Art2 - Song2", segundo=0), tema("t5", "Art5 - Song5", segundo=0)]}
    carpeta = root / "1001tracklists" / "sets"
    carpeta.mkdir(parents=True)
    with (root / "1001tracklists" / "candidatos.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        escritor = csv.DictWriter(handle, fieldnames=["veto", "id", "dj", "uso"])
        escritor.writeheader()
        escritor.writerow({"veto": "", "id": "aaa", "dj": "DJ Uno & DJ Dos", "uso": "pares"})
        escritor.writerow({"veto": "", "id": "bbb", "dj": "DJ Tres", "uso": "pares"})
        escritor.writerow({"veto": "x", "id": "ccc", "dj": "Vetado", "uso": "pares"})
    for set_id, temas in sets.items():
        (carpeta / f"{set_id}.json").write_text(json.dumps({"id": set_id, "titulo": set_id, "temas": temas}), encoding="utf-8")
    detalles = {"sample_start_ms": 60000, "sample_end_ms": 180000, "track_length_ms": 400000, "bpm": 128, "key": "A Minor"}
    tanda(remoto, "beatport_previews", "lote_01", [
        {"clave_1001": "t1", "estado": "ok", "beatport_id": "100", "archivo": "100.mp3", "bytes": len(MP3["100"]),
         "md5": hashlib.md5(MP3["100"]).hexdigest(), "via": "busqueda", "nivel_busqueda": "B", "detalles": dict(detalles, track_id=100)},
        {"clave_1001": "t2", "estado": "ok", "beatport_id": "200", "archivo": "200.mp3", "bytes": len(MP3["200"]),
         "md5": hashlib.md5(MP3["200"]).hexdigest(), "via": "1001", "detalles": dict(detalles, track_id=200)},
        {"clave_1001": "t3", "estado": "repetido", "beatport_id": "100", "igual_a": "t1"},
        {"clave_1001": "t5", "estado": "sin_id"}], {"100.mp3": MP3["100"], "200.mp3": MP3["200"]})
    tanda(remoto, "youtube_temas", "yt_01", [
        {"clave_1001": "t5", "estado": "ok", "video_id": "-Ab_c", "archivo": "-Ab_c.m4a", "ext": "m4a", "bytes": len(M4A),
         "md5": hashlib.md5(M4A).hexdigest(), "duracion_s": 300, "titulo_youtube": "Song5", "canal": "Art5 - Topic", "topic": True}],
        {"-Ab_c.m4a": M4A})
    tanda(remoto, "youtube_temas", "prueba_x", [], {})
    listados = {c: listado(remoto, c) for c in b.FUENTES.values()}
    datos = b.armar(root, remoto, listados)
    b.escribir(out, datos, "abc")
    for s in datos["shards"]:  # lo que hace publish en el Drive: copiar cada tanda con su nombre nuevo
        destino = out / s["shard"]
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remoto / s["origin"], destino)
    return out


def test_armado_y_lectura(tmp_path):
    out = armar_mini(tmp_path)
    ds = djsets.DJSets(out)
    assert [s["set_id"] for s in ds.sets()] == ["aaa", "bbb"]          # el vetado no entra
    assert {t["track_id"] for t in ds.tracks()} == {"t1", "t2", "t3", "t5"}  # t4 solo aparece encima de otro
    assert {t["track_id"] for t in ds.tracks(with_audio=True)} == {"t1", "t2", "t3", "t5"}
    assert ds.track("t3")["recording_id"] == "bp_100" and ds.track("t3")["audio_member"] == "100.mp3"
    assert ds.track("t5")["audio_source"] == "youtube" and ds.track("t5")["audio_is_full_track"] == "1"
    assert ds.track("t5")["audio_attempts"] == "beatport:sin_id > youtube:ok"
    assert ds.track("t1")["audio_duration_s"] == "120.0" and ds.track("t1")["bp_bpm"] == "128"
    assert ds.track("t2")["n_sets"] == "2"
    assert ds.audio_bytes("t3") == MP3["100"] and ds.audio_bytes("t5") == M4A
    assert ds.pairs() == [("aaa", "t1", "t2"), ("bbb", "t2", "t5")]  # «ID - ID» entre t2 y t3 corta el par
    filas = ds.set_tracks("aaa")
    assert [f["track_id"] for f in filas] == ["t1", "t2", "", "", "t3"] and filas[2]["with_previous"] == "1"
    assert [f["cue_s"] for f in filas] == ["0", "300", "", "", "700"]
    assert [f["cue_s"] for f in ds.set_tracks("bbb")] == ["0", ""]      # un 0 fuera de la primera fila = sin tiempo
    assert ds.sets()[1]["share_with_cue"] == "0.500"
    assert set(ds.split()) == {"aaa", "bbb"}
    info = ds.info()
    assert info["counts"]["tracks_with_audio"] == 4 and info["counts"]["recordings"] == 3
    assert info["counts"]["shards"] == {"beatport": 1, "youtube": 1}
    assert ds.validate(full=True) == []


def test_unpack_se_retoma_y_validate_detecta(tmp_path):
    out = armar_mini(tmp_path)
    ds = djsets.DJSets(out)
    assert ds.unpack() == 3
    ruta = ds.audio_path("t5")
    assert ruta is not None and ruta.name == "yt_-Ab_c.m4a" and ruta.read_bytes() == M4A
    assert djsets.DJSets(out).unpack() == 0
    assert djsets.DJSets(out).audio_bytes("t1") == MP3["100"]          # ahora desde audio_files/
    tracks = out / "metadata" / "tracks.csv"
    tracks.write_text(tracks.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert djsets.DJSets(out).validate() == ["md5 distinto: metadata/tracks.csv"]


def test_cli_validate(tmp_path, capsys):
    out = armar_mini(tmp_path)
    assert djsets.main(["validate", str(out)]) == 0
    assert "OK" in capsys.readouterr().out


def test_split_por_primer_dj():
    assert b.clave_dj("John Digweed & Chris Fortier") == "john digweed"
    assert b.clave_dj("Adam Beyer b2b Cirez D") == "adam beyer"
    assert b.split_de("John Digweed") == b.split_de("John Digweed & Chris Fortier")
