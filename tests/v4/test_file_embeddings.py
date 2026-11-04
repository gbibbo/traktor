"""
PURPOSE: Tests de src/v4/pipeline/file_embeddings.py: embedding y lugar en la organización dentro de los tags
         (MP3 con GEOB respetando la versión ID3, FLAC con comentarios Vorbis), track_uid intacto, sin
         reescribir lo que ya está, WAV sin tocar y un paquete sintético de punta a punta.
CHANGELOG:
  - 2026-11-04: Creación.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.tags import audio_payload_hash  # noqa: E402
from src.v4.pipeline import file_embeddings as fe  # noqa: E402
from src.v4.pipeline.organize import Library, OrgStore, build  # noqa: E402
from tests.v4.test_organize import PARAMS, _library  # noqa: E402

PLACE = {"org": "t", "version": 3, "track_uid": "u" * 64, "l1": 2, "l2": 1, "position": 7, "x": 1.5, "y": -2.25,
         "carpeta": "C · Techno", "playlist": "C2 (128-130)"}


def _mp3(path: Path, version: int = 3) -> Path:
    from mutagen.id3 import ID3, TIT2
    path.write_bytes(bytes(range(256)) * 40)  # el "audio": cualquier byte sirve para el hash
    tags = ID3()
    tags.add(TIT2(encoding=3, text=["Tema"]))
    tags.save(str(path), v2_version=version)
    return path


def _flac(path: Path) -> Path:
    import soundfile as sf
    sf.write(str(path), np.zeros((4410, 2), dtype="float32"), 44100, format="FLAC")
    return path


def test_mp3_roundtrip_keeps_id3_version_and_audio(tmp_path):
    from mutagen.id3 import ID3
    vec = np.arange(512, dtype=np.float32) / 7
    for version in (3, 4):
        f = _mp3(tmp_path / f"v{version}.mp3", version)
        before = audio_payload_hash(f)
        fe.write_track(f, vec, PLACE, "clap_full")
        got_vec, got_place = fe.read_track(f, "clap_full")
        assert np.array_equal(got_vec, vec) and got_place == PLACE
        assert audio_payload_hash(f) == before                 # el track_uid no cambia
        assert ID3(str(f)).version[1] == version               # 2.3 sigue 2.3
        assert str(ID3(str(f))["TIT2"]) == "Tema"              # lo demás queda


def test_flac_roundtrip_keeps_audio(tmp_path):
    import soundfile as sf
    f = _flac(tmp_path / "a.flac")
    before = audio_payload_hash(f)
    vec = np.linspace(-1, 1, 512, dtype=np.float32)
    fe.write_track(f, vec, PLACE, "clap_full")
    got_vec, got_place = fe.read_track(f, "clap_full")
    assert np.array_equal(got_vec, vec) and got_place == PLACE
    assert audio_payload_hash(f) == before
    assert sf.read(str(f))[0].shape == (4410, 2)             # sigue siendo audio que se lee


def test_nothing_stored_reads_as_none(tmp_path):
    f = tmp_path / "sin.mp3"
    f.write_bytes(b"\xff\xfb" + b"\0" * 100)
    assert fe.read_track(f, "clap_full") == (None, None)
    assert fe.read_track(_flac(tmp_path / "b.flac"), "clap_full") == (None, None)


def test_package_end_to_end(tmp_path):
    art = _library(tmp_path)
    build(OrgStore(art, "t"), Library(art, "clap_full"), dict(PARAMS), [""])
    assign, _, _ = OrgStore(art, "t").load()
    uids = list(assign["track_uid"][:3])
    pkg = tmp_path / "TRAKTOR ML"
    (pkg / "Musica" / "A" / "A1").mkdir(parents=True)
    files = [_mp3(pkg / "Musica/A/A1/uno.mp3"), _flac(pkg / "Musica/A/A1/dos.flac")]
    (pkg / "Musica/A/A1/tres.wav").write_bytes(b"RIFF" + b"\0" * 100)
    pd.DataFrame([{"carpeta": "A · X", "playlist": "A1", "posicion": str(i + 1), "track_uid": u,
                   "archivo": "Musica\\A\\A1\\" + name}
                  for i, (u, name) in enumerate(zip(uids, ["uno.mp3", "dos.flac", "tres.wav"]))]
                 ).to_csv(pkg / "_organizacion.csv", index=False, encoding="utf-8-sig")
    plan, rep = fe.package_plan(pkg, art, "t")
    assert rep == "clap_full" and len(plan) == 3
    assert fe.check_package(plan, rep) == {"bien": 0, "faltan": 2, "distintos": 0, "wav_aiff": 1}
    wav_before = (pkg / "Musica/A/A1/tres.wav").read_bytes()
    result = fe.write_package(plan, rep)
    assert result["counts"] == {"escritos": 2, "ya_estaban": 0, "wav_aiff": 1, "sin_datos": 0, "errores": 0}
    assert (pkg / "Musica/A/A1/tres.wav").read_bytes() == wav_before   # WAV sin tocar
    vec, place = fe.read_track(files[0], rep)
    row = assign.set_index("track_uid").loc[uids[0]]
    assert place["track_uid"] == uids[0] and place["l1"] == int(row["l1"]) and place["position"] == int(row["position"])
    assert fe.check_package(plan, rep)["bien"] == 2
    again = fe.write_package(plan, rep)                                # lo que ya está no se reescribe
    assert again["counts"]["escritos"] == 0 and again["counts"]["ya_estaban"] == 2
    only = fe.package_plan(pkg, art, "t", folder="B")[0]               # --folder: solo esa subcarpeta
    assert only == []
