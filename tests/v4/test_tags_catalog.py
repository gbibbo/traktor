"""
PURPOSE: Tests de src/v4/common/tags.py y de las extensiones de catálogo para bibliotecas con
         subcarpetas: parsing de BPM y comentarios de Mixed In Key, token idempotente, hash de
         audio estable al editar tags (MP3 y FLAC reales escritos con soundfile), lectura de tags,
         deduplicado por track_uid prefiriendo la copia organizada y duración con respaldo mutagen.
CHANGELOG:
  - 2026-09-27: Creación inicial.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.catalog import _copy_penalty, build_catalog, dedupe_by_uid  # noqa: E402
from src.v4.common.tags import (  # noqa: E402
    audio_payload_hash, parse_bpm, parse_mik_comment, payload_range, read_tags, with_token, write_comment,
)


def test_parse_bpm():
    assert parse_bpm("128") == 128.0
    assert parse_bpm("127,98") == 127.98
    assert parse_bpm("1280") == 128.0
    assert parse_bpm("abc") is None
    assert parse_bpm(None) is None
    assert parse_bpm("5") is None


def test_parse_mik_comment():
    assert parse_mik_comment("8A - Energy 6") == ("8A", 6)
    assert parse_mik_comment("10B/7A - Energy 6") == ("10B", 6)
    assert parse_mik_comment("3B - Energy 6 ") == ("3B", 6)
    assert parse_mik_comment("Toolroom") == (None, None)
    assert parse_mik_comment(None) == (None, None)


def test_with_token_idempotent():
    assert with_token("8A - Energy 6", "Vocal") == "8A - Energy 6 - Vocal"
    assert with_token("8A - Energy 6 - Vocal", "Vocal") == "8A - Energy 6 - Vocal"
    assert with_token("", "Vocal") == "Vocal"
    assert with_token(None, "Vocal") == "Vocal"
    # "Vocals" en el título de otra cosa no cuenta como la marca
    assert with_token("Vocalstation", "Vocal") == "Vocalstation - Vocal"


def test_payload_range_skips_id3v2_and_v1():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "x.mp3"
        tag_body = b"\x00" * 20
        id3v2 = b"ID3\x03\x00\x00" + bytes([0, 0, 0, len(tag_body)]) + tag_body
        audio = b"\xff\xfb" + b"A" * 500
        id3v1 = b"TAG" + b"\x00" * 125
        p.write_bytes(id3v2 + audio + id3v1)
        assert payload_range(p) == (len(id3v2), len(id3v2) + len(audio))


def _write_audio(path: Path, fmt: str) -> bool:
    import soundfile as sf
    rng = np.random.default_rng(0)
    data = (rng.standard_normal((44100 * 2, 2)) * 0.1).astype(np.float32)
    try:
        sf.write(str(path), data, 44100, format=fmt)
        return True
    except Exception:  # noqa: BLE001  (libsndfile sin soporte MP3)
        return False


@pytest.mark.parametrize("suffix,fmt", [(".mp3", "MP3"), (".flac", "FLAC")])
def test_hash_stable_after_comment_write(suffix, fmt):
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / f"Artist - Title{suffix}"
        if not _write_audio(p, fmt):
            pytest.skip(f"soundfile no puede escribir {fmt}")
        before = audio_payload_hash(p)
        write_comment(p, "8A - Energy 6")
        assert read_tags(p)["tag_comment"] == "8A - Energy 6"
        assert read_tags(p)["mik_energy"] == 6
        write_comment(p, with_token(read_tags(p)["tag_comment"], "Vocal"))
        assert read_tags(p)["tag_comment"] == "8A - Energy 6 - Vocal"
        assert audio_payload_hash(p) == before


def test_copy_penalty():
    assert _copy_penalty("2020 new - copia/a.mp3") == 1
    assert _copy_penalty("2020 old/old 4/a.mp3") == 1
    assert _copy_penalty("#1 BIBO/PRO/Nuevitas 5/a.mp3") == 0
    assert _copy_penalty("2020 new/Tranqui/a.mp3") == 0
    assert _copy_penalty("Oldschool/a.mp3") == 0


def test_dedupe_prefers_organized_copy():
    cat = pd.DataFrame({
        "track_uid": ["u1", "u1", "u2", "u1"],
        "filename": ["a.mp3", "a.mp3", "b.mp3", "a.mp3"],
        "rel_path": ["2020 new - copia/a.mp3", "2020 new/Tranqui/a.mp3", "2019/b.mp3", "2020 old/old 4/a.mp3"],
    })
    kept, dropped = dedupe_by_uid(cat)
    assert sorted(kept["rel_path"]) == ["2019/b.mp3", "2020 new/Tranqui/a.mp3"]
    assert len(dropped) == 2
    assert set(dropped["kept_rel_path"]) == {"2020 new/Tranqui/a.mp3"}


def test_build_catalog_recursive_with_tags_and_dedup(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "lib"
        (root / "Crate").mkdir(parents=True)
        (root / "Crate - copia").mkdir(parents=True)
        src = root / "Crate" / "Artist - Title.flac"
        if not _write_audio(src, "FLAC"):
            pytest.skip("soundfile no puede escribir FLAC")
        write_comment(src, "5A - Energy 7")
        (root / "Crate - copia" / src.name).write_bytes(src.read_bytes())
        write_comment(root / "Crate - copia" / src.name, "otro comentario")  # mismo audio, otros tags
        monkeypatch.setenv("TRAKTOR_ARTIFACTS_ROOT", str(Path(tmp) / "artifacts"))
        cat = build_catalog(root, "unit", {}, recursive=True, with_tags=True, hash_mode="audio")
        assert len(cat) == 1
        row = cat.iloc[0]
        assert row["rel_path"] == "Crate/Artist - Title.flac"
        assert row["folder"] == "Crate"
        assert row["mik_energy"] == 7 and row["tag_camelot"] == "5A"
        assert (Path(tmp) / "artifacts" / "unit" / "duplicates.csv").exists()


def test_duration_falls_back_to_mutagen(monkeypatch):
    import soundfile as sf
    from src.v4.common import catalog as catalog_mod
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "x.flac"
        if not _write_audio(p, "FLAC"):
            pytest.skip("soundfile no puede escribir FLAC")

        def boom(*a, **k):
            raise RuntimeError("bad map offset")
        monkeypatch.setattr(sf, "info", boom)
        assert abs(catalog_mod._get_duration(p) - 2.0) < 0.05


def test_without_token():
    from src.v4.common.tags import without_token
    assert without_token("8A - Energy 6 - Vocal", "Vocal") == "8A - Energy 6"
    assert without_token("Vocal - 5A - Energy 7", "Vocal") == "5A - Energy 7"
    assert without_token("Vocal", "Vocal") == ""
    assert without_token("8A - Energy 6", "Vocal") == "8A - Energy 6"
    assert without_token(None, "Vocal") == ""
