"""
PURPOSE: Tests de tools/traktor_live/sugerencias_itunes.py: XML de iTunes legible, orden de la playlist,
         rutas codificadas, rondas sin temas repetidos y que no pisa ni borra un XML ajeno.
CHANGELOG:
  - 2026-11-04: Creación inicial.
"""
import plistlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.traktor_live.sugerencias_itunes import (  # noqa: E402
    LIBRARY_ID, PLAYLIST_NAME, Track, build_library, file_url, path_probe, pick, remove_library,
    settings_itunes_path, write_library,
)


def _tracks(n):
    return [Track(path=Path(f"C:\\TRAKTOR ML\\Musica\\Tema {i}.mp3"), title=f"Tema {i}", artist="A",
                  seconds=300.5, bpm="126", key="8A") for i in range(n)]


def test_file_url_encodes_spaces_accents_and_symbols():
    url = file_url(Path("C:\\TRAKTOR ML\\Música\\Tom & Jerry #1.mp3"))
    assert url == "file://localhost/C:/TRAKTOR%20ML/M%C3%BAsica/Tom%20%26%20Jerry%20%231.mp3"


def test_library_roundtrip_keeps_order_and_marks_round(tmp_path):
    chosen = _tracks(3)
    out = tmp_path / "iTunes" / "iTunes Music Library.xml"
    write_library(build_library(chosen, round_no=2), out)
    with open(out, "rb") as f:
        lib = plistlib.load(f)
    assert lib["Library Persistent ID"] == LIBRARY_ID
    pl = next(p for p in lib["Playlists"] if p["Name"] == PLAYLIST_NAME)
    ids = [it["Track ID"] for it in pl["Playlist Items"]]
    assert [lib["Tracks"][str(i)]["Name"] for i in ids] == ["Tema 0", "Tema 1", "Tema 2"]
    assert lib["Tracks"][str(ids[0])]["Comments"] == "Ronda 2 · 1/3"
    assert lib["Tracks"][str(ids[0])]["Total Time"] == 300500
    assert [lib["Tracks"][str(i)]["Composer"] for i in ids] == ["C 95 %", "C 92 %", "C 89 %"]
    assert lib["Tracks"][str(ids[1])]["Grouping"] == "G 92 %"
    assert lib["Tracks"][str(ids[2])]["Album Artist"] == "AA 89 %"
    # una segunda ronda reemplaza el archivo propio
    write_library(build_library(_tracks(2), round_no=3), out)
    assert not out.with_name(out.name + ".tmp").exists()


def test_raw_hash_keeps_hash_unencoded():
    assert file_url(Path("C:\\M\\#1 BIBO\\a b.mp3"), raw_hash=True) == "file://localhost/C:/M/#1%20BIBO/a%20b.mp3"


def test_path_probe_one_track_per_character_class():
    names = ["#1 BIBO\\x.mp3", "Melodic House & Techno\\y.mp3", "a+b.mp3", "a;b.mp3", "Shlømo.mp3", "plain.mp3"]
    tracks = [Track(path=Path("C:\\TRAKTOR ML\\Musica\\" + n), title=n, artist="A", seconds=1, bpm="1", key="1")
              for n in names] + [Track(path=Path("C:\\Música\\#2\\z.mp3"), title="z", artist="A", seconds=1,
                                       bpm="1", key="1")]
    probe = path_probe(tracks, "C:\\TRAKTOR ML\\Musica")
    assert [(label, t.title, raw) for label, t, raw in probe] == [
        ("#", names[0], False), ("# sin codificar", names[0], True), ("&", names[1], False),
        ("+", names[2], False), (";", names[3], False), ("no ASCII", names[4], False), ("simple", names[5], False)]
    lib = build_library([t for _, t, _ in probe], 0, labels=[lb for lb, _, _ in probe],
                        raw_hash=[r for _, _, r in probe])
    locs = [v["Location"] for v in lib["Tracks"].values()]
    assert "%231%20BIBO" in locs[0] and "/#1%20BIBO" in locs[1]
    assert lib["Tracks"]["1002"]["Name"] == "[prueba # sin codificar] " + names[0]
    assert lib["Tracks"]["1001"]["Persistent ID"] != lib["Tracks"]["1002"]["Persistent ID"]


def test_rounds_do_not_repeat_tracks():
    tracks = _tracks(35)
    r1, r2, r3 = (pick(tracks, r, 10) for r in (1, 2, 3))
    assert len({t.path for t in r1 + r2 + r3}) == 30
    assert pick(tracks, 1, 10) == r1                       # orden fijo entre corridas
    with pytest.raises(ValueError):
        pick(tracks, 4, 10)


def test_never_overwrites_or_removes_a_foreign_library(tmp_path):
    out = tmp_path / "iTunes Music Library.xml"
    with open(out, "wb") as f:
        plistlib.dump({"Library Persistent ID": "0000000000000000", "Tracks": {}}, f)
    with pytest.raises(FileExistsError):
        write_library(build_library(_tracks(1), 1), out)
    with pytest.raises(FileExistsError):
        remove_library(out)
    assert out.exists()


def test_remove_own_library(tmp_path):
    out = tmp_path / "iTunes Music Library.xml"
    write_library(build_library(_tracks(1), 1), out)
    assert remove_library(out) and not out.exists()
    assert remove_library(out) is False


def test_settings_path_is_read_from_tsi(tmp_path):
    tsi = tmp_path / "Traktor Settings.tsi"
    tsi.write_text('<?xml version="1.0"?><NIXML><TraktorSettings>'
                   '<Entry Name="Browser.Dir.ITunesMusic" Type="3" Value="C:\\Users\\x\\Music\\iTunes\\'
                   'iTunes Music Library.xml"></Entry></TraktorSettings></NIXML>', encoding="utf-8")
    assert settings_itunes_path(tsi) == Path("C:\\Users\\x\\Music\\iTunes\\iTunes Music Library.xml")
