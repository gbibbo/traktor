"""
PURPOSE: Tests de traktor_collection.py con un paquete y una colección de Traktor falsos: prepare
         saca solo los temas sin carátula en Traktor, con carátula en el archivo y sin cues propios,
         y actualiza el texto de los demás; playlists se niega si falta un tema y, si están todos,
         agrega la carpeta con sus playlists en orden (y la reemplaza al repetir).
CHANGELOG:
  - 2026-10-30: Creación inicial.
"""
import io
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path, PureWindowsPath

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.v4.pipeline.traktor_collection import _has_own_cues, add_playlists, prepare  # noqa: E402
from src.v4.pipeline.usb_package import copy_files, plan_layout, unique_files, write_package_files  # noqa: E402
from test_usb_package import _fixture, tiny_mp3  # noqa: E402


def _tag(path: Path, title: str, cover: bool):
    from mutagen.id3 import APIC, ID3, TALB, TIT2, TPE1, TPE4
    from PIL import Image
    tiny_mp3(path)
    t = ID3()
    t.add(TIT2(encoding=3, text=[title]))
    t.add(TPE1(encoding=3, text=["New Artist"]))
    t.add(TALB(encoding=3, text=["New Album"]))
    t.add(TPE4(encoding=3, text=["Extended Mix"]))
    if cover:
        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, format="JPEG")
        t.add(APIC(encoding=0, mime="image/jpeg", type=3, desc="", data=buf.getvalue()))
    t.save(path, v2_version=3)


def _entry(path: Path, title: str, cover: bool, cues: str = "") -> str:
    from xml.sax.saxutils import quoteattr
    p = PureWindowsPath(str(path))
    d = "/:" + "".join(f"{x}/:" for x in p.parts[1:-1])
    info = '<INFO PLAYTIME="300"' + (' COVERARTID="001/ABC"' if cover else "") + "></INFO>"
    return (f'<ENTRY TITLE={quoteattr(title)} ARTIST="Old"><LOCATION DIR={quoteattr(d)} FILE={quoteattr(p.name)} '
            f'VOLUME="{p.drive}" VOLUMEID="x"></LOCATION>{info}{cues}</ENTRY>\n')


GRID = ('<CUE_V2 NAME="AutoGrid" TYPE="0" START="10.0" HOTCUE="0"></CUE_V2>'
        '<CUE_V2 NAME="AutoGrid" TYPE="4" START="10.0" HOTCUE="-1"></CUE_V2>')
OWN = GRID + '<CUE_V2 NAME="drop" TYPE="0" START="95.5" HOTCUE="1"></CUE_V2>'


def test_has_own_cues():
    e = ET.fromstring(f"<ENTRY>{GRID}</ENTRY>")
    assert not _has_own_cues(e) and _has_own_cues(ET.fromstring(f"<ENTRY>{OWN}</ENTRY>"))


def test_prepare_and_playlists():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        tracks, specs = _fixture(t)
        pkg = t / "TRAKTOR ML"
        pl = plan_layout(specs, tracks)
        files = unique_files(pl)
        copy_files(files, pkg)
        write_package_files(pkg, tracks, specs, pl, "TRAKTOR ML test v1")
        path = {p.track_uid: pkg / p.rel_path for p in files}
        _tag(path["u1"], "One (Extended Mix)", cover=True)    # sin carátula en Traktor: se saca
        _tag(path["u2"], "Two", cover=True)                   # sin carátula pero con cue propio: queda
        other = t / "other.mp3"
        tiny_mp3(other)
        col = t / "collection.nml"
        col.write_text(
            '<?xml version="1.0" encoding="UTF-8" standalone="no" ?>\n<NML VERSION="20"><COLLECTION ENTRIES="3">\n'
            + _entry(path["u1"], "Old 1", False, GRID) + _entry(path["u2"], "Old 2", False, OWN)
            + _entry(other, "Mine", False, GRID)
            + '</COLLECTION><PLAYLISTS><NODE TYPE="FOLDER" NAME="$ROOT"><SUBNODES COUNT="1">'
            '<NODE TYPE="PLAYLIST" NAME="Mine"><PLAYLIST ENTRIES="1" TYPE="LIST" UUID="u">'
            f'<ENTRY><PRIMARYKEY TYPE="TRACK" KEY="{PureWindowsPath(str(path["u1"])).drive}x"></PRIMARYKEY></ENTRY>'
            '</PLAYLIST></NODE></SUBNODES></NODE></PLAYLISTS></NML>\n', encoding="utf-8")
        bdir = t / "bk"

        stats = prepare(col, pkg, bdir)
        assert (stats["removed"], stats["kept"], stats["updated"]) == (1, 1, 1)
        root = ET.parse(col).getroot()
        titles = {e.get("TITLE"): e for e in root.find("COLLECTION")}
        assert set(titles) == {"Two", "Mine"} and root.find("COLLECTION").get("ENTRIES") == "2"
        two = titles["Two"]
        assert two.get("ARTIST") == "New Artist" and two.find("ALBUM").get("TITLE") == "New Album"
        assert two.find("INFO").get("REMIXER") == "Extended Mix" and len(two.findall("CUE_V2")) == 3
        assert titles["Mine"].get("ARTIST") == "Old"           # fuera del paquete no se toca
        assert len(list(bdir.glob("collection_*.nml"))) == 1

        # faltan temas del paquete en la colección: no escribe nada
        before = col.read_bytes()
        stats = add_playlists(col, pkg, bdir)
        assert stats["missing"] == 2 and col.read_bytes() == before

        # como si Traktor hubiera agregado la carpeta Musica
        text = col.read_text(encoding="utf-8").replace(
            "</COLLECTION>", _entry(path["u1"], "One (Extended Mix)", True) + _entry(path["u3"], "Three", True)
            + "</COLLECTION>")
        col.write_text(text, encoding="utf-8")
        for _ in range(2):                                    # repetir reemplaza la carpeta
            stats = add_playlists(col, pkg, bdir)
            assert stats["missing"] == 0
        root = ET.parse(col).getroot()
        top = root.find("PLAYLISTS/NODE/SUBNODES")
        assert [n.get("NAME") for n in top.findall("NODE")] == ["Mine", "TRAKTOR ML test v1"]
        assert top.get("COUNT") == "2"
        base = top.findall("NODE")[1]
        folder = base.find("SUBNODES/NODE")
        assert folder.get("NAME") == "A · Techno / House" and base.find("SUBNODES").get("COUNT") == "1"
        pls = folder.findall("SUBNODES/NODE")
        assert [p.get("NAME") for p in pls] == [s.name for s in specs]
        coll_keys = {e.find("LOCATION").get("VOLUME") + e.find("LOCATION").get("DIR") + e.find("LOCATION").get("FILE")
                     for e in root.find("COLLECTION")}
        for p, s in zip(pls, specs):
            keys = [pk.get("KEY") for pk in p.iter("PRIMARYKEY")]
            assert p.find("PLAYLIST").get("ENTRIES") == str(len(s.track_uids)) and all(k in coll_keys for k in keys)
            assert [k.split("/:")[-1] for k in keys] == [path[u].name for u in s.track_uids]
