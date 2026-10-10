"""
PURPOSE: Tests del paquete portátil para Traktor (src/v4/pipeline/usb_package.py): nombres válidos
         en FAT32, ubicación por playlist sin choques, plantilla NML con unidad y carpeta
         reemplazables (incluido lo que hace el .ps1), m3u8 relativos, copia reanudable y
         verificación.
CHANGELOG:
  - 2026-10-10: Preparar Traktor con colección local falsa (ficha copiada, nombre, artista/título).
  - 2026-10-10: Creación inicial.
"""
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path, PureWindowsPath

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.dj_export import PlaylistSpec  # noqa: E402
from src.v4.pipeline.usb_package import (  # noqa: E402
    copy_files, fill_template, nml_dir_prefix, plan_layout, safe_filename, safe_name, stale_files,
    unique_files, verify_files, write_package_files,
)


def _fixture(tmp: Path):
    src = tmp / "src"
    (src / "a").mkdir(parents=True)
    (src / "b").mkdir()
    files = {"u1": src / "a" / "x.mp3", "u2": src / "b" / "x.mp3", "u3": src / "a" / "R&B y.flac"}
    for i, p in enumerate(files.values()):
        p.write_bytes(bytes([i]) * (100 + i))
    tracks = pd.DataFrame({
        "track_uid": list(files), "source_path": [str(p) for p in files.values()],
        "rel_path": ["a/x.mp3", "b/x.mp3", "a/R&B y.flac"],
        "artist": ["A", "B", "Céline"], "title": ["One", "Two", "Three (Remix)"],
        "tag_comment": ["Vocal", None, None], "duration_s": [300.0, 310.0, 320.0],
    }).set_index("track_uid")
    specs = [PlaylistSpec("A · Techno / House", "A1 Techno / House (120-125)", ["u1", "u2"]),
             PlaylistSpec("A · Techno / House", "A2 R&B", ["u3", "u1"])]
    return tracks, specs


def test_safe_names():
    assert safe_name("A · Techno (Peak Time / Driving)") == "A - Techno (Peak Time - Driving)"
    assert safe_name('bad:name?*"<>|. ') == "bad_name______"
    assert safe_filename("a: b.mp3") == "a_ b.mp3"
    assert safe_filename("x" * 300 + ".flac").endswith(".flac") and len(safe_filename("x" * 300 + ".flac")) == 120


def test_plan_layout_dedupes_and_disambiguates():
    with tempfile.TemporaryDirectory() as t:
        tracks, specs = _fixture(Path(t))
        pl = plan_layout(specs, tracks)
        assert len(pl) == 4 and len(unique_files(pl)) == 3
        rel = {p.track_uid: p.rel_path for p in unique_files(pl)}
        d = r"Musica\A - Techno - House\A1 Techno - House (120-125)"
        assert rel["u1"] == d + r"\x.mp3" and rel["u2"] == d + r"\x (2).mp3"
        assert rel["u3"].startswith(r"Musica\A - Techno - House\A2 R&B" + "\\")
        # el tema repetido apunta al mismo archivo; posiciones 1-based por playlist
        assert [(p.playlist[:2], p.position, p.rel_path == rel[p.track_uid]) for p in pl] == \
               [("A1", 1, True), ("A1", 2, True), ("A2", 1, True), ("A2", 2, True)]


def test_nml_dir_prefix():
    assert nml_dir_prefix(PureWindowsPath(r"E:\TRAKTOR ML")) == ("E:", "/:TRAKTOR ML/:")
    assert nml_dir_prefix(PureWindowsPath(r"C:\Users\x\TRAKTOR ML")) == ("C:", "/:Users/:x/:TRAKTOR ML/:")


def _ps_available():
    return sys.platform == "win32" and shutil.which("powershell") is not None


def test_package_end_to_end():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        tracks, specs = _fixture(t)
        pkg = t / "dest" / "TRAKTOR ML"
        pl = plan_layout(specs, tracks)
        files = unique_files(pl)
        assert copy_files(files, pkg) == (3, 0)
        assert copy_files(files, pkg) == (0, 3)  # reanudable
        write_package_files(pkg, tracks, specs, pl, "TRAKTOR ML test v1")
        assert verify_files(files, pkg, "hash") == []

        tpl = (pkg / "_plantilla.nml").read_text(encoding="utf-8")
        assert "@@VOL@@" in tpl and "__PKG__" not in tpl
        filled = fill_template(tpl, PureWindowsPath(r"E:\otra & carpeta\TRAKTOR ML"))
        root = ET.fromstring(filled.split("\n", 1)[1])
        keys = [pk.get("KEY") for pk in root.iter("PRIMARYKEY")]
        assert len(keys) == 4 and len(set(keys)) == 3
        assert keys[0] == "E:/:otra & carpeta/:TRAKTOR ML/:Musica/:A - Techno - House/:A1 Techno - House (120-125)/:x.mp3"
        locs = root.findall("COLLECTION/ENTRY/LOCATION")
        assert len(locs) == 3 and {l.get("VOLUME") for l in locs} == {"E:"}

        # traktor.nml real apunta a archivos que existen
        real = ET.parse(pkg / "traktor.nml").getroot()
        for loc in real.iter("LOCATION"):
            p = loc.get("VOLUME") + loc.get("DIR").replace("/:", "\\") + loc.get("FILE")
            assert Path(p).exists(), p

        # m3u8 relativos resuelven desde su carpeta
        m3u = next((pkg / "Playlists m3u8").rglob("A1*.m3u8"))
        lines = [ln for ln in m3u.read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")]
        assert len(lines) == 2 and all((m3u.parent / ln.replace("\\", "/")).resolve().exists() for ln in lines)

        assert (pkg / "Preparar Traktor.bat").read_bytes().startswith(b"@echo off")
        assert stale_files(files, pkg) == []
        (pkg / "Musica" / "viejo.mp3").write_bytes(b"x")
        assert [f.name for f in stale_files(files, pkg)] == ["viejo.mp3"]

        if _ps_available():
            _check_prepare_script(t, pkg)

def _nml_loc(path: Path) -> str:
    p = PureWindowsPath(str(path))
    d = "/:" + "".join(f"{x}/:" for x in p.parts[1:-1])
    return f'<LOCATION DIR="{d}" FILE="{p.name}" VOLUME="{p.drive}" VOLUMEID="abc"/>', p.drive + d + p.name


def _check_prepare_script(t: Path, pkg: Path):
    """El .ps1 real sobre una copia movida del paquete y una colección de Traktor falsa."""
    local = t / "local"
    (local / "music" / "a").mkdir(parents=True)
    (local / "music" / "z").mkdir()
    l1, l2, l3 = local / "music" / "a" / "x.mp3", local / "music" / "z" / "x.mp3", local / "music" / "renamed.flac"
    for f in (l1, l2, l3):
        f.write_bytes(b"local")
    (loc1, k1), (loc2, _), (loc3, k3) = _nml_loc(l1), _nml_loc(l2), _nml_loc(l3)
    gone, _ = _nml_loc(local / "music" / "gone.mp3")
    collection = local / "collection.nml"
    collection.write_text(f"""<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<NML VERSION="19"><HEAD COMPANY="www.native-instruments.com" PROGRAM="Traktor"/><COLLECTION ENTRIES="4">
<ENTRY TITLE="One" ARTIST="A">{loc1}<INFO PLAYTIME="300" PLAYTIME_FLOAT="300.5"/><TEMPO BPM="128.000000" BPM_QUALITY="100.000000"/><CUE_V2 NAME="AutoGrid" TYPE="4" START="12.3"/></ENTRY>
<ENTRY TITLE="Other" ARTIST="Z">{loc2}<INFO PLAYTIME="999"/></ENTRY>
<ENTRY TITLE="Three (Remix)" ARTIST="Celine">{loc3}<INFO PLAYTIME="321"/><TEMPO BPM="124.000000"/></ENTRY>
<ENTRY TITLE="Two" ARTIST="B">{gone}<INFO PLAYTIME="310"/></ENTRY>
</COLLECTION></NML>
""", encoding="utf-8")

    moved = t / "otra" / "TRAKTOR ML"
    shutil.copytree(pkg, moved)
    (moved / "traktor.nml").unlink()
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                        str(moved / "_preparar_traktor.ps1"), "-Collection", str(collection)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "por nombre: 1, por artista y titulo: 1" in r.stdout, r.stdout
    assert "Copia del paquete (Traktor la analiza al cargarla): 1" in r.stdout
    assert "No encontrados: 0" in r.stdout

    text = (moved / "traktor.nml").read_text(encoding="utf-8")
    assert text.startswith('<?xml version="1.0" encoding="UTF-8" standalone="no" ?>\n<NML')
    root = ET.fromstring(text.split("\n", 1)[1])
    coll = root.find("COLLECTION")
    assert coll.get("ENTRIES") == "3" and len(coll.findall("ENTRY")) == 3
    by_title = {e.get("TITLE"): e for e in coll.findall("ENTRY")}
    # la ficha local se copia tal cual (análisis y cues), no la del paquete
    assert by_title["One"].find("TEMPO").get("BPM") == "128.000000"
    assert by_title["One"].find("CUE_V2").get("START") == "12.3"
    assert by_title["Three (Remix)"].get("ARTIST") == "Celine"
    pkg_key = by_title["Two"].find("LOCATION")
    assert pkg_key.get("VOLUME") == PureWindowsPath(str(moved)).drive
    k2 = pkg_key.get("VOLUME") + pkg_key.get("DIR") + pkg_key.get("FILE")
    assert "/:otra/:TRAKTOR ML/:Musica/:" in k2
    keys = [pk.get("KEY") for pk in root.iter("PRIMARYKEY")]
    assert keys == [k1, k2, k3, k1]
    rep = (moved / "_informe.txt").read_text(encoding="utf-8").splitlines()
    assert len(rep) == 3 and sum(l.startswith("copia del paquete") for l in rep) == 1

    # sin temas en la colección: todo sale del paquete
    empty = local / "empty.nml"
    empty.write_text('<?xml version="1.0" encoding="UTF-8"?><NML VERSION="19"><COLLECTION ENTRIES="0"/></NML>',
                     encoding="utf-8")
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                        str(moved / "_preparar_traktor.ps1"), "-Collection", str(empty)],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "Copia del paquete (Traktor la analiza al cargarla): 3" in r.stdout, r.stdout
    root = ET.fromstring((moved / "traktor.nml").read_text(encoding="utf-8").split("\n", 1)[1])
    for loc in root.iter("LOCATION"):
        assert Path(loc.get("VOLUME") + loc.get("DIR").replace("/:", "\\") + loc.get("FILE")).exists()


def test_verify_detects_problems():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        tracks, specs = _fixture(t)
        pkg = t / "dest"
        files = unique_files(plan_layout(specs, tracks))
        copy_files(files, pkg)
        (pkg / files[0].rel_path).write_bytes(b"\xff" * 100)  # mismo tamaño, otro contenido
        (pkg / files[1].rel_path).unlink()
        assert verify_files(files, pkg, "size") == [f"falta: {files[1].rel_path}"]
        probs = verify_files(files, pkg, "hash")
        assert len(probs) == 2 and probs[0].startswith("contenido distinto")
