"""
PURPOSE: Tests del paquete portátil para Traktor (src/v4/pipeline/usb_package.py): nombres válidos
         en FAT32, ubicación por playlist sin choques, plantilla NML con unidad y carpeta
         reemplazables (incluido lo que hace el .ps1), m3u8 relativos, copia reanudable y
         verificación.
CHANGELOG:
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
    files = {"u1": src / "a" / "x.mp3", "u2": src / "b" / "x.mp3", "u3": src / "a" / "R&B: y?.flac"
             if sys.platform != "win32" else src / "a" / "R&B y.flac"}
    for i, p in enumerate(files.values()):
        p.write_bytes(bytes([i]) * (100 + i))
    tracks = pd.DataFrame({
        "track_uid": list(files), "source_path": [str(p) for p in files.values()],
        "artist": ["A", "B", None], "title": ["One", "Two", None],
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

        if _ps_available():  # el .ps1 genera lo mismo que fill_template para su carpeta real
            moved = t / "otra" / "TRAKTOR ML"
            shutil.copytree(pkg, moved)
            (moved / "traktor.nml").unlink()
            r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                                str(moved / "_preparar_traktor.ps1")], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            assert "Faltan: 0" in r.stdout
            got = (moved / "traktor.nml").read_text(encoding="utf-8")
            assert got == fill_template(tpl, PureWindowsPath(str(moved.resolve())))


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
