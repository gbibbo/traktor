"""
PURPOSE: Tests de tools/playlist_review/pagina_escucha.py: lectura de las listas, marca de los temas comunes, rutas como
         file:// y que la clave de la escucha no entre en la página.
CHANGELOG:
  - 2026-11-13: Creación inicial.
"""
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.playlist_review import pagina_escucha as pe  # noqa: E402


def lista(path: Path, temas):
    lineas = ["#EXTM3U"]
    for titulo, ruta in temas:
        lineas += [f"#EXTINF:-1,{titulo}", ruta]
    path.write_text("\n".join(lineas) + "\n", encoding="utf-8")


@pytest.fixture()
def carpeta(tmp_path):
    d = tmp_path / "e99" / "listas"
    d.mkdir(parents=True)
    base = r"C:\Música ñ\carpeta"
    p1 = ("Uno - Principal", base + r"\Uno - Principal.mp3")
    lista(d / "01_A.m3u8", [p1, ("X - Común", base + r"\X.mp3"), ("Y - Solo A", base + r"\Y.mp3")])
    lista(d / "01_B.m3u8", [p1, ("Z - Solo B", base + r"\Z.mp3"), ("X - Común", base + r"\X.mp3")])
    p2 = ("Dos - Principal", base + r"\Dos.mp3")
    lista(d / "02_A.m3u8", [p2, ("M - a", base + r"\M.mp3")])
    lista(d / "02_B.m3u8", [p2, ("N - b", base + r"\N.mp3")])
    (d.parent / "clave.json").write_text('{"01": "SECRETO_DE_LA_CLAVE"}', encoding="utf-8")
    (d / "clave.json").write_text('{"01": "SECRETO_DE_LA_CLAVE"}', encoding="utf-8")
    return d


def datos_de(html: str) -> dict:
    return json.loads(re.search(r"const D = (\{.*?\});\nconst CLAVE", html, re.S).group(1))


def test_arma_los_pares_y_marca_los_comunes(carpeta):
    d = pe.armar(carpeta)
    assert [t["n"] for t in d["temas"]] == ["01", "02"]
    t1 = d["temas"][0]
    assert t1["p"]["t"] == "Uno - Principal"
    assert [(c["t"], c["c"]) for c in t1["A"]] == [("X - Común", 1), ("Y - Solo A", 0)]
    assert [(c["t"], c["c"]) for c in t1["B"]] == [("Z - Solo B", 0), ("X - Común", 1)]
    assert t1["comunes"] == 1 and d["temas"][1]["comunes"] == 0
    assert t1["p"]["u"].startswith("file:///C:/M%C3%BAsica%20%C3%B1/")


def test_la_pagina_no_lleva_la_clave(carpeta, capsys):
    out = carpeta.parent / "escucha.html"
    assert pe.main(["--listas", str(carpeta)]) == 0
    html = out.read_text(encoding="utf-8")
    assert "SECRETO_DE_LA_CLAVE" not in html
    d = datos_de(html)
    assert d["id"] == "e99" and len(d["temas"]) == 2
    assert "[OK]" in capsys.readouterr().out


def test_falla_si_el_principal_no_coincide(carpeta):
    lista(carpeta / "02_B.m3u8", [("Otro", r"C:\otro.mp3"), ("N - b", r"C:\N.mp3")])
    with pytest.raises(SystemExit):
        pe.armar(carpeta)
