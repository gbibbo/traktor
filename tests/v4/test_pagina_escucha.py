"""
PURPOSE: Tests de tools/playlist_review/pagina_escucha.py: lectura de las listas, elección de los primeros temas propios
         de cada lista, rutas como file://, duración estimada y que la clave de la escucha no entre en la página.
CHANGELOG:
  - 2026-11-13: Creación inicial.
  - 2026-11-13: Versión corta: cada lista aporta sus primeros temas propios; duración estimada.
"""
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.playlist_review import pagina_escucha as pe  # noqa: E402

BASE = r"C:\Música ñ\carpeta"


def tema(nombre):
    return (nombre, BASE + "\\" + nombre + ".mp3")


def lista(path: Path, temas):
    lineas = ["#EXTM3U"]
    for titulo, ruta in temas:
        lineas += [f"#EXTINF:-1,{titulo}", ruta]
    path.write_text("\n".join(lineas) + "\n", encoding="utf-8")


@pytest.fixture()
def carpeta(tmp_path):
    d = tmp_path / "e99" / "listas"
    d.mkdir(parents=True)
    p1 = tema("Principal uno")
    lista(d / "01_A.m3u8", [p1, tema("comun 1"), tema("a1"), tema("a2"), tema("comun 2"), tema("a3"), tema("a4")])
    lista(d / "01_B.m3u8", [p1, tema("b1"), tema("comun 2"), tema("b2"), tema("comun 1"), tema("b3"), tema("b4")])
    p2 = tema("Principal dos")
    lista(d / "02_A.m3u8", [p2, tema("m")])
    lista(d / "02_B.m3u8", [p2, tema("n")])
    (d.parent / "clave.json").write_text('{"01": "SECRETO_DE_LA_CLAVE"}', encoding="utf-8")
    (d / "clave.json").write_text('{"01": "SECRETO_DE_LA_CLAVE"}', encoding="utf-8")
    return d


def datos_de(html: str) -> dict:
    return json.loads(re.search(r"const D = (\{.*?\});\nconst CLAVE", html, re.S).group(1))


def test_cada_lista_aporta_sus_primeros_temas_propios(carpeta):
    d = pe.armar(carpeta, por_lista=3)
    assert [t["n"] for t in d["temas"]] == ["01", "02"]
    t1 = d["temas"][0]
    assert t1["p"]["t"] == "Principal uno" and t1["comunes"] == 2
    assert [(c["t"], c["puesto"]) for c in t1["A"]] == [("a1", 2), ("a2", 3), ("a3", 5)]
    assert [(c["t"], c["puesto"]) for c in t1["B"]] == [("b1", 1), ("b2", 3), ("b3", 5)]
    assert [c["t"] for c in d["temas"][1]["A"]] == ["m"]
    assert t1["p"]["u"].startswith("file:///C:/M%C3%BAsica%20%C3%B1/")


def test_duracion_estimada(carpeta):
    d = pe.armar(carpeta, por_lista=3)
    # tema 1: 7 fragmentos; tema 2: 3 fragmentos; 10 s cada uno más 5 s para decidir por tema
    assert pe.minutos(d, 10) == round((7 * 10 + 5 + 3 * 10 + 5) / 60)


def test_la_pagina_no_lleva_la_clave(carpeta, capsys):
    out = carpeta.parent / "escucha.html"
    assert pe.main(["--listas", str(carpeta)]) == 0
    html = out.read_text(encoding="utf-8")
    assert "SECRETO_DE_LA_CLAVE" not in html
    d = datos_de(html)
    assert d["id"] == "e99" and d["seg"] == 10 and len(d["temas"]) == 2
    assert "[OK]" in capsys.readouterr().out


def test_falla_si_el_principal_no_coincide(carpeta):
    lista(carpeta / "02_B.m3u8", [tema("otro principal"), tema("n")])
    with pytest.raises(SystemExit):
        pe.armar(carpeta)


def test_falla_si_las_listas_son_iguales(carpeta):
    lista(carpeta / "02_B.m3u8", [tema("Principal dos"), tema("m")])
    with pytest.raises(SystemExit):
        pe.armar(carpeta)
