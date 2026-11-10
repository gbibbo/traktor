"""
PURPOSE: Curado y lectura de páginas de 1001Tracklists (tools/dj_sets/tracklists_1001.py), con fragmentos de
         HTML escritos a mano: duraciones e identificados de las filas, descarte de sets comerciales, los dos
         usos (trayectoria y pares), topes por DJ y por programa, y calidad de un set.
CHANGELOG:
  - 2026-11-09: Creación.
"""
from pathlib import Path

import pytest

from tools.dj_sets import tracklists_1001 as t

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def config():
    return t.cargar_config(REPO / "config" / "dj_sets" / "tracklists_1001.json")


def fila(titulo, minutos=360, total=90, identificados=80, generos=("Techno",), fecha="2025-01-01", iconos=(), id_="x"):
    return {"id": id_, "ruta": f"/tracklist/{id_}/a.html", "titulo": titulo, "fecha": fecha, "minutos": minutos, "total": total,
            "identificados": identificados, "vistas": 1, "generos": list(generos), "iconos": list(iconos), "consultas": []}


FILA_HTML = """<div class="bItm action oItm" onclick="x" data-id="14ycg27k" id="14ycg27k"><img src="a"><div class="bCont">
<div class="bTitle" onclick="y"><a href="/tracklist/14ycg27k/massano.html" class="">Massano @ All Night Long, Lofi &amp; Co</a></div></div>
<div class="mediaRow iRow"><div class="badge views" title="tracklist views" data-count="1166"><i></i>1.2k</div>
<div title="tracklist date"><i></i>2026-09-12</div><div title="IDed tracks / total tracks"><i></i>{ided}</div>
<div title="play time"><i></i>{dur}</div><div><i title="with youtube video" class="fa fa-video-camera"></i></div>
<div title="musicstyle(s)"><i></i>Tech House, Techno</div></div></div> """


@pytest.mark.parametrize("texto,minutos", [("6h ", 360), ("2h 1m", 121), ("45m", 45), ("1d 2h", 1560), ("", 0)])
def test_minutos(texto, minutos):
    assert t.minutos_de_texto(texto) == minutos


@pytest.mark.parametrize("texto,esperado", [("90/105", (90, 105)), ("all /73", (73, 73)), ("00/15", (0, 15)), ("", (0, 0))])
def test_identificados(texto, esperado):
    assert t.identificados_de_texto(texto) == esperado


def test_parsear_lista():
    filas = t.parsear_lista(FILA_HTML.format(ided="90/105", dur="6h ") + FILA_HTML.format(ided="all /73", dur="2h 1m"))
    assert len(filas) == 2
    assert filas[0]["titulo"] == "Massano @ All Night Long, Lofi & Co"
    assert (filas[0]["identificados"], filas[0]["total"], filas[0]["minutos"]) == (90, 105, 360)
    assert filas[0]["generos"] == ["Tech House", "Techno"]
    assert filas[0]["vistas"] == 1166
    assert filas[0]["iconos"] == ["with youtube video"]
    assert filas[1]["identificados"] == 73


@pytest.mark.parametrize("titulo,generos,motivo", [
    ("David Guetta @ Club", ("Techno",), "comercial_dj"),
    ("Martin Garrix @ Club", ("House",), "comercial_dj"),
    ("Bassjackers @ All Night Long, Toffler", ("Mainstage", "Techno"), "comercial_dj"),
    ("Desconocido @ Club", ("Mainstage", "Techno"), "comercial_genero"),
    ("Desconocido @ Club", ("Trance", "Techno"), "fuera_de_estilo"),
    ("Desconocido @ Club", ("Latin Music",), "comercial_genero"),
    ("Desconocido @ Club", ("Salsa",), "sin_genero_aceptado"),
])
def test_descarta_comerciales(config, titulo, generos, motivo):
    salida = t.evaluar(fila(titulo, generos=generos), config)
    assert salida["motivo"] == motivo and salida["uso"] == ""


def test_dj_comercial_en_b2b(config):
    assert t.evaluar(fila("Carl Cox & David Guetta @ Club"), config)["motivo"] == "comercial_dj"


@pytest.mark.parametrize("dj", ["ARTBAT", "Argy", "Boris Brejcha", "Solomun", "Tale Of Us", "Charlotte de Witte", "Amelie Lens", "Nina Kraviz"])
def test_no_son_comerciales_para_gabriel(config, dj):
    assert dj in config["djs"], "también se buscan"
    assert t.evaluar(fila(f"{dj} @ Club"), config)["motivo"] == ""


@pytest.mark.parametrize("titulo,generos", [
    # Las etiquetas del sitio y los festivales grandes no sacan a un DJ de las semillas.
    ("John Digweed @ Sziget Events Hall", ("Progressive House", "Mainstage")),
    ("Charlotte de Witte @ Mainstage, Tomorrowland", ("Hard Dance",)),
    ("Carl Cox @ Resistance Stage, Ultra Music Festival Miami", ("Techno",)),
    ("Mathame @ Tomorrowland", ("Melodic House/Techno",)),
])
def test_las_semillas_no_salen_por_etiqueta_ni_evento(config, titulo, generos):
    assert t.evaluar(fila(titulo, generos=generos), config)["motivo"] == ""


def test_un_festival_no_vuelve_comercial_a_nadie(config):
    assert t.evaluar(fila("Desconocido @ Stage, Tomorrowland"), config)["motivo"] == ""


@pytest.mark.parametrize("titulo,minutos,total,ided,uso,motivo", [
    ("A @ Club", 360, 90, 80, "trayectoria+pares", ""),
    # Noche larga poco identificada: sirve para la trayectoria, no para pares (el piso de ID es menor).
    ("A @ Club", 360, 80, 50, "trayectoria", ""),
    # Sets cortos: alcanzan para saber qué temas van juntos (pedido de Gabriel).
    ("A @ Club", 90, 20, 19, "pares", ""),
    ("A - Programa 1149", 120, 21, 20, "pares", ""),
    # Una lista de 8 h con 17 temas está incompleta.
    ("A @ Club", 480, 17, 16, "", "densidad"),
    ("A @ Club", 45, 12, 12, "", "corto"),
    ("A @ Club", 0, 12, 12, "", "sin_duracion"),
    ("A @ Club", 120, 30, 15, "", "pocos_identificados"),
    # Stream: no es una noche frente a un público, pero sus pares valen.
    ("A @ Twitch Marathon", 360, 90, 80, "pares", ""),
    ("A + More @ Festival", 360, 90, 80, "pares", ""),
])
def test_usos(config, titulo, minutos, total, ided, uso, motivo):
    salida = t.evaluar(fila(titulo, minutos=minutos, total=total, identificados=ided), config)
    assert (salida["uso"], salida["motivo"]) == (uso, motivo)


def test_pig_and_dan_no_es_b2b():
    assert t.djs_de("Pig&Dan") == ["Pig&Dan"]
    assert t.djs_de("Carl Cox & Joseph Capriati") == ["Carl Cox", "Joseph Capriati"]
    assert t.djs_de("A + More") == ["A"]


def test_contiene_nombre_exige_palabra_entera():
    assert t.contiene_nombre("Hernán Cattáneo", "Hernan Cattaneo & Edu Imbernon")
    assert not t.contiene_nombre("Anna", "Annabel Ross")
    assert not t.contiene_nombre("ANNA", "Anna Lunoe & Dza")
    assert t.contiene_nombre("ANNA", "ANNA & Antdot")
    assert t.contiene_nombre("Carl Cox", "Carl Cox presents Friends")


def test_topes_por_dj_y_programa(config):
    largas = [t.evaluar(fila("A @ Club", id_=f"a{i}"), config) for i in range(20)]
    aceptados, descartados = t.seleccionar(largas, {})
    assert len(aceptados) == t.TOPE_POR_DJ
    assert {f["motivo"] for f in descartados} == {"tope_dj"}

    programas = [t.evaluar(fila(f"{chr(65 + i % 26)}{i} - Show {i}", minutos=120, total=20, identificados=20, id_=f"p{i}"), config) for i in range(40)]
    aceptados, descartados = t.seleccionar(programas, {})
    assert len(aceptados) == t.TOPE_POR_PROGRAMA
    assert {f["motivo"] for f in descartados} == {"tope_programa"}


def test_trayectorias_van_primero(config):
    corta = t.evaluar(fila("B - Show 1", minutos=120, total=20, identificados=20, id_="p", iconos=["with youtube video"]), config)
    larga = t.evaluar(fila("A @ Club", id_="t"), config)
    aceptados, _ = t.seleccionar([corta, larga], {})
    assert [f["id"] for f in aceptados] == ["t", "p"]


def test_grupo_noche():
    assert t.grupo_noche({"titulo": "A @ Club Part 1"}) == t.grupo_noche({"titulo": "A @ Club Part 2"}) != ""
    assert t.grupo_noche({"titulo": "A @ Club"}) == ""


def test_consultas_nombres_cortos_y_ventanas(config):
    consultas = {c["id"]: c for c in t.generar_consultas(config)}
    assert consultas["dj__anna"]["texto"] == "ANNA @"
    assert consultas["dj__carl-cox"]["texto"] == "Carl Cox"
    assert consultas["dj__carl-cox"]["duracion"] == 60
    assert t.campos_formulario(consultas["termino__all-night-long__2025"])["endDate"] == "2025-12-31"
    assert "endDate" not in t.campos_formulario(consultas["termino__all-night-long__2026"])
    assert len(consultas) == len(t.generar_consultas(config))  # ids únicos
    assert not any(c["dj"] in config["djs_comerciales"] for c in consultas.values() if c["tipo"] == "dj")


TEMA_HTML = """<div id="tlp_{tlp}" class="tlpTog bItm tlpItem trRow4{extra}" data-trno="3" data-id="{tlp}" data-isided="{ided}" data-trackid="ujn67dhf" onclick="r">
<div class="bPlay"><input id="tlp{tlp}_cue_seconds" type="hidden" value="{cue}" form="f"><span id="tlp3_tracknumber_value" class="fontXL"> {num} </span></div>
<div class="bCont tl"><meta itemprop="name" content="{nombre}"><meta itemprop="byArtist" content="Hardt Antoine">
<meta itemprop="publisher" content="&lt;span&gt;CROSSTOWN REBELS&lt;/span&gt;"><meta itemprop="duration" content="PT5M58S"><meta itemprop="genre" content="{genero}">
<meta itemprop="url" content="/track/ujn67dhf/a/index.html"><span class="trackValue notranslate blueTxt" id="tr_1"> <span class="notranslate">{nombre}<a href="/x"><i></i></a></span> </span> <span class="italic trackEditData spL">(Acappella)</span> <span class="iBlock">x</span>
<div class="iRow grow mediaRow" data-trackid="1122250"><i onclick="new MediaViewer(this,null,{{idObject:5,idItem:1,idSource: '13'}})"></i><i onclick="new MediaViewer(this,null,{{idObject:5,idItem:1,idSource: 'affiliate'}})"></i></div>
</div></div>
"""

SET_HTML = (
    '<html><head><title>A @ Club 2026-07-25</title><link rel="canonical" href="https://x/tracklist/abc/a.html"></head><body>'
    '<div id="mediaLink55" data-idmedia="55" data-idsource="13" class="mediaLink mediaItemzMrESxTnw0M"><meta itemprop="duration" content="PT493M15S"></div>'
    '<div id="mediaLink56" data-idmedia="56" data-idsource="16" class="mediaLink "><iframe id="mc" src="https://mixcloud/x?a=1&amp;b=2"></iframe></div>'
    + TEMA_HTML.format(tlp=1, extra="", ided="true", cue=0, num="01", nombre="Hardt Antoine - Dreamstate", genero="House")
    + TEMA_HTML.format(tlp=2, extra=" con", ided="true", cue=120, num="w/", nombre="Otro - Acapella", genero="House")
    + TEMA_HTML.format(tlp=3, extra="", ided="true", cue=300, num="02", nombre="Biblio Artista - Tema", genero="Mainstage")
    + TEMA_HTML.format(tlp=4, extra="", ided="false", cue=600, num="03", nombre="ID - ID", genero="")
    + TEMA_HTML.format(tlp=5, extra="", ided="true", cue=900, num="04", nombre="C - D", genero="House")
    + "</body></html>"
)


def test_parsear_set():
    dato = t.parsear_set(SET_HTML)
    assert dato["titulo"] == "A @ Club 2026-07-25"
    assert dato["url"] == "https://x/tracklist/abc/a.html"
    assert [g["id_externo"] for g in dato["grabaciones"]] == ["zMrESxTnw0M", ""]
    assert dato["grabaciones"][1]["iframe"] == "https://mixcloud/x?a=1&b=2"
    assert dato["grabaciones"][0]["duracion_s"] == 493 * 60 + 15
    tema = dato["temas"][0]
    assert (tema["nombre"], tema["sello"], tema["duracion_s"], tema["segundo"], tema["id_numerico"]) == ("Hardt Antoine - Dreamstate", "CROSSTOWN REBELS", 358, 0, "1122250")
    assert (tema["visible"], tema["edicion"]) == ("Hardt Antoine - Dreamstate", "(Acappella)")
    assert tema["fuentes"] == ["13", "affiliate"]
    assert dato["temas"][1]["junto_al_anterior"] and dato["temas"][1]["numero"] == "w/"
    assert not dato["temas"][3]["identificado"]


def test_texto_visible_con_span_anidados():
    # La estructura real del sitio: artista, separador y título, cada uno con sus propios span.
    bloque = ('<span class="trackValue notranslate blueTxt" id="tr_u"> <span class="notranslate blueTxt">Hardt Antoine'
              '<span title="open artist page" class="tgHid spL"><a href="/a"><i class="fa"></i></a></span></span>'
              '<span class="notranslate "> - </span><span class="blueTxt"><span class="notranslate">Dreamstate</span>'
              '<span title="open track page"><a href="/t"><i></i></a></span></span> </span> '
              '<span class="iBlock notranslate"><span title="label">CROSSTOWN REBELS</span></span>')
    assert t.texto_visible(bloque) == "Hardt Antoine - Dreamstate"


def test_medir_set():
    medida = t.medir_set(t.parsear_set(SET_HTML), {"biblio artista"}, {"Mainstage"})
    # La fila tocada encima no cuenta; de los 4 temas, 3 identificados y un solo par consecutivo identificado.
    assert (medida["temas"], medida["pct_id"], medida["pares_consecutivos"], medida["tramo_mas_largo"]) == (4, 75, 1, 2)
    # El tiempo 0 del primer tema es válido.
    assert (medida["con_tiempo"], medida["pct_tiempo"]) == (4, 100)
    assert medida["pct_genero_comercial"] == 33
    assert medida["pct_artista_en_biblioteca"] == 25
