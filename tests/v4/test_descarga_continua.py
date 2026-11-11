"""
PURPOSE: El encadenamiento de D5/D6 (tools/dj_sets/descarga_continua.py), sin red: qué paso toca en cada carril según
         lo que hay en disco (rescate, buscar, enlaces, bajar, lote nuevo, YouTube), las marcas de vacío, las esperas
         tras un corte y cuándo se da todo por terminado.
CHANGELOG:
  - 2026-11-10: Creación.
"""
import json
from pathlib import Path

from tools.dj_sets import beatport_previews as bpv
from tools.dj_sets import descarga_continua as dc


def fila(clave, **extra):
    return dict({"clave": clave, "nombre": f"A - {clave}", "artista": "A", "titulo": clave, "con_boton": "si",
                 "nivel_busqueda": "B"}, **extra)


def corrida(base: Path, filas, tandas=0, temas=()):
    temas = list(temas)
    bpv.escribir_tabla(base / "temas.csv", filas)
    for k in range(tandas):
        (base / f"{bpv.nombre_tanda(k)}.json").write_text(json.dumps({"temas": temas}), encoding="utf-8")


def test_la_cadena_completa_paso_a_paso(tmp_path):
    bp_dir, yt_dir = tmp_path / "beatport_previews", tmp_path / "youtube_temas"
    corrida(bp_dir / "lote_01", [fila("x")], tandas=1)  # ya rescatado por rescate_01
    corrida(bp_dir / "prueba3", [fila("p")], tandas=1)  # pruebas: no se rescatan
    corrida(bp_dir / "lote_04", [fila(f"l{i}") for i in range(600)], tandas=1)
    corrida(bp_dir / "rescate_01", [fila("a"), fila("b")], tandas=1)
    corrida(yt_dir / "yt_01", [fila(f"y{i}") for i in range(250)], tandas=1)
    temporal = str(bp_dir / "tmp")

    # Primero lo corto: el rescate del rescate terminado, mientras lote_04 sigue a medias.
    assert dc.tarea_beatport(tmp_path, 10) == ("rescate", ["--corrida", "res_rescate_01", "rescate", "--de", "rescate_01"])
    corrida(bp_dir / "res_rescate_01", [fila("a", nivel_busqueda=""), fila("b", nivel_busqueda="")])
    assert dc.tarea_beatport(tmp_path, 10) == ("buscar", ["--corrida", "res_rescate_01", "buscar"])
    assert dc.tarea_1001(tmp_path, 0) is None  # sin buscar todavía, no se piden enlaces

    # Buscado: «b» tiene botón y la búsqueda no lo encontró: espera el enlace; Beatport sigue con lote_04.
    corrida(bp_dir / "res_rescate_01", [fila("a", nivel_busqueda="D", con_boton=""), fila("b", nivel_busqueda="C")])
    assert dc.tarea_beatport(tmp_path, 10) == ("bajar", ["--corrida", "lote_04", "bajar", "--temporal", temporal, "--max-tandas", "1"])
    assert dc.tarea_1001(tmp_path, 7) == ("sets", ["bajar", "--n", "200"])  # los sets primero
    assert dc.tarea_1001(tmp_path, 0) == ("enlaces", ["--corrida", "res_rescate_01", "enlaces", "--solo-sin-id", "--n", "200"])

    # Con el enlace pedido, se baja el rescate antes que el lote grande.
    corrida(bp_dir / "res_rescate_01", [fila("a", nivel_busqueda="D", con_boton=""), fila("b", nivel_busqueda="C", fuente_1001="1")])
    assert dc.tarea_1001(tmp_path, 0) is None
    assert dc.tarea_beatport(tmp_path, 10) == ("bajar", ["--corrida", "res_rescate_01", "bajar", "--temporal", temporal, "--max-tandas", "1"])

    # YouTube: primero termina yt_01; el rescate bajado da una corrida nueva sin repetir los temas de yt_01.
    assert dc.tarea_youtube(tmp_path) == ("bajar", ["--corrida", "yt_01", "bajar", "--temporal", str(yt_dir / "tmp")])
    corrida(yt_dir / "yt_01", [fila(f"y{i}") for i in range(250)], tandas=2)
    assert dc.tarea_youtube(tmp_path) is None  # el rescate todavía no está bajado
    corrida(bp_dir / "res_rescate_01", [fila("a", nivel_busqueda="D", con_boton=""), fila("b", nivel_busqueda="C", fuente_1001="1")], tandas=1)
    assert dc.tarea_youtube(tmp_path) == ("elegir", ["--corrida", "yt_res_rescate_01", "elegir", "--de", "res_rescate_01",
                                                    "--excluir", "yt_01"])
    dc.marcar_vacia(yt_dir / "yt_res_rescate_01")
    assert dc.tarea_youtube(tmp_path) is None

    # lote_04 terminado: su rescate; si no tuvo temas sin resolver, queda marcado y sigue un lote nuevo con los sets nuevos.
    assert dc.tarea_beatport(tmp_path, 10) == ("bajar", ["--corrida", "lote_04", "bajar", "--temporal", temporal, "--max-tandas", "1"])
    corrida(bp_dir / "lote_04", [fila(f"l{i}") for i in range(600)], tandas=2)
    assert dc.tarea_beatport(tmp_path, 10) == ("rescate", ["--corrida", "res_lote_04", "rescate", "--de", "lote_04"])
    dc.marcar_vacia(bp_dir / "res_lote_04")
    assert dc.tarea_beatport(tmp_path, 10) == ("elegir", ["--corrida", "lote_05", "elegir", "--todos", "--n", "1000000",
                                                          "--excluir", "lote_01", "--excluir", "lote_04"])
    (bp_dir / "sets_en_lotes.txt").write_text("10", encoding="utf-8")
    assert dc.tarea_beatport(tmp_path, 10) is None
    assert dc.tarea_beatport(tmp_path, 11) is not None  # llegó un set nuevo

    # Todo hecho solo si los tres carriles están ociosos, no quedan sets y nada tiene tarea.
    o = dc.Orquesta(tmp_path, Path("python"), Path("navegadores"))
    o.ocioso = {"1001": True, "beatport": True, "youtube": True}
    o.sets_pendientes = 0
    assert dc.todo_hecho(o)
    o.sets_pendientes = None  # no se pudo contar: no se da por terminado
    assert not dc.todo_hecho(o)
    o.sets_pendientes, o.ocioso["youtube"] = 0, False
    assert not dc.todo_hecho(o)


def test_esperas_crecen_con_tope_y_la_pausa_de_youtube_se_guarda(tmp_path):
    assert [dc.espera_corte("1001", n) / 3600 for n in (1, 2, 3, 4, 9)] == [2, 3, 4, 4, 4]
    assert [dc.espera_corte("youtube", n) / 60 for n in (1, 2, 3, 4)] == [60, 90, 120, 120]
    assert [dc.espera_corte("beatport", n) / 60 for n in (1, 2, 3, 4)] == [30, 60, 120, 120]
    assert [dc.espera_error(n) / 60 for n in (1, 2, 3, 4, 5)] == [10, 20, 40, 60, 60]
    assert dc.nueva_pausa_yt(6, 3 * 3600) == 9 and dc.nueva_pausa_yt(9, 60) == 9  # al arrancar: el mismo bloqueo
    assert dc.nueva_pausa_yt(25, 3600) == dc.PAUSA_YT_TOPE
    assert dc.leer_pausa_yt(tmp_path) == dc.PAUSA_YT_INICIAL
    (tmp_path / "pausa_s.txt").write_text("13.5", encoding="utf-8")
    assert dc.leer_pausa_yt(tmp_path) == 13.5


def test_corrida_completa_por_nombre_de_tanda(tmp_path):
    corrida(tmp_path / "c", [fila(str(i)) for i in range(501)], tandas=1)
    assert not dc.completa(tmp_path / "c", 500)
    (tmp_path / "c" / "tanda_00000_v1.json.viejo").write_text("{}", encoding="utf-8")
    assert not dc.completa(tmp_path / "c", 500)
    (tmp_path / "c" / "tanda_00001.json").write_text("{}", encoding="utf-8")
    assert dc.completa(tmp_path / "c", 500)
    assert not dc.completa(tmp_path / "vacia", 500)


def test_el_guardian_relanza_solo_si_correr_murio_sin_terminar_ni_pausa(tmp_path):
    # 2026-11-11 13:28: correr murió sin dejar rastro y la descarga quedó parada 19 min.
    vivos = {111}
    root = tmp_path / "dj_sets"
    assert dc.que_hace_el_guardian(root, vivos.__contains__) == ("sin_disco", 0)
    root.mkdir()
    assert dc.que_hace_el_guardian(root, vivos.__contains__) == ("relanzar", 0)  # nunca arrancó
    (root / "descarga_continua.pid").write_text("111", encoding="utf-8")
    assert dc.que_hace_el_guardian(root, vivos.__contains__) == ("corre", 111)
    vivos.clear()
    assert dc.que_hace_el_guardian(root, vivos.__contains__) == ("relanzar", 111)  # se murió
    (root / "PARAR").write_text("", encoding="utf-8")
    assert dc.que_hace_el_guardian(root, vivos.__contains__)[0] == "pausa"  # pausa de Gabriel: no se relanza
    (root / dc.FIN_MARCA).write_text("", encoding="utf-8")
    assert dc.que_hace_el_guardian(root, vivos.__contains__)[0] == "fin"


def test_el_registro_no_tumba_el_programa_si_el_disco_falla(tmp_path):
    o = dc.Orquesta(tmp_path / "no_existe", Path("python"), Path("navegadores"))
    o.log("main", "esto no se puede escribir y no debe romper nada")  # la carpeta no existe: OSError, ignorado


def test_la_copia_al_drive_deja_afuera_solo_cache_y_temporales():
    # 2026-11-11 (Gabriel): nada del dataset solo en el pendrive. rclone copy (nunca sync: no borra en el Drive).
    cmd = dc.comando_copia(Path("D:/datos/dj_sets"), "rclone", "erclm:traktor_ml/dj_sets")
    assert cmd[:4] == ["rclone", "copy", str(Path("D:/datos/dj_sets")), "erclm:traktor_ml/dj_sets"]
    excluidos = [cmd[i + 1] for i, x in enumerate(cmd) if x == "--exclude"]
    assert set(excluidos) == set(dc.EXCLUIR_COPIA)
    assert all("cache" in e or "tmp" in e or e in ("descarga_continua.pid", "espera_*.txt", "PARAR") for e in excluidos)
    assert "sync" not in cmd and "--delete-excluded" not in cmd


def test_la_espera_ante_un_sitio_sobrevive_a_un_reinicio(tmp_path):
    o = dc.Orquesta(tmp_path, Path("python"), Path("navegadores"))
    o.parar.set()  # que no espere de verdad
    assert dc.espera_pendiente(tmp_path, "youtube") == 0
    o.esperar_sitio("youtube", 3600, "verificación anti-robots")
    assert 3500 < dc.espera_pendiente(tmp_path, "youtube") <= 3601  # la hora se guarda redondeada al segundo
    assert dc.espera_pendiente(tmp_path, "1001") == 0
    (tmp_path / "espera_1001.txt").write_text("1", encoding="utf-8")  # ya pasó
    assert dc.espera_pendiente(tmp_path, "1001") == 0


def test_youtube_adelanta_los_sin_boton_de_un_lote_a_medias(tmp_path):
    # 2026-11-10: YouTube se quedaba sin trabajo hasta que terminaran lote_04, su rescate y sus enlaces; los temas sin
    # botón que Beatport no dio no esperan nada.
    bp_dir, yt_dir = tmp_path / "beatport_previews", tmp_path / "youtube_temas"
    filas = [fila(f"s{i}", con_boton="") for i in range(250)] + [fila(f"b{i}") for i in range(50)] + [fila(f"x{i}") for i in range(700)]
    sin_resolver = [{"clave_1001": f["clave"], "estado": "sin_id"} for f in filas[:300]]
    corrida(bp_dir / "lote_04", filas, tandas=1, temas=sin_resolver)
    corrida(yt_dir / "yt_01", [fila("otro")], tandas=1)
    tarea = dc.tarea_youtube(tmp_path)
    assert tarea == ("elegir", ["--corrida", "yt_lote_04_01", "elegir", "--de", "lote_04", "--solo-sin-boton", "--excluir", "yt_01"])
    # Ya tomados: con menos de 200 nuevos y el lote a medias, espera a juntar más.
    corrida(yt_dir / "yt_lote_04_01", [fila(f"s{i}") for i in range(100, 250)])
    assert dc.tarea_youtube(tmp_path) == ("bajar", ["--corrida", "yt_lote_04_01", "bajar", "--temporal", str(yt_dir / "tmp")])
    corrida(yt_dir / "yt_lote_04_01", [fila(f"s{i}") for i in range(100, 250)], tandas=1)
    assert dc.tarea_youtube(tmp_path) is None
    # Con el lote terminado, van los que queden aunque sean menos de 200.
    corrida(bp_dir / "lote_04", filas, tandas=2, temas=sin_resolver)
    assert dc.tarea_youtube(tmp_path)[1][:2] == ["--corrida", "yt_lote_04_02"]
    # lote_01..03 no: sus temas ya pasaron por rescate_01 y yt_01.
    corrida(bp_dir / "lote_02", filas[:300], tandas=1, temas=sin_resolver)
    assert "lote_02" not in dc.tarea_youtube(tmp_path)[1]
