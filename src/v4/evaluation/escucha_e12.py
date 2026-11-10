"""
PURPOSE: La escucha E12 del plan de playlists en vivo (sección 12.9, «La escucha E12, con las respuestas de Gabriel»):
         la lista de Parecidos de hoy (solo CLAP crudo) frente a la lista nueva (todos los factores, con los pesos que
         decidió E16 y el CLAP que decidió V8), con los filtros en vivo, a ciegas y en orden al azar.
           elegir  -> <root>/e12/principales.json: 10 temas principales al azar entre los que Gabriel tocó en sus sets
                      (D1), repartidos por estilo y BPM, con semilla fija; se escribe una sola vez, antes de armar ninguna
                      lista
           listas  -> <root>/e12/listas/<n>_A.m3u8 y <n>_B.m3u8 (el tema principal primero y los 30 de la lista) y
                      <root>/e12/clave.json (qué lista es cuál; Gabriel no la abre hasta contestar)
           python src/v4/evaluation/escucha_e12.py --root "D:/TRAKTOR ML datos/dj_sets" elegir
           python src/v4/evaluation/escucha_e12.py --root "D:/TRAKTOR ML datos/dj_sets" listas
         Reglas (Gabriel 2026-11-10 y agente, fijadas antes de elegir):
           Temas principales: los de D1 que están en el índice con BPM y tonalidad (como en vivo), ordenados por género
             del índice y BPM, en 10 tramos del mismo tamaño; uno al azar de cada tramo (semilla 0). Así quedan repartidos
             por estilo y BPM sin mirar ninguna lista.
           Lista de hoy: CLAP crudo con los filtros en vivo (BPM a ±20 y tonalidad compatible), sin el principal.
           Lista nueva: todos los factores de fusion_vistas (vistas en z, bpm, clave y genero en desvíos de los negativos
             de E16), con los pesos de la ecuación aprendida de E16 si E16 dice que los datos la distinguen y si no pesos
             iguales; con v8c en lugar de clap_c si V8 pasó con D3 y con E16. Mismos filtros.
           A ciegas: cuál es A y cuál B, al azar con semilla 0 por tema principal.
           Lectura: con 10 preferencias decide 9 o 10 de 10 (p 0,02 y 0,002, binomial a dos colas); 8 de 10 es un
             indicio (p 0,11). Los valores por defecto del recomendador no cambian hasta que Gabriel lo diga.
CHANGELOG:
  - 2026-11-10: Creación (E12).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from src.v4.evaluation import fusion_vistas as fv  # noqa: E402
from src.v4.evaluation import juez_d5 as jd  # noqa: E402
from src.v4.evaluation import zona_sets as zs  # noqa: E402
from tools.traktor_live import recomendador as rc  # noqa: E402

N_PRINCIPALES = 10
SEMILLA = 0
PACKAGE = Path("C:/TRAKTOR ML")


def elegir_principales(candidatos: List[int], genero: List[str], bpm: np.ndarray, n: int = N_PRINCIPALES,
                       semilla: int = SEMILLA) -> List[int]:
    """Ordena por (género, BPM, fila), parte en n tramos iguales y toma uno al azar de cada tramo."""
    orden = sorted(candidatos, key=lambda i: (genero[i] or "~", float(bpm[i]), i))
    rng = np.random.default_rng(semilla)
    return [int(t[rng.integers(len(t))]) for t in np.array_split(np.asarray(orden), n)]


def elegir(root: Path) -> Dict[str, object]:
    f = root / "e12" / "principales.json"
    if f.exists():
        raise SystemExit(f"{f} ya existe: los temas principales se eligen una sola vez")
    _, items = rc.load_index(rc.DEFAULT_INDEX)
    bib = zs.cargar_biblioteca(items)
    crudos = {k: fv.cargar_rep(v, bib.uids) for k, v in fv.REPS.items()}
    E = fv.espacio(fv.construir_vistas(crudos, bib.bpm), crudos)
    lado, _ = fv.lado_biblioteca(items, bib, E, fv.escala_z({"clap": E["clap"]}))
    d1 = zs.fuente_d1(root / "sets" / "d1.csv", bib)
    tocados = sorted({i for filas in d1.sets.values() for i in filas})
    cand = [i for i in tocados if np.isfinite(lado.bpm[i]) and lado.clave[i] >= 0]
    por_uid = {it.uid: it for it in items}
    genero = [por_uid[u].genre or "" for u in bib.uids]
    elegidos = elegir_principales(cand, genero, lado.bpm)
    out = {"creado": dt.datetime.now().isoformat(timespec="seconds"), "commit": fv.git_head(), "semilla": SEMILLA,
           "candidatos": len(cand), "tocados_en_d1": len(tocados),
           "principales": [{"uid": bib.uids[i], "artista": por_uid[bib.uids[i]].artist, "titulo": por_uid[bib.uids[i]].title,
                            "genero": genero[i], "bpm": round(float(lado.bpm[i]), 2), "clave": fv.CAMELOT[lado.clave[i]],
                            "archivo": por_uid[bib.uids[i]].rel} for i in elegidos]}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def listas(root: Path) -> Dict[str, object]:
    princ = json.loads((root / "e12" / "principales.json").read_text(encoding="utf-8"))
    e16 = json.loads((root / "sets" / "juez_d5.json").read_text(encoding="utf-8"))
    v8 = root / "e17" / "v8_clap_sin_codec.json"
    v8_pasa = False
    if v8.exists() and "v8_umbral_3_en_e16" in e16:
        u = e16["v8_umbral_3_en_e16"]
        v8_pasa = bool(json.loads(v8.read_text(encoding="utf-8"))["veredicto"]["pasa_con_d3"] and u["pasa_a"] and u["pasa_b"])
    data, items = rc.load_index(rc.DEFAULT_INDEX)
    bib = zs.cargar_biblioteca(items)
    crudos = {k: fv.cargar_rep(v, bib.uids) for k, v in fv.REPS.items()}
    vistas = fv.construir_vistas(crudos, bib.bpm)
    if v8_pasa:
        jd.vistas_v8(root, vistas)
    E = fv.espacio(vistas, crudos)
    escala = fv.escala_z(E)
    lado, tablas = fv.lado_biblioteca(items, bib, E, escala)
    if v8_pasa:
        lado.E["v8c"] = E["v8c"]
    ecu = e16["ecuacion"]
    aprendida = ecu["lista_de_la_escucha_e12"] == "ecuación aprendida"
    cols = list(jd.FACTORES_V8 if v8_pasa else fv.FACTORES)
    tabla_pesos = ecu["pesos_v8c"] if v8_pasa else ecu["pesos"]
    pesos = {c: (tabla_pesos[c]["coef_en_desvios"] if aprendida else 1.0) for c in cols}
    ref = {c: tuple(tabla_pesos[c]["media_y_desvio_de_referencia"]) for c in cols}
    pos = {u: i for i, u in enumerate(bib.uids)}
    por_uid = {it.uid: it for it in items}
    rng = np.random.default_rng(SEMILLA)
    clave, salida = [], root / "e12" / "listas"
    salida.mkdir(parents=True, exist_ok=True)
    comun = []
    for n, p in enumerate(princ["principales"], 1):
        i = pos[p["uid"]]
        fx, ok = fv.factores(lado, i, lado, tablas)
        ok = ok.copy()
        ok[i] = False
        Z = np.column_stack([np.nan_to_num((fx[c] - ref[c][0]) / ref[c][1], nan=0.0) for c in cols])
        nueva = Z @ np.asarray([pesos[c] for c in cols])
        hoy = fx["clap"]
        top = {}
        for nombre, s in (("hoy", hoy), ("nueva", nueva)):
            c = np.where(ok)[0]
            top[nombre] = [int(x) for x in c[np.argsort(-s[c], kind="stable")[:fv.TOP]]]
        comun.append(len(set(top["hoy"]) & set(top["nueva"])))
        a_es_hoy = bool(rng.integers(2))
        for letra, nombre in (("A", "hoy" if a_es_hoy else "nueva"), ("B", "nueva" if a_es_hoy else "hoy")):
            filas = [i] + top[nombre]
            texto = "#EXTM3U\n" + "".join(f"#EXTINF:-1,{por_uid[bib.uids[j]].artist} - {por_uid[bib.uids[j]].title}\n"
                                          f"{PACKAGE / por_uid[bib.uids[j]].rel}\n" for j in filas)
            (salida / f"{n:02d}_{letra}.m3u8").write_text(texto, encoding="utf-8")
        clave.append({"n": n, "uid": p["uid"], "A": "hoy" if a_es_hoy else "nueva", "B": "nueva" if a_es_hoy else "hoy",
                      "temas_en_comun": comun[-1], "candidatos": int(ok.sum())})
    out = {"creado": dt.datetime.now().isoformat(timespec="seconds"), "commit": fv.git_head(), "lista_nueva":
           {"pesos": "ecuación aprendida de E16" if aprendida else "iguales", "clap": "v8c" if v8_pasa else "clap_c",
            "pesos_por_factor": pesos}, "temas_en_comun_de_30": comun, "listas": clave}
    (root / "e12" / "clave.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return {k: v for k, v in out.items() if k != "listas"}


def main() -> int:
    ap = argparse.ArgumentParser(description="E12: la escucha a ciegas de la lista de hoy frente a la nueva")
    ap.add_argument("--root", default=str(zs.DEFAULT_ROOT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("elegir")
    sub.add_parser("listas")
    args = ap.parse_args()
    root = Path(args.root)
    res = elegir(root) if args.cmd == "elegir" else listas(root)
    print(json.dumps(res, ensure_ascii=False, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
