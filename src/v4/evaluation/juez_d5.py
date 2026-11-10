"""
PURPOSE: E16 del plan de playlists en vivo (sección 12.9): el juez «mismo set» con los sets de D5 (1001Tracklists)
         cruzados con la biblioteca, como fuente nueva junto a D3, y con sus pares la ecuación de factores ajustada de
         nuevo (P12). Usa los embeddings de la biblioteca: no depende de los previews.
           (a) las tablas de la unidad 1 (AUC de «mismo set» frente al azar de mismo BPM en la biblioteca) con D5 aparte,
               D3 aparte y las dos juntas;
           (b) la ecuación: pares de D3 y D5 que pasan los filtros en vivo, 20 negativos por par, cada DJ fuera del
               entrenamiento (un DJ de D3 y de D5 es el mismo grupo; un set de D5 que ya está en D3 no entra); AUC de
               CLAP crudo, las tres vistas, todos los factores con pesos iguales y la ecuación aprendida; los pesos con
               su intervalo y si los datos los distinguen de los pesos iguales; D1 aparte (nada se ajusta con él);
           (c) la tonalidad: en los mismos pares de un mismo set y su azar, cuántas veces más que al azar aparece cada
               relación (P7) y cada distancia entre tónicas, con la tonalidad de Traktor (la del vivo) y con la de
               Beatport (bp_key), que es la única variable que cambia (Gabriel, DECISIONS 2026-11-10, «después de E15»,
               punto 2; la regla de compatibilidad no se toca);
           (d) la parte de E16 del umbral 3 de V8 (si hay e17/direcciones.npz): v8c en lugar de clap_c en D5.
           python src/v4/evaluation/juez_d5.py [--root "D:/TRAKTOR ML datos/dj_sets"] -> <root>/sets/juez_d5.json
         Regla para la escucha E12 (Gabriel, DECISIONS 2026-11-10, punto 8; fijada antes de medir): la lista nueva usa
         la ecuación aprendida solo si su AUC fuera de muestra supera al de todos los factores con pesos iguales con un
         intervalo (remuestreo de DJs) que no toca el 0; si no, pesos iguales.
CHANGELOG:
  - 2026-11-10: Creación (E16).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from src.v4.common import harmonic  # noqa: E402
from src.v4.evaluation import codec_clap as cc  # noqa: E402
from src.v4.evaluation import fusion_vistas as fv  # noqa: E402
from src.v4.evaluation import pares_texto as pt  # noqa: E402
from src.v4.evaluation import zona_sets as zs  # noqa: E402
from tools.traktor_live import recomendador as rc  # noqa: E402

BEATPORT_PARQUET = REPO_ROOT / "artifacts" / "v4" / "datasets" / "musica" / "features" / "beatport.parquet"
BOOT_SETS = 1000
FACTORES_V8 = ("v8c",) + fv.FACTORES[1:]


def juntar_fuentes(nombre: str, fs: Sequence[zs.Fuente]) -> zs.Fuente:
    """Varias fuentes del espacio de la biblioteca en una (los ids de set no se pisan)."""
    sets, dj = {}, {}
    for f in fs:
        sets.update(f.sets)
        dj.update(f.dj)
    return zs.Fuente(nombre, "biblioteca", fs[0].V, fs[0].bpm, fs[0].artista, sets, dj, {f.nombre: f.curado for f in fs})


def vistas_v8(root: Path, vistas: Dict[str, fv.Vista]) -> Optional[int]:
    """Suma las vistas v8 y v8c (CLAP sin la dirección del códec de E17, con el k que eligió V8). None si no hay E17."""
    f = root / "e17" / "direcciones.npz"
    if not f.exists():
        return None
    v8 = root / "e17" / "v8_clap_sin_codec.json"
    k = json.loads(v8.read_text(encoding="utf-8"))["direcciones"]["k_elegido"] if v8.exists() else 1
    d = np.load(f)
    Q = cc.base(d["clap_u1"], d["clap_extra"], k)
    vistas["v8"] = fv.Vista("clap", None, Q)
    vistas["v8c"] = fv.Vista("clap", vistas["clap_c"].mu, Q)
    return k


def claves_beatport(uids: Sequence[str], parquet: Path = BEATPORT_PARQUET) -> np.ndarray:
    """Índice Camelot (fv.CAMELOT) de la tonalidad de Beatport de cada tema (niveles A y B); −1 si no hay."""
    import pandas as pd
    b = pd.read_parquet(parquet, columns=["track_uid", "level", "bp_key"])
    b = b[b.level.isin(["A", "B"]) & b.bp_key.notna()]
    k = {u: fv.indice_clave(harmonic.to_camelot(x)) for u, x in zip(b.track_uid, b.bp_key)}
    return np.asarray([k.get(u, -1) for u in uids])


def tabla_tonalidad(cat_real: Sequence[Optional[object]], W: np.ndarray, K: np.ndarray, cat_azar: Sequence[Optional[object]],
                    p: np.ndarray, rng: np.random.Generator, veces: int = BOOT_SETS) -> Dict[str, Dict[str, object]]:
    """Por categoría: fracción pesada entre los pares de un mismo set y en su azar, el cociente y su intervalo
    remuestreando sets (el azar queda fijo)."""
    cr = np.asarray([None if c is None else str(c) for c in cat_real], dtype=object)
    ca = np.asarray([None if c is None else str(c) for c in cat_azar], dtype=object)
    okr, oka = np.asarray([c is not None for c in cr]), np.asarray([c is not None for c in ca])
    cats = sorted({c for c in cr[okr]} | {c for c in ca[oka]}, key=lambda x: (len(x), x))
    fa = {c: p[oka][ca[oka] == c].sum() / p[oka].sum() for c in cats}
    sets = np.unique(K[okr])
    por = {s: np.where(okr & (K == s))[0] for s in sets}

    def fr(ii):
        w = W[ii]
        return {c: w[cr[ii] == c].sum() / w.sum() for c in cats}

    base = fr(np.where(okr)[0])
    bs = {c: [] for c in cats}
    for _ in range(veces):
        f = fr(np.concatenate([por[s] for s in rng.choice(sets, len(sets), replace=True)]))
        for c in cats:
            if fa[c] > 0:
                bs[c].append(f[c] / fa[c])
    return {c: {"mismo_set": round(float(base[c]), 4), "azar": round(float(fa[c]), 4),
                "veces": round(float(base[c] / fa[c]), 3) if fa[c] > 0 else None,
                "ic95_por_sets": [round(float(np.percentile(bs[c], 2.5)), 3), round(float(np.percentile(bs[c], 97.5)), 3)] if bs[c] else None,
                "pares": int(np.sum(cr[okr] == c))} for c in cats}


def medir(root: Path, semilla: int = 0) -> Dict[str, object]:
    t0 = time.perf_counter()
    data, items = rc.load_index(rc.DEFAULT_INDEX)
    bib = zs.cargar_biblioteca(items)
    crudos = {k: fv.cargar_rep(v, bib.uids) for k, v in fv.REPS.items()}
    vistas = fv.construir_vistas(crudos, bib.bpm)
    k_v8 = vistas_v8(root, vistas)
    E = fv.espacio(vistas, crudos)
    escala = fv.escala_z(E)
    metodos = {m: fv.METODOS[m] for m in ("clap crudo", "clap centrado", "maest centrado", "estilos", "z: las tres")}
    if k_v8 is not None:
        metodos.update({"v8 crudo": {"v8": 1.0}, "v8 centrado": {"v8c": 1.0}, "z: las tres con v8c": {"v8c": 1.0, "maest_c": 1.0, "estilos": 1.0}})
    C = {m: fv.concatenar(E, p, escala) for m, p in metodos.items()}
    d1 = zs.fuente_d1(root / "sets" / "d1.csv", bib)
    d3 = zs.fuente_d3(zs.DEFAULT_D3, bib)
    d5 = zs.fuente_d5(root, bib, d3)
    d35 = juntar_fuentes("d3_y_d5", [d3, d5])
    I5, *_ = zs.pares_del_set(d5.sets, d5.artista)
    res: Dict[str, object] = {"creado": dt.datetime.now().isoformat(timespec="seconds"), "commit": fv.git_head(), "semilla": semilla,
                              "indice": {"creado": data.get("creado"), "temas": len(bib.uids)}, "v8_k": k_v8,
                              "fuentes": {"d3": d3.curado, "d5": d5.curado, "d1": d1.curado,
                                          "pares_del_mismo_set": {"d5": int(len(I5))},
                                          "djs_d5": len({zs.estricto(v) for v in d5.dj.values()}),
                                          "djs_en_d3_y_d5": len({zs.estricto(v) for v in d5.dj.values()} & {zs.estricto(v) for v in d3.dj.values()})}}

    # (a) tablas de vistas
    tabla: Dict[str, object] = {}
    for nombre, f in (("d3", d3), ("d5", d5), ("d3_y_d5", d35)):
        nulo = fv.nulo_bpm(f, bib.bpm, semilla)
        filas = {}
        rb = ab = None
        for m in metodos:
            Cm = C[m]
            real, azar = fv.dot_pares(Cm, nulo.I, Cm, nulo.J), fv.dot_pares(Cm, nulo.a, Cm, nulo.b)
            if m == "clap crudo":
                rb, ab = real, azar
            filas[m] = {"auc": fv.r4(fv.auc_con_nulo(nulo, real, azar)),
                        "dif_con_clap_crudo_ic95_por_sets": fv.auc_dif_por_sets(nulo, real, azar, rb, ab) if m != "clap crudo" else [0.0, 0.0]}
            if m == "v8 centrado":
                rc_, ac_ = fv.dot_pares(C["clap centrado"], nulo.I, C["clap centrado"], nulo.J), fv.dot_pares(C["clap centrado"], nulo.a, C["clap centrado"], nulo.b)
                filas[m]["dif_con_clap_centrado"] = fv.r4(fv.auc_con_nulo(nulo, real, azar) - fv.auc_con_nulo(nulo, rc_, ac_))
                filas[m]["dif_con_clap_centrado_ic95_por_sets"] = fv.auc_dif_por_sets(nulo, real, azar, rc_, ac_)
        tabla[nombre] = {"pares": int(len(nulo.I)), "sets": int(len(np.unique(nulo.K))), "metodos": filas}
        print(f"  {nombre}: {len(nulo.I)} pares; " + "  ".join(f"{m} {v['auc']}" for m, v in filas.items()), flush=True)
    res["vistas"] = tabla

    # (b) la ecuación
    lado_bib, tablas = fv.lado_biblioteca(items, bib, E, escala)
    if k_v8 is not None:
        lado_bib.E["v8"], lado_bib.E["v8c"] = E["v8"], E["v8c"]
    rng = np.random.default_rng(semilla)
    partes = [fv.pares_ecuacion("d3_biblioteca", d3, lado_bib, lado_bib, tablas, rng, mismo_espacio=True),
              fv.pares_ecuacion("d5_biblioteca", d5, lado_bib, lado_bib, tablas, rng, mismo_espacio=True)]
    P = fv.Pares.juntar(partes)
    P_d1 = fv.pares_ecuacion("d1", d1, lado_bib, lado_bib, tablas, rng, mismo_espacio=True)
    print(f"pares de la ecuación: {[p.resumen for p in partes]} ({time.perf_counter() - t0:.0f} s)", flush=True)
    _, st_todo = fv.estandarizar(P.X, fv.FACTORES, P.y == 0, P.w)
    oof = {"ecuación aprendida (P12)": fv.validar_por_grupo(P, fv.FACTORES),
           "ecuación aprendida, solo vistas": fv.validar_por_grupo(P, list(fv.VISTAS_P11))}
    puntajes = {**fv.puntajes_base(P, st_todo), **oof}
    if k_v8 is not None:
        _, st_v8 = fv.estandarizar(P.X, FACTORES_V8, P.y == 0, P.w)
        puntajes["v8 crudo"] = P.X["v8"]
        puntajes["todos los factores con v8c, pesos iguales"] = fv.estandarizar(P.X, FACTORES_V8, P.y == 0, P.w, st_v8)[0].sum(axis=1)
        puntajes["ecuación aprendida con v8c"] = fv.validar_por_grupo(P, FACTORES_V8)
    fuentes = np.unique(P.fuente)
    base = puntajes["clap crudo"]
    ecu: Dict[str, object] = {"pares": P.resumen, "pares_d1": P_d1.resumen, "factores": list(fv.FACTORES), "C": fv.LR_C}
    ecu["auc_fuera_de_muestra"] = {
        n: {"todas": fv.r4(fv.auc_pares(P, s)), **{f: fv.r4(fv.auc_pares(P, s, P.fuente == f)) for f in fuentes},
            "dif_con_clap_crudo_ic95_por_djs": fv.auc_dif_por_grupos(P, s, base, semilla) if n != "clap crudo" else [0.0, 0.0]}
        for n, s in puntajes.items()}
    iguales = puntajes["todos los factores, pesos iguales"]
    apr = puntajes["ecuación aprendida (P12)"]
    dif = fv.auc_pares(P, apr) - fv.auc_pares(P, iguales)
    ic = fv.auc_dif_por_grupos(P, apr, iguales, semilla)
    ecu["aprendida_frente_a_pesos_iguales"] = {"dif_auc": fv.r4(dif), "ic95_por_djs": ic,
                                               "los_datos_la_distinguen": bool(ic and ic[0] > 0)}
    ecu["lista_de_la_escucha_e12"] = "ecuación aprendida" if ecu["aprendida_frente_a_pesos_iguales"]["los_datos_la_distinguen"] else "pesos iguales"
    Z, st_all = fv.estandarizar(P.X, fv.FACTORES, P.y == 0, P.w)
    modelo = fv.ajustar(Z, P.y, P.w)
    gs = np.unique(P.grupo)
    por = {g: np.where(P.grupo == g)[0] for g in gs}
    rng_b = np.random.default_rng(semilla)
    boot = []
    for _ in range(fv.BOOT_DJ // 2):
        ii = np.concatenate([por[g] for g in rng_b.choice(gs, len(gs), replace=True)])
        if len(np.unique(P.y[ii])) == 2:
            Zi, _ = fv.estandarizar({c: P.X[c][ii] for c in fv.FACTORES}, fv.FACTORES, P.y[ii] == 0, P.w[ii])
            boot.append(fv.ajustar(Zi, P.y[ii], P.w[ii]).coef_[0])
    boot = np.asarray(boot)
    rel = boot / boot.mean(axis=1, keepdims=True)
    ecu["pesos"] = {c: {"coef_en_desvios": round(float(modelo.coef_[0][k]), 3),
                        "ic95_por_djs": [round(float(np.percentile(boot[:, k], 2.5)), 3), round(float(np.percentile(boot[:, k], 97.5)), 3)],
                        "relativo_a_la_media_de_los_pesos": round(float(modelo.coef_[0][k] / modelo.coef_[0].mean()), 3),
                        "relativo_ic95": [round(float(np.percentile(rel[:, k], 2.5)), 3), round(float(np.percentile(rel[:, k], 97.5)), 3)],
                        "media_y_desvio_de_referencia": [round(st_all[c][0], 4), round(st_all[c][1], 4)]}
                    for k, c in enumerate(fv.FACTORES)}
    ecu["intercepto"] = round(float(modelo.intercept_[0]), 3)
    if k_v8 is not None:                         # la misma ecuación con v8c, para la lista nueva si V8 pasa
        Zv, st_v8_all = fv.estandarizar(P.X, FACTORES_V8, P.y == 0, P.w)
        mv = fv.ajustar(Zv, P.y, P.w)
        ecu["pesos_v8c"] = {c: {"coef_en_desvios": round(float(mv.coef_[0][k]), 3),
                                "media_y_desvio_de_referencia": [round(st_v8_all[c][0], 4), round(st_v8_all[c][1], 4)]}
                            for k, c in enumerate(FACTORES_V8)}
    # D1 aparte
    Zd1, _ = fv.estandarizar(P_d1.X, fv.FACTORES, P_d1.y == 0, P_d1.w, st_all)
    s_d1 = {**fv.puntajes_base(P_d1, st_todo), "ecuación aprendida (P12)": modelo.decision_function(Zd1)}
    ecu["d1_auc"] = {n: fv.r4(fv.auc_pares(P_d1, s)) for n, s in s_d1.items()}

    def puntuar(fx: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        uno = np.ones(len(fx["clap"]), dtype=bool)
        Zx, _ = fv.estandarizar(fx, fv.FACTORES, uno, np.ones(len(uno)), st_all)
        Zt, _ = fv.estandarizar(fx, fv.FACTORES, uno, np.ones(len(uno)), st_todo)
        out = {"clap crudo": fx["clap"], "todos los factores, pesos iguales": Zt.sum(axis=1),
               "ecuación aprendida (P12)": modelo.decision_function(Zx)}
        if k_v8 is not None:
            out["todos los factores con v8c, pesos iguales"] = fv.estandarizar(fx, FACTORES_V8, uno, np.ones(len(uno)), st_v8)[0].sum(axis=1)
        return out
    viv = fv.acierto_en_vivo(d1, lado_bib, tablas, puntuar)
    rec = [int(x) for x in viv.pop("_recuperacion")]
    ecu["d1_acierto_en_vivo"] = {"recuperacion_de_los_filtros": rec,
                                 **{n: {**fv.comparar(h, viv["clap crudo"], semilla), "candidatos_mediana": int(np.median(h[:, 2]))}
                                    for n, h in viv.items()}}
    res["ecuacion"] = ecu
    print(f"ecuación lista ({time.perf_counter() - t0:.0f} s)", flush=True)

    # (c) tonalidad de Traktor frente a la de Beatport, en los mismos pares
    nulo = fv.nulo_bpm(d35, bib.bpm, semilla)
    kt, kb = lado_bib.clave, claves_beatport(bib.uids)
    con_las_dos = (kt >= 0) & (kb >= 0)
    mr, ma = con_las_dos[nulo.I] & con_las_dos[nulo.J], con_las_dos[nulo.a] & con_las_dos[nulo.b]
    rng_t = np.random.default_rng(semilla)
    ton: Dict[str, object] = {"pares_mismo_set_con_las_dos": int(mr.sum()), "pares_mismo_set": int(len(nulo.I)),
                              "azar_con_las_dos": int(ma.sum()),
                              "temas_con_las_dos": int(con_las_dos.sum()),
                              "igual_en_los_temas": round(float(np.mean(kt[con_las_dos] == kb[con_las_dos])), 3)}
    for fuente, kk in (("traktor", kt), ("beatport", kb)):
        cam = [fv.CAMELOT[k] if k >= 0 else "?" for k in kk]
        def cats(A, B, m, fn):
            return [fn(cam[a], cam[b]) if ok else None for a, b, ok in zip(A, B, m)]
        ton[fuente] = {"relacion": tabla_tonalidad(cats(nulo.I, nulo.J, mr, pt.relacion_clave), nulo.W, nulo.K,
                                                   cats(nulo.a, nulo.b, ma, pt.relacion_clave), nulo.p, rng_t),
                       "distancia_entre_tonicas": tabla_tonalidad(cats(nulo.I, nulo.J, mr, pt.distancia_tonica), nulo.W, nulo.K,
                                                                  cats(nulo.a, nulo.b, ma, pt.distancia_tonica), nulo.p, rng_t)}
    res["tonalidad"] = ton

    # (d) la parte de E16 del umbral 3 de V8: D5
    if k_v8 is not None:
        a5 = tabla["d5"]["metodos"]["v8 centrado"]
        e_v8 = fv.auc_pares(P, puntajes["todos los factores con v8c, pesos iguales"], P.fuente == "d5_biblioteca")
        e_cc = fv.auc_pares(P, iguales, P.fuente == "d5_biblioteca")
        res["v8_umbral_3_en_e16"] = {"k": k_v8, "a_auc_d5_biblioteca_dif_v8c_menos_clap_c": a5["dif_con_clap_centrado"],
                                     "a_ic95": a5["dif_con_clap_centrado_ic95_por_sets"],
                                     "b_auc_d5_ecuacion_dif_v8c_menos_clap_c": fv.r4(e_v8 - e_cc),
                                     "pasa_a": a5["dif_con_clap_centrado"] >= -cc.UMBRAL_AUC, "pasa_b": (e_v8 - e_cc) >= -cc.UMBRAL_AUC}
    res["segundos"] = round(time.perf_counter() - t0, 1)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="E16: juez «mismo set» con D5 × biblioteca y la ecuación ajustada de nuevo")
    ap.add_argument("--root", default=str(zs.DEFAULT_ROOT))
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--low-priority", action="store_true")
    args = ap.parse_args()
    if args.low_priority:
        from src.v4.pipeline.extract_representations import lower_priority
        lower_priority()
    res = medir(Path(args.root), args.semilla)
    out = Path(args.root) / "sets" / "juez_d5.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: res["ecuacion"][k] for k in ("aprendida_frente_a_pesos_iguales", "lista_de_la_escucha_e12", "pesos")}, ensure_ascii=False)[:3000])
    print(f"[OK] -> {out} ({res['segundos']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
