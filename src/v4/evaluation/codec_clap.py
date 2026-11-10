"""
PURPOSE: E17 y V8 del plan de playlists en vivo (sección 12.9, «Después de E15»). E17: ¿el corrimiento de CLAP entre
         un preview de Beatport y el archivo de la biblioteca es de la compresión o del nivel? Se recodifica el mismo
         audio de 20 archivos de la biblioteca de 300 kbps o más y se mide cuánto se mueve cada vista y si se mueve
         sobre la dirección preview − biblioteca de E15. V8: CLAP sin esa dirección (estimada en E17, sin previews),
         con umbrales fijados antes de medir.
           recodificar -> <root>/e17/emb/<track_uid>.npz: CLAP, MAEST capa 7 y estilos de cada segmento del extractor
                          en 5 variantes (original, MP3 a 192, 128 y 96 kbps, y 6 dB menos sin recodificar)
           e17         -> <root>/e17/e17_codec.json y <root>/e17/direcciones.npz
           v8          -> <root>/e17/v8_clap_sin_codec.json
           python src/v4/evaluation/codec_clap.py --root "D:/TRAKTOR ML datos/dj_sets" recodificar --low-priority --max 2
           python src/v4/evaluation/codec_clap.py --root "D:/TRAKTOR ML datos/dj_sets" e17
           python src/v4/evaluation/codec_clap.py --root "D:/TRAKTOR ML datos/dj_sets" v8
         PROTOCOLO, fijado el 2026-11-10 antes de recodificar ni medir nada (umbrales de V8: los del pedido de Gabriel y
         la fila V8 del plan; las decisiones operativas son del agente y están marcadas así):
         E17
           Muestra: temas del índice con bitrate de 300 kbps o más (tamaño × 8 / duración, de catalog.parquet, como la
             auditoría), sin los 166 temas de E15 (para que la dirección no salga de los temas con los que V8 se
             evalúa). Ordenados por track_uid, permutados con semilla 0; los 20 primeros cuyo archivo se abre.
           Audio: los 3 segmentos de 30 s del extractor de la biblioteca (35-65 % del tema; dj_segment_starts), en
             estéreo. Variantes, cada una del mismo audio decodificado:
               original     el segmento tal cual (su mezcla a mono es la que vio el extractor);
               mp3_192/128/96  el segmento estéreo codificado con LAME (ffmpeg libmp3lame, CBR, la misma frecuencia de
                            muestreo) y decodificado; se alinea con el original por correlación (el retardo del códec);
               menos_6db    el original por 10^(−6/20), sin codificar.
           Embeddings: los del extractor (ClapBackend y MaestHfBackend); el del tema es la media de los 3 segmentos.
             Control: el original tiene que coincidir con el embedding guardado de la biblioteca (coseno ≥ 0,999).
           Medidas por vista cruda (CLAP, MAEST capa 7, estilos; vectores unitarios):
             coseno de cada variante con el original (por segmento y por tema);
             dirección de E15: d = media de unit(preview, ventana V4) − unit(biblioteca) en los 146 temas con
               same_cut; û = d/|d|; σ = desvío de unit(biblioteca)·û en los 1760 temas;
             corrimiento de cada variante sobre û: s = media de (unit(variante) − unit(original))·û / σ; el de
               referencia es el del códec del preview, s_ref = lo mismo con (preview − recorte) en la submuestra de
               control de E15 (el mismo tramo de audio);
             coseno entre la diferencia media de cada variante y d, y fracción de la diferencia que cae sobre û.
           Regla (decide CLAP; MAEST y estilos se informan):
             «compresión» si s(96) ≥ 0,5·s_ref y |s(−6 dB)| ≤ 0,25·s(96);
             «nivel» si s(−6 dB) ≥ 0,5·s_ref y |s(96)| ≤ 0,25·s(−6 dB);
             «ninguna de las dos» si s(96) y s(−6 dB) quedan bajo 0,25·s_ref;
             «las dos» en cualquier otro caso.
           Dirección para V8 (sin previews): u1 = media de unit(mp3_96) − unit(original) en los 20 temas, normalizada.
             Direcciones de reserva (agente): las 4 primeras componentes principales de las 60 diferencias de las tres
             tasas, sin la parte sobre u1.
         V8
           Vistas (agente): v8 = unit(unit(x) − (unit(x)·u)u) (reemplaza a CLAP crudo, la lista de hoy); v8c = lo mismo
             después de restar la media de la biblioteca (reemplaza a clap_c en la fusión y en la lista nueva de E12).
             Las dos con la clase Vista de fusion_vistas. Con k direcciones, u es la base de u1 y las k − 1 primeras de
             reserva. Decide v8c, que es la vista que iría a la lista nueva; v8 se informa con las mismas medidas.
           Umbrales:
             (1) puesto: en los 146 previews de E15 con same_cut (ventana V4), transformados con la media de la
                 biblioteca (sin centrar aparte), el tema verdadero queda primero en el 90 % o más y entre los 5
                 primeros en el 97 % o más. Muestra del umbral (agente): la de los umbrales de E15; el pedido dice
                 «los 166», y los 20 sin same_cut pueden ser otra versión del tema: los 166 y los 20 se informan.
             (2) vecinos: anclas = temas de la biblioteca de menos de 200 kbps; exceso = fracción de sus 10 vecinos
                 de menos de 200 kbps − la esperada por el género de cada vecino (la tasa de su género sin él mismo;
                 sin género es un género). El exceso de v8c no supera al de MAEST centrado en las mismas anclas.
             (3) juez de la unidad 1, mismos pares: (a) AUC de D3 × biblioteca con el nulo de mismo BPM, v8c frente a
                 clap_c; (b) AUC en los pares de D3 de la ecuación (los de la unidad 1, semilla 0) de «todos los
                 factores con pesos iguales» con v8c en lugar de clap_c. Ninguno pierde más de 0,02. La misma prueba
                 con los pares de E16 se corre en la unidad de E16; el veredicto de V8 queda provisorio hasta entonces.
                 D1 se informa aparte (AUC y acierto en 30 con los filtros en vivo).
             (4) se informa, sin umbral: cuántos de los 30 temas de la lista de hoy (CLAP crudo con los filtros en
                 vivo) cambian por tema principal con v8, y cuántos de la lista de todos los factores con v8c.
           Más direcciones: si (1) no pasa con una, se prueban 2 a 5 y se informan todas; decide la menor que pasa
             (1); (2), (3) y (4) se miden con ella (o con una si ninguna pasa).
           MAEST (centrado) y estilos con su propia dirección de E17 (una): las mismas medidas, solo informadas.
CHANGELOG:
  - 2026-11-10: Creación (E17 y V8, auditoría de E15).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from src.v4.evaluation import dominio_previews as dp  # noqa: E402
from src.v4.evaluation import fusion_vistas as fv  # noqa: E402
from src.v4.evaluation import zona_sets as zs  # noqa: E402
from tools.dj_sets.piloto import unit  # noqa: E402
from tools.traktor_live import recomendador as rc  # noqa: E402

CATALOG = REPO_ROOT / "artifacts" / "v4" / "datasets" / "musica" / "catalog.parquet"
SEMILLA = 0
N_E17 = 20
KBPS_ALTO = 300.0
KBPS_BAJO = 200.0
TASAS = (192, 128, 96)
ATENUACION_DB = 6.0
VARIANTES = ("original",) + tuple(f"mp3_{t}" for t in TASAS) + ("menos_6db",)
SEG = (30.0, 0, 3, 0)                    # segmentation de config/v4.yaml: 30 s, 3 en el medio
MAX_RETARDO = 4096                       # muestras de retardo del códec que busca la alineación
VECINOS = 10
UMBRAL_PRIMERO = 0.90
UMBRAL_TOP5 = 0.97
UMBRAL_AUC = 0.02
MAX_DIRECCIONES = 5
REPS_CRUDAS = {"clap": "clap", "maest": "l7", "estilos": "styles"}   # nombre en fusion_vistas -> clave del .npz


# ---------------------------------------------------------------------------
# Muestra y audio
# ---------------------------------------------------------------------------

def bitrates(uids: Sequence[str], catalog: Path = CATALOG) -> np.ndarray:
    """kbps de cada tema (tamaño × 8 / duración, como la auditoría), alineado con uids; nan si falta."""
    import pandas as pd
    c = pd.read_parquet(catalog, columns=["track_uid", "filesize_bytes", "duration_s"])
    k = dict(zip(c.track_uid, c.filesize_bytes * 8 / c.duration_s / 1000.0))
    return np.asarray([k.get(u, np.nan) for u in uids], dtype=np.float64)


def rutas(uids: Sequence[str], catalog: Path = CATALOG) -> Dict[str, Path]:
    import pandas as pd
    c = pd.read_parquet(catalog, columns=["track_uid", "source_path"])
    return {u: Path(p) for u, p in zip(c.track_uid, c.source_path) if u in set(uids)}


def muestra_e17(uids: Sequence[str], kbps: np.ndarray, excluir: set, existe, n: int = N_E17,
                semilla: int = SEMILLA) -> List[str]:
    """n temas de KBPS_ALTO o más fuera de `excluir`, al azar con semilla fija (orden por uid y permutación); se salta
    los que existe(uid) da falso."""
    cand = sorted(u for u, k in zip(uids, kbps) if np.isfinite(k) and k >= KBPS_ALTO and u not in excluir)
    out = []
    for i in np.random.default_rng(semilla).permutation(len(cand)):
        if existe(cand[i]):
            out.append(cand[i])
            if len(out) == n:
                break
    return out


def segmentos_estereo(path: Path, seg: Tuple[float, int, int, int] = SEG) -> Tuple[List[np.ndarray], int]:
    """Los segmentos del extractor (read_dj_segments) sin mezclar a mono: (n, 2) float32 cada uno."""
    import soundfile as sf
    from src.v4.common.audio_utils import dj_segment_starts
    try:
        with sf.SoundFile(str(path)) as f:
            sr, total = f.samplerate, f.frames
            n = int(seg[0] * sr)
            out = []
            for s in dj_segment_starts(total, n, seg[1], seg[2], seg[3]):
                f.seek(s)
                x = f.read(n, dtype="float32", always_2d=True)
                if x.shape[1] == 1:
                    x = np.repeat(x, 2, axis=1)
                out.append(np.pad(x[:, :2], ((0, n - len(x)), (0, 0))) if len(x) < n else x[:, :2])
        return out, sr
    except Exception:  # noqa: BLE001  (los que libsndfile no abre van por ffmpeg, como en el extractor)
        import mutagen
        info = mutagen.File(str(path)).info
        sr = int(getattr(info, "sample_rate", 44100) or 44100)
        total = int(float(info.length) * sr)
        n = int(seg[0] * sr)
        out = []
        for s in dj_segment_starts(total, n, seg[1], seg[2], seg[3]):
            raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{s / sr:.6f}", "-i", str(path), "-t", f"{seg[0]}",
                                  "-ac", "2", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True, check=True).stdout
            x = np.frombuffer(raw, dtype="<f4").reshape(-1, 2)[:n]
            out.append(np.pad(x, ((0, n - len(x)), (0, 0))).astype(np.float32))
        return out, sr


def retardo(ref: np.ndarray, y: np.ndarray, sr: int, max_lag: int = MAX_RETARDO) -> int:
    """Muestras que `y` va atrasada respecto de `ref` (mono), por correlación sobre 2 s desde el segundo 5."""
    a = int(5 * sr)
    L = int(2 * sr)
    r = ref[a:a + L]
    w = y[max(0, a - max_lag):a + L + max_lag]
    if len(r) < L or np.abs(r).sum() == 0:
        return 0
    c = np.correlate(w, r, mode="valid")
    return int(np.argmax(c)) - (a - max(0, a - max_lag))


def mp3_ida_y_vuelta(x: np.ndarray, sr: int, kbps: int) -> np.ndarray:
    """El segmento estéreo (n, 2) codificado a MP3 CBR con LAME y decodificado, alineado y del mismo largo."""
    enc = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "2", "-i", "-",
                          "-c:a", "libmp3lame", "-b:a", f"{kbps}k", "-f", "mp3", "-"],
                         input=np.ascontiguousarray(x, dtype="<f4").tobytes(), capture_output=True, check=True).stdout
    dec = subprocess.run(["ffmpeg", "-v", "error", "-f", "mp3", "-i", "-", "-ac", "2", "-ar", str(sr), "-f", "f32le", "-"],
                         input=enc, capture_output=True, check=True).stdout
    y = np.frombuffer(dec, dtype="<f4").reshape(-1, 2)
    lag = retardo(x.mean(axis=1), y.mean(axis=1), sr)
    y = y[lag:] if lag >= 0 else np.pad(y, ((-lag, 0), (0, 0)))
    n = len(x)
    return (np.pad(y, ((0, n - len(y)), (0, 0))) if len(y) < n else y[:n]).astype(np.float32)


def variantes(segs: Sequence[np.ndarray], sr: int) -> Dict[str, List[np.ndarray]]:
    """Mono de cada variante, segmento a segmento."""
    out = {"original": [s.mean(axis=1) for s in segs]}
    for t in TASAS:
        out[f"mp3_{t}"] = [mp3_ida_y_vuelta(s, sr, t).mean(axis=1) for s in segs]
    g = np.float32(10 ** (-ATENUACION_DB / 20))
    out["menos_6db"] = [s.mean(axis=1) * g for s in segs]
    return out


def recodificar(root: Path, hf_cache: Optional[str], maximo: Optional[int] = None) -> None:
    _, items = rc.load_index(rc.DEFAULT_INDEX)
    uids = [it.uid for it in items if it.vec is not None]
    e15 = {t.uid for t in dp.congelar(root, dp.muestra(root, uids))[0]}
    paths = rutas(uids)
    elegidos = muestra_e17(uids, bitrates(uids), e15, lambda u: u in paths and paths[u].exists())
    out_dir = root / "e17" / "emb"
    out_dir.mkdir(parents=True, exist_ok=True)
    f = root / "e17" / "muestra.json"
    if not f.exists():
        f.write_text(json.dumps({"creado": dt.datetime.now().isoformat(timespec="seconds"), "semilla": SEMILLA,
                                 "track_uids": elegidos, "excluidos_de_e15": len(e15)}, indent=1), encoding="utf-8")
    elegidos = json.loads(f.read_text(encoding="utf-8"))["track_uids"]
    pend = [u for u in elegidos if not (out_dir / f"{u}.npz").exists()][:maximo]
    print(f"{len(elegidos)} temas elegidos; faltan {len(pend)}", flush=True)
    modelos = dp.Modelos(hf_cache)
    t0 = time.perf_counter()
    for n, u in enumerate(pend, 1):
        try:
            segs, sr = segmentos_estereo(paths[u])
            res = {"sr": np.int32(sr)}
            for nombre, mono in variantes(segs, sr).items():
                for k, v in modelos.embed(mono, sr).items():
                    res[f"{nombre}_{k}"] = v
            if not all(np.isfinite(v).all() for v in res.values()):
                raise ValueError("valores no finitos")
            dest = out_dir / f"{u}.npz"
            np.savez_compressed(dest.with_suffix(".tmp.npz"), **res)
            dest.with_suffix(".tmp.npz").replace(dest)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] {u[:12]} {paths[u].name}: {type(exc).__name__}: {exc}", flush=True)
        el = time.perf_counter() - t0
        print(f"  {n}/{len(pend)} {paths[u].name[:60]} | {el / n:.0f} s por tema, {el / 60:.1f} min", flush=True)


# ---------------------------------------------------------------------------
# E17: medidas
# ---------------------------------------------------------------------------

def corrimiento(D: np.ndarray, u: np.ndarray, sigma: float) -> float:
    """Media de la proyección de las diferencias sobre la dirección unitaria u, en desvíos de la biblioteca."""
    return float(np.mean(D @ u) / sigma)


def veredicto_e17(s: Dict[str, float], s_ref: float) -> str:
    s96, s6 = s["mp3_96"], s["menos_6db"]
    if s96 >= 0.5 * s_ref and abs(s6) <= 0.25 * s96:
        return "compresión"
    if s6 >= 0.5 * s_ref and abs(s96) <= 0.25 * s6:
        return "nivel"
    if s96 < 0.25 * s_ref and s6 < 0.25 * s_ref:
        return "ninguna de las dos"
    return "las dos"


def direcciones_reserva(D_todas: np.ndarray, u1: np.ndarray, k: int = MAX_DIRECCIONES - 1) -> np.ndarray:
    """(k, d): las primeras componentes principales de las diferencias sin su parte sobre u1, ortogonales a u1."""
    R = D_todas - np.outer(D_todas @ u1, u1)
    _, _, Vt = np.linalg.svd(R, full_matrices=False)          # sin centrar: lo sistemático que queda, no solo la varianza
    return Vt[:k]


def base(u1: np.ndarray, extra: np.ndarray, k: int) -> np.ndarray:
    """Base ortonormal (d, k) con u1 y las k − 1 primeras de reserva."""
    Q, _ = np.linalg.qr(np.column_stack([u1] + [extra[i] for i in range(k - 1)]))
    return Q


def datos_e15(root: Path, uids: Sequence[str]) -> Tuple[list, Dict[str, np.ndarray], list]:
    """Temas de E15 con embedding, sus npz y los de la submuestra de control con recorte."""
    temas, control = dp.congelar(root, dp.muestra(root, uids))
    emb = {t.bp_id: np.load(root / "e15" / "emb" / f"{t.bp_id}.npz") for t in temas
           if (root / "e15" / "emb" / f"{t.bp_id}.npz").exists()}
    temas = [t for t in temas if t.bp_id in emb]
    ctrl = [t for t in temas if t.bp_id in set(control) and "cut_clap" in emb[t.bp_id]]
    return temas, emb, ctrl


def medir_e17(root: Path) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    t0 = time.perf_counter()
    data, items = rc.load_index(rc.DEFAULT_INDEX)
    bib = zs.cargar_biblioteca(items)
    crudos = {k: unit(fv.cargar_rep(v, bib.uids)) for k, v in fv.REPS.items()}
    pos = {u: i for i, u in enumerate(bib.uids)}
    elegidos = json.loads((root / "e17" / "muestra.json").read_text(encoding="utf-8"))["track_uids"]
    emb = {u: np.load(root / "e17" / "emb" / f"{u}.npz") for u in elegidos if (root / "e17" / "emb" / f"{u}.npz").exists()}
    us = [u for u in elegidos if u in emb]
    temas, e15, ctrl = datos_e15(root, bib.uids)
    principal = [t for t in temas if t.same_cut]
    kb = bitrates(bib.uids)
    en_e15 = np.zeros(len(bib.uids), dtype=bool)
    en_e15[[pos[t.uid] for t in temas]] = True
    res: Dict[str, object] = {"creado": dt.datetime.now().isoformat(timespec="seconds"), "commit": fv.git_head(),
                              "indice": {"creado": data.get("creado"), "temas": len(bib.uids)},
                              "muestra": {"elegidos": len(elegidos), "con_embedding": len(us),
                                          "kbps": [round(float(kb[pos[u]]), 1) for u in us],
                                          "sr": [int(emb[u]["sr"]) for u in us]},
                              "e15": {"temas": len(temas), "same_cut": len(principal), "control_con_recorte": len(ctrl)}}
    direcciones: Dict[str, np.ndarray] = {}
    por_rep: Dict[str, object] = {}
    for rep, clave in REPS_CRUDAS.items():
        L = crudos[rep]
        tema = {v: unit(np.stack([emb[u][f"{v}_{clave}"].mean(axis=0) for u in us])) for v in VARIANTES}
        orig_lib = L[[pos[u] for u in us]]
        r: Dict[str, object] = {"control_original_frente_a_la_biblioteca": {
            "min": round(float(np.min(np.sum(tema["original"] * orig_lib, axis=1))), 5),
            "mediana": round(float(np.median(np.sum(tema["original"] * orig_lib, axis=1))), 5)}}
        # dirección de E15 y su escala
        P = unit(np.stack([e15[t.bp_id][f"prev_{clave}"].mean(axis=0) for t in principal]))
        B = L[[pos[t.uid] for t in principal]]
        d = (P - B).mean(axis=0)
        uh = d / np.linalg.norm(d)
        proy = L @ uh
        sigma = float(proy.std())
        Pc = unit(np.stack([e15[t.bp_id][f"prev_{clave}"].mean(axis=0) for t in ctrl]))
        Cc = unit(np.stack([e15[t.bp_id][f"cut_{clave}"].mean(axis=0) for t in ctrl]))
        s_ref = corrimiento(Pc - Cc, uh, sigma)
        r["direccion_e15"] = {"temas": len(principal), "norma": round(float(np.linalg.norm(d)), 4), "desvio_biblioteca": round(sigma, 5),
                              "corrimiento_preview_frente_a_biblioteca": round(corrimiento(P - B, uh, sigma), 3),
                              "corrimiento_codec_del_preview_s_ref": round(s_ref, 3), "control_temas": len(ctrl)}
        # la auditoría: proyección de la biblioteca por bitrate, fuera de E15 (verificación)
        ref = proy[(kb >= KBPS_ALTO) & ~en_e15].mean()
        r["biblioteca_por_bitrate_fuera_de_e15"] = {
            nombre: {"temas": int(m.sum()), "corrimiento": round(float((proy[m].mean() - ref) / sigma), 3)}
            for nombre, m in (("300_o_mas", (kb >= KBPS_ALTO) & ~en_e15),
                              ("200_a_300", (kb >= KBPS_BAJO) & (kb < KBPS_ALTO) & ~en_e15),
                              ("menos_de_200", (kb < KBPS_BAJO) & ~en_e15))}
        r["previews_corrimiento_frente_a_300_o_mas"] = round(float(((P @ uh).mean() - ref) / sigma), 3)
        # variantes
        var: Dict[str, object] = {}
        s: Dict[str, float] = {}
        D_todas = []
        for v in VARIANTES[1:]:
            seg = np.concatenate([np.sum(unit(emb[u][f"{v}_{clave}"]) * unit(emb[u][f"original_{clave}"]), axis=1) for u in us])
            D = tema[v] - tema["original"]
            if v.startswith("mp3_"):
                D_todas.append(D)
            dm = D.mean(axis=0)
            s[v] = corrimiento(D, uh, sigma)
            nrm = np.linalg.norm(D, axis=1)
            var[v] = {"coseno_segmento_mediana": round(float(np.median(seg)), 4), "coseno_segmento_p10": round(float(np.percentile(seg, 10)), 4),
                      "coseno_tema_mediana": round(float(np.median(np.sum(tema[v] * tema["original"], axis=1))), 4),
                      "corrimiento_sobre_e15": round(s[v], 3),
                      "coseno_diferencia_media_con_e15": round(float(dm @ d / (np.linalg.norm(dm) * np.linalg.norm(d) + 1e-12)), 3),
                      "fraccion_de_la_diferencia_sobre_e15": round(float(np.mean((D @ uh) ** 2 / np.maximum(nrm ** 2, 1e-18))), 3),
                      "norma_diferencia_media_frente_a_e15": round(float(np.linalg.norm(dm) / np.linalg.norm(d)), 3)}
        r["variantes"] = var
        r["veredicto"] = veredicto_e17(s, s_ref)
        u1 = (tema["mp3_96"] - tema["original"]).mean(axis=0)
        u1 = u1 / np.linalg.norm(u1)
        extra = direcciones_reserva(np.vstack(D_todas), u1)
        direcciones[f"{rep}_u1"] = u1
        direcciones[f"{rep}_extra"] = extra
        r["direccion_v8"] = {"coseno_u1_con_e15": round(float(u1 @ uh), 3),
                             "corrimiento_previews_sobre_u1": round(float(np.mean((P - B) @ u1) / float((L @ u1).std())), 3)}
        por_rep[rep] = r
        print(f"  {rep:8s} s_ref {s_ref:+.2f} | " + "  ".join(f"{v} {s[v]:+.2f}" for v in s) + f" -> {r['veredicto']}", flush=True)
    res["por_vista"] = por_rep
    res["veredicto"] = por_rep["clap"]["veredicto"]
    res["segundos"] = round(time.perf_counter() - t0, 1)
    return res, direcciones


# ---------------------------------------------------------------------------
# V8: medidas
# ---------------------------------------------------------------------------

def exceso_vecinos(V: np.ndarray, bajo: np.ndarray, genero: np.ndarray, k: int = VECINOS,
                   anclas: Optional[np.ndarray] = None) -> Dict[str, float]:
    """En las anclas (por defecto, los temas bajos): fracción de sus k vecinos bajos y la esperada por el género de
    cada vecino (tasa de su género sin él mismo; −1 = sin género, que cuenta como un género)."""
    S = V @ V.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argpartition(-S, k, axis=1)[:, :k]
    g = np.where(genero < 0, genero.max() + 1, genero)
    n_g = np.bincount(g)
    b_g = np.bincount(g, weights=bajo.astype(float))
    p = (b_g[g] - bajo) / np.maximum(n_g[g] - 1, 1)
    a = np.where(bajo)[0] if anclas is None else anclas
    obs = bajo[nn[a]].mean(axis=1)
    esp = p[nn[a]].mean(axis=1)
    return {"anclas": int(len(a)), "observado": round(float(obs.mean()), 4), "esperado_por_genero": round(float(esp.mean()), 4),
            "exceso": round(float(obs.mean() - esp.mean()), 4)}


def cambios_top(s_a: np.ndarray, s_b: np.ndarray, ok: np.ndarray, top: int = fv.TOP) -> int:
    """Cuántos de los `top` primeros de s_a (entre los ok) no están entre los de s_b."""
    c = np.where(ok)[0]
    if not len(c):
        return 0
    t = min(top, len(c))
    a = set(c[np.argsort(-s_a[c], kind="stable")[:t]])
    b = set(c[np.argsort(-s_b[c], kind="stable")[:t]])
    return t - len(a & b)


def resumen(x: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    return {"n": int(len(x)), "media": round(float(x.mean()), 2), "mediana": float(np.median(x)),
            "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)), "cero": round(float(np.mean(x == 0)), 3)}


def medir_v8(root: Path, semilla: int = SEMILLA) -> Dict[str, object]:
    t0 = time.perf_counter()
    data, items = rc.load_index(rc.DEFAULT_INDEX)
    bib = zs.cargar_biblioteca(items)
    crudos = {k: fv.cargar_rep(v, bib.uids) for k, v in fv.REPS.items()}
    vistas = fv.construir_vistas(crudos, bib.bpm)
    dirs = np.load(root / "e17" / "direcciones.npz")
    mu = {k: vistas[k].mu for k in ("clap_c", "maest_c")}
    for k in range(1, MAX_DIRECCIONES + 1):
        Q = base(dirs["clap_u1"], dirs["clap_extra"], k)
        vistas[f"v8_{k}"] = fv.Vista("clap", None, Q)
        vistas[f"v8c_{k}"] = fv.Vista("clap", mu["clap_c"], Q)
    vistas["maest_v8c"] = fv.Vista("maest", mu["maest_c"], base(dirs["maest_u1"], dirs["maest_extra"], 1))
    vistas["estilos_v8"] = fv.Vista("estilos", None, base(dirs["estilos_u1"], dirs["estilos_extra"], 1))
    E = fv.espacio(vistas, crudos)
    escala = fv.escala_z(E)
    pos = {u: i for i, u in enumerate(bib.uids)}
    res: Dict[str, object] = {"creado": dt.datetime.now().isoformat(timespec="seconds"), "commit": fv.git_head(), "semilla": semilla,
                              "indice": {"creado": data.get("creado"), "temas": len(bib.uids)}}
    print(f"vistas listas ({time.perf_counter() - t0:.0f} s)", flush=True)

    # (1) puesto de los previews de E15, ventana V4, con la media de la biblioteca
    temas, e15, _ = datos_e15(root, bib.uids)
    grupos = {"same_cut_146": [t for t in temas if t.same_cut], "todos_166": temas, "sin_same_cut_20": [t for t in temas if not t.same_cut]}
    nombres_v = ["clap", "clap_c"] + [f"v8_{k}" for k in range(1, MAX_DIRECCIONES + 1)] + \
                [f"v8c_{k}" for k in range(1, MAX_DIRECCIONES + 1)] + ["maest_c", "maest_v8c", "estilos", "estilos_v8"]
    puesto: Dict[str, object] = {}
    for g, sub in grupos.items():
        idx = np.asarray([pos[t.uid] for t in sub])
        crudo = {r: np.stack([e15[t.bp_id][f"prev_{c}"].mean(axis=0) for t in sub]) for r, c in REPS_CRUDAS.items()}
        puesto[g] = {v: dp.resumen_puesto(dp.puesto(vistas[v](crudo[vistas[v].rep]), E[v], idx)) for v in nombres_v}
    res["puesto"] = puesto
    pasa1 = {k: puesto["same_cut_146"][f"v8c_{k}"]["primero"] >= UMBRAL_PRIMERO and puesto["same_cut_146"][f"v8c_{k}"]["entre_5"] >= UMBRAL_TOP5
             for k in range(1, MAX_DIRECCIONES + 1)}
    k_dec = next((k for k in range(1, MAX_DIRECCIONES + 1) if pasa1[k]), 1)
    res["direcciones"] = {"pasa_puesto_con_k": pasa1, "k_elegido": k_dec}
    print("puesto: " + "  ".join(f"k{k} {puesto['same_cut_146'][f'v8c_{k}']['primero']:.3f}" for k in pasa1) + f" -> k = {k_dec}", flush=True)

    # (2) vecinos
    kb = bitrates(bib.uids)
    bajo = kb < KBPS_BAJO
    gen_nombres = sorted({it.genre for it in items if it.vec is not None and it.genre})
    gid = {g: i for i, g in enumerate(gen_nombres)}
    por_uid = {it.uid: it for it in items}
    genero = np.asarray([gid.get(por_uid[u].genre, -1) for u in bib.uids])
    alto = np.where(kb >= KBPS_ALTO)[0]
    res["vecinos"] = {"temas_bajo_200_kbps": int(bajo.sum()),
                      "anclas_bajo_200": {v: exceso_vecinos(E[v], bajo, genero) for v in nombres_v},
                      "anclas_300_o_mas": {v: exceso_vecinos(E[v], bajo, genero, anclas=alto) for v in ("clap", "clap_c", f"v8_{k_dec}", f"v8c_{k_dec}", "maest_c")}}
    print(f"vecinos listos ({time.perf_counter() - t0:.0f} s)", flush=True)

    # (3) juez de la unidad 1: D3 × biblioteca con el nulo de mismo BPM, y los pares de la ecuación (misma semilla y orden)
    v8, v8c = f"v8_{k_dec}", f"v8c_{k_dec}"
    tempo = zs.cargar_tempo(root / "sets" / "tempo_temas.csv")
    d1 = zs.fuente_d1(root / "sets" / "d1.csv", bib)
    d3 = zs.fuente_d3(zs.DEFAULT_D3, bib)
    bed_todo, dup = zs.fuente_bedrock(root, tempo, bib)
    l7, st, hay = fv.maest_de_bedrock(root, bed_todo)
    bed = fv.subfuente(bed_todo, hay)
    juez: Dict[str, object] = {}
    nulo_d3 = fv.nulo_bpm(d3, bib.bpm, semilla)
    nulo_d1 = fv.nulo_bpm(d1, bib.bpm, semilla)
    comparables = {"clap": ["clap"] + [f"v8_{k}" for k in range(1, MAX_DIRECCIONES + 1)],
                   "clap_c": ["clap_c"] + [f"v8c_{k}" for k in range(1, MAX_DIRECCIONES + 1)],
                   "maest_c": ["maest_c", "maest_v8c"], "estilos": ["estilos", "estilos_v8"]}
    auc_vistas: Dict[str, object] = {}
    for b, vs in comparables.items():
        rb, ab = fv.dot_pares(E[b], nulo_d3.I, E[b], nulo_d3.J), fv.dot_pares(E[b], nulo_d3.a, E[b], nulo_d3.b)
        for v in vs:
            rv, av = fv.dot_pares(E[v], nulo_d3.I, E[v], nulo_d3.J), fv.dot_pares(E[v], nulo_d3.a, E[v], nulo_d3.b)
            auc_vistas[v] = {"auc_d3_biblioteca": fv.r4(fv.auc_con_nulo(nulo_d3, rv, av)),
                             "dif_con": b, "dif_ic95_por_sets": fv.auc_dif_por_sets(nulo_d3, rv, av, rb, ab) if v != b else [0.0, 0.0],
                             "auc_d1": fv.r4(fv.auc_con_nulo(nulo_d1, fv.dot_pares(E[v], nulo_d1.I, E[v], nulo_d1.J),
                                                             fv.dot_pares(E[v], nulo_d1.a, E[v], nulo_d1.b)))}
    juez["auc_d3_biblioteca_nulo_mismo_bpm"] = auc_vistas

    extra = (v8, v8c, "maest_v8c", "estilos_v8")
    lado_bib, tablas = fv.lado_biblioteca(items, bib, E, escala)
    for v in extra:
        lado_bib.E[v] = E[v]
    rng = np.random.default_rng(semilla)
    partes = [fv.pares_ecuacion("d3_biblioteca", d3, lado_bib, lado_bib, tablas, rng, mismo_espacio=True)]
    if hay.any():
        E_bed = fv.espacio(vistas, {"clap": bed.V, "maest": l7[hay], "estilos": st[hay]})
        lado_bed = fv.Lado({v: E_bed[v] for v in ("clap",) + fv.VISTAS_P11 + extra}, bed.bpm, np.full(len(bed.V), -1),
                           np.full(len(bed.V), -1), bed.artista)
        partes.append(fv.pares_ecuacion("bedrock", bed, lado_bed, lado_bib, tablas, rng, mismo_espacio=False, excluir_pool=dup))
    P = fv.Pares.juntar(partes)
    P_d1 = fv.pares_ecuacion("d1", d1, lado_bib, lado_bib, tablas, rng, mismo_espacio=True)
    variantes_cols = {"todos, clap_c (unidad 1)": fv.FACTORES,
                      f"todos, {v8c}": (v8c,) + fv.FACTORES[1:],
                      "todos, maest_v8c": ("clap_c", "maest_v8c") + fv.FACTORES[2:],
                      "todos, estilos_v8": ("clap_c", "maest_c", "estilos_v8") + fv.FACTORES[3:],
                      f"todos, {v8c}, maest_v8c y estilos_v8": (v8c, "maest_v8c", "estilos_v8") + fv.FACTORES[3:]}
    stats = {n: fv.estandarizar(P.X, cols, P.y == 0, P.w)[1] for n, cols in variantes_cols.items()}
    puntos = {"clap crudo": P.X["clap"], f"{v8} crudo": P.X[v8]}
    puntos.update({n: fv.estandarizar(P.X, cols, P.y == 0, P.w, stats[n])[0].sum(axis=1) for n, cols in variantes_cols.items()})
    m3 = P.fuente == "d3_biblioteca"
    idx3 = np.where(m3)[0]
    P3 = fv.Pares({}, P.y[idx3], P.w[idx3], P.grupo[idx3], P.fuente[idx3], P.consenso[idx3], {})
    base_de = {f"{v8} crudo": "clap crudo"}
    base_de.update({n: "todos, clap_c (unidad 1)" for n in variantes_cols if n != "todos, clap_c (unidad 1)"})
    ecu = {}
    for n, s in puntos.items():
        a = fv.auc_pares(P, s, m3)
        fila = {"d3": fv.r4(a), "bedrock": fv.r4(fv.auc_pares(P, s, P.fuente == "bedrock")), "todas": fv.r4(fv.auc_pares(P, s))}
        if n in base_de:
            b = puntos[base_de[n]]
            fila["dif_d3_con"] = base_de[n]
            fila["dif_d3"] = fv.r4(a - fv.auc_pares(P, b, m3))
            fila["dif_d3_ic95_por_djs"] = fv.auc_dif_por_grupos(P3, s[idx3], b[idx3], semilla)
        ecu[n] = fila
    juez["pares"] = P.resumen
    juez["ecuacion_auc"] = ecu
    # D1 aparte
    s_d1 = {"clap crudo": P_d1.X["clap"], f"{v8} crudo": P_d1.X[v8]}
    s_d1.update({n: fv.estandarizar(P_d1.X, cols, P_d1.y == 0, P_d1.w, stats[n])[0].sum(axis=1) for n, cols in variantes_cols.items()})
    juez["d1_auc_pares"] = {n: fv.r4(fv.auc_pares(P_d1, s)) for n, s in s_d1.items()}

    def puntuar(fx: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        uno = np.ones(len(fx["clap"]), dtype=bool)
        out = {"clap crudo": fx["clap"], f"{v8} crudo": fx[v8]}
        for n in ("todos, clap_c (unidad 1)", f"todos, {v8c}"):
            out[n] = fv.estandarizar(fx, variantes_cols[n], uno, np.ones(len(uno)), stats[n])[0].sum(axis=1)
        return out
    viv = fv.acierto_en_vivo(d1, lado_bib, tablas, puntuar)
    rec = [int(x) for x in viv.pop("_recuperacion")]
    juez["d1_acierto_en_vivo"] = {"recuperacion_de_los_filtros": rec,
                                  **{n: {**fv.comparar(h, viv["clap crudo" if "crudo" in n else "todos, clap_c (unidad 1)"], semilla),
                                         "candidatos_mediana": int(np.median(h[:, 2]))} for n, h in viv.items()}}
    res["juez"] = juez
    print(f"juez listo ({time.perf_counter() - t0:.0f} s)", flush=True)

    # (4) cuántos de los 30 cambian por tema principal
    d1_filas = sorted({i for filas in d1.sets.values() for i in filas})
    cambios = {"lista de hoy: clap crudo -> " + v8: [], f"todos los factores: clap_c -> {v8c}": []}
    uno_n = np.ones(len(bib.uids))
    for i in range(len(bib.uids)):
        fx, ok = fv.factores(lado_bib, i, lado_bib, tablas)
        ok = ok.copy()
        ok[i] = False
        z0 = fv.estandarizar(fx, variantes_cols["todos, clap_c (unidad 1)"], ok, uno_n, stats["todos, clap_c (unidad 1)"])[0].sum(axis=1)
        z1 = fv.estandarizar(fx, variantes_cols[f"todos, {v8c}"], ok, uno_n, stats[f"todos, {v8c}"])[0].sum(axis=1)
        cambios["lista de hoy: clap crudo -> " + v8].append(cambios_top(fx["clap"], fx[v8], ok))
        cambios[f"todos los factores: clap_c -> {v8c}"].append(cambios_top(z0, z1, ok))
    res["cambios_en_los_30"] = {n: {"biblioteca": resumen(c), "temas_de_d1": resumen(np.asarray(c)[d1_filas])} for n, c in cambios.items()}

    # veredicto con los umbrales fijados
    a3 = auc_vistas[v8c]
    e3 = ecu[f"todos, {v8c}"]
    ver = {"k": k_dec, "vista": v8c,
           "pasa_1_puesto": pasa1[k_dec],
           "pasa_2_vecinos": res["vecinos"]["anclas_bajo_200"][v8c]["exceso"] <= res["vecinos"]["anclas_bajo_200"]["maest_c"]["exceso"],
           "pasa_3a_auc_d3_biblioteca": a3["auc_d3_biblioteca"] >= auc_vistas["clap_c"]["auc_d3_biblioteca"] - UMBRAL_AUC,
           "pasa_3b_auc_d3_ecuacion": e3["dif_d3"] >= -UMBRAL_AUC,
           "e16": "pendiente: se mide en la unidad de E16"}
    ver["pasa_con_d3"] = all(ver[c] for c in ("pasa_1_puesto", "pasa_2_vecinos", "pasa_3a_auc_d3_biblioteca", "pasa_3b_auc_d3_ecuacion"))
    res["veredicto"] = ver
    res["segundos"] = round(time.perf_counter() - t0, 1)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="E17 (el códec en CLAP) y V8 (CLAP sin la dirección del códec)")
    ap.add_argument("--root", default=str(zs.DEFAULT_ROOT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("recodificar")
    r.add_argument("--hf-cache", default=None)
    r.add_argument("--low-priority", action="store_true")
    r.add_argument("--max", type=int, default=None, help="Solo los primeros N temas que faltan (prueba chica)")
    sub.add_parser("e17")
    sub.add_parser("v8")
    args = ap.parse_args()
    root = Path(args.root)
    if args.cmd == "recodificar":
        if args.low_priority:
            from src.v4.pipeline.extract_representations import lower_priority
            lower_priority()
        recodificar(root, args.hf_cache, args.max)
        return 0
    if args.cmd == "e17":
        res, dirs = medir_e17(root)
        np.savez(root / "e17" / "direcciones.npz", **dirs)
        out = root / "e17" / "e17_codec.json"
    else:
        res = medir_v8(root)
        out = root / "e17" / "v8_clap_sin_codec.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res.get("veredicto"), ensure_ascii=False))
    print(f"[OK] -> {out} ({res['segundos']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
