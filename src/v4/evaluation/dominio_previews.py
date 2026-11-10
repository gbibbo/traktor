"""
PURPOSE: E15 del plan de playlists en vivo (sección 12.9): control de dominio de los previews de Beatport. En los temas
         de la biblioteca que ya tienen su preview bajado (el mismo id de Beatport), ¿el embedding del preview cae
         cerca del embedding del mismo tema en la biblioteca? Decide la ventana de V7 y si un par puede mezclar preview
         y biblioteca o solo preview con preview. Lee los MP3 de los .tar del Drive (erclm:traktor_ml/beatport_previews)
         en streaming, de a una tanda, y guarda solo embeddings; no le pide nada a Beatport.
           extraer  -> <root>/e15/emb/<bp_id>.npz: embeddings por segmento de 30 s (CLAP, MAEST capa 7, estilos) del
                       preview y, en la submuestra de control, de la regla del extractor y del recorte de la biblioteca
           medir    -> <root>/e15/e15_previews.json
           python src/v4/evaluation/dominio_previews.py --root "D:/TRAKTOR ML datos/dj_sets" extraer --low-priority
           python src/v4/evaluation/dominio_previews.py --root "D:/TRAKTOR ML datos/dj_sets" medir
         PROTOCOLO, fijado el 2026-11-10 antes de mirar los datos (el de la auditoría en el plan, sección 12.9, con
         estas decisiones operativas del agente, que no cambian ningún umbral):
           Muestra: los temas del índice con id de Beatport (niveles A y B) cuyo preview está en un manifiesto con
             estado ok. Principal: los que tienen same_cut; los demás se informan aparte.
           Submuestra de control: 40 temas con same_cut elegidos al azar con semilla 0 (ordenados por id de Beatport y
             permutados). Ahí se calculan el recorte de la biblioteca y la regla del extractor. Motivo: el cómputo
             completo pasa de 3 h de CPU (2,7 s por segmento en MAEST y 0,6 s en CLAP, medidos el 2026-11-10) y el
             umbral de pesado es una hora; con la submuestra queda cerca de una hora.
           Ventanas del preview (120 s; empieza en sample_start_ms del tema):
             V4      el preview entero en 4 segmentos de 30 s (0-30, 30-60, 60-90, 90-120); la media.
             Vtramo  la parte que cae en el 35-65 % del tema, a la resolución de esos 4 segmentos: los que tienen 15 s
                     o más dentro del tramo; si ninguno, el que más tiene.
             Vext    la regla del extractor tal cual (read_dj_segments(preview, 30, 0, 3, 0)): línea de base, solo en la
                     submuestra.
           Recorte (control, submuestra): el archivo de la biblioteca cortado en [sample_start_ms, sample_end_ms], en
             los mismos 4 segmentos. Preview frente a recorte, segmento a segmento, aísla el códec; el recorte frente
             al embedding de la biblioteca aísla el tramo.
           Medidas, por vista (clap crudo, clap_c, maest_c, estilos; con la transformación y la escala z de
             fusion_vistas.py ajustadas en la biblioteca):
             puesto del tema verdadero entre los 1760 (cuántos temas de la biblioteca quedan más parecidos al preview);
             Spearman, sobre los pares de temas de la muestra, entre el parecido preview con preview y el parecido
               biblioteca con biblioteca: por vista, para las tres vistas sumadas en z (P11) y para todos los factores
               con pesos iguales (vistas en z más bpm, clave y genero con las fórmulas del recomendador, en desvíos de
               200 000 pares al azar de la biblioteca dentro de los filtros en vivo; del lado de los previews, el BPM,
               la tonalidad y el género de Beatport de su manifiesto; del lado de la biblioteca, los de Traktor o los
               tags y el género del índice);
             desplazamiento entre dominios: norma de la diferencia de medias (vectores unitarios) frente a la
               dispersión de la biblioteca, y puesto y Spearman con cada dominio centrado por su propia media.
           Umbrales (los de la auditoría): códec, coseno mediano de 0,97 o más entre el segmento del preview y el del
             recorte, en CLAP y en MAEST capa 7 sin centrar; con la ventana elegida, el tema verdadero primero en el
             90 % o más y entre los 5 primeros en el 97 % o más, en clap_c y en maest_c (las vistas de la fusión;
             CLAP crudo se informa); Spearman de 0,90 o más en el puntaje de todos los factores. Estilos, sin umbral.
           Ventana elegida: entre V4 y Vtramo, la de mayor Spearman en el puntaje de todos los factores; si difieren
             en menos de 0,01, la de más primeros puestos sumando clap_c y maest_c; si siguen iguales, V4.
CHANGELOG:
  - 2026-11-10: Creación (E15, plan sección 12.9).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from src.v4.common import harmonic  # noqa: E402
from src.v4.evaluation import fusion_vistas as fv  # noqa: E402
from src.v4.evaluation import zona_sets as zs  # noqa: E402
from tools.dj_sets.piloto import unit  # noqa: E402
from tools.traktor_live import recomendador as rc  # noqa: E402

BEATPORT_PARQUET = REPO_ROOT / "artifacts" / "v4" / "datasets" / "musica" / "features" / "beatport.parquet"
PACKAGE = Path("C:/TRAKTOR ML")
SEG_S = 30.0
PREVIEW_S = 120.0
N_SEG = 4
N_CONTROL = 40
SEMILLA = 0
MIN_EN_TRAMO_S = 15.0
UMBRAL_CODEC = 0.97
UMBRAL_PRIMERO = 0.90
UMBRAL_TOP5 = 0.97
UMBRAL_SPEARMAN = 0.90
PARES_ESTANDAR = 200_000
VISTAS = ("clap", "clap_c", "maest_c", "estilos")
REP_DE = {"clap": "clap", "maest": "l7", "estilos": "styles"}     # crudo de fusion_vistas -> clave en el .npz


# ---------------------------------------------------------------------------
# Muestra y ventanas
# ---------------------------------------------------------------------------

@dataclass
class Tema:
    uid: str
    bp_id: str
    corrida: str
    tanda: str
    archivo: str                 # MP3 dentro del .tar
    detalles: dict
    same_cut: bool
    confirmed: bool
    archivo_biblioteca: Optional[str] = None


def muestra(root: Path, uids: Sequence[str], parquet: Path = BEATPORT_PARQUET) -> List[Tema]:
    """Los temas del índice con id de Beatport (A o B) cuyo preview está bajado (estado ok en un manifiesto)."""
    import pandas as pd
    b = pd.read_parquet(parquet)
    b = b[b.track_uid.isin(set(uids)) & b.level.isin(["A", "B"]) & b.bp_track_id.notna()]
    por_id = {str(int(float(x))): r for x, r in zip(b.bp_track_id, b.itertuples())}
    paquete = {}
    if (PACKAGE / "_organizacion.csv").exists():
        with open(PACKAGE / "_organizacion.csv", encoding="utf-8-sig", newline="") as f:
            paquete = {r["track_uid"]: r["archivo"] for r in csv.DictReader(f)}
    out: Dict[str, Tema] = {}
    for f in sorted((root / "beatport_previews").glob("*/tanda_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for t in d.get("temas", []):
            bid = str(t.get("beatport_id") or "")
            if t.get("estado") == "ok" and bid in por_id and bid not in out:
                r = por_id[bid]
                out[bid] = Tema(r.track_uid, bid, f.parent.name, f.stem, t["archivo"], t["detalles"],
                                bool(r.same_cut) if r.same_cut == r.same_cut else False,
                                bool(r.confirmed) if r.confirmed == r.confirmed else False, paquete.get(r.track_uid))
    return sorted(out.values(), key=lambda x: int(x.bp_id))


def submuestra_control(temas: Sequence[Tema], n: int = N_CONTROL, semilla: int = SEMILLA) -> List[str]:
    """Ids de Beatport de n temas con same_cut, al azar con semilla fija (orden por id y permutación)."""
    ids = sorted((t.bp_id for t in temas if t.same_cut), key=int)
    return [ids[i] for i in np.random.default_rng(semilla).permutation(len(ids))[:n]]


def segmentos_en_tramo(start_ms: float, length_ms: float, n_seg: int = N_SEG, seg_s: float = SEG_S,
                       min_s: float = MIN_EN_TRAMO_S) -> List[int]:
    """Índices de los segmentos de 30 s del preview con min_s o más dentro del 35-65 % del tema; si ninguno, el que
    más tiene."""
    lo, hi = 0.35 * length_ms / 1000.0, 0.65 * length_ms / 1000.0
    s0 = start_ms / 1000.0
    dentro = [max(0.0, min(s0 + (k + 1) * seg_s, hi) - max(s0 + k * seg_s, lo)) for k in range(n_seg)]
    elegidos = [k for k, d in enumerate(dentro) if d >= min_s]
    return elegidos or [int(np.argmax(dentro))]


def cortar(x: np.ndarray, sr: int, n_seg: int = N_SEG, seg_s: float = SEG_S) -> List[np.ndarray]:
    """n_seg segmentos consecutivos de seg_s desde el comienzo; lo que falte se completa con ceros."""
    n = int(seg_s * sr)
    out = []
    for k in range(n_seg):
        s = x[k * n:(k + 1) * n]
        out.append(np.pad(s, (0, n - len(s))).astype(np.float32) if len(s) < n else s.astype(np.float32))
    return out


def leer_mono(path: Path, inicio_s: float = 0.0, dur_s: Optional[float] = None) -> Tuple[np.ndarray, int]:
    """Audio mono float32 desde inicio_s, dur_s segundos (todo si None); soundfile y, si no abre, ffmpeg."""
    import soundfile as sf
    try:
        with sf.SoundFile(str(path)) as f:
            sr = f.samplerate
            f.seek(int(inicio_s * sr))
            x = f.read(-1 if dur_s is None else int(dur_s * sr), dtype="float32", always_2d=True).mean(axis=1)
        return x, sr
    except Exception:  # noqa: BLE001
        sr = 44100
        cmd = ["ffmpeg", "-v", "error", "-ss", str(inicio_s)] + ([] if dur_s is None else ["-t", str(dur_s)]) + \
              ["-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        return np.frombuffer(raw, dtype="<f4").copy(), sr


class Modelos:
    """CLAP y MAEST con el mismo código que la biblioteca (extract_representations)."""

    def __init__(self, hf_cache: Optional[str]):
        from src.v4.pipeline.extract_representations import ClapBackend, MaestHfBackend, resample
        self.resample = resample
        self.clap = ClapBackend(hf_cache)
        self.maest = MaestHfBackend(None)

    def embed(self, segs: Sequence[np.ndarray], sr: int) -> Dict[str, np.ndarray]:
        """(n, d) por segmento: clap (512), l7 (768), styles (400)."""
        c = [self.clap.embed(self.resample(s, sr, self.clap.sr))[""] for s in segs]
        m = [self.maest.embed(self.resample(s, sr, self.maest.sr)) for s in segs]
        return {"clap": np.stack(c), "l7": np.stack([x["_l7"] for x in m]), "styles": np.stack([x["_styles"] for x in m])}


def congelar(root: Path, temas: Sequence[Tema]) -> Tuple[List[Tema], List[str]]:
    """La muestra y la submuestra de control se fijan la primera vez (<root>/e15/muestra.json): la bajada sigue y el
    conjunto de previews crece, pero lo que se mide no cambia."""
    f = root / "e15" / "muestra.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))
        por_id = {t.bp_id: t for t in temas}
        return [por_id[i] for i in d["bp_ids"] if i in por_id], d["control"]
    control = submuestra_control(temas)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"creado": dt.datetime.now().isoformat(timespec="seconds"), "semilla": SEMILLA,
                             "bp_ids": [t.bp_id for t in temas], "control": control}, indent=1), encoding="utf-8")
    return list(temas), control


def youtube_con_preview(root: Path) -> int:
    """Temas con audio de YouTube y preview de Beatport a la vez (el control de YouTube del protocolo)."""
    prev = set()
    for f in (root / "beatport_previews").glob("*/tanda_*.json"):
        prev |= {t["clave_1001"] for t in json.loads(f.read_text(encoding="utf-8")).get("temas", []) if t.get("estado") == "ok"}
    yt = [t for f in (root / "youtube_temas").glob("*/tanda_*.json")
          for t in json.loads(f.read_text(encoding="utf-8")).get("temas", []) if t.get("estado") == "ok"]
    return sum(t.get("clave_1001") in prev for t in yt)


def miembros(cmd: Sequence[str], nombres: Sequence[str], destino: Path) -> List[str]:
    """Lee en streaming el .tar que escribe `cmd` (rclone cat <remoto>) y guarda en destino solo los archivos pedidos;
    corta la lectura cuando los tiene todos. Devuelve los que encontró. El .tar no queda en disco."""
    faltan = set(nombres)
    hechos = []
    proc = subprocess.Popen(list(cmd), stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            for m in tar:
                if m.name in faltan:
                    (destino / m.name).write_bytes(tar.extractfile(m).read())
                    faltan.discard(m.name)
                    hechos.append(m.name)
                    if not faltan:
                        break
    finally:
        proc.kill()
        proc.wait()
    return hechos


def extraer(root: Path, hf_cache: Optional[str], max_tandas: Optional[int] = None) -> None:
    from src.v4.common.audio_utils import read_dj_segments
    from src.v4.common.remote_storage import find_rclone, remote_root
    _, items = rc.load_index(rc.DEFAULT_INDEX)
    temas, control = congelar(root, muestra(root, [it.uid for it in items]))
    control = set(control)
    out_dir = root / "e15" / "emb"
    tmp = root / "e15" / "tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    rclone, base = find_rclone(), remote_root()
    pendientes = [t for t in temas if not (out_dir / f"{t.bp_id}.npz").exists()]
    print(f"{len(temas)} temas con preview ({sum(t.same_cut for t in temas)} con same_cut), control {len(control)}; "
          f"faltan {len(pendientes)}", flush=True)
    modelos = Modelos(hf_cache)
    t0, hechos = time.perf_counter(), 0
    por_tanda: Dict[Tuple[str, str], List[Tema]] = {}
    for t in pendientes:
        por_tanda.setdefault((t.corrida, t.tanda), []).append(t)
    for (corrida, tanda), grupo in sorted(por_tanda.items())[:max_tandas]:
        ta = time.perf_counter()
        bajados = set(miembros([rclone, "cat", f"{base}/beatport_previews/{corrida}/{tanda}.tar"], [t.archivo for t in grupo], tmp))
        tb = time.perf_counter() - ta
        for t in grupo:
            mp3 = tmp / t.archivo
            if t.archivo not in bajados:
                print(f"[WARN] {corrida}/{tanda}: falta {t.archivo} en el .tar", flush=True)
                continue
            try:
                x, sr = leer_mono(mp3)
                res = {f"prev_{k}": v for k, v in modelos.embed(cortar(x, sr), sr).items()}
                res["prev_segundos"] = np.float32(len(x) / sr)
                if t.bp_id in control:
                    ext, sr2 = read_dj_segments(mp3, SEG_S, 0, 3, 0)
                    res.update({f"ext_{k}": v for k, v in modelos.embed(ext, sr2).items()})
                    lib = PACKAGE / t.archivo_biblioteca if t.archivo_biblioteca else None
                    if lib is not None and lib.exists():
                        s0 = t.detalles["sample_start_ms"] / 1000.0
                        y, sr3 = leer_mono(lib, s0, PREVIEW_S)
                        res.update({f"cut_{k}": v for k, v in modelos.embed(cortar(y, sr3), sr3).items()})
                        res["cut_segundos"] = np.float32(len(y) / sr3)
                if not all(np.isfinite(v).all() for v in res.values()):
                    raise ValueError("valores no finitos")
                dest = out_dir / f"{t.bp_id}.npz"
                np.savez_compressed(dest.with_suffix(".tmp.npz"), **res)
                dest.with_suffix(".tmp.npz").replace(dest)
                hechos += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] {t.bp_id}: {type(exc).__name__}: {exc}", flush=True)
            finally:
                mp3.unlink(missing_ok=True)
        el = time.perf_counter() - t0
        print(f"  {corrida}/{tanda}: {len(grupo)} temas, lectura del .tar {tb:.0f} s | {hechos} hechos, "
              f"{el / max(hechos, 1):.1f} s por tema, {el / 60:.1f} min", flush=True)
    print(f"[OK] {hechos} temas nuevos en {out_dir}", flush=True)


# ---------------------------------------------------------------------------
# Medidas
# ---------------------------------------------------------------------------

def puesto(P: np.ndarray, L: np.ndarray, verdadero: np.ndarray) -> np.ndarray:
    """Para cada fila de P (un preview ya transformado), cuántos temas de L le quedan más parecidos que el verdadero."""
    S = P @ L.T
    v = S[np.arange(len(P)), verdadero]
    return (S > v[:, None]).sum(axis=1)


def resumen_puesto(r: np.ndarray) -> Dict[str, float]:
    return {"n": int(len(r)), "primero": round(float(np.mean(r == 0)), 4), "entre_5": round(float(np.mean(r < 5)), 4),
            "mediana": float(np.median(r)), "p90": float(np.percentile(r, 90)), "peor": int(r.max())}


def pares_de(n: int) -> Tuple[np.ndarray, np.ndarray]:
    i, j = np.triu_indices(n, 1)
    return i, j


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def estandar_factores(lado: fv.Lado, t: fv.Tablas, semilla: int = SEMILLA, n: int = PARES_ESTANDAR) -> Dict[str, Tuple[float, float]]:
    """Media y desvío de bpm, clave y genero en pares al azar de la biblioteca que pasan los filtros en vivo."""
    rng = np.random.default_rng(semilla)
    por_ancla = max(1, n // 2000)
    vals: Dict[str, list] = {"bpm": [], "clave": [], "genero": []}
    for a in rng.integers(0, len(lado.bpm), 2000):
        fx, ok = fv.factores(lado, int(a), lado, t)
        ok[a] = False
        cand = np.where(ok)[0]
        if not len(cand):
            continue
        sel = rng.choice(cand, por_ancla)
        for c in vals:
            vals[c].append(fx[c][sel])
    out = {}
    for c, v in vals.items():
        x = np.concatenate(v)
        x = x[np.isfinite(x)]
        out[c] = (float(x.mean()), float(x.std()) or 1.0)
    return out


def puntajes_pares(lado: fv.Lado, t: fv.Tablas, est: Dict[str, Tuple[float, float]]) -> Dict[str, np.ndarray]:
    """Para todos los pares (i<j) de los temas del lado: cada vista (coseno), P11 (suma de z) y todos los factores con
    pesos iguales (z de las vistas más bpm, clave y genero en desvíos de `est`; desconocido = 0)."""
    n = len(lado.bpm)
    i, j = pares_de(n)
    out: Dict[str, np.ndarray] = {}
    filas = [fv.factores(lado, a, lado, t)[0] for a in range(n)]
    for c in ("clap",) + fv.VISTAS_P11 + ("bpm", "clave", "genero"):
        if c in filas[0]:
            out[c] = np.asarray([filas[a][c][b] for a, b in zip(i, j)], dtype=np.float64)
    out["p11"] = sum(out[v] for v in fv.VISTAS_P11)
    z = [np.nan_to_num((out[c] - est[c][0]) / est[c][1], nan=0.0) for c in ("bpm", "clave", "genero")]
    out["todos_los_factores"] = out["p11"] + sum(z)
    return out


def lado_previews(E: Dict[str, np.ndarray], temas: Sequence[Tema], t: fv.Tablas) -> fv.Lado:
    gid = {g: k for k, g in enumerate(t.generos)}
    bpm = np.asarray([float(x.detalles.get("bpm") or np.nan) for x in temas], dtype=np.float64)
    clave = np.asarray([fv.indice_clave(harmonic.to_camelot(x.detalles.get("key") or "")) for x in temas])
    genero = np.asarray([gid.get((x.detalles.get("genre"), None), -1) for x in temas])
    return fv.Lado(E, bpm, clave, genero, ["" for _ in temas])


def medir(root: Path) -> Dict[str, object]:
    t0 = time.perf_counter()
    data, items = rc.load_index(rc.DEFAULT_INDEX)
    bib = zs.cargar_biblioteca(items)
    crudos = {k: fv.cargar_rep(v, bib.uids) for k, v in fv.REPS.items()}
    vistas = fv.construir_vistas(crudos, bib.bpm)
    E = fv.espacio(vistas, crudos)
    escala = fv.escala_z(E)
    temas_todos, control = congelar(root, muestra(root, bib.uids))
    control = set(control)
    emb = {t.bp_id: np.load(root / "e15" / "emb" / f"{t.bp_id}.npz") for t in temas_todos if (root / "e15" / "emb" / f"{t.bp_id}.npz").exists()}
    temas = [t for t in temas_todos if t.bp_id in emb]
    pos = {u: i for i, u in enumerate(bib.uids)}
    generos_prev = sorted({(t.detalles.get("genre"), None) for t in temas if t.detalles.get("genre")}, key=str)
    lado_bib, tablas = fv.lado_biblioteca(items, bib, E, escala, generos_prev)
    est = estandar_factores(lado_bib, tablas)

    def ventana(t: Tema, pref: str, cual: str) -> Dict[str, np.ndarray]:
        z = emb[t.bp_id]
        if cual == "V4":
            k = list(range(N_SEG))
        elif cual == "Vtramo":
            k = segmentos_en_tramo(t.detalles["sample_start_ms"], t.detalles["track_length_ms"])
        else:
            k = list(range(len(z[f"{pref}_clap"])))
        return {r: z[f"{pref}_{REP_DE[r]}"][k].mean(axis=0) for r in REP_DE}

    def transformar(crudo: Dict[str, np.ndarray], mu_propia: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        out = {}
        for v in VISTAS:
            vs = vistas[v]
            if mu_propia is not None and vs.mu is not None:
                vs = fv.Vista(vs.rep, mu_propia[vs.rep], vs.Q)
            out[v] = vs(crudo[vs.rep])
        return out

    def evaluar(sub: Sequence[Tema], pref: str, cual: str) -> Dict[str, object]:
        crudo = {r: np.stack([ventana(t, pref, cual)[r] for t in sub]) for r in REP_DE}
        idx = np.asarray([pos[t.uid] for t in sub])
        Pv = transformar(crudo)
        mu_prev = {r: unit(crudo[r]).mean(axis=0) for r in REP_DE}
        Pa = transformar(crudo, mu_prev)
        res: Dict[str, object] = {"temas": len(sub)}
        res["puesto"] = {v: resumen_puesto(puesto(Pv[v], E[v], idx)) for v in VISTAS}
        res["puesto_centrado_aparte"] = {v: resumen_puesto(puesto(Pa[v], E[v], idx)) for v in ("clap_c", "maest_c")}
        lp = puntajes_pares(lado_previews({v: Pv[v] for v in ("clap",) + fv.VISTAS_P11}, sub, tablas), tablas, est)
        lpa = puntajes_pares(lado_previews({v: Pa[v] for v in ("clap",) + fv.VISTAS_P11}, sub, tablas), tablas, est)
        sub_lado = fv.Lado({v: lado_bib.E[v][idx] for v in lado_bib.E}, lado_bib.bpm[idx], lado_bib.clave[idx], lado_bib.genero[idx],
                           [lado_bib.artista[i] for i in idx])
        lb = puntajes_pares(sub_lado, tablas, est)
        res["spearman"] = {c: round(spearman(lp[c], lb[c]), 4) for c in ("clap",) + fv.VISTAS_P11 + ("p11", "todos_los_factores")}
        # diagnóstico: cuánto del puntaje de todos los factores es el audio y cuánto los metadatos de Beatport
        ok = {c: np.isfinite(lp[c]) & np.isfinite(lb[c]) for c in ("bpm", "clave", "genero")}
        res["spearman_metadatos"] = {c: round(spearman(lp[c][ok[c]], lb[c][ok[c]]), 4) if ok[c].sum() > 2 else None for c in ok}
        lmix = puntajes_pares(fv.Lado({v: Pv[v] for v in ("clap",) + fv.VISTAS_P11}, sub_lado.bpm, sub_lado.clave, sub_lado.genero,
                                      sub_lado.artista), tablas, est)
        res["spearman"]["todos_los_factores_con_metadatos_de_la_biblioteca"] = round(spearman(lmix["todos_los_factores"], lb["todos_los_factores"]), 4)
        lp_bpm = np.asarray([float(t.detalles.get("bpm") or np.nan) for t in sub])
        lp_key = np.asarray([fv.indice_clave(harmonic.to_camelot(t.detalles.get("key") or "")) for t in sub])
        gid = {g: k for k, g in enumerate(tablas.generos)}
        lp_gen = np.asarray([gid.get((t.detalles.get("genre"), None), -1) for t in sub])
        bpm_ok = np.isfinite(lp_bpm) & np.isfinite(sub_lado.bpm)
        key_ok = (lp_key >= 0) & (sub_lado.clave >= 0)
        res["metadatos_beatport_frente_a_biblioteca"] = {
            "bpm_a_1": round(float(np.mean(np.abs(lp_bpm[bpm_ok] - sub_lado.bpm[bpm_ok]) <= 1.0)), 3), "bpm_con_los_dos": int(bpm_ok.sum()),
            "bpm_doble_o_mitad": round(float(np.mean(np.abs(np.abs(np.log2(lp_bpm[bpm_ok] / sub_lado.bpm[bpm_ok])) - 1) < 0.02)), 3),
            "tonalidad_igual": round(float(np.mean(lp_key[key_ok] == sub_lado.clave[key_ok])), 3), "tonalidad_con_las_dos": int(key_ok.sum()),
            "genero_igual": round(float(np.mean([tablas.generos[a][0] == tablas.generos[b][0] for a, b in zip(lp_gen, sub_lado.genero) if a >= 0 and b >= 0])), 3),
            "genero_con_los_dos": int(np.sum((lp_gen >= 0) & (sub_lado.genero >= 0)))}
        res["spearman_centrado_aparte"] = {c: round(spearman(lpa[c], lb[c]), 4) for c in fv.VISTAS_P11 + ("p11", "todos_los_factores")}
        res["coseno_con_el_mismo_tema"] = {v: round(float(np.median(np.sum(Pv[v] * E[v][idx], axis=1))), 4) for v in VISTAS}
        desp = {}
        for r in REP_DE:
            A, B = unit(crudo[r]), unit(crudos[r][idx])
            d = A.mean(axis=0) - B.mean(axis=0)
            disp = float(np.sqrt(np.mean(np.sum((B - B.mean(axis=0)) ** 2, axis=1))))
            desp[r] = {"norma_diferencia_de_medias": round(float(np.linalg.norm(d)), 4), "dispersion_biblioteca": round(disp, 4),
                       "cociente": round(float(np.linalg.norm(d)) / disp, 3)}
        res["desplazamiento"] = desp
        return res

    principal = [t for t in temas if t.same_cut]
    otros = [t for t in temas if not t.same_cut]
    ctrl = [t for t in principal if t.bp_id in control and "ext_clap" in emb[t.bp_id]]
    ctrl_cut = [t for t in ctrl if "cut_clap" in emb[t.bp_id]]
    res: Dict[str, object] = {
        "creado": dt.datetime.now().isoformat(timespec="seconds"), "commit": fv.git_head(),
        "indice": {"creado": data.get("creado"), "temas": len(bib.uids)},
        "muestra": {"con_preview": len(temas_todos), "con_embedding": len(temas), "same_cut": len(principal), "otros": len(otros),
                    "confirmed": sum(t.confirmed for t in temas), "control_elegidos": len(control), "control_con_regla_del_extractor": len(ctrl),
                    "control_con_recorte": len(ctrl_cut),
                    "segundos_del_preview": {"mediana": float(np.median([float(emb[t.bp_id]["prev_segundos"]) for t in temas])) if temas else None},
                    "corridas": sorted({t.corrida for t in temas}), "tandas": len({(t.corrida, t.tanda) for t in temas})},
        "youtube_con_preview": youtube_con_preview(root),
        "estandar_factores": {c: [round(m, 4), round(s, 4)] for c, (m, s) in est.items()},
        "umbrales": {"codec": UMBRAL_CODEC, "primero": UMBRAL_PRIMERO, "entre_5": UMBRAL_TOP5, "spearman": UMBRAL_SPEARMAN},
    }
    res["principal"] = {w: evaluar(principal, "prev", w) for w in ("V4", "Vtramo")}
    res["otros"] = {w: evaluar(otros, "prev", w) for w in ("V4", "Vtramo")} if len(otros) >= 3 else None
    if len(ctrl) >= 3:
        res["control"] = {"preview_V4": evaluar(ctrl, "prev", "V4"), "preview_Vtramo": evaluar(ctrl, "prev", "Vtramo"),
                          "preview_Vext": evaluar(ctrl, "ext", "Vext")}
    if len(ctrl_cut) >= 3:
        res["control"]["recorte_V4"] = evaluar(ctrl_cut, "cut", "V4")
        res["control"]["recorte_Vtramo"] = evaluar(ctrl_cut, "cut", "Vtramo")
        codec = {}
        for r in REP_DE:
            c = np.concatenate([np.sum(unit(emb[t.bp_id][f"prev_{REP_DE[r]}"]) * unit(emb[t.bp_id][f"cut_{REP_DE[r]}"]), axis=1) for t in ctrl_cut])
            codec[r] = {"mediana": round(float(np.median(c)), 4), "p10": round(float(np.percentile(c, 10)), 4), "segmentos": int(len(c))}
        res["control"]["codec_por_segmento"] = codec
    # ventana elegida y veredicto con los umbrales fijados de antemano
    sp = {w: res["principal"][w]["spearman"]["todos_los_factores"] for w in ("V4", "Vtramo")}
    pr = {w: sum(res["principal"][w]["puesto"][v]["primero"] for v in ("clap_c", "maest_c")) for w in ("V4", "Vtramo")}
    if abs(sp["V4"] - sp["Vtramo"]) >= 0.01:
        elegida = max(sp, key=sp.get)
    elif abs(pr["V4"] - pr["Vtramo"]) > 1e-9:
        elegida = max(pr, key=pr.get)
    else:
        elegida = "V4"
    w = res["principal"][elegida]
    veredicto = {
        "ventana_elegida": elegida,
        "codec": {r: res.get("control", {}).get("codec_por_segmento", {}).get(r, {}).get("mediana") for r in ("clap", "maest")},
        "pasa_codec": all((res.get("control", {}).get("codec_por_segmento", {}).get(r, {}).get("mediana") or 0) >= UMBRAL_CODEC for r in ("clap", "maest")),
        "pasa_puesto": all(w["puesto"][v]["primero"] >= UMBRAL_PRIMERO and w["puesto"][v]["entre_5"] >= UMBRAL_TOP5 for v in ("clap_c", "maest_c")),
        "pasa_spearman": w["spearman"]["todos_los_factores"] >= UMBRAL_SPEARMAN,
    }
    if veredicto["pasa_puesto"] and veredicto["pasa_spearman"]:
        veredicto["que_sigue"] = "V7 con la ventana elegida; los pares pueden mezclar preview y biblioteca"
    elif veredicto["pasa_spearman"]:
        veredicto["que_sigue"] = "los previews se comparan entre ellos, centrados aparte"
    else:
        veredicto["que_sigue"] = "los previews no sirven de juez de los modelos: decirle a Gabriel antes de gastar GPU en V7"
    res["veredicto"] = veredicto
    res["segundos"] = round(time.perf_counter() - t0, 1)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="E15: control de dominio de los previews de Beatport")
    ap.add_argument("--root", default=str(zs.DEFAULT_ROOT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extraer")
    e.add_argument("--hf-cache", default=None)
    e.add_argument("--low-priority", action="store_true")
    e.add_argument("--max-tandas", type=int, default=None, help="Solo las primeras N tandas (prueba chica)")
    sub.add_parser("medir")
    args = ap.parse_args()
    root = Path(args.root)
    if args.cmd == "extraer":
        if args.low_priority:
            from src.v4.pipeline.extract_representations import lower_priority
            lower_priority()
        extraer(root, args.hf_cache, args.max_tandas)
        return 0
    res = medir(root)
    out = root / "e15" / "e15_previews.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res["veredicto"], ensure_ascii=False))
    print(f"[OK] -> {out} ({res['segundos']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
