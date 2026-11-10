"""
PURPOSE: MAEST (capa 7 y los 400 estilos Discogs) de los temas originales de una carpeta de youtube_audio.py, con
         el mismo modelo y la misma segmentación que la biblioteca (extract_representations.MaestHfBackend:
         mtg-upf/discogs-maest-30s-pw-129e a 16 kHz; 3 tramos de 30 s del 35-65 %, read_dj_segments; media de los
         tramos). Es la vista que les faltaba a los originales de Bedrock para la fusión de vistas de Parecidos
         (plan de playlists en vivo, sección 12, P11; src/v4/evaluation/fusion_vistas.py). Por archivo deja
         <root>/maest/<carpeta>/<nombre>.npz con:
           l7       (768,)  media de los tokens de la capa 7, media de los tramos (como maesthf_full_l7);
           styles   (400,)  sigmoide de los 400 estilos, media de los tramos (como maesthf_full_styles).
         Solo archivos de hasta 20 min (los temas; las mezclas enteras quedan afuera, como el «dj» de
         clap_blocks.py). Lo ya calculado no se repite. --low-priority para las corridas largas.
         --verificar N: recalcula N temas de la biblioteca (paquete C:/TRAKTOR ML) y compara con
         maesthf_full_l7 y maesthf_full_styles (mismo espacio), con el tiempo por tema.
CHANGELOG:
  - 2026-11-09: Creación (unidad 1 de PROMPT_FUSION_Y_CONCEPTOS.md: MAEST de los 179 originales de Bedrock).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from src.v4.common.audio_utils import read_dj_segments  # noqa: E402
from src.v4.pipeline.extract_representations import MaestHfBackend, lower_priority, resample  # noqa: E402

DEFAULT_ROOT = REPO_ROOT / "artifacts" / "v4" / "dj_sets"
LIB_REPS = REPO_ROOT / "artifacts" / "v4" / "datasets" / "musica" / "representations"
PACKAGE = Path("C:/TRAKTOR ML")
SEGMENT_S, N_MID = 30.0, 3            # config/v4.yaml, segmentation (la de la biblioteca)
MAX_S = 20 * 60


def embed(backend: MaestHfBackend, path: Path) -> Dict[str, np.ndarray]:
    """{"l7", "styles"} del tema como en la biblioteca: media de los 3 tramos de 30 s del 35-65 %."""
    segs, sr = read_dj_segments(path, SEGMENT_S, 0, N_MID, 0)
    out = [backend.embed(resample(s, sr, backend.sr)) for s in segs]
    return {"l7": np.mean([o["_l7"] for o in out], axis=0).astype(np.float32),
            "styles": np.mean([o["_styles"] for o in out], axis=0).astype(np.float32)}


def duration_s(path: Path) -> Optional[float]:
    try:
        import mutagen
        info = mutagen.File(str(path))
        return float(info.info.length) if info is not None else None
    except Exception:  # noqa: BLE001
        return None


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def verify(backend: MaestHfBackend, n: int) -> None:
    """Recalcula n temas de la biblioteca y compara con lo guardado (coseno), con el tiempo por tema."""
    reps = {k: (np.load(LIB_REPS / f"maesthf_full_{s}" / "embeddings.npy"),
                {u: i for i, u in enumerate(json.loads((LIB_REPS / f"maesthf_full_{s}" / "track_uids.json").read_text(encoding="utf-8")))})
            for k, s in (("l7", "l7"), ("styles", "styles"))}
    with open(PACKAGE / "_organizacion.csv", encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["track_uid"] in reps["l7"][1]]
    rows = rows[:: max(1, len(rows) // max(1, n))][:n]
    for r in rows:
        t = time.perf_counter()
        v = embed(backend, PACKAGE / r["archivo"])
        c = {k: cos(v[k], E[pos[r["track_uid"]]]) for k, (E, pos) in reps.items()}
        print(f"  l7 {c['l7']:.4f}  estilos {c['styles']:.4f}  {time.perf_counter() - t:.1f} s  {r['archivo']}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="MAEST (capa 7 y estilos) de los originales de una carpeta de audio de sets")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--carpeta", action="append", help="Carpeta de <root>/audio (id de playlist); repetible")
    ap.add_argument("--limit", type=int, default=None, help="Solo los primeros N archivos de cada carpeta (prueba)")
    ap.add_argument("--hf-cache", default=None)
    ap.add_argument("--verificar", type=int, default=0, help="Recalcular N temas de la biblioteca contra maesthf_full_*")
    ap.add_argument("--low-priority", action="store_true")
    args = ap.parse_args()
    if args.low_priority:
        lower_priority()
    backend = MaestHfBackend(args.hf_cache)
    if args.verificar:
        verify(backend, args.verificar)
    root = Path(args.root)
    for carpeta in args.carpeta or []:
        folder = root / "audio" / carpeta
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        out_dir = root / "maest" / carpeta
        out_dir.mkdir(parents=True, exist_ok=True)
        files = [e["archivo"] for e in manifest["entradas"] if e.get("archivo")][: args.limit]
        hechos, largos, fallos, t0 = 0, 0, 0, time.perf_counter()
        for name in files:
            out = out_dir / (Path(name).stem + ".npz")
            if out.exists():
                continue
            dur = duration_s(folder / name)
            if dur is None or dur > MAX_S:
                largos += 1
                continue
            try:
                res = embed(backend, folder / name)
            except Exception as exc:  # noqa: BLE001
                fallos += 1
                print(f"[WARN] {carpeta}/{name}: {type(exc).__name__}: {exc}", flush=True)
                continue
            if not all(np.isfinite(v).all() for v in res.values()):
                fallos += 1
                print(f"[WARN] {carpeta}/{name}: valores no finitos", flush=True)
                continue
            tmp = out.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **res)
            tmp.replace(out)
            hechos += 1
            if hechos % 10 == 0:
                print(f"  {carpeta}: {hechos} temas, {(time.perf_counter() - t0) / hechos:.1f} s/tema", flush=True)
        print(f"[OK] {carpeta}: {hechos} nuevos, {largos} sin duración o de más de 20 min, {fallos} fallos "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
