"""
PURPOSE: Analiza cada tema una sola vez y guarda en caché su espectro por semitono en bloques de 0,5 s
         (src/v4/harmonic_tiv/frontend.py), con y sin HPSS y en tres variantes (crudo, blanqueado,
         NNLS). Todo lo demás (secciones, ponderaciones, perfiles de croma, métricas de compatibilidad)
         se calcula después sobre el caché, sin volver a tocar el audio.
         Entrada: un manifiesto CSV con columnas id,path (lo arma build_eval_sets.py) o --dataset-name
         (lee catalog.parquet: track_uid, source_path). Salida: <out>/<id>.npz + meta.json.
         Uso:
           python tools/harmonic_tiv/extract_semitones.py --manifest <csv> --out <dir> [--folder X]
               [--limit N] [--workers 8] [--low-priority]
CHANGELOG:
  - 2026-11-06: Creación (rama harmonic-tiv).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.v4.harmonic_tiv import frontend as fe  # noqa: E402

CACHE_VERSION = "semitones_v2"  # v2: piso del blanqueo (v1 descartado)


def _low_priority() -> None:
    """Prioridad baja (los procesos hijos la heredan). En Windows de 64 bits, GetCurrentProcess devuelve
    el pseudo-handle -1: sin declarar restype/argtypes, ctypes lo trunca a 32 bits y la llamada falla en
    silencio (pasó el 2026-11-06), por eso se declaran y se verifica el resultado."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            k32.SetPriorityClass.restype = wintypes.BOOL
            if not k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000):   # BELOW_NORMAL
                print(f"aviso: no se pudo bajar la prioridad (error {ctypes.get_last_error()})", file=sys.stderr)
        else:
            os.nice(10)
    except Exception as exc:  # noqa: BLE001
        print(f"aviso: no se pudo bajar la prioridad: {exc}", file=sys.stderr)


def _worker_init(low_priority: bool) -> None:
    import torch
    torch.set_num_threads(1)
    if low_priority:
        _low_priority()


def analyze_file(track_id: str, path: str, out_dir: str, cfg_dict: dict) -> dict:
    out_path = Path(out_dir) / f"{track_id}.npz"
    if out_path.exists():
        return {"id": track_id, "status": "cached"}
    t0 = time.time()
    try:
        cfg = fe.FrontendConfig(**cfg_dict)
        x = fe.load_mono(Path(path), cfg.sr)
        if len(x) < cfg.sr * 5:
            return {"id": track_id, "status": "too_short", "seconds": len(x) / cfg.sr}
        matrix = fe.logfreq_matrix(cfg.sr, cfg.n_fft)
        dictionary = fe.nnls_dictionary(cfg.nnls_shape, cfg.nnls_harmonics)
        arrays = {"duration_s": np.float32(len(x) / cfg.sr)}
        for pre in ("orig", "hpss"):
            sig = x if pre == "orig" else fe.hpss_harmonic(x, cfg)
            res = fe.analyze_signal(sig, cfg, matrix=matrix, dictionary=dictionary)
            for k, v in res.items():
                arrays[f"{pre}__{k}"] = np.asarray(v)
        tmp = out_path.with_name(out_path.stem + ".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, out_path)
        bad = [k for k, v in arrays.items() if np.issubdtype(np.asarray(v).dtype, np.floating)
               and not np.all(np.isfinite(np.asarray(v, dtype=np.float64)))]
        return {"id": track_id, "status": "ok", "seconds": round(time.time() - t0, 1),
                "duration_s": float(arrays["duration_s"]), "nonfinite": bad}
    except Exception as exc:  # noqa: BLE001
        return {"id": track_id, "status": "error", "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(limit=3)}


def read_manifest(path: Path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [(r["id"], r["path"]) for r in csv.DictReader(f)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, help="CSV con columnas id,path")
    ap.add_argument("--dataset-name", help="alternativa: catalog.parquet del dataset (track_uid, source_path)")
    ap.add_argument("--out", type=Path, required=True, help="carpeta del caché")
    ap.add_argument("--folder", help="solo rutas que contengan este texto (prueba chica)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--low-priority", action="store_true")
    args = ap.parse_args()

    if args.manifest:
        items = read_manifest(args.manifest)
    elif args.dataset_name:
        import pandas as pd
        from src.v4.common.config_loader import load_config
        from src.v4.common.path_resolver import resolve_artifacts_root
        cat = pd.read_parquet(resolve_artifacts_root(load_config()) / args.dataset_name / "catalog.parquet")
        items = list(zip(cat["track_uid"], cat["source_path"]))
    else:
        ap.error("falta --manifest o --dataset-name")
    if args.folder:
        items = [(i, p) for i, p in items if args.folder.lower() in str(p).lower()]
    if args.limit:
        items = items[: args.limit]
    if args.low_priority:
        _low_priority()      # el proceso principal también; los trabajadores lo heredan
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = fe.FrontendConfig()
    meta = {"cache_version": CACHE_VERSION, "frontend": cfg.as_dict(),
            "variants": ["raw_lin", "white_lin", "white_nnls"], "preprocs": ["orig", "hpss"]}
    meta_path = args.out / "meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old != meta:
            print(f"ERROR: {meta_path} tiene otra configuración; usar otra carpeta", file=sys.stderr)
            return 2
    else:
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"{len(items)} temas -> {args.out} con {args.workers} procesos", flush=True)
    t0 = time.time()
    counts: dict = {}
    log_path = args.out / "extract_log.jsonl"
    done: set = set()
    pending = list(items)
    rounds_without_progress = 0
    # Si un proceso muere (memoria, decodificador), el pool se rompe: se arma otro con lo que falta.
    # Tres rondas seguidas sin avanzar = un archivo que tumba el proceso siempre: se informa y se corta.
    with open(log_path, "a", encoding="utf-8") as log:
        while pending and rounds_without_progress < 3:
            before = len(done)
            try:
                with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init,
                                         initargs=(args.low_priority,), max_tasks_per_child=25) as ex:
                    futs = {ex.submit(analyze_file, i, p, str(args.out), cfg.as_dict()): i for i, p in pending}
                    for fut in as_completed(futs):
                        r = fut.result()
                        done.add(futs[fut])
                        counts[r["status"]] = counts.get(r["status"], 0) + 1
                        if r["status"] != "cached":
                            log.write(json.dumps(r, ensure_ascii=False) + "\n")
                            log.flush()
                        if r["status"] == "error":
                            print(f"  ERROR {r['id'][:12]}: {r['error']}", flush=True)
                        n = len(done)
                        if n % 25 == 0 or n == len(items):
                            print(f"  {n}/{len(items)} {counts} {(time.time() - t0) / 60:.1f} min", flush=True)
            except BrokenProcessPool as exc:
                print(f"  pool roto ({exc}); se reintenta lo pendiente", flush=True)
            pending = [(i, p) for i, p in pending if i not in done]
            rounds_without_progress = 0 if len(done) > before else rounds_without_progress + 1
    if pending:
        counts["crashed"] = len(pending)
        with open(args.out / "crashed.txt", "w", encoding="utf-8") as f:
            f.writelines(f"{i}\t{p}\n" for i, p in pending)
        print(f"  {len(pending)} temas tumban el proceso: ver crashed.txt", flush=True)
    print("listo:", counts)
    return 0 if not (counts.get("error", 0) or counts.get("crashed", 0)) else 1


if __name__ == "__main__":
    sys.exit(main())
