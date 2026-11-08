"""
PURPOSE: CLAP del comienzo y del final de cada tema de la biblioteca (línea «comienzos y finales» del plan de
         playlists en vivo, después de la revalidación de D7: lo único que predecía el tema siguiente estaba
         cerca de los cambios, y el vector de la biblioteca sale del centro del tema, 3 tramos de 30 s del
         35-65 %). Por cada tema del índice de recomendación (tools/traktor_live/indice_recomendacion.json)
         decodifica solo los primeros y los últimos 60 s con ffmpeg y saca, con el mismo modelo y
         preprocesado que la biblioteca (ClapBackend, ventanas de 10 s), un vector por bloque de 30 s:
           comienzo (n, 2, 512)  0-30 s y 30-60 s;
           final    (n, 2, 512)  los 30 s antes de los últimos 30 s, y los últimos 30 s;
           uids, dur.
         Escribe <root>/biblioteca/clap_extremos.npz (fuera de Git). Lo ya calculado no se repite (guarda
         cada 50 temas). --limite N para la prueba chica, --low-priority para la corrida completa.
           python tools/dj_sets/clap_extremos.py --root "D:/TRAKTOR ML datos/dj_sets" --limite 5
CHANGELOG:
  - 2026-11-07: Creación.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from src.v4.pipeline.extract_representations import lower_priority, resample, split_windows  # noqa: E402
from tools.dj_sets.clap_blocks import Embedder, probe  # noqa: E402
from tools.traktor_live import recomendador as rc  # noqa: E402

DEFAULT_ROOT = REPO_ROOT / "artifacts" / "v4" / "dj_sets"
EDGE_S = 60.0
BLOCK_S = 30.0


def decode(path: Path, sr: int, start: Optional[float], from_end: bool) -> np.ndarray:
    """EDGE_S segundos mono float32: desde `start`, o los últimos (from_end)."""
    seek = ["-sseof", f"-{EDGE_S}"] if from_end else ["-ss", f"{start or 0.0}"]
    p = subprocess.run(["ffmpeg", "-v", "error", *seek, "-t", f"{EDGE_S}", "-i", str(path), "-ac", "1", "-ar", str(sr),
                        "-f", "f32le", "-"], capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "replace")[-300:])
    return np.frombuffer(p.stdout, dtype="<f4")


def edge_vectors(emb: Embedder, path: Path) -> Dict[str, np.ndarray]:
    sr, dur = probe(path)
    out = {"dur": np.float32(dur)}
    for name, from_end in (("comienzo", False), ("final", True)):
        x = decode(path, sr, 0.0, from_end)
        n = int(sr * BLOCK_S)
        blocks = [x[:n], x[n:2 * n]]
        wins, owner = [], []
        for k, b in enumerate(blocks):
            w = split_windows(resample(b, sr, emb.b.sr), emb.b.sr * 10) if len(b) >= sr * 10 else []
            wins += w
            owner += [k] * len(w)
        W = emb.windows(wins)
        own = np.asarray(owner)
        out[name] = np.stack([W[own == k].mean(axis=0) if (own == k).any() else np.full(512, np.nan, np.float32)
                              for k in range(2)]).astype(np.float32)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="CLAP del comienzo y del final de cada tema de la biblioteca")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--indice", default=str(rc.DEFAULT_INDEX))
    ap.add_argument("--paquete", default=None, help="Carpeta del paquete (por defecto, la del índice)")
    ap.add_argument("--hf-cache", default=None)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--limite", type=int, default=None)
    ap.add_argument("--low-priority", action="store_true")
    args = ap.parse_args()
    if args.low_priority:
        lower_priority()
    data, items = rc.load_index(Path(args.indice))
    pkg = Path(args.paquete or data["paquete"])
    dest = Path(args.root) / "biblioteca"
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / "clap_extremos.npz"
    done: Dict[str, Dict[str, np.ndarray]] = {}
    if out_path.exists():
        with np.load(out_path) as z:              # cerrado al salir: en Windows, abierto no se puede reemplazar
            for i, u in enumerate(z["uids"]):
                done[str(u)] = {"comienzo": z["comienzo"][i].copy(), "final": z["final"][i].copy(),
                                "dur": z["dur"][i]}
    todo = [it for it in items if it.uid not in done][: args.limite]
    emb = Embedder(args.hf_cache, args.batch)

    def save() -> None:
        uids = sorted(done)
        np.savez_compressed(out_path.with_suffix(".tmp.npz"), uids=np.asarray(uids),
                            comienzo=np.stack([done[u]["comienzo"] for u in uids]),
                            final=np.stack([done[u]["final"] for u in uids]),
                            dur=np.asarray([done[u]["dur"] for u in uids], dtype=np.float32))
        out_path.with_suffix(".tmp.npz").replace(out_path)
    t0, failed = time.perf_counter(), 0
    for k, it in enumerate(todo, 1):
        try:
            done[it.uid] = edge_vectors(emb, pkg / it.rel)
        except (RuntimeError, OSError, ValueError) as exc:
            failed += 1
            print(f"  [ERROR] {it.rel}: {exc}", flush=True)
        if k % 50 == 0 or k == len(todo):
            save()
            el = time.perf_counter() - t0
            print(f"  {k}/{len(todo)} temas, {el / k:.1f} s por tema, {failed} con error", flush=True)
    print(f"[OK] {len(done)} temas con comienzo y final -> {out_path}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
