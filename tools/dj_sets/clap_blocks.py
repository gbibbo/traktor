"""
PURPOSE: CLAP por bloques de 30 s de cada archivo de audio de una carpeta de youtube_audio.py (plan de
         playlists en vivo, T11), con el mismo modelo y el mismo preprocesado que la biblioteca
         (extract_representations.ClapBackend: laion/clap-htsat-unfused a 48 kHz, cada tramo de 30 s en 3
         ventanas de 10 s; remuestreo con resample_poly desde la frecuencia del archivo). Lee el audio en
         streaming con ffmpeg (un set de 4 h no entra en memoria). Por archivo deja
         <root>/clap/<carpeta>/<nombre>.npz con:
           blocks   (n, 512)  media de las ventanas de cada bloque;
           windows  (m, 512)  cada ventana de 10 s, y win_block (m,) el bloque de cada una;
           starts   (n,)      segundo de inicio de cada bloque; dur: duración en segundos;
           dj       (512,)    solo en archivos de hasta 20 min: el embedding como en la biblioteca
                              (3 tramos de 30 s del 35-65 %, read_dj_segments), para comparar con clap_full.
         Lo ya calculado no se repite. --low-priority para las corridas largas.
         --verificar N: recalcula N temas de la biblioteca y compara con clap_full (mismo espacio).
CHANGELOG:
  - 2026-11-07: Creación (piloto de trayectorias de DJs).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from src.v4.common.audio_utils import read_dj_segments  # noqa: E402
from src.v4.pipeline.extract_representations import ClapBackend, lower_priority, resample, split_windows  # noqa: E402

DEFAULT_ROOT = REPO_ROOT / "artifacts" / "v4" / "dj_sets"
BLOCK_S = 30.0
DJ_MAX_S = 20 * 60
MIN_TAIL_S = 10.0


def probe(path: Path) -> Tuple[int, float]:
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                        "stream=sample_rate:format=duration", "-of", "json", str(path)],
                       capture_output=True, text=True, check=True)
    d = json.loads(p.stdout)
    return int(d["streams"][0]["sample_rate"]), float(d["format"]["duration"])


def stream_blocks(path: Path, sr: int, block_s: float = BLOCK_S):
    """Bloques mono float32 de block_s segundos a la frecuencia del archivo (el último, si dura >= 10 s)."""
    n = int(sr * block_s)
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                            stdout=subprocess.PIPE)
    try:
        i = 0
        while True:
            raw = proc.stdout.read(n * 4)
            if not raw:
                break
            x = np.frombuffer(raw, dtype="<f4")
            if len(x) < n and len(x) < MIN_TAIL_S * sr:
                break
            yield i * block_s, x
            i += 1
    finally:
        proc.stdout.close()
        proc.wait()


class Embedder:
    def __init__(self, hf_cache: Optional[str], batch: int):
        self.b = ClapBackend(hf_cache)
        self.batch = batch

    def windows(self, wins: List[np.ndarray]) -> np.ndarray:
        out = []
        for i in range(0, len(wins), self.batch):
            chunk = wins[i:i + self.batch]
            inputs = self.b.processor(audio=chunk, sampling_rate=self.b.sr, return_tensors="pt")
            with self.b.torch.no_grad():
                feats = self.b.model.get_audio_features(**inputs)
            if not self.b.torch.is_tensor(feats):
                feats = getattr(feats, "audio_embeds", None) if getattr(feats, "audio_embeds", None) is not None else feats.pooler_output
            out.append(feats.reshape(len(chunk), -1).numpy().astype(np.float32))
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)

    def dj(self, path: Path) -> np.ndarray:
        segs, sr = read_dj_segments(path, BLOCK_S, 0, 3, 0)
        return np.stack([self.b.embed(resample(s, sr, self.b.sr))[""] for s in segs]).mean(axis=0)


def embed_file(emb: Embedder, path: Path) -> dict:
    sr, dur = probe(path)
    starts, wins, win_block = [], [], []
    for k, (t0, x) in enumerate(stream_blocks(path, sr)):
        w = split_windows(resample(x, sr, emb.b.sr), emb.b.sr * 10)
        starts.append(t0)
        wins += w
        win_block += [k] * len(w)
    W = emb.windows(wins)
    wb = np.asarray(win_block, dtype=np.int32)
    B = np.stack([W[wb == k].mean(axis=0) for k in range(len(starts))]) if starts else np.zeros((0, 512), np.float32)
    out = {"blocks": B.astype(np.float32), "windows": W, "win_block": wb,
           "starts": np.asarray(starts, dtype=np.float32), "dur": np.float32(dur)}
    if dur <= DJ_MAX_S:
        out["dj"] = emb.dj(path).astype(np.float32)
    return out


def verify(emb: Embedder, n: int) -> None:
    """Recalcula n temas de la biblioteca como clap_full y compara con lo guardado (coseno)."""
    import csv
    rep = REPO_ROOT / "artifacts" / "v4" / "datasets" / "musica" / "representations" / "clap_full"
    E = np.load(rep / "embeddings.npy")
    uids = json.loads((rep / "track_uids.json").read_text(encoding="utf-8"))
    row = {u: i for i, u in enumerate(uids)}
    pkg = Path("C:/TRAKTOR ML")
    with open(pkg / "_organizacion.csv", encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["track_uid"] in row][:: max(1, 1700 // max(1, n))][:n]
    for r in rows:
        p = pkg / r["archivo"]
        v = emb.dj(p)
        g = E[row[r["track_uid"]]]
        cos = float(v @ g / (np.linalg.norm(v) * np.linalg.norm(g)))
        print(f"  coseno {cos:.4f}  {r['archivo']}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="CLAP por bloques de 30 s (piloto de trayectorias)")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--carpeta", action="append", help="Carpeta de <root>/audio (id de playlist o nombre); repetible")
    ap.add_argument("--limit", type=int, default=None, help="Solo los primeros N archivos de cada carpeta (prueba)")
    ap.add_argument("--batch", type=int, default=24, help="Ventanas de 10 s por pasada del modelo")
    ap.add_argument("--hf-cache", default=None)
    ap.add_argument("--verificar", type=int, default=0, help="Recalcular N temas de la biblioteca contra clap_full")
    ap.add_argument("--low-priority", action="store_true")
    args = ap.parse_args()
    if args.low_priority:
        lower_priority()
    emb = Embedder(args.hf_cache, args.batch)
    if args.verificar:
        verify(emb, args.verificar)
    root = Path(args.root)
    for carpeta in args.carpeta or []:
        folder = root / "audio" / carpeta
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        out_dir = root / "clap" / carpeta
        out_dir.mkdir(parents=True, exist_ok=True)
        files = [e["archivo"] for e in manifest["entradas"] if e.get("archivo")][: args.limit]
        for i, name in enumerate(files, 1):
            out = out_dir / (Path(name).stem + ".npz")
            if out.exists():
                continue
            t = time.perf_counter()
            res = embed_file(emb, folder / name)
            tmp = out.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **res)
            tmp.replace(out)
            el = time.perf_counter() - t
            h = float(res["dur"]) / 3600
            print(f"  {carpeta} {i}/{len(files)} {name}: {len(res['starts'])} bloques, {h * 60:.1f} min de audio, "
                  f"{el:.0f} s ({el / max(h, 1e-6) / 60:.1f} min por hora de audio)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
