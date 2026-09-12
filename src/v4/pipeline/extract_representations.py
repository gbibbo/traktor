"""
PURPOSE: Extraer representaciones congeladas (Discogs-EffNet, MAEST, MERT por capas, CLAP) para un
         dataset, en CPU, con la misma segmentación DJ de V4 (config segmentation.*: 3 x 30 s del tramo
         central por defecto). Reentrante: cachea un .npy por tema y variante y ensambla al final.
         Salida por variante: artifacts/<dataset>/representations/<variante>/{embeddings.npy,
         track_uids.json, manifest.json}, alineada por track_uids.json (fase 2 de
         docs/plans/representation_model_plan.md).
         Backends: 'effnet' y 'maest' requieren essentia-tensorflow (venv Python 3.10, sin torch);
         'mert' y 'clap' requieren torch + transformers (venv Python 3.11). --sources full,hpss añade
         variantes percusivas por HPSS (torch o scipy).
CHANGELOG:
  - 2026-09-12: Creación inicial.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.audio_utils import get_dj_segments, hpss_percussive  # noqa: E402
from src.v4.common.config_loader import load_config  # noqa: E402
from src.v4.common.path_resolver import resolve_dataset_artifacts  # noqa: E402

ESSENTIA_MODELS_DIR = REPO_ROOT / "models" / "essentia"
EFFNET_PB = "discogs-effnet-bs64-1.pb"
MAEST_PB = "discogs-maest-30s-pw-1.pb"
MERT_NAME = "m-a-p/MERT-v1-330M"
CLAP_NAME = "laion/clap-htsat-unfused"
MAEST_LAYERS = (7,)  # capa 7 = recomendada por MTG; cada capa extra es una pasada más del modelo de 350 MB
MERT_VARIANTS = {"last": [24], "last4": [21, 22, 23, 24], "l7": [7], "l12": [12]}


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

def load_mono(path: Path):
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1).astype(np.float32), sr


def resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio
    from scipy.signal import resample_poly
    g = math.gcd(sr, target_sr)
    return resample_poly(audio, target_sr // g, sr // g).astype(np.float32)


# ---------------------------------------------------------------------------
# Backends: cada uno devuelve {sufijo_variante: vector} para un segmento a su sample rate
# ---------------------------------------------------------------------------

class EffnetBackend:
    sr = 16000
    name = "effnet"

    def __init__(self, models_dir: Path):
        import essentia.standard as es
        self.model = es.TensorflowPredictEffnetDiscogs(graphFilename=str(models_dir / EFFNET_PB), output="PartitionedCall:1")
        self.info = {"model": EFFNET_PB, "output": "PartitionedCall:1", "pooling": "mean frames, mean segments", "dim": 1280}

    def embed(self, seg: np.ndarray) -> Dict[str, np.ndarray]:
        frames = self.model(seg)  # (n_frames, 1280)
        return {"": np.asarray(frames, dtype=np.float32).mean(axis=0)}


class MaestBackend:
    sr = 16000
    name = "maest"

    def __init__(self, models_dir: Path, layers=MAEST_LAYERS):
        import essentia.standard as es
        self.models = {L: es.TensorflowPredictMAEST(graphFilename=str(models_dir / MAEST_PB),
                                                    input="serving_default_melspectrogram",
                                                    output=f"StatefulPartitionedCall:{L}")
                       for L in layers}
        self.info = {"model": MAEST_PB, "layers": list(layers), "pooling": "mean tokens, mean segments", "dim": 768}

    def embed(self, seg: np.ndarray) -> Dict[str, np.ndarray]:
        out = {}
        for L, model in self.models.items():
            tokens = np.asarray(model(seg), dtype=np.float32)  # (batch, n_tokens, 768) o (n_tokens, 768)
            out[f"_l{L}"] = tokens.reshape(-1, tokens.shape[-1]).mean(axis=0)
        return out


class MertBackend:
    sr = 24000
    name = "mert"

    def __init__(self, hf_cache: Optional[str]):
        import torch
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        kwargs = {"trust_remote_code": True}
        if hf_cache:
            kwargs["cache_dir"] = hf_cache
        self.torch = torch
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(MERT_NAME, **kwargs)
        self.model = AutoModel.from_pretrained(MERT_NAME, **kwargs).eval()
        # transformers>=5 ya no rellena hidden_states en el código remoto de MERT: se capturan con hooks.
        # Estado 0 = entrada de la capa 1; estados 1..23 = salida de cada capa; estado 24 = last_hidden_state
        # (con la layer norm final, como en la convención de HF para modelos stable-layer-norm).
        self._captured: List = []
        layers = self.model.encoder.layers
        layers[0].register_forward_pre_hook(lambda m, args: self._captured.append(args[0].detach()))
        for layer in layers[:-1]:
            layer.register_forward_hook(lambda m, args, out: self._captured.append((out[0] if isinstance(out, tuple) else out).detach()))
        self.n_layers = len(layers)
        self.info = {"model": MERT_NAME, "variants": MERT_VARIANTS, "pooling": "mean time per layer, mean segments",
                     "dim": 1024, "hidden_states": "hooks: input of layer 1, outputs of layers 1..23, last_hidden_state"}

    def embed(self, seg: np.ndarray) -> Dict[str, np.ndarray]:
        inputs = self.processor(seg, sampling_rate=self.sr, return_tensors="pt")
        self._captured.clear()
        with self.torch.no_grad():
            out = self.model(**inputs, return_dict=True)
        hidden = list(self._captured) + [out.last_hidden_state.detach()]
        if len(hidden) != self.n_layers + 1:
            raise RuntimeError(f"MERT: capturados {len(hidden)} estados, esperados {self.n_layers + 1}")
        layers = self.torch.stack(hidden, dim=0).squeeze(1).mean(dim=1).numpy()  # (25, 1024)
        res = {"_layers": layers.astype(np.float32)}
        for suffix, idx in MERT_VARIANTS.items():
            res[f"_{suffix}"] = layers[idx].mean(axis=0).astype(np.float32)
        return res


class ClapBackend:
    sr = 48000
    name = "clap"

    def __init__(self, hf_cache: Optional[str]):
        import torch
        from transformers import ClapModel, ClapProcessor
        kwargs = {"cache_dir": hf_cache} if hf_cache else {}
        self.torch = torch
        self.processor = ClapProcessor.from_pretrained(CLAP_NAME, **kwargs)
        self.model = ClapModel.from_pretrained(CLAP_NAME, **kwargs).eval()
        self.info = {"model": CLAP_NAME, "pooling": "audio projection, mean segments", "dim": 512}

    def embed(self, seg: np.ndarray) -> Dict[str, np.ndarray]:
        inputs = self.processor(audio=seg, sampling_rate=self.sr, return_tensors="pt")
        with self.torch.no_grad():
            feats = self.model.get_audio_features(**inputs)
        if not self.torch.is_tensor(feats):  # transformers>=5: ModelOutput con la proyección en pooler_output
            feats = getattr(feats, "audio_embeds", None) if getattr(feats, "audio_embeds", None) is not None else feats.pooler_output
        vec = feats.reshape(-1).numpy().astype(np.float32)
        if vec.shape[0] != 512:
            raise RuntimeError(f"CLAP: dimensión inesperada {vec.shape}")
        return {"": vec}


def build_backend(name: str, models_dir: Path, hf_cache: Optional[str], maest_layers=MAEST_LAYERS):
    if name == "effnet":
        return EffnetBackend(models_dir)
    if name == "maest":
        return MaestBackend(models_dir, maest_layers)
    if name == "mert":
        return MertBackend(hf_cache)
    if name == "clap":
        return ClapBackend(hf_cache)
    raise ValueError(f"backend desconocido: {name}")


# ---------------------------------------------------------------------------
# Extracción
# ---------------------------------------------------------------------------

def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def segmentation_params(config: dict) -> dict:
    seg = config.get("segmentation", {})
    return {"segment_duration_s": float(seg.get("segment_duration_s", 30.0)),
            "n_intro": int(seg.get("n_intro_segments", 0)), "n_mid": int(seg.get("n_mid_segments", 3)),
            "n_outro": int(seg.get("n_outro_segments", 0))}


def variant_dir(rep_root: Path, backend: str, source: str, suffix: str) -> Path:
    return rep_root / f"{backend}_{source}{suffix}"


def run(dataset_name: str, config: dict, backends: List[str], sources: List[str], audio_root: Optional[Path],
        max_tracks: Optional[int], models_dir: Path, hf_cache: Optional[str], maest_layers=MAEST_LAYERS) -> None:
    artifacts = resolve_dataset_artifacts(dataset_name, config)
    catalog = pd.read_parquet(artifacts / "catalog.parquet")
    if max_tracks:
        catalog = catalog.head(max_tracks)
    rep_root = artifacts / "representations"
    seg_params = segmentation_params(config)
    print(f"[INFO] {len(catalog)} temas | backends={backends} | sources={sources} | segmentación={seg_params}")

    loaded = {b: build_backend(b, models_dir, hf_cache, maest_layers) for b in backends}
    t0 = time.time()
    failed: Dict[str, str] = {}
    for i, row in enumerate(catalog.itertuples(), 1):
        uid = row.track_uid
        path = (audio_root / row.filename) if audio_root else Path(row.source_path)
        # ¿ya está todo en caché para este tema?
        pending = [(b, s) for b in backends for s in sources
                   if not (rep_root / f"{b}_{s}" / "cache" / f"{uid}.npz").exists()]
        if not pending:
            continue
        try:
            audio, sr = load_mono(path)
            segs_full = get_dj_segments(audio, sr, seg_params["segment_duration_s"],
                                        seg_params["n_intro"], seg_params["n_mid"], seg_params["n_outro"])
            # HPSS solo sobre los segmentos (90 s), no sobre el tema entero: 4-5x más barato
            per_source = {"full": segs_full}
            if "hpss" in sources:
                per_source["hpss"] = [hpss_percussive(seg) for seg in segs_full]
            for b, s in pending:
                backend = loaded[b]
                segs = per_source[s]
                acc: Dict[str, List[np.ndarray]] = {}
                for seg in segs:
                    for suffix, vec in backend.embed(resample(seg, sr, backend.sr)).items():
                        acc.setdefault(suffix, []).append(vec)
                cache = rep_root / f"{b}_{s}" / "cache"
                cache.mkdir(parents=True, exist_ok=True)
                np.savez(cache / f"{uid}.npz", **{(k or "_"): np.mean(v, axis=0) for k, v in acc.items()})
        except Exception as exc:  # noqa: BLE001
            failed[uid] = f"{type(exc).__name__}: {exc}"
            print(f"[WARN] {row.filename}: {failed[uid]}")
        if i % 10 == 0 or i == len(catalog):
            el = time.time() - t0
            print(f"[INFO] {i}/{len(catalog)} temas | {el/60:.1f} min | {el/i:.1f} s/tema", flush=True)

    # Ensamblar por variante
    for b in backends:
        for s in sources:
            cache = rep_root / f"{b}_{s}" / "cache"
            uids, per_key = [], {}
            for row in catalog.itertuples():
                f = cache / f"{row.track_uid}.npz"
                if not f.exists():
                    continue
                data = np.load(f)
                uids.append(row.track_uid)
                for k in data.files:
                    per_key.setdefault(k, []).append(data[k])
            for k, vecs in per_key.items():
                if k == "_layers":
                    continue  # matriz (25,1024) por tema: se conserva solo en caché
                suffix = "" if k == "_" else k
                out = variant_dir(rep_root, b, s, suffix)
                out.mkdir(parents=True, exist_ok=True)
                mat = np.stack(vecs).astype(np.float32)
                np.save(out / "embeddings.npy", mat)
                (out / "track_uids.json").write_text(json.dumps(uids), encoding="utf-8")
                manifest = {"variant": out.name, "backend": b, "source": s, "n_tracks": len(uids),
                            "dim": int(mat.shape[1]), "segmentation": seg_params, "backend_info": loaded[b].info,
                            "failed_uids": {u: e for u, e in failed.items()}, "git_commit": git_commit(),
                            "created": dt.datetime.now(dt.timezone.utc).isoformat(), "finite": bool(np.isfinite(mat).all())}
                (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                print(f"[INFO] {out.name}: {mat.shape} finite={manifest['finite']}")
    if failed:
        print(f"[WARN] {len(failed)} temas fallidos")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extrae representaciones congeladas (CPU) con la segmentación DJ de V4.")
    parser.add_argument("--dataset-name", default="test_20")
    parser.add_argument("--config", default=None)
    parser.add_argument("--audio-root", default=None, help="Carpeta de audio (une con catalog.filename); útil en WSL")
    parser.add_argument("--models", default="effnet,maest", help="effnet,maest (essentia-tensorflow) | mert,clap (torch)")
    parser.add_argument("--sources", default="full", help="full,hpss")
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument("--models-dir", default=str(ESSENTIA_MODELS_DIR))
    parser.add_argument("--hf-cache", default=os.environ.get("TRAKTOR_HF_CACHE"))
    parser.add_argument("--threads", type=int, default=None, help="Hilos para torch (por defecto los del sistema)")
    parser.add_argument("--maest-layers", default=",".join(str(L) for L in MAEST_LAYERS), help="Capas MAEST a extraer, p. ej. 7,12")
    args = parser.parse_args()

    if args.threads:
        try:
            import torch
            torch.set_num_threads(args.threads)
        except ImportError:
            pass
    config = load_config(Path(args.config) if args.config else None)
    run(args.dataset_name, config, args.models.split(","), args.sources.split(","),
        Path(args.audio_root) if args.audio_root else None, args.max_tracks, Path(args.models_dir), args.hf_cache,
        tuple(int(x) for x in args.maest_layers.split(",")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
