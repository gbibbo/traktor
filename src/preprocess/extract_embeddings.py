#!/usr/bin/env python3
"""
PURPOSE: Extract MERT-v1-330M embeddings for TRAKTOR ML V3 pipeline.
         Reads manifest.parquet, loads audio, segments it, runs MERT inference,
         and saves a [N, 1024] embedding matrix for downstream PCA/clustering.

CHANGELOG:
    2026-02-05: Initial implementation for V3 MERT pipeline.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

from tqdm import tqdm
from transformers import AutoModel, Wav2Vec2FeatureExtractor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "m-a-p/MERT-v1-330M"
TARGET_SR = 24_000
SEGMENT_DURATION = 5.0          # seconds
SEGMENT_SAMPLES = int(SEGMENT_DURATION * TARGET_SR)  # 120,000
PADDING_SECONDS = 45.0          # skip first/last N seconds
MAX_SEGMENTS = 8
EMBEDDING_DIM = 1024
NUM_LAYERS_POOL = 4             # average last N hidden layers


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_mert_model(
    device: str,
) -> Tuple[AutoModel, Wav2Vec2FeatureExtractor]:
    """Load MERT-v1-330M and its feature extractor.

    Returns the model in eval mode on *device* and the feature extractor.
    """
    print(f"[INFO] Loading model: {MODEL_NAME}")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()
    model.to(device)
    print(f"[INFO] Model loaded on {device}")
    return model, processor


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def load_and_prepare_audio(file_path: Path, target_sr: int = TARGET_SR) -> torch.Tensor:
    """Load an audio file, convert to mono, and resample to *target_sr*.

    Returns a 1-D float tensor at *target_sr* Hz.
    """
    # Use soundfile directly — torchcodec in .local breaks torchaudio.load().
    data, sr = sf.read(str(file_path), dtype="float32")  # (samples,) or (samples, channels)
    waveform = torch.from_numpy(data)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # (1, samples)
    else:
        waveform = waveform.T  # (channels, samples) — sf returns (samples, channels)

    # Stereo → mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform.squeeze(0)  # [num_samples]


def segment_audio(
    waveform: torch.Tensor,
    sample_rate: int = TARGET_SR,
    segment_duration: float = SEGMENT_DURATION,
    padding_seconds: float = PADDING_SECONDS,
    max_segments: int = MAX_SEGMENTS,
) -> torch.Tensor:
    """Extract uniformly-spaced segments from the middle of *waveform*.

    Skips the first and last *padding_seconds*. If the usable portion is
    shorter than one segment, falls back to a centre crop.

    Returns a tensor of shape ``[num_segments, segment_samples]``.
    """
    total_samples = waveform.shape[0]
    pad_samples = int(padding_seconds * sample_rate)
    seg_samples = int(segment_duration * sample_rate)

    usable_start = pad_samples
    usable_end = total_samples - pad_samples

    # Fallback: usable portion too short → centre crop
    if usable_end - usable_start < seg_samples:
        centre = total_samples // 2
        start = max(0, centre - seg_samples // 2)
        end = start + seg_samples
        # If the whole track is shorter than one segment, pad with zeros
        if end > total_samples:
            segment = torch.zeros(seg_samples)
            segment[: total_samples - start] = waveform[start:]
            return segment.unsqueeze(0)
        return waveform[start:end].unsqueeze(0)

    # Calculate how many segments fit and distribute uniformly
    usable_len = usable_end - usable_start
    n_segments = min(max_segments, usable_len // seg_samples)
    n_segments = max(1, n_segments)

    if n_segments == 1:
        starts = np.array([usable_start], dtype=np.int64)
    else:
        starts = np.linspace(
            usable_start, usable_end - seg_samples, n_segments, dtype=np.int64,
        )

    segments = [waveform[int(s) : int(s) + seg_samples] for s in starts]
    return torch.stack(segments)  # [n_segments, seg_samples]


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
def extract_embedding(
    segments: torch.Tensor,
    model: AutoModel,
    processor: Wav2Vec2FeatureExtractor,
    device: str,
) -> np.ndarray:
    """Run MERT inference on *segments* and return a 1024-dim embedding.

    Steps:
        1. Process segments through the feature extractor.
        2. Forward pass with ``output_hidden_states=True``.
        3. Average the last 4 hidden layers (element-wise).
        4. Mean-pool across the time dimension.
        5. Mean-pool across segments.

    Returns a numpy array of shape ``(1024,)``.
    """
    # segments: [n_segments, seg_samples]
    segments_list = [seg.numpy() for seg in segments]

    inputs = processor(
        segments_list,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states is a tuple of (num_layers+1) tensors of shape [batch, time, dim]
    hidden_states = outputs.hidden_states
    last_layers = torch.stack(hidden_states[-NUM_LAYERS_POOL:])  # [4, batch, time, 1024]
    pooled_layers = last_layers.mean(dim=0)   # [batch, time, 1024]
    pooled_time = pooled_layers.mean(dim=1)   # [batch, 1024]
    embedding = pooled_time.mean(dim=0)       # [1024]

    return embedding.cpu().numpy()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def _checkpoint_dir(dataset_dir: Path) -> Path:
    return dataset_dir / "embeddings"


def load_checkpoint(
    dataset_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Load an existing checkpoint. Returns ``None`` if not found or invalid."""
    ckpt_path = _checkpoint_dir(dataset_dir) / "checkpoint.json"
    if not ckpt_path.exists():
        return None

    try:
        with open(ckpt_path) as f:
            meta = json.load(f)

        emb_path = _checkpoint_dir(dataset_dir) / "checkpoint_embeddings.npy"
        ids_path = _checkpoint_dir(dataset_dir) / "checkpoint_track_ids.npy"
        if not emb_path.exists() or not ids_path.exists():
            print("[WARN] Checkpoint metadata found but .npy files missing — starting fresh")
            return None

        embeddings = np.load(emb_path)
        track_ids = np.load(ids_path, allow_pickle=True).tolist()

        if embeddings.shape[0] != len(track_ids):
            print("[WARN] Checkpoint shape mismatch — starting fresh")
            return None

        meta["embeddings"] = embeddings
        meta["track_ids"] = track_ids
        return meta

    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"[WARN] Corrupt checkpoint ({exc}) — starting fresh")
        return None


def save_checkpoint(
    dataset_dir: Path,
    last_idx: int,
    total_tracks: int,
    embeddings: np.ndarray,
    track_ids: List[str],
) -> None:
    """Persist current progress to disk."""
    emb_dir = _checkpoint_dir(dataset_dir)
    emb_dir.mkdir(parents=True, exist_ok=True)

    np.save(emb_dir / "checkpoint_embeddings.npy", embeddings)
    np.save(emb_dir / "checkpoint_track_ids.npy", np.array(track_ids, dtype=object))

    meta = {
        "last_processed_idx": last_idx,
        "total_tracks": total_tracks,
        "n_saved": len(track_ids),
        "timestamp": datetime.now().isoformat(),
        "model_name": MODEL_NAME,
    }
    with open(emb_dir / "checkpoint.json", "w") as f:
        json.dump(meta, f, indent=2)


def clear_checkpoint(dataset_dir: Path) -> None:
    """Remove checkpoint files after successful completion."""
    emb_dir = _checkpoint_dir(dataset_dir)
    for name in ("checkpoint.json", "checkpoint_embeddings.npy", "checkpoint_track_ids.npy"):
        p = emb_dir / name
        if p.exists():
            p.unlink()


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------
def process_tracks(
    manifest_df: pd.DataFrame,
    model: AutoModel,
    processor: Wav2Vec2FeatureExtractor,
    device: str,
    dataset_dir: Path,
    checkpoint_every: int = 25,
) -> Tuple[np.ndarray, List[str], List[Dict[str, str]]]:
    """Iterate over all tracks, extract embeddings, and checkpoint periodically.

    Returns
    -------
    embeddings : np.ndarray, shape [N_success, 1024]
    track_ids  : list[str]
    failures   : list[dict] with keys ``track_id`` and ``reason``
    """
    total = len(manifest_df)
    embeddings_list: List[np.ndarray] = []
    track_ids: List[str] = []
    failures: List[Dict[str, str]] = []
    start_idx = 0

    # --- Attempt resume from checkpoint ---
    ckpt = load_checkpoint(dataset_dir)
    if ckpt is not None:
        start_idx = ckpt["last_processed_idx"] + 1
        embeddings_list = list(ckpt["embeddings"])
        track_ids = ckpt["track_ids"]
        print(f"[INFO] Resuming from checkpoint — already processed {start_idx}/{total} tracks")
    else:
        print(f"[INFO] Starting fresh — {total} tracks to process")

    # --- Processing loop ---
    for idx in tqdm(
        range(start_idx, total),
        desc="Extracting embeddings",
        unit="track",
        initial=start_idx,
        total=total,
    ):
        row = manifest_df.iloc[idx]
        tid = row["track_id"]
        fpath = Path(row["file_path"])

        try:
            waveform = load_and_prepare_audio(fpath)
            segments = segment_audio(waveform)
            embedding = extract_embedding(segments, model, processor, device)

            embeddings_list.append(embedding)
            track_ids.append(tid)

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            failures.append({"track_id": tid, "reason": "CUDA OOM"})
            tqdm.write(f"[WARN] OOM on {fpath.name} — skipped")

        except Exception as exc:
            failures.append({"track_id": tid, "reason": str(exc)[:200]})
            tqdm.write(f"[WARN] Failed {fpath.name}: {exc}")

        # --- Periodic checkpoint ---
        if checkpoint_every > 0 and (idx + 1) % checkpoint_every == 0:
            save_checkpoint(
                dataset_dir, idx, total,
                np.array(embeddings_list), track_ids,
            )
            tqdm.write(f"[CKPT] Saved checkpoint at track {idx + 1}/{total}")

        # --- Periodic GPU cache clear ---
        if device == "cuda" and (idx + 1) % 50 == 0:
            torch.cuda.empty_cache()

    embeddings = np.array(embeddings_list) if embeddings_list else np.empty((0, EMBEDDING_DIM))
    return embeddings, track_ids, failures


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save_results(
    dataset_dir: Path,
    embeddings: np.ndarray,
    track_ids: List[str],
) -> Path:
    """Save final embeddings, track IDs, and extraction metadata."""
    emb_dir = dataset_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    emb_path = emb_dir / "embeddings_mert.npy"
    ids_path = emb_dir / "track_ids.npy"
    meta_path = emb_dir / "metadata.json"

    np.save(emb_path, embeddings)
    np.save(ids_path, np.array(track_ids, dtype=object))

    meta = {
        "model": MODEL_NAME,
        "date": datetime.now().isoformat(),
        "n_tracks": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "target_sr": TARGET_SR,
        "segment_duration_s": SEGMENT_DURATION,
        "padding_s": PADDING_SECONDS,
        "max_segments": MAX_SEGMENTS,
        "num_layers_pooled": NUM_LAYERS_POOL,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    clear_checkpoint(dataset_dir)
    return emb_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MERT-v1-330M embeddings for TRAKTOR ML V3",
    )
    parser.add_argument(
        "--dataset-name", type=str, required=True,
        help="Dataset name (reads artifacts/dataset/<name>/manifest.parquet)",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=25,
        help="Save checkpoint every N tracks (default: 25, 0 to disable)",
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
        help="Processing device (default: auto)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # --- Resolve paths ---
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "artifacts" / "dataset" / args.dataset_name
    manifest_path = dataset_dir / "manifest.parquet"

    # --- Device ---
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # --- Header ---
    print("=" * 70)
    print("TRAKTOR ML V3 — Extract MERT Embeddings")
    print("=" * 70)
    print(f"  Dataset         : {args.dataset_name}")
    print(f"  Manifest        : {manifest_path}")
    print(f"  Output dir      : {dataset_dir / 'embeddings'}")
    print(f"  Model           : {MODEL_NAME}")
    print(f"  Device          : {device}", end="")
    if device == "cuda":
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print()
    print(f"  Checkpoint freq : every {args.checkpoint_every} tracks")
    print()

    # --- Validate manifest ---
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        return 1

    try:
        manifest_df = pd.read_parquet(manifest_path)
    except Exception as exc:
        print(f"[ERROR] Cannot read manifest: {exc}")
        return 1

    required_cols = {"track_id", "file_path", "duration"}
    missing = required_cols - set(manifest_df.columns)
    if missing:
        print(f"[ERROR] Manifest missing columns: {missing}")
        return 1

    print(f"[INFO] Loaded manifest: {len(manifest_df)} tracks")

    # --- Load model ---
    t0 = time.time()
    try:
        model, processor = load_mert_model(device)
    except Exception as exc:
        print(f"[ERROR] Failed to load model: {exc}")
        return 1
    print(f"[INFO] Model loaded in {time.time() - t0:.1f}s")
    print()

    # --- Extract embeddings ---
    t0 = time.time()
    embeddings, track_ids, failures = process_tracks(
        manifest_df, model, processor, device,
        dataset_dir, args.checkpoint_every,
    )
    elapsed = time.time() - t0

    if embeddings.shape[0] == 0:
        print("[ERROR] No embeddings extracted — all tracks failed")
        return 1

    # --- Save ---
    emb_path = save_results(dataset_dir, embeddings, track_ids)

    # --- Summary ---
    print()
    print("=" * 70)
    print("Embedding Extraction Summary")
    print("=" * 70)
    print(f"  Tracks processed : {embeddings.shape[0]}")
    print(f"  Tracks failed    : {len(failures)}")
    print(f"  Embeddings shape : {embeddings.shape}")
    size_mb = emb_path.stat().st_size / (1024 * 1024)
    print(f"  Output file      : {emb_path} ({size_mb:.1f} MB)")
    if elapsed > 0 and embeddings.shape[0] > 0:
        per_track = elapsed / embeddings.shape[0]
        print(f"  Elapsed time     : {elapsed:.1f}s ({per_track:.2f}s per track)")

    if failures:
        print(f"\n  Failed tracks (showing up to 10):")
        for f in failures[:10]:
            print(f"    - {f['track_id'][:12]}…: {f['reason']}")

    print("\n" + "=" * 70)
    print("[SUCCESS] Embeddings extracted successfully!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
