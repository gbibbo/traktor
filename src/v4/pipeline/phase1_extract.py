"""
PURPOSE: Phase 1 — Extracción de features: Essentia (BPM/key/beats), Demucs (drums stem),
         MERT embeddings (full mix + drums percusivo). Reentrante via progress_shard_XX.json.
         EJECUTAR SOLO VIA SLURM (partición a100, GPU). Nunca en el nodo de login.
CHANGELOG:
  - 2026-03-01: Añadido run_id en progress_shard_XX.json. Si el run_id no coincide con el
                run actual (e.g. smoke test vs full run), se resetea el checkpoint para
                evitar que tracks marcados como procesados sean silenciosamente omitidos.
  - 2026-02-28: Creación inicial V4. Reentrancia, sharding, logging JSONL.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.audio_utils import get_dj_segments, load_audio_torch, validate_audio_file
from src.v4.common.catalog import load_catalog
from src.v4.common.config_loader import load_config
from src.v4.common.demucs_utils import load_demucs_model, process_track_stems
from src.v4.common.embedding_utils import MERTEmbedder
from src.v4.common.logging_utils import get_slurm_job_id, log_event, open_phase_log
from src.v4.common.path_resolver import (
    resolve_dataset_artifacts,
    resolve_hf_cache,
    resolve_logs_root,
    resolve_torch_cache,
)
from src.v4.config import (
    ESSENTIA_SAMPLE_RATE,
    MERT_SAMPLE_RATE,
    SEGMENT_DURATION_S,
    N_INTRO_SEGMENTS,
    N_MID_SEGMENTS,
    N_OUTRO_SEGMENTS,
)


def compute_run_id() -> str:
    """Genera un run_id estable para este Slurm job (o 'local' si no hay Slurm).

    Si hay SLURM_JOB_ID, es único por job — detecta correctamente smoke_test vs full_run.
    Si no hay Slurm, usa 'local' para que re-runs locales preserven el checkpoint.
    """
    slurm_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_ARRAY_JOB_ID")
    if slurm_id:
        return f"slurm_{slurm_id}"
    return "local"


def load_progress(progress_path: Path, expected_run_id: str) -> dict:
    """Cargar estado reentrante. Resetea si run_id no coincide (run diferente)."""
    empty = {"processed": [], "failed": [], "last_checkpoint": 0, "run_id": expected_run_id}
    if not progress_path.exists():
        return empty
    with open(progress_path) as f:
        data = json.load(f)
    stored_run_id = data.get("run_id", "")
    if stored_run_id and stored_run_id != expected_run_id:
        print(f"[WARN] Progress mismatch: stored run_id={stored_run_id!r}, "
              f"current={expected_run_id!r} — resetting shard checkpoint.")
        return empty
    return data


def save_progress(progress_path: Path, progress: dict) -> None:
    import datetime
    progress["updated_utc"] = datetime.datetime.utcnow().isoformat()
    tmp = progress_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(progress, f, indent=2)
    tmp.rename(progress_path)


def extract_essentia_features(audio_44k: np.ndarray) -> dict:
    """
    Extraer BPM, key y beat ticks con Essentia a 44.1kHz.

    Returns: dict con bpm, bpm_confidence, beat_ticks, beat_confidence, key, key_confidence.
    """
    import essentia.standard as es

    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
    bpm, beat_ticks, bpm_confidence, _, beat_loudness = rhythm_extractor(audio_44k)
    beat_confidence = float(np.mean(beat_loudness)) if len(beat_loudness) > 0 else 0.0

    key_extractor = es.KeyExtractor()
    key, scale, key_confidence = key_extractor(audio_44k)
    key_str = f"{key} {scale}"

    return {
        "bpm": float(bpm),
        "bpm_confidence": float(bpm_confidence),
        "beat_ticks": beat_ticks.tolist(),
        "beat_confidence": beat_confidence,
        "key": key_str,
        "key_confidence": float(key_confidence),
    }


def setup_caches(config: dict) -> None:
    """Configurar variables de entorno de cache (HF, torch) antes de cargar modelos."""
    hf_cache = resolve_hf_cache(config)
    if hf_cache:
        hf_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf_cache)
        os.environ["TRANSFORMERS_CACHE"] = str(hf_cache)

    torch_cache = resolve_torch_cache(config)
    if torch_cache:
        torch_cache.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(torch_cache)


def main() -> int:
    parser = argparse.ArgumentParser(description="TRAKTOR ML V4 — Phase 1: Feature Extraction (GPU)")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--max-tracks", type=int, default=None, help="Limitar N tracks (smoke test)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    print("=" * 70)
    print(f"[INFO] TRAKTOR ML V4 — Phase 1: Feature Extraction")
    print(f"[INFO] Dataset: {args.dataset_name} | Shard: {args.shard_id}/{args.num_shards} | Device: {args.device}")
    print(f"[INFO] SLURM Job: {get_slurm_job_id() or 'N/A'}")
    print("=" * 70)

    config = load_config(Path(args.config) if args.config else None)
    setup_caches(config)

    artifacts_dir = resolve_dataset_artifacts(args.dataset_name, config)
    logs_root = resolve_logs_root(args.dataset_name, config)
    embeddings_dir = artifacts_dir / "embeddings" / "shards"
    features_dir = artifacts_dir / "features" / "shards"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    shard_tag = f"shard_{args.shard_id:02d}"
    progress_path = embeddings_dir / f"progress_{shard_tag}.json"
    run_id = compute_run_id()

    log_fh = open_phase_log(logs_root, f"phase1_extract_{shard_tag}")
    log_event(log_fh, {
        "phase": "phase1",
        "dataset_name": args.dataset_name,
        "event_type": "phase_start",
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "device": args.device,
    })

    # Cargar catálogo y seleccionar shard
    catalog = load_catalog(args.dataset_name, config)
    track_uids = catalog["track_uid"].tolist()
    source_paths = {row["track_uid"]: row["source_path"] for _, row in catalog.iterrows()}

    # Sharding
    shard_uids = track_uids[args.shard_id::args.num_shards]
    if args.max_tracks is not None:
        shard_uids = shard_uids[:args.max_tracks]
    print(f"[INFO] Shard {args.shard_id}: {len(shard_uids)} tracks to process")

    # Cargar estado reentrante (resetea si run_id no coincide con job anterior)
    progress = load_progress(progress_path, expected_run_id=run_id)
    already_processed = set(progress["processed"])
    remaining = [uid for uid in shard_uids if uid not in already_processed]
    print(f"[INFO] Already processed: {len(already_processed)}, remaining: {len(remaining)}")

    if not remaining:
        print("[INFO] All tracks already processed. Nothing to do.")
        log_fh.close()
        return 0

    # Cargar modelos
    print("[INFO] Loading Demucs model...")
    demucs_model, demucs_sr = load_demucs_model(device=args.device)

    hf_cache_str = os.environ.get("HF_HOME")
    print("[INFO] Loading MERT model...")
    embedder = MERTEmbedder(device=args.device, hf_cache=hf_cache_str)

    # Buffers
    mert_perc_list = []
    mert_full_list = []
    uid_list = []
    bpm_key_rows = []

    seg_cfg = config.get("segmentation", {})
    beat_conf_threshold = float(seg_cfg.get("beat_conf_threshold", 0.5))

    t0 = time.time()
    for idx, uid in enumerate(remaining):
        audio_path = Path(source_paths[uid])
        t_track = time.time()
        try:
            if not validate_audio_file(audio_path):
                raise ValueError(f"Invalid audio file: {audio_path}")

            # 1. Cargar a 44.1kHz para Essentia
            waveform_44k, _ = load_audio_torch(audio_path, target_sr=ESSENTIA_SAMPLE_RATE)
            audio_44k = waveform_44k.squeeze(0).numpy()

            # 2. Essentia: BPM + key + beats
            feats = extract_essentia_features(audio_44k)

            # 3. Demucs: separar drums → resample a 24kHz
            drums_24k, full_24k = process_track_stems(
                audio_path, demucs_model, demucs_sr, device=args.device, target_sr=MERT_SAMPLE_RATE
            )

            # 4. Segmentos DJ
            beat_ticks = np.array(feats["beat_ticks"]) * ESSENTIA_SAMPLE_RATE if feats["beat_ticks"] else None
            segs_perc = get_dj_segments(
                drums_24k, MERT_SAMPLE_RATE,
                beat_ticks=beat_ticks,
                bpm=feats["bpm"],
                beat_confidence=feats["beat_confidence"],
                beat_conf_threshold=beat_conf_threshold,
            )
            segs_full = get_dj_segments(
                full_24k, MERT_SAMPLE_RATE,
                beat_ticks=beat_ticks,
                bpm=feats["bpm"],
                beat_confidence=feats["beat_confidence"],
                beat_conf_threshold=beat_conf_threshold,
            )

            # 5. MERT embeddings → agregar
            emb_perc = embedder.aggregate_segments(embedder.embed_segments(segs_perc))
            emb_full = embedder.aggregate_segments(embedder.embed_segments(segs_full))

            mert_perc_list.append(emb_perc)
            mert_full_list.append(emb_full)
            uid_list.append(uid)
            bpm_key_rows.append({
                "track_uid": uid,
                "bpm": feats["bpm"],
                "bpm_confidence": feats["bpm_confidence"],
                "beat_confidence": feats["beat_confidence"],
                "key": feats["key"],
                "key_confidence": feats["key_confidence"],
            })

            progress["processed"].append(uid)
            duration_ms = int((time.time() - t_track) * 1000)
            log_event(log_fh, {
                "phase": "phase1",
                "dataset_name": args.dataset_name,
                "event_type": "track_ok",
                "track_uid": uid,
                "filepath": str(audio_path),
                "duration_ms": duration_ms,
                "bpm": feats["bpm"],
                "key": feats["key"],
            })
            print(f"  [{idx+1}/{len(remaining)}] OK: {audio_path.name} "
                  f"({duration_ms}ms, BPM={feats['bpm']:.1f}, key={feats['key']})")

        except Exception as e:
            progress["failed"].append(uid)
            log_event(log_fh, {
                "phase": "phase1",
                "dataset_name": args.dataset_name,
                "event_type": "track_failed",
                "track_uid": uid,
                "filepath": str(audio_path),
                "error": str(e),
            })
            print(f"  [{idx+1}/{len(remaining)}] FAIL: {audio_path.name} — {e}")

        # Checkpoint
        if (idx + 1) % args.checkpoint_every == 0 and uid_list:
            _save_checkpoint(embeddings_dir, features_dir, shard_tag,
                             uid_list, mert_perc_list, mert_full_list, bpm_key_rows)
            progress["last_checkpoint"] = idx + 1
            save_progress(progress_path, progress)
            print(f"  [CHECKPOINT] Saved {len(uid_list)} tracks")

    # Checkpoint final
    if uid_list:
        _save_checkpoint(embeddings_dir, features_dir, shard_tag,
                         uid_list, mert_perc_list, mert_full_list, bpm_key_rows)
    save_progress(progress_path, progress)

    elapsed = time.time() - t0
    log_event(log_fh, {
        "phase": "phase1",
        "dataset_name": args.dataset_name,
        "event_type": "phase_end",
        "shard_id": args.shard_id,
        "n_ok": len(uid_list),
        "n_failed": len(progress["failed"]),
        "elapsed_s": elapsed,
    })
    log_fh.close()

    print("=" * 70)
    print(f"[INFO] Phase 1 shard {args.shard_id} complete in {elapsed:.1f}s")
    print(f"[INFO] Processed: {len(uid_list)} OK, {len(progress['failed'])} failed")
    print("=" * 70)
    return 0


def _save_checkpoint(
    embeddings_dir: Path,
    features_dir: Path,
    shard_tag: str,
    uid_list: list,
    mert_perc_list: list,
    mert_full_list: list,
    bpm_key_rows: list,
) -> None:
    """Guardar estado parcial de shard."""
    import pandas as pd

    np.save(embeddings_dir / f"mert_perc_{shard_tag}.npy", np.stack(mert_perc_list))
    np.save(embeddings_dir / f"mert_full_{shard_tag}.npy", np.stack(mert_full_list))
    with open(embeddings_dir / f"track_uids_{shard_tag}.json", "w") as f:
        json.dump(uid_list, f)
    pd.DataFrame(bpm_key_rows).to_parquet(features_dir / f"bpm_key_{shard_tag}.parquet", index=False)


if __name__ == "__main__":
    sys.exit(main())
