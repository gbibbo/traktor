"""
TEST BLOCK 1: Integration test de common utilities.
Ejecutar desde repo root: python tests/v4/test_block1_common.py

Este test NO carga modelos (MERT, Demucs). Solo valida:
  1. Config carga correctamente y paths resuelven
  2. Catálogo se construye correctamente sobre test_20
  3. Audio se carga y segmenta correctamente
  4. Módulo demucs_utils importa sin errores y sin 16000 hardcoded
  5. Módulo embedding_utils importa sin errores con clase MERTEmbedder
  6. logging_utils funciona (crear log JSONL con 2 eventos)
"""
import sys
import os
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)


def test_config_and_paths():
    """Config carga -> paths resuelven -> artifacts root y audio root son válidos."""
    from src.v4.common.config_loader import load_config
    from src.v4.common.path_resolver import resolve_artifacts_root, resolve_dataset_audio_root

    config = load_config()
    assert "paths" in config
    assert "clustering" in config

    artifacts = resolve_artifacts_root(config)
    assert artifacts is not None

    audio_root = resolve_dataset_audio_root("test_20", config)
    assert audio_root.exists(), f"Audio root not found: {audio_root}"
    print(f"  OK: config loads, artifacts={artifacts}, audio={audio_root}")


def test_catalog():
    """Catálogo se construye con n_ok > 0, UIDs únicos, n_found == expected_n (si definido)."""
    from src.v4.common.config_loader import load_config
    from src.v4.common.path_resolver import resolve_dataset_audio_root
    from src.v4.common.catalog import build_catalog, load_catalog
    from src.v4.common.audio_utils import get_audio_files

    config = load_config()
    audio_root = resolve_dataset_audio_root("test_20", config)

    # n_files_found debe coincidir con expected_n (total archivos MP3 escaneados)
    all_files = get_audio_files(audio_root)
    expected_n = config.get("datasets", {}).get("test_20", {}).get("expected_n")
    if expected_n is not None:
        assert len(all_files) == expected_n, (
            f"Expected {expected_n} audio files found, got {len(all_files)}"
        )

    # Construir catálogo (solo tracks válidos)
    catalog = build_catalog(audio_root, "test_20", config)
    n_ok = len(catalog)
    assert n_ok > 0, "Catalog has 0 valid tracks"
    assert catalog["track_uid"].nunique() == n_ok, "Duplicate track_uids in catalog"
    assert catalog["duration_s"].notna().all(), "Some tracks have null duration"

    # Reload test
    catalog2 = load_catalog("test_20", config)
    assert len(catalog2) == n_ok, f"Reload mismatch: {len(catalog2)} != {n_ok}"

    print(f"  OK: catalog has {n_ok}/{len(all_files)} valid tracks, all UIDs unique")


def test_audio_loading_and_segmentation():
    """Cargar 1 track a 44.1kHz, segmentar en modo fallback -> 4 segmentos de 5s exactos."""
    from src.v4.common.config_loader import load_config
    from src.v4.common.path_resolver import resolve_dataset_audio_root
    from src.v4.common.audio_utils import get_audio_files, load_audio_torch, get_dj_segments, validate_audio_file
    from src.v4.config import ESSENTIA_SAMPLE_RATE
    import numpy as np

    config = load_config()
    audio_root = resolve_dataset_audio_root("test_20", config)
    files = get_audio_files(audio_root)
    assert len(files) > 0

    # Encontrar primer archivo válido
    test_file = None
    for f in files:
        if validate_audio_file(f):
            test_file = f
            break
    assert test_file is not None, "No valid audio file found"

    waveform, sr = load_audio_torch(test_file, target_sr=ESSENTIA_SAMPLE_RATE)
    assert sr == ESSENTIA_SAMPLE_RATE, f"Expected SR {ESSENTIA_SAMPLE_RATE}, got {sr}"
    assert waveform.shape[0] == 1, f"Expected mono (1 channel), got {waveform.shape[0]}"

    audio_np = waveform.squeeze(0).numpy()
    segments = get_dj_segments(audio_np, sr)

    expected_n_segs = 4  # 1 intro + 2 mid + 1 outro
    expected_len = int(5.0 * sr)
    assert len(segments) == expected_n_segs, f"Expected {expected_n_segs} segments, got {len(segments)}"
    for i, seg in enumerate(segments):
        assert len(seg) == expected_len, f"Segment {i}: expected {expected_len} samples, got {len(seg)}"

    print(f"  OK: loaded {test_file.name} at {sr}Hz, {len(segments)} segments of {expected_len} samples")


def test_demucs_import():
    """demucs_utils importa sin errores y sin 16000 hardcoded."""
    import inspect
    import src.v4.common.demucs_utils as du

    source = inspect.getsource(du)
    assert "16000" not in source, "Found hardcoded 16000 in demucs_utils.py!"

    assert hasattr(du, "load_demucs_model")
    assert hasattr(du, "separate_stems")
    assert hasattr(du, "stem_to_mono_numpy")
    assert hasattr(du, "process_track_stems")
    print("  OK: demucs_utils imports, no hardcoded 16000, all functions present")


def test_mert_import():
    """embedding_utils importa sin errores. MERTEmbedder tiene los métodos esperados."""
    from src.v4.common.embedding_utils import MERTEmbedder

    assert hasattr(MERTEmbedder, "embed_audio")
    assert hasattr(MERTEmbedder, "embed_segments")
    assert hasattr(MERTEmbedder, "aggregate_segments")
    print("  OK: MERTEmbedder class exists with expected methods")


def test_logging_utils():
    """logging_utils crea log JSONL con 2 eventos parseable."""
    import json
    from src.v4.common.logging_utils import open_phase_log, log_event, compute_config_hash

    with tempfile.TemporaryDirectory() as tmpdir:
        from pathlib import Path
        fh = open_phase_log(Path(tmpdir), "test_phase")
        log_event(fh, {"phase": "test", "dataset_name": "test_20", "event_type": "test_start"})
        log_event(fh, {"phase": "test", "dataset_name": "test_20", "event_type": "test_end"})
        fh.close()

        log_files = list(Path(tmpdir).glob("*.jsonl"))
        assert len(log_files) == 1
        lines = log_files[0].read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            evt = json.loads(line)
            assert "timestamp_utc" in evt
            assert "event_type" in evt

    h = compute_config_hash({"a": 1, "b": 2})
    assert len(h) == 8
    print("  OK: logging_utils creates valid JSONL, compute_config_hash works")


if __name__ == "__main__":
    print("=" * 70)
    print("TEST BLOCK 1: Common utilities integration")
    print("=" * 70)
    tests = [
        test_config_and_paths,
        test_catalog,
        test_audio_loading_and_segmentation,
        test_demucs_import,
        test_mert_import,
        test_logging_utils,
    ]
    passed, failed = 0, 0
    for test in tests:
        print(f"\n[{test.__name__}]")
        try:
            test()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL: {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"BLOCK 1: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    print("BLOCK 1: ALL TESTS PASSED")
