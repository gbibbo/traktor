"""
TEST BLOCK 2: Verificación de pipeline scripts y Slurm jobs.
Parte A: Tests locales sin GPU (automático).
Parte B: Smoke test GPU (vía Slurm, requiere intervención humana).

Ejecutar desde repo root: python tests/v4/test_block2_pipeline.py
"""
import ast
import importlib.util
import json
import subprocess
import sys
import os
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)


def test_pipeline_imports():
    """Todos los scripts de pipeline importan sin errores de sintaxis."""
    import importlib
    modules = [
        "src.v4.pipeline.phase0_ingest",
        "src.v4.pipeline.phase1_extract",
        "src.v4.pipeline.phase1_merge_shards",
    ]
    for mod in modules:
        spec = importlib.util.spec_from_file_location(mod, os.path.join(REPO_ROOT, mod.replace(".", "/") + ".py"))
        # Solo compilar (verificar sintaxis), no ejecutar
        import ast
        src_path = os.path.join(REPO_ROOT, mod.replace(".", "/") + ".py")
        with open(src_path) as f:
            source = f.read()
        ast.parse(source)
        print(f"  OK: {mod} syntax valid")


def test_slurm_jobs_syntax():
    """Todos los Slurm jobs pasan bash -n (verificación de sintaxis)."""
    import glob
    jobs = glob.glob(os.path.join(REPO_ROOT, "slurm/jobs/v4/*.job"))
    assert len(jobs) > 0, "No V4 Slurm jobs found"
    for job in sorted(jobs):
        result = subprocess.run(["bash", "-n", job], capture_output=True, text=True)
        if result.returncode != 0:
            raise AssertionError(f"Syntax error in {job}:\n{result.stderr}")
        print(f"  OK: {os.path.basename(job)} syntax valid")


def test_phase0_runs():
    """Phase 0 ejecuta sobre test_20 y genera catalog.parquet."""
    from src.v4.common.config_loader import load_config
    from src.v4.common.path_resolver import resolve_dataset_artifacts

    config = load_config()
    result = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "src/v4/pipeline/phase0_ingest.py"),
         "--dataset-name", "test_20"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(f"Phase 0 failed:\n{result.stdout}\n{result.stderr}")

    artifacts_dir = resolve_dataset_artifacts("test_20", config)
    catalog_path = artifacts_dir / "catalog.parquet"
    report_path = artifacts_dir / "ingest_report.json"

    assert catalog_path.exists(), f"catalog.parquet not found: {catalog_path}"
    assert report_path.exists(), f"ingest_report.json not found: {report_path}"

    import pandas as pd
    catalog = pd.read_parquet(catalog_path)
    assert len(catalog) > 0, "Catalog is empty"
    assert catalog["track_uid"].nunique() == len(catalog), "Duplicate track_uids!"

    with open(report_path) as f:
        report = json.load(f)
    assert report["n_catalog_entries"] > 0
    assert report["n_catalog_entries"] == len(catalog)

    print(f"  OK: Phase 0 generated catalog with {len(catalog)} tracks, report OK")


def test_merge_shards_dummy():
    """phase1_merge_shards funciona con shards dummy (np.random)."""
    import numpy as np
    import pandas as pd
    from src.v4.common.catalog import build_catalog, load_catalog
    from src.v4.common.config_loader import load_config
    from src.v4.common.path_resolver import resolve_dataset_artifacts, resolve_dataset_audio_root

    config = load_config()
    artifacts_dir = resolve_dataset_artifacts("test_20", config)

    # Necesita catálogo existente
    catalog = load_catalog("test_20", config)
    uids = catalog["track_uid"].tolist()[:6]  # usar primeros 6 como dummy

    shards_dir = artifacts_dir / "embeddings" / "shards"
    features_shards_dir = artifacts_dir / "features" / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    features_shards_dir.mkdir(parents=True, exist_ok=True)

    # Crear 2 shards dummy
    for i, (shard_uids, shard_tag) in enumerate([
        (uids[:3], "shard_98"),
        (uids[3:6], "shard_99"),
    ]):
        n = len(shard_uids)
        np.save(shards_dir / f"mert_perc_{shard_tag}.npy", np.random.randn(n, 1024).astype(np.float32))
        np.save(shards_dir / f"mert_full_{shard_tag}.npy", np.random.randn(n, 1024).astype(np.float32))
        with open(shards_dir / f"track_uids_{shard_tag}.json", "w") as f:
            json.dump(shard_uids, f)
        pd.DataFrame([{"track_uid": u, "bpm": 128.0, "bpm_confidence": 0.9,
                        "beat_confidence": 0.8, "key": "C minor", "key_confidence": 0.7}
                       for u in shard_uids]).to_parquet(
            features_shards_dir / f"bpm_key_{shard_tag}.parquet", index=False)

    # Ejecutar merge
    result = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "src/v4/pipeline/phase1_merge_shards.py"),
         "--dataset-name", "test_20"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(f"Merge failed:\n{result.stdout}\n{result.stderr}")

    # Verificar outputs
    perc_path = artifacts_dir / "embeddings" / "mert_perc.npy"
    full_path = artifacts_dir / "embeddings" / "mert_full.npy"
    assert perc_path.exists()
    assert full_path.exists()

    perc = np.load(perc_path)
    full = np.load(full_path)
    assert perc.shape == (6, 1024), f"Expected (6,1024), got {perc.shape}"
    assert full.shape == (6, 1024), f"Expected (6,1024), got {full.shape}"

    manifest_path = artifacts_dir / "run_manifest.json"
    assert manifest_path.exists()
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["n_tracks_merged"] == 6

    # Limpiar shards dummy
    for shard_tag in ["shard_98", "shard_99"]:
        for f in [
            shards_dir / f"mert_perc_{shard_tag}.npy",
            shards_dir / f"mert_full_{shard_tag}.npy",
            shards_dir / f"track_uids_{shard_tag}.json",
            features_shards_dir / f"bpm_key_{shard_tag}.parquet",
        ]:
            if f.exists():
                f.unlink()

    print(f"  OK: merge_shards dummy test passed, shapes {perc.shape}, manifest OK")


if __name__ == "__main__":
    print("=" * 70)
    print("TEST BLOCK 2 — Parte A: Pipeline verification (no GPU)")
    print("=" * 70)

    tests = [
        test_pipeline_imports,
        test_slurm_jobs_syntax,
        test_phase0_runs,
        test_merge_shards_dummy,
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
    print(f"BLOCK 2 Parte A: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    print("BLOCK 2 Parte A: ALL TESTS PASSED")
    print("")
    print(">>> Parte B (GPU): submit smoke_test_gpu.job:")
    print("    ./slurm/tools/on_submit.sh sbatch slurm/jobs/v4/smoke_test_gpu.job test_20")
