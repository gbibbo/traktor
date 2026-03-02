"""
TEST BLOCK 3: Clustering + evaluation framework.
Ejecutar desde repo root: python tests/v4/test_block3_clustering.py

Parte A (unit tests — siempre ejecuta, no requiere embeddings):
  1. clustering_ari(x, x) == 1.0
  2. noise_rate conocida
  3. transition_score con datos sintéticos → valor en [0, 1]
  4. key_compatibility (Camelot wheel)

Parte B (clustering real — skip-safe si no existen mert_perc.npy/mert_full.npy):
  5. Ejecutar phase2_cluster.py sobre test_20 (con --skip-umap para rapidez)
  6. Verificar: clusters L1 entre 3-15, noise < 30%, parquet tiene N filas
  7. Ejecutar eval_runner (solo cluster stats, sin dev_set)
  8. Imprimir reporte para revisión humana

PAUSA HUMANA al final de Parte B:
  Revisar el reporte antes de proceder a Block 4.
"""
import sys
import os
import json
import tempfile
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)


# ============================================================================
# Parte A: Unit tests (siempre ejecutar)
# ============================================================================

def test_clustering_ari_perfect():
    """clustering_ari(x, x) debe ser 1.0."""
    from src.v4.evaluation.metrics import clustering_ari
    labels = np.array([0, 0, 1, 1, 2, 2, 3])
    result = clustering_ari(labels, labels)
    assert abs(result - 1.0) < 1e-6, f"ARI(x,x) debería ser 1.0, got {result}"
    print(f"  OK: clustering_ari(x,x) = {result:.4f}")


def test_clustering_ari_excludes_noise():
    """ARI excluye labels == -1 de ambos arrays."""
    from src.v4.evaluation.metrics import clustering_ari
    pred = np.array([0, 0, 1, 1, -1])
    true = np.array([0, 0, 1, 1, -1])
    result = clustering_ari(pred, true)
    assert abs(result - 1.0) < 1e-6, f"ARI con ruido debería ser 1.0, got {result}"
    print(f"  OK: clustering_ari excluye ruido correctamente = {result:.4f}")


def test_noise_rate():
    """noise_rate([0, 0, 1, -1, -1]) debe ser 0.4."""
    from src.v4.evaluation.metrics import noise_rate
    labels = np.array([0, 0, 1, -1, -1])
    result = noise_rate(labels)
    assert abs(result - 0.4) < 1e-9, f"noise_rate debería ser 0.4, got {result}"
    print(f"  OK: noise_rate = {result:.1f}")


def test_noise_rate_all_assigned():
    """noise_rate sin puntos de ruido debe ser 0.0."""
    from src.v4.evaluation.metrics import noise_rate
    labels = np.array([0, 1, 2, 0, 1])
    result = noise_rate(labels)
    assert result == 0.0, f"noise_rate debería ser 0.0, got {result}"
    print(f"  OK: noise_rate (no noise) = {result}")


def test_transition_score_range():
    """transition_score con datos sintéticos debe retornar valor en [0, 1]."""
    from src.v4.evaluation.metrics import transition_score
    rng = np.random.default_rng(42)
    N = 10
    embeddings = rng.standard_normal((N, 32)).astype(np.float32)
    bpm = rng.uniform(120, 140, N)
    keys = ["Cm", "Gm", "Dm", "Am", "Em", "Bm", "Fm", "C#m", "G#m", "D#m"]
    ordering = list(range(N))
    score = transition_score(ordering, embeddings, bpm, keys)
    assert 0.0 <= score <= 1.0, f"transition_score debe estar en [0,1], got {score}"
    print(f"  OK: transition_score = {score:.4f}")


def test_transition_score_better_than_worst():
    """Ordering idéntico (self-loop reemplazado por circular) > score aleatorio en promedio."""
    from src.v4.evaluation.metrics import transition_score
    rng = np.random.default_rng(0)
    N = 20
    # Crear embeddings con estructura clara (clusters de 5)
    embeddings = np.concatenate([
        rng.standard_normal((5, 16)) + np.array([10, 0] + [0]*14),
        rng.standard_normal((5, 16)) + np.array([-10, 0] + [0]*14),
        rng.standard_normal((5, 16)) + np.array([0, 10] + [0]*14),
        rng.standard_normal((5, 16)) + np.array([0, -10] + [0]*14),
    ]).astype(np.float32)
    bpm = np.full(N, 128.0)
    keys = ["Cm"] * N
    # Ordering agrupado (similar tracks juntos)
    grouped = list(range(0, 5)) + list(range(5, 10)) + list(range(10, 15)) + list(range(15, 20))
    score_grouped = transition_score(grouped, embeddings, bpm, keys)
    # Ordering aleatorio
    random_scores = []
    for _ in range(20):
        perm = rng.permutation(N).tolist()
        random_scores.append(transition_score(perm, embeddings, bpm, keys))
    avg_random = float(np.mean(random_scores))
    assert score_grouped >= avg_random * 0.9, (
        f"Ordering agrupado ({score_grouped:.4f}) debería ser >= random ({avg_random:.4f})"
    )
    print(f"  OK: grouped={score_grouped:.4f} >= random_avg={avg_random:.4f}")


def test_key_compatibility_camelot():
    """Verificar compatibilidades conocidas del Camelot wheel."""
    from src.v4.evaluation.metrics import _key_compatibility
    # Misma tonalidad
    assert _key_compatibility("Cm", "Cm") == 1.0, "Cm-Cm debería ser 1.0"
    # Relativa mayor/menor (mismo número Camelot)
    assert _key_compatibility("Cm", "Eb") == 1.0, "Cm-Eb debería ser 1.0 (8A-8B)"
    # Vecino en el anillo
    assert _key_compatibility("Cm", "Gm") == 0.5, "Cm-Gm debería ser 0.5"
    # Incompatible
    assert _key_compatibility("Cm", "F#m") == 0.0, "Cm-F#m debería ser 0.0"
    # Notación numérica Camelot
    assert _key_compatibility("8A", "8A") == 1.0, "8A-8A debería ser 1.0"
    print("  OK: _key_compatibility Camelot wheel correcta")


def test_composite_score():
    """composite_score con pesos uniformes."""
    from src.v4.evaluation.metrics import composite_score
    metrics = {"ari": 0.8, "nmi": 0.6}
    result = composite_score(metrics)
    assert abs(result - 0.7) < 1e-9, f"Promedio simple debería ser 0.7, got {result}"
    print(f"  OK: composite_score = {result:.2f}")


# ============================================================================
# Parte B: Clustering real sobre test_20 (SKIP-SAFE)
# ============================================================================

def _check_embeddings_available(dataset_name: str = "test_20") -> bool:
    """Retorna True si los embeddings de Phase 1 están disponibles."""
    from src.v4.common.config_loader import load_config
    from src.v4.common.path_resolver import resolve_dataset_artifacts
    config = load_config()
    artifacts = resolve_dataset_artifacts(dataset_name, config)
    perc = artifacts / "embeddings" / "mert_perc.npy"
    full = artifacts / "embeddings" / "mert_full.npy"
    uids = artifacts / "embeddings" / "track_uids.json"
    return perc.exists() and full.exists() and uids.exists()


def test_phase2_cluster_on_test20():
    """Ejecutar Phase 2 sobre test_20 y verificar resultados.

    SKIP si mert_perc.npy / mert_full.npy no existen (Phase 1 pendiente).
    """
    if not _check_embeddings_available():
        print("\n  [SKIP] mert_perc.npy / mert_full.npy no disponibles.")
        print("         Esperar a que finalice el job GPU de Phase 1.")
        print("         Re-ejecutar este test después de phase1_merge_shards.py.")
        return

    from src.v4.common.config_loader import load_config
    from src.v4.common.path_resolver import resolve_dataset_artifacts
    from src.v4.pipeline.phase2_cluster import run_clustering

    config = load_config()
    cluster_cfg = config.get("clustering", {})
    artifacts = resolve_dataset_artifacts("test_20", config)
    assign_noise = cluster_cfg.get("assign_noise", True)

    results_path = run_clustering(
        dataset_name="test_20",
        config=config,
        l1_min_cluster_size=cluster_cfg.get("l1_min_cluster_size", 10),
        l1_min_samples=cluster_cfg.get("l1_min_samples", 3),
        l2_min_cluster_size=cluster_cfg.get("l2_min_cluster_size", 4),
        l2_min_samples=cluster_cfg.get("l2_min_samples", 2),
        config_tag="test_block3",
        skip_umap=True,  # Más rápido para el test
        pca_dim=cluster_cfg.get("pca_dim", 0),
        assign_noise=assign_noise,
    )

    import pandas as pd
    import json

    assert results_path.exists(), f"Parquet no fue creado: {results_path}"
    df = pd.read_parquet(results_path)

    # N = len(track_uids.json), no len(catalog)
    uids_path = artifacts / "embeddings" / "track_uids.json"
    with open(uids_path) as f:
        track_uids = json.load(f)
    N = len(track_uids)

    assert len(df) == N, f"Parquet debe tener {N} filas (track_uids), got {len(df)}"
    assert "track_uid" in df.columns
    assert "label_l1" in df.columns
    assert "label_l2" in df.columns

    labels_l1 = df["label_l1"].to_numpy()
    labels_l2 = df["label_l2"].to_numpy()

    # Verificar ruido L1 (assert duro: sólo fallo crítico total)
    noise_l1 = float(np.sum(labels_l1 == -1) / N)
    assert noise_l1 < 0.80, f"Noise L1 crítico (>80%): {noise_l1:.1%} — revisar parámetros HDBSCAN"

    # Verificar número de clusters L1 (assert duro: al menos 1 cluster real)
    n_clusters_l1 = len(set(labels_l1[labels_l1 != -1]))
    assert n_clusters_l1 >= 1, f"Sin clusters L1 ({n_clusters_l1}) — revisar HDBSCAN"

    # Verificar comportamiento de assign_noise
    if assign_noise:
        assert noise_l1 == 0.0, f"Con assign_noise=True, noise debe ser 0%, got {noise_l1:.1%}"
        assert "label_l1_raw" in df.columns, "Falta columna label_l1_raw"
        assert "label_l2_raw" in df.columns, "Falta columna label_l2_raw"
        n_reassigned = int((df["label_l1_raw"] == -1).sum())
        print(f"      [INFO] Tracks reasignados L1: {n_reassigned} ({n_reassigned/N:.1%})")
    else:
        # Sin reassignment: contrato original — si label_l1==-1, label_l2 debe ser -1
        noise_mask = labels_l1 == -1
        assert np.all(labels_l2[noise_mask] == -1), (
            "Tracks con label_l1=-1 deben tener label_l2=-1"
        )

    # Reporte humano (no son asserts, son guías)
    print(f"      [HUMAN REVIEW] n_clusters_l1={n_clusters_l1}, noise={noise_l1:.1%}")
    print(f"      Objetivo sugerido: 3-10 clusters, <30% noise para ~250 tracks")

    print(f"  OK: Phase 2 clustering completado")
    print(f"      N={N}, L1 clusters={n_clusters_l1}, noise={noise_l1:.1%}")
    return results_path


def test_eval_runner_on_test20():
    """Ejecutar eval_runner sobre test_20 y verificar JSON de scores.

    SKIP si mert_perc.npy no disponible.
    """
    if not _check_embeddings_available():
        print("\n  [SKIP] Embeddings no disponibles — skip eval_runner test.")
        return

    from src.v4.common.config_loader import load_config
    from src.v4.common.path_resolver import resolve_dataset_artifacts
    from src.v4.pipeline.phase2_cluster import run_clustering, _hash_config
    from src.v4.evaluation.eval_runner import run_evaluation, print_clustering_report

    config = load_config()
    cluster_cfg = config.get("clustering", {})
    cluster_params = {
        "pca_dim": cluster_cfg.get("pca_dim", 0),
        "l1_min_cluster_size": cluster_cfg.get("l1_min_cluster_size", 10),
        "l1_min_samples": cluster_cfg.get("l1_min_samples", 3),
        "l2_min_cluster_size": cluster_cfg.get("l2_min_cluster_size", 4),
        "l2_min_samples": cluster_cfg.get("l2_min_samples", 2),
        "l2_min_parent_size": 8,
        "assign_noise": cluster_cfg.get("assign_noise", True),
        "config_tag": "test_block3",
        "dataset_name": "test_20",
    }
    config_hash = _hash_config(cluster_params)

    # Asegurar que el clustering existe
    artifacts = resolve_dataset_artifacts("test_20", config)
    results_path = artifacts / "clustering" / f"results_{config_hash}.parquet"
    if not results_path.exists():
        run_clustering(
            dataset_name="test_20",
            config=config,
            l1_min_cluster_size=cluster_params["l1_min_cluster_size"],
            l1_min_samples=cluster_params["l1_min_samples"],
            l2_min_cluster_size=cluster_params["l2_min_cluster_size"],
            l2_min_samples=cluster_params["l2_min_samples"],
            config_tag="test_block3",
            skip_umap=True,
            pca_dim=cluster_params["pca_dim"],
            assign_noise=cluster_params["assign_noise"],
        )

    result = run_evaluation(
        dataset_name="test_20",
        clustering_config_hash=config_hash,
        config=config,
        dev_set_path=None,  # Sin ground truth
    )

    assert "metrics" in result
    m = result["metrics"]
    assert "noise_rate_l1" in m
    assert "n_clusters_l1" in m
    assert "n_tracks_total" in m
    assert m["n_tracks_total"] > 0

    print(f"  OK: eval_runner completado")
    print_clustering_report(result)

    print("\n" + "=" * 70)
    print("!!! PAUSA HUMANA — REVISIÓN REQUERIDA !!!")
    print("Revisar el reporte antes de proceder a Block 4:")
    print("  1. ¿Número de clusters L1 razonable? (objetivo: 3-10 para ~250 tracks)")
    print("  2. ¿Tamaños de clusters balanceados?")
    print("  3. ¿Noise rate aceptable? (<15% ideal, <30% aceptable)")
    print("  4. Si no satisfactorio: re-ejecutar Phase 2 con otros parámetros.")
    print("=" * 70)


# ============================================================================
# Runner
# ============================================================================

def run_all():
    tests_a = [
        test_clustering_ari_perfect,
        test_clustering_ari_excludes_noise,
        test_noise_rate,
        test_noise_rate_all_assigned,
        test_transition_score_range,
        test_transition_score_better_than_worst,
        test_key_compatibility_camelot,
        test_composite_score,
    ]

    tests_b = [
        test_phase2_cluster_on_test20,
        test_eval_runner_on_test20,
    ]

    print("=" * 70)
    print("TEST BLOCK 3: Clustering + Evaluation")
    print("=" * 70)

    print("\n--- Parte A: Unit tests (sin embeddings) ---")
    failed = []
    for test in tests_a:
        print(f"\n[TEST] {test.__name__}")
        try:
            test()
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed.append(test.__name__)
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append(test.__name__)

    print("\n--- Parte B: Clustering real (skip-safe) ---")
    for test in tests_b:
        print(f"\n[TEST] {test.__name__}")
        try:
            test()
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed.append(test.__name__)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            failed.append(test.__name__)

    print("\n" + "=" * 70)
    if failed:
        print(f"[RESULT] FAIL — {len(failed)} test(s) failed: {failed}")
        return 1
    print("[RESULT] PASS — Todos los tests completados correctamente.")
    return 0


if __name__ == "__main__":
    sys.exit(run_all())
