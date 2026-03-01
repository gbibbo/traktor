"""
TEST BLOCK 4: Export pipeline — Phase 3 (naming) + Phase 4 (ordering) + Phase 5 (M3U export).
Ejecutar desde repo root: python tests/v4/test_block4_export.py

Parte A (automática):
  1. Importar los 3 scripts sin errores.
  2. Phase 3: names_<hash>.json existe, ningún nombre es vacío.
  3. Phase 4: ordered_<hash>.parquet existe, columna 'position' presente.
  4. Phase 5: directorio playlists/V4_<N>/ existe, ≥1 M3U, total == N canónico.
  5. Transition score promedio >= baseline aleatorio (seed=0, 25 permutaciones).
  6. _summary.txt existe.

Parte B (revisión humana):
  Imprime quality report con primeros 5 tracks por playlist L2.

PAUSA HUMANA al final de Parte B:
  Revisar ordering antes de usar playlists en Traktor.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)


# ============================================================================
# Helpers
# ============================================================================

def _get_N_canonical(artifacts_dir: Path) -> int:
    """N canónico: len(track_uids.json)."""
    uids_path = artifacts_dir / "embeddings" / "track_uids.json"
    if not uids_path.exists():
        raise FileNotFoundError(f"track_uids.json not found: {uids_path}")
    with open(uids_path) as f:
        return len(json.load(f))


def _check_embeddings_available(dataset_name: str = "test_20") -> bool:
    from src.v4.common.config_loader import load_config
    from src.v4.common.path_resolver import resolve_dataset_artifacts
    config = load_config()
    artifacts = resolve_dataset_artifacts(dataset_name, config)
    return (
        (artifacts / "embeddings" / "mert_perc.npy").exists()
        and (artifacts / "embeddings" / "mert_full.npy").exists()
        and (artifacts / "embeddings" / "track_uids.json").exists()
    )


def _check_clustering_available(dataset_name: str = "test_20") -> bool:
    from src.v4.common.config_loader import load_config
    from src.v4.common.path_resolver import resolve_dataset_artifacts
    config = load_config()
    artifacts = resolve_dataset_artifacts(dataset_name, config)
    clustering_dir = artifacts / "clustering"
    return any(clustering_dir.glob("results_*.parquet")) if clustering_dir.exists() else False


# ============================================================================
# Parte A: Tests automáticos
# ============================================================================

def test_imports():
    """Los 3 scripts importan sin errores."""
    import src.v4.pipeline.phase3_name  # noqa
    import src.v4.pipeline.phase4_order  # noqa
    import src.v4.pipeline.phase5_export  # noqa
    print("  OK: phase3_name, phase4_order, phase5_export importan correctamente")


def test_phase3_naming():
    """Phase 3: names_<hash>.json existe, ningún nombre es vacío o None."""
    if not _check_clustering_available():
        print("  [SKIP] No hay clustering results — Phase 3 no puede ejecutarse aún")
        return

    from src.v4.common.config_loader import load_config
    from src.v4.pipeline.phase3_name import run_naming

    config = load_config()
    names_path = run_naming(dataset_name="test_20", config=config)

    assert names_path.exists(), f"names JSON no encontrado: {names_path}"

    with open(names_path, encoding="utf-8") as f:
        names = json.load(f)

    assert len(names) > 0, "names JSON está vacío"
    for key, name in names.items():
        assert name and str(name).strip(), f"Nombre vacío para clave {key!r}"

    print(f"  OK: Phase 3 — {len(names)} nombres generados, ninguno vacío")
    return names_path


def test_phase4_ordering():
    """Phase 4: ordered_<hash>.parquet existe, columna 'position' presente, sin NaN."""
    if not _check_clustering_available() or not _check_embeddings_available():
        print("  [SKIP] No hay clustering o embeddings — Phase 4 no puede ejecutarse aún")
        return

    import pandas as pd
    from src.v4.common.config_loader import load_config
    from src.v4.pipeline.phase4_order import run_ordering

    config = load_config()
    ordered_path = run_ordering(dataset_name="test_20", config=config)

    assert ordered_path.exists(), f"ordered parquet no encontrado: {ordered_path}"
    df = pd.read_parquet(ordered_path)

    assert "position" in df.columns, "Columna 'position' no está en ordered parquet"

    # Verificar que tracks en clusters (no noise) tienen posición asignada
    clustered = df[df["label_l2"] >= 0]
    if not clustered.empty:
        assert clustered["position"].notna().all(), \
            f"Tracks en clusters con position NaN: {clustered['position'].isna().sum()}"

    print(f"  OK: Phase 4 — ordered_<hash>.parquet con {len(df)} filas, columna 'position' presente")
    return ordered_path


def test_phase5_export():
    """Phase 5: playlists generadas, total tracks == N canónico, M3U válidos."""
    if not _check_clustering_available():
        print("  [SKIP] No hay clustering results — Phase 5 no puede ejecutarse aún")
        return

    import pandas as pd
    from src.v4.common.config_loader import load_config
    from src.v4.common.path_resolver import resolve_dataset_artifacts
    from src.v4.pipeline.phase5_export import run_export

    config = load_config()
    artifacts_dir = resolve_dataset_artifacts("test_20", config)
    N = _get_N_canonical(artifacts_dir)

    out_dir = run_export(
        dataset_name="test_20",
        config=config,
        windows_audio_dir=r"C:\Música\test_export",  # Dir de prueba (no real)
    )

    assert out_dir.exists(), f"Directorio de playlists no creado: {out_dir}"

    m3u_files = list(out_dir.rglob("*.m3u"))
    assert len(m3u_files) >= 1, "No se generó ningún M3U"

    # Verificar header EXTM3U y contar tracks totales
    total_tracks = 0
    for m3u_path in m3u_files:
        content = m3u_path.read_text(encoding="utf-8")
        assert content.startswith("#EXTM3U"), f"{m3u_path.name} no empieza con #EXTM3U"
        # Contar líneas no-header, no-EXTINF
        track_lines = [l for l in content.splitlines() if l.strip() and not l.startswith("#")]
        total_tracks += len(track_lines)

    assert total_tracks == N, \
        f"Total tracks en M3U ({total_tracks}) != N canónico ({N})"

    # Verificar _summary.txt
    summary_path = out_dir / "_summary.txt"
    assert summary_path.exists(), "_summary.txt no encontrado"
    summary_content = summary_path.read_text(encoding="utf-8")
    non_header_lines = [l for l in summary_content.splitlines()
                        if l.strip() and not l.startswith("=") and not l.startswith("-")]
    assert len(non_header_lines) >= 2, "_summary.txt parece estar vacío"

    print(f"  OK: Phase 5 — {len(m3u_files)} M3U files, {total_tracks}/{N} tracks, _summary.txt OK")
    return out_dir


def test_transition_score_vs_random():
    """Transition score del ordering real >= baseline aleatorio (seed=0, 25 perms)."""
    if not _check_clustering_available() or not _check_embeddings_available():
        print("  [SKIP] No hay clustering/embeddings para test de transition score")
        return

    import pandas as pd
    from src.v4.common.config_loader import load_config
    from src.v4.common.path_resolver import resolve_dataset_artifacts
    from src.v4.evaluation.metrics import transition_score
    from src.v4.pipeline.phase4_order import essentia_to_camelot

    config = load_config()
    artifacts_dir = resolve_dataset_artifacts("test_20", config)

    # Cargar ordered parquet (auto-detect)
    clustering_dir = artifacts_dir / "clustering"
    candidates = sorted(clustering_dir.glob("ordered_*.parquet"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        print("  [SKIP] No ordered parquet — ejecutar test_phase4_ordering primero")
        return

    df = pd.read_parquet(candidates[-1])
    bpm_df = pd.read_parquet(artifacts_dir / "features" / "bpm_key.parquet")
    bpm_df = bpm_df.set_index("track_uid")

    mert_full = np.load(artifacts_dir / "embeddings" / "mert_full.npy")

    with open(artifacts_dir / "embeddings" / "track_uids.json") as f:
        uid_list = json.load(f)
    uid_to_idx = {uid: i for i, uid in enumerate(uid_list)}

    real_scores = []
    random_scores = []
    rng = np.random.default_rng(0)

    # Analizar todos los L2 subclusters con >= 3 tracks
    l1_labels = [l for l in df["label_l1"].unique() if l != -1]
    for l1 in sorted(l1_labels):
        mask_l1 = df["label_l1"] == l1
        l2_labels = [l for l in df.loc[mask_l1, "label_l2"].unique() if l >= 0]
        for l2 in sorted(l2_labels):
            mask_l2 = mask_l1 & (df["label_l2"] == l2)
            sub = df[mask_l2].sort_values("position")
            if len(sub) < 3:
                continue
            uids = sub["track_uid"].tolist()
            idxs = [uid_to_idx[u] for u in uids if u in uid_to_idx]
            if len(idxs) < 3:
                continue

            emb = mert_full[idxs]
            bpms = np.array([bpm_df.loc[u, "bpm"] if u in bpm_df.index else 128.0 for u in uids[:len(idxs)]])
            keys = [essentia_to_camelot(str(bpm_df.loc[u, "key"]) if u in bpm_df.index else "?")
                    for u in uids[:len(idxs)]]

            ordered_perm = list(range(len(idxs)))
            real_scores.append(transition_score(ordered_perm, emb, bpms, keys))

            for _ in range(25):
                perm = rng.permutation(len(idxs)).tolist()
                random_scores.append(transition_score(perm, emb, bpms, keys))

    if not real_scores:
        print("  [SKIP] No hay subclusters con ≥3 tracks para test de ordering")
        return

    real_mean = float(np.mean(real_scores))
    baseline_mean = float(np.mean(random_scores))

    assert real_mean >= baseline_mean - 1e-3, (
        f"Ordering real ({real_mean:.4f}) < baseline aleatorio ({baseline_mean:.4f})"
    )
    print(f"  OK: transition_score real={real_mean:.4f} >= baseline={baseline_mean:.4f} (25 perms, seed=0)")


# ============================================================================
# Parte B: Reporte humano de calidad
# ============================================================================

def print_quality_report():
    """Imprime primeros 5 tracks por playlist L2 con BPM y key."""
    if not _check_clustering_available():
        print("  [SKIP] No hay playlists generadas para el reporte")
        return

    import pandas as pd
    from src.v4.common.config_loader import load_config
    from src.v4.common.path_resolver import resolve_dataset_artifacts
    from src.v4.pipeline.phase4_order import essentia_to_camelot

    config = load_config()
    artifacts_dir = resolve_dataset_artifacts("test_20", config)
    clustering_dir = artifacts_dir / "clustering"

    candidates = sorted(clustering_dir.glob("ordered_*.parquet"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        print("  [SKIP] No ordered parquet disponible")
        return

    df = pd.read_parquet(candidates[-1])
    bpm_df = pd.read_parquet(artifacts_dir / "features" / "bpm_key.parquet").set_index("track_uid")

    catalog_success = None
    cs_path = artifacts_dir / "catalog_success.parquet"
    if cs_path.exists():
        catalog_success = pd.read_parquet(cs_path).set_index("track_uid")

    print("\n" + "=" * 70)
    print("=== QUALITY REPORT — REVISIÓN HUMANA ===")
    print("=" * 70)

    for l1 in sorted([l for l in df["label_l1"].unique() if l != -1]):
        mask_l1 = df["label_l1"] == l1
        for l2 in sorted([l for l in df.loc[mask_l1, "label_l2"].unique() if l >= 0]):
            mask_l2 = mask_l1 & (df["label_l2"] == l2)
            sub = df[mask_l2].sort_values("position")
            n = len(sub)
            bpms = [bpm_df.loc[u, "bpm"] if u in bpm_df.index else float("nan")
                    for u in sub["track_uid"]]
            keys = [essentia_to_camelot(str(bpm_df.loc[u, "key"]) if u in bpm_df.index else "?")
                    for u in sub["track_uid"]]

            bpm_arr = np.array([b for b in bpms if not np.isnan(b)])
            bpm_range = f"{bpm_arr.min():.0f}–{bpm_arr.max():.0f}" if len(bpm_arr) > 0 else "N/A"

            print(f"\nL1={l1} L2={l2} | {n} tracks | BPM {bpm_range}")
            print("-" * 50)

            for i, (_, row) in enumerate(sub.head(5).iterrows()):
                uid = row["track_uid"]
                bpm_val = bpms[i] if i < len(bpms) else float("nan")
                key_val = keys[i] if i < len(keys) else "?"
                if catalog_success is not None and uid in catalog_success.index:
                    artist = catalog_success.loc[uid].get("artist", "?")
                    title = catalog_success.loc[uid].get("title", "?")
                    name = f"{artist} - {title}"
                else:
                    name = row.get("filename", uid[:8])
                bpm_str = f"{bpm_val:.1f}" if not np.isnan(bpm_val) else "?"
                print(f"  {i+1}. {str(name)[:45]:<45} BPM={bpm_str} Key={key_val}")

            if n > 5:
                print(f"  ... ({n - 5} tracks más)")

    print("\n" + "=" * 70)
    print("!!! PAUSA HUMANA — REVISIÓN REQUERIDA !!!")
    print("Revisar antes de importar playlists en Traktor:")
    print("  1. ¿BPM ranges coherentes dentro de cada playlist?")
    print("  2. ¿Ordering tiene sentido (BPMs cercanos consecutivos, keys compatibles)?")
    print("  3. Opcionalmente: escuchar transiciones en 2-3 playlists.")
    print("=" * 70)


# ============================================================================
# Runner
# ============================================================================

def run_all():
    tests_a = [
        test_imports,
        test_phase3_naming,
        test_phase4_ordering,
        test_phase5_export,
        test_transition_score_vs_random,
    ]

    print("=" * 70)
    print("TEST BLOCK 4: Export pipeline (Naming + Ordering + M3U Export)")
    print("=" * 70)

    print("\n--- Parte A: Tests automáticos ---")
    failed = []
    for test in tests_a:
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

    print("\n--- Parte B: Reporte humano de calidad ---")
    try:
        print_quality_report()
    except Exception as e:
        print(f"  [WARN] Error en quality report: {e}")

    print("\n" + "=" * 70)
    if failed:
        print(f"[RESULT] FAIL — {len(failed)} test(s) failed: {failed}")
        return 1
    print("[RESULT] PASS — Todos los tests completados correctamente.")
    return 0


if __name__ == "__main__":
    sys.exit(run_all())
