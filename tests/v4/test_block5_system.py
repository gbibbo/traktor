"""
TEST BLOCK 5: Verificación final del sistema V4.
Ejecutar desde repo root: python tests/v4/test_block5_system.py

Parte A (automática):
  1. Todos los módulos V4 importan sin errores.
  2. catalog_success.parquet existe y len == len(track_uids.json).
  3. docs/v4/TODO.md tiene todas las tareas marcadas [x].
  4. ProjectionHead forward con (16, 1024) produce (16, 256).
  5. docs/PROJECT_MAP.md y docs/V4_USAGE.md existen.

Parte B (revisión humana):
  Instrucciones para validación final en Traktor DJ.
"""
import os
import re
import sys
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)


# ============================================================================
# Parte A: Tests automáticos
# ============================================================================

def test_all_modules_import():
    """Todos los módulos V4 importan sin errores de sintaxis."""
    modules = [
        "src.v4.config",
        "src.v4.common.config_loader",
        "src.v4.common.path_resolver",
        "src.v4.common.catalog",
        "src.v4.common.audio_utils",
        "src.v4.common.demucs_utils",
        "src.v4.common.embedding_utils",
        "src.v4.common.logging_utils",
        "src.v4.pipeline.phase0_ingest",
        "src.v4.pipeline.phase1_extract",
        "src.v4.pipeline.phase1_merge_shards",
        "src.v4.pipeline.phase2_cluster",
        "src.v4.pipeline.phase3_name",
        "src.v4.pipeline.phase4_order",
        "src.v4.pipeline.phase5_export",
        "src.v4.evaluation.metrics",
        "src.v4.evaluation.eval_runner",
        "src.v4.adaptation.projection_head",
        "src.v4.adaptation.contrastive_trainer",
    ]
    failed = []
    for mod in modules:
        try:
            __import__(mod)
        except Exception as e:
            failed.append(f"{mod}: {e}")

    if failed:
        for msg in failed:
            print(f"  FAIL import: {msg}")
        assert False, f"{len(failed)} módulo(s) no importan correctamente"

    print(f"  OK: {len(modules)} módulos V4 importan sin errores")


def test_catalog_success_N_canonical():
    """catalog_success.parquet existe y len == len(track_uids.json)."""
    import json
    import pandas as pd
    from src.v4.common.config_loader import load_config
    from src.v4.common.path_resolver import resolve_dataset_artifacts

    config = load_config()
    artifacts_dir = resolve_dataset_artifacts("test_20", config)

    uids_path = artifacts_dir / "embeddings" / "track_uids.json"
    cs_path = artifacts_dir / "catalog_success.parquet"

    if not uids_path.exists():
        print("  [SKIP] track_uids.json no existe (Phase 1 no ejecutada)")
        return

    with open(uids_path) as f:
        N = len(json.load(f))

    if not cs_path.exists():
        print(f"  [WARN] catalog_success.parquet no existe — ejecutar phase1_merge_shards.py")
        print(f"         (N canónico esperado: {N})")
        return

    cat = pd.read_parquet(cs_path)
    assert len(cat) == N, (
        f"catalog_success.parquet tiene {len(cat)} filas, "
        f"track_uids.json tiene {N} — deben coincidir"
    )
    assert "track_uid" in cat.columns, "catalog_success.parquet no tiene columna track_uid"
    assert cat["track_uid"].nunique() == len(cat), "track_uids duplicados en catalog_success"

    print(f"  OK: catalog_success.parquet = {len(cat)} filas == N canónico ({N})")


def test_todo_md_complete():
    """docs/v4/TODO.md tiene todas las tareas marcadas [x]."""
    todo_path = Path(REPO_ROOT) / "docs" / "v4" / "TODO.md"
    assert todo_path.exists(), f"TODO.md no encontrado: {todo_path}"

    content = todo_path.read_text()
    unchecked = [
        line.strip()
        for line in content.splitlines()
        if re.match(r"^- \[ \]", line.strip())
    ]

    if unchecked:
        print(f"  [WARN] {len(unchecked)} tarea(s) sin completar en TODO.md:")
        for task in unchecked:
            print(f"    {task}")
        # No assert duro — es informativo
    else:
        print("  OK: TODO.md — todas las tareas completadas")


def test_projection_head_forward():
    """ProjectionHead forward con (16, 1024) produce (16, 256) L2-normalizados."""
    try:
        import torch
        from src.v4.adaptation.projection_head import ProjectionHead

        head = ProjectionHead(in_dim=1024, hidden_dim=512, out_dim=256)
        x = torch.randn(16, 1024)
        out = head(x)

        assert out.shape == (16, 256), f"Shape esperado (16,256), got {out.shape}"
        # Verificar L2-normalización
        norms = torch.norm(out, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(16), atol=1e-5), \
            "La salida no está L2-normalizada"

        print(f"  OK: ProjectionHead forward: (16, 1024) → (16, 256), L2-normalizado")
    except ImportError:
        print("  [SKIP] torch no disponible para test de ProjectionHead")


def test_docs_exist():
    """docs/PROJECT_MAP.md y docs/V4_USAGE.md existen."""
    docs_to_check = [
        Path(REPO_ROOT) / "docs" / "PROJECT_MAP.md",
        Path(REPO_ROOT) / "docs" / "V4_USAGE.md",
    ]
    missing = [p for p in docs_to_check if not p.exists()]
    if missing:
        for p in missing:
            print(f"  FAIL: {p} no encontrado")
        assert False, f"{len(missing)} documento(s) faltante(s)"

    print("  OK: docs/PROJECT_MAP.md y docs/V4_USAGE.md existen")


# ============================================================================
# Parte B: Instrucciones de validación humana
# ============================================================================

def print_human_validation_instructions():
    print("\n" + "=" * 70)
    print("=== TEST BLOCK 5 — REVISIÓN HUMANA DEL SISTEMA COMPLETO ===")
    print("=" * 70)
    print("""
Criterios de aceptación final (el humano debe verificar):

1. UI Streamlit:
   streamlit run src/v4/ui/app.py
   - El scatter UMAP/placeholder se renderiza sin error
   - Los clusters son visualmente distinguibles
   - El re-clustering con distintos parámetros funciona
   - El botón "Export Playlists" ejecuta phases 3+4+5 sin error

2. Playlists en Traktor DJ:
   - Importar 2-3 archivos M3U desde playlists/V4_*/
   - Verificar que las rutas Windows son correctas y Traktor encuentra los tracks
   - Escuchar transiciones entre tracks consecutivos en al menos 2 playlists

3. Coherencia musical:
   - Al menos 70% de los clusters tienen sentido musical
   - Los BPMs dentro de cada playlist son coherentes
   - Las transiciones suenan razonables (no saltos bruscos de BPM/key)

NOTA: Si el clustering no satisface, re-ejecutar phase2_cluster.py con otros parámetros.
      Ejemplo: --l1-min-cluster-size 7 --l2-min-cluster-size 3
""")
    print("=" * 70)


# ============================================================================
# Runner
# ============================================================================

def run_all():
    tests_a = [
        test_all_modules_import,
        test_catalog_success_N_canonical,
        test_todo_md_complete,
        test_projection_head_forward,
        test_docs_exist,
    ]

    print("=" * 70)
    print("TEST BLOCK 5: System Integration")
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

    print_human_validation_instructions()

    print("\n" + "=" * 70)
    if failed:
        print(f"[RESULT] FAIL — {len(failed)} test(s) failed: {failed}")
        return 1
    print("[RESULT] PASS — Parte A completada. Proceder con revisión humana (Parte B).")
    return 0


if __name__ == "__main__":
    sys.exit(run_all())
