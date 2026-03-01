"""
PURPOSE: Phase 3 — Naming semántico de clusters L1/L2.
         Estrategia 1 (primaria): genre voting desde beatport_genre_norm en catalog_success.
         Estrategia 2 (fallback): nombres genéricos A, B, C... / A1, A2...
         Guarda clustering/names_<hash>.json para uso en Phase 4 y 5.
CHANGELOG:
  - 2026-03-01: Creación inicial V4.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.config_loader import load_config
from src.v4.common.path_resolver import resolve_dataset_artifacts


# ---------------------------------------------------------------------------
# Utilidades de naming (inline de legacy/v2/scripts/common/clustering_utils.py)
# ---------------------------------------------------------------------------

def _cluster_to_letter(cluster_id: int) -> str:
    """0 → 'A', 1 → 'B', ... 25 → 'Z', 26 → 'AA', ..."""
    if cluster_id < 0:
        return "Noise"
    letters = []
    n = cluster_id
    while True:
        letters.append(chr(ord("A") + (n % 26)))
        n = n // 26 - 1
        if n < 0:
            break
    return "".join(reversed(letters))


def _simplify_genre_name(genre: str) -> str:
    """'Electronic---Techno' → 'Techno'. Elimina prefijos de categoría."""
    if not genre or genre in ("unknown", "error", "Noise"):
        return str(genre)
    if "---" in genre:
        return genre.split("---", 1)[1]
    return genre


def _top_genres(track_uids: list, catalog: pd.DataFrame, top_n: int = 2) -> str:
    """Devuelve los top_n géneros más frecuentes para un conjunto de track_uids."""
    subset = catalog[catalog["track_uid"].isin(track_uids)]
    col = "beatport_genre_norm" if "beatport_genre_norm" in subset.columns else None
    if col is None:
        return ""
    genres = subset[col].dropna().apply(_simplify_genre_name)
    genres = genres[genres.str.strip() != ""]
    if genres.empty:
        return ""
    counts = genres.value_counts()
    top = counts.head(top_n).index.tolist()
    return " / ".join(top)


# ---------------------------------------------------------------------------
# Lógica principal
# ---------------------------------------------------------------------------

def _find_latest_results(clustering_dir: Path) -> Optional[Path]:
    """Auto-detecta el último results_<hash>.parquet por tiempo de modificación."""
    candidates = sorted(clustering_dir.glob("results_*.parquet"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def run_naming(
    dataset_name: str,
    config: dict,
    config_hash: Optional[str] = None,
) -> Path:
    """
    Genera nombres para cada cluster L1 y L2.

    Returns: Ruta al names_<hash>.json generado.
    """
    artifacts_dir = resolve_dataset_artifacts(dataset_name, config)
    clustering_dir = artifacts_dir / "clustering"
    clustering_dir.mkdir(parents=True, exist_ok=True)

    # Resolver config_hash
    if config_hash is None:
        results_path = _find_latest_results(clustering_dir)
        if results_path is None:
            raise FileNotFoundError(
                f"No results_*.parquet found in {clustering_dir}. "
                "Run phase2_cluster.py first."
            )
        config_hash = results_path.stem.replace("results_", "")
    else:
        results_path = clustering_dir / f"results_{config_hash}.parquet"
        if not results_path.exists():
            raise FileNotFoundError(f"Clustering results not found: {results_path}")

    print(f"[INFO] Loading clustering results: {results_path.name}")
    df = pd.read_parquet(results_path)

    # Cargar catalog_success (N canónico)
    catalog_success_path = artifacts_dir / "catalog_success.parquet"
    if catalog_success_path.exists():
        catalog = pd.read_parquet(catalog_success_path)
    else:
        # Fallback a catalog.parquet (menos robusto, pero no bloquea)
        catalog_path = artifacts_dir / "catalog.parquet"
        catalog = pd.read_parquet(catalog_path) if catalog_path.exists() else pd.DataFrame()
        if catalog.empty:
            print("[WARN] No catalog_success.parquet ni catalog.parquet — usando nombres genéricos")

    # Merge df con catalog para acceder a géneros
    if not catalog.empty and "track_uid" in catalog.columns:
        df_merged = df.merge(catalog[["track_uid"] + [c for c in catalog.columns if c not in ("track_uid",)]],
                             on="track_uid", how="left")
    else:
        df_merged = df.copy()

    names: dict = {}

    # Cluster L1 noise
    names["l1_-1"] = "Noise"

    l1_labels = sorted([lbl for lbl in df["label_l1"].unique() if lbl != -1])

    print("\n" + "=" * 70)
    print(f"{'Cluster':<12} {'Nombre':<30} {'N tracks':>8}")
    print("-" * 52)

    for l1 in l1_labels:
        mask_l1 = df["label_l1"] == l1
        tracks_l1 = df.loc[mask_l1, "track_uid"].tolist()

        # Nombre L1
        genre_name = _top_genres(tracks_l1, catalog) if not catalog.empty else ""
        if genre_name:
            l1_name = genre_name
        else:
            l1_name = f"Group {_cluster_to_letter(l1)}"

        key_l1 = f"l1_{l1}"
        names[key_l1] = l1_name
        print(f"  L1_{_cluster_to_letter(l1):<10} {l1_name:<30} {len(tracks_l1):>8}")

        # Sub-clusters L2
        l2_labels = sorted([lbl for lbl in df.loc[mask_l1, "label_l2"].unique() if lbl != -1])

        if not l2_labels:
            # Cluster L1 pequeño → un único subgrupo trivial
            key_l2 = f"l1_{l1}_l2_0"
            names[key_l2] = f"{_cluster_to_letter(l1)}1"
        else:
            # Ruido L2 dentro del cluster L1
            noise_l2_mask = mask_l1 & (df["label_l2"] == -1)
            if noise_l2_mask.any():
                names[f"l1_{l1}_l2_-1"] = f"{_cluster_to_letter(l1)}_Noise"

            for l2 in l2_labels:
                mask_l2 = mask_l1 & (df["label_l2"] == l2)
                tracks_l2 = df.loc[mask_l2, "track_uid"].tolist()
                genre_l2 = _top_genres(tracks_l2, catalog) if not catalog.empty else ""
                if genre_l2 and genre_l2 != l1_name:
                    l2_name = f"{_cluster_to_letter(l1)}{l2 + 1} {genre_l2}"
                else:
                    l2_name = f"{_cluster_to_letter(l1)}{l2 + 1}"
                key_l2 = f"l1_{l1}_l2_{l2}"
                names[key_l2] = l2_name
                print(f"    L2_{_cluster_to_letter(l1)}{l2 + 1:<8} {l2_name:<28} {len(tracks_l2):>8}")

    print("=" * 70)

    # Guardar names JSON
    names_path = clustering_dir / f"names_{config_hash}.json"
    with open(names_path, "w", encoding="utf-8") as f:
        json.dump(names, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Saved names: {names_path}")
    return names_path


def main() -> int:
    parser = argparse.ArgumentParser(description="TRAKTOR ML V4 — Phase 3: Semantic Naming")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--config-hash", default=None,
                        help="Hash del config de clustering (auto-detect si no se provee)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("[INFO] TRAKTOR ML V4 — Phase 3: Semantic Naming")
    print(f"[INFO] Dataset: {args.dataset_name}")
    print("=" * 70)

    config = load_config(Path(args.config) if args.config else None)
    run_naming(
        dataset_name=args.dataset_name,
        config=config,
        config_hash=args.config_hash,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
