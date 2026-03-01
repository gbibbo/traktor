"""
PURPOSE: Phase 5 — Exportar playlists M3U compatibles con Traktor DJ.
         Estructura de output: playlists/V4_<N>/ con carpetas por cluster L1
         y archivos M3U por sub-cluster L2. Rutas Windows para uso directo en Traktor.
         Total de tracks exportados siempre == N canónico (track_uids.json).
CHANGELOG:
  - 2026-03-01: Creación inicial V4.
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.config_loader import load_config
from src.v4.common.path_resolver import resolve_dataset_artifacts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_version_dir(base_dir: Path) -> Path:
    """Auto-incrementa V4_1, V4_2, ... en base_dir."""
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = []
    for item in base_dir.iterdir():
        m = re.match(r"^V4_(\d+)$", item.name)
        if m and item.is_dir():
            existing.append(int(m.group(1)))
    n = max(existing) + 1 if existing else 1
    return base_dir / f"V4_{n}"


def _sanitize_dirname(name: str) -> str:
    """Convierte nombre de cluster en nombre de directorio seguro."""
    safe = re.sub(r'[\\/:*?"<>|]', "_", name)
    return safe.strip()[:50]  # Max 50 chars


def _track_label(row: pd.Series) -> str:
    """Genera label EXTINF: 'Artist - Title' o filename stem como fallback."""
    artist = row.get("artist", None)
    title = row.get("title", None)
    if pd.notna(artist) and pd.notna(title) and str(artist).strip() and str(title).strip():
        return f"{artist} - {title}"
    filename = row.get("filename", "Unknown")
    return str(Path(filename).stem)


def _windows_path(windows_audio_dir: str, filename: str) -> str:
    """Construye ruta Windows absoluta: backslash, sin usar Path de Linux."""
    base = windows_audio_dir.rstrip("/\\")
    return f"{base}\\{filename}"


def _write_m3u(m3u_path: Path, tracks: pd.DataFrame, windows_audio_dir: str) -> None:
    """Escribe un archivo M3U con header EXTM3U. UTF-8."""
    with open(m3u_path, "w", encoding="utf-8") as f:
        f.write("#EXTM3U\n")
        for _, row in tracks.iterrows():
            label = _track_label(row)
            filename = row.get("filename", "")
            wpath = _windows_path(windows_audio_dir, filename)
            f.write(f"#EXTINF:-1,{label}\n")
            f.write(f"{wpath}\n")


def _find_latest_ordered(clustering_dir: Path) -> Optional[Path]:
    candidates = sorted(clustering_dir.glob("ordered_*.parquet"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _find_latest_names(clustering_dir: Path, config_hash: str) -> Optional[Path]:
    p = clustering_dir / f"names_{config_hash}.json"
    return p if p.exists() else None


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_export(
    dataset_name: str,
    config: dict,
    config_hash: Optional[str] = None,
    windows_audio_dir: Optional[str] = None,
) -> Path:
    """
    Genera playlists M3U desde ordered_<hash>.parquet y names_<hash>.json.

    Returns: Ruta al directorio playlists/V4_<N>/ generado.
    """
    artifacts_dir = resolve_dataset_artifacts(dataset_name, config)
    clustering_dir = artifacts_dir / "clustering"
    embeddings_dir = artifacts_dir / "embeddings"
    playlists_base = REPO_ROOT / "playlists"

    # Windows audio dir desde config o CLI override
    if windows_audio_dir is None:
        windows_audio_dir = (
            config.get("paths", {}).get("local_windows_audio_dir")
            or r"C:\Música\2020 new - copia"
        )

    # Resolver config_hash
    if config_hash is None:
        ordered_path = _find_latest_ordered(clustering_dir)
        if ordered_path is None:
            raise FileNotFoundError(
                f"No ordered_*.parquet found in {clustering_dir}. "
                "Run phase4_order.py first."
            )
        config_hash = ordered_path.stem.replace("ordered_", "")
    else:
        ordered_path = clustering_dir / f"ordered_{config_hash}.parquet"
        if not ordered_path.exists():
            raise FileNotFoundError(f"Ordered results not found: {ordered_path}")

    names_path = _find_latest_names(clustering_dir, config_hash)
    if names_path is None:
        print("[WARN] names_<hash>.json not found — using generic names")
        names = {}
    else:
        with open(names_path, encoding="utf-8") as f:
            names = json.load(f)

    print(f"[INFO] Loading ordered results: {ordered_path.name}")
    df = pd.read_parquet(ordered_path)

    # Cargar catalog_success para metadata (artist/title/filename)
    catalog_success_path = artifacts_dir / "catalog_success.parquet"
    if catalog_success_path.exists():
        catalog = pd.read_parquet(catalog_success_path)
    else:
        catalog_path = artifacts_dir / "catalog.parquet"
        catalog = pd.read_parquet(catalog_path) if catalog_path.exists() else pd.DataFrame()

    if not catalog.empty and "track_uid" in catalog.columns:
        df = df.merge(
            catalog[["track_uid", "filename"] +
                    [c for c in catalog.columns if c in ("artist", "title")]],
            on="track_uid", how="left"
        )

    # N canónico = len(track_uids.json)
    uids_path = embeddings_dir / "track_uids.json"
    with open(uids_path) as f:
        N_canonical = len(json.load(f))

    # Crear directorio de output versionado
    out_dir = _next_version_dir(playlists_base)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {out_dir}")

    total_exported = 0
    summary_lines = []

    # Tracks de ruido L1 → All_Noise.m3u
    noise_df = df[df["label_l1"] == -1].copy()
    if not noise_df.empty:
        noise_path = out_dir / "All_Noise.m3u"
        _write_m3u(noise_path, noise_df, windows_audio_dir)
        total_exported += len(noise_df)
        summary_lines.append(f"Noise        | All_Noise.m3u                     | {len(noise_df):>4} tracks")
        print(f"  [OK] All_Noise.m3u ({len(noise_df)} tracks)")

    l1_labels = sorted([l for l in df["label_l1"].unique() if l != -1])

    for l1 in l1_labels:
        l1_key = f"l1_{l1}"
        l1_name = names.get(l1_key, f"Group_{l1}")
        l1_letter = _cluster_to_letter_export(l1)
        l1_dirname = f"L1_{l1_letter}_{_sanitize_dirname(l1_name)}"
        l1_dir = out_dir / l1_dirname
        l1_dir.mkdir(exist_ok=True)

        mask_l1 = df["label_l1"] == l1
        l2_labels = sorted([l for l in df.loc[mask_l1, "label_l2"].unique() if l >= 0])

        # Tracks L2-noise dentro de L1 → L1_X_Noise.m3u
        noise_l2 = df[mask_l1 & (df["label_l2"] == -1)].copy()
        if not noise_l2.empty:
            if noise_l2["position"].max() > 0:
                noise_l2 = noise_l2.sort_values("position")
            fn = f"L2_{l1_letter}_Noise.m3u"
            _write_m3u(l1_dir / fn, noise_l2, windows_audio_dir)
            total_exported += len(noise_l2)
            summary_lines.append(f"L1_{l1_letter} Noise  | {fn:<35} | {len(noise_l2):>4} tracks")

        # L2 subclusters triviales (label_l2 == 0, cluster L1 pequeño)
        if not l2_labels:
            # Toda el cluster L1 como un único playlist
            tracks_l1 = df[mask_l1].copy()
            if tracks_l1["position"].max() > 0:
                tracks_l1 = tracks_l1.sort_values("position")
            l2_name = names.get(f"l1_{l1}_l2_0", f"{l1_letter}1")
            fn = f"L2_{_sanitize_dirname(l2_name)}.m3u"
            _write_m3u(l1_dir / fn, tracks_l1, windows_audio_dir)
            total_exported += len(tracks_l1)
            summary_lines.append(f"L1_{l1_letter} L2_1    | {fn:<35} | {len(tracks_l1):>4} tracks")
            print(f"  [OK] {l1_dirname}/{fn} ({len(tracks_l1)} tracks)")
            continue

        for l2 in l2_labels:
            l2_key = f"l1_{l1}_l2_{l2}"
            l2_name = names.get(l2_key, f"{l1_letter}{l2 + 1}")
            mask_l2 = mask_l1 & (df["label_l2"] == l2)
            tracks_l2 = df[mask_l2].copy()
            if tracks_l2["position"].max() > 0:
                tracks_l2 = tracks_l2.sort_values("position")
            fn = f"L2_{_sanitize_dirname(l2_name)}.m3u"
            _write_m3u(l1_dir / fn, tracks_l2, windows_audio_dir)
            total_exported += len(tracks_l2)
            summary_lines.append(f"L1_{l1_letter}        | {fn:<35} | {len(tracks_l2):>4} tracks")
            print(f"  [OK] {l1_dirname}/{fn} ({len(tracks_l2)} tracks)")

    # Verificar total
    if total_exported != N_canonical:
        print(f"[WARN] total_exported={total_exported} != N_canonical={N_canonical} — "
              "revisar si hay tracks sin posición asignada")
    else:
        print(f"[INFO] Export check OK: {total_exported} == N_canonical ({N_canonical})")

    # Generar _summary.txt
    summary_path = out_dir / "_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"TRAKTOR ML V4 — Playlist Export Summary\n")
        f.write(f"Dataset: {dataset_name} | N tracks: {N_canonical}\n")
        f.write(f"Windows audio dir: {windows_audio_dir}\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Cluster':<12} | {'Archivo M3U':<35} | {'N':>6}\n")
        f.write("-" * 60 + "\n")
        for line in summary_lines:
            f.write(line + "\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'TOTAL':<50} | {total_exported:>6}\n")

    print("\n" + "=" * 70)
    print(f"[INFO] Export complete: {total_exported} tracks in {out_dir}")
    print(f"[INFO] Summary: {summary_path}")
    print("=" * 70)

    return out_dir


def _cluster_to_letter_export(n: int) -> str:
    """0→'A', 1→'B', ..., 25→'Z', 26→'AA'."""
    letters = []
    while True:
        letters.append(chr(ord("A") + (n % 26)))
        n = n // 26 - 1
        if n < 0:
            break
    return "".join(reversed(letters))


def main() -> int:
    parser = argparse.ArgumentParser(description="TRAKTOR ML V4 — Phase 5: M3U Export")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--config-hash", default=None)
    parser.add_argument("--windows-audio-dir", default=None,
                        help="Ruta raíz Windows de audio (override de config)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("[INFO] TRAKTOR ML V4 — Phase 5: M3U Export")
    print(f"[INFO] Dataset: {args.dataset_name}")
    if args.windows_audio_dir:
        print(f"[INFO] Windows audio dir (override): {args.windows_audio_dir}")
    print("=" * 70)

    config = load_config(Path(args.config) if args.config else None)
    run_export(
        dataset_name=args.dataset_name,
        config=config,
        config_hash=args.config_hash,
        windows_audio_dir=args.windows_audio_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
