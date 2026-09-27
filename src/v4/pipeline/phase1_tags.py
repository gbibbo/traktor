"""
PURPOSE: Phase 1 (variante tags) — BPM, tonalidad y energía desde los tags ya leídos por Phase 0
         (catálogo con read_tags), sin Essentia. Escribe features/bpm_key.parquet con el mismo
         contrato que phase1_extract (track_uid, bpm, key) más energy (Mixed In Key, 1-10) y la
         fuente de cada valor, para que Phase 4 ordene sin cambios. Sirve en Windows, donde
         Essentia no tiene build nativo. --estimate-missing estima el BPM que falte en los tags con
         src/v4/common/tempo.py sobre los segmentos DJ (bpm_source = "estimate").
CHANGELOG:
  - 2026-09-27: Creación inicial (MVP biblioteca completa). --estimate-missing.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.audio_utils import read_dj_segments  # noqa: E402
from src.v4.common.catalog import load_catalog  # noqa: E402
from src.v4.common.config_loader import load_config  # noqa: E402
from src.v4.common.path_resolver import resolve_dataset_artifacts  # noqa: E402
from src.v4.common.tempo import estimate_bpm  # noqa: E402


def bpm_key_from_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    """Tabla bpm_key desde columnas tag_* del catálogo. Falta de dato = NaN / None, nunca inventado."""
    required = {"track_uid", "tag_bpm", "tag_camelot"}
    missing = required - set(catalog.columns)
    if missing:
        raise ValueError(f"El catálogo no tiene tags ({sorted(missing)}); correr Phase 0 con read_tags: true")
    out = pd.DataFrame({
        "track_uid": catalog["track_uid"],
        "bpm": pd.to_numeric(catalog["tag_bpm"], errors="coerce"),
        "key": catalog["tag_camelot"].where(catalog["tag_camelot"].notna(), None),
        "energy": pd.to_numeric(catalog.get("mik_energy"), errors="coerce"),
    })
    out["bpm_source"] = out["bpm"].notna().map({True: "tags", False: None})
    out["key_source"] = out["key"].notna().map({True: "tags", False: None})
    return out


def fill_missing_bpm(table: pd.DataFrame, catalog: pd.DataFrame, seg_cfg: dict) -> pd.DataFrame:
    """Estima el BPM de las filas sin BPM leyendo los segmentos DJ del archivo."""
    paths = catalog.set_index("track_uid")["source_path"]
    missing = table.index[table["bpm"].isna()]
    for n, i in enumerate(missing, 1):
        path = Path(paths[table.at[i, "track_uid"]])
        try:
            segs, sr = read_dj_segments(path, float(seg_cfg.get("segment_duration_s", 30.0)),
                                        int(seg_cfg.get("n_intro_segments", 0)), int(seg_cfg.get("n_mid_segments", 3)),
                                        int(seg_cfg.get("n_outro_segments", 0)))
            bpm = estimate_bpm(segs, sr)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] {path.name}: {type(exc).__name__}: {exc}")
            bpm = None
        if bpm is not None:
            table.at[i, "bpm"] = bpm
            table.at[i, "bpm_source"] = "estimate"
        if n % 25 == 0 or n == len(missing):
            print(f"[INFO] BPM estimado {n}/{len(missing)}", flush=True)
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description="TRAKTOR ML V4 — Phase 1 (tags): BPM/tonalidad/energía desde tags")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--estimate-missing", action="store_true", help="Estimar desde el audio el BPM que falte")
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else None)
    catalog = load_catalog(args.dataset_name, config)
    table = bpm_key_from_catalog(catalog)
    features_dir = resolve_dataset_artifacts(args.dataset_name, config) / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    out = features_dir / "bpm_key.parquet"
    if args.estimate_missing:
        table["bpm_source"] = table["bpm_source"].astype(object)
        if out.exists():  # reutilizar estimaciones previas (mismo track_uid = mismo audio)
            prev = pd.read_parquet(out)
            prev = prev[prev["bpm_source"] == "estimate"].set_index("track_uid")["bpm"]
            reuse = table["bpm"].isna() & table["track_uid"].isin(prev.index)
            table.loc[reuse, "bpm"] = table.loc[reuse, "track_uid"].map(prev)
            table.loc[reuse, "bpm_source"] = "estimate"
        table = fill_missing_bpm(table, catalog, config.get("segmentation", {}))
    table.to_parquet(out, index=False)
    n = len(table)
    print(f"[INFO] bpm_key.parquet: {n} temas | BPM {table['bpm'].notna().sum()}/{n} "
          f"(estimados {int((table['bpm_source'] == 'estimate').sum())}) | "
          f"tonalidad {table['key'].notna().sum()}/{n} | energía {table['energy'].notna().sum()}/{n} → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
