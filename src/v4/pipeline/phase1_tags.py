"""
PURPOSE: Phase 1 (variante tags) — BPM, tonalidad y energía desde los tags ya leídos por Phase 0
         (catálogo con read_tags), sin Essentia. Escribe features/bpm_key.parquet con el mismo
         contrato que phase1_extract (track_uid, bpm, key) más energy (Mixed In Key, 1-10) y la
         fuente de cada valor, para que Phase 4 ordene sin cambios. Sirve en Windows, donde
         Essentia no tiene build nativo. --estimate-missing estima el BPM que falte en los tags con
         src/v4/common/tempo.py sobre los segmentos DJ (bpm_source = "estimate").
         --traktor completa antes el BPM y la tonalidad que falten con los de la colección de Traktor
         (collection.nml; bpm_source/key_source = "traktor"): Traktor guarda su análisis ahí, no en el
         archivo. Se empareja por nombre de archivo y duración (±2 s), solo si hay una coincidencia.
CHANGELOG:
  - 2026-09-27: Creación inicial (MVP biblioteca completa). --estimate-missing.
  - 2026-11-07: --traktor (Gabriel: BPM y tonalidad de los temas descargados los calcula Traktor al cargarlos;
                el proyecto los leía solo de los tags, y Traktor no los escribe en el archivo).
"""
import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

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


TRAKTOR_DURATION_TOLERANCE_S = 2.0


def traktor_bpm_key(collection: Path) -> Dict[str, List[Tuple[float, str, float]]]:
    """Por nombre de archivo (casefold): (BPM, Camelot, duración en s) de cada entrada de collection.nml. La
    tonalidad con traktor_key de las sugerencias en vivo (tools/traktor_live/recomendador.py)."""
    from tools.traktor_live.recomendador import traktor_key

    out: Dict[str, List[Tuple[float, str, float]]] = {}
    for e in ET.parse(collection).getroot().iter("ENTRY"):
        loc = e.find("LOCATION")
        if loc is None or not loc.get("FILE"):
            continue
        tempo, key, info = e.find("TEMPO"), e.find("MUSICAL_KEY"), e.find("INFO")
        try:
            bpm = float(tempo.get("BPM")) if tempo is not None and tempo.get("BPM") else 0.0
        except ValueError:
            bpm = 0.0
        try:
            duration = float((info.get("PLAYTIME_FLOAT") or info.get("PLAYTIME") or 0)) if info is not None else 0.0
        except ValueError:
            duration = 0.0
        camelot = traktor_key(key.get("VALUE")) if key is not None else "?"
        out.setdefault(loc.get("FILE").casefold(), []).append((bpm, camelot, duration))
    return out


def fill_from_traktor(table: pd.DataFrame, catalog: pd.DataFrame, collection: Path) -> Tuple[int, int]:
    """BPM y tonalidad que falten, de la colección de Traktor. Devuelve (BPM completados, tonalidades)."""
    entries = traktor_bpm_key(collection)
    for c in ("key", "bpm_source", "key_source"):  # una columna toda vacía puede venir como float
        table[c] = table[c].astype(object)
    meta = catalog.drop_duplicates("track_uid").set_index("track_uid")
    n_bpm = n_key = 0
    for i in table.index[table["bpm"].isna() | table["key"].isna()]:
        uid = table.at[i, "track_uid"]
        if uid not in meta.index:
            continue
        candidates = entries.get(Path(str(meta.at[uid, "source_path"])).name.casefold(), [])
        duration = pd.to_numeric(meta.at[uid, "duration_s"], errors="coerce") if "duration_s" in meta.columns else None
        if duration is not None and pd.notna(duration):
            candidates = [c for c in candidates if not c[2] or abs(c[2] - duration) <= TRAKTOR_DURATION_TOLERANCE_S]
        if len(candidates) != 1:  # ninguna, o más de una con ese nombre: no se adivina
            continue
        bpm, camelot, _ = candidates[0]
        if pd.isna(table.at[i, "bpm"]) and bpm > 0:
            table.at[i, "bpm"], table.at[i, "bpm_source"] = bpm, "traktor"
            n_bpm += 1
        if (table.at[i, "key"] is None or pd.isna(table.at[i, "key"])) and camelot != "?":
            table.at[i, "key"], table.at[i, "key_source"] = camelot, "traktor"
            n_key += 1
    return n_bpm, n_key


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
    parser.add_argument("--traktor", action="store_true",
                        help="Antes de estimar, completar BPM y tonalidad que falten con la colección de Traktor")
    parser.add_argument("--traktor-collection", type=Path, default=None,
                        help="collection.nml (por defecto la más reciente de Documentos/Native Instruments)")
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else None)
    catalog = load_catalog(args.dataset_name, config)
    table = bpm_key_from_catalog(catalog)
    features_dir = resolve_dataset_artifacts(args.dataset_name, config) / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    out = features_dir / "bpm_key.parquet"
    if args.traktor:
        from src.v4.common.traktor_nml import find_collection
        collection = args.traktor_collection or find_collection()
        if collection is None or not Path(collection).is_file():
            print("[INFO] Sin colección de Traktor en este ordenador: BPM y tonalidad solo de los tags.")
        else:
            n_bpm, n_key = fill_from_traktor(table, catalog, Path(collection))
            print(f"[INFO] De la colección de Traktor ({collection}): BPM {n_bpm}, tonalidad {n_key}", flush=True)
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
