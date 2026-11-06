"""
PURPOSE: Categoría, género y subgénero de Beatport de cada tema con coincidencia (niveles A y B de
         features/beatport.parquet), leídos de la página del tema en beatport.com
         (BeatportClient.track_genre). Beatport tiene tres niveles: categoría ("Electronic"), género
         ("Techno (Peak Time / Driving)") y subgénero opcional ("Psy-Techno"); la búsqueda que usó
         beatport_lookup.py guardaba solo el género. Salida: features/beatport_genres.parquet y .csv
         (track_uid, bp_track_id, ids y nombres de los tres niveles, y bp_genre para comparar). Cada
         página queda en artifacts/v4/beatport_cache/tracks/<id>.json: repetir la corrida no vuelve a
         preguntar. --limit para la corrida chica de prueba; --low-priority para la completa.
CHANGELOG:
  - 2026-11-06: Creación (taxonomía de géneros para la recomendación en vivo).
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.beatport import BeatportBlocked, BeatportClient  # noqa: E402
from src.v4.common.config_loader import load_config  # noqa: E402
from src.v4.common.path_resolver import resolve_artifacts_root, resolve_dataset_artifacts  # noqa: E402

COLUMNS = ["track_uid", "bp_track_id", "category_id", "category", "genre_id", "genre", "sub_genre_id",
           "sub_genre", "bp_genre"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Categoría, género y subgénero de Beatport por tema")
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Solo los primeros N temas (prueba)")
    ap.add_argument("--min-interval", type=float, default=1.5, help="Segundos mínimos entre pedidos")
    ap.add_argument("--offline", action="store_true", help="Solo la caché, sin pedidos nuevos")
    ap.add_argument("--output", default="beatport_genres", help="Nombre base de la salida en features/")
    ap.add_argument("--low-priority", action="store_true", help="Prioridad baja del proceso (corridas largas)")
    args = ap.parse_args()

    if args.low_priority:
        from src.v4.pipeline.extract_representations import lower_priority
        lower_priority()
    config = load_config(Path(args.config) if args.config else None)
    features = resolve_dataset_artifacts(args.dataset_name, config) / "features"
    bp = pd.read_parquet(features / "beatport.parquet")
    bp = bp[bp["bp_track_id"].notna()]
    if args.limit:
        bp = bp.head(args.limit)
    client = BeatportClient(resolve_artifacts_root(config).parent / "beatport_cache",
                            min_interval=args.min_interval, offline=args.offline)
    print(f"[beatport_genres] {args.dataset_name}: {len(bp)} temas con coincidencia en Beatport", flush=True)
    rows, t0 = [], time.monotonic()
    try:
        for i, r in enumerate(bp.itertuples(index=False), 1):
            g = client.track_genre(int(r.bp_track_id), r.bp_url if isinstance(r.bp_url, str) else None) or {}
            rows.append({"track_uid": r.track_uid, "bp_track_id": int(r.bp_track_id),
                         **{k: g.get(k) for k in COLUMNS[2:8]}, "bp_genre": r.bp_genre})
            if i % 100 == 0:
                print(f"  {i}/{len(bp)} ({client.n_requests} pedidos, {time.monotonic() - t0:.0f} s)", flush=True)
    except BeatportBlocked as exc:
        print(f"[ERROR] {exc}. Se guarda lo leído hasta acá; repetir la corrida sigue desde la caché.")
    out = pd.DataFrame(rows, columns=COLUMNS)
    for c in ("category_id", "genre_id", "sub_genre_id"):
        out[c] = out[c].astype("Int64")
    out.to_parquet(features / f"{args.output}.parquet", index=False)
    out.to_csv(features / f"{args.output}.csv", index=False, encoding="utf-8-sig")
    read = out["genre"].notna()
    same = (out.loc[read, "genre"] == out.loc[read, "bp_genre"]).sum()
    print(f"[OK] {features / args.output}.parquet: {len(out)} temas; leídos {int(read.sum())}, con subgénero "
          f"{int(out['sub_genre'].notna().sum())}; género igual al de la búsqueda {int(same)} de {int(read.sum())}; "
          f"{client.n_requests} pedidos en {time.monotonic() - t0:.0f} s")
    print(out.loc[read, ["category", "genre", "sub_genre"]].value_counts(dropna=False).head(30).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
