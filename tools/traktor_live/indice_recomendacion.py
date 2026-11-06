"""
PURPOSE: Arma el índice de la recomendación en vivo (recomendador.py): un JSON con cada tema de la
         biblioteca (organización del paquete, _organizacion.csv), su track_uid, su archivo en el paquete,
         su ruta original (para reconocerlo cuando suena desde Música/), BPM y tonalidad de los tags y su
         embedding CLAP normalizado (float32 en base64). Corre una vez, con el Python del pipeline
         (numpy y pandas), y otra cada vez que cambie la biblioteca; el servicio en vivo lo lee con la
         biblioteca estándar. La tonalidad y el BPM que analizó Traktor se leen en vivo de su colección y
         mandan sobre los de los tags (DECISIONS 2026-11-06, punto 8).
         Por defecto: datos en artifacts/v4/datasets/musica (en este PC, el pendrive), representación
         clap_full, paquete en <unidad>:\\TRAKTOR ML (el primero con _organizacion.csv) y salida en
         artifacts/v4/traktor_live/indice_recomendacion.json (fuera de Git).
CHANGELOG:
  - 2026-11-06: Creación (etapa de recomendación, unidad 2: sugerencias por CLAP).
  - 2026-11-06: Versión 2: género por tema (features/genre_pred.parquet: el de Beatport y, si falta, el del
                clasificador, con su fuente), para el componente "genero" del puntaje.
  - 2026-11-06: El clasificador solo con confianza >= 0.6 (el umbral de DECISIONS 2026-09-29, punto 5); el
                gate de Discogs y la confianza baja quedan sin género (el componente no cuenta).
  - 2026-11-06: Versión 3: subgénero de Beatport (features/beatport_genres.parquet, página del tema) cuando
                su género es el del tema, para la taxonomía de géneros (config/genre_taxonomy/).
"""
from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import json
import string
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from tools.traktor_live import traktor_estado as te  # noqa: E402

DEFAULT_DATA = REPO_ROOT / "artifacts" / "v4" / "datasets" / "musica"
DEFAULT_REP = "clap_full"
DEFAULT_OUT = te.DEFAULT_LOG_DIR / "indice_recomendacion.json"
PACKAGE_NAME = "TRAKTOR ML"
ORG_CSV = "_organizacion.csv"
INDEX_VERSION = 3
MIN_CLASSIFIER_CONF = 0.6             # DECISIONS 2026-09-29, punto 5


def genre_of(beatport_genre, final_genre, final_source, model_conf) -> Optional[Dict[str, str]]:
    """{"genero", "fuente"} de un tema: el de Beatport (A o B, confirmado o no); si no hay, el del
    clasificador con confianza >= MIN_CLASSIFIER_CONF; si no, None."""
    if isinstance(beatport_genre, str) and beatport_genre:
        return {"genero": beatport_genre, "fuente": "beatport"}
    try:
        conf = float(model_conf)
    except (TypeError, ValueError):
        conf = 0.0
    if final_source == "classifier" and isinstance(final_genre, str) and final_genre and conf >= MIN_CLASSIFIER_CONF:
        return {"genero": final_genre, "fuente": "clasificador"}
    return None


def add_subgenres(genres: Dict[str, Dict], rows) -> None:
    """Anota el subgénero de Beatport (filas con track_uid, genre, sub_genre) en los temas cuyo género del
    índice es el mismo: un subgénero solo tiene sentido dentro de su género."""
    for r in rows:
        g = genres.get(r.track_uid)
        if g is not None and isinstance(r.sub_genre, str) and r.sub_genre and r.genre == g["genero"]:
            g["subgenero"] = r.sub_genre


def find_package(platform: Optional[str] = None) -> Optional[Path]:
    """Primer <unidad>:\\TRAKTOR ML (Windows) o /Volumes/<disco>/TRAKTOR ML (macOS) con _organizacion.csv."""
    system = platform or sys.platform
    if system == "win32":
        roots = [Path(f"{d}:\\") for d in string.ascii_uppercase]
    else:
        roots = sorted(Path("/Volumes").glob("*")) if Path("/Volumes").exists() else []
    for r in roots:
        p = r / PACKAGE_NAME
        try:
            if (p / ORG_CSV).exists():
                return p
        except OSError:
            continue
    return None


def build_index(rows: List[Dict[str, str]], uids: List[str], embeddings, tags: Dict[str, Dict], rep: str,
                genres: Optional[Dict[str, Dict]] = None) -> dict:
    """rows: _organizacion.csv; uids/embeddings: la representación (misma fila); tags: track_uid ->
    {"bpm", "key"}; genres: track_uid -> {"genero", "fuente", "subgenero" opcional}. Devuelve el índice."""
    genres = genres or {}
    import numpy as np
    emb = np.asarray(embeddings, dtype="float64")
    norms = np.linalg.norm(emb, axis=1)
    row_of = {u: i for i, u in enumerate(uids)}
    tracks = []
    for r in rows:
        uid = r["track_uid"]
        i = row_of.get(uid)
        vec = None
        if i is not None and np.isfinite(emb[i]).all() and norms[i] > 0:
            unit = (emb[i] / norms[i]).astype("<f4")
            vec = base64.b64encode(unit.tobytes()).decode("ascii")
        t = tags.get(uid) or {}
        bpm = t.get("bpm")
        tracks.append({
            "uid": uid, "archivo": r["archivo"],
            "original": "\\".join(x for x in (r.get("carpeta_original", ""), r.get("nombre_original", "")) if x),
            "artista": r.get("artista", ""), "titulo": r.get("titulo", ""),
            "carpeta": r.get("carpeta", ""), "playlist": r.get("playlist", ""),
            "bpm": None if bpm is None or bpm != bpm else round(float(bpm), 2),
            "key": t.get("key") or None, "emb": vec,
            "genero": (genres.get(uid) or {}).get("genero"), "genero_fuente": (genres.get(uid) or {}).get("fuente"),
            "subgenero": (genres.get(uid) or {}).get("subgenero"),
        })
    return {"version": INDEX_VERSION, "rep": rep, "dim": int(emb.shape[1]) if emb.ndim == 2 else 0,
            "creado": dt.datetime.now().isoformat(timespec="seconds"), "temas": tracks}


def main() -> int:
    ap = argparse.ArgumentParser(description="Índice de la recomendación en vivo (CLAP, BPM y tonalidad por tema)")
    ap.add_argument("--datos", default=str(DEFAULT_DATA), help="Carpeta del dataset (representations/, features/)")
    ap.add_argument("--rep", default=DEFAULT_REP, help="Representación (por defecto clap_full)")
    ap.add_argument("--paquete", default=None, help="Carpeta del paquete con _organizacion.csv (por defecto <unidad>:\\TRAKTOR ML)")
    ap.add_argument("--salida", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    import numpy as np
    import pandas as pd

    data = Path(args.datos)
    package = Path(args.paquete) if args.paquete else find_package()
    if package is None or not (package / ORG_CSV).exists():
        print(f"[ERROR] No encontré el paquete con {ORG_CSV} (--paquete)")
        return 1
    rep_dir = data / "representations" / args.rep
    emb = np.load(rep_dir / "embeddings.npy")
    uids = json.loads((rep_dir / "track_uids.json").read_text(encoding="utf-8"))
    if emb.shape[0] != len(uids):
        print(f"[ERROR] {rep_dir}: {emb.shape[0]} embeddings y {len(uids)} track_uids")
        return 1
    with open(package / ORG_CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    tags: Dict[str, Dict] = {}
    bk_path = data / "features" / "bpm_key.parquet"
    if bk_path.exists():
        bk = pd.read_parquet(bk_path)
        for r in bk.itertuples(index=False):
            tags[r.track_uid] = {"bpm": r.bpm, "key": r.key if isinstance(r.key, str) and r.key else None}
    genres: Dict[str, Dict] = {}
    gp_path = data / "features" / "genre_pred.parquet"
    if gp_path.exists():
        gp = pd.read_parquet(gp_path)
        for r in gp.itertuples(index=False):
            g = genre_of(r.beatport_genre, r.final_genre, r.final_source, r.model_conf)
            if g is not None:
                genres[r.track_uid] = g
    sub_path = data / "features" / "beatport_genres.parquet"
    if sub_path.exists():
        add_subgenres(genres, pd.read_parquet(sub_path).itertuples(index=False))
    index = build_index(rows, uids, emb, tags, args.rep, genres)
    index["paquete"] = str(package)
    out = Path(args.salida)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out)
    t = index["temas"]
    print(f"[OK] {out}: {len(t)} temas del paquete {package}; con embedding {sum(1 for x in t if x['emb'])}, "
          f"con BPM {sum(1 for x in t if x['bpm'])}, con tonalidad de tags {sum(1 for x in t if x['key'])}, "
          f"con género {sum(1 for x in t if x['genero'])} (de Beatport {sum(1 for x in t if x['genero_fuente'] == 'beatport')}, "
          f"con subgénero {sum(1 for x in t if x['subgenero'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
