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
INDEX_VERSION = 1


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


def build_index(rows: List[Dict[str, str]], uids: List[str], embeddings, tags: Dict[str, Dict], rep: str) -> dict:
    """rows: _organizacion.csv; uids/embeddings: la representación (misma fila); tags: track_uid ->
    {"bpm", "key"}. Devuelve el índice (dict serializable)."""
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
    index = build_index(rows, uids, emb, tags, args.rep)
    index["paquete"] = str(package)
    out = Path(args.salida)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out)
    t = index["temas"]
    print(f"[OK] {out}: {len(t)} temas del paquete {package}; con embedding {sum(1 for x in t if x['emb'])}, "
          f"con BPM {sum(1 for x in t if x['bpm'])}, con tonalidad de tags {sum(1 for x in t if x['key'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
