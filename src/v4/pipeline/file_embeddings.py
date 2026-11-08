"""
PURPOSE: Guardar dentro de cada archivo de música (en sus tags) su embedding y su lugar en la organización, para
         que la música y sus datos viajen juntos (DECISIONS 2026-11-04; docs/reports/
         embeddings_en_los_archivos_2026-11-04.md: Traktor 4.4.1 y Rekordbox 7.2.19 los conservan al escribir
         sus propios tags).
           - MP3: dos marcos ID3 GEOB, «TRAKTOR_ML <rep> v1» (el embedding, float32 little-endian) y
             «TRAKTOR_ML organizacion v1» (JSON: organización, versión, carpeta, playlist, posición, punto del
             mapa y track_uid). Se respeta la versión ID3 del archivo (2.3 o 2.4).
           - FLAC: los mismos datos en dos comentarios Vorbis (TRAKTOR_ML_<REP>_V1 en base64,
             TRAKTOR_ML_ORGANIZACION_V1).
           - WAV y AIFF no se tocan: su track_uid es el hash del archivo entero y cambiaría.
         Cada archivo se verifica: el embedding y el lugar se releen idénticos y el hash del audio sin tags
         (track_uid) es el mismo antes y después. Lo que ya está escrito igual no se reescribe.
         Uso (el paquete de usb_package.py, con su _organizacion.csv):
           file_embeddings.py write --package "D:\\TRAKTOR ML" [--folder "<subcarpeta de Musica>"] [--low-priority]
           file_embeddings.py check --package "D:\\TRAKTOR ML"
CHANGELOG:
  - 2026-11-07: same() no compara la versión de la organización guardada en el lugar: es la versión en que se
                fijó ese lugar. Cada agregado sube la versión, y por eso «Actualizar el pendrive» reescribía (y
                verificaba con dos hashes de audio) los 1760 archivos aunque solo hubieran entrado 10 temas.
                package_plan(only_uids=…): solo esos temas (los nuevos o los que cambiaron de lugar).
  - 2026-11-04: Creación.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import sys
import time
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.config_loader import load_config  # noqa: E402
from src.v4.common.embedding_utils import load_track_embeddings  # noqa: E402
from src.v4.common.path_resolver import resolve_dataset_artifacts  # noqa: E402
from src.v4.common.tags import audio_payload_hash  # noqa: E402
from src.v4.pipeline.organize import OrgStore  # noqa: E402

WRITABLE = (".mp3", ".flac")
ORG_KEY = "TRAKTOR_ML organizacion v1"
REPORT = "file_embeddings_report.json"  # en los artifacts del dataset


def emb_key(rep: str) -> str:
    return f"TRAKTOR_ML {rep} v1"


def vorbis_key(key: str) -> str:
    return key.upper().replace(" ", "_")


# ---------------------------------------------------------------------------
# Un archivo
# ---------------------------------------------------------------------------

def read_track(path: Path, rep: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """(embedding, lugar) guardados en el archivo; None si no están."""
    path = Path(path)
    if path.suffix.lower() == ".flac":
        from mutagen.flac import FLAC
        tags = FLAC(str(path))
        raw = (tags.get(vorbis_key(emb_key(rep))) or [None])[0]
        place = (tags.get(vorbis_key(ORG_KEY)) or [None])[0]
        vec = np.frombuffer(base64.b64decode(raw), dtype="<f4") if raw else None
        return vec, json.loads(place) if place else None
    from mutagen.id3 import ID3, ID3NoHeaderError
    try:
        tags = ID3(str(path))
    except ID3NoHeaderError:
        return None, None
    g, o = tags.get("GEOB:" + emb_key(rep)), tags.get("GEOB:" + ORG_KEY)
    vec = np.frombuffer(g.data, dtype="<f4") if g else None
    return vec, json.loads(o.data.decode("utf-8")) if o else None


def write_track(path: Path, vec: np.ndarray, place: Dict, rep: str) -> None:
    path = Path(path)
    data = np.asarray(vec, dtype="<f4").tobytes()
    text = json.dumps(place, ensure_ascii=False, sort_keys=True)
    if path.suffix.lower() == ".flac":
        from mutagen.flac import FLAC
        tags = FLAC(str(path))
        tags[vorbis_key(emb_key(rep))] = [base64.b64encode(data).decode("ascii")]
        tags[vorbis_key(ORG_KEY)] = [text]
        tags.save()
        return
    if path.suffix.lower() != ".mp3":
        raise ValueError(f"formato que no se escribe: {path.suffix}")
    from mutagen.id3 import GEOB, ID3, ID3NoHeaderError
    try:
        tags = ID3(str(path))
    except ID3NoHeaderError:
        tags = ID3()
    version = tags.version[1] if tags.version and tags.version[1] in (3, 4) else 3
    tags.add(GEOB(encoding=3, mime="application/octet-stream", filename=f"{rep}.f32", desc=emb_key(rep), data=data))
    tags.add(GEOB(encoding=3, mime="application/json", filename="organizacion.json", desc=ORG_KEY,
                  data=text.encode("utf-8")))
    tags.save(str(path), v2_version=version)


PLACE_IGNORED = ("version",)   # la versión en que se fijó el lugar: no hace distinto un lugar que no cambió


def _comparable(place: Optional[Dict]) -> Optional[Dict]:
    return None if place is None else {k: v for k, v in place.items() if k not in PLACE_IGNORED}


def same(vec: Optional[np.ndarray], place: Optional[Dict], want_vec: np.ndarray, want_place: Dict) -> bool:
    return (vec is not None and _comparable(place) == _comparable(want_place)
            and np.array_equal(vec, np.asarray(want_vec, dtype="<f4")))


# ---------------------------------------------------------------------------
# El paquete
# ---------------------------------------------------------------------------

def package_plan(package: Path, artifacts: Path, org_name: str, folder: Optional[str] = None,
                 only_uids: Optional[set] = None) -> Tuple[List[Dict], str]:
    """Qué escribir en cada archivo del paquete: embedding y lugar en la versión actual de la organización.
    only_uids: solo esos temas (sin leer el resto de los archivos)."""
    store = OrgStore(artifacts, org_name)
    meta = store.meta()
    rep = meta["params"]["rep"]
    assign, _, version = store.load()
    by_uid = assign.set_index("track_uid")
    uids, M = load_track_embeddings(artifacts, rep=rep)
    row_of = {u: i for i, u in enumerate(uids)}
    index = pd.read_csv(Path(package) / "_organizacion.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    plan = []
    for r in index.itertuples():
        rel = PureWindowsPath(r.archivo)
        if folder and (len(rel.parts) < 2 or rel.parts[1] != folder):
            continue
        if only_uids is not None and r.track_uid not in only_uids:
            continue
        a = by_uid.loc[r.track_uid] if r.track_uid in by_uid.index else None
        place = None if a is None else {
            "org": org_name, "version": int(version), "track_uid": r.track_uid,
            "l1": int(a["l1"]), "l2": int(a["l2"]), "position": int(a["position"]),
            "x": round(float(a["x"]), 6), "y": round(float(a["y"]), 6),
            "carpeta": r.carpeta, "playlist": r.playlist}
        plan.append({"path": Path(package) / rel.as_posix(), "uid": r.track_uid, "place": place,
                     "vec": M[row_of[r.track_uid]] if r.track_uid in row_of else None})
    return plan, rep


def write_package(plan: List[Dict], rep: str, dry_run: bool = False) -> Dict:
    counts = {"escritos": 0, "ya_estaban": 0, "wav_aiff": 0, "sin_datos": 0, "errores": 0}
    problems: List[Dict] = []
    t0 = time.time()
    for k, item in enumerate(plan, 1):
        path = item["path"]
        if path.suffix.lower() not in WRITABLE:
            counts["wav_aiff"] += 1
            continue
        if item["vec"] is None or item["place"] is None:
            counts["sin_datos"] += 1
            problems.append({"archivo": str(path), "motivo": "sin embedding o fuera de la organización"})
            continue
        try:
            if same(*read_track(path, rep), item["vec"], item["place"]):
                counts["ya_estaban"] += 1
            elif not dry_run:
                before = audio_payload_hash(path)
                write_track(path, item["vec"], item["place"], rep)
                after = audio_payload_hash(path)
                if after != before:
                    raise RuntimeError("el hash del audio cambió al escribir")
                if not same(*read_track(path, rep), item["vec"], item["place"]):
                    raise RuntimeError("lo escrito no se relee igual")
                counts["escritos"] += 1
        except Exception as exc:  # noqa: BLE001 (un archivo raro se informa y se sigue)
            counts["errores"] += 1
            problems.append({"archivo": str(path), "motivo": f"{type(exc).__name__}: {exc}"})
        if k % 25 == 0 or k == len(plan):
            print(f"[INFO] {k}/{len(plan)} temas · {time.time() - t0:.0f} s", flush=True)
    return {"counts": counts, "problems": problems, "seconds": round(time.time() - t0, 1)}


def check_package(plan: List[Dict], rep: str) -> Dict:
    counts = {"bien": 0, "faltan": 0, "distintos": 0, "wav_aiff": 0}
    for item in plan:
        if item["path"].suffix.lower() not in WRITABLE:
            counts["wav_aiff"] += 1
            continue
        vec, place = read_track(item["path"], rep)
        if vec is None and place is None:
            counts["faltan"] += 1
        elif item["vec"] is not None and same(vec, place, item["vec"], item["place"]):
            counts["bien"] += 1
        else:
            counts["distintos"] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Embedding y lugar en la organización dentro de cada archivo")
    parser.add_argument("mode", choices=("write", "check"))
    parser.add_argument("--package", required=True, help="Carpeta del paquete (con _organizacion.csv)")
    parser.add_argument("--dataset-name", default="musica")
    parser.add_argument("--org-name", default="biblioteca")
    parser.add_argument("--config", default=None)
    parser.add_argument("--folder", default=None, help="Solo una subcarpeta de Musica (prueba chica)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--low-priority", action="store_true")
    args = parser.parse_args()
    if args.low_priority:
        from src.v4.pipeline.extract_representations import lower_priority
        lower_priority()
    artifacts = resolve_dataset_artifacts(args.dataset_name, load_config(Path(args.config) if args.config else None))
    plan, rep = package_plan(Path(args.package), artifacts, args.org_name, args.folder)
    if args.mode == "check":
        print(json.dumps(check_package(plan, rep), ensure_ascii=False))
        return 0
    result = write_package(plan, rep, args.dry_run)
    result.update({"package": str(args.package), "folder": args.folder, "rep": rep, "org": args.org_name,
                   "dry_run": args.dry_run, "when": dt.datetime.now().isoformat(timespec="seconds")})
    if not args.dry_run:
        (artifacts / REPORT).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("counts", "seconds")}, ensure_ascii=False))
    for p in result["problems"][:20]:
        print("[WARN]", p["archivo"], "·", p["motivo"])
    return 1 if result["counts"]["errores"] else 0


if __name__ == "__main__":
    sys.exit(main())
