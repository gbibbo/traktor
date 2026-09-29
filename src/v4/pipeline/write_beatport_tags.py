"""
PURPOSE: Escribe en los archivos los metadatos de Beatport y el género del modelo, con las reglas que
         decidió Gabriel el 2026-09-29 (docs/DECISIONS.md), a partir de features/beatport.parquet y
         features/genre_pred.parquet:
           - match A, o B confirmado por sello, fecha o duración: Artist, Remixers, Label, Genre y
             Released de Beatport (un campo que Beatport no trae no se toca);
           - C o D con género del clasificador de confianza >= --min-conf: solo Genre;
           - C o D sin género confiable (confianza baja, gate MAEST sin validar, sin
             representaciones): Genre vacío con --clear-unreliable-genre; si no, no se toca;
           - B sin confirmar: nada.
         Solo MP3 y FLAC: en WAV y AIFF el hash de audio incluye los tags y el track_uid cambiaría.
         Sin --write es una simulación: features/tag_plan.csv con el valor actual y el nuevo de cada
         campo, leídos de los archivos en ese momento. Con --write guarda antes un respaldo
         (features/tag_backup_<fecha>.csv con los valores anteriores), escribe, relee y verifica que
         el hash del audio sin tags no cambie. --revert <csv> restaura los valores anteriores.
CHANGELOG:
  - 2026-09-29: Creación inicial.
"""
import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.catalog import load_catalog  # noqa: E402
from src.v4.common.config_loader import load_config  # noqa: E402
from src.v4.common.path_resolver import resolve_dataset_artifacts  # noqa: E402
from src.v4.common.tags import audio_payload_hash  # noqa: E402

FIELDS = ("artist", "remixers", "label", "genre", "released")
ID3_FRAMES = {"artist": "TPE1", "remixers": "TPE4", "label": "TPUB", "genre": "TCON", "released": "TDRC"}
VORBIS_KEYS = {"artist": "artist", "remixers": "remixer", "label": "label", "genre": "genre", "released": "date"}
WRITABLE = (".mp3", ".flac")
DEFAULT_MIN_CONF = 0.6


def _s(v) -> str:
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()


# ---------------------------------------------------------------------------
# Plan (puro)
# ---------------------------------------------------------------------------

def build_plan(bp: pd.DataFrame, pred: pd.DataFrame, catalog: pd.DataFrame,
               min_conf: float = DEFAULT_MIN_CONF, clear_unreliable: bool = False) -> pd.DataFrame:
    """Una fila por tema con la acción y el valor nuevo de cada campo.
    new_<campo>: None = no se toca, "" = se vacía, texto = se escribe."""
    pred = pred.drop_duplicates("track_uid").set_index("track_uid")
    cat = catalog.drop_duplicates("track_uid").set_index("track_uid")
    rows = []
    for r in bp.drop_duplicates("track_uid").itertuples():
        path = _s(cat.loc[r.track_uid, "source_path"]) if r.track_uid in cat.index else ""
        new: Dict[str, Optional[str]] = {f: None for f in FIELDS}
        p = pred.loc[r.track_uid] if r.track_uid in pred.index else None
        if r.level == "A" or (r.level == "B" and bool(r.confirmed)):
            action, source = "beatport", f"beatport_{r.level}"
            for f in FIELDS:
                v = _s(getattr(r, f"new_{f}"))
                new[f] = v or None
        elif r.level in ("C", "D"):
            conf = float(p["model_conf"]) if p is not None and pd.notna(p["model_conf"]) else None
            if p is not None and p["model_source"] == "classifier" and conf is not None and conf >= min_conf:
                action, source = "model", f"classifier p={conf:.2f}"
                new["genre"] = _s(p["model_genre"])
            else:
                reason = ("sin representaciones" if p is None else
                          "gate sin validar" if p["model_source"] == "gate_discogs" else f"confianza {conf:.2f}")
                action, source = ("clear_genre" if clear_unreliable else "keep"), f"sin género confiable ({reason})"
                if clear_unreliable:
                    new["genre"] = ""
        else:
            action, source = "keep", "B sin confirmar"
        if Path(path).suffix.lower() not in WRITABLE and action != "keep":
            action, source = "skip_format", f"{Path(path).suffix.lower()}: el track_uid cambiaría"
            new = {f: None for f in FIELDS}
        rows.append({"track_uid": r.track_uid, "source_path": path, "rel_path": r.rel_path, "level": r.level,
                     "action": action, "source": source, **{f"new_{f}": new[f] for f in FIELDS}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Lectura / escritura de campos
# ---------------------------------------------------------------------------

def read_fields(path: Path) -> Dict[str, Optional[List[str]]]:
    """Valores actuales (lista de textos) de cada campo; None si el campo no está."""
    import mutagen
    audio = mutagen.File(str(path))
    out: Dict[str, Optional[List[str]]] = {f: None for f in FIELDS}
    if audio is None or audio.tags is None:
        return out
    if Path(path).suffix.lower() == ".flac":
        for f, k in VORBIS_KEYS.items():
            vals = audio.tags.get(k)
            out[f] = [str(v) for v in vals] if vals else None
    else:
        for f, fid in ID3_FRAMES.items():
            frame = audio.tags.get(fid)
            out[f] = [str(v) for v in frame.text] if frame is not None and frame.text else None
    return out


def write_fields(path: Path, values: Dict[str, Optional[List[str]]]) -> None:
    """values[campo] = lista de textos a escribir, o None para borrar el campo. Otros campos intactos.
    En MP3 conserva la versión ID3 del archivo (2.3 o 2.4), como tags.write_comment."""
    import mutagen
    path = Path(path)
    if path.suffix.lower() == ".flac":
        from mutagen.flac import FLAC
        audio = FLAC(str(path))
        if audio.tags is None:
            audio.add_tags()
        for f, vals in values.items():
            k = VORBIS_KEYS[f]
            if vals:
                audio.tags[k] = list(vals)
            elif k in audio.tags:
                del audio.tags[k]
        audio.save()
        return
    from mutagen import id3
    audio = mutagen.File(str(path))
    if audio is None:
        raise ValueError(f"no se pudo abrir {path}")
    if audio.tags is None:
        audio.add_tags()
    tags = audio.tags
    version = tags.version[1] if getattr(tags, "version", None) and tags.version[1] in (3, 4) else 3
    for f, vals in values.items():
        fid = ID3_FRAMES[f]
        tags.delall(fid)
        if vals:
            tags.add(getattr(id3, fid)(encoding=3, text=list(vals)))
    audio.save(v2_version=version)


def _changes(cur: Dict[str, Optional[List[str]]], row) -> Dict[str, Optional[List[str]]]:
    """Solo los campos cuyo valor cambia. released se compara en los primeros 10 caracteres."""
    out = {}
    for f in FIELDS:
        new = getattr(row, f"new_{f}") if hasattr(row, f"new_{f}") else row[f"new_{f}"]
        if new is None or (isinstance(new, float) and pd.isna(new)):
            continue
        old = "; ".join(cur[f] or [])
        same = old[:10] == new[:10] if f == "released" and len(new) == 10 else old == new
        if not same:
            out[f] = [new] if new else None
    return out


# ---------------------------------------------------------------------------
# Simulación, escritura y reversión
# ---------------------------------------------------------------------------

def simulate(plan: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in plan.itertuples():
        rec = {"track_uid": r.track_uid, "rel_path": r.rel_path, "level": r.level, "action": r.action, "source": r.source}
        if r.action in ("beatport", "model", "clear_genre"):
            try:
                cur = read_fields(Path(r.source_path))
            except Exception as exc:  # noqa: BLE001
                rec.update(action="error", source=f"{type(exc).__name__}: {exc}")
                rows.append(rec)
                continue
            ch = _changes(cur, r)
            for f in FIELDS:
                rec[f"cur_{f}"] = "; ".join(cur[f] or [])
                rec[f"new_{f}"] = getattr(r, f"new_{f}")
            rec["changes"] = ", ".join(ch)
        rows.append(rec)
    return pd.DataFrame(rows)


def apply(plan: pd.DataFrame, backup_path: Path) -> pd.DataFrame:
    """Escribe fila por fila. El respaldo se guarda cada 50 archivos para poder revertir aunque se corte."""
    log: List[dict] = []
    todo = plan[plan["action"].isin(["beatport", "model", "clear_genre"])]
    for i, r in enumerate(todo.itertuples(), 1):
        path = Path(r.source_path)
        rec = {"track_uid": r.track_uid, "source_path": str(path), "action": r.action, "source": r.source}
        try:
            cur = read_fields(path)
            ch = _changes(cur, r)
            rec["old"] = json.dumps({f: cur[f] for f in ch}, ensure_ascii=False)
            rec["new"] = json.dumps(ch, ensure_ascii=False)
            if not ch:
                rec["status"] = "unchanged"
            elif not os.access(path, os.W_OK):
                rec["status"] = "read_only"   # atributo de solo lectura: no se cambia sin pedirlo
            else:
                before = audio_payload_hash(path)
                write_fields(path, ch)
                after_vals = read_fields(path)
                if any((after_vals[f] or None) != (v or None) for f, v in ch.items()):
                    rec["status"] = "verify_failed"
                elif audio_payload_hash(path) != before:
                    rec["status"] = "uid_changed"
                else:
                    rec["status"] = "written"
                rec["uid_matches_catalog"] = before == r.track_uid
        except Exception as exc:  # noqa: BLE001
            rec.update(status="error", detail=f"{type(exc).__name__}: {exc}")
        log.append(rec)
        if i % 50 == 0 or i == len(todo):
            pd.DataFrame(log).to_csv(backup_path, index=False, encoding="utf-8")
            print(f"[tags] {i}/{len(todo)} {pd.Series([x['status'] for x in log]).value_counts().to_dict()}", flush=True)
    return pd.DataFrame(log)


def revert(backup_csv: Path) -> int:
    log = pd.read_csv(backup_csv, encoding="utf-8")
    n = 0
    for r in log[log["status"].isin(["written", "verify_failed", "uid_changed"])].itertuples():
        old = json.loads(r.old)
        write_fields(Path(r.source_path), {f: (v or None) for f, v in old.items()})
        n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description="Escribe tags de Beatport y género del modelo (simulación por defecto)")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--min-conf", type=float, default=DEFAULT_MIN_CONF)
    parser.add_argument("--clear-unreliable-genre", action="store_true",
                        help="Vaciar Genre en C/D sin género confiable (si no, no se toca)")
    parser.add_argument("--folder", default=None, help="Solo temas cuyo rel_path contiene este texto (prueba)")
    parser.add_argument("--write", action="store_true", help="Escribir de verdad (con respaldo)")
    parser.add_argument("--revert", default=None, help="CSV de respaldo: restaurar los valores anteriores")
    parser.add_argument("--low-priority", action="store_true")
    args = parser.parse_args()

    if args.revert:
        print(f"[tags] restaurados {revert(Path(args.revert))} archivos")
        return 0
    if args.low_priority:
        from src.v4.pipeline.extract_representations import lower_priority
        lower_priority()
    config = load_config(Path(args.config) if args.config else None)
    feats = resolve_dataset_artifacts(args.dataset_name, config) / "features"
    plan = build_plan(pd.read_parquet(feats / "beatport.parquet"), pd.read_parquet(feats / "genre_pred.parquet"),
                      load_catalog(args.dataset_name, config), args.min_conf, args.clear_unreliable_genre)
    if args.folder:
        plan = plan[plan["rel_path"].str.contains(args.folder, case=False, regex=False)]
    print(f"[tags] {len(plan)} temas; acciones: {plan['action'].value_counts().to_dict()}")

    if not args.write:
        sim = simulate(plan)
        sim.to_csv(feats / "tag_plan.csv", index=False, encoding="utf-8-sig")
        touched = sim[sim.get("changes", pd.Series(dtype=str)).fillna("") != ""]
        per_field = {f: int(touched["changes"].str.contains(rf"\b{f}\b").sum()) for f in FIELDS}
        print(f"[tags] simulación -> {feats / 'tag_plan.csv'}: {len(touched)} archivos cambiarían; por campo {per_field}")
        return 0

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log = apply(plan, feats / f"tag_backup_{stamp}.csv")
    print(f"[tags] respaldo: {feats / f'tag_backup_{stamp}.csv'}; estados {log['status'].value_counts().to_dict()}")
    bad = log[log["status"].isin(["error", "verify_failed", "uid_changed"])]
    return 1 if len(bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
