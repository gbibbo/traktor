"""
PURPOSE: Temas repetidos desde la línea de comandos y para la app (regla en src/v4/common/duplicates.py;
         DECISIONS 2026-09-29). Nunca borra archivos:
           - candidates: grupos de posibles repetidos que todavía no tienen decisión, con la capa, la
             calidad de cada copia, la que se quedaría y si se resuelve sola (--json para guardarlos);
           - auto: registra como decisión de la regla los grupos que se resuelven solos (sonido y
             nombre); los demás quedan como temas separados;
           - decide / import: veredictos de Gabriel (--keep/--drop o --distinct; o un JSON);
           - apply: saca las copias descartadas de una organización (versión 'dedupe', se deshace);
           - move-copies: mueve las copias descartadas a <música>/_copias/<misma ruta> y corrige el
             catálogo, las decisiones y duplicates.csv; restore-copies las devuelve (moved_copies.csv);
           - list: las decisiones guardadas.
         La app (review_app.py) corre auto + apply + move-copies al agregar música nueva.
CHANGELOG:
  - 2026-09-29: Creación inicial.
  - 2026-09-29: auto, move-copies / restore-copies y candidatos que incluyen la música nueva (scope).
                move-copies no saca archivos de la carpeta de otro dataset (test_20 es un enlace a
                «Música/2020 new - copia»); restore-copies --path devuelve solo algunas.
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

from src.v4.common import duplicates as dup  # noqa: E402
from src.v4.common.config_loader import load_config  # noqa: E402
from src.v4.common.path_resolver import resolve_dataset_artifacts, resolve_dataset_audio_root  # noqa: E402
from src.v4.common.tags import read_release_tags  # noqa: E402
from src.v4.pipeline import organize  # noqa: E402

COPIES_DIR = "_copias"
MOVED_FILE = "moved_copies.csv"


def _catalog(artifacts: Path) -> pd.DataFrame:
    return pd.read_parquet(artifacts / "catalog.parquet").drop_duplicates("track_uid")


def _tracks_by_rel(cat: pd.DataFrame, rels: List[str]) -> List[Dict]:
    by_rel = dict(zip(cat["rel_path"], cat["track_uid"]))
    out = []
    for r in rels:
        r = r.replace("\\", "/").removeprefix("Música/")
        if r not in by_rel:
            raise ValueError(f"No está en el catálogo: {r}")
        out.append({"track_uid": by_rel[r], "rel_path": r})
    return out


def candidates(artifacts: Path, audio_root: Path, org_name: str, scope: Optional[str] = None) -> List[Dict]:
    """Grupos sin decisión entre los temas de la organización; con scope, también los temas de esa carpeta
    que todavía no están en ella, y solo los grupos que los tocan (música nueva)."""
    store = organize.OrgStore(artifacts, org_name)
    lib = organize.Library(artifacts, store.meta()["params"]["rep"])
    in_org = [u for u in store.load()[0]["track_uid"] if u in lib.rep_index and u not in lib.excluded]
    new = [u for u in lib.in_scope(scope) if u not in set(in_org)] if scope else []
    uids = in_org + new
    if len(uids) < 2:
        return []
    cat = lib.catalog.reindex(uids)
    title = [t if isinstance(t, str) and t.strip() else Path(str(f)).stem for t, f in zip(cat["title"], cat["filename"])]
    mix = [(read_release_tags(audio_root / r).get("tag_remixer") or dup.bracket_mix(t, Path(r).stem))
           for r, t in zip(cat["rel_path"], title)]
    groups = dup.find_candidates(uids, lib.emb(uids), cat["duration_s"].to_numpy(float), cat["artist"].fillna("").tolist(),
                                 title, mix, skip=dup.decided_pairs(dup.load_decisions(artifacts)),
                                 focus=set(new) if scope else None)
    out = []
    for g in groups:
        rows = [{"track_uid": uids[i], "rel_path": cat["rel_path"].iloc[i], "dur": round(float(cat["duration_s"].iloc[i]), 1),
                 "quality": dup.read_quality(audio_root / cat["rel_path"].iloc[i])} for i in g["members"]]
        keep, auto = dup.resolve(g, [r["quality"] for r in rows], [r["rel_path"] for r in rows])
        out.append({**{k: g[k] for k in ("layer", "sim", "dur_diff")}, "tracks": rows,
                    "keep": rows[keep]["track_uid"] if keep is not None else None, "auto": auto})
    return out


def auto_resolve(artifacts: Path, audio_root: Path, org_name: str, scope: Optional[str] = None) -> Dict:
    """Registra los grupos que la regla resuelve sola; devuelve {resolved, pending} (grupos)."""
    groups = candidates(artifacts, audio_root, org_name, scope)
    for g in groups:
        if g["auto"]:
            dup.add_decision(artifacts, "mismo", g["tracks"], keep=g["keep"], layer=g["layer"], by="regla",
                             note=f"similitud {g['sim']}")
    return {"resolved": [g for g in groups if g["auto"]], "pending": [g for g in groups if not g["auto"]]}


def apply(artifacts: Path, org_name: str) -> int:
    store = organize.OrgStore(artifacts, org_name)
    drop = dup.dropped(artifacts)
    n = sum(u in drop for u in store.load()[0]["track_uid"])
    return organize.remove_tracks(store, drop, detail={"n_decisions": len(dup.load_decisions(artifacts))}) if n else \
        store.meta()["current_version"]


# ---------------------------------------------------------------------------
# Mover las copias a _copias (y volverlas)
# ---------------------------------------------------------------------------

def _relink(artifacts: Path, audio_root: Path, mapping: Dict[str, str]) -> None:
    """Rutas nuevas en catalog.parquet, duplicate_decisions.json y duplicates.csv (mapping: vieja -> nueva)."""
    if not mapping:
        return
    cat = pd.read_parquet(artifacts / "catalog.parquet")
    m = cat["rel_path"].isin(mapping)
    cat.loc[m, "rel_path"] = cat.loc[m, "rel_path"].map(mapping)
    cat.loc[m, "source_path"] = [str(Path(audio_root).resolve() / r) for r in cat.loc[m, "rel_path"]]
    cat.loc[m, "folder"] = [Path(r).parent.as_posix() if Path(r).parent != Path(".") else "" for r in cat.loc[m, "rel_path"]]
    cat.to_parquet(artifacts / "catalog.parquet", index=False)
    decisions = dup.load_decisions(artifacts)
    for d in decisions:
        for t in d["tracks"]:
            t["rel_path"] = mapping.get(t["rel_path"], t["rel_path"])
    dup.save_decisions(artifacts, decisions)
    dpath = artifacts / "duplicates.csv"
    if dpath.exists():
        dd = pd.read_csv(dpath)
        for col in ("rel_path", "kept_rel_path"):
            dd[col] = dd[col].map(lambda r: mapping.get(r, r))
        dd.to_csv(dpath, index=False)


def protected_roots(config: Dict, dataset: str) -> List[Path]:
    """Carpetas de audio de los otros datasets (resueltas: test_20 es un enlace a una carpeta de Música).
    Sus archivos no se mueven: el dataset perdería temas."""
    out = []
    for name in (config.get("datasets") or {}):
        if name == dataset:
            continue
        try:
            out.append(resolve_dataset_audio_root(name, config).resolve())
        except FileNotFoundError:
            continue
    return out


def move_copies(artifacts: Path, audio_root: Path, dry_run: bool = False,
                protected: Optional[List[Path]] = None) -> Dict:
    """Mueve cada copia descartada a <música>/_copias/<su ruta>. No pisa nada: si el destino existe, el
    archivo no está o pertenece a la carpeta de otro dataset (protected), la saltea. Registra cada
    movimiento en moved_copies.csv (para restore_copies)."""
    audio_root = Path(audio_root)
    protected = [Path(x).resolve() for x in (protected or [])]
    cat = _catalog(artifacts)
    drop = dup.dropped(artifacts)
    todo, skipped = [], []
    for u, rel in zip(cat["track_uid"], cat["rel_path"]):
        if u not in drop or rel.startswith(COPIES_DIR + "/"):
            continue
        src, dst = audio_root / rel, audio_root / COPIES_DIR / rel
        item = {"track_uid": u, "from": rel, "to": f"{COPIES_DIR}/{rel}"}
        owner = next((p for p in protected if src.resolve().is_relative_to(p)), None)
        if owner is not None:
            skipped.append({**item, "error": f"(es parte de otro dataset: {owner.name})"})
        else:
            (todo if src.is_file() and not dst.exists() else skipped).append(item)
    if dry_run:
        return {"moved": todo, "skipped": skipped}
    moved = []
    for mv in todo:
        dst = audio_root / mv["to"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(audio_root / mv["from"], dst)
            moved.append({**mv, "date": dt.datetime.now().isoformat(timespec="seconds")})
        except OSError as exc:  # archivo en uso (Rekordbox/Traktor abiertos), permisos
            skipped.append({**mv, "error": str(exc)})
    _relink(artifacts, audio_root, {m["from"]: m["to"] for m in moved})
    if moved:
        log = artifacts / MOVED_FILE
        prev = pd.read_csv(log) if log.exists() else pd.DataFrame(columns=["track_uid", "from", "to", "date"])
        pd.concat([prev, pd.DataFrame(moved)], ignore_index=True).to_csv(log, index=False)
    return {"moved": moved, "skipped": skipped}


def restore_copies(artifacts: Path, audio_root: Path, only: Optional[List[str]] = None) -> Dict:
    """Devuelve a su lugar las copias de moved_copies.csv (only: solo esas rutas originales); las que no se
    pueden, quedan en el registro."""
    audio_root = Path(audio_root)
    log = artifacts / MOVED_FILE
    if not log.exists():
        return {"restored": [], "left": []}
    rows = pd.read_csv(log).to_dict("records")
    restored, left = [], []
    for r in reversed(rows):
        src, dst = audio_root / r["to"], audio_root / r["from"]
        if only is not None and r["from"] not in only:
            left.append(r)
        elif src.is_file() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(src, dst)
            restored.append(r)
        else:
            left.append(r)
    _relink(artifacts, audio_root, {r["to"]: r["from"] for r in restored})
    pd.DataFrame(list(reversed(left)), columns=["track_uid", "from", "to", "date"]).to_csv(log, index=False)
    return {"restored": restored, "left": left}


def main() -> int:
    parser = argparse.ArgumentParser(description="Temas repetidos: candidatos, veredictos y aplicación")
    parser.add_argument("--dataset-name", default="musica")
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--org-name", default="biblioteca")
    c.add_argument("--scope", default=None, help="carpeta con música nueva (solo grupos que la tocan)")
    c.add_argument("--json", default=None)
    au = sub.add_parser("auto")
    au.add_argument("--org-name", default="biblioteca")
    au.add_argument("--scope", default=None)
    d = sub.add_parser("decide")
    d.add_argument("--keep", default=None, help="ruta (relativa a la carpeta de música) de la copia que se queda")
    d.add_argument("--drop", action="append", default=[], help="ruta de una copia que sale (repetible)")
    d.add_argument("--distinct", nargs="+", default=None, help="rutas de temas que NO son el mismo")
    d.add_argument("--layer", default="")
    d.add_argument("--by", default="gabriel")
    d.add_argument("--note", default="")
    i = sub.add_parser("import")
    i.add_argument("--file", required=True)
    a = sub.add_parser("apply")
    a.add_argument("--org-name", default="biblioteca")
    mc = sub.add_parser("move-copies")
    mc.add_argument("--dry-run", action="store_true")
    rc = sub.add_parser("restore-copies")
    rc.add_argument("--path", action="append", default=None, help="ruta original de una copia (repetible; sin esto, todas)")
    sub.add_parser("list")
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else None)
    artifacts = resolve_dataset_artifacts(args.dataset_name, config)
    audio_root = resolve_dataset_audio_root(args.dataset_name, config)
    if args.cmd == "candidates":
        groups = candidates(artifacts, audio_root, args.org_name, args.scope)
        for k, g in enumerate(groups, 1):
            print(f"{k}. [{g['layer']}] similitud {g['sim']} · Δ {g['dur_diff']} s · "
                  f"{'se resuelve sola' if g['auto'] else 'queda para decidir'}")
            for r in g["tracks"]:
                print(f"   {'★' if r['track_uid'] == g['keep'] else '·'} {dup.quality_label(r['quality']):<16} "
                      f"{r['dur']:>6.0f} s  {r['rel_path']}")
        if args.json:
            Path(args.json).write_text(json.dumps(groups, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[INFO] {len(groups)} grupos sin decisión ({sum(g['auto'] for g in groups)} se resuelven solos)")
    elif args.cmd == "auto":
        r = auto_resolve(artifacts, audio_root, args.org_name, args.scope)
        print(f"[INFO] {len(r['resolved'])} grupos resueltos por la regla; {len(r['pending'])} quedan para decidir")
    elif args.cmd == "decide":
        cat = _catalog(artifacts)
        if args.distinct:
            e = dup.add_decision(artifacts, "distintos", _tracks_by_rel(cat, args.distinct), layer=args.layer,
                                 by=args.by, note=args.note)
        else:
            tracks = _tracks_by_rel(cat, [args.keep, *args.drop])
            e = dup.add_decision(artifacts, "mismo", tracks, keep=tracks[0]["track_uid"], layer=args.layer,
                                 by=args.by, note=args.note)
        print(f"[INFO] Decisión guardada ({e['verdict']}, {len(e['tracks'])} temas)")
    elif args.cmd == "import":
        cat = _catalog(artifacts)
        items = json.loads(Path(args.file).read_text(encoding="utf-8"))
        for it in items:
            tracks = _tracks_by_rel(cat, it["tracks"])
            keep = _tracks_by_rel(cat, [it["keep"]])[0]["track_uid"] if it["verdict"] == "mismo" else None
            dup.add_decision(artifacts, it["verdict"], tracks, keep=keep, layer=it.get("layer", ""),
                             by=it.get("by", "gabriel"), note=it.get("note", ""))
        print(f"[INFO] {len(items)} decisiones importadas; copias descartadas en total: {len(dup.dropped(artifacts))}")
    elif args.cmd == "apply":
        v = apply(artifacts, args.org_name)
        print(f"[INFO] Organización '{args.org_name}': versión actual {v}")
    elif args.cmd == "move-copies":
        r = move_copies(artifacts, audio_root, args.dry_run, protected_roots(config, args.dataset_name))
        for m in r["moved"]:
            print(f"   {'(prueba) ' if args.dry_run else ''}{m['from']}  ->  {m['to']}")
        for m in r["skipped"]:
            print(f"   salteada: {m['from']} {m.get('error', '(no está o el destino ya existe)')}")
        print(f"[INFO] {len(r['moved'])} copias {'se moverían' if args.dry_run else 'movidas'} a {COPIES_DIR}; "
              f"{len(r['skipped'])} salteadas")
    elif args.cmd == "restore-copies":
        r = restore_copies(artifacts, audio_root, args.path)
        print(f"[INFO] {len(r['restored'])} copias devueltas a su lugar; {len(r['left'])} quedaron en el registro")
    elif args.cmd == "list":
        for d in dup.load_decisions(artifacts):
            keep = next((t["rel_path"] for t in d["tracks"] if t["track_uid"] == d.get("keep")), None)
            print(f"{d['date']} {d['verdict']:<9} [{d['layer']}] {d['by']:<7} se queda: {keep or '—'} · {len(d['tracks'])} temas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
