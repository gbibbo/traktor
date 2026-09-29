"""
PURPOSE: Temas repetidos desde la línea de comandos (regla en src/v4/common/duplicates.py; DECISIONS
         2026-09-29). Nunca borra ni mueve archivos:
           - candidates: grupos de posibles repetidos entre los temas de una organización que todavía no
             tienen decisión, con la capa, la calidad de cada copia, la que se quedaría y si se resuelve
             sola (--json para guardarlos);
           - decide: registrar un veredicto de Gabriel (--keep/--drop, o --distinct);
           - import: varios veredictos desde un JSON [{verdict, keep, tracks, layer, note}] con rutas
             relativas a la carpeta de música;
           - apply: sacar las copias descartadas de una organización (versión nueva 'dedupe', se deshace
             con organize.py / la app);
           - list: las decisiones guardadas.
CHANGELOG:
  - 2026-09-29: Creación inicial.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common import duplicates as dup  # noqa: E402
from src.v4.common.config_loader import load_config  # noqa: E402
from src.v4.common.path_resolver import resolve_dataset_artifacts, resolve_dataset_audio_root  # noqa: E402
from src.v4.common.tags import read_release_tags  # noqa: E402
from src.v4.pipeline import organize  # noqa: E402


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


def candidates(artifacts: Path, audio_root: Path, org_name: str) -> List[Dict]:
    store = organize.OrgStore(artifacts, org_name)
    lib = organize.Library(artifacts, store.meta()["params"]["rep"])
    assign, _, _ = store.load()
    uids = [u for u in assign["track_uid"] if u in lib.rep_index]
    cat = lib.catalog.reindex(uids)
    title = [t if isinstance(t, str) and t.strip() else Path(f).stem for t, f in zip(cat["title"], cat["filename"])]
    mix = [(read_release_tags(audio_root / r).get("tag_remixer") or dup.bracket_mix(t, Path(r).stem))
           for r, t in zip(cat["rel_path"], title)]
    groups = dup.find_candidates(uids, lib.emb(uids), cat["duration_s"].to_numpy(float), cat["artist"].fillna("").tolist(),
                                 title, mix, skip=dup.decided_pairs(dup.load_decisions(artifacts)))
    out = []
    for g in groups:
        rows = [{"track_uid": uids[i], "rel_path": cat["rel_path"].iloc[i], "dur": round(float(cat["duration_s"].iloc[i]), 1),
                 "quality": dup.read_quality(audio_root / cat["rel_path"].iloc[i])} for i in g["members"]]
        keep, auto = dup.resolve(g, [r["quality"] for r in rows])
        out.append({**{k: g[k] for k in ("layer", "sim", "dur_diff")}, "tracks": rows,
                    "keep": rows[keep]["track_uid"] if keep is not None else None, "auto": auto})
    return out


def apply(artifacts: Path, org_name: str) -> int:
    store = organize.OrgStore(artifacts, org_name)
    drop = dup.dropped(artifacts)
    n = sum(u in drop for u in store.load()[0]["track_uid"])
    return organize.remove_tracks(store, drop, detail={"n_decisions": len(dup.load_decisions(artifacts))}) if n else \
        store.meta()["current_version"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Temas repetidos: candidatos, veredictos y aplicación")
    parser.add_argument("--dataset-name", default="musica")
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--org-name", default="biblioteca")
    c.add_argument("--json", default=None)
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
    sub.add_parser("list")
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else None)
    artifacts = resolve_dataset_artifacts(args.dataset_name, config)
    audio_root = resolve_dataset_audio_root(args.dataset_name, config)
    if args.cmd == "candidates":
        groups = candidates(artifacts, audio_root, args.org_name)
        for k, g in enumerate(groups, 1):
            print(f"{k}. [{g['layer']}] similitud {g['sim']} · Δ {g['dur_diff']} s · "
                  f"{'se resuelve sola' if g['auto'] else 'hay que preguntar'}")
            for r in g["tracks"]:
                print(f"   {'★' if r['track_uid'] == g['keep'] else '·'} {dup.quality_label(r['quality']):<16} "
                      f"{r['dur']:>6.0f} s  {r['rel_path']}")
        if args.json:
            Path(args.json).write_text(json.dumps(groups, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[INFO] {len(groups)} grupos sin decisión ({sum(g['auto'] for g in groups)} se resuelven solos)")
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
    elif args.cmd == "list":
        for d in dup.load_decisions(artifacts):
            keep = next((t["rel_path"] for t in d["tracks"] if t["track_uid"] == d.get("keep")), None)
            print(f"{d['date']} {d['verdict']:<9} [{d['layer']}] se queda: {keep or '—'} · {len(d['tracks'])} temas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
