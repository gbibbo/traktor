"""
PURPOSE: Generar la página autónoma "Revisión de playlists" para que Gabriel evalúe, escuchando, si la
         organización en carpetas y playlists tiene sentido antes de importarla en Rekordbox/Traktor.
         La página muestra un mapa 2D (UMAP de Phase 2) con la carpeta y la playlist elegidas, cada
         playlist en su orden (BPM, tonalidad, energía, Vocal, género del tag y carpeta de origen),
         escucha de la secuencia (un fragmento de cada tema) y veredictos por playlist y por tema que
         se guardan en el navegador y se exportan a CSV.
         Se escribe DENTRO de la carpeta de audio del dataset (rutas relativas: el navegador reproduce
         los archivos locales sin servidor). Admite varias organizaciones (una por config de Phase 2):
         con --blind se muestran como A/B en orden aleatorio y la correspondencia se guarda aparte.
CHANGELOG:
  - 2026-09-27: Creación inicial. Veredictos guardados por dataset + config_hash (no por generación).
"""
import argparse
import datetime as dt
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.config_loader import load_config  # noqa: E402
from src.v4.common.path_resolver import resolve_dataset_artifacts, resolve_dataset_audio_root  # noqa: E402

TEMPLATE = Path(__file__).with_name("template.html")
PAGE_NAME = "_revision_playlists.html"
_VOCAL = re.compile(r"(?<!\w)Vocal(?!\w)", re.IGNORECASE)


def _letter(n: int) -> str:
    """0→'A', 25→'Z', 26→'AA' (igual que Phase 3/5)."""
    out = []
    while True:
        out.append(chr(ord("A") + n % 26))
        n = n // 26 - 1
        if n < 0:
            return "".join(reversed(out))


def short_folder(folder: str) -> str:
    """Carpeta de origen legible: sin el prefijo '#1 BIBO/PRO/' y como mucho las 2 últimas partes."""
    parts = [p for p in str(folder or "").split("/") if p]
    if parts[:2] == ["#1 BIBO", "PRO"]:
        parts = parts[2:]
    return " / ".join(parts[-2:])


def _num(v, nd: Optional[int] = None):
    if v is None or pd.isna(v):
        return None
    return round(float(v), nd) if nd is not None else int(v)


def build_tracks(catalog: pd.DataFrame, bpm_key: pd.DataFrame, uids: List[str]) -> List[Dict]:
    """Un registro compacto por tema, en el orden de uids."""
    cat = catalog.drop_duplicates("track_uid").set_index("track_uid")
    bk = bpm_key.drop_duplicates("track_uid").set_index("track_uid")
    out = []
    for u in uids:
        r = cat.loc[u]
        b = bk.loc[u] if u in bk.index else None
        comment = r.get("tag_comment")
        out.append({
            "u": u[:16],
            "p": str(r["rel_path"]),
            "a": "" if pd.isna(r.get("artist")) else str(r.get("artist")),
            "t": "" if pd.isna(r.get("title")) else str(r.get("title")),
            "g": "" if pd.isna(r.get("tag_genre")) else str(r.get("tag_genre")),
            "fs": short_folder(r.get("folder", "")),
            "b": _num(b["bpm"], 1) if b is not None else None,
            "k": None if b is None or pd.isna(b["key"]) else str(b["key"]),
            "e": _num(b["energy"]) if b is not None and "energy" in b.index else None,
            "v": bool(isinstance(comment, str) and _VOCAL.search(comment)),
        })
    return out


def load_org(artifacts: Path, config_hash: str) -> Dict:
    """Carpetas (L1) y playlists (L2) ordenadas de una corrida de Phases 2-4, más el UMAP."""
    cdir = artifacts / "clustering"
    ordered = pd.read_parquet(cdir / f"ordered_{config_hash}.parquet")
    names = json.loads((cdir / f"names_{config_hash}.json").read_text(encoding="utf-8"))
    cfg = json.loads((cdir / f"config_{config_hash}.json").read_text(encoding="utf-8"))
    if not {"umap_x", "umap_y"} <= set(ordered.columns) or not ordered[["umap_x", "umap_y"]].abs().to_numpy().any():
        raise ValueError(f"ordered_{config_hash} no tiene UMAP: correr phase2_cluster.py sin --skip-umap")
    return {"hash": config_hash, "rep": cfg.get("rep", "mert"), "ordered": ordered, "names": names}


def org_payload(org: Dict, uid_index: Dict[str, int], label: str, n_suggested: int = 8) -> Dict:
    """id = config_hash: los veredictos guardados en el navegador siguen a la organización aunque se
    regenere la página o cambie la letra A/B."""
    org_id = org["hash"]
    df = org["ordered"].sort_values(["label_l1", "label_l2", "position"])
    folders = []
    for l1, g1 in df.groupby("label_l1", sort=True):
        letter = "Ruido" if l1 < 0 else _letter(int(l1))
        l1_name = org["names"].get(f"l1_{l1}", f"Group {letter}")
        fname = f"{letter} · {l1_name}" if not l1_name.startswith("Group") else letter
        pls = []
        for l2, g2 in g1.groupby("label_l2", sort=True):
            pname = org["names"].get(f"l1_{l1}_l2_{l2}", f"{letter}{int(l2) + 1}" if l2 >= 0 else f"{letter} varios")
            pls.append({"id": f"{org_id}:{l1}:{l2}", "name": pname,
                        "tracks": [uid_index[u] for u in g2["track_uid"]], "suggested": False})
        folders.append({"id": f"{org_id}:{l1}", "name": fname, "playlists": pls})
    # Sugeridas: la playlist más grande de cada una de las n carpetas más grandes
    by_size = sorted(folders, key=lambda f: -sum(len(p["tracks"]) for p in f["playlists"]))
    for f in by_size[:n_suggested]:
        max(f["playlists"], key=lambda p: len(p["tracks"]))["suggested"] = True
    xy_df = df.set_index("track_uid")[["umap_x", "umap_y"]]
    order = list(xy_df.index)
    return {"id": org_id, "label": label, "folders": folders,
            "trackIdx": [uid_index[u] for u in order],
            "xy": [[round(float(x), 4), round(float(y), 4)] for x, y in xy_df.to_numpy()]}


def render(data: Dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8")
    assert "/*__DATA__*/null" in html
    return html.replace("/*__DATA__*/null", payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Página HTML para revisar la organización escuchando")
    parser.add_argument("--dataset-name", default="musica")
    parser.add_argument("--config", default=None)
    parser.add_argument("--org", action="append", default=[],
                        help="config_hash de Phase 2 (repetible). Default: el ordered_*.parquet más reciente.")
    parser.add_argument("--blind", action="store_true", help="Mostrar las organizaciones como A/B en orden aleatorio")
    parser.add_argument("--seed", type=int, default=None, help="Semilla del orden a ciegas")
    parser.add_argument("--out", default=None, help=f"Default: <carpeta de audio>/{PAGE_NAME}")
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else None)
    artifacts = resolve_dataset_artifacts(args.dataset_name, config)
    hashes = args.org or [sorted((artifacts / "clustering").glob("ordered_*.parquet"),
                                 key=lambda p: p.stat().st_mtime)[-1].stem.replace("ordered_", "")]
    orgs = [load_org(artifacts, h) for h in hashes]
    catalog = pd.read_parquet(artifacts / "catalog.parquet")
    bpm_key = pd.read_parquet(artifacts / "features" / "bpm_key.parquet")

    uids = list(dict.fromkeys(u for o in orgs for u in o["ordered"]["track_uid"]))
    uid_index = {u: i for i, u in enumerate(uids)}
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M")

    if args.blind and len(orgs) > 1:
        rng = random.Random(args.seed)
        rng.shuffle(orgs)
        labels = [f"Organización {_letter(i)}" for i in range(len(orgs))]
        key = {lab: {"config_hash": o["hash"], "rep": o["rep"]} for lab, o in zip(labels, orgs)}
        key_path = artifacts / "evaluation" / f"review_key_{run_id}.json"
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_text(json.dumps(key, indent=2), encoding="utf-8")
        print(f"[INFO] A ciegas: correspondencia en {key_path} (no está en la página)")
    else:
        labels = [f"{o['rep']} ({o['hash']})" if len(orgs) > 1 else "Organización actual" for o in orgs]

    data = {"run_id": run_id, "dataset": args.dataset_name,
            "tracks": build_tracks(catalog, bpm_key, uids),
            "orgs": [org_payload(o, uid_index, lab) for o, lab in zip(orgs, labels)]}
    out = Path(args.out) if args.out else resolve_dataset_audio_root(args.dataset_name, config) / PAGE_NAME
    out.write_text(render(data), encoding="utf-8")
    n_pl = sum(len(f["playlists"]) for o in data["orgs"] for f in o["folders"])
    print(f"[INFO] {len(uids)} temas, {len(orgs)} organización(es), {n_pl} playlists → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
