"""
PURPOSE: Baja el audio de las compilaciones mezcladas (D7) que youtube_buscar.py validó por duraciones:
         solo los archivos emparejados (cada disco entero en las «continua», cada pista en las
         «por_tema»), a <root>/audio/d7_<grupo>/ con un manifest.json que dice qué disco o qué pista de
         MusicBrainz es cada archivo. Antes de bajar vuelve a listar la playlist y comprueba otra vez la
         duración de cada archivo (por si la playlist cambió de orden). Lo ya bajado no se repite. Después,
         clap_blocks.py --carpeta d7_<grupo> saca los embeddings, y validar_d7.py revisa por título que cada
         archivo sea la compilación (la versión 3 de la búsqueda solo miraba duraciones).
           python tools/dj_sets/bajar_d7.py --root D:/... [--limite N] [--artista "Sasha"]
CHANGELOG:
  - 2026-11-07: Creación (escalar D7 después del piloto con Digweed).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from tools.dj_sets.youtube_audio import download, existing, list_playlist  # noqa: E402
from tools.dj_sets.youtube_buscar import DISC_TOL_S, TRACK_TOL_S  # noqa: E402

DEFAULT_ROOT = REPO_ROOT / "artifacts" / "v4" / "dj_sets"


def wanted(group: Dict, release: Dict, entries: List[Dict]) -> List[Dict]:
    """Los archivos a bajar con su disco o su pista; los que ya no coinciden por duración, fuera."""
    y = group["youtube"]
    out = []
    if y["tipo"] == "continua":
        for k, j in enumerate(y["archivo_por_disco"]):
            if j is None or j >= len(entries):
                continue
            want = sum((t["ms"] or 0) for t in release["media"][k]["pistas"]) / 1000
            d = entries[j].get("duration")
            if d and abs(d - want) <= DISC_TOL_S:
                out.append({"disco": k + 1, "pista": None, **entries[j]})
    else:
        tracks = [(m["disco"], t) for m in release["media"] for t in m["pistas"]]
        for (disc, t), j in zip(tracks, y["archivo_por_pista"]):
            if j is None or j >= len(entries):
                continue
            d = entries[j].get("duration")
            if d and abs(d - (t["ms"] or 0) / 1000) <= TRACK_TOL_S:
                out.append({"disco": disc, "pista": t["posicion"], **entries[j]})
    return out


def run_group(root: Path, gid: str, group: Dict, pause: float) -> Dict:
    y = group["youtube"]
    release = json.loads((root / "musicbrainz" / f"{y['release']}.json").read_text(encoding="utf-8"))
    entries = [e for e in list_playlist(f"https://www.youtube.com/playlist?list={y['playlist']}").get("entries") or []
               if e.get("id")]
    folder = root / "audio" / f"d7_{gid}"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, e in enumerate(wanted(group, release, entries), 1):
        f = existing(folder, e["id"])
        if f is None:
            f = download(f"https://www.youtube.com/watch?v={e['id']}", folder, f"{i:03d}_{e['id']}")
            time.sleep(pause)
        rows.append({"indice": i, "id": e["id"], "titulo": e.get("title"), "duracion": e.get("duration"),
                     "disco": e["disco"], "pista": e["pista"], "archivo": f.name if f else None})
    manifest = {"tipo": "d7", "grupo": gid, "artista": group["artista"], "titulo": group["grupo"]["titulo"],
                "release": y["release"], "playlist": y["playlist"], "forma": y["tipo"], "entradas": rows}
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Audio de las compilaciones D7 validadas")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--artista", action="append", default=None)
    ap.add_argument("--limite", type=int, default=None, help="Solo los primeros N grupos (prueba)")
    ap.add_argument("--pausa", type=float, default=2.0)
    args = ap.parse_args()
    root = Path(args.root)
    found = json.loads((root / "youtube" / "busqueda.json").read_text(encoding="utf-8"))
    groups = [(gid, g) for gid, g in found.items() if g.get("version") in (3, 4) and g["youtube"].get("estado") == "ok"]
    if args.artista:
        wanted_artists = {a.casefold() for a in args.artista}
        groups = [(gid, g) for gid, g in groups if g["artista"].casefold() in wanted_artists]
    groups = groups[: args.limite]
    hours = 0.0
    for n, (gid, g) in enumerate(groups, 1):
        try:
            m = run_group(root, gid, g, args.pausa)
        except RuntimeError as exc:
            print(f"  {n}/{len(groups)} {g['artista']} | {g['grupo']['titulo']}: [ERROR] {exc}", flush=True)
            continue
        ok = [r for r in m["entradas"] if r["archivo"]]
        h = sum(r["duracion"] or 0 for r in ok) / 3600
        hours += h
        print(f"  {n}/{len(groups)} {g['artista']} | {g['grupo']['titulo']}: {len(ok)}/{len(m['entradas'])} archivos, "
              f"{h:.1f} h ({m['forma']})", flush=True)
    print(f"[OK] {len(groups)} compilaciones, {hours:.0f} h de audio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
