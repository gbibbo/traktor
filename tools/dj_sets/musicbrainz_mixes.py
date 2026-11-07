"""
PURPOSE: Listas de temas de compilaciones mezcladas (DJ-mix) desde MusicBrainz, para el plan de playlists
         en vivo (docs/plans/recomendacion_playlists_plan.md, fuente D7). Por cada lanzamiento guarda
         <root>/musicbrainz/<release_id>.json: título, fecha, y por disco las pistas con posición,
         artista, título, duración y recording id. La duración de cada pista de un CD mezclado da el
         límite de cada tema dentro de la mezcla continua de ese disco.
           buscar   --artista "John Digweed"  -> los grupos DJ-mix que no son radio, con sus lanzamientos
           bajar    --release <mbid> [...]    -> el JSON de cada lanzamiento
           artistas --lista config/dj_sets/artistas_spotify.txt -> <root>/musicbrainz/artistas.json: por
                    artista, sus grupos DJ-mix que no son radio (para elegir qué compilaciones bajar)
         MusicBrainz pide una consulta por segundo e identificarse: Fetcher de tools/genre_taxonomy.
CHANGELOG:
  - 2026-11-07: Creación (piloto de trayectorias de DJs, DECISIONS 2026-11-07, puntos 12 y 13).
  - 2026-11-07: Subcomando artistas: recorre la lista de artistas de Gabriel.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from tools.genre_taxonomy.referencias import Fetcher  # noqa: E402

MB = "https://musicbrainz.org/ws/2"
DEFAULT_ROOT = REPO_ROOT / "artifacts" / "v4" / "dj_sets"


def credit(ac: List[dict]) -> str:
    return "".join(a.get("name", "") + a.get("joinphrase", "") for a in ac or [])


def release_summary(r: dict) -> Dict:
    """El JSON de /release/<id>?inc=recordings+artist-credits -> lo que usa el piloto."""
    media = []
    for m in r.get("media", []):
        tracks = [{"posicion": int(t.get("position") or 0), "artista": credit(t.get("artist-credit")),
                   "titulo": t.get("title", ""), "ms": t.get("length"),
                   "recording": (t.get("recording") or {}).get("id")} for t in m.get("tracks", [])]
        media.append({"disco": int(m.get("position") or 0), "formato": m.get("format"), "pistas": tracks})
    return {"id": r.get("id"), "titulo": r.get("title"), "artista": credit(r.get("artist-credit")),
            "fecha": r.get("date"), "media": media}


def search(f: Fetcher, artist: str, limit: int = 100) -> List[dict]:
    query = f'secondarytype:"dj-mix" AND NOT primarytype:broadcast AND artist:"{artist}"'
    d = f.json(f"{MB}/release-group?" + urllib.parse.urlencode({"query": query, "fmt": "json", "limit": limit}))
    return [{"id": g["id"], "titulo": g.get("title"), "fecha": g.get("first-release-date"),
             "tipos": [g.get("primary-type")] + (g.get("secondary-types") or [])} for g in d.get("release-groups", [])]


def main() -> int:
    ap = argparse.ArgumentParser(description="Compilaciones mezcladas (DJ-mix) desde MusicBrainz")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("buscar")
    b.add_argument("--artista", required=True)
    d = sub.add_parser("bajar")
    d.add_argument("--release", action="append", required=True)
    a = sub.add_parser("artistas")
    a.add_argument("--lista", default=str(REPO_ROOT / "config" / "dj_sets" / "artistas_spotify.txt"))
    for p in (b, d, a):
        p.add_argument("--root", default=str(DEFAULT_ROOT))
    args = ap.parse_args()
    f = Fetcher(pause=1.5)
    out = Path(args.root) / "musicbrainz"
    out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "artistas":
        names = [x.strip() for x in Path(args.lista).read_text(encoding="utf-8").splitlines()
                 if x.strip() and not x.lstrip().startswith("#")]
        res = []
        for i, name in enumerate(names, 1):
            groups = search(f, name)
            res.append({"artista": name, "grupos": groups})
            print(f"  {i}/{len(names)} {name}: {len(groups)} DJ-mix", flush=True)
        (out / "artistas.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        con = [r for r in res if r["grupos"]]
        print(f"[OK] {len(con)} de {len(res)} artistas con compilaciones DJ-mix; "
              f"{sum(len(r['grupos']) for r in res)} grupos en total")
        return 0
    if args.cmd == "buscar":
        groups = search(f, args.artista)
        (out / f"buscar_{args.artista.replace(' ', '_')}.json").write_text(
            json.dumps(groups, ensure_ascii=False, indent=1), encoding="utf-8")
        for g in groups:
            print(g["fecha"], "|", g["titulo"], "|", g["id"])
        return 0
    for rid in args.release:
        r = f.json(f"{MB}/release/{rid}?inc=recordings+artist-credits&fmt=json")
        s = release_summary(r)
        (out / f"{rid}.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
        tot = [sum((t["ms"] or 0) for t in m["pistas"]) / 60000 for m in s["media"]]
        print(f"[OK] {s['titulo']} ({s['fecha']}): {len(s['media'])} discos, "
              f"{sum(len(m['pistas']) for m in s['media'])} pistas, minutos por disco {[round(x, 1) for x in tot]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
