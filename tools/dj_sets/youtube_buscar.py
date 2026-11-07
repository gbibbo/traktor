"""
PURPOSE: Busca en YouTube las compilaciones mezcladas (DJ-mix) de los artistas de Gabriel, para escalar la
         fuente D7 del plan de playlists en vivo. Por cada grupo de <root>/musicbrainz/artistas.json
         (musicbrainz_mixes.py artistas) baja de MusicBrainz sus lanzamientos con la duración de cada
         pista, busca playlists en YouTube y valida cada una por duraciones, porque los títulos no
         alcanzan (hay playlists de fans con «... cd1 (01)», genéricas o de otro DJ):
           por_tema   la playlist trae el set cortado por tema: cada pista del CD mezclado tiene un
                      archivo de la misma duración (±3 s). Da el orden, el límite y la identidad de cada
                      tema (la de MusicBrainz, por la posición);
           continua   la playlist trae la mezcla de cada disco entera: un archivo dura lo mismo que un
                      disco (±8 s: con Cattáneo, los aciertos difieren en 0 a 7 s y los discos de otro DJ
                      en 10 a 23 s, porque los CD duran casi todos 74 a 80 min). El límite de cada tema sale de las duraciones del CD,
                      como en el piloto con Digweed; si además hay originales sueltos, mejor.
         Escribe <root>/youtube/busqueda.json (grupo, lanzamiento elegido, playlist, tipo, cobertura y el
         emparejamiento pista-archivo) y guarda cada lanzamiento en <root>/musicbrainz/<id>.json. No baja
         audio: eso lo hace youtube_audio.py con la playlist elegida.
           python tools/dj_sets/youtube_buscar.py --root D:/... --artista "Sasha" --artista "Carl Cox"
CHANGELOG:
  - 2026-11-07: Creación (escalar D7 después del piloto con Digweed). Validación por duraciones contra
                MusicBrainz en lugar de los títulos; discos a ±8 s (no 1 %), con la diferencia anotada.
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from tools.dj_sets.musicbrainz_mixes import MB, release_summary  # noqa: E402
from tools.dj_sets.piloto import norm  # noqa: E402
from tools.dj_sets.youtube_audio import list_playlist, ytdlp  # noqa: E402
from tools.genre_taxonomy.referencias import Fetcher  # noqa: E402

DEFAULT_ROOT = REPO_ROOT / "artifacts" / "v4" / "dj_sets"
PLAYLIST_FILTER = "EgIQAw%3D%3D"
TRACK_TOL_S = 3.0
DISC_TOL_S = 8.0
MIN_TRACK_COVERAGE = 0.6


def match_tracks_by_duration(tracks_s: Sequence[float], entries_s: Sequence[Optional[float]],
                             tol: float = TRACK_TOL_S) -> List[Optional[int]]:
    """Para cada pista (en orden), el índice de un archivo de la misma duración (±tol) que no se haya
    usado; se prefiere el más cercano en posición al anterior emparejado (las playlists siguen el orden)."""
    used, out, last = set(), [], -1
    for t in tracks_s:
        cands = [j for j, d in enumerate(entries_s) if d and j not in used and abs(d - t) <= tol]
        if not cands:
            out.append(None)
            continue
        j = min(cands, key=lambda j: (j <= last, abs(j - last - 1), abs(entries_s[j] - t)))
        used.add(j)
        out.append(j)
        last = j
    return out


def match_discs(discs_s: Sequence[float], entries_s: Sequence[Optional[float]]) -> List[Optional[int]]:
    """Para cada disco, un archivo que dure lo mismo (±DISC_TOL_S)."""
    used, out = set(), []
    for t in discs_s:
        cands = [j for j, d in enumerate(entries_s) if d and j not in used and abs(d - t) <= DISC_TOL_S]
        j = min(cands, key=lambda j: abs(entries_s[j] - t)) if cands else None
        if j is not None:
            used.add(j)
        out.append(j)
    return out


def validate(release: Dict, entries: Sequence[Dict]) -> Dict:
    """Tipo y cobertura de una playlist de YouTube frente a un lanzamiento de MusicBrainz (release_summary)."""
    tracks = [t for m in release["media"] for t in m["pistas"]]
    t_s = [(t["ms"] or 0) / 1000 for t in tracks]
    d_s = [sum((t["ms"] or 0) for t in m["pistas"]) / 1000 for m in release["media"]]
    e_s = [e.get("duration") for e in entries]
    by_track = match_tracks_by_duration(t_s, e_s) if all(t_s) else [None] * len(t_s)
    by_disc = match_discs(d_s, e_s) if all(d_s) else [None] * len(d_s)
    cov_t = sum(j is not None for j in by_track) / len(by_track) if by_track else 0.0
    cov_d = sum(j is not None for j in by_disc) / len(by_disc) if by_disc else 0.0
    kind = "continua" if cov_d >= 0.5 else ("por_tema" if cov_t >= MIN_TRACK_COVERAGE else None)
    return {"tipo": kind, "cobertura_temas": round(cov_t, 3), "cobertura_discos": round(cov_d, 3),
            "pistas": len(tracks), "discos": len(d_s), "horas": round(sum(d_s) / 3600, 2),
            "archivo_por_pista": by_track, "archivo_por_disco": by_disc,
            "diferencia_por_disco_s": [None if j is None else round(e_s[j] - t, 1) for j, t in zip(by_disc, d_s)]}


def title_score(artist: str, title: str, candidate: str) -> float:
    got = norm(candidate)
    return max(difflib.SequenceMatcher(None, norm(f"{artist} {title}"), got).ratio(),
               difflib.SequenceMatcher(None, norm(title), got).ratio() * 0.95)


def search_playlists(query: str, n: int = 6) -> List[Dict]:
    url = "https://www.youtube.com/results?" + urllib.parse.urlencode({"search_query": query}) + f"&sp={PLAYLIST_FILTER}"
    p = ytdlp("--flat-playlist", "-J", "--playlist-end", str(n), url)
    if p.returncode != 0:
        return []
    return [e for e in (json.loads(p.stdout).get("entries") or []) if e.get("id")]


def releases_of(f: Fetcher, root: Path, group_id: str) -> List[Dict]:
    cache = root / "musicbrainz" / "grupos" / f"{group_id}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    d = f.json(f"{MB}/release?release-group={group_id}&inc=recordings+artist-credits&fmt=json&limit=25")
    rels = [release_summary(r) for r in d.get("releases", [])]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rels, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in rels:
        (root / "musicbrainz" / f"{r['id']}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return rels


def best_for(f: Fetcher, root: Path, artist: str, group: Dict, pause: float, max_lists: int = 5) -> Dict:
    rels = [r for r in releases_of(f, root, group["id"]) if r["media"] and all(m["pistas"] for m in r["media"])]
    if not rels:
        return {"estado": "sin lanzamiento con pistas en MusicBrainz"}
    seen, cands = set(), []
    for q in (f"{artist} {group['titulo']}", f"{group['titulo']} mixed by {artist}", f"{group['titulo']} DJ Mix"):
        for c in search_playlists(q):
            if c["id"] not in seen:
                seen.add(c["id"])
                cands.append(c)
        time.sleep(pause)
    cands = sorted(cands, key=lambda c: -title_score(artist, group["titulo"], c.get("title") or ""))[:max_lists]
    best = None
    for c in cands:
        try:
            d = list_playlist(f"https://www.youtube.com/playlist?list={c['id']}")
        except RuntimeError:
            continue
        time.sleep(pause)
        entries = [e for e in d.get("entries") or [] if e.get("id")]
        for r in rels:
            v = validate(r, entries)
            rank = (v["tipo"] == "continua", v["tipo"] == "por_tema", v["cobertura_discos"], v["cobertura_temas"])
            if v["tipo"] and (best is None or rank > best["_rank"]):
                best = {"_rank": rank, "playlist": c["id"], "titulo": d.get("title"),
                        "canal": d.get("channel") or d.get("uploader"), "archivos": len(entries),
                        "release": r["id"], "release_titulo": r["titulo"], **v}
    if best is None:
        return {"estado": "sin playlist que coincida por duraciones", "playlists_revisadas": len(cands)}
    best.pop("_rank")
    return {"estado": "ok", **best}


def main() -> int:
    ap = argparse.ArgumentParser(description="Compilaciones mezcladas en YouTube, validadas por duraciones (D7)")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--artista", action="append", default=None, help="Solo estos artistas (por defecto, todos)")
    ap.add_argument("--pausa", type=float, default=2.0)
    ap.add_argument("--rehacer", action="store_true", help="Volver a buscar los grupos ya revisados")
    args = ap.parse_args()
    root = Path(args.root)
    groups = json.loads((root / "musicbrainz" / "artistas.json").read_text(encoding="utf-8"))
    if args.artista:
        wanted = {a.casefold() for a in args.artista}
        groups = [g for g in groups if g["artista"].casefold() in wanted]
    out_path = root / "youtube" / "busqueda.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() and not args.rehacer else {}
    f = Fetcher(pause=1.5)
    for a in groups:
        for g in a["grupos"]:
            if g["id"] in done and done[g["id"]].get("version") == 3:
                continue
            res = best_for(f, root, a["artista"], g, args.pausa)
            done[g["id"]] = {"version": 3, "artista": a["artista"], "grupo": g, "youtube": res}
            tag = res["estado"] if res["estado"] != "ok" else (
                f"{res['tipo']}, {res['horas']} h, temas {res['cobertura_temas']:.0%}, discos {res['cobertura_discos']:.0%} "
                f"({res['titulo']} | {res['canal']})")
            print(f"  {a['artista']} | {g['titulo']}: {tag}", flush=True)
            out_path.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = [v["youtube"] for v in done.values() if v["youtube"].get("estado") == "ok"]
    print(f"[OK] {len(done)} grupos revisados; en YouTube {len(ok)}: "
          f"{sum(1 for v in ok if v['tipo'] == 'continua')} continuas y {sum(1 for v in ok if v['tipo'] == 'por_tema')} "
          f"por tema, {sum(v['horas'] for v in ok):.0f} h de sets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
