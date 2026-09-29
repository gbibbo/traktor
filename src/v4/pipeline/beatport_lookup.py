"""
PURPOSE: Busca en Beatport cada tema de un dataset y guarda lo encontrado como tabla aparte, SIN
         tocar los archivos de audio:
           - features/<output>.parquet y .csv (UTF-8 con BOM, para Excel): nivel de match (A = ISRC
             exacto, B = misma versión, C = solo otras versiones, D = nada), datos de Beatport
             (género, sello, fecha, mezcla, artistas, remixers, ISRC, BPM, tonalidad, URL) y los
             tags que se propondrían (Artist, Title, Remixers, Label, Genre, Released) al lado de
             los actuales;
           - features/<output>_summary.json: conteos por nivel, coincidencias con los tags actuales
             y cuántos archivos cambiarían en cada campo.
         Solo A y B llevan género de Beatport; C y D quedan para el modelo (genre_model.py).
         confirmed = A, o B con sello, fecha o duración iguales a las del archivo. Beatport asigna el
         género por lanzamiento (un compilado puede cambiarlo): Genre sale del lanzamiento que
         coincide con el archivo y bp_original_genre guarda el del lanzamiento más antiguo.
         Las búsquedas se guardan en artifacts/v4/beatport_cache (compartida entre datasets): repetir
         la corrida no vuelve a preguntar. --limit / --folder para la corrida chica de prueba.
CHANGELOG:
  - 2026-09-29: Creación inicial.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.beatport import (  # noqa: E402
    BeatportBlocked, BeatportClient, MatchResult, TrackQuery, clean_text, match, norm, proposed_tags,
    is_version_text, split_title_mix,
)
from src.v4.common.catalog import load_catalog  # noqa: E402
from src.v4.common.config_loader import load_config  # noqa: E402
from src.v4.common.path_resolver import resolve_artifacts_root, resolve_dataset_artifacts  # noqa: E402
from src.v4.common.tags import read_release_tags  # noqa: E402

# Campos que se escribirían (pedido de Gabriel). El título de Beatport queda solo como referencia.
FIELDS = ("artist", "remixers", "label", "genre", "released")
MAX_QUERIES = 6


def _s(v) -> str:
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()


def _split_artist_title(text: str) -> Tuple[str, str]:
    text = clean_text(text)
    if " - " in text:
        artist, title = text.split(" - ", 1)
        return artist.strip(), title.strip()
    return "", text


def build_queries(row: pd.Series, rel: Dict[str, Optional[str]]) -> List[Tuple[TrackQuery, bool]]:
    """Variantes de consulta, de la más confiable a la menos; (consulta, buscar en Beatport):
      1. tags del archivo; si el título trae "Artista - Tema" manda esa parte (en los rips de YouTube
         el tag de artista suele ser el canal: "Anjunadeep", "Toolroom Records");
      2. nombre del archivo;
      3. artista y título invertidos ("Finder - Carl Cox"): reusa los candidatos ya traídos."""
    mix_tag = _s(rel.get("tag_remixer"))
    if mix_tag and not is_version_text(mix_tag):   # TPE4 con solo el nombre del remixer
        mix_tag = f"{mix_tag} Remix"
    common = dict(isrcs=tuple(x for x in [_s(rel.get("tag_isrc")).upper()] if x),
                  label=_s(row.get("tag_label")) or None, date=_s(rel.get("tag_date")) or None,
                  duration_s=float(row["duration_s"]) if pd.notna(row.get("duration_s")) else None)
    stem_artist, stem_title = _split_artist_title(Path(_s(row.get("filename"))).stem)
    tag_artist, tag_title = _s(row.get("tag_artist")), _s(row.get("tag_title"))
    pairs: List[Tuple[str, str, bool]] = []
    if tag_title:
        in_title_artist, in_title = _split_artist_title(tag_title)
        pairs.append((in_title_artist or tag_artist or stem_artist, in_title, True))
    pairs.append((stem_artist or tag_artist, stem_title, True))
    if pairs[0][0]:
        pairs.append((pairs[0][1], pairs[0][0], False))
    out: List[Tuple[TrackQuery, bool]] = []
    seen = set()
    for artist, title, fetch in pairs:
        base, mix_in_title = split_title_mix(title)
        key = (norm(artist), norm(base))
        if not base or key in seen:
            continue
        seen.add(key)
        mix = mix_tag or mix_in_title or split_title_mix(stem_title)[1]
        out.append((TrackQuery(artists=clean_text(artist), title=base, mix=mix, **common), fetch))
    return out


_LEVEL_RANK = {"A": 3, "B": 2, "C": 1, "D": 0}


def _confident(res: MatchResult) -> bool:
    return res.level == "A" or (res.level == "B" and (res.label_match or res.date_match or res.same_cut))


def lookup(client: BeatportClient, variants: List[Tuple[TrackQuery, bool]]
           ) -> Tuple[MatchResult, TrackQuery, List[str]]:
    """Recorre las variantes hasta un match confiable (A, o B confirmado por sello, fecha o duración);
    si no llega, devuelve el mejor nivel alcanzado. A lo sumo MAX_QUERIES pedidos por tema."""
    cands: List[dict] = []
    used: List[str] = []
    best, best_q = None, variants[0][0]
    for q, fetch in variants:
        for query in (q.search_queries() if fetch else []):
            if len(used) >= MAX_QUERIES or query in used:
                continue
            used.append(query)
            cands += client.search(query)
            if _confident(match(q, cands)):
                break
        res = match(q, cands)
        if best is None or (_LEVEL_RANK[res.level], _confident(res)) > (_LEVEL_RANK[best.level], _confident(best)):
            best, best_q = res, q
        if _confident(best):
            break
    return best or MatchResult(level="D"), best_q, used


def record(row: pd.Series, rel: Dict[str, Optional[str]], q: TrackQuery, res: MatchResult,
           queries: List[str]) -> dict:
    b = res.best or {}
    new = proposed_tags(q, res) if res.level in ("A", "B") else {}
    rec = {
        "track_uid": row["track_uid"], "rel_path": _s(row.get("rel_path")),
        "cur_artist": _s(row.get("tag_artist")), "cur_title": _s(row.get("tag_title")),
        "cur_remixers": _s(rel.get("tag_remixer")), "cur_label": _s(row.get("tag_label")),
        "cur_genre": _s(row.get("tag_genre")), "cur_released": _s(rel.get("tag_date")),
        "q_artist": q.artists, "q_title": q.title, "q_mix": q.mix, "q_isrc": "; ".join(q.isrcs),
        "level": res.level, "confirmed": _confident(res), "n_candidates": res.n_candidates, "n_pool": res.n_pool,
        "label_match": res.label_match, "date_match": res.date_match, "same_cut": res.same_cut,
        "title_sim": round(res.title_sim, 3), "artist_score": round(res.artist_score, 3),
        "bp_track_id": b.get("track_id"), "bp_title": b.get("title"), "bp_mix": b.get("mix"),
        "bp_artists": "; ".join(b.get("artists") or []), "bp_remixers": "; ".join(b.get("remixers") or []),
        "bp_label": b.get("label"), "bp_release_date": b.get("release_date"), "bp_genre": b.get("genre"),
        "bp_isrc": b.get("isrc"), "bp_bpm": b.get("bpm"), "bp_key": b.get("key"),
        "bp_length_s": b.get("length_s"), "bp_url": b.get("url"),
        "same_version_genres": "; ".join(res.same_version_genres), "genre_conflict": res.genre_conflict,
        "bp_original_genre": res.original_genre,
        "hint_genres": "; ".join(res.hint_genres),
        "queries": " | ".join(queries),
    }
    for f in FIELDS:
        rec[f"new_{f}"] = new.get(f)
    return rec


def summarize_run(df: pd.DataFrame) -> dict:
    n = len(df)
    ab = df[df["level"].isin(["A", "B"])]
    out: dict = {"n_tracks": n, "levels": df["level"].value_counts().reindex(list("ABCD"), fill_value=0).to_dict(),
                 "B_confirmed": int(((df["level"] == "B") & df["confirmed"]).sum())}
    changes, filled = {}, {}
    for f in FIELDS:
        cur = ab[f"cur_{f}"].map(norm)
        new = ab[f"new_{f}"].map(lambda v: norm(_s(v)))
        changes[f] = int(((cur != new) & (new != "")).sum())
        filled[f] = int(((cur == "") & (new != "")).sum())
    out["would_change"] = changes
    out["would_fill_empty"] = filled
    has_genre = ab[ab["cur_genre"] != ""]
    out["genre_equal_to_current"] = {"equal": int((has_genre["cur_genre"].map(norm) == has_genre["new_genre"].map(lambda v: norm(_s(v)))).sum()),
                                     "of": len(has_genre)}
    for col in ("label_match", "date_match", "same_cut"):
        out[col] = int(ab[col].sum())
    out["genre_conflict"] = int(ab["genre_conflict"].sum())
    out["genre_differs_from_original_release"] = int((ab["bp_original_genre"].notna()
                                                      & (ab["bp_original_genre"] != ab["new_genre"])).sum())
    out["extended_to_original_mix"] = int(((ab["cur_remixers"].str.contains("extended", case=False))
                                           & (ab["new_remixers"] == "Original Mix")).sum())
    out["remixers_extended_mix"] = int((ab["new_remixers"].fillna("").str.contains("extended", case=False)).sum())
    out["beatport_genres"] = ab["new_genre"].value_counts().to_dict()
    diff = has_genre[has_genre["cur_genre"].map(norm) != has_genre["new_genre"].map(lambda v: norm(_s(v)))]
    out["genre_changes_top"] = (diff.groupby(["cur_genre", "new_genre"]).size().sort_values(ascending=False)
                                .head(25).reset_index().apply(lambda r: f"{r['cur_genre']} -> {r['new_genre']}: {r[0]}", axis=1)
                                .tolist())
    return out


def save(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / f"{name}.parquet", index=False)
    df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="Busca los temas de un dataset en Beatport (sin escribir tags)")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Solo los primeros N temas (prueba)")
    parser.add_argument("--folder", default=None, help="Solo temas cuyo rel_path contiene este texto (prueba)")
    parser.add_argument("--output", default="beatport", help="Nombre base de la salida en features/")
    parser.add_argument("--min-interval", type=float, default=1.2, help="Segundos mínimos entre pedidos")
    parser.add_argument("--offline", action="store_true", help="Solo la caché, sin pedidos nuevos")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--low-priority", action="store_true", help="Prioridad baja del proceso (corridas largas)")
    args = parser.parse_args()

    if args.low_priority:
        from src.v4.pipeline.extract_representations import lower_priority
        lower_priority()
    config = load_config(Path(args.config) if args.config else None)
    artifacts = resolve_dataset_artifacts(args.dataset_name, config)
    catalog = load_catalog(args.dataset_name, config)
    if args.folder:
        catalog = catalog[catalog["rel_path"].str.contains(args.folder, case=False, regex=False)]
    if args.limit:
        catalog = catalog.head(args.limit)
    client = BeatportClient(resolve_artifacts_root(config).parent / "beatport_cache",
                            min_interval=args.min_interval, offline=args.offline)
    out_dir = artifacts / "features"
    print(f"[beatport] {args.dataset_name}: {len(catalog)} temas -> {out_dir / args.output}.parquet")

    rows: List[dict] = []
    t0 = time.monotonic()
    status = 0
    try:
        for i, (_, row) in enumerate(catalog.iterrows(), 1):
            rel = read_release_tags(Path(row["source_path"]))
            res, q, used = lookup(client, build_queries(row, rel))
            rows.append(record(row, rel, q, res, used))
            if i % 25 == 0 or i == len(catalog):
                el = time.monotonic() - t0
                lv = pd.Series([r["level"] for r in rows]).value_counts().to_dict()
                print(f"[beatport] {i}/{len(catalog)} pedidos={client.n_requests} "
                      f"{el / 60:.1f} min (ETA {el / i * (len(catalog) - i) / 60:.0f} min) niveles={lv}", flush=True)
            if i % args.checkpoint_every == 0:
                save(pd.DataFrame(rows), out_dir, args.output)
    except BeatportBlocked as exc:
        print(f"[ERROR] {exc}. Guardo lo hecho ({len(rows)} temas); la caché permite retomar.")
        status = 2
    df = pd.DataFrame(rows)
    save(df, out_dir, args.output)
    summary = summarize_run(df) if len(df) else {}
    summary.update({"dataset": args.dataset_name, "requests": client.n_requests,
                    "minutes": round((time.monotonic() - t0) / 60, 1), "complete": status == 0})
    (out_dir / f"{args.output}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                                         encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
