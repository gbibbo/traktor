"""
PURPOSE: Completa la metadata de los temas descargados (download_JIJIJI) con las mismas reglas que la
         biblioteca, reutilizando lo que ya existe, sin reescribirlo:
           - Beatport: beatport_lookup.build_queries / lookup / record (misma caché,
             artifacts/v4/beatport_cache) y write_beatport_tags.build_plan / apply: con match A, o B
             confirmado por sello, fecha o duración, Artist, Remixers, Label, Genre y Released de Beatport
             (DECISIONS 2026-09-29), con respaldo y verificación de que el audio no cambia. Sin modelo de
             género: lo que Beatport no confirma queda sin género.
           - Vocal: tag_vocals.clap_probe_vocals / write_tags (CLAP del tema entero y el clasificador
             lineal; Vocal si al menos el 5 % del tema tiene voz, DECISIONS 2026-10-05).
         Ni BPM ni tonalidad (Gabriel, 2026-11-07). Recibe archivos sueltos (los de una descarga), no un
         dataset: arma para ellos las filas con la forma del catálogo (read_tags, read_release_tags,
         audio_payload_hash). Es idempotente: solo escribe lo que cambia, y las cachés de Beatport y de CLAP
         evitan volver a buscar o analizar. Si Beatport pide la verificación anti-robots de Cloudflare, un
         pedido de prueba lo ve enseguida: esos temas quedan en download_tags/beatport_pending.txt y se
         completan en la próxima corrida. Respaldos en --backup-dir (por defecto artifacts/v4/download_tags);
         --revert CSV los restaura. Necesita el Python del pipeline (pandas, torch, transformers): el
         orquestador de descargas lo lanza al terminar cada descarga.
CHANGELOG:
  - 2026-11-07: Creación (Gabriel: los temas descargados salían solo con la metadata de Spotify).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.beatport import BeatportBlocked, BeatportClient  # noqa: E402
from src.v4.common.config_loader import load_config  # noqa: E402
from src.v4.common.path_resolver import resolve_artifacts_root  # noqa: E402
from src.v4.common.tags import audio_payload_hash, read_release_tags, read_tags  # noqa: E402
from src.v4.pipeline import beatport_lookup, tag_vocals, write_beatport_tags  # noqa: E402

AUDIO = (".mp3", ".flac")
PENDING_NAME = "beatport_pending.txt"   # temas que esperan a Beatport (bloqueado la última vez)


def beatport_available() -> Tuple[bool, str]:
    """Un pedido de prueba: si Cloudflare pide la verificación anti-robots («Just a moment…», visto el
    2026-11-07), BeatportClient pasaría ~5 minutos reintentando. Entonces se deja para la próxima vez."""
    import requests
    from src.v4.common.beatport import SEARCH_URL, USER_AGENT

    try:
        r = requests.get(SEARCH_URL, params={"q": "techno"}, timeout=30,
                         headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})
    except requests.RequestException as exc:
        return False, type(exc).__name__
    if r.status_code == 200:
        return True, ""
    return False, ("pide la verificación anti-robots de Cloudflare" if r.headers.get("cf-mitigated") == "challenge"
                   else f"respondió {r.status_code}")


def read_pending(path: Path) -> List[Path]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [Path(line) for line in lines if line.strip() and Path(line).is_file()]


def write_pending(path: Path, files: List[Path]) -> None:
    files = list(dict.fromkeys(files))
    if files:
        path.write_text("\n".join(map(str, files)) + "\n", encoding="utf-8")
    else:
        path.unlink(missing_ok=True)


def catalog_rows(files: List[Path]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Optional[str]]]]:
    """Filas con la forma del catálogo (las columnas que leen beatport_lookup y tag_vocals)."""
    import mutagen

    rows, rels = [], {}
    for path in files:
        try:
            length = float(mutagen.File(str(path)).info.length)
        except Exception:  # noqa: BLE001
            length = None
        uid = audio_payload_hash(path)
        rows.append({"track_uid": uid, "source_path": str(path), "rel_path": path.name, "filename": path.name,
                     "duration_s": length, **read_tags(path)})
        rels[uid] = read_release_tags(path)
    return pd.DataFrame(rows), rels


def beatport_tags(catalog: pd.DataFrame, rels: Dict[str, Dict[str, Optional[str]]], client: BeatportClient,
                  backup: Path) -> pd.DataFrame:
    records = []
    for _, row in catalog.iterrows():
        variants = beatport_lookup.build_queries(row, rels[row["track_uid"]])
        if not variants:
            continue
        res, q, used = beatport_lookup.lookup(client, variants)
        records.append(beatport_lookup.record(row, rels[row["track_uid"]], q, res, used))
    if not records:
        return pd.DataFrame()
    no_model = pd.DataFrame(columns=["track_uid", "model_source", "model_conf", "model_genre"])
    plan = write_beatport_tags.build_plan(pd.DataFrame(records), no_model, catalog)
    log = write_beatport_tags.apply(plan, backup)
    return plan.merge(log[["track_uid", "status"]], on="track_uid", how="left") if len(log) else plan


def vocal_tags(catalog: pd.DataFrame, cache_dir: Path, backup: Path) -> pd.DataFrame:
    tracks = catalog[["track_uid", "source_path"]]
    vocals = tag_vocals.clap_probe_vocals(tracks, cache_dir)
    if vocals.empty:
        return vocals
    tag_vocals.write_tags(tracks, vocals, backup)
    return vocals


def main() -> int:
    parser = argparse.ArgumentParser(description="Completa la metadata de temas descargados (Beatport y Vocal)")
    parser.add_argument("files", nargs="*", type=Path, help="Archivos o carpetas (MP3/FLAC)")
    parser.add_argument("--backup-dir", type=Path, default=None,
                        help="Respaldos de los tags anteriores (default: artifacts/v4/download_tags)")
    parser.add_argument("--no-beatport", action="store_true")
    parser.add_argument("--no-vocal", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Beatport solo desde la caché")
    parser.add_argument("--revert", type=Path, default=None,
                        help="Restaura un respaldo (beatport_tags_*.csv o vocal_tags_*.csv)")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "v4.yaml")
    args = parser.parse_args()

    if args.revert:
        reverter = tag_vocals.revert if args.revert.name.startswith("vocal_") else write_beatport_tags.revert
        print(f"[metadata] {reverter(args.revert)} archivos restaurados desde {args.revert}")
        return 0

    files: List[Path] = []
    for item in args.files:
        if item.is_dir():
            files += sorted(p for p in item.iterdir() if p.suffix.lower() in AUDIO)
        elif item.suffix.lower() in AUDIO and item.is_file():
            files.append(item)
    files = list(dict.fromkeys(p.resolve() for p in files))
    if not files:
        print("[metadata] No hay MP3 ni FLAC para completar.")
        return 0

    tools_root = resolve_artifacts_root(load_config(args.config)).parent  # artifacts/v4
    backup_dir = args.backup_dir or tools_root / "download_tags"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    catalog, rels = catalog_rows(files)
    code = 0

    if not args.no_beatport:
        pending_path = backup_dir / PENDING_NAME
        pending = [p for p in read_pending(pending_path) if p not in files]
        ok, why = (True, "") if args.offline else beatport_available()
        plan = pd.DataFrame()
        if not ok:
            write_pending(pending_path, pending + files)
            print(f"[metadata] Beatport {why}: género y sello quedan para la próxima descarga "
                  f"({len(pending) + len(files)} temas en espera).", flush=True)
            code = 1
        else:
            if pending:
                print(f"[metadata] También {len(pending)} temas que esperaban a Beatport.", flush=True)
                more, more_rels = catalog_rows(pending)
                bp_catalog, bp_rels = pd.concat([catalog, more], ignore_index=True), {**rels, **more_rels}
            else:
                bp_catalog, bp_rels = catalog, rels
            client = BeatportClient(tools_root / "beatport_cache", offline=args.offline)
            try:
                plan = beatport_tags(bp_catalog, bp_rels, client, backup_dir / f"beatport_tags_{stamp}.csv")
                write_pending(pending_path, [])
            except BeatportBlocked as exc:
                write_pending(pending_path, pending + files)
                print(f"[metadata] Beatport dejó de responder: {exc}. Lo que faltó queda para la próxima descarga.",
                      flush=True)
                code = 1
        if len(plan):
            found = plan[plan["action"] == "beatport"]
            written = int((plan.get("status") == "written").sum()) if "status" in plan else 0
            print(f"[metadata] Beatport: {len(found)} de {len(plan)} temas confirmados "
                  f"({', '.join(f'{k}: {v}' for k, v in found['level'].value_counts().sort_index().items()) or '-'}); "
                  f"{written} archivos actualizados. Sin Beatport (quedan sin género): {len(plan) - len(found)}.",
                  flush=True)
            for r in plan.itertuples():
                genre = r.new_genre if r.action == "beatport" else "—"
                print(f"[metadata]   {Path(r.source_path).name}: {r.level} · {genre}", flush=True)

    if not args.no_vocal:
        vocals = vocal_tags(catalog, tools_root / "download_tags" / "clap_wholetrack_cache",
                            backup_dir / f"vocal_tags_{stamp}.csv")
        if len(vocals):
            print(f"[metadata] Vocal: {int(vocals['is_vocal'].sum())} de {len(vocals)} temas "
                  f"(al menos el {tag_vocals.DEFAULT_MIN_FRACTION:.0%} del tema con voz).", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
