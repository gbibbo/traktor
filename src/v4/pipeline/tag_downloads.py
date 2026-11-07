"""
PURPOSE: Deja los temas descargados (download_JIJIJI) listos para la colección y las playlists, con las mismas
         reglas que la biblioteca y reutilizando lo que ya existe, sin reescribirlo:
           - Beatport: beatport_lookup.build_queries / lookup / record (misma caché,
             artifacts/v4/beatport_cache) y write_beatport_tags.build_plan / apply: con match A, o B
             confirmado por sello, fecha o duración, Artist, Remixers, Label, Genre y Released de Beatport
             (DECISIONS 2026-09-29), con respaldo y verificación de que el audio no cambia. Sin modelo de
             género: lo que Beatport no confirma queda sin género.
           - Vocal: tag_vocals.clap_probe_vocals / write_tags (CLAP del tema entero y el clasificador
             lineal; Vocal si al menos el 5 % del tema tiene voz, DECISIONS 2026-10-05).
           - Embedding: CLAP `clap_full` como extract_representations (3 segmentos de 30 s del centro,
             ventanas de 10 s, promedio), guardado dentro del archivo con file_embeddings.write_track (GEOB
             «TRAKTOR_ML clap_full v1», como en el pendrive) y en la caché del dataset `musica`
             (representations/clap_full/cache/<track_uid>.npz): «Agregar música» lo reutiliza.
         Ni BPM ni tonalidad (Gabriel, 2026-11-07). Recibe archivos sueltos, no un dataset: arma las filas con
         la forma del catálogo (read_tags, read_release_tags, audio_payload_hash). Idempotente: solo escribe lo
         que cambia, y las cachés de Beatport y de CLAP evitan buscar o analizar de nuevo.
         Dos modos:
           - tag_downloads.py ARCHIVOS|CARPETAS: todos de una vez;
           - tag_downloads.py --watch COLA.jsonl --done HECHOS.jsonl: tema por tema, mientras la descarga sigue
             (el orquestador agrega cada tema apenas está bajado o grabado; {"end": true} lo cierra). Los
             modelos se cargan una sola vez, con prioridad baja y pocos hilos, para no quitarle CPU a la
             grabación.
         Si Beatport pide la verificación anti-robots de Cloudflare, un pedido de prueba lo ve enseguida: esos
         temas quedan en download_tags/beatport_pending.txt y se completan en la próxima corrida. Respaldos en
         --backup-dir (por defecto artifacts/v4/download_tags); --revert CSV los restaura. Necesita el Python
         del pipeline (pandas, torch, transformers).
CHANGELOG:
  - 2026-11-07: Creación (Gabriel: los temas descargados salían solo con la metadata de Spotify).
  - 2026-11-07: Tema por tema (--watch) mientras la descarga sigue, y el embedding clap_full en el archivo y en
                la caché del dataset (Gabriel: que la música quede lista para la colección y las playlists, y
                que todo dure lo menos posible).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.beatport import BeatportBlocked, BeatportClient  # noqa: E402
from src.v4.common.config_loader import load_config  # noqa: E402
from src.v4.common.path_resolver import resolve_artifacts_root, resolve_dataset_artifacts  # noqa: E402
from src.v4.common.tags import audio_payload_hash, read_release_tags, read_tags  # noqa: E402
from src.v4.pipeline import beatport_lookup, file_embeddings, tag_vocals, write_beatport_tags  # noqa: E402

AUDIO = (".mp3", ".flac")
PENDING_NAME = "beatport_pending.txt"   # temas que esperan a Beatport (bloqueado la última vez)
REP = "clap_full"
DATASET = "musica"
BEATPORT_RETRY_S = 600                  # bloqueado: se vuelve a probar cada 10 minutos


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


class Models:
    """CLAP cargado una sola vez: ventanas del tema entero para la voz (vocal_eval.ClapDetector) y clap_full
    para el embedding (extract_representations.build_backend). Se cargan recién cuando hacen falta."""

    def __init__(self, config: dict):
        self.config = config
        self.detector = None
        self.backend = None

    def vocal_windows(self, path: Path, uid: str, cache: Path) -> None:
        """Las ventanas que clap_probe_vocals lee de su caché: con el modelo ya cargado, no lo carga de nuevo."""
        target = cache / f"{uid}.npz"
        if target.exists():
            return
        if self.detector is None:
            from src.v4.evaluation.vocal_eval import ClapDetector
            self.detector = ClapDetector()
        starts, _, emb = self.detector._embed(path)
        cache.mkdir(parents=True, exist_ok=True)
        np.savez(target, starts=starts, emb=emb)

    def clap_full(self, path: Path, uid: str, cache: Path) -> np.ndarray:
        """El mismo embedding y el mismo formato de caché que extract_representations --models clap."""
        target = cache / f"{uid}.npz"
        if target.exists():
            return np.load(target)["_"]
        from src.v4.common.audio_utils import read_dj_segments
        from src.v4.pipeline.extract_representations import (ESSENTIA_MODELS_DIR, build_backend, resample,
                                                              segmentation_params)
        if self.backend is None:
            self.backend = build_backend("clap", ESSENTIA_MODELS_DIR, os.environ.get("TRAKTOR_HF_CACHE"))
        seg = segmentation_params(self.config)
        segments, sr = read_dj_segments(path, seg["segment_duration_s"], seg["n_intro"], seg["n_mid"],
                                        seg["n_outro"])
        acc: Dict[str, List[np.ndarray]] = {}
        for segment in segments:
            for suffix, vec in self.backend.embed(resample(segment, sr, self.backend.sr)).items():
                acc.setdefault(suffix, []).append(vec)
        payload = {(k or "_"): (np.concatenate(v, axis=0) if k.startswith("_win") else np.mean(v, axis=0))
                   for k, v in acc.items()}
        cache.mkdir(parents=True, exist_ok=True)
        np.savez(target, **payload)
        return payload["_"]


def embed_in_file(path: Path, uid: str, vec: np.ndarray) -> bool:
    """El embedding dentro del archivo (file_embeddings): True si se escribió, False si ya estaba igual."""
    current, place = file_embeddings.read_track(path, REP)
    want = {**(place or {}), "track_uid": uid}
    if file_embeddings.same(current, place, vec, want):
        return False
    file_embeddings.write_track(path, vec, want, REP)
    return True


def caches(config: dict, tools_root: Path) -> Tuple[Path, Path]:
    """Cachés de CLAP: las del dataset `musica` si está en este ordenador (así «Agregar música» las reutiliza),
    si no, en artifacts/v4/download_tags."""
    try:
        rep_root = resolve_dataset_artifacts(DATASET, config) / "representations"
        if rep_root.is_dir():
            return rep_root / "clap_wholetrack" / "cache", rep_root / REP / "cache"
    except Exception:  # noqa: BLE001
        pass
    base = tools_root / "download_tags"
    return base / "clap_wholetrack_cache", base / "clap_full_cache"


class Beatport:
    """Beatport con su pedido de prueba: si está bloqueado, se vuelve a probar cada BEATPORT_RETRY_S."""

    def __init__(self, cache_dir: Path, offline: bool):
        self.client = BeatportClient(cache_dir, offline=offline)
        self.offline = offline
        self.ok: Optional[bool] = True if offline else None
        self.why = ""
        self.next_probe = 0.0

    def ready(self) -> bool:
        if self.ok is None or (not self.ok and time.monotonic() >= self.next_probe):
            self.ok, self.why = beatport_available()
            self.next_probe = time.monotonic() + BEATPORT_RETRY_S
        return bool(self.ok)

    def blocked(self, why: str) -> None:
        self.ok, self.why, self.next_probe = False, why, time.monotonic() + BEATPORT_RETRY_S


def process(files: List[Path], *, config: dict, tools_root: Path, backup_dir: Path, beatport: Optional[Beatport],
            models: Optional[Models], stamp: str) -> List[Dict[str, Any]]:
    """Beatport, Vocal y embedding de estos archivos. Un resultado por archivo (para el registro y HECHOS)."""
    catalog, rels = catalog_rows(files)
    results = {row.track_uid: {"path": row.source_path, "name": Path(row.source_path).name}
               for row in catalog.itertuples()}

    if beatport is not None:
        pending_path = backup_dir / PENDING_NAME
        pending = [p for p in read_pending(pending_path) if p not in files]
        if not beatport.ready():
            write_pending(pending_path, pending + files)
            for r in results.values():
                r["genre"] = "espera a Beatport"
        else:
            bp_catalog, bp_rels = catalog, rels
            if pending:
                more, more_rels = catalog_rows(pending)
                bp_catalog, bp_rels = pd.concat([catalog, more], ignore_index=True), {**rels, **more_rels}
            try:
                plan = beatport_tags(bp_catalog, bp_rels, beatport.client,
                                     backup_dir / f"beatport_tags_{stamp}_{catalog.track_uid.iloc[0][:10]}.csv")
                write_pending(pending_path, [])
            except BeatportBlocked as exc:
                beatport.blocked(str(exc))
                write_pending(pending_path, pending + files)
                plan = pd.DataFrame()
            for r in plan.itertuples() if len(plan) else []:
                if r.track_uid in results:
                    results[r.track_uid]["genre"] = r.new_genre if r.action == "beatport" else "sin Beatport"
                    results[r.track_uid]["level"] = r.level
                elif r.action == "beatport":
                    print(f"[metadata]   {Path(r.source_path).name} (esperaba a Beatport): {r.new_genre}", flush=True)

    if models is not None:
        vocal_cache, full_cache = caches(config, tools_root)
        for row in catalog.itertuples():
            try:
                models.vocal_windows(Path(row.source_path), row.track_uid, vocal_cache)
            except Exception as exc:  # noqa: BLE001 (clap_probe_vocals lo vuelve a intentar y avisa)
                results[row.track_uid]["error"] = f"voz: {type(exc).__name__}: {exc}"
        vocals = vocal_tags(catalog, vocal_cache, backup_dir / f"vocal_tags_{stamp}_{catalog.track_uid.iloc[0][:10]}.csv")
        for v in vocals.itertuples():
            results[v.track_uid]["vocal"] = bool(v.is_vocal)
        for row in catalog.itertuples():
            try:
                vec = models.clap_full(Path(row.source_path), row.track_uid, full_cache)
                results[row.track_uid]["embedding"] = "escrito" if embed_in_file(
                    Path(row.source_path), row.track_uid, vec) else "ya estaba"
            except Exception as exc:  # noqa: BLE001
                results[row.track_uid]["embedding"] = f"falló: {type(exc).__name__}: {exc}"
    return list(results.values())


def describe(result: Dict[str, Any]) -> str:
    parts = [str(result.get("genre") or "—")]
    if "vocal" in result:
        parts.append("Vocal" if result["vocal"] else "sin voz")
    if "embedding" in result:
        parts.append(f"embedding {result['embedding']}")
    if result.get("error"):
        parts.append(result["error"])
    return f"[metadata] {result['name']}: " + " · ".join(parts)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Las líneas completas de un JSONL que otro proceso va escribiendo (la última puede estar a medias)."""
    try:
        raw = path.read_bytes()
    except OSError:
        return []
    out = []
    for line in raw.split(b"\n")[:-1]:
        try:
            item = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


def watch(queue_path: Path, done_path: Path, **kwargs: Any) -> int:
    """Tema por tema, apenas aparece en la cola; termina con {"end": true} cuando no queda nada."""
    done = {str(item.get("path")) for item in read_jsonl(done_path)}
    while True:
        items = read_jsonl(queue_path)
        todo = [str(i["path"]) for i in items if i.get("path") and str(i["path"]) not in done]
        for path in todo:
            if not Path(path).is_file():
                result = {"path": path, "name": Path(path).name, "error": "ya no está"}
            else:
                try:
                    result = process([Path(path)], **kwargs)[0]
                except Exception as exc:  # noqa: BLE001 (un tema que falla no frena a los demás)
                    result = {"path": path, "name": Path(path).name, "error": f"{type(exc).__name__}: {exc}"}
            done.add(path)
            with done_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(describe(result), flush=True)
        if not todo:
            if any(i.get("end") for i in items):
                return 0
            time.sleep(1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Metadata y embedding de temas descargados (Beatport, Vocal, CLAP)")
    parser.add_argument("files", nargs="*", type=Path, help="Archivos o carpetas (MP3/FLAC)")
    parser.add_argument("--watch", type=Path, default=None, help="Cola JSONL de temas terminados (modo tema por tema)")
    parser.add_argument("--done", type=Path, default=None, help="Con --watch: JSONL con el resultado de cada tema")
    parser.add_argument("--backup-dir", type=Path, default=None,
                        help="Respaldos de los tags anteriores (default: artifacts/v4/download_tags)")
    parser.add_argument("--no-beatport", action="store_true")
    parser.add_argument("--no-vocal", action="store_true", help="Sin voz ni embedding (sin CLAP)")
    parser.add_argument("--offline", action="store_true", help="Beatport solo desde la caché")
    parser.add_argument("--threads", type=int, default=None, help="Hilos de torch (default: un tercio de los núcleos)")
    parser.add_argument("--revert", type=Path, default=None,
                        help="Restaura un respaldo (beatport_tags_*.csv o vocal_tags_*.csv)")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "v4.yaml")
    args = parser.parse_args()

    if args.revert:
        reverter = tag_vocals.revert if args.revert.name.startswith("vocal_") else write_beatport_tags.revert
        print(f"[metadata] {reverter(args.revert)} archivos restaurados desde {args.revert}")
        return 0

    config = load_config(args.config)
    tools_root = resolve_artifacts_root(config).parent  # artifacts/v4
    backup_dir = args.backup_dir or tools_root / "download_tags"
    backup_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(config=config, tools_root=tools_root, backup_dir=backup_dir,
                  beatport=None if args.no_beatport else Beatport(tools_root / "beatport_cache", args.offline),
                  models=None if args.no_vocal else Models(config), stamp=datetime.now().strftime("%Y%m%d_%H%M%S"))

    if args.watch:
        if args.done is None:
            parser.error("--watch necesita --done")
        from src.v4.pipeline.extract_representations import lower_priority
        lower_priority()  # la grabación (captura en tiempo real) va primero
        try:
            import torch
            torch.set_num_threads(args.threads or max(2, (os.cpu_count() or 6) // 3))
        except ImportError:
            pass
        return watch(args.watch, args.done, **kwargs)

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
    results = process(files, **kwargs)
    for result in results:
        print(describe(result), flush=True)
    beatport = kwargs["beatport"]
    if beatport is not None and not beatport.ok:
        print(f"[metadata] Beatport {beatport.why}: género y sello quedan para la próxima descarga.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
