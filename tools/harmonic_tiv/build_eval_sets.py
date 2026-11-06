"""
PURPOSE: Arma los conjuntos de evaluación y ajuste de la compatibilidad armónica (protocolo en
         docs/plans/armonia_tiv_protocolo.md). No escucha nada ni toca el audio: solo junta etiquetas.
           1. thesis48: los 48 temas de la tesis de Gabriel (24 Progressive House a 124 BPM, 24 Techno a
              128, dos por cada tonalidad menor; tonalidad de Beatport verificada a mano, switch points
              manuales). Manifiesto con id = hash del audio (igual que track_uid de los catálogos).
           2. keys_musica: tonalidad de cada tema de `musica` según fuentes distintas: tag del archivo
              (Mixed In Key, en Camelot), Beatport (solo matches confirmados) y el análisis de Traktor
              (MUSICAL_KEY de collection.nml). Consenso = al menos dos fuentes iguales y ninguna distinta.
           3. transitions_history: transiciones reales de Gabriel en los History de Traktor: B arranca
              mientras A sigue corriendo y B corre >= 60 s. Los temas se atan al catálogo por la ruta
              relativa a la carpeta Música y, si no, por el hash del audio del archivo (contenido).
           4. splits: dev/test por artista (hash estable del artista principal) para los temas, y por
              fecha para las sesiones (las más viejas = dev).
         Uso: python tools/harmonic_tiv/build_eval_sets.py --out <dir> [--thesis-src <clon de harmonic_mix>]
CHANGELOG:
  - 2026-11-06: Creación (rama harmonic-tiv).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.v4.common.harmonic import to_camelot  # noqa: E402
from src.v4.common.tags import audio_payload_hash  # noqa: E402
from src.v4.common.traktor_nml import entry_path, find_collection  # noqa: E402

# Traktor MUSICAL_KEY VALUE: 0..11 = C..B mayor, 12..23 = C..B menor
_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def traktor_value_to_camelot(v) -> str:
    try:
        i = int(v)
    except (TypeError, ValueError):
        return "?"
    if not 0 <= i <= 23:
        return "?"
    return to_camelot(f"{_NOTES[i % 12]} {'minor' if i >= 12 else 'major'}")


def norm_rel(p: str) -> str:
    """Ruta relativa a la carpeta Música, en minúsculas, con / y sin acentos raros de normalización."""
    s = unicodedata.normalize("NFC", str(p)).replace("\\", "/").lower()
    for marker in ("/música/", "/musica/"):
        if marker in s:
            s = s.split(marker, 1)[1]
            break
    return s.lstrip("/")


def primary_artist(a: str) -> str:
    a = (a or "").lower()
    a = re.split(r",|&| feat\.? | ft\.? | x | vs\.? | and ", a)[0]
    return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKD", a))


def split_of(key: str, salt: str = "harmonic_tiv_v1") -> str:
    h = hashlib.sha256((salt + key).encode("utf-8")).digest()[0]
    return "dev" if h % 2 == 0 else "test"


# --------------------------------------------------------------------------- 1. thesis48

def build_thesis48(src: Path, dst: Path) -> pd.DataFrame:
    rows = []
    for genre in ("progressive_house", "techno"):
        for ann in sorted((src / "annotations" / genre).glob("*.json")):
            audio = src / "music" / genre / (ann.stem + ".mp3")
            if not audio.exists():
                raise FileNotFoundError(audio)
            local = dst / "audio" / genre / audio.name
            if not local.exists():
                local.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(audio, local)
            a = json.loads(ann.read_text(encoding="utf-8"))
            rows.append({"id": audio_payload_hash(local), "path": str(local), "genre": genre, "name": ann.stem,
                         "camelot": f"{int(a['key'])}{a['mode']}", "bpm": a["bpm"],
                         "switch_in": a.get("time_intro"), "switch_out": a.get("time_outro"),
                         "switch_in2": a.get("time_intro2"), "switch_out2": a.get("time_outro2")})
    # Los 20 clips del experimento de escucha (P1 = objetivo, P2 = candidato; sin respuestas en el repo)
    exp = src / "music" / "experiment"
    if exp.exists():
        for f in sorted(exp.glob("*.mp3")):
            local = dst / "audio" / "experiment" / f.name
            if not local.exists():
                local.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, local)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 2. tonalidades de musica

def traktor_index(collection: Path) -> pd.DataFrame:
    root = ET.parse(collection).getroot()
    rows = []
    for e in root.find("COLLECTION").findall("ENTRY"):
        mk = e.find("MUSICAL_KEY")
        info = e.find("INFO")
        rows.append({"path": str(entry_path(e)),
                     "traktor_value": mk.get("VALUE") if mk is not None else None,
                     "traktor_info_key": info.get("KEY") if info is not None else None})
    return pd.DataFrame(rows)


def match_paths_to_uids(paths, cat: pd.DataFrame, hash_cache: dict) -> list:
    """track_uid de cada ruta: por ruta relativa a Música (orig_rel_path o rel_path) y si no, por el hash
    del audio del archivo (si existe). None si no se pudo."""
    by_rel = {}
    for col in ("orig_rel_path", "rel_path"):
        for uid, rel in zip(cat["track_uid"], cat[col].fillna("")):
            by_rel.setdefault(norm_rel(rel), uid)
    by_src = {str(p).lower(): uid for uid, p in zip(cat["track_uid"], cat["source_path"])}
    uids = set(cat["track_uid"])
    out = []
    for p in paths:
        uid = by_src.get(str(p).lower()) or by_rel.get(norm_rel(p))
        if uid is None and Path(p).exists():
            if p not in hash_cache:
                try:
                    hash_cache[p] = audio_payload_hash(Path(p))
                except Exception:  # noqa: BLE001
                    hash_cache[p] = None
            uid = hash_cache[p] if hash_cache[p] in uids else None
        out.append(uid)
    return out


def build_keys(cat: pd.DataFrame, beatport: pd.DataFrame, trk: pd.DataFrame, hash_cache: dict) -> pd.DataFrame:
    df = cat[["track_uid", "artist", "title", "source_path", "duration_s"]].copy()
    df["k_tag"] = [to_camelot(a if isinstance(a, str) and a else b)
                   for a, b in zip(cat["tag_camelot"], cat["tag_key"])]
    bp = beatport[beatport["confirmed"].fillna(False)][["track_uid", "bp_key"]]
    df = df.merge(bp, on="track_uid", how="left")
    df["k_beatport"] = df["bp_key"].map(to_camelot)
    trk = trk.copy()
    trk["track_uid"] = match_paths_to_uids(trk["path"], cat, hash_cache)
    trk = trk.dropna(subset=["track_uid"]).drop_duplicates("track_uid")
    trk["k_traktor"] = trk["traktor_value"].map(traktor_value_to_camelot)
    df = df.merge(trk[["track_uid", "k_traktor"]], on="track_uid", how="left")
    for c in ("k_tag", "k_beatport", "k_traktor"):
        df[c] = df[c].fillna("?")

    def consensus(row):
        ks = [row[c] for c in ("k_tag", "k_beatport", "k_traktor") if row[c] != "?"]
        if len(ks) >= 2 and len(set(ks)) == 1:
            return ks[0]
        return "?"
    df["k_consensus"] = df.apply(consensus, axis=1)
    df["n_sources"] = (df[["k_tag", "k_beatport", "k_traktor"]] != "?").sum(axis=1)
    df["split"] = [split_of(primary_artist(a) or uid) for a, uid in zip(df["artist"].fillna(""), df["track_uid"])]
    return df.drop(columns=["bp_key"])


# --------------------------------------------------------------------------- 3. transiciones reales

def build_transitions(history_dir: Path, cat: pd.DataFrame, hash_cache: dict, min_play_s: float = 60.0):
    rows, stats = [], {"sessions": 0, "entries": 0, "played": 0, "matched": 0}
    for f in sorted(history_dir.glob("*.nml")):
        root = ET.parse(f).getroot()
        coll = {e.find("LOCATION").get("VOLUME", "") + e.find("LOCATION").get("DIR", "") + e.find("LOCATION").get("FILE", ""):
                str(entry_path(e)) for e in root.find("COLLECTION").findall("ENTRY") if e.find("LOCATION") is not None}
        pl = root.find(".//PLAYLIST")
        if pl is None:
            continue
        stats["sessions"] += 1
        seq = []
        for e in pl.findall("ENTRY"):
            pk, ext = e.find("PRIMARYKEY"), e.find("EXTENDEDDATA")
            if pk is None or ext is None:
                continue
            key = pk.get("KEY", "")
            path = coll.get(key)
            if path is None:  # PRIMARYKEY KEY = VOLUME + DIR + FILE
                continue
            start = float(ext.get("STARTDATE", 0)) * 86400 + float(ext.get("STARTTIME", 0))
            seq.append({"path": path, "start": start, "dur": float(ext.get("DURATION", 0)),
                        "deck": ext.get("DECK")})
        stats["entries"] += len(seq)
        uids = match_paths_to_uids([s["path"] for s in seq], cat, hash_cache)
        for s, u in zip(seq, uids):
            s["uid"] = u
        played = sorted([s for s in seq if s["dur"] >= min_play_s], key=lambda s: s["start"])
        stats["played"] += len(played)
        stats["matched"] += sum(s["uid"] is not None for s in played)
        for i, b in enumerate(played):
            # A = el tema anterior más reciente que seguía corriendo cuando arrancó B (mezclados)
            prev = [a for a in played[:i] if a["start"] < b["start"] < a["start"] + a["dur"] and a["deck"] != b["deck"]]
            if not prev:
                continue
            a = max(prev, key=lambda a: a["start"])
            rows.append({"session": f.stem, "a_uid": a["uid"], "b_uid": b["uid"], "a_path": a["path"],
                         "b_path": b["path"], "overlap_s": round(a["start"] + a["dur"] - b["start"], 1),
                         "a_dur": round(a["dur"], 1), "b_dur": round(b["dur"], 1),
                         "deck_a": a["deck"], "deck_b": b["deck"], "b_start": b["start"]})
    return pd.DataFrame(rows), stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path, required=True, help="carpeta del dataset musica (catalog.parquet)")
    ap.add_argument("--thesis-src", type=Path, help="clon de github.com/gbibbo/harmonic_mix")
    ap.add_argument("--thesis-dst", type=Path, help="dónde copiar el audio de thesis48 (fuera de Git)")
    ap.add_argument("--history", type=Path, help="carpeta History de Traktor (por defecto, junto a collection.nml)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}

    if args.thesis_src:
        t48 = build_thesis48(args.thesis_src, args.thesis_dst or args.out / "thesis48")
        t48.to_csv(args.out / "thesis48_manifest.csv", index=False, encoding="utf-8")
        report["thesis48"] = {"n": len(t48), "by_genre": t48["genre"].value_counts().to_dict()}

    cat = pd.read_parquet(args.dataset_root / "catalog.parquet")
    beatport = pd.read_parquet(args.dataset_root / "features" / "beatport.parquet")
    collection = find_collection()
    hash_cache_path = args.out / "hash_cache.json"
    hash_cache = json.loads(hash_cache_path.read_text(encoding="utf-8")) if hash_cache_path.exists() else {}
    trk = traktor_index(collection)
    keys = build_keys(cat, beatport, trk, hash_cache)
    keys.to_csv(args.out / "keys_musica.csv", index=False, encoding="utf-8")
    manifest = keys[["track_uid", "source_path"]].rename(columns={"track_uid": "id", "source_path": "path"})
    manifest.to_csv(args.out / "musica_manifest.csv", index=False, encoding="utf-8")

    def agree(a, b):
        m = (keys[a] != "?") & (keys[b] != "?")
        return {"n_both": int(m.sum()), "exact": round(float((keys.loc[m, a] == keys.loc[m, b]).mean()), 3) if m.any() else None}
    report["keys"] = {"n_tracks": len(keys), "collection": str(collection),
                      "coverage": {c: int((keys[c] != "?").sum()) for c in ("k_tag", "k_beatport", "k_traktor", "k_consensus")},
                      "agreement": {"tag_vs_beatport": agree("k_tag", "k_beatport"),
                                    "tag_vs_traktor": agree("k_tag", "k_traktor"),
                                    "beatport_vs_traktor": agree("k_beatport", "k_traktor")},
                      "consensus_by_split": keys[keys.k_consensus != "?"]["split"].value_counts().to_dict(),
                      "split_sizes": keys["split"].value_counts().to_dict()}

    hist = args.history or (collection.parent / "History")
    trans, stats = build_transitions(hist, cat, hash_cache)
    hash_cache_path.write_text(json.dumps(hash_cache), encoding="utf-8")
    if len(trans):
        sessions = sorted(trans["session"].unique())
        half = sessions[: len(sessions) // 2]
        trans["split"] = ["dev" if s in half else "test" for s in trans["session"]]
    trans.to_csv(args.out / "transitions_history.csv", index=False, encoding="utf-8")
    both = trans.dropna(subset=["a_uid", "b_uid"]) if len(trans) else trans
    report["history"] = {**stats, "transitions": len(trans), "transitions_both_in_catalog": len(both),
                         "by_split": both["split"].value_counts().to_dict() if len(both) else {}}
    (args.out / "eval_sets_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
