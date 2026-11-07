"""
PURPOSE: D1 del plan de playlists en vivo: los sets de Gabriel leídos de la historia de Traktor
         (Documents/Native Instruments/Traktor 4.4.1/History/*.nml), como filas del almacén de sets
         (docs/plans/recomendacion_playlists_plan.md, sección 3.2). Cada fila es un tema al aire:
           set_id, fuente ("D1"), archivo de historia, posicion, inicio (fecha y hora), minuto (desde el
           primer tema al aire del set), segundos_al_aire, deck, artista, titulo, clave de Traktor y
           track_uid del índice de recomendación (o vacío).
         Reglas (auditoría 2026-11-07):
           - al aire = PLAYEDPUBLIC 1 y DURATION >= 120 s;
           - la hora es STARTDATE (año << 16 | mes << 8 | día) + STARTTIME (segundos del día);
           - un set nuevo empieza después de 20 min sin nada al aire (corta las sesiones dejadas abiertas);
           - se guardan los sets con 8 o más temas al aire.
         El tema se reconoce en el índice por su ruta (la del paquete o la original de Música/, como
         recomendador.identify) y, si no, por artista y título normalizados.
         Escribe <root>/sets/d1.csv y un resumen.
CHANGELOG:
  - 2026-11-07: Creación (etapa 1 del plan: almacén de sets con D1).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from src.v4.common.traktor_nml import entry_key  # noqa: E402
from tools.traktor_live import recomendador as rc  # noqa: E402

DEFAULT_ROOT = REPO_ROOT / "artifacts" / "v4" / "dj_sets"
DEFAULT_HISTORY = Path.home() / "Documents" / "Native Instruments" / "Traktor 4.4.1" / "History"
MIN_ON_AIR_S = 120.0
GAP_S = 20 * 60
MIN_TRACKS = 8
FIELDS = ["set_id", "fuente", "historia", "posicion", "inicio", "minuto", "segundos_al_aire", "deck",
          "artista", "titulo", "clave_traktor", "track_uid", "reconocido_por"]


def start_of(date_code: str, seconds: str) -> Optional[dt.datetime]:
    try:
        d, s = int(date_code), float(seconds)
        return dt.datetime(d >> 16, (d >> 8) & 0xFF, d & 0xFF) + dt.timedelta(seconds=s)
    except (TypeError, ValueError):
        return None


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", s)
    s = re.sub(r"\b(feat|ft|featuring)\b.*", " ", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def on_air(path: Path) -> List[Dict]:
    """Temas al aire de un archivo de historia, en orden de inicio."""
    root = ET.parse(path).getroot()
    coll = {entry_key(e): e for e in root.iter("ENTRY") if e.find("LOCATION") is not None}
    rows = []
    for pe in root.iter("ENTRY"):
        pk, ext = pe.find("PRIMARYKEY"), pe.find("EXTENDEDDATA")
        if pk is None or ext is None or ext.get("PLAYEDPUBLIC") != "1":
            continue
        try:
            secs = float(ext.get("DURATION") or 0)
        except ValueError:
            continue
        start = start_of(ext.get("STARTDATE"), ext.get("STARTTIME"))
        if secs < MIN_ON_AIR_S or start is None:
            continue
        key = pk.get("KEY") or ""
        e = coll.get(key)
        rows.append({"historia": path.name, "inicio": start, "segundos_al_aire": secs, "deck": ext.get("DECK"),
                     "artista": e.get("ARTIST", "") if e is not None else "",
                     "titulo": e.get("TITLE", "") if e is not None else "", "clave_traktor": key})
    return sorted(rows, key=lambda r: r["inicio"])


def split_sets(rows: List[Dict], gap_s: float = GAP_S) -> List[List[Dict]]:
    """Corta donde pasan más de gap_s sin nada al aire (desde el fin del último tema que sonó)."""
    sets, cur, end = [], [], None
    for r in rows:
        r_end = r["inicio"] + dt.timedelta(seconds=r["segundos_al_aire"])
        if cur and (r["inicio"] - end).total_seconds() > gap_s:
            sets.append(cur)
            cur, end = [], None
        cur.append(r)
        end = r_end if end is None else max(end, r_end)
    if cur:
        sets.append(cur)
    return sets


class Identifier:
    """track_uid del índice de recomendación por ruta (sufijos, como recomendador.identify) o por
    artista y título."""

    def __init__(self, index_path: Path):
        _, items = rc.load_index(index_path)
        self.by_suffix: Dict[str, str] = {}
        self.by_name: Dict[Tuple[str, str], str] = {}
        for it in items:
            for k in (rc.norm(it.rel), rc.norm(it.original)):
                if k:
                    self.by_suffix.setdefault(k, it.uid)
            a, t = norm_text(it.artist.split(",")[0].split("&")[0]), norm_text(it.title)
            if a and t:
                self.by_name.setdefault((a, t), it.uid)

    def __call__(self, key: str, artist: str, title: str) -> Tuple[str, str]:
        path = "/".join(p for p in key.replace("/:", "/").split("/") if p)
        for s in rc.suffixes(path):
            uid = self.by_suffix.get(s)
            if uid:
                return uid, "ruta"
        uid = self.by_name.get((norm_text(artist.split(",")[0].split("&")[0]), norm_text(title)))
        return (uid, "artista_titulo") if uid else ("", "")


def build(history: Path, ident: Optional[Identifier]) -> Tuple[List[Dict], Dict]:
    rows_all, files = [], sorted(history.glob("*.nml"))
    for p in files:
        rows_all += on_air(p)
    rows_all.sort(key=lambda r: r["inicio"])
    out, n_sets, short = [], 0, 0
    for s in split_sets(rows_all):
        if len(s) < MIN_TRACKS:
            short += 1
            continue
        n_sets += 1
        t0 = s[0]["inicio"]
        sid = f"D1_{t0:%Y%m%d_%H%M}"
        for i, r in enumerate(s):
            uid, how = ident(r["clave_traktor"], r["artista"], r["titulo"]) if ident else ("", "")
            out.append({"set_id": sid, "fuente": "D1", "historia": r["historia"], "posicion": i,
                        "inicio": r["inicio"].isoformat(timespec="seconds"),
                        "minuto": round((r["inicio"] - t0).total_seconds() / 60, 2),
                        "segundos_al_aire": round(r["segundos_al_aire"], 1), "deck": r["deck"],
                        "artista": r["artista"], "titulo": r["titulo"], "clave_traktor": r["clave_traktor"],
                        "track_uid": uid, "reconocido_por": how})
    by_set: Dict[str, List[Dict]] = {}
    for r in out:
        by_set.setdefault(r["set_id"], []).append(r)
    trans = sum(len(v) - 1 for v in by_set.values())
    both = sum(1 for v in by_set.values() for a, b in zip(v, v[1:]) if a["track_uid"] and b["track_uid"])
    hours = {k: round((dt.datetime.fromisoformat(v[-1]["inicio"]) - dt.datetime.fromisoformat(v[0]["inicio"])).total_seconds() / 3600
                      + v[-1]["segundos_al_aire"] / 3600, 2) for k, v in by_set.items()}
    summary = {"archivos": len(files), "sets": n_sets, "sesiones_cortas_descartadas": short, "temas": len(out),
               "transiciones": trans, "temas_en_indice": sum(1 for r in out if r["track_uid"]),
               "transiciones_con_los_dos": both, "sets_de_3h_o_mas": {k: h for k, h in hours.items() if h >= 3}}
    return out, summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Sets de Gabriel desde la historia de Traktor (D1)")
    ap.add_argument("--historia", default=str(DEFAULT_HISTORY))
    ap.add_argument("--indice", default=str(rc.DEFAULT_INDEX))
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    args = ap.parse_args()
    ident = Identifier(Path(args.indice)) if Path(args.indice).exists() else None
    rows, summary = build(Path(args.historia), ident)
    out = Path(args.root) / "sets"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "d1.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    (out / "d1_resumen.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
