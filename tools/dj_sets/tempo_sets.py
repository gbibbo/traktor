"""
PURPOSE: BPM estimado de los temas de sets profesionales que no lo tienen (plan de playlists en vivo, sección
         11, medidas E9 y E10: las líneas de base piden pares del mismo rango de BPM y un orden por BPM solo).
           - D7 validados (sets/d7_temas.csv y d7_clap.npz): un tramo de hasta 60 s en el centro de cada tema,
             lejos de los cambios. Es el tempo al que sonó en la mezcla, no el del original.
           - Originales de Bedrock del piloto: 3 tramos de 30 s del 35-65 %, los mismos del embedding.
         Usa src/v4/common/tempo.py (numpy y scipy, 95-150 BPM), el estimador de phase1_tags. Escribe
         <root>/sets/tempo_temas.csv (fuente, clave, bpm, inicio_s, segundos); lo ya calculado no se repite.
         --verificar N compara con el BPM del índice en N temas de la biblioteca. --limite N para la prueba
         chica, --low-priority para la corrida completa.
           python tools/dj_sets/tempo_sets.py --root "D:/TRAKTOR ML datos/dj_sets" --verificar 20
CHANGELOG:
  - 2026-11-08: Creación (unidad A1 de la sección 11 del plan).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from src.v4.common.audio_utils import read_dj_segments  # noqa: E402
from src.v4.common.tempo import estimate_bpm  # noqa: E402
from src.v4.pipeline.extract_representations import lower_priority  # noqa: E402
from tools.dj_sets.piloto import bounds  # noqa: E402
from tools.dj_sets.trayectorias_d7 import accepted  # noqa: E402
from tools.traktor_live import recomendador as rc  # noqa: E402

DEFAULT_ROOT = REPO_ROOT / "artifacts" / "v4" / "dj_sets"
SR = 44100
EDGE_S = 60.0
EXCERPT_S = 60.0
BEDROCK = ("PLTywFuVWois0FgkZNrMZMEHfbSF9hc_TC", "PLTywFuVWois3bEYbWvO18JWhOzBYsYLCH",
           "PLTywFuVWois3fWN_T47ysRqmMsk3mCaXv")
FIELDS = ["fuente", "clave", "bpm", "inicio_s", "segundos"]


def excerpt(a: float, b: float, edge: float = EDGE_S, length: float = EXCERPT_S) -> Tuple[float, float]:
    """(inicio, duración) del tramo a medir dentro de [a, b]: hasta `length` segundos en el centro, a más de
    `edge` de cada cambio; en un tema corto, el medio del tema."""
    span = b - a
    if span <= 0:
        return a, 0.0
    if span - 2 * edge >= 20.0:
        low, high = a + edge, b - edge
    else:
        low, high = a + span / 4, b - span / 4
    dur = min(length, high - low)
    return (low + high - dur) / 2, dur


def decode(path: Path, start: float, seconds: float, sr: int = SR) -> np.ndarray:
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{seconds:.3f}", "-i", str(path), "-ac", "1",
                        "-ar", str(sr), "-f", "f32le", "-"], capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "replace")[-300:])
    return np.frombuffer(p.stdout, dtype="<f4")


def d7_spans(root: Path, validation: Optional[Dict], mode: str = "limites") -> Iterator[Tuple[int, Optional[Tuple[Path, float, float]], str, str]]:
    """(fila, (archivo, inicio, fin) o None, artista, título), en el mismo orden que trayectorias_d7.build_store.
    En las «por_tema» el fin es None: dura lo que dure el archivo."""
    fila = 0
    for man_path in sorted((root / "audio").glob("d7_*/manifest.json")):
        man = json.loads(man_path.read_text(encoding="utf-8"))
        folder = man_path.parent.name
        mb = json.loads((root / "musicbrainz" / f"{man['release']}.json").read_text(encoding="utf-8"))
        files = [e for e in man["entradas"] if e.get("archivo")]

        def usable(e: Optional[Dict]) -> Optional[Path]:
            if e is None or not (root / "clap" / folder / (Path(e["archivo"]).stem + ".npz")).exists():
                return None
            return root / "audio" / folder / e["archivo"]

        for disc in mb["media"]:
            n, tracks = disc["disco"], disc["pistas"]
            per: List[Optional[Tuple[Path, float, Optional[float]]]] = []
            if man["forma"] == "continua":
                p = usable(next((x for x in files if x["disco"] == n and accepted(validation, folder, x["archivo"], mode)), None))
                per = [(p, a, b) for a, b in bounds([t["ms"] for t in tracks])] if p is not None else [None] * len(tracks)
            else:
                for t in tracks:
                    p = usable(next((x for x in files if x["disco"] == n and x["pista"] == t["posicion"]
                                     and accepted(validation, folder, x["archivo"], mode)), None))
                    per.append((p, 0.0, None) if p is not None else None)
            for t, src in zip(tracks, per):
                yield fila, src, t.get("artista") or "", t.get("titulo") or ""
                fila += 1


def load_done(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as fh:
        return {r["clave"]: r for r in csv.DictReader(fh)}


def save(path: Path, rows: Dict[str, Dict[str, object]]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows.values():
            w.writerow({k: r.get(k, "") for k in FIELDS})
    tmp.replace(path)


def duration(root: Path, path: Path) -> float:
    z = np.load(root / "clap" / path.parent.name / (path.stem + ".npz"))
    return float(z["dur"])


def d7_todo(root: Path, limit: Optional[int]) -> Iterator[Tuple[str, Tuple[Path, float, float]]]:
    """Tareas de D7: solo las filas con vector (borde de 60 s). Comprueba que el orden sea el de d7_temas.csv."""
    validation = json.loads((root / "sets" / "d7_validacion.json").read_text(encoding="utf-8"))
    with (root / "sets" / "d7_temas.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    V = np.load(root / "sets" / "d7_clap.npz")["borde_60"]
    n = 0
    for fila, src, artista, titulo in d7_spans(root, validation):
        r = rows[fila]
        if r["artista"] != artista or r["titulo"] != titulo:
            raise RuntimeError(f"La fila {fila} no coincide con d7_temas.csv: {artista} - {titulo}")
        if src is None or np.isnan(V[fila, 0]):
            continue
        if src[2] is None:
            src = (src[0], 0.0, duration(root, src[0]))
        yield f"d7:{fila}", src
        n += 1
        if limit and n >= limit:
            return


def bedrock_todo(root: Path, limit: Optional[int]) -> Iterator[Tuple[str, Path]]:
    n = 0
    for folder in BEDROCK:
        man = json.loads((root / "audio" / folder / "manifest.json").read_text(encoding="utf-8"))
        for e in man["entradas"]:
            if not e.get("archivo") or "continuous mix" in (e.get("titulo") or "").lower():
                continue
            yield f"bedrock:{folder}:{e['archivo']}", root / "audio" / folder / e["archivo"]
            n += 1
            if limit and n >= limit:
                return


def verify(index_path: Path, n: int) -> None:
    """BPM estimado frente al del índice en n temas de la biblioteca, repartidos a lo largo del índice."""
    data = json.loads(index_path.read_text(encoding="utf-8"))
    package = Path(data["paquete"])
    temas = [t for t in data["temas"] if t.get("bpm") and t.get("archivo") and (package / t["archivo"]).exists()]
    if not temas:
        print(f"[ERROR] No encuentro los temas del paquete en {package}")
        return
    picks = temas[:: max(1, len(temas) // n)][:n]
    err = []
    t0 = time.perf_counter()
    for t in picks:
        segs, sr = read_dj_segments(package / t["archivo"], 30.0, 0, 3, 0)
        est = estimate_bpm(segs, sr)
        if est is None:
            continue
        e = est - float(t["bpm"])
        err.append(e)
        print(f"  índice {float(t['bpm']):6.1f}  estimado {est:6.2f}  error {e:+6.2f}  {t['artista'][:24]} - {t['titulo'][:30]}", flush=True)
    a = np.abs(np.asarray(err))
    print(f"[verificar] {len(a)} temas: error absoluto mediano {np.median(a):.2f}; a ±2 BPM {np.mean(a <= 2):.0%}; "
          f"a ±4 BPM {np.mean(a <= 4):.0%}; {(time.perf_counter() - t0) / max(1, len(picks)):.1f} s por tema")


def main() -> int:
    ap = argparse.ArgumentParser(description="BPM estimado de los temas de D7 y de los originales de Bedrock")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--indice", default=str(rc.DEFAULT_INDEX))
    ap.add_argument("--limite", type=int, default=None, help="Solo los primeros N de cada fuente (prueba chica)")
    ap.add_argument("--verificar", type=int, default=0, help="Compara con el BPM del índice en N temas y termina")
    ap.add_argument("--low-priority", action="store_true")
    args = ap.parse_args()
    if args.low_priority:
        lower_priority()
    if args.verificar:
        verify(Path(args.indice), args.verificar)
        return 0
    root = Path(args.root)
    out = root / "sets" / "tempo_temas.csv"
    done: Dict[str, Dict[str, object]] = dict(load_done(out))
    t0, nuevos, fallos = time.perf_counter(), 0, 0
    for clave, (path, a, b) in d7_todo(root, args.limite):
        if clave in done:
            continue
        ini, seg = excerpt(a, b)
        try:
            est = estimate_bpm([decode(path, ini, seg)], SR) if seg >= 8 else None
        except RuntimeError as exc:
            print(f"[WARN] {clave}: {exc}", flush=True)
            est, fallos = None, fallos + 1
        done[clave] = {"fuente": "d7", "clave": clave, "bpm": "" if est is None else est, "inicio_s": round(ini, 1), "segundos": round(seg, 1)}
        nuevos += 1
        if nuevos % 25 == 0:
            save(out, done)
            print(f"  {nuevos} temas, {(time.perf_counter() - t0) / nuevos:.1f} s por tema", flush=True)
    for clave, path in bedrock_todo(root, args.limite):
        if clave in done:
            continue
        try:
            segs, sr = read_dj_segments(path, 30.0, 0, 3, 0)
            est = estimate_bpm(segs, sr)
        except Exception as exc:  # noqa: BLE001  (un archivo que no se decodifica no corta la corrida)
            print(f"[WARN] {clave}: {type(exc).__name__}: {exc}", flush=True)
            est, fallos = None, fallos + 1
        done[clave] = {"fuente": "bedrock", "clave": clave, "bpm": "" if est is None else est, "inicio_s": "", "segundos": 90}
        nuevos += 1
        if nuevos % 25 == 0:
            save(out, done)
            print(f"  {nuevos} temas, {(time.perf_counter() - t0) / nuevos:.1f} s por tema", flush=True)
    save(out, done)
    bpms = np.asarray([float(r["bpm"]) for r in done.values() if r["bpm"] != ""])
    print(f"[OK] {len(done)} temas ({nuevos} nuevos, {fallos} fallos, {len(done) - len(bpms)} sin BPM) -> {out}; "
          f"BPM p5-p50-p95: {np.percentile(bpms, 5):.1f} {np.median(bpms):.1f} {np.percentile(bpms, 95):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
