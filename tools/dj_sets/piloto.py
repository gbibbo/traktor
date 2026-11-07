"""
PURPOSE: Análisis del piloto de trayectorias de DJs (plan de playlists en vivo, etapa 2, punto 4) sobre
         compilaciones de Bedrock que traen la mezcla continua de cada CD y cada tema original suelto.
         Junta la lista de MusicBrainz (orden y duración de cada tema en cada CD), el manifest de
         youtube_audio.py y los .npz de clap_blocks.py, y mide:
           1. identificación: cada bloque de 30 s de la mezcla, ¿cae más cerca de su tema original que de
              los otros temas del disco y de los de la biblioteca (clap_full)? (lejos del cambio de tema y
              en la zona de transición, por separado);
           2. separación en temas: picos de novedad de la secuencia CLAP contra los límites reales
              (precisión, recuperación y F a ±30 s y ±60 s), con una línea de base de cortes regulares;
           3. trayectoria: distancia coseno entre temas consecutivos (originales, como en la biblioteca)
              frente a pares al azar del mismo disco y de la biblioteca, por hora del set; variación
              dentro de cada tema de la mezcla, por hora;
           4. sets crudos (sin límites conocidos): las mismas medidas por bloque y por hora.
         Escribe <root>/piloto/resultados.json.
CHANGELOG:
  - 2026-11-07: Creación.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)

DEFAULT_ROOT = REPO_ROOT / "artifacts" / "v4" / "dj_sets"
BLOCK_S = 30.0
EDGE_S = 30.0                       # zona de transición a cada lado de un cambio de tema
_NOISE = re.compile(r"\[(official audio|official video|official|audio)\]|\((official audio|official)\)", re.I)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = _NOISE.sub(" ", s)
    s = re.sub(r"\b(original mix|extended mix|mixed|live)\b", " ", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _yt_title(s: str) -> str:
    return norm(s.split(" - ", 1)[1] if " - " in (s or "") else s)


def pair_score(track: Dict, entry_title: str) -> float:
    """Parecido entre una pista de MusicBrainz y el título de YouTube: el de «artista título» completo o,
    si el artista se escribe distinto (alfabeto, colaboradores), el del título solo, un poco rebajado.
    Un título de YouTube cuyas palabras están todas en el de MusicBrainz (versiones con nombre largo)
    cuenta como 0.9 del título solo."""
    full = difflib.SequenceMatcher(None, norm(f"{track.get('artista', '')} {track.get('titulo', '')}"),
                                   norm(entry_title)).ratio()
    mt, yt = norm(track.get("titulo", "")), _yt_title(entry_title)
    title = difflib.SequenceMatcher(None, mt, yt).ratio()
    words = [w for w in yt.split() if len(w) >= 4]
    contained = 0.9 if words and set(yt.split()) <= set(mt.split()) else 0.0
    return max(full, 0.9 * title, contained)


def match_tracks(tracks: Sequence[Dict], entries: Sequence[Dict], min_ratio: float = 0.55) -> List[Optional[int]]:
    """Para cada pista de MusicBrainz ({artista, titulo}), el índice de la entrada de YouTube ({titulo}) que
    mejor coincide (pair_score), o None si ninguna pasa min_ratio. Cada entrada se usa una vez (la mejor
    gana)."""
    scores = []
    for i, t in enumerate(tracks):
        for j, e in enumerate(entries):
            scores.append((pair_score(t, e.get("titulo") or ""), i, j))
    out: List[Optional[int]] = [None] * len(tracks)
    used = set()
    for r, i, j in sorted(scores, reverse=True):
        if r < min_ratio:
            break
        if out[i] is None and j not in used:
            out[i], used = j, used | {j}
    return out


def bounds(ms: Sequence[Optional[int]]) -> List[Tuple[float, float]]:
    """Inicio y fin (s) de cada pista dentro de la mezcla continua, por las duraciones acumuladas."""
    t, out = 0.0, []
    for m in ms:
        d = (m or 0) / 1000.0
        out.append((t, t + d))
        t += d
    return out


def unit(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)


def novelty(B: np.ndarray, k: int) -> np.ndarray:
    """Distancia coseno entre la media de los k bloques anteriores y la de los k siguientes, en cada borde
    entre bloques (posición i = entre el bloque i-1 y el i)."""
    U = unit(B)
    n = len(U)
    nov = np.zeros(n)
    for i in range(1, n):
        a, b = U[max(0, i - k):i].mean(axis=0), U[i:min(n, i + k)].mean(axis=0)
        nov[i] = 1.0 - float(unit(a) @ unit(b))
    return nov


def peaks(nov: np.ndarray, min_gap: int, n_peaks: Optional[int] = None, thr: Optional[float] = None) -> List[int]:
    """Picos de mayor a menor con separación mínima min_gap; hasta n_peaks o por encima de thr."""
    order = np.argsort(-nov)
    chosen: List[int] = []
    for i in order:
        if i == 0 or (thr is not None and nov[i] < thr) or (n_peaks is not None and len(chosen) >= n_peaks):
            continue
        if all(abs(int(i) - c) >= min_gap for c in chosen):
            chosen.append(int(i))
    return sorted(chosen)


def prf(est: Sequence[float], ref: Sequence[float], tol: float) -> Dict[str, float]:
    """Precisión, recuperación y F de cortes estimados contra reales (cada real se empareja una vez)."""
    used, hit = set(), 0
    for e in est:
        best = min(((abs(e - r), j) for j, r in enumerate(ref) if j not in used), default=None)
        if best is not None and best[0] <= tol:
            used.add(best[1])
            hit += 1
    p = hit / len(est) if est else 0.0
    r = hit / len(ref) if ref else 0.0
    return {"precision": round(p, 3), "recuperacion": round(r, 3), "f": round(2 * p * r / (p + r), 3) if p + r else 0.0}


def summary(x: Sequence[float]) -> Dict[str, float]:
    a = np.asarray([v for v in x if v == v], dtype=float)
    if not len(a):
        return {"n": 0}
    return {"n": int(len(a)), "media": round(float(a.mean()), 4), "mediana": round(float(np.median(a)), 4),
            "p25": round(float(np.percentile(a, 25)), 4), "p75": round(float(np.percentile(a, 75)), 4)}


def load(root: Path, carpeta: str, archivo: str) -> Optional[dict]:
    p = root / "clap" / carpeta / (Path(archivo).stem + ".npz")
    return dict(np.load(p)) if p.exists() else None


def album(root: Path, carpeta: str, release: str, lib: np.ndarray, rng: np.random.Generator) -> Dict:
    man = json.loads((root / "audio" / carpeta / "manifest.json").read_text(encoding="utf-8"))
    mb = json.loads((root / "musicbrainz" / f"{release}.json").read_text(encoding="utf-8"))
    entries = man["entradas"]
    mixes = {int(m.group(1)): e for e in entries if (m := re.search(r"continuous mix (\d+)", e.get("titulo") or "", re.I))}
    originals = [e for e in entries if not re.search(r"continuous mix", e.get("titulo") or "", re.I)]
    orig_emb = {}
    for j, e in enumerate(originals):
        d = load(root, carpeta, e["archivo"]) if e.get("archivo") else None
        if d is not None and "dj" in d:
            orig_emb[j] = (d["dj"], d["blocks"].mean(axis=0))
    keys = sorted(orig_emb)
    O_dj = unit(np.stack([orig_emb[j][0] for j in keys])) if keys else np.zeros((0, 512))
    O_mean = unit(np.stack([orig_emb[j][1] for j in keys])) if keys else np.zeros((0, 512))
    L = unit(lib)
    res = {"album": man.get("titulo"), "release": release, "discos": [], "identificacion": {}, "separacion": {},
           "trayectoria": {}}
    ident = {k: [] for k in ("rango_disco_dj", "rango_disco_media", "rango_con_biblioteca", "rango_transicion")}
    track_level, seq, seq_hour, seq_pos = [], [], [], []
    pos = 0                                       # posición global de la pista en el set (todos los discos)
    within_by_hour: Dict[int, List[float]] = {}
    seg = {"k2_30": [], "k2_60": [], "k3_30": [], "k3_60": [], "regular_30": [], "regular_60": []}
    offset = 0.0
    for disc in mb["media"]:
        n = disc["disco"]
        e = mixes.get(n)
        d = load(root, carpeta, e["archivo"]) if e and e.get("archivo") else None
        tr = disc["pistas"]
        bd = bounds([t["ms"] for t in tr])
        m = match_tracks(tr, originals)
        info = {"disco": n, "pistas": len(tr), "emparejadas": sum(1 for x in m if x is not None and x in orig_emb),
                "minutos_mb": round(bd[-1][1] / 60, 1) if bd else 0,
                "minutos_mezcla": round(float(d["dur"]) / 60, 1) if d is not None else None}
        res["discos"].append(info)
        for i, (t, j) in enumerate(zip(tr, m)):
            if j is not None and j in orig_emb:
                seq.append(orig_emb[j][0])
                seq_hour.append(int((offset + bd[i][0]) // 3600))
                seq_pos.append(pos + i)
        pos += len(tr)
        if d is None:
            offset += bd[-1][1] if bd else 0
            continue
        B, starts = unit(d["blocks"]), d["starts"]
        mid = starts + BLOCK_S / 2
        for i, ((a, b), j) in enumerate(zip(bd, m)):
            inside = (mid >= a) & (mid < b)
            if j is None or j not in orig_emb or not inside.any():
                continue
            col = keys.index(j)
            core = inside & (mid >= a + EDGE_S) & (mid < b - EDGE_S)
            edge = inside & ~core
            for kind, mask in (("core", core), ("edge", edge)):
                for bi in np.where(mask)[0]:
                    s_dj, s_mean = O_dj @ B[bi], O_mean @ B[bi]
                    if kind == "core":
                        ident["rango_disco_dj"].append(int((s_dj > s_dj[col]).sum()) + 1)
                        ident["rango_disco_media"].append(int((s_mean > s_mean[col]).sum()) + 1)
                        ident["rango_con_biblioteca"].append(int((s_dj > s_dj[col]).sum() + (L @ B[bi] > s_dj[col]).sum()) + 1)
                    else:
                        ident["rango_transicion"].append(int((s_dj > s_dj[col]).sum()) + 1)
            if core.any():
                v = unit(B[core].mean(axis=0))
                track_level.append(int(np.argmax(O_dj @ v)) == col)
                within = 1.0 - B[core] @ v
                within_by_hour.setdefault(int((offset + a) // 3600), []).extend(within.tolist())
        ref = [a for a, _ in bd[1:]]
        mean_len = np.mean([b - a for a, b in bd]) if bd else 0
        for k in (2, 3):
            nov = novelty(B, k)
            est = [float(starts[i]) for i in peaks(nov, min_gap=4, n_peaks=len(ref))]
            for tol in (30, 60):
                seg[f"k{k}_{tol}"].append(prf(est, ref, tol))
        regular = list(np.arange(mean_len, float(d["dur"]) - 1, mean_len)) if mean_len else []
        for tol in (30, 60):
            seg[f"regular_{tol}"].append(prf(regular, ref, tol))
        offset += float(d["dur"])

    def top(r, k):
        return round(float(np.mean(np.asarray(r) <= k)), 3) if r else None
    for name, r in ident.items():
        res["identificacion"][name] = {"bloques": len(r), "top1": top(r, 1), "top5": top(r, 5),
                                       "mediana": float(np.median(r)) if r else None}
    res["identificacion"]["candidatos_disco"] = len(keys)
    res["identificacion"]["candidatos_con_biblioteca"] = len(keys) + len(L)
    res["identificacion"]["tema_por_mayoria_de_bloques"] = round(float(np.mean(track_level)), 3) if track_level else None
    for key, lst in seg.items():
        res["separacion"][key] = {m: round(float(np.mean([x[m] for x in lst])), 3) for m in ("precision", "recuperacion", "f")} if lst else None
    if len(seq) > 2:
        S = unit(np.stack(seq))
        adj = [i for i in range(len(S) - 1) if seq_pos[i + 1] == seq_pos[i] + 1]   # solo pistas vecinas
        steps = np.asarray([1.0 - float(S[i] @ S[i + 1]) for i in adj])
        step_hour = [seq_hour[i + 1] for i in adj]
        rnd_album = [1.0 - float(S[a] @ S[b]) for a, b in rng.integers(0, len(S), size=(2000, 2)) if a != b]
        rnd_lib = [1.0 - float(L[a] @ L[b]) for a, b in rng.integers(0, len(L), size=(2000, 2)) if a != b]
        res["trayectoria"] = {
            "pasos_consecutivos": summary(steps), "pares_al_azar_del_disco": summary(rnd_album),
            "pares_al_azar_biblioteca": summary(rnd_lib),
            "pasos_por_hora": {str(h): summary([s for s, hh in zip(steps, step_hour) if hh == h])
                               for h in sorted(set(step_hour))},
            "variacion_dentro_del_tema_por_hora": {str(h): summary(v) for h, v in sorted(within_by_hour.items())},
            "distancia_al_primer_tema_por_hora": {str(h): summary([1.0 - float(S[0] @ S[i]) for i in range(len(S)) if seq_hour[i] == h])
                                                  for h in sorted(set(seq_hour))},
        }
    return res


def raw_set(root: Path, carpeta: str) -> Dict:
    man = json.loads((root / "audio" / carpeta / "manifest.json").read_text(encoding="utf-8"))
    e = man["entradas"][0]
    d = load(root, carpeta, e["archivo"]) if e.get("archivo") else None
    if d is None:
        return {"set": man.get("titulo"), "estado": "sin embeddings"}
    B, starts = unit(d["blocks"]), d["starts"]
    step = 1.0 - np.sum(B[1:] * B[:-1], axis=1)
    hours = (starts[1:] // 3600).astype(int)
    nov = novelty(B, 3)
    est = peaks(nov, min_gap=6, thr=float(np.percentile(nov, 90)))
    lens = np.diff([0.0] + [float(starts[i]) for i in est] + [float(d["dur"])]) / 60
    return {"set": man.get("titulo"), "horas": round(float(d["dur"]) / 3600, 2),
            "paso_entre_bloques_por_hora": {str(h): summary(step[hours == h]) for h in sorted(set(hours.tolist()))},
            "temas_estimados": len(est) + 1, "minutos_por_tema_estimado": summary(lens)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Análisis del piloto de trayectorias")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--album", action="append", default=[], help="carpeta:release_id (CD de MusicBrainz)")
    ap.add_argument("--set", action="append", default=[], help="Carpeta de un set crudo")
    args = ap.parse_args()
    root = Path(args.root)
    lib = np.load(REPO_ROOT / "artifacts" / "v4" / "datasets" / "musica" / "representations" / "clap_full" / "embeddings.npy")
    rng = np.random.default_rng(0)
    out = {"albums": [album(root, *a.split(":", 1), lib, rng) for a in args.album],
           "sets_crudos": [raw_set(root, s) for s in args.set]}
    dest = root / "piloto"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "resultados.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
