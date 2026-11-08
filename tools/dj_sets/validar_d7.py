"""
PURPOSE: Valida que cada archivo bajado de las compilaciones mezcladas (D7) sea de verdad esa compilación.
         La validación de youtube_buscar.py por duraciones no alcanza para los discos enteros («continua»):
         en una playlist de colección con decenas de sets de unos 75 minutos, otro set dura lo mismo (±8 s)
         por azar (2026-11-07: Sasha GU013 disco 2 era «John Digweed - GU 019 Los Angeles CD1»). Dos pruebas:
           1. Título (decide). El título de YouTube tiene que coincidir con la compilación:
              - las palabras propias del título de MusicBrainz (sin las de serie, como «Global Underground»,
                que aparecen en 3 o más compilaciones, ni las vacías) están al menos a la mitad;
              - si el título de YouTube trae números (sin contar el del disco), alguno coincide con los de
                la compilación (los años valen también con dos cifras);
              - si trae un número de disco, es el asignado;
              - sin el nombre del DJ, hace falta además un número de la compilación que coincida;
              - siempre tiene que haber algo a favor: una palabra propia o un número que coincidan.
              En las «por_tema», el 80 % de las pistas tiene que coincidir por artista y título
              (piloto.pair_score >= 0,55).
           2. Audio: si los límites de MusicBrainz sirven para separar los temas. Para los discos enteros con
              3 temas o más, la novedad CLAP en los límites (con el mejor desfase de ±90 s, porque el video
              puede empezar corrido) frente a los mismos límites desplazados al azar (p por permutación,
              300). Con p > 0,5 el disco es el correcto pero sus límites no se usan tema a tema.
         Escribe <root>/sets/d7_validacion.json, que trayectorias_d7.py usa para quedarse con lo validado.
           python tools/dj_sets/validar_d7.py --root "D:/TRAKTOR ML datos/dj_sets"
CHANGELOG:
  - 2026-11-07: Creación (los discos ajenos que encontró la revisión de D7).
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from tools.dj_sets.piloto import bounds, norm, novelty, pair_score, unit  # noqa: E402

DEFAULT_ROOT = REPO_ROOT / "artifacts" / "v4" / "dj_sets"
STOP = set("the presents present mixed mix by series in at of and vol volume part cd disc disk live a de la "
           "compilation edition dj continuous continous full hd album official playlist session tape mins "
           "with feat".split())
_DISC = re.compile(r"\b(?:cd|disc|disk|continuous mix|continous mix)\s*[-#]?\s*0?(\d+)\b", re.I)
PAIR_MIN = 0.55
TRACK_SHARE = 0.8


def clean(s: str) -> str:
    """Normaliza juntando las siglas («F.A.C.T.» -> «fact», «'98» -> «98»); los demás puntos separan
    palabras («1990.Sasha.Bedroom» -> «1990 sasha bedroom»)."""
    s = re.sub(r"(?<![^\W\d_])([^\W\d_])\.(?=[^\W\d_](?:\.|\b))", r"\1", s or "")
    return norm(re.sub(r"['’´]", "", s))


def disc_number(title: str) -> Optional[int]:
    m = _DISC.search(title or "")
    return int(m.group(1)) if m else None


def numbers(title: str) -> Set[int]:
    """Números del título sin el del disco; los años también con dos cifras."""
    s = _DISC.sub(" ", re.sub(r"['’´]", "", title or "").replace(".", " "))
    out: Set[int] = set()
    for x in re.findall(r"\d+", s):
        n = int(x)
        out.add(n)
        if 1950 <= n <= 2099:
            out.add(n % 100)
    return out


def series_words(release_titles: Iterable[str], min_count: int = 3) -> Set[str]:
    """Palabras que aparecen en min_count títulos o más: son de la serie, no de la compilación."""
    c = collections.Counter()
    for t in release_titles:
        c.update(set(clean(t).split()))
    return {w for w, k in c.items() if k >= min_count}


def title_check(dj: str, release_title: str, disc: int, n_discs: int, yt_title: str,
                series: Set[str]) -> Tuple[bool, str]:
    y = clean(yt_title)
    ywords = set(y.split())
    dj_words = [w for w in clean(dj).split() if len(w) > 2]
    has_dj = bool(dj_words) and dj_words[-1] in ywords
    rwords = [w for w in clean(release_title).split()
              if w not in STOP and w not in dj_words and not w.isdigit() and len(w) > 1]
    own = [w for w in rwords if w not in series] or rwords
    rn, yn = numbers(release_title), numbers(yt_title)
    d = disc_number(yt_title)
    if n_discs > 1 and d is not None and d != disc:
        return False, f"dice disco {d}, asignado {disc}"
    if yn and rn and not (rn & yn):
        return False, f"números {sorted(yn)} frente a {sorted(rn)}"
    hits = sum(w in ywords for w in own)
    if own and hits / len(own) < 0.5:
        return False, f"faltan palabras propias {own}"
    num_hit = bool(rn & yn)
    if not has_dj and not num_hit:
        return False, "sin el DJ ni un número de la compilación"
    if not hits and not num_hit:
        return False, "nada a favor"
    return True, ""


OFFSETS = np.arange(-90, 91, 10)            # s: el video puede empezar corrido respecto del disco
BOUNDARY_P_MAX = 0.5


def boundary_p(z: Dict[str, np.ndarray], ms: Sequence[Optional[int]], rng: np.random.Generator,
               n_null: int = 300) -> Optional[float]:
    """p de que la novedad CLAP en los límites de MusicBrainz (con el mejor desfase de ±90 s) sea tan alta
    como con límites desplazados al azar (también con su mejor desfase)."""
    if len(ms) < 3 or len(z["starts"]) < 6:
        return None
    B, starts, dur = unit(z["blocks"]), z["starts"], float(z["dur"])
    nov = novelty(B, 2)
    pct = np.argsort(np.argsort(nov)) / max(1, len(nov) - 1)

    def score(times: Sequence[float]) -> float:
        idx = [int(np.argmin(np.abs(starts + 15 - t))) for t in times]
        return float(np.mean([pct[max(0, i - 1):i + 2].max() for i in idx]))
    ref = np.asarray([a for a, _ in bounds(ms)[1:]])

    def best(times: np.ndarray) -> float:
        return max(score((times + o) % dur) for o in OFFSETS)
    obs = best(ref)
    null = [best((ref + s) % dur) for s in rng.uniform(60, max(61, dur - 60), n_null)]
    return round((1 + sum(x >= obs for x in null)) / (n_null + 1), 4)


def validate_all(root: Path) -> Dict:
    mans = sorted((root / "audio").glob("d7_*/manifest.json"))
    data = []
    for m in mans:
        man = json.loads(m.read_text(encoding="utf-8"))
        mb = json.loads((root / "musicbrainz" / f"{man['release']}.json").read_text(encoding="utf-8"))
        data.append((m.parent.name, man, mb))
    series = series_words(mb["titulo"] for _, _, mb in data)
    rng = np.random.default_rng(0)
    out: Dict[str, Dict] = {}
    for folder, man, mb in data:
        files: Dict[str, Dict] = {}
        if man["forma"] == "continua":
            for e in man["entradas"]:
                if not e.get("archivo"):
                    continue
                ok, why = title_check(man["artista"], mb["titulo"], e["disco"], len(mb["media"]),
                                      e.get("titulo") or "", series)
                disc = next(x for x in mb["media"] if x["disco"] == e["disco"])
                p = root / "clap" / folder / (Path(e["archivo"]).stem + ".npz")
                bp = boundary_p(np.load(p), [t["ms"] for t in disc["pistas"]], rng) if p.exists() else None
                files[e["archivo"]] = {"disco": e["disco"], "titulo": e.get("titulo"), "valido": ok, "motivo": why,
                                       "p_limites": bp, "pistas_disco": len(disc["pistas"]),
                                       "limites_usables": ok and (bp is None or bp <= BOUNDARY_P_MAX)}
            valid = any(f["valido"] for f in files.values())
            why = ""
        else:
            tracks = {(x["disco"], t["posicion"]): t for x in mb["media"] for t in x["pistas"]}
            got = [e for e in man["entradas"] if e.get("archivo")]
            good = [e for e in got if pair_score(tracks[(e["disco"], e["pista"])], e.get("titulo") or "") >= PAIR_MIN]
            valid = bool(got) and len(good) >= TRACK_SHARE * len(got)
            why = f"{len(good)}/{len(got)} pistas coinciden por título"
            for e in got:
                files[e["archivo"]] = {"disco": e["disco"], "pista": e["pista"], "titulo": e.get("titulo"),
                                       "valido": valid, "limites_usables": valid}
        out[folder] = {"dj": man["artista"], "compilacion": mb["titulo"], "forma": man["forma"], "valida": valid,
                       "motivo": why, "archivos": files}
    return {"palabras_de_serie": sorted(series), "compilaciones": out}


def main() -> int:
    ap = argparse.ArgumentParser(description="Validación por título y por límites de los archivos D7")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    args = ap.parse_args()
    root = Path(args.root)
    res = validate_all(root)
    dest = root / "sets"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "d7_validacion.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    comps = res["compilaciones"].values()
    cont = [f for c in comps if c["forma"] == "continua" for f in c["archivos"].values()]
    ok = [f for f in cont if f["valido"]]
    pt = [c for c in comps if c["forma"] == "por_tema"]
    ps_ok = [f["p_limites"] for f in ok if f["p_limites"] is not None]
    ps_bad = [f["p_limites"] for f in cont if not f["valido"] and f["p_limites"] is not None]
    usable = [f for f in ok if f["limites_usables"]]
    print(f"[OK] discos enteros válidos {len(ok)} de {len(cont)} ({len(usable)} con límites usables); "
          f"por tema {sum(c['valida'] for c in pt)} de {len(pt)}")
    if ps_ok and ps_bad:
        print(f"     p de los límites (mediana): válidos {np.median(ps_ok):.3f} (n {len(ps_ok)}), "
              f"rechazados {np.median(ps_bad):.3f} (n {len(ps_bad)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
