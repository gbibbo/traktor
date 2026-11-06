"""
PURPOSE: Prueba de escucha a ciegas según la decisión del auditor del 2026-11-06 (P4 modificada ahora ->
         P5(b) después; docs/plans/armonia_tiv_protocolo.md, sección 9). Dos fases:
           pilot    FEM#10 contra «sin blanqueo», ambos como brazo TIV, en contextos donde eligen distinto.
                    Sirve para congelar UNA configuración; es exploratorio (sin umbral de significación).
           confirm  la configuración congelada contra dos controles, en contextos nuevos y representativos:
                    C1 = el recomendador actual (tools/traktor_live/recomendador.py en main: tonalidad de
                         Traktor con los tags de respaldo, BPM a ±20, corte Camelot >= 0,60 transponiendo
                         hasta ±2, orden por CLAP);
                    C2 = sin criterio armónico (mismo pool y orden CLAP, sin filtro tonal, sin transponer).
                    C2 solo se confirma si pasa C1. Si los dos brazos eligen el mismo tema con la misma
                    transposición, el contexto es un empate y no se escucha.
         Brazo TIV: la misma política que C1 con el filtro tonal reemplazado por uno TIV de la misma
         selectividad media (umbral calibrado en el split dev); transposición hasta ±2 con un costo por
         semitono también calibrado en dev; orden por CLAP.
         Regiones de la mezcla (fijas antes de escuchar): 2 de cada 3 contextos outro->intro y 1 central
         (render.region_starts); mismo tramo del principal en las dos mezclas; plantilla fija.
         Independencia: ningún tema aparece en más de un ítem (principal o elegido) entre todas las fases;
         los principales salen de grupos de artistas distintos para piloto, C1 y C2.
         score: U = (victorias + 0,5 · empates) / N con N = 60 contextos por comparación (incluye los
         empates automáticos); binomial exacta unilateral sobre los no empates; aprueba si U >= 0,60 y
         p < 0,05, en orden C1 -> C2. 20 repetidos fuera de N; hacen falta 16 acuerdos.
         Uso:
           python tools/harmonic_tiv/listening_test.py build --phase pilot --out <dir>
           python tools/harmonic_tiv/listening_test.py build --phase confirm --config '{"spec":"raw_lin"}'
               --used-from <dir del piloto>/key.json --out <dir>
           python tools/harmonic_tiv/listening_test.py score --dir <dir> --answers <respuestas.json>
CHANGELOG:
  - 2026-11-06: Creación (rama harmonic-tiv).
  - 2026-11-06: Rehecha según el auditor: controles que reproducen el recomendador actual y "sin armonía",
                contextos representativos con empates automáticos, U con empates, contraste secuencial,
                20 repetidos (16 acuerdos), piloto separado, temas sin repetir entre ítems (antes un
                principal podía ser el candidato de otro ítem), regiones de mezcla reales. La página sale
                de listening_page.html.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import shutil
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.v4.common.harmonic import best_key_shift  # noqa: E402
from src.v4.harmonic_tiv import features as ft  # noqa: E402
from src.v4.harmonic_tiv import render as rd  # noqa: E402
from src.v4.harmonic_tiv import tiv as tv  # noqa: E402

BPM_RANGE = 20.0            # como recomendador.py
KEY_MIN = 0.60              # corte Camelot aprobado
MAX_SHIFT = 2
PILOT_CONFIGS = {"FEM#10": {}, "sin blanqueo": {"spec": "raw_lin"}}
N_PER_COMPARISON = 60
N_PILOT = 30
N_REPEATS = 20
MIN_REPEAT_AGREEMENTS = 16
U_MIN = 0.60
ALPHA = 0.05
MODE_WEIGHTS = (("outro_intro", 2), ("central", 1))
PAGE = Path(__file__).with_name("listening_page.html")


def artist_key(a: str) -> str:
    a = (a or "").lower()
    a = re.split(r",|&| feat\.? | ft\.? | x | vs\.? | and ", a)[0]
    return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKD", a))


def artist_group(a: str, salt: str = "escucha_tiv_v1") -> str:
    """pilot / c1 / c2 por hash estable del artista principal (los principales no comparten artista)."""
    h = hashlib.sha256((salt + artist_key(a)).encode("utf-8")).digest()[0] % 3
    return ("pilot", "c1", "c2")[h]


class Library:
    """Lo que el recomendador ve de cada tema: tonalidad (Traktor manda sobre los tags), BPM (Traktor manda
    sobre los tags), embedding CLAP, y el TIV de cada configuración."""

    def __init__(self, eval_dir: Path, dataset_root: Path, cache: Path):
        k = pd.read_csv(eval_dir / "keys_musica.csv")
        bpm = pd.read_parquet(dataset_root / "features" / "bpm_key.parquet")[["track_uid", "bpm"]]
        k = k.merge(bpm, on="track_uid", how="left").set_index("track_uid")
        d = dataset_root / "representations" / "clap_full"
        emb = np.load(d / "embeddings.npy").astype(np.float64)
        emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        uids = json.loads((d / "track_uids.json").read_text(encoding="utf-8"))
        clap = {u: emb[i] for i, u in enumerate(uids)}
        self.cache = cache
        self.df = k
        self.ids = [u for u in k.index if (cache / f"{u}.npz").exists()]
        self.key = {u: (k.at[u, "k_traktor"] if k.at[u, "k_traktor"] != "?" else k.at[u, "k_tag"]) for u in self.ids}
        self.bpm = {u: (float(k.at[u, "traktor_bpm"]) if pd.notna(k.at[u, "traktor_bpm"])
                        else (float(k.at[u, "bpm"]) if pd.notna(k.at[u, "bpm"]) else None)) for u in self.ids}
        self.vec = {u: clap.get(u) for u in self.ids}
        self.split = k["split"].to_dict()
        self.artist = k["artist"].fillna("").to_dict()
        self.tivs: dict = {}

    def tiv(self, name: str, change: dict) -> dict:
        if name not in self.tivs:
            self.tivs[name] = ft.collection_tivs(self.cache, self.ids, ft.THESIS_FEM10.with_(**change))
        return self.tivs[name]

    def pool(self, t: str, used: set) -> list:
        """Candidatos del recomendador: otro tema, sin usar, BPM a ±20 del principal, con CLAP si el
        principal lo tiene."""
        b0, v0 = self.bpm.get(t), self.vec.get(t)
        out = []
        for x in self.ids:
            if x == t or x in used:
                continue
            if b0 is not None and (self.bpm.get(x) is None or abs(self.bpm[x] - b0) > BPM_RANGE):
                continue
            if v0 is not None and self.vec.get(x) is None:
                continue
            out.append(x)
        return out

    def clap(self, t: str, x: str) -> float:
        v0, v = self.vec.get(t), self.vec.get(x)
        return float(v0 @ v) if v0 is not None and v is not None else 0.0


# --------------------------------------------------------------------------- políticas

def camelot_pass(key_of: dict, t: str, pool: list):
    """Filtro del recomendador actual: (pasan, {x: transposición sugerida})."""
    k0 = key_of.get(t, "?")
    if k0 == "?":
        return list(pool), {x: 0 for x in pool}
    passing, shifts = [], {}
    for x in pool:
        kx = key_of.get(x, "?")
        if kx == "?":
            continue
        score, shift, _ = best_key_shift(k0, kx, max_shift=MAX_SHIFT)
        if score >= KEY_MIN:
            passing.append(x)
            shifts[x] = int(shift)
    return passing, shifts


def tiv_scores(tivs: dict, t: str, pool: list, lam: float):
    """Por candidato: (score TIV con la mejor transposición en ±2 menos lam por semitono, transposición)."""
    shifts = np.arange(-MAX_SHIFT, MAX_SHIFT + 1)
    C = np.stack([tivs[x] for x in pool])
    hc = tv.hc_tivlib(tivs[t][None, None, :], tv.transpose(C, shifts))          # (M, 5)
    val = (1.0 - hc) - lam * np.abs(shifts)[None, :]
    best = np.argmax(val - 1e-12 * np.abs(shifts)[None, :], axis=1)              # empate -> menor |s|
    return val[np.arange(len(pool)), best], shifts[best]


def tiv_pass(tivs: dict, t: str, pool: list, lam: float, tau: float):
    pool = [x for x in pool if x in tivs]
    if not pool or t not in tivs:
        return [], {}
    sc, sh = tiv_scores(tivs, t, pool, lam)
    return [x for x, s in zip(pool, sc) if s >= tau], {x: int(s) for x, s in zip(pool, sh)}


def pick(sim, t: str, passing: list, shifts: dict):
    """El recomendador ordena por CLAP: el primero es la sugerencia."""
    if not passing:
        return None, None
    best = max(passing, key=lambda x: sim(t, x))
    return best, shifts.get(best, 0)


def calibrate(lib: Library, tivs: dict, seed: int = 0, n_targets: int = 300) -> dict:
    """En el split dev: lam = 0,25 · mediana del rango 5-95 % del score TIV sin transponer en el pool
    (un semitono cuesta lo mismo, en proporción, que en harmonic.py: 0,15 de un rango de 0,6); tau =
    umbral para que el filtro TIV deje pasar, en promedio, la misma fracción del pool que el filtro
    Camelot (misma selectividad)."""
    rng = np.random.default_rng(seed)
    dev = [u for u in lib.ids if lib.split.get(u) == "dev" and lib.key.get(u, "?") != "?" and u in tivs]
    dev = list(rng.permutation(dev))[:n_targets]
    spreads, fracs, pools = [], [], []
    for t in dev:
        pool = [x for x in lib.pool(t, set()) if x in tivs]
        if len(pool) < 30:
            continue
        hc0 = 1.0 - tv.hc_tivlib(tivs[t][None, :], np.stack([tivs[x] for x in pool]))
        spreads.append(np.percentile(hc0, 95) - np.percentile(hc0, 5))
        passing, _ = camelot_pass(lib.key, t, pool)
        fracs.append(len(passing) / len(pool))
        pools.append((t, pool))
    lam = 0.25 * float(np.median(spreads))
    target_frac = float(np.mean(fracs))
    allv = [tiv_scores(tivs, t, pool, lam)[0] for t, pool in pools]
    grid = np.quantile(np.concatenate(allv), np.linspace(0.0, 1.0, 401))
    best_tau, best_err = float(grid[0]), 1e9
    for tau in grid:
        err = abs(np.mean([(sc >= tau).mean() for sc in allv]) - target_frac)
        if err < best_err:
            best_tau, best_err = float(tau), err
    return {"lambda": lam, "tau": best_tau, "camelot_pass_frac_dev": round(target_frac, 4),
            "tiv_pass_frac_dev": round(float(np.mean([(sc >= best_tau).mean() for sc in allv])), 4),
            "n_dev_targets": len(pools)}


def make_tiv_arm(lib: Library, tivs: dict, cal: dict):
    return lambda t, pool: pick(lib.clap, t, *tiv_pass(tivs, t, pool, cal["lambda"], cal["tau"]))


# --------------------------------------------------------------------------- selección de contextos

def contexts(lib: Library, group: str, rng: np.random.Generator):
    """Principales del split test, del grupo de artistas pedido, en orden al azar (representativos)."""
    ts = [u for u in lib.ids if lib.split.get(u) == "test" and artist_group(lib.artist.get(u, "")) == group
          and lib.bpm.get(u) is not None and lib.vec.get(u) is not None]
    return list(rng.permutation(ts))


def choose_mode(rng: np.random.Generator) -> str:
    modes = [m for m, w in MODE_WEIGHTS for _ in range(w)]
    return modes[int(rng.integers(0, len(modes)))]


def build_items(targets, n, pool_fn, arm_a, arm_b, used, rng, only_disagreements):
    """Contextos en el orden de `targets`. Ningún tema se repite entre ítems (principal ni elegidos):
    `used` se comparte entre comparaciones y fases. Devuelve (ítems, motivos de los saltados)."""
    items, skipped = [], {"tema_repetido": 0, "pool_chico": 0, "sin_eleccion": 0, "acuerdo": 0}
    for t in targets:
        if len(items) >= n:
            break
        if t in used:
            skipped["tema_repetido"] += 1
            continue
        pool = pool_fn(t, used | {t})
        if len(pool) < 30:
            skipped["pool_chico"] += 1
            continue
        a, sa = arm_a(t, pool)
        b, sb = arm_b(t, pool)
        if a is None or b is None:
            skipped["sin_eleccion"] += 1
            continue
        same = a == b and sa == sb
        if same and only_disagreements:
            skipped["acuerdo"] += 1
            continue
        items.append({"target": t, "pick_a": a, "shift_a": int(sa), "pick_b": b, "shift_b": int(sb),
                      "auto_tie": bool(same), "mode": choose_mode(rng)})
        used.update({t, a, b})
    return items, skipped


def track_info(lib: Library, u: str) -> rd.TrackInfo:
    row = lib.df.loc[u]
    return rd.TrackInfo(uid=u, path=row["source_path"], bpm=float(lib.bpm[u]),
                        cache=ft.load_cache(lib.cache / f"{u}.npz"),
                        grid_start_s=float(row["grid_start_s"]) if pd.notna(row["grid_start_s"]) else None,
                        grid_bpm=float(row["grid_bpm"]) if pd.notna(row["grid_bpm"]) else None)


def write_page(out: Path, page_items: list, test_id: str) -> None:
    html = PAGE.read_text(encoding="utf-8").replace("__ITEMS__", json.dumps(page_items)).replace("__TEST_ID__", test_id)
    (out / "index.html").write_text(html, encoding="utf-8")


def render_items(lib, items, out, rng, label_a, label_b, tag, key_items, page_items, render=True):
    """Renderiza los contextos que no son empate automático; el orden de las dos mezclas va al azar."""
    for n, it in enumerate(items, 1):
        iid = f"{tag}{n:03d}"
        key_items[iid] = {**it, "comparison": tag, "arm_a": label_a, "arm_b": label_b, "repeat_of": None}
        if it["auto_tie"]:
            continue
        fa, fb = out / "audio" / f"{secrets.token_hex(4)}.mp3", out / "audio" / f"{secrets.token_hex(4)}.mp3"
        params = {}
        if render:
            params = rd.render_pair(track_info(lib, it["target"]), track_info(lib, it["pick_a"]),
                                    track_info(lib, it["pick_b"]), fa, fb, shift_a=it["shift_a"],
                                    shift_b=it["shift_b"], mode=it["mode"])
        a_first = bool(rng.integers(0, 2))
        c1, c2 = (fa, fb) if a_first else (fb, fa)
        key_items[iid].update(params)
        key_items[iid]["clip_of_a"] = 1 if a_first else 2
        page_items.append({"id": iid, "c1": f"audio/{c1.name}", "c2": f"audio/{c2.name}"})
        print(f"  {iid} {it['mode']} {params.get('target_start_s')}", flush=True)


def add_repeats(out, rng, key_items, page_items, n_rep, copy_audio=True):
    """Repetidos con el mismo audio, otros nombres y el orden invertido; fuera de N."""
    rendered = [pi for pi in page_items if key_items[pi["id"]]["repeat_of"] is None]
    for r, idx in enumerate(rng.choice(len(rendered), size=min(n_rep, len(rendered)), replace=False), 1):
        orig = rendered[idx]
        new_c = []
        for c in (orig["c2"], orig["c1"]):
            dst = out / "audio" / f"{secrets.token_hex(4)}.mp3"
            if copy_audio:
                shutil.copy2(out / c, dst)
            new_c.append(f"audio/{dst.name}")
        rid = f"r{r:03d}"
        ok = key_items[orig["id"]]
        key_items[rid] = {**ok, "clip_of_a": 3 - ok["clip_of_a"], "repeat_of": orig["id"]}
        page_items.append({"id": rid, "c1": new_c[0], "c2": new_c[1]})


def shuffle_spaced(page_items, key_items, rng):
    """Orden al azar; un repetido queda a 5 ítems o más de su original."""
    rep_of = {pi["id"]: key_items[pi["id"]]["repeat_of"] for pi in page_items}
    page = list(page_items)
    for _ in range(2000):
        page = [page_items[i] for i in rng.permutation(len(page_items))]
        where = {pi["id"]: n for n, pi in enumerate(page)}
        if all(abs(where[i] - where[o]) >= 5 for i, o in rep_of.items() if o):
            break
    return page


def cmd_build(args) -> int:
    rng = np.random.default_rng(args.seed)
    lib = Library(args.eval_dir, args.dataset_root, args.cache)
    out = args.out
    (out / "audio").mkdir(parents=True, exist_ok=True)
    used: set = set()
    if args.used_from:
        used |= set(json.loads(Path(args.used_from).read_text(encoding="utf-8")).get("used", []))
    test_id = secrets.token_hex(4)
    key = {"test_id": test_id, "phase": args.phase, "seed": args.seed, "items": {}, "calibration": {}, "skipped": {}}
    page_items: list = []
    render = not args.dry_run
    if args.phase == "pilot":
        arms = {}
        for name, change in PILOT_CONFIGS.items():
            tivs = lib.tiv(name, change)
            cal = calibrate(lib, tivs)
            key["calibration"][name] = {"change": change, **cal}
            arms[name] = make_tiv_arm(lib, tivs, cal)
            print(f"calibración {name}: {cal}", flush=True)
        a_name, b_name = list(PILOT_CONFIGS)
        items, skipped = build_items(contexts(lib, "pilot", rng), args.n or N_PILOT, lib.pool, arms[a_name],
                                     arms[b_name], used, rng, only_disagreements=True)
        key["skipped"]["pilot"] = skipped
        render_items(lib, items, out, rng, a_name, b_name, "p", key["items"], page_items, render)
    else:
        change = json.loads(args.config)
        tivs = lib.tiv("congelada", change)
        cal = calibrate(lib, tivs)
        key["calibration"]["congelada"] = {"change": change, **cal}
        print(f"calibración: {cal}", flush=True)
        arm_tiv = make_tiv_arm(lib, tivs, cal)

        def arm_camelot(t, pool):
            return pick(lib.clap, t, *camelot_pass(lib.key, t, pool))

        def arm_none(t, pool):
            return pick(lib.clap, t, list(pool), {x: 0 for x in pool})
        for tag, ctrl, ctrl_name in (("c1", arm_camelot, "recomendador actual"), ("c2", arm_none, "sin armonía")):
            items, skipped = build_items(contexts(lib, tag, rng), args.n or N_PER_COMPARISON, lib.pool, arm_tiv,
                                         ctrl, used, rng, only_disagreements=False)
            key["skipped"][tag] = skipped
            print(f"{tag}: {len(items)} contextos, {sum(i['auto_tie'] for i in items)} empates automáticos, "
                  f"saltados {skipped}", flush=True)
            render_items(lib, items, out, rng, "tiv", ctrl_name, tag, key["items"], page_items, render)
        add_repeats(out, rng, key["items"], page_items, args.n_repeat, copy_audio=render)
    key["used"] = sorted(used)
    write_page(out, shuffle_spaced(page_items, key["items"], rng), test_id)
    (out / "key.json").write_text(json.dumps(key, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"listo: {len(page_items)} ítems a escuchar -> {out / 'index.html'}; clave en key.json")
    return 0


# --------------------------------------------------------------------------- puntaje

def outcome(item: dict, answer) -> str:
    """'a' gana el brazo a, 'b' el b, 'tie' empate (automático o del oyente)."""
    if item["auto_tie"]:
        return "tie"
    v = int(answer)
    if v == 0:
        return "tie"
    return "a" if v == item["clip_of_a"] else "b"


def summarize(outs: list, n_expected: int) -> dict:
    """U con empates a 0,5 y binomial exacta unilateral sobre los no empates (prueba pareada de signos)."""
    from scipy.stats import binomtest
    w, l_, t = outs.count("a"), outs.count("b"), outs.count("tie")
    n = len(outs)
    res = {"N": n, "N_esperado": n_expected, "victorias": w, "derrotas": l_, "empates": t,
           "U": round((w + 0.5 * t) / n, 3) if n else None,
           "p_unilateral": round(binomtest(w, w + l_, 0.5, alternative="greater").pvalue, 4) if w + l_ else None}
    res["aprueba"] = bool(n == n_expected and res["U"] is not None and res["U"] >= U_MIN
                          and res["p_unilateral"] is not None and res["p_unilateral"] < ALPHA)
    return res


def score(key: dict, a: dict) -> dict:
    items = key["items"]
    res = {"fase": key["phase"],
           "sin_responder": len([i for i, it in items.items() if not it["auto_tie"] and i not in a])}
    reps = [(i, it) for i, it in items.items() if it["repeat_of"]]
    agree = [outcome(it, a[i]) == outcome(items[it["repeat_of"]], a[it["repeat_of"]])
             for i, it in reps if i in a and it["repeat_of"] in a]
    res["repetidos"] = {"n": len(agree), "acuerdos": int(sum(agree)), "minimo": MIN_REPEAT_AGREEMENTS}
    main = {i: it for i, it in items.items() if not it["repeat_of"] and (it["auto_tie"] or i in a)}
    if key["phase"] == "pilot":
        outs = [outcome(it, a.get(i, 0)) for i, it in main.items()]
        s = summarize(outs, len([1 for it in items.values() if not it["repeat_of"]]))
        s.pop("aprueba")
        first = next(iter(items.values()))
        res["piloto"] = {**s, "brazo_a": first["arm_a"], "brazo_b": first["arm_b"]}
        # se congela la de mayor U; empate exacto -> el brazo a (FEM#10, la referencia)
        res["congelar"] = first["arm_a"] if s["U"] is not None and s["U"] >= 0.5 else first["arm_b"]
        return res
    for tag in ("c1", "c2"):
        outs = [outcome(it, a.get(i, 0)) for i, it in main.items() if it["comparison"] == tag]
        res[tag] = summarize(outs, N_PER_COMPARISON)
    if res["repetidos"]["acuerdos"] < MIN_REPEAT_AGREEMENTS:
        res["veredicto"] = "NO CONCLUYENTE: menos de 16 acuerdos en los 20 repetidos"
    elif not res["c1"]["aprueba"]:
        res["veredicto"] = "NO SE INTEGRA: no supera al recomendador actual (evidencia insuficiente, no equivalencia)"
    elif not res["c2"]["aprueba"]:
        res["veredicto"] = "SUPERA al recomendador actual, pero no se confirma contra 'sin armonía'"
    else:
        res["veredicto"] = "SE INTEGRA: supera a los dos controles con los umbrales fijados"
    return res


def cmd_score(args) -> int:
    key = json.loads((args.dir / "key.json").read_text(encoding="utf-8"))
    ans = json.loads(Path(args.answers).read_text(encoding="utf-8"))
    if ans.get("test_id") != key["test_id"]:
        print("ERROR: las respuestas son de otra prueba", file=sys.stderr)
        return 2
    res = score(key, ans["answers"])
    print(json.dumps(res, indent=2, ensure_ascii=False))
    (args.dir / f"score_{key['test_id']}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--phase", choices=["pilot", "confirm"], required=True)
    b.add_argument("--config", default="{}", help="(confirm) configuración congelada: cambios a FEM#10 en JSON")
    b.add_argument("--used-from", type=Path, help="(confirm) key.json del piloto: sus temas no se repiten")
    b.add_argument("--eval-dir", type=Path, default=Path(r"D:\TRAKTOR ML datos\harmonic_tiv"))
    b.add_argument("--cache", type=Path, default=Path(r"D:\TRAKTOR ML datos\harmonic_tiv\cache_semitones_v2\musica"))
    b.add_argument("--dataset-root", type=Path, default=Path(r"D:\TRAKTOR ML datos\musica"))
    b.add_argument("--n", type=int, help="contextos por comparación (por defecto 30 en el piloto, 60 en confirm)")
    b.add_argument("--n-repeat", type=int, default=N_REPEATS)
    b.add_argument("--seed", type=int, default=20261106)
    b.add_argument("--dry-run", action="store_true", help="elige y calibra sin renderizar audio")
    b.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--dir", type=Path, required=True)
    s.add_argument("--answers", type=Path, required=True)
    args = ap.parse_args()
    return cmd_build(args) if args.cmd == "build" else cmd_score(args)


if __name__ == "__main__":
    sys.exit(main())
