"""
PURPOSE: Etapa C del protocolo (docs/plans/armonia_tiv_protocolo.md): prueba de escucha a ciegas que
         decide si el TIV entra al recomendador.
           build  Elige ítems del split test donde la regla Camelot (K0: harmonic.py con la tonalidad MIK,
                  sin transponer) y el finalista TIV eligen candidatos distintos entre los mismos
                  candidatos (los `pool_k` más parecidos por CLAP con BPM compatible; empates por CLAP).
                  Renderiza las dos transiciones (src/v4/harmonic_tiv/render.py), las nombra con tokens
                  al azar, agrega ítems repetidos con el orden invertido (consistencia de Gabriel) y
                  escribe una página local (index.html) que guarda las respuestas y las exporta en JSON.
                  La clave (qué mezcla es cuál) queda en key.json, fuera de la página.
           score  Lee las respuestas y la clave: victorias del TIV sin empates, binomial exacta
                  unilateral, por estrato, empates, y acuerdo de Gabriel consigo mismo en los repetidos.
         Estratos: S1 = Camelot empata entre las dos elecciones (el TIV desempata); S2 = el TIV elige un
         tema que Camelot puntúa más bajo.
         Uso:
           python tools/harmonic_tiv/listening_test.py build --config '{"section":"center30"}' --out <dir>
           python tools/harmonic_tiv/listening_test.py score --dir <dir> --answers <respuestas.json>
CHANGELOG:
  - 2026-11-06: Creación (rama harmonic-tiv).
"""
from __future__ import annotations

import argparse
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
from src.v4.harmonic_tiv import evaluate as ev  # noqa: E402
from src.v4.harmonic_tiv import features as ft  # noqa: E402
from src.v4.harmonic_tiv import render as rd  # noqa: E402
from src.v4.harmonic_tiv import tiv as tv  # noqa: E402


def _artist_key(a: str) -> str:
    a = (a or "").lower()
    a = re.split(r",|&| feat\.? | ft\.? | x | vs\.? | and ", a)[0]
    return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKD", a))


def load_clap(dataset_root: Path):
    d = dataset_root / "representations" / "clap_full"
    emb = np.load(d / "embeddings.npy").astype(np.float64)
    uids = json.loads((d / "track_uids.json").read_text(encoding="utf-8"))
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    return {u: emb[i] for i, u in enumerate(uids)}


def select_items(args, keys: pd.DataFrame, tivs: dict, clap: dict, metric: str, rng: np.random.Generator):
    k = keys.set_index("track_uid")
    ids = [u for u in k.index if u in tivs and u in clap]
    bpm = {u: (k.at[u, "traktor_bpm"] if pd.notna(k.at[u, "traktor_bpm"]) else k.at[u, "bpm"]) for u in ids}
    artist = {u: _artist_key(str(k.at[u, "artist"])) for u in ids}
    fn = tv.METRICS[metric]
    quotas = dict(q.split(":") for q in args.strata.split(","))
    quotas = {s: int(n) for s, n in quotas.items()}
    got = {s: [] for s in quotas}
    targets = [u for u in ids if k.at[u, "split"] == "test" and k.at[u, "k_tag"] != "?"]
    rng.shuffle(targets)
    used = set()
    for t in targets:
        if all(len(got[s]) >= quotas[s] for s in quotas):
            break
        pool = [x for x in ids if x != t and artist[x] != artist[t] and ev.bpm_compatible(bpm[t], bpm[x])]
        if len(pool) < args.pool_k:
            continue
        sims = np.array([clap[t] @ clap[x] for x in pool])
        top = [pool[i] for i in np.argsort(-sims)[: args.pool_k]]
        sim_of = {x: float(clap[t] @ clap[x]) for x in top}
        cam = {x: best_key_shift(k.at[t, "k_tag"], k.at[x, "k_tag"], max_shift=0)[0] for x in top}
        tsc = {x: 1.0 - float(fn(tivs[t], tivs[x])) for x in top}
        pick_k = max(top, key=lambda x: (cam[x], sim_of[x]))
        pick_t = max(top, key=lambda x: (tsc[x], sim_of[x]))
        if pick_k == pick_t or pick_k in used or pick_t in used:
            continue
        stratum = "S1" if abs(cam[pick_t] - cam[pick_k]) < 1e-9 else "S2"
        if len(got[stratum]) >= quotas[stratum]:
            continue
        got[stratum].append({"target": t, "k_pick": pick_k, "t_pick": pick_t, "stratum": stratum,
                             "camelot_k": cam[pick_k], "camelot_t": cam[pick_t],
                             "tiv_k": round(tsc[pick_k], 4), "tiv_t": round(tsc[pick_t], 4),
                             "clap_k": round(sim_of[pick_k], 4), "clap_t": round(sim_of[pick_t], 4),
                             "key_target": k.at[t, "k_tag"], "key_k": k.at[pick_k, "k_tag"], "key_t": k.at[pick_t, "k_tag"]})
        used.update([t, pick_k, pick_t])
    return [it for s in got for it in got[s]], {s: len(v) for s, v in got.items()}


def track_info(u: str, k: pd.DataFrame, cache_dir: Path) -> rd.TrackInfo:
    row = k.loc[u]
    bpm = row["traktor_bpm"] if pd.notna(row["traktor_bpm"]) else row["bpm"]
    return rd.TrackInfo(uid=u, path=row["source_path"], bpm=float(bpm), cache=ft.load_cache(cache_dir / f"{u}.npz"),
                        grid_start_s=float(row["grid_start_s"]) if pd.notna(row["grid_start_s"]) else None,
                        grid_bpm=float(row["grid_bpm"]) if pd.notna(row["grid_bpm"]) else None)


PAGE = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Escucha armónica</title>
<style>
:root{{--bg:#fafafa;--fg:#1b1b1b;--mut:#666;--card:#fff;--line:#ddd;--acc:#2a5bd7;--sel:#e7eefc}}
@media (prefers-color-scheme: dark){{:root{{--bg:#151515;--fg:#eee;--mut:#aaa;--card:#1f1f1f;--line:#333;--acc:#7da2ff;--sel:#24314f}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.45 system-ui,sans-serif}}
main{{max-width:760px;margin:0 auto;padding:16px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:14px 0}}
audio{{width:100%;margin:4px 0 10px}} .row{{display:flex;gap:8px;flex-wrap:wrap}}
button{{font:inherit;padding:8px 12px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}}
button.on{{background:var(--sel);border-color:var(--acc)}} .mut{{color:var(--mut);font-size:14px}}
</style></head><body><main>
<h1>Escucha armónica</h1>
<p>En cada ítem, el tema que suena primero es el mismo en las dos mezclas; cambia el tema que entra.
Elegí la mezcla que suena <b>más armónica</b> (menos choque de notas), o «iguales» si no notás diferencia.
Las respuestas se guardan en este navegador; al terminar, «Exportar respuestas».</p>
<p class="mut" id="prog"></p>
<div id="items"></div>
<p><button id="exp">Exportar respuestas</button></p>
</main><script>
const ITEMS = {items};
const KEY = "escucha_armonica_{test_id}";
let ans = {{}};
try {{ ans = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{ ans = {{}}; }}
function save() {{ try {{ localStorage.setItem(KEY, JSON.stringify(ans)); }} catch (e) {{}} prog(); }}
function prog() {{ document.getElementById("prog").textContent = Object.keys(ans).length + " de " + ITEMS.length + " respondidos"; }}
const box = document.getElementById("items");
ITEMS.forEach((it, i) => {{
  const d = document.createElement("div"); d.className = "card";
  d.innerHTML = `<b>Ítem ${{i + 1}}</b><div class="mut">Mezcla 1</div><audio controls preload="none" src="${{it.c1}}"></audio>
    <div class="mut">Mezcla 2</div><audio controls preload="none" src="${{it.c2}}"></audio>
    <div class="row"><button data-v="1">1 más armónica</button><button data-v="2">2 más armónica</button><button data-v="0">Iguales</button></div>`;
  d.querySelectorAll("button").forEach(b => {{
    if (ans[it.id] === b.dataset.v) b.classList.add("on");
    b.onclick = () => {{ ans[it.id] = b.dataset.v; d.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b)); save(); }};
  }});
  box.appendChild(d);
}});
document.getElementById("exp").onclick = () => {{
  const blob = new Blob([JSON.stringify({{test_id: "{test_id}", answers: ans}}, null, 1)], {{type: "application/json"}});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "respuestas_{test_id}.json"; a.click();
}};
prog();
</script></body></html>
"""


def cmd_build(args) -> int:
    rng = np.random.default_rng(args.seed)
    keys = pd.read_csv(args.eval_dir / "keys_musica.csv")
    bpm = pd.read_parquet(args.dataset_root / "features" / "bpm_key.parquet")[["track_uid", "bpm"]]
    keys = keys.merge(bpm, on="track_uid", how="left")
    change = json.loads(args.config)
    metric = change.pop("metric", "tivlib")
    cfg = ft.THESIS_FEM10.with_(**change)
    tivs = ft.collection_tivs(args.cache, keys["track_uid"], cfg)
    clap = load_clap(args.dataset_root)
    items, counts = select_items(args, keys, tivs, clap, metric, rng)
    print(f"ítems por estrato: {counts}", flush=True)
    out = args.out
    (out / "audio").mkdir(parents=True, exist_ok=True)
    k = keys.set_index("track_uid")
    test_id = secrets.token_hex(4)
    key, page_items = {"test_id": test_id, "config": cfg.as_dict(), "metric": metric, "items": {}}, []
    for n, it in enumerate(items, 1):
        tok_a, tok_b = secrets.token_hex(4), secrets.token_hex(4)
        fa, fb = out / "audio" / f"{tok_a}.mp3", out / "audio" / f"{tok_b}.mp3"
        params = rd.render_pair(track_info(it["target"], k, args.cache), track_info(it["k_pick"], k, args.cache),
                                track_info(it["t_pick"], k, args.cache), fa, fb)
        tiv_first = bool(rng.integers(0, 2))
        c1, c2 = (fb, fa) if tiv_first else (fa, fb)
        item_id = f"i{n:03d}"
        key["items"][item_id] = {**it, **params, "tiv_clip": 1 if tiv_first else 2, "repeat_of": None}
        page_items.append({"id": item_id, "c1": f"audio/{c1.name}", "c2": f"audio/{c2.name}"})
        print(f"  {n}/{len(items)} {it['stratum']} {params}", flush=True)
    # repetidos: mismo audio con otros nombres y el orden invertido
    reps = rng.choice(len(page_items), size=min(args.n_repeat, len(page_items)), replace=False)
    for r, idx in enumerate(reps, 1):
        orig = page_items[idx]
        new_c = []
        for c in (orig["c2"], orig["c1"]):
            dst = out / "audio" / f"{secrets.token_hex(4)}.mp3"
            shutil.copy2(out / c, dst)
            new_c.append(f"audio/{dst.name}")
        rid = f"r{r:03d}"
        ok = key["items"][orig["id"]]
        key["items"][rid] = {**ok, "tiv_clip": 3 - ok["tiv_clip"], "repeat_of": orig["id"]}
        page_items.append({"id": rid, "c1": new_c[0], "c2": new_c[1]})
    # orden al azar; los repetidos no pueden quedar a menos de 5 ítems de su original
    rep_of = {pi["id"]: key["items"][pi["id"]]["repeat_of"] for pi in page_items}
    for _ in range(1000):
        page = [page_items[i] for i in rng.permutation(len(page_items))]
        where = {pi["id"]: n for n, pi in enumerate(page)}
        if all(abs(where[i] - where[o]) >= 5 for i, o in rep_of.items() if o):
            break
    (out / "index.html").write_text(PAGE.format(items=json.dumps(page), test_id=test_id), encoding="utf-8")
    (out / "key.json").write_text(json.dumps(key, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"listo: {len(page)} ítems ({len(items)} + {len(reps)} repetidos) -> {out / 'index.html'}")
    return 0


def cmd_score(args) -> int:
    from scipy.stats import binomtest
    key = json.loads((args.dir / "key.json").read_text(encoding="utf-8"))
    ans = json.loads(Path(args.answers).read_text(encoding="utf-8"))
    if ans.get("test_id") != key["test_id"]:
        print("ERROR: las respuestas son de otra prueba", file=sys.stderr)
        return 2
    a = ans["answers"]
    rows = []
    for iid, it in key["items"].items():
        if iid not in a:
            continue
        v = int(a[iid])
        outcome = "tie" if v == 0 else ("tiv" if v == it["tiv_clip"] else "camelot")
        rows.append({"id": iid, "stratum": it["stratum"], "repeat_of": it["repeat_of"], "outcome": outcome})
    df = pd.DataFrame(rows)
    main_rows = df[df.repeat_of.isna()]
    res = {"answered": len(df), "items_main": len(main_rows)}

    def summary(d):
        w, l_, t = (d.outcome == "tiv").sum(), (d.outcome == "camelot").sum(), (d.outcome == "tie").sum()
        out = {"tiv_wins": int(w), "camelot_wins": int(l_), "ties": int(t)}
        if w + l_:
            bt = binomtest(int(w), int(w + l_), 0.5, alternative="greater")
            out.update({"tiv_share_no_ties": round(w / (w + l_), 3), "p_one_sided": round(bt.pvalue, 4),
                        "ci95_low": round(bt.proportion_ci(0.95).low, 3)})
        return out
    res["all"] = summary(main_rows)
    res["by_stratum"] = {s: summary(d) for s, d in main_rows.groupby("stratum")}
    reps = df[df.repeat_of.notna()]
    agree = [o == df.set_index("id").at[r, "outcome"] for r, o in zip(reps.repeat_of, reps.outcome)
             if r in set(df.id)]
    res["repeat_consistency"] = {"n": len(agree), "agree": round(float(np.mean(agree)), 3) if agree else None}
    s = res["all"]
    ok_consistency = res["repeat_consistency"]["agree"] is not None and res["repeat_consistency"]["agree"] >= 0.70
    if not ok_consistency:
        res["verdict"] = "NO CONCLUYENTE (consistencia < 70 % o sin repetidos)"
    elif s.get("tiv_share_no_ties", 0) >= 0.60 and s.get("p_one_sided", 1) < 0.05:
        res["verdict"] = "ÉXITO: el TIV gana según el criterio del protocolo"
    else:
        res["verdict"] = "NO SUPERA: el criterio del protocolo no se cumple"
    print(json.dumps(res, indent=2, ensure_ascii=False))
    (args.dir / f"score_{key['test_id']}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--config", default="{}", help="cambios a FEM#10 en JSON (campos de TivConfig y 'metric')")
    b.add_argument("--eval-dir", type=Path, default=Path(r"D:\TRAKTOR ML datos\harmonic_tiv"))
    b.add_argument("--cache", type=Path, default=Path(r"D:\TRAKTOR ML datos\harmonic_tiv\cache_semitones_v2\musica"))
    b.add_argument("--dataset-root", type=Path, default=Path(r"D:\TRAKTOR ML datos\musica"))
    b.add_argument("--strata", default="S1:30,S2:30")
    b.add_argument("--n-repeat", type=int, default=10)
    b.add_argument("--pool-k", type=int, default=30)
    b.add_argument("--seed", type=int, default=20261106)
    b.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--dir", type=Path, required=True)
    s.add_argument("--answers", type=Path, required=True)
    args = ap.parse_args()
    return cmd_build(args) if args.cmd == "build" else cmd_score(args)


if __name__ == "__main__":
    sys.exit(main())
