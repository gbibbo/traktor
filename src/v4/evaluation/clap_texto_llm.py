"""
PURPOSE: Banco del LLM que reescribe los conceptos: manda pedidos escritos como los escribiría un DJ apurado por el circuito
         real (Worker con su LLM, agente con CLAP sobre la biblioteca, sin escribir playlists) y compara qué encuentra el
         texto tal como lo escribió la persona contra lo que encuentra el texto curado. Referencias: estilos de MAEST y
         etiqueta Vocal (como en clap_texto.py). Si la referencia no aparece, solo se anota qué hizo el LLM. Sirve para
         cambiar las instrucciones del LLM sabiendo si mejoraron; no dice si la lista suena bien: eso lo decide Gabriel.
         Necesita un Worker corriendo:
           python src/v4/evaluation/clap_texto_llm.py --url http://127.0.0.1:8787 --salida banco.json
         con TRAKTOR_APP_PASSWORD y TRAKTOR_CONCEPT_DEVICE_TOKEN en el entorno. Gasta neuronas de Workers AI.
CHANGELOG:
  - 2026-11-09: Creación inicial.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.evaluation.clap_texto import Biblioteca, Texto, average_precision, bootstrap_ic, nombre_estilo, overlap, precision_top  # noqa: E402
from tools.traktor_live import conceptos_agente as ca  # noqa: E402

# Pedido -> referencia: un estilo de MAEST, "vocal": True/False, o nada (solo se mira qué hace el LLM).
PEDIDOS = [
    ("Quiero música de cumpleaños con cornetas", {}),
    ("algo oscuro tipo after sin voces", {"estilo": "deep techno", "vocal": False}),
    ("poneme un acid bien duro", {"estilo": "acid"}),
    ("funky disco con guitarrita", {"estilo": "disco"}),
    ("musica tribal con tambores", {"estilo": "tribal house"}),
    ("techno a lo amelie lens", {"estilo": "hard techno"}),
    ("algo para el atardecer en la playa", {"estilo": "downtempo"}),
    ("temazos con voz de mujer", {"vocal": True}),
    ("piano house alegre de los 90", {"estilo": "house"}),
    ("trance bien arriba", {"estilo": "progressive trance"}),
    ("minimal raro con clicks", {"estilo": "minimal"}),
    ("ambient sin bombo", {"estilo": "ambient"}),
    ("electro robotico", {"estilo": "electro"}),
    ("dub techno con eco", {"estilo": "dub techno"}),
    ("musica para el gym reventada", {"estilo": "hard techno"}),
    ("deep house suave pa cenar", {"estilo": "deep house"}),
    ("psicodelico rapido", {"estilo": "psy-trance"}),
    ("nu disco bailable", {"estilo": "nu-disco"}),
]
PROHIBIDO = re.compile(r"\b(this is|sound of|in the style of|tracks?|songs?|music by)\b", re.I)


def pedir(url: str, ruta: str, cuerpo: Optional[dict], cabecera: Dict[str, str], timeout: float = 120.0):
    data = None if cuerpo is None else json.dumps(cuerpo).encode("utf-8")
    req = urllib.request.Request(url + ruta, data=data, method="POST" if cuerpo is not None else "GET",
                                 headers={"Content-Type": "application/json", "User-Agent": "banco", **cabecera})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def medir(b: Biblioteca, t: Texto, pedido: str, textos: List[str], ref: dict, estilos: dict) -> dict:
    crudo = b.V @ t.una(pedido)
    curado = b.V @ t.promedio(textos)
    fila = {"overlap_crudo_curado": overlap(crudo, curado)}
    if "estilo" in ref:
        pos = next((m for e, m in estilos.items() if nombre_estilo(e) == ref["estilo"]), None)
        if pos is not None:
            fila.update(ap_crudo=average_precision(crudo, pos), ap_curado=average_precision(curado, pos),
                        p30_crudo=precision_top(crudo, pos), p30_curado=precision_top(curado, pos))
    if "vocal" in ref:
        pos = b.vocal if ref["vocal"] else ~b.vocal
        fila.update(p30_vocal_crudo=precision_top(crudo, pos), p30_vocal_curado=precision_top(curado, pos), vocal_ref=bool(ref["vocal"]))
    return fila


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="El LLM que reescribe conceptos, por el circuito real")
    ap.add_argument("--url", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--n", type=int, default=len(PEDIDOS))
    args = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    app = {"X-App-Password": os.environ["TRAKTOR_APP_PASSWORD"]}
    token = os.environ["TRAKTOR_CONCEPT_DEVICE_TOKEN"]
    b = Biblioteca()
    t = Texto(4)
    estilos = b.estilos_presentes()
    biblioteca = ca.cargar_biblioteca()
    agente = ca.Agente(ca.WorkerClient(args.url, token), lambda lvl, txt: None, cargar=lambda: biblioteca, codificar=t.backend.embed_text,
                       nml=lambda bib, chosen, nombre: "Concepto (banco)", results_log=None)
    filas = []
    for pedido, ref in PEDIDOS[: args.n]:
        t0 = time.perf_counter()
        status, creado = pedir(args.url, "/api/conceptos", {"texto": pedido}, app)
        t1 = time.perf_counter()
        if status != 201:
            filas.append({"pedido": pedido, "error": f"HTTP {status}: {creado}"})
            print(f"[ERROR] {pedido}: {status} {creado}", flush=True)
            continue
        ca.bucle(agente, threading.Event(), job_id=creado["id"], una_vez=True, poll_s=0.2, poll_activo_s=0.2, activo_s=4.0)
        t2 = time.perf_counter()
        _, final = pedir(args.url, f"/api/jobs/{creado['id']}", None, app)
        concepto = final["concept"]
        textos = [x["texto"] for x in concepto["textos"]]
        todos = [x["texto"] for x in creado["concept"]["textos"]]
        fila = {"pedido": pedido, "ref": ref, "estado": final["status"], "entendi": concepto.get("entendi"), "no_se_oye": concepto.get("no_se_oye"),
                "nombre": concepto.get("nombre"), "textos_iniciales": todos, "textos_finales": textos, "rondas": concepto.get("rondas"),
                "ronda_final": concepto.get("ronda"), "problema": concepto.get("problema"), "resumen": concepto.get("resumen"),
                "s_llm_inicial": round(t1 - t0, 1), "s_resto": round(t2 - t1, 1), **medir(b, t, pedido, textos, ref, estilos),
                "palabras": [len(x.split()) for x in textos], "prohibidas": [x for x in textos if PROHIBIDO.search(x)],
                "no_ascii": [x for x in textos if re.search(r"[^\x20-\x7e]", x)]}
        filas.append(fila)
        print(f"[OK] {pedido} -> {concepto.get('entendi')} | ronda {concepto.get('ronda')} | {fila['s_llm_inicial']}s + {fila['s_resto']}s", flush=True)
    resumen = {}
    con_estilo = [f for f in filas if "ap_curado" in f]
    if con_estilo:
        delta = np.array([f["ap_curado"] - f["ap_crudo"] for f in con_estilo])
        m, lo, hi = bootstrap_ic(delta)
        resumen["estilo"] = {"n": len(con_estilo), "ap_crudo": float(np.mean([f["ap_crudo"] for f in con_estilo])),
                             "ap_curado": float(np.mean([f["ap_curado"] for f in con_estilo])), "delta": m, "ic95": [lo, hi],
                             "mejoran": int((delta > 0).sum())}
    con_vocal = [f for f in filas if "p30_vocal_curado" in f]
    if con_vocal:
        resumen["vocal"] = {"n": len(con_vocal), "crudo": float(np.mean([f["p30_vocal_crudo"] for f in con_vocal])),
                            "curado": float(np.mean([f["p30_vocal_curado"] for f in con_vocal]))}
    Path(args.salida).write_text(json.dumps({"resumen": resumen, "filas": filas}, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(resumen, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
