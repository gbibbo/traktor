"""
PURPOSE: Cómo cambia lo que encuentra CLAP según cómo se escribe el texto, medido sobre la biblioteca (1760 temas,
         clap_full). De acá salen las reglas y el vocabulario de las instrucciones del LLM que reescribe los conceptos
         (cloudflare/concept-ai-worker.js). Informe: docs/reports/clap_texto_2026-11-09.md.
         Referencias independientes (ninguna es la verdad; el juez de una lista es Gabriel escuchando):
           estilos    los 400 estilos Discogs de MAEST (representations/maesthf_full_styles): positivo = el 10 % de los
                      temas con más activación del estilo, solo estilos presentes en la biblioteca;
           genero     género confirmado de Beatport (índice de recomendación, genero_fuente == "beatport");
           vocal      etiqueta Vocal (clap_probe);
           instrumento  clases de AudioSet de AST (representations/ast_full, si existe): positivo = el 5 % superior.
         Pruebas: forma de la frase (F), sensibilidad a paráfrasis (P), promediar frases (E), negación (N), idioma (L),
         conceptos no sonoros (A), longitud (S), palabras de instrumento (I), señales para elegir entre frases (Q) y
         barrido de vocabulario (V). Una sola variable por comparación; intervalos por bootstrap sobre conceptos.
         Uso: python src/v4/evaluation/clap_texto.py --pruebas F,P,E,N,L,A,S,I,Q,V --low-priority
CHANGELOG:
  - 2026-11-09: Creación inicial.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)
from src.v4.evaluation.clap_texto_frases import (  # noqa: E402
    ACUSTICO, ARTISTAS, CAPTIONS, CONCEPTOS_PARAFRASIS, FORMAS, IDIOMA, INSTRUMENTOS, LONGITUD, NEGACION, NULOS,
    VOCABULARIO,
)

DATA = REPO_ROOT / "artifacts" / "v4" / "datasets" / "musica"
INDEX = REPO_ROOT / "artifacts" / "v4" / "traktor_live" / "indice_recomendacion.json"
OUT = DATA / "evaluation" / "clap_texto"
TOP = 30
STYLE_MIN_MEAN = 0.15                          # activación media del 10 % superior: el estilo está en la biblioteca
RNG_SEED = 20261109


# ---------------------------------------------------------------- datos

class Biblioteca:
    def __init__(self):
        index = json.loads(INDEX.read_text(encoding="utf-8"))
        self.temas = [t for t in index["temas"] if t.get("emb")]
        self.uids = [t["uid"] for t in self.temas]
        self.V = np.stack([np.frombuffer(base64.b64decode(t["emb"]), dtype="<f4") for t in self.temas]).astype(np.float64)
        self.genero = np.array([t["genero"] if t.get("genero_fuente") == "beatport" else "" for t in self.temas], dtype=object)
        self.artista = [t.get("artista") or "" for t in self.temas]
        self.bpm = np.array([t["bpm"] if t.get("bpm") else np.nan for t in self.temas], dtype=np.float64)
        vocal = pd.read_parquet(DATA / "features" / "vocals_clap_probe.parquet").set_index("track_uid")["is_vocal"]
        self.vocal = np.array([bool(vocal.get(u, False)) for u in self.uids])
        self.estilos, self.estilo_nombres = self._alinear("maesthf_full_styles", self._nombres_maest())
        self.ast, self.ast_nombres = None, []
        cache = DATA / "representations" / "ast_full" / "cache"
        if cache.exists() and any(cache.glob("*.npz")):
            self.ast, self.ast_nombres = self._ast_desde_cache(cache), self._nombres_ast()

    def _nombres_maest(self) -> List[str]:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained("mtg-upf/discogs-maest-30s-pw-129e", trust_remote_code=True, local_files_only=True)
        return [cfg.id2label[i] for i in range(len(cfg.id2label))]

    def _nombres_ast(self) -> List[str]:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", local_files_only=True)
        return [cfg.id2label[i] for i in range(len(cfg.id2label))]

    def _alinear(self, rep: str, nombres: List[str]) -> Tuple[np.ndarray, List[str]]:
        d = DATA / "representations" / rep
        mat = np.load(d / "embeddings.npy")
        row = {u: i for i, u in enumerate(json.loads((d / "track_uids.json").read_text(encoding="utf-8")))}
        filas = [row.get(u) for u in self.uids]
        if any(f is None for f in filas):                # cobertura parcial: el resto queda en NaN y no cuenta
            out = np.full((len(filas), mat.shape[1]), np.nan, dtype=np.float32)
            for i, f in enumerate(filas):
                if f is not None:
                    out[i] = mat[f]
            return out, nombres
        return mat[filas], nombres

    def _ast_desde_cache(self, cache: Path) -> np.ndarray:
        """AST por tema desde la caché (puede cubrir solo una parte: extract_representations --shard). Sin dato, NaN."""
        filas = {}
        for f in cache.glob("*.npz"):
            with np.load(f) as z:
                if "_probmean" in z.files:
                    filas[f.stem] = z["_probmean"]
        dim = len(next(iter(filas.values())))
        out = np.full((len(self.uids), dim), np.nan, dtype=np.float32)
        for i, u in enumerate(self.uids):
            if u in filas:
                out[i] = filas[u]
        return out

    def estilos_presentes(self) -> Dict[str, np.ndarray]:
        """{estilo: máscara del 10 % superior} para los estilos que de verdad están en la biblioteca."""
        out = {}
        for j, nombre in enumerate(self.estilo_nombres):
            a = self.estilos[:, j]
            umbral = np.nanquantile(a, 0.9)
            pos = a >= umbral
            if np.nanmean(a[pos]) >= STYLE_MIN_MEAN:
                out[nombre] = pos
        return out


class Texto:
    """CLAP de texto con caché: la misma frase no se codifica dos veces."""

    def __init__(self, threads: int = 4):
        import torch
        from src.v4.pipeline.extract_representations import ClapBackend
        torch.set_num_threads(threads)
        self.backend = ClapBackend(None)
        self.cache: Dict[str, np.ndarray] = {}

    def __call__(self, frases: Sequence[str]) -> np.ndarray:
        faltan = [f for f in dict.fromkeys(frases) if f not in self.cache]
        for i in range(0, len(faltan), 32):
            lote = faltan[i:i + 32]
            for f, v in zip(lote, self.backend.embed_text(lote)):
                self.cache[f] = np.asarray(v, dtype=np.float64)
        return np.stack([self.cache[f] for f in frases])

    def una(self, frase: str) -> np.ndarray:
        return self([frase])[0]

    def promedio(self, frases: Sequence[str]) -> np.ndarray:
        m = self(list(frases)).mean(axis=0)
        return m / np.linalg.norm(m)


# ---------------------------------------------------------------- métricas

def average_precision(scores: np.ndarray, pos: np.ndarray) -> float:
    orden = np.argsort(-scores, kind="stable")
    hits = pos[orden]
    if hits.sum() == 0:
        return float("nan")
    precision = np.cumsum(hits) / (np.arange(hits.size) + 1)
    return float((precision * hits).sum() / hits.sum())


def auc(scores: np.ndarray, pos: np.ndarray) -> float:
    rangos = pd.Series(scores).rank().to_numpy()
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((rangos[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def precision_top(scores: np.ndarray, pos: np.ndarray, k: int = TOP) -> float:
    return float(pos[np.argsort(-scores, kind="stable")[:k]].mean())


def top_idx(scores: np.ndarray, k: int = TOP) -> np.ndarray:
    return np.argsort(-scores, kind="stable")[:k]


def overlap(a: np.ndarray, b: np.ndarray, k: int = TOP) -> float:
    return len(set(top_idx(a, k)) & set(top_idx(b, k))) / k


def contraste(scores: np.ndarray, k: int = TOP) -> float:
    """Cuánto se separa la lista del resto: media de los k primeros, en desvíos de la biblioteca."""
    return float((np.sort(scores)[-k:].mean() - scores.mean()) / scores.std())


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def bootstrap_ic(valores: np.ndarray, n: int = 4000) -> Tuple[float, float, float]:
    """Media e intervalo del 95 % remuestreando conceptos."""
    v = valores[np.isfinite(valores)]
    rng = np.random.default_rng(RNG_SEED)
    medias = v[rng.integers(0, v.size, (n, v.size))].mean(axis=1)
    return float(v.mean()), float(np.quantile(medias, 0.025)), float(np.quantile(medias, 0.975))


def nombre_estilo(etiqueta: str) -> str:
    return etiqueta.split("---")[-1].lower()


# ---------------------------------------------------------------- pruebas

def prueba_formas(b: Biblioteca, t: Texto) -> dict:
    """F: la misma palabra en distintas formas de frase. Referencia: estilos de MAEST y géneros de Beatport."""
    estilos = b.estilos_presentes()
    filas, por_forma = [], {f: [] for f in FORMAS}
    por_forma["caption"], por_forma["genero+caption"] = [], []
    for etiqueta, pos in estilos.items():
        g = nombre_estilo(etiqueta)
        fila = {"estilo": g, "n_pos": int(pos.sum())}
        base = None
        for forma, plantilla in FORMAS.items():
            s = b.V @ t.una(plantilla.format(g))
            ap = average_precision(s, pos)
            fila[forma] = ap
            por_forma[forma].append(ap)
            base = s if forma == "bare" else base
            fila[f"auc_{forma}"] = auc(s, pos)
            fila[f"ov_{forma}"] = overlap(s, base) if forma != "bare" else 1.0
        if g in CAPTIONS:
            sc = b.V @ t.una(CAPTIONS[g])
            sg = b.V @ t.una(f"{g}, {CAPTIONS[g][0].lower()}{CAPTIONS[g][1:]}")
            fila["caption"], fila["genero+caption"] = average_precision(sc, pos), average_precision(sg, pos)
            por_forma["caption"].append(fila["caption"])
            por_forma["genero+caption"].append(fila["genero+caption"])
        filas.append(fila)
    df = pd.DataFrame(filas)
    resumen = {}
    bare = df["bare"].to_numpy()
    for forma in FORMAS:
        delta = df[forma].to_numpy() - bare
        m, lo, hi = bootstrap_ic(delta)
        resumen[forma] = {"ap_media": float(df[forma].mean()), "delta_vs_bare": m, "ic95": [lo, hi],
                          "auc_media": float(df[f"auc_{forma}"].mean()), "overlap_con_bare": float(df[f"ov_{forma}"].mean())}
    sub = df.dropna(subset=["caption"])
    for k in ("caption", "genero+caption"):
        delta = sub[k].to_numpy() - sub["bare"].to_numpy()
        m, lo, hi = bootstrap_ic(delta)
        resumen[k] = {"ap_media": float(sub[k].mean()), "delta_vs_bare": m, "ic95": [lo, hi], "n": int(len(sub)),
                      "bare_mismo_subconjunto": float(sub["bare"].mean())}
    # Beatport: otra referencia, misma pregunta
    generos = pd.Series(b.genero[b.genero != ""]).value_counts()
    generos = [g for g, n in generos.items() if n >= 40]
    conocidos = b.genero != ""
    beat = []
    for g in generos:
        limpio = g.split("(")[0].replace("/", " ").strip().lower()
        pos = b.genero[conocidos] == g
        fila = {"genero": g}
        for forma, plantilla in FORMAS.items():
            fila[forma] = average_precision((b.V @ t.una(plantilla.format(limpio)))[conocidos], pos)
        beat.append(fila)
    bdf = pd.DataFrame(beat).set_index("genero")
    return {"n_estilos": int(len(df)), "formas": resumen, "beatport_ap": bdf.mean().to_dict(),
            "beatport_detalle": bdf.round(3).to_dict(orient="index"), "estilos": df.round(4).to_dict(orient="records")}


def prueba_parafrasis(b: Biblioteca, t: Texto) -> dict:
    """P: frases que para una persona dicen lo mismo. ¿Cuánto coinciden sus listas?"""
    filas = []
    for concepto, frases in CONCEPTOS_PARAFRASIS.items():
        S = [b.V @ t.una(f) for f in frases]
        n = len(S)
        ov = [overlap(S[i], S[j]) for i in range(n) for j in range(i + 1, n)]
        rho = [spearman(S[i], S[j]) for i in range(n) for j in range(i + 1, n)]
        filas.append({"concepto": concepto, "n": n, "overlap_medio": float(np.mean(ov)), "overlap_min": float(np.min(ov)),
                      "overlap_max": float(np.max(ov)), "spearman_medio": float(np.mean(rho))})
    df = pd.DataFrame(filas)
    # Referencia: dos conceptos distintos
    conceptos = list(CONCEPTOS_PARAFRASIS)
    S0 = {c: b.V @ t.una(CONCEPTOS_PARAFRASIS[c][0]) for c in conceptos}
    otros = [overlap(S0[a], S0[c]) for i, a in enumerate(conceptos) for c in conceptos[i + 1:]]
    return {"overlap_parafrasis": float(df["overlap_medio"].mean()), "overlap_conceptos_distintos": float(np.mean(otros)),
            "detalle": df.round(3).to_dict(orient="records")}


def prueba_ensamble(b: Biblioteca, t: Texto) -> dict:
    """E: ¿promediar varias frases da una lista más estable y mejor que una sola?"""
    filas = []
    for concepto, frases in CONCEPTOS_PARAFRASIS.items():
        if len(frases) < 6:
            continue
        impares, pares = frases[::2], frases[1::2]
        s1 = [b.V @ t.una(f) for f in frases]
        ea, eb = b.V @ t.promedio(impares), b.V @ t.promedio(pares)
        sueltas = [overlap(s1[i], s1[j]) for i in range(len(s1)) for j in range(i + 1, len(s1))]
        filas.append({"concepto": concepto, "una_frase": float(np.mean(sueltas)), "tres_frases": overlap(ea, eb)})
    df = pd.DataFrame(filas)
    delta = df["tres_frases"].to_numpy() - df["una_frase"].to_numpy()
    m, lo, hi = bootstrap_ic(delta)
    # Con referencia: promedio de plantillas contra una sola forma
    estilos = b.estilos_presentes()
    ganan = []
    for etiqueta, pos in estilos.items():
        g = nombre_estilo(etiqueta)
        sola = average_precision(b.V @ t.una(FORMAS["music"].format(g)), pos)
        todas = average_precision(b.V @ t.promedio([p.format(g) for p in FORMAS.values()]), pos)
        ganan.append(todas - sola)
    m2, lo2, hi2 = bootstrap_ic(np.array(ganan))
    return {"estabilidad": {"una_frase": float(df["una_frase"].mean()), "tres_frases": float(df["tres_frases"].mean()),
                            "delta": m, "ic95": [lo, hi]},
            "ap_promedio_plantillas_vs_una": {"delta": m2, "ic95": [lo2, hi2], "n": len(ganan)},
            "detalle": df.round(3).to_dict(orient="records")}


def prueba_negacion(b: Biblioteca, t: Texto) -> dict:
    """N: ¿entiende CLAP «sin»? Referencia: etiqueta Vocal."""
    filas = []
    for c in NEGACION:
        s = {nombre: b.V @ t.una(f.format(c)) for nombre, f in
             {"solo": "{}", "con": "{} with vocals", "sin": "{} without vocals", "no": "{} no vocals",
              "instrumental": "instrumental {}"}.items()}
        fila = {"concepto": c}
        for k, v in s.items():
            fila[f"vocal30_{k}"] = float(b.vocal[top_idx(v)].mean())
        fila["texto_cos_con_sin"] = float(t.una(f"{c} with vocals") @ t.una(f"{c} without vocals"))
        fila["overlap_con_sin"] = overlap(s["con"], s["sin"])
        fila["overlap_solo_sin"] = overlap(s["solo"], s["sin"])
        filas.append(fila)
    df = pd.DataFrame(filas)
    return {"medias": df.drop(columns="concepto").mean().round(3).to_dict(), "base_vocal": float(b.vocal.mean()),
            "detalle": df.round(3).to_dict(orient="records")}


def prueba_idioma(b: Biblioteca, t: Texto) -> dict:
    """L: la misma idea en español y en inglés."""
    filas = []
    for es, en in IDIOMA:
        se, si = b.V @ t.una(es), b.V @ t.una(en)
        filas.append({"es": es, "en": en, "overlap": overlap(se, si), "spearman": spearman(se, si), "contraste_es": contraste(se),
                      "contraste_en": contraste(si), "texto_cos": float(t.una(es) @ t.una(en))})
    df = pd.DataFrame(filas)
    # Referencia: el nombre del estilo traducido no existe, así que se compara contra la frase en inglés
    return {"overlap_medio": float(df["overlap"].mean()), "spearman_medio": float(df["spearman"].mean()),
            "contraste_es": float(df["contraste_es"].mean()), "contraste_en": float(df["contraste_en"].mean()),
            "detalle": df.round(3).to_dict(orient="records")}


def prueba_no_sonoro(b: Biblioteca, t: Texto) -> dict:
    """A: ocasiones y abstracciones frente a su descripción sonora, y frases sin sentido como línea de base."""
    def lista(frase: str) -> dict:
        s = b.V @ t.una(frase)
        i = top_idx(s)
        gen = pd.Series([g or "?" for g in b.genero[i]]).value_counts().head(3).to_dict()
        return {"contraste": contraste(s), "top1": float(s.max()), "mediana_biblioteca": float(np.median(s)), "generos": gen,
                "ejemplos": [f"{b.artista[k][:24]} - {b.temas[k]['titulo'][:32]}" for k in i[:4]]}
    filas = []
    for ocasion, descripcion in ACUSTICO:
        filas.append({"ocasion": ocasion, "descripcion": descripcion, "ocasion_datos": lista(ocasion), "descripcion_datos": lista(descripcion),
                      "overlap": overlap(b.V @ t.una(ocasion), b.V @ t.una(descripcion))})
    nulos = {f: contraste(b.V @ t.una(f)) for f in NULOS}
    return {"contraste_nulos": nulos, "contraste_nulos_medio": float(np.mean(list(nulos.values()))), "detalle": filas}


def prueba_longitud(b: Biblioteca, t: Texto) -> dict:
    """S: la misma idea, cada vez más larga."""
    filas = []
    for concepto, frases in LONGITUD.items():
        ultima = b.V @ t.una(frases[-1])
        s = [b.V @ t.una(f) for f in frases]
        for f, x in zip(frases, s):
            filas.append({"concepto": concepto, "palabras": len(f.split()), "frase": f, "contraste": contraste(x),
                          "overlap_con_larga": overlap(x, ultima), "overlap_con_corta": overlap(x, s[0])})
    return {"detalle": pd.DataFrame(filas).round(3).to_dict(orient="records")}


def prueba_instrumentos(b: Biblioteca, t: Texto) -> dict:
    """I: palabras de instrumento contra las clases de AudioSet de AST."""
    if b.ast is None:
        return {"estado": "sin AST: correr extract_representations.py --models ast"}
    cubiertos = np.isfinite(b.ast[:, 0])
    filas = []
    for clase, frases in INSTRUMENTOS.items():
        if clase not in b.ast_nombres:
            continue
        a = b.ast[cubiertos, b.ast_nombres.index(clase)]
        umbral = np.quantile(a, 0.95)
        pos = a >= umbral
        for frase in frases:
            s = (b.V @ t.una(frase))[cubiertos]
            filas.append({"clase_audioset": clase, "frase": frase, "auc": auc(s, pos), "ap": average_precision(s, pos),
                          "base": float(pos.mean()), "p30": precision_top(s, pos), "spearman": spearman(s, a),
                          "ast_umbral": float(umbral)})
    return {"n_temas": int(cubiertos.sum()), "detalle": pd.DataFrame(filas).round(4).to_dict(orient="records")}


def prueba_artistas(b: Biblioteca, t: Texto) -> dict:
    """¿Sirve nombrar a un artista? Referencia: sus temas en la biblioteca."""
    filas = []
    for artista in ARTISTAS:
        pos = np.array([artista.lower() in a.lower() for a in b.artista])
        if pos.sum() < 4:
            continue
        s = b.V @ t.una(artista)
        s2 = b.V @ t.una(f"a track by {artista}")
        filas.append({"artista": artista, "n": int(pos.sum()), "auc": auc(s, pos), "auc_track_by": auc(s2, pos),
                      "en_top30": int(pos[top_idx(s)].sum()), "base30": float(pos.mean() * TOP)})
    return {"detalle": filas, "auc_medio": float(np.mean([f["auc"] for f in filas])) if filas else None}


def prueba_senales(b: Biblioteca, t: Texto) -> dict:
    """Q: dadas varias frases para un mismo estilo, ¿qué número señala a la mejor? Spearman dentro de cada estilo."""
    estilos = b.estilos_presentes()
    senales = {"contraste": [], "coherencia": [], "acuerdo": [], "top1": []}
    for etiqueta, pos in estilos.items():
        g = nombre_estilo(etiqueta)
        S = [b.V @ t.una(p.format(g)) for p in FORMAS.values()]
        if g in CAPTIONS:
            S.append(b.V @ t.una(CAPTIONS[g]))
        ap = np.array([average_precision(s, pos) for s in S])
        if np.std(ap) < 1e-6:
            continue
        cont = np.array([contraste(s) for s in S])
        coh = []
        for s in S:
            A = b.V[top_idx(s)]
            coh.append(float((A @ A.T)[np.triu_indices(TOP, 1)].mean()))
        ac = np.array([np.mean([overlap(s, o) for o in S if o is not s]) for s in S])
        top1 = np.array([float(np.sort(s)[-1]) for s in S])
        for nombre, valores in (("contraste", cont), ("coherencia", np.array(coh)), ("acuerdo", ac), ("top1", top1)):
            senales[nombre].append(spearman(valores, ap))
    out = {}
    for k, v in senales.items():
        m, lo, hi = bootstrap_ic(np.array(v))
        out[k] = {"spearman_con_ap": m, "ic95": [lo, hi], "n": len(v)}
    return out


def prueba_vocabulario(b: Biblioteca, t: Texto) -> dict:
    """V: qué palabras mueven la lista. Estable = la misma palabra en tres formas da casi la misma lista."""
    formas = ("{}", "{} music", "the sound of {}")
    filas = []
    media_lib = b.V.mean(axis=0)
    media_lib = media_lib / np.linalg.norm(media_lib)
    for grupo, palabras in VOCABULARIO.items():
        for p in palabras:
            S = [b.V @ t.una(f.format(p)) for f in formas]
            ov = np.mean([overlap(S[i], S[j]) for i in range(3) for j in range(i + 1, 3)])
            m = t.promedio([f.format(p) for f in formas])
            filas.append({"grupo": grupo, "palabra": p, "estable": float(ov), "contraste": contraste(b.V @ m),
                          "lejos_de_la_media": float(1 - m @ media_lib)})
    df = pd.DataFrame(filas)
    # lista de cada palabra: cuánto se parece a la de las demás (una palabra útil tiene su propia lista)
    tops = {r.palabra: set(top_idx(b.V @ t.promedio([f.format(r.palabra) for f in formas]))) for r in df.itertuples()}
    palabras = list(tops)
    propio = []
    for p in palabras:
        otros = [len(tops[p] & tops[o]) / TOP for o in palabras if o != p]
        propio.append({"palabra": p, "solapamiento_medio": float(np.mean(otros)), "maximo": float(np.max(otros))})
    df = df.merge(pd.DataFrame(propio), on="palabra")
    return {"resumen_por_grupo": df.groupby("grupo")[["estable", "contraste", "solapamiento_medio"]].mean().round(3).to_dict(orient="index"),
            "detalle": df.round(3).to_dict(orient="records")}


PRUEBAS: Dict[str, Tuple[str, Callable]] = {
    "F": ("formas", prueba_formas), "P": ("parafrasis", prueba_parafrasis), "E": ("ensamble", prueba_ensamble),
    "N": ("negacion", prueba_negacion), "L": ("idioma", prueba_idioma), "A": ("no_sonoro", prueba_no_sonoro),
    "S": ("longitud", prueba_longitud), "I": ("instrumentos", prueba_instrumentos), "R": ("artistas", prueba_artistas),
    "Q": ("senales", prueba_senales), "V": ("vocabulario", prueba_vocabulario),
}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Cómo depende de la frase lo que encuentra CLAP en la biblioteca")
    ap.add_argument("--pruebas", default=",".join(PRUEBAS))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--low-priority", action="store_true")
    args = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if args.low_priority:
        from src.v4.pipeline.extract_representations import lower_priority
        lower_priority()
    b = Biblioteca()
    t = Texto(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] {len(b.uids)} temas; {len(b.estilos_presentes())} estilos presentes; AST: {'sí' if b.ast is not None else 'no'}")
    for clave in args.pruebas.split(","):
        nombre, funcion = PRUEBAS[clave]
        res = funcion(b, t)
        (OUT / f"{nombre}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        print(f"[OK] {clave} {nombre} -> {OUT / (nombre + '.json')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
