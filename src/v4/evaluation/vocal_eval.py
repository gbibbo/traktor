"""
PURPOSE: Evaluación de detectores de voz contra datos públicos etiquetados (protocolo en
         docs/plans/20260929_deteccion_voz.md). Primer banco: Electrobyte (Romero-Arenas et al. 2022,
         CC BY 4.0, Zenodo 6757945): 90 temas de música electrónica con voz / sin voz marcada por
         tramos, partición 60/15/15 (train/valid/test).
         - Cada detector da un puntaje de voz por ventana a lo largo del tema entero; se proyecta a
           una grilla de 1 s (media de las ventanas que cubren el centro de cada segundo).
         - Etiqueta de un segundo = voz si más de la mitad del segundo está marcada "sing".
         - Métricas por segundo: AUC (sin umbral), y con el umbral elegido en valid (máxima exactitud
           balanceada): exactitud, exactitud balanceada, precisión, recall y F1; intervalo del 95 %
           por bootstrap sobre temas.
         Detectores: 'clap' (zero-shot, el detector actual de tag_vocals, sobre todo el tema), 'ast'
         (AudioSet: máximo de las clases de voz), 'hdemucs' (energía del stem de voz de
         torchaudio HDEMUCS_HIGH_MUSDB_PLUS relativa a la mezcla). Los puntajes se guardan en
         artifacts/v4/vocal_eval/<detector>/ para no recalcular.
CHANGELOG:
  - 2026-09-29: Creación inicial.
  - 2026-09-29: Detector 'clap_probe' (lineal sobre CLAP, entrenado con Electrobyte train); el último
                pedazo de HDemucs se rellena con silencio (el modelo exige largo fijo).
  - 2026-10-05: --save-probe guarda clap_probe como coeficientes para tag_vocals.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

ELECTROBYTE = REPO_ROOT / "data" / "public" / "electrobyte" / "Electrobyte"
CACHE = REPO_ROOT / "artifacts" / "v4" / "vocal_eval"
AST_VOICE = ("Speech", "Narration, monologue", "Singing", "Choir", "Male singing", "Female singing",
             "Child singing", "Synthetic singing", "Rapping", "Humming", "Vocal music", "A capella")


# ---------------------------------------------------------------------------
# Datos y etiquetas (puro)
# ---------------------------------------------------------------------------

def read_lab(path: Path) -> List[Tuple[float, float, str]]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3:
            out.append((float(parts[0]), float(parts[1]), parts[2]))
    return out


def frame_labels(segments: List[Tuple[float, float, str]], n_frames: int, hop: float = 1.0) -> np.ndarray:
    """1 si más de la mitad del segundo i está marcada como voz ('sing')."""
    cover = np.zeros(n_frames)
    for b, e, lab in segments:
        if lab != "sing":
            continue
        for i in range(int(b // hop), min(n_frames, int(np.ceil(e / hop)))):
            cover[i] += max(0.0, min(e, (i + 1) * hop) - max(b, i * hop))
    return (cover / hop > 0.5).astype(np.int8)


def windows_to_frames(starts: np.ndarray, win: float, scores: np.ndarray, n_frames: int, hop: float = 1.0) -> np.ndarray:
    """Puntaje por segundo: media de las ventanas que cubren el centro del segundo (NaN si ninguna)."""
    out = np.full(n_frames, np.nan)
    centers = (np.arange(n_frames) + 0.5) * hop
    for i, c in enumerate(centers):
        m = (starts <= c) & (c < starts + win)
        if m.any():
            out[i] = float(np.mean(scores[m]))
    return out


def electrobyte_split(split: str, root: Path = ELECTROBYTE) -> List[Tuple[str, Path, Path]]:
    return [(a.stem, a, root / "labels" / f"{a.stem}.lab") for a in sorted((root / "audio" / split).glob("*.mp3"))]


# ---------------------------------------------------------------------------
# Métricas (puro)
# ---------------------------------------------------------------------------

def auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUC por rangos (Mann-Whitney), con empates promediados."""
    from scipy.stats import rankdata
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def binary_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {"accuracy": (tp + tn) / max(1, len(y)), "balanced_accuracy": (rec + spec) / 2, "precision": prec,
            "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def best_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """Umbral de máxima exactitud balanceada (se elige en valid, nunca en test)."""
    cands = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 197)))
    return float(max(cands, key=lambda t: binary_metrics(y, (s >= t).astype(int))["balanced_accuracy"]))


def bootstrap_ci(per_track: List[Tuple[np.ndarray, np.ndarray]], thr: float, n: int = 1000, seed: int = 0) -> Dict[str, List[float]]:
    rng = np.random.default_rng(seed)
    stats = {k: [] for k in ("balanced_accuracy", "f1", "auc")}
    for _ in range(n):
        idx = rng.integers(0, len(per_track), len(per_track))
        y = np.concatenate([per_track[i][0] for i in idx]); s = np.concatenate([per_track[i][1] for i in idx])
        m = binary_metrics(y, (s >= thr).astype(int))
        stats["balanced_accuracy"].append(m["balanced_accuracy"]); stats["f1"].append(m["f1"]); stats["auc"].append(auc(y, s))
    return {k: [round(float(np.nanquantile(v, 0.025)), 3), round(float(np.nanquantile(v, 0.975)), 3)] for k, v in stats.items()}


# ---------------------------------------------------------------------------
# Detectores: puntaje por ventana (starts, win, scores) sobre el tema entero
# ---------------------------------------------------------------------------

def ffmpeg_bin() -> str:
    return os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"


def decode(path: Path, sr: int, channels: int = 1) -> np.ndarray:
    raw = subprocess.run([ffmpeg_bin(), "-v", "error", "-i", str(path), "-ac", str(channels), "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    wav = np.frombuffer(raw, dtype=np.float32)
    return wav.reshape(-1, channels).T if channels > 1 else wav


class ClapDetector:
    name, sr, win, hop = "clap", 48000, 10.0, 5.0

    def __init__(self):
        import torch
        from transformers import ClapModel, ClapProcessor
        from src.v4.pipeline.extract_representations import CLAP_NAME
        from src.v4.pipeline.tag_vocals import INSTRUMENTAL_PROMPTS, VOCAL_PROMPTS
        self.torch = torch
        self.proc = ClapProcessor.from_pretrained(CLAP_NAME)
        self.model = ClapModel.from_pretrained(CLAP_NAME).eval()

        def centroid(prompts):  # misma fórmula que tag_vocals.clap_window_probs, calculada una vez
            with torch.no_grad():
                t = self.model.get_text_features(**self.proc(text=prompts, return_tensors="pt", padding=True))
            if not torch.is_tensor(t):
                t = t.text_embeds if getattr(t, "text_embeds", None) is not None else t.pooler_output
            t = t.numpy()
            t = t / np.linalg.norm(t, axis=1, keepdims=True)
            c = t.mean(axis=0)
            return c / np.linalg.norm(c)
        self.classes = np.stack([centroid(VOCAL_PROMPTS), centroid(INSTRUMENTAL_PROMPTS)])
        self.scale = float(self.model.logit_scale_a.detach().exp()) if hasattr(self.model, "logit_scale_a") else 100.0

    def probs(self, emb: np.ndarray) -> np.ndarray:
        w = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        logits = self.scale * (w @ self.classes.T)
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p[:, 0] / p.sum(axis=1)

    def score(self, path: Path):
        starts, win, emb = self.embed(path)
        return starts, win, self.probs(emb)

    def embed(self, path: Path):
        """Embeddings CLAP por ventana (con caché en vocal_eval/clap_emb/)."""
        cdir = CACHE / "clap_emb"
        cdir.mkdir(parents=True, exist_ok=True)
        f = cdir / f"{Path(path).stem}.npz"
        if f.exists():
            z = np.load(f)
            return z["starts"], float(z["win"]), z["emb"]
        starts, win, emb = self._embed(path)
        np.savez(f, starts=starts, win=win, emb=emb)
        return starts, win, emb

    def _embed(self, path: Path):
        wav = decode(path, self.sr)
        starts = np.arange(0, max(1, len(wav) - int(self.win * self.sr) + 1), int(self.hop * self.sr))
        wins = [wav[a:a + int(self.win * self.sr)] for a in starts]
        mats = []
        for b in range(0, len(wins), 8):
            inp = self.proc(audio=wins[b:b + 8], sampling_rate=self.sr, return_tensors="pt")
            with self.torch.no_grad():
                f = self.model.get_audio_features(**inp)
            if not self.torch.is_tensor(f):
                f = f.audio_embeds if getattr(f, "audio_embeds", None) is not None else f.pooler_output
            mats.append(f.numpy().astype(np.float32))
        return starts / self.sr, self.win, np.vstack(mats)


class AstDetector:
    name, sr, win, hop = "ast", 16000, 10.24, 5.0

    def __init__(self):
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        from src.v4.pipeline.extract_representations import AST_NAME
        self.torch = torch
        self.fe = ASTFeatureExtractor.from_pretrained(AST_NAME)
        self.model = ASTForAudioClassification.from_pretrained(AST_NAME).eval()
        labels = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]
        self.idx = [labels.index(l) for l in AST_VOICE if l in labels]

    def score(self, path: Path):
        wav = decode(path, self.sr)
        w = int(self.win * self.sr)
        starts = np.arange(0, max(1, len(wav) - w + 1), int(self.hop * self.sr))
        out = []
        for b in range(0, len(starts), 8):
            wins = [wav[a:a + w] for a in starts[b:b + 8]]
            inp = self.fe(wins, sampling_rate=self.sr, return_tensors="pt")
            with self.torch.no_grad():
                p = self.torch.sigmoid(self.model(**inp).logits).numpy()
            out.append(p[:, self.idx].max(axis=1))
        return starts / self.sr, self.win, np.concatenate(out)


class HDemucsDetector:
    """Energía del stem de voz relativa a la mezcla, en dB, por ventana de 1 s (sin solapamiento)."""
    name, sr, win, hop = "hdemucs", 44100, 1.0, 1.0

    def __init__(self):
        import torch
        import torchaudio
        self.torch = torch
        bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
        self.model = bundle.get_model().eval()
        self.sources = list(self.model.sources)

    def score(self, path: Path, chunk_s: float = 10.0, overlap_s: float = 1.0):
        torch = self.torch
        wav = decode(path, self.sr, channels=2)
        x = torch.from_numpy(np.ascontiguousarray(wav))
        ref = x.mean(0)
        x = (x - ref.mean()) / (ref.std() + 1e-8)
        n, step, ov = x.shape[1], int(chunk_s * self.sr), int(overlap_s * self.sr)
        vocals = torch.zeros(n)
        weight = torch.zeros(n)
        fade = torch.ones(step + ov)
        fade[:ov] = torch.linspace(0, 1, ov); fade[-ov:] = torch.linspace(1, 0, ov)
        vi = self.sources.index("vocals")
        for a in range(0, n, step):
            b = min(n, a + step + ov)
            seg = x[:, a:b]
            if seg.shape[1] < step + ov:  # el modelo exige largo fijo: el último pedazo se rellena con silencio
                seg = torch.nn.functional.pad(seg, (0, step + ov - seg.shape[1]))
            with torch.no_grad():
                out = self.model(seg[None])[0, vi].mean(0)[: b - a]
            f = fade[: b - a].clone()
            if a == 0:
                f[:ov] = 1
            if b == n:
                f[-min(ov, b - a):] = torch.maximum(f[-min(ov, b - a):], torch.ones(min(ov, b - a)))
            vocals[a:b] += out * f
            weight[a:b] += f
        vocals = (vocals / weight.clamp(min=1e-6)).numpy()
        mix = x.mean(0).numpy()
        hop = int(self.hop * self.sr)
        nf = n // hop
        ev = np.array([np.mean(vocals[i * hop:(i + 1) * hop] ** 2) for i in range(nf)])
        em = np.array([np.mean(mix[i * hop:(i + 1) * hop] ** 2) for i in range(nf)])
        return np.arange(nf) * self.hop, self.win, 10 * np.log10((ev + 1e-10) / (em + 1e-10))


def window_voice_fraction(segs: List[Tuple[float, float, str]], starts: np.ndarray, win: float) -> np.ndarray:
    """Fracción de cada ventana [a, a + win) marcada como voz."""
    out = np.zeros(len(starts))
    for i, a in enumerate(starts):
        out[i] = sum(max(0.0, min(e, a + win) - max(b, a)) for b, e, lab in segs if lab == "sing") / win
    return out


class ClapProbeDetector:
    """Clasificador lineal (StandardScaler + regresión logística balanceada) sobre los embeddings CLAP
    por ventana, entrenado con Electrobyte train (ventana = voz si más de la mitad tiene voz). C se elige
    por AUC de ventanas en valid; el umbral por segundo se elige después, también en valid."""
    name, win, hop = "clap_probe", 10.0, 5.0
    cache_scores = False

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.clap = ClapDetector()

        def xy(split):
            X, y = [], []
            for name, audio, lab in electrobyte_split(split):
                starts, win, emb = self.clap.embed(audio)
                X.append(emb)
                y.append((window_voice_fraction(read_lab(lab), np.asarray(starts), win) > 0.5).astype(int))
                print(f"[clap_probe] embeddings {split} {name[:40]}", flush=True)
            return np.vstack(X), np.concatenate(y)
        Xtr, ytr = xy("train")
        Xva, yva = xy("valid")
        best = None
        for C in (0.001, 0.01, 0.1, 1.0):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced")).fit(Xtr, ytr)
            a = auc(yva, clf.predict_proba(Xva)[:, 1])
            if best is None or a > best[0]:
                best = (a, C, clf)
        self.info = {"C": best[1], "auc_valid_windows": round(best[0], 3), "train_windows": int(len(ytr)),
                     "train_voice_share": round(float(ytr.mean()), 3)}
        self.clf = best[2]

    def score(self, path: Path):
        starts, win, emb = self.clap.embed(path)
        return starts, win, self.clf.predict_proba(emb)[:, 1]


PROBE_PATH = REPO_ROOT / "models" / "vocal_probe" / "clap_probe.npz"


def save_probe(path: Path = PROBE_PATH) -> Path:
    """Guarda el clasificador lineal como coeficientes (sin pickle) con el umbral por segundo elegido en
    Electrobyte valid (report_clap_probe.json)."""
    det = ClapProbeDetector()
    scaler, lr = det.clf.named_steps["standardscaler"], det.clf.named_steps["logisticregression"]
    thr = json.loads((CACHE / "report_clap_probe.json").read_text(encoding="utf-8"))["threshold_from_valid"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=scaler.mean_, scale=scaler.scale_, coef=lr.coef_[0], intercept=lr.intercept_[0],
             frame_threshold=thr, C=det.info["C"], win=det.win, hop=det.hop)
    return path


DETECTORS = {"clap": ClapDetector, "ast": AstDetector, "hdemucs": HDemucsDetector, "clap_probe": ClapProbeDetector}


# ---------------------------------------------------------------------------
# Banco por tema: MTG-Jamendo voice_instrumental (unánime) en géneros de baile
# ---------------------------------------------------------------------------

JAMENDO = REPO_ROOT / "data" / "public" / "mtg_jamendo"
DANCE_GENRES = {"techno", "house", "deephouse", "minimal", "trance", "dance", "edm", "club", "breakbeat"}
JAMENDO_URL = "https://prod-1.storage.jamendo.com/?trackid={id}&format=mp31"


def jamendo_sample(seed: int = 0) -> List[dict]:
    """Todos los temas con voz y la misma cantidad de instrumentales (al azar, semilla fija), de géneros de
    baile, con etiqueta voice/instrumental unánime; mitad 'dev' (para elegir T) y mitad 'test', por clase."""
    import csv
    def read(name):
        with open(JAMENDO / name, encoding="utf-8") as f:
            r = csv.reader(f, delimiter="\t"); next(r)
            return {row[0]: row for row in r}
    ann, tags = read("music-classification-annotations-clean.tsv"), read("raw_30s_cleantags.tsv")
    rows = []
    for t, row in ann.items():
        lab = next((x.split("---")[1].split(",")[0] for x in row[5:] if x.startswith("voice_instrumental---")), None)
        genres = {x.split("---")[1] for x in tags.get(t, [""] * 6)[5:] if x.startswith("genre---")}
        if lab in ("voice", "instrumental") and genres & DANCE_GENRES:
            rows.append({"track_id": t, "id": t.split("_")[1].lstrip("0"), "label": int(lab == "voice"),
                         "genres": ",".join(sorted(genres)), "duration": float(row[4])})
    rng = np.random.default_rng(seed)
    voice = [r for r in rows if r["label"] == 1]
    inst = [r for r in rows if r["label"] == 0]
    inst = [inst[i] for i in sorted(rng.choice(len(inst), size=len(voice), replace=False))]
    out = []
    for group in (voice, inst):
        perm = rng.permutation(len(group))
        for k, i in enumerate(perm):
            out.append({**group[i], "split": "dev" if k % 2 == 0 else "test"})
    return out


def fetch_jamendo(items: List[dict], min_interval: float = 1.0) -> int:
    import requests
    adir = JAMENDO / "audio"
    adir.mkdir(parents=True, exist_ok=True)
    n = 0
    for it in items:
        f = adir / f"{it['track_id']}.mp3"
        if f.exists() and f.stat().st_size > 10000:
            continue
        r = requests.get(JAMENDO_URL.format(id=it["id"]), timeout=60)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("audio/"):
            f.write_bytes(r.content)
            n += 1
        else:
            print(f"[jamendo] {it['track_id']}: {r.status_code} {r.headers.get('content-type')}")
        time.sleep(min_interval)
    return n


def track_level(det_name: str, frame_thr: float) -> dict:
    """Por tema: segundos con voz (puntaje por segundo >= umbral elegido en Electrobyte valid). T en dev,
    métricas en test; también AUC de los segundos con voz sin umbral."""
    det = DETECTORS[det_name]()
    items = jamendo_sample()
    res = {"dev": [], "test": []}
    t0 = time.time()
    for i, it in enumerate(items, 1):
        path = JAMENDO / "audio" / f"{it['track_id']}.mp3"
        if not path.exists():
            continue
        starts, win, scores = track_scores(det, "jamendo_" + it["track_id"], path)
        n = int(np.floor(max(starts) + win))
        s = windows_to_frames(np.asarray(starts), win, np.asarray(scores), n)
        s = s[~np.isnan(s)]
        res[it["split"]].append((it["label"], float((s >= frame_thr).sum()), float((s >= frame_thr).mean())))
        if i % 20 == 0:
            print(f"[{det_name}] jamendo {i}/{len(items)} {time.time() - t0:.0f} s", flush=True)
    dev, test = np.array(res["dev"]), np.array(res["test"])
    out = {"detector": det_name, "frame_threshold": frame_thr, "n_dev": len(dev), "n_test": len(test)}
    for col, name in ((1, "seconds"), (2, "fraction")):
        T = best_threshold(dev[:, 0].astype(int), dev[:, col])
        m = binary_metrics(test[:, 0].astype(int), (test[:, col] >= T).astype(int))
        per = [(np.array([int(l)]), np.array([v])) for l, v in zip(test[:, 0], test[:, col])]
        out[name] = {"T_from_dev": round(T, 3), "auc_dev": round(auc(dev[:, 0].astype(int), dev[:, col]), 3),
                     "auc_test": round(auc(test[:, 0].astype(int), test[:, col]), 3),
                     "test": {k: round(v, 3) for k, v in m.items()}, "test_ci95": bootstrap_ci(per, T)}
    return out


# ---------------------------------------------------------------------------
# Corrida
# ---------------------------------------------------------------------------

def track_scores(det, name: str, path: Path) -> Tuple[np.ndarray, float, np.ndarray]:
    if not getattr(det, "cache_scores", True):   # depende de un modelo entrenado: no se guarda
        return det.score(path)
    cdir = CACHE / det.name
    cdir.mkdir(parents=True, exist_ok=True)
    f = cdir / f"{name}.npz"
    if f.exists():
        z = np.load(f)
        return z["starts"], float(z["win"]), z["scores"]
    starts, win, scores = det.score(path)
    np.savez(f, starts=starts, win=win, scores=scores)
    return starts, win, scores


def collect(det, split: str, limit: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    items = electrobyte_split(split)[: limit or None]
    per = []
    t0 = time.time()
    for i, (name, audio, lab) in enumerate(items, 1):
        segs = read_lab(lab)
        n = int(np.floor(segs[-1][1]))
        starts, win, scores = track_scores(det, name, audio)
        s = windows_to_frames(np.asarray(starts), win, np.asarray(scores), n)
        y = frame_labels(segs, n)
        ok = ~np.isnan(s)
        per.append((y[ok], s[ok]))
        print(f"[{det.name}] {split} {i}/{len(items)} {time.time() - t0:.0f} s", flush=True)
    return per


def evaluate(det_name: str, limit: int = 0) -> dict:
    det = DETECTORS[det_name]()
    val, test = collect(det, "valid", limit), collect(det, "test", limit)
    yv, sv = np.concatenate([p[0] for p in val]), np.concatenate([p[1] for p in val])
    yt, st = np.concatenate([p[0] for p in test]), np.concatenate([p[1] for p in test])
    thr = best_threshold(yv, sv)
    rep = {"detector": det_name, "frames_valid": int(len(yv)), "frames_test": int(len(yt)),
           "voice_share_test": round(float(yt.mean()), 3), "threshold_from_valid": round(thr, 4),
           "auc_valid": round(auc(yv, sv), 3), "auc_test": round(auc(yt, st), 3),
           "test": {k: round(v, 3) for k, v in binary_metrics(yt, (st >= thr).astype(int)).items()},
           "test_ci95": bootstrap_ci(test, thr)}
    if hasattr(det, "info"):
        rep["model"] = det.info
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description="Evalúa detectores de voz: Electrobyte por segundo (valid para el umbral, "
                                             "test para medir) y MTG-Jamendo por tema (--track-level)")
    ap.add_argument("--detector", choices=sorted(DETECTORS))
    ap.add_argument("--limit", type=int, default=0, help="Solo los primeros N temas de cada partición (prueba)")
    ap.add_argument("--fetch-jamendo", action="store_true", help="Bajar el audio del banco MTG-Jamendo (CC)")
    ap.add_argument("--save-probe", action="store_true",
                    help="Entrenar clap_probe y guardarlo en models/vocal_probe/clap_probe.npz (lo usa tag_vocals)")
    ap.add_argument("--track-level", action="store_true", help="Evaluar por tema en MTG-Jamendo")
    ap.add_argument("--low-priority", action="store_true")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    if args.low_priority:
        from src.v4.pipeline.extract_representations import lower_priority
        lower_priority()
    if args.fetch_jamendo:
        items = jamendo_sample()
        print(f"[jamendo] {len(items)} temas ({sum(i['label'] for i in items)} con voz); bajados ahora: {fetch_jamendo(items)}")
        return 0
    if args.save_probe:
        out = save_probe()
        print(f"[probe] guardado en {out}")
        return 0
    if not args.detector:
        ap.error("falta --detector")
    import torch
    torch.set_num_threads(args.threads)
    if args.track_level:
        frame_thr = json.loads((CACHE / f"report_{args.detector}.json").read_text(encoding="utf-8"))["threshold_from_valid"]
        rep = track_level(args.detector, frame_thr)
        out = CACHE / f"report_{args.detector}_tracks.json"
        out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(rep, ensure_ascii=False, indent=1))
        return 0
    rep = evaluate(args.detector, args.limit)
    out = CACHE / f"report_{args.detector}{'_limit' + str(args.limit) if args.limit else ''}.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
