"""
PURPOSE: Etiqueta "Vocal" por tema con modelos preentrenados, a partir de las ventanas de 10 s que
         guarda extract_representations (3 segmentos x 3 ventanas del tramo central):
           - método 'clap': zero-shot con CLAP (textos "con voces" vs "instrumental") sobre
             '_winclap' de representations/clap_full/cache (sin costo extra de audio);
           - método 'ast': probabilidad máxima de clases de canto/habla de AudioSet (AST) sobre
             '_winprob' de representations/ast_full/cache.
         Regla: con --mean-threshold, Vocal si la P(voz) media de las ventanas >= umbral (default de
         CLAP 0.145: calibrado el 2026-09-27 contra la carpeta "2020 new/Vocal" de Gabriel, AUC 0.906,
         exploratorio, y bajado de 0.15 a pedido de Gabriel para incluir "I Don't Feel Like Dancing"
         de Scissor Sisters, que quedaba justo debajo); sin él, Vocal si la fracción de ventanas con voz >= --min-coverage. Escribe
         features/vocals.parquet y, con --write-tags, agrega " - Vocal" al comentario del archivo
         (MP3/AIFF/FLAC) guardando un CSV de respaldo; --revert <csv> restaura los comentarios.
         --check-list N escribe features/vocal_check.m3u8: los N temas más cerca del umbral (mitad
         arriba, mitad abajo) para verificar la etiqueta escuchando.
CHANGELOG:
  - 2026-09-27: Creación inicial (pedido de Gabriel: etiqueta Vocal en la metadata). Umbral 0.145.
                Solo se cuentan y etiquetan temas del catálogo actual (la caché puede tener otros).
"""
import argparse
import datetime as dt
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.catalog import load_catalog, update_catalog_columns  # noqa: E402
from src.v4.common.config_loader import load_config  # noqa: E402
from src.v4.common.path_resolver import resolve_dataset_artifacts  # noqa: E402
from src.v4.common.tags import WRITABLE_SUFFIXES, read_tags, with_token, write_comment  # noqa: E402

VOCAL_PROMPTS = [
    "a song with a singer singing", "music with singing vocals", "a female voice singing over a beat",
    "a male voice singing over a beat", "a dance track with vocals", "a person rapping or talking over music",
]
INSTRUMENTAL_PROMPTS = [
    "instrumental music", "an instrumental electronic track with no vocals", "instrumental techno",
    "instrumental house music with drums and synths", "a beat with no voice",
]
AST_SINGING = ("Singing", "Male singing", "Female singing", "Child singing", "Synthetic singing",
               "Vocal music", "Rapping", "Choir", "Chant")
AST_SPEECH = ("Speech", "Male speech, man speaking", "Female speech, woman speaking", "Narration, monologue")
TOKEN = "Vocal"
DEFAULT_CLAP_THRESHOLD = 0.145


def clap_window_probs(win_embs: Dict[str, np.ndarray], hf_cache=None) -> Dict[str, np.ndarray]:
    """P(voz) por ventana: softmax entre las medias de textos vocal/instrumental (escala de CLAP)."""
    import torch
    from transformers import ClapModel, ClapProcessor
    from src.v4.pipeline.extract_representations import CLAP_NAME
    kwargs = {"cache_dir": hf_cache} if hf_cache else {}
    proc = ClapProcessor.from_pretrained(CLAP_NAME, **kwargs)
    model = ClapModel.from_pretrained(CLAP_NAME, **kwargs).eval()

    def _text(prompts: List[str]) -> np.ndarray:
        with torch.no_grad():
            t = model.get_text_features(**proc(text=prompts, return_tensors="pt", padding=True))
        if not torch.is_tensor(t):
            t = getattr(t, "text_embeds", None) if getattr(t, "text_embeds", None) is not None else t.pooler_output
        t = t.numpy()
        t = t / np.linalg.norm(t, axis=1, keepdims=True)
        c = t.mean(axis=0)
        return c / np.linalg.norm(c)

    classes = np.stack([_text(VOCAL_PROMPTS), _text(INSTRUMENTAL_PROMPTS)])  # (2, 512)
    scale = float(model.logit_scale_a.detach().exp()) if hasattr(model, "logit_scale_a") else 100.0
    out = {}
    for uid, w in win_embs.items():
        w = w / np.linalg.norm(w, axis=1, keepdims=True)
        logits = scale * (w @ classes.T)
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        out[uid] = (p[:, 0] / p.sum(axis=1)).astype(np.float32)
    return out


def ast_window_probs(win_probs: Dict[str, np.ndarray], labels: List[str], include_speech: bool) -> Dict[str, np.ndarray]:
    """P(voz) por ventana = máximo de las clases de canto (y habla si include_speech)."""
    names = set(AST_SINGING) | (set(AST_SPEECH) if include_speech else set())
    idx = [i for i, lab in enumerate(labels) if lab in names]
    return {uid: w[:, idx].max(axis=1) for uid, w in win_probs.items()}


def summarize(window_probs: Dict[str, np.ndarray], threshold: float, min_coverage: float,
              mean_threshold: Optional[float] = None) -> pd.DataFrame:
    rows = []
    for uid, p in window_probs.items():
        coverage = float((p >= threshold).mean())
        mean = float(p.mean())
        is_vocal = mean >= mean_threshold if mean_threshold is not None else coverage >= min_coverage
        rows.append({"track_uid": uid, "vocal_max": float(p.max()), "vocal_mean": mean,
                     "vocal_coverage": coverage, "is_vocal": is_vocal})
    return pd.DataFrame(rows)


def check_list(vocals: pd.DataFrame, catalog: pd.DataFrame, n: int, out_path: Path,
               score: str = "vocal_mean", threshold: float = DEFAULT_CLAP_THRESHOLD) -> pd.DataFrame:
    """M3U8 con n//2 temas justo arriba y n//2 justo abajo del umbral (orden: arriba primero)."""
    from src.v4.common.dj_export import write_m3u8
    v = vocals.merge(catalog[["track_uid", "source_path", "artist", "title", "duration_s"]], on="track_uid")
    above = v[v[score] >= threshold].nsmallest(n // 2, score)
    below = v[v[score] < threshold].nlargest(n - n // 2, score)
    picked = pd.concat([above.assign(label="Vocal"), below.assign(label="sin Vocal")])
    write_m3u8(out_path, picked.set_index("track_uid"), picked["track_uid"].tolist())
    return picked


def _load_windows(cache_dir: Path, key: str) -> Dict[str, np.ndarray]:
    out = {}
    for f in sorted(cache_dir.glob("*.npz")):
        data = np.load(f)
        if key in data.files:
            out[f.stem] = data[key]
    return out


def write_tags(catalog: pd.DataFrame, vocals: pd.DataFrame, backup_path: Path) -> pd.DataFrame:
    """Agrega TOKEN al comentario de los temas Vocal. Respaldo CSV con el comentario anterior."""
    rows = []
    merged = catalog.merge(vocals[["track_uid", "is_vocal"]], on="track_uid", how="inner")
    for r in merged[merged["is_vocal"]].itertuples():
        path = Path(r.source_path)
        if path.suffix.lower() not in WRITABLE_SUFFIXES:
            rows.append({"track_uid": r.track_uid, "source_path": str(path), "old_comment": None,
                         "new_comment": None, "status": "skipped_format", "detail": path.suffix.lower()})
            continue
        old = read_tags(path)["tag_comment"]
        new = with_token(old, TOKEN)
        status, detail = "unchanged", ""
        if new != (old or "") and not os.access(path, os.W_OK):
            status = "read_only"  # atributo de solo lectura: no se cambia sin pedirlo
        elif new != (old or ""):
            try:
                write_comment(path, new)
                status = "written"
            except Exception as exc:  # noqa: BLE001
                status, detail = "error", f"{type(exc).__name__}: {exc}"
        rows.append({"track_uid": r.track_uid, "source_path": str(path), "old_comment": old,
                     "new_comment": new, "status": status, "detail": detail})
    log = pd.DataFrame(rows)
    log.to_csv(backup_path, index=False, encoding="utf-8")
    return log


def revert(backup_csv: Path) -> int:
    log = pd.read_csv(backup_csv, encoding="utf-8")
    n = 0
    for r in log[log["status"] == "written"].itertuples():
        write_comment(Path(r.source_path), "" if pd.isna(r.old_comment) else str(r.old_comment))
        n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description="Etiqueta Vocal por tema (CLAP zero-shot o AST AudioSet)")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--method", choices=("clap", "ast"), default="clap")
    parser.add_argument("--threshold", type=float, default=0.5, help="P(voz) mínima para contar una ventana")
    parser.add_argument("--min-coverage", type=float, default=0.34, help="Fracción mínima de ventanas con voz")
    parser.add_argument("--mean-threshold", type=float, default=None,
                        help="Vocal si P(voz) media >= umbral (default 0.145 con --method clap)")
    parser.add_argument("--ast-include-speech", action="store_true", help="Contar habla (spoken word) como voz")
    parser.add_argument("--write-tags", action="store_true", help="Escribir ' - Vocal' en el comentario")
    parser.add_argument("--revert", default=None, help="CSV de respaldo: restaurar comentarios")
    parser.add_argument("--check-list", type=int, default=0, help="N temas cerca del umbral → features/vocal_check.m3u8")
    args = parser.parse_args()

    if args.revert:
        print(f"[INFO] Restaurados {revert(Path(args.revert))} comentarios")
        return 0

    config = load_config(Path(args.config) if args.config else None)
    artifacts = resolve_dataset_artifacts(args.dataset_name, config)
    rep_root = artifacts / "representations"
    if args.method == "clap":
        wins = _load_windows(rep_root / "clap_full" / "cache", "_winclap")
        probs = clap_window_probs(wins)
    else:
        from transformers import AutoConfig
        from src.v4.pipeline.extract_representations import AST_NAME
        cfg = AutoConfig.from_pretrained(AST_NAME)
        labels = [cfg.id2label[i] for i in range(len(cfg.id2label))]
        probs = ast_window_probs(_load_windows(rep_root / "ast_full" / "cache", "_winprob"), labels,
                                 args.ast_include_speech)
    catalog = load_catalog(args.dataset_name, config)
    probs = {uid: p for uid, p in probs.items() if uid in set(catalog["track_uid"])}
    if not probs:
        print(f"[ERROR] No hay ventanas en caché para el método {args.method}; correr extract_representations")
        return 1

    mean_thr = args.mean_threshold if args.mean_threshold is not None else (DEFAULT_CLAP_THRESHOLD if args.method == "clap" else None)
    vocals = summarize(probs, args.threshold, args.min_coverage, mean_thr)
    vocals["method"] = args.method
    features = artifacts / "features"
    features.mkdir(parents=True, exist_ok=True)
    out = features / f"vocals_{args.method}.parquet"
    vocals.to_parquet(out, index=False)
    rule = f"media >= {mean_thr}" if mean_thr is not None else f"cobertura >= {args.min_coverage} (umbral {args.threshold})"
    print(f"[INFO] {vocals['is_vocal'].sum()}/{len(vocals)} temas Vocal ({args.method}, {rule}) → {out}")

    if args.check_list and mean_thr is not None:
        picked = check_list(vocals, catalog, args.check_list, features / "vocal_check.m3u8", threshold=mean_thr)
        print(f"[INFO] Lista de chequeo ({len(picked)} temas) → {features / 'vocal_check.m3u8'}")
        for r in picked.itertuples():
            print(f"    {r.label:9s} {r.vocal_mean:.3f}  {r.artist} - {r.title}")

    if args.write_tags:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = features / f"vocal_tags_backup_{stamp}.csv"
        log = write_tags(catalog, vocals, backup)
        print(f"[INFO] Comentarios: {log['status'].value_counts().to_dict()} | respaldo: {backup}")
        written = log[log["status"] == "written"].set_index("track_uid")["new_comment"]
        if not written.empty:
            # El catálogo refleja el comentario nuevo (lo usan los exports de Rekordbox/Traktor)
            comments = catalog.set_index("track_uid")["tag_comment"].astype(object)
            comments.update(written)
            update_catalog_columns(args.dataset_name, config, comments.reset_index())
    return 0


if __name__ == "__main__":
    sys.exit(main())
