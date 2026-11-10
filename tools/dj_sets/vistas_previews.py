"""
PURPOSE: V7 del plan de playlists en vivo (sección 12.9): las vistas de cada preview de Beatport bajado por D6 (CLAP,
         MAEST capa 7 y los 400 estilos Discogs), con la ventana de E15 (V4: el preview entero en 4 segmentos de 30 s) y
         el embedding de cada segmento además de su media. Una salida por tanda, alineada por el id de Beatport:
           <remoto>/v7_vistas/<corrida>/<tanda>.npz   bp_id (str), clap (n, 4, 512), l7 (n, 4, 768), styles (n, 4, 400),
                                                      segundos (n), meta (json: modelos, commit, dispositivo, fallidos)
         Lee el manifiesto (<remoto>/beatport_previews/<corrida>/<tanda>.json) y el .tar de la tanda del Drive con
         open_remote(); no le pide nada a Beatport. Es un trabajo de gpuq (AGENTS.md, «Compute»; docs/GPUQ.md): se
         retoma por tanda (salta las que ya están en el remoto) y deja el .tar en GPUQ_SCRATCH solo mientras la procesa.
         Mismo código de modelos que la biblioteca y E15 (ClapBackend, MaestHfBackend, resample), en lotes y en GPU si
         hay; `verificar` compara una salida con los embeddings de E15 de los mismos previews.
           python tools/dj_sets/vistas_previews.py correr [--corridas lote_01,lote_02] [--max-tandas N] [--max-temas N]
                                                          [--tandas lote_03/tanda_00003] [--destino v7_prueba]
           python tools/dj_sets/vistas_previews.py verificar --npz <salida.npz> [--e15 "D:/TRAKTOR ML datos/dj_sets/e15/emb"]
CHANGELOG:
  - 2026-11-10: Creación (V7).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)

CORRIDAS = ("piloto", "res_piloto", "lote_01", "lote_02", "lote_03", "lote_04", "rescate_01", "res_rescate_01")
ORIGEN = "beatport_previews"
DESTINO = "v7_vistas"
SEG_S = 30.0
N_SEG = 4
LOTE_CLAP = 48               # ventanas de 10 s por pasada de CLAP
LOTE_MAEST = 16              # segmentos de 30 s por pasada de MAEST


def cortar(x: np.ndarray, sr: int, n_seg: int = N_SEG, seg_s: float = SEG_S) -> List[np.ndarray]:
    """n_seg segmentos consecutivos de seg_s desde el comienzo, con ceros si falta (= dominio_previews.cortar, V4)."""
    n = int(seg_s * sr)
    out = []
    for k in range(n_seg):
        s = x[k * n:(k + 1) * n]
        out.append(np.pad(s, (0, n - len(s))).astype(np.float32) if len(s) < n else s.astype(np.float32))
    return out


def leer_mono(path: Path) -> Tuple[np.ndarray, int]:
    """Audio mono float32 (= dominio_previews.leer_mono sin recorte): soundfile y, si no abre, ffmpeg."""
    import soundfile as sf
    try:
        with sf.SoundFile(str(path)) as f:
            return f.read(-1, dtype="float32", always_2d=True).mean(axis=1), f.samplerate
    except Exception:  # noqa: BLE001
        sr = 44100
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                             capture_output=True, check=True).stdout
        return np.frombuffer(raw, dtype="<f4").copy(), sr


class Modelos:
    """CLAP y MAEST de extract_representations, en lotes y en el dispositivo pedido. Un segmento da lo mismo que
    ClapBackend.embed (media de sus ventanas de 10 s) y MaestHfBackend.embed (media de los tokens de la capa 7 y
    sigmoide de los 400 estilos)."""

    def __init__(self, hf_cache: Optional[str], device: str):
        from src.v4.pipeline.extract_representations import ClapBackend, MaestHfBackend, resample, split_windows
        import torch
        self.torch, self.resample, self.split = torch, resample, split_windows
        self.device = device
        self.clap = ClapBackend(hf_cache)
        self.maest = MaestHfBackend(None)          # como E15: MAEST en la caché por defecto de Hugging Face
        self.clap.model.to(device)
        self.maest.model.to(device)
        self.info = {"clap": self.clap.info, "maest": self.maest.info, "device": device,
                     "torch": torch.__version__, "transformers": __import__("transformers").__version__}

    def _clap(self, wins: List[np.ndarray]) -> np.ndarray:
        out = []
        for a in range(0, len(wins), LOTE_CLAP):
            inputs = self.clap.processor(audio=wins[a:a + LOTE_CLAP], sampling_rate=self.clap.sr, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self.torch.no_grad():
                f = self.clap.model.get_audio_features(**inputs)
            if not self.torch.is_tensor(f):
                f = getattr(f, "audio_embeds", None) if getattr(f, "audio_embeds", None) is not None else f.pooler_output
            out.append(f.float().cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def _maest(self, segs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        l7, st = [], []
        for a in range(0, len(segs), LOTE_MAEST):
            inputs = self.maest.fe(segs[a:a + LOTE_MAEST], sampling_rate=self.maest.sr, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self.torch.no_grad():
                out = self.maest.model(**inputs, output_hidden_states=True)
            l7.append(out.hidden_states[self.maest.layer].mean(dim=1).float().cpu().numpy())
            st.append(self.torch.sigmoid(out.logits).float().cpu().numpy())
        return np.concatenate(l7).astype(np.float32), np.concatenate(st).astype(np.float32)

    def embed(self, previews: Sequence[Tuple[np.ndarray, int]]) -> Dict[str, np.ndarray]:
        """(n, N_SEG, d) por vista para n previews (audio mono, sr)."""
        segs = [s for x, sr in previews for s in cortar(x, sr)]
        srs = [sr for _, sr in previews for _ in range(N_SEG)]
        w48 = [self.split(self.resample(s, sr, self.clap.sr), self.clap.sr * 10) for s, sr in zip(segs, srs)]
        nw = len(w48[0])
        if any(len(w) != nw for w in w48):
            raise ValueError("segmentos de largo distinto")
        clap = self._clap([v for w in w48 for v in w]).reshape(len(segs), nw, -1).mean(axis=1)
        l7, st = self._maest([self.resample(s, sr, self.maest.sr) for s, sr in zip(segs, srs)])
        n = len(previews)
        return {"clap": clap.reshape(n, N_SEG, -1), "l7": l7.reshape(n, N_SEG, -1), "styles": st.reshape(n, N_SEG, -1)}


def temas_ok(manifiesto: dict) -> List[dict]:
    """Los temas de la tanda con preview bajado, sin repetir el id de Beatport."""
    vistos, out = set(), []
    for t in manifiesto.get("temas", []):
        bid = str(t.get("beatport_id") or "")
        if t.get("estado") == "ok" and t.get("archivo") and bid and bid not in vistos:
            vistos.add(bid)
            out.append(t)
    return out


def procesar_tanda(remote, corrida: str, tanda: str, modelos: Modelos, scratch: Path, max_temas: Optional[int] = None,
                   lote: int = 16, commit: str = "", destino: str = DESTINO) -> Dict[str, object]:
    """Una tanda: baja el manifiesto y el .tar, embebe cada preview y sube <DESTINO>/<corrida>/<tanda>.npz."""
    t0 = time.perf_counter()
    man = remote.get_json(f"{ORIGEN}/{corrida}/{tanda}.json")
    temas = temas_ok(man)[:max_temas]
    tar_local = scratch / f"{corrida}_{tanda}.tar"
    remote.get(f"{ORIGEN}/{corrida}/{tanda}.tar", tar_local)
    t_bajada = time.perf_counter() - t0
    por_archivo = {t["archivo"]: t for t in temas}
    ids, segundos, fallidos = [], [], []
    acc: Dict[str, List[np.ndarray]] = {"clap": [], "l7": [], "styles": []}
    pendientes: List[Tuple[str, np.ndarray, int]] = []

    def vaciar():
        if not pendientes:
            return
        e = modelos.embed([(x, sr) for _, x, sr in pendientes])
        for k in acc:
            acc[k].append(e[k])
        ids.extend(b for b, _, _ in pendientes)
        segundos.extend(len(x) / sr for _, x, sr in pendientes)
        pendientes.clear()

    try:
        with tarfile.open(tar_local) as tar, tempfile.TemporaryDirectory(dir=scratch) as tmp:
            for m in tar:
                t = por_archivo.get(m.name)
                if t is None:
                    continue
                mp3 = Path(tmp) / Path(m.name).name
                try:
                    mp3.write_bytes(tar.extractfile(m).read())
                    x, sr = leer_mono(mp3)
                    if not len(x) or not np.isfinite(x).all():
                        raise ValueError("audio vacío o no finito")
                    pendientes.append((str(t["beatport_id"]), x, sr))
                except Exception as exc:  # noqa: BLE001
                    fallidos.append({"bp_id": str(t["beatport_id"]), "error": f"{type(exc).__name__}: {exc}"[:200]})
                finally:
                    mp3.unlink(missing_ok=True)
                if len(pendientes) >= lote:
                    vaciar()
                    print(f"    {corrida}/{tanda}: {len(ids)}/{len(temas)} ({time.perf_counter() - t0:.0f} s)", flush=True)
            vaciar()
    finally:
        tar_local.unlink(missing_ok=True)
    faltan = sorted(set(str(t["beatport_id"]) for t in temas) - set(ids) - {f["bp_id"] for f in fallidos})
    fallidos += [{"bp_id": b, "error": "no está en el .tar"} for b in faltan]
    out = {k: (np.concatenate(v) if v else np.zeros((0, N_SEG, d), np.float32)) for (k, v), d in zip(acc.items(), (512, 768, 400))}
    if not all(np.isfinite(v).all() for v in out.values()):
        raise ValueError(f"{corrida}/{tanda}: embeddings no finitos")
    seg = time.perf_counter() - t0
    meta = {"corrida": corrida, "tanda": tanda, "temas_ok": len(temas), "hechos": len(ids), "fallidos": fallidos,
            "ventana": "V4: el preview entero en 4 segmentos de 30 s (E15)", "commit": commit, "modelos": modelos.info,
            "segundos": round(seg, 1), "segundos_bajada": round(t_bajada, 1), "gpuq_job": os.environ.get("GPUQ_JOB_ID")}
    npz = scratch / f"{corrida}_{tanda}.npz"
    np.savez_compressed(npz, bp_id=np.asarray(ids, dtype=str), segundos=np.asarray(segundos, dtype=np.float32),
                        meta=np.asarray(json.dumps(meta, ensure_ascii=False)), **out)
    remote.put(npz, f"{destino}/{corrida}/{tanda}.npz")
    npz.unlink(missing_ok=True)
    return meta


def correr(corridas: Sequence[str], max_tandas: Optional[int], max_temas: Optional[int], device: str,
           hf_cache: Optional[str], lote: int, destino: str = DESTINO, tandas: Optional[Sequence[str]] = None) -> int:
    from src.v4.common.remote_storage import open_remote
    remote = open_remote()
    scratch = Path(os.environ.get("GPUQ_SCRATCH") or tempfile.gettempdir()) / "v7"
    scratch.mkdir(parents=True, exist_ok=True)
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    commit = os.environ.get("GPUQ_CODE_SHA") or ""
    pend = []
    for c in corridas:
        hay = remote.ls(f"{ORIGEN}/{c}")
        hechos = remote.ls(f"{destino}/{c}")
        for f in sorted(hay):
            listo = f[:-4] + ".npz" in hechos
            elegida = not tandas or f"{c}/{f[:-4]}" in tandas
            if f.startswith("tanda_") and f.endswith(".tar") and f[:-4] + ".json" in hay and not listo and elegida:
                pend.append((c, f[:-4]))
    print(f"{len(pend)} tandas por hacer en {', '.join(corridas)}; dispositivo {device}", flush=True)
    pend = pend[:max_tandas]
    if not pend:
        return 0
    modelos = Modelos(hf_cache, device)
    print(f"modelos listos: {json.dumps(modelos.info, ensure_ascii=False)[:300]}", flush=True)
    t0, total = time.perf_counter(), 0
    for c, t in pend:
        meta = procesar_tanda(remote, c, t, modelos, scratch, max_temas, lote, commit, destino)
        total += meta["hechos"]
        el = time.perf_counter() - t0
        print(f"  {c}/{t}: {meta['hechos']}/{meta['temas_ok']} previews, {len(meta['fallidos'])} fallidos, "
              f"{meta['segundos']:.0f} s (bajada {meta['segundos_bajada']:.0f} s) | {total} en {el / 60:.1f} min, "
              f"{el / max(total, 1):.2f} s por preview", flush=True)
    print(f"[OK] {total} previews en {len(pend)} tandas", flush=True)
    return 0


def verificar(npz: Path, e15: Path) -> Dict[str, object]:
    """Coseno, segmento a segmento, con los embeddings de E15 (prev_*) de los mismos previews."""
    z = np.load(npz)
    cos: Dict[str, List[float]] = {"clap": [], "l7": [], "styles": []}
    n = 0
    for i, b in enumerate(z["bp_id"]):
        f = e15 / f"{b}.npz"
        if not f.exists():
            continue
        e = np.load(f)
        n += 1
        for k in cos:
            A, B = z[k][i], e[f"prev_{k}"]
            cos[k] += list(np.sum(A * B, axis=1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)))
    return {"previews_en_e15": n, **{k: {"min": round(float(np.min(v)), 5), "mediana": round(float(np.median(v)), 5)} for k, v in cos.items() if v}}


def main() -> int:
    ap = argparse.ArgumentParser(description="V7: vistas de los previews de Beatport (trabajo de gpuq)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("correr")
    c.add_argument("--corridas", default=",".join(CORRIDAS))
    c.add_argument("--max-tandas", type=int, default=None)
    c.add_argument("--max-temas", type=int, default=None, help="Solo los primeros N previews de cada tanda (prueba chica)")
    c.add_argument("--device", default="auto")
    c.add_argument("--hf-cache", default=None)
    c.add_argument("--lote", type=int, default=16, help="Previews por pasada de los modelos")
    c.add_argument("--destino", default=DESTINO, help="Carpeta del remoto para las salidas (v7_prueba en el trabajo corto)")
    c.add_argument("--tandas", default="", help="Solo estas tandas: corrida/tanda_00003,...")
    v = sub.add_parser("verificar")
    v.add_argument("--npz", required=True)
    v.add_argument("--e15", default="D:/TRAKTOR ML datos/dj_sets/e15/emb")
    args = ap.parse_args()
    if args.cmd == "correr":
        return correr([x for x in args.corridas.split(",") if x], args.max_tandas, args.max_temas, args.device,
                      args.hf_cache, args.lote, args.destino, [x for x in args.tandas.split(",") if x])
    print(json.dumps(verificar(Path(args.npz), Path(args.e15)), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
