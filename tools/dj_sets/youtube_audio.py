"""
PURPOSE: Audio de YouTube para aprender trayectorias de DJs (plan de playlists en vivo, fuentes D7 y D8;
         DECISIONS 2026-11-07, punto 12: Gabriel autoriza bajar de YouTube para esta aplicación). Baja
         una playlist o un video a <root>/audio/<id>/ con yt-dlp, sin recodificar (m4a si hay), y deja
         <root>/audio/<id>/manifest.json con cada entrada: índice, id de YouTube, título, canal,
         duración y archivo. Lo ya bajado no se vuelve a bajar. Pausa entre descargas.
           python tools/dj_sets/youtube_audio.py --playlist <id o url> [--root D:/...] [--limit N]
           python tools/dj_sets/youtube_audio.py --video <id o url> --nombre <carpeta>
         El audio queda fuera de Git (artifacts/ o la carpeta de datos en D:).
CHANGELOG:
  - 2026-11-07: Creación (piloto de trayectorias de DJs).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.v4.common  # noqa: E402,F401  (salida UTF-8 en consolas cp1252)

DEFAULT_ROOT = REPO_ROOT / "artifacts" / "v4" / "dj_sets"
AUDIO_EXTS = (".m4a", ".webm", ".opus", ".mp3", ".ogg")


def ytdlp(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "yt_dlp", *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def list_playlist(url: str) -> Dict:
    p = ytdlp("--flat-playlist", "-J", url)
    if p.returncode != 0:
        raise RuntimeError(f"yt-dlp no pudo listar {url}: {p.stderr.strip()[-300:]}")
    return json.loads(p.stdout)


def existing(folder: Path, vid: str) -> Optional[Path]:
    for ext in AUDIO_EXTS:
        hits = list(folder.glob(f"*_{vid}{ext}"))
        if hits:
            return hits[0]
    return None


def download(url: str, folder: Path, stem: str) -> Optional[Path]:
    p = ytdlp("-f", "bestaudio[ext=m4a]/bestaudio", "--no-playlist", "--no-progress",
              "-o", str(folder / f"{stem}.%(ext)s"), url)
    if p.returncode != 0:
        print(f"  [ERROR] {url}: {p.stderr.strip()[-200:]}", flush=True)
        return None
    for ext in AUDIO_EXTS:
        f = folder / f"{stem}{ext}"
        if f.exists():
            return f
    return None


def run(entries: List[Dict], folder: Path, meta: Dict, pause: float) -> Dict:
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, e in enumerate(entries, 1):
        vid = e["id"]
        f = existing(folder, vid)
        if f is None:
            f = download(f"https://www.youtube.com/watch?v={vid}", folder, f"{i:03d}_{vid}")
            time.sleep(pause)
        rows.append({"indice": i, "id": vid, "titulo": e.get("title"), "canal": e.get("channel") or e.get("uploader"),
                     "duracion": e.get("duration"), "archivo": f.name if f else None})
        print(f"  {i}/{len(entries)} {'ok ' if f else 'FALTA'} {e.get('title')}", flush=True)
    manifest = {**meta, "entradas": rows}
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Audio de YouTube para el piloto de trayectorias")
    ap.add_argument("--playlist")
    ap.add_argument("--video")
    ap.add_argument("--nombre", help="Carpeta para --video (por defecto, el id)")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pausa", type=float, default=3.0, help="Segundos entre descargas")
    args = ap.parse_args()
    root = Path(args.root) / "audio"
    if args.playlist:
        url = args.playlist if args.playlist.startswith("http") else f"https://www.youtube.com/playlist?list={args.playlist}"
        d = list_playlist(url)
        entries = [e for e in d.get("entries") or [] if e.get("id")][: args.limit]
        meta = {"tipo": "playlist", "id": d.get("id"), "titulo": d.get("title"),
                "canal": d.get("channel") or d.get("uploader"), "url": url}
        m = run(entries, root / d["id"], meta, args.pausa)
    elif args.video:
        url = args.video if args.video.startswith("http") else f"https://www.youtube.com/watch?v={args.video}"
        p = ytdlp("-J", "--no-playlist", url)
        if p.returncode != 0:
            print(f"[ERROR] {p.stderr.strip()[-300:]}")
            return 1
        v = json.loads(p.stdout)
        meta = {"tipo": "video", "id": v["id"], "titulo": v.get("title"), "canal": v.get("channel"),
                "url": url, "descripcion": v.get("description")}
        m = run([{"id": v["id"], "title": v.get("title"), "channel": v.get("channel"), "duration": v.get("duration")}],
                root / (args.nombre or v["id"]), meta, args.pausa)
    else:
        ap.error("--playlist o --video")
    ok = sum(1 for r in m["entradas"] if r["archivo"])
    print(f"[OK] {m['titulo']}: {ok}/{len(m['entradas'])} archivos")
    return 0 if ok == len(m["entradas"]) else 2


if __name__ == "__main__":
    sys.exit(main())
