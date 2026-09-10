"""
PURPOSE: Generar la página autónoma "Anotador DJ" (tools/dj_feedback/dj_feedback.html) que
         permite al DJ escuchar la colección desde su disco local y registrar tres tipos
         de evidencia sin servidor: tripletas (A: ¿B o C?), agrupación con semillas
         (subconjunto + grupos manuales + K objetivo) y calificación de transiciones.
         Embebe la lista de archivos de audio del dataset y la ruta Windows para las M3U.
         También escribe playlists/feedback/all_tracks.m3u para cargar toda la colección
         en Traktor.
CHANGELOG:
  - 2026-09-10: Creación inicial.
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.v4.common.audio_utils import get_audio_files  # noqa: E402
from src.v4.common.config_loader import load_config  # noqa: E402
from src.v4.common.path_resolver import resolve_dataset_audio_root  # noqa: E402

TEMPLATE = Path(__file__).with_name("template.html")
OUTPUT = Path(__file__).with_name("dj_feedback.html")


def write_m3u(path: Path, windows_dir: str, filenames: list[str]) -> None:
    windows_dir = windows_dir.rstrip("\\/")
    lines = ["#EXTM3U"]
    for name in filenames:
        lines.append(f"#EXTINF:-1,{Path(name).stem}")
        lines.append(f"{windows_dir}\\{name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the DJ feedback HTML page.")
    parser.add_argument("--dataset-name", default="test_20")
    parser.add_argument("--config", default=None)
    parser.add_argument("--windows-audio-dir", default=None,
                        help="Override paths.local_windows_audio_dir from config.")
    parser.add_argument("--m3u-out", default=str(REPO_ROOT / "playlists" / "feedback" / "all_tracks.m3u"))
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else None)
    audio_root = resolve_dataset_audio_root(args.dataset_name, config)
    windows_dir = args.windows_audio_dir or config.get("paths", {}).get("local_windows_audio_dir", "C:\\Music")

    filenames = [p.name for p in get_audio_files(audio_root)]
    if not filenames:
        print(f"[ERROR] No audio files found in {audio_root}")
        return 1

    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("__TRACKS_JSON__", json.dumps(filenames, ensure_ascii=False))
    html = html.replace("__WINDOWS_DIR__", windows_dir.replace('"', "&quot;"))
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"[INFO] Wrote {OUTPUT} with {len(filenames)} tracks from {audio_root}")

    m3u_path = Path(args.m3u_out)
    write_m3u(m3u_path, windows_dir, filenames)
    print(f"[INFO] Wrote {m3u_path} ({len(filenames)} entries, root {windows_dir})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
