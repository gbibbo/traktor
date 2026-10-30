"""
PURPOSE: Ejecuta spotify_soulseek_orchestrator.py para la app de descargas y permite detenerlo
         con un archivo de señal: cuando aparece STOP_FILE se interrumpe el hilo principal como
         un Ctrl+C, así el orquestador marca el run como interrumpido y cierra sus procesos hijos.
         (En Windows no se puede mandar Ctrl+C a un proceso sin consola compartida.)
         Uso: python run_job.py STOP_FILE <argumentos del orquestador...>
CHANGELOG:
  - 2026-10-30: Creación inicial.
"""
from __future__ import annotations

import _thread
import sys
import threading
import time
from pathlib import Path

DOWNLOAD_DIR = Path(__file__).resolve().parents[1]


def watch_stop_file(stop_file: Path) -> None:
    while not stop_file.exists():
        time.sleep(0.3)
    print("\nDetenido desde la app.", flush=True)
    _thread.interrupt_main()


def main() -> int:
    if len(sys.argv) < 3:
        print("Uso: run_job.py STOP_FILE PLAYLIST_URL [opciones]", file=sys.stderr)
        return 2
    stop_file = Path(sys.argv[1])
    threading.Thread(target=watch_stop_file, args=(stop_file,), daemon=True).start()
    sys.path.insert(0, str(DOWNLOAD_DIR))
    import spotify_soulseek_orchestrator as orchestrator

    sys.argv = [str(DOWNLOAD_DIR / "spotify_soulseek_orchestrator.py"), *sys.argv[2:]]
    try:
        return orchestrator.main()
    except KeyboardInterrupt:  # llegó antes de que el orquestador armara su propio manejo
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
