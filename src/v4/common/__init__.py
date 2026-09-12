"""
PURPOSE: Paquete de utilidades comunes de TRAKTOR ML V4.
         Al importarse, fuerza stdout/stderr a UTF-8 para que los prints con
         caracteres no ASCII (flechas →, ✓, acentos) no aborten en consolas
         Windows con codepage cp1252. Es un no-op donde los streams ya son UTF-8
         o no soportan reconfigure (p. ej. algunos entornos de test/captura).
         Portabilidad local-first: permite correr el pipeline en Windows nativo
         sin depender de PYTHONUTF8/PYTHONIOENCODING.
CHANGELOG:
  - 2026-09-12: UTF-8 stdout/stderr al importar (portabilidad Windows/CPU local).
"""
import sys


def _enable_utf8_console() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            enc = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
            if enc != "utf8":
                reconfigure(encoding="utf-8")
        except Exception:
            pass


_enable_utf8_console()
