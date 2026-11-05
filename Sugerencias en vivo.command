#!/bin/bash
# Sugerencias en vivo en Traktor con un botón del controlador (tools/traktor_live/sugerencias_vivo.py), macOS.
# Doble clic en el Finder abre la Terminal. Primero abrir Traktor. La ventana tiene que quedar abierta.
# Guía: docs/SUGERENCIAS_EN_VIVO.md. Acepta las mismas opciones, p. ej. --aprender o --midi K2.
cd "$(dirname "$0")" || exit 1
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
echo "Sugerencias en vivo: Ctrl+C para salir."
"$PY" tools/traktor_live/sugerencias_vivo.py "$@"
status=$?
if [ $status -ne 0 ]; then read -r -p "Enter para cerrar..." _; fi
exit $status
