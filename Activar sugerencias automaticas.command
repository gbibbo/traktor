#!/bin/bash
# Deja las sugerencias en vivo en inicio automático (macOS, LaunchAgent): arrancan solas cada vez que se
# abre Traktor, sin ventanas (tools/traktor_live/sugerencias_servicio.py). Hace falta una sola vez.
# Para quitarlo: correr este archivo con "quitar"; para ver el estado, con "estado".
# Guía: docs/SUGERENCIAS_EN_VIVO.md.
cd "$(dirname "$0")" || exit 1
fail() { echo "[ERROR] $1"; read -r -p "Enter para cerrar..." _; exit 1; }
if [ ! -x .venv/bin/python ]; then
  echo "Primera vez: creo el entorno de Python en .venv ..."
  python3 -m venv .venv || fail "No encontré Python 3: instalar Python 3.11 de python.org y volver a abrir esto."
fi
if ! .venv/bin/python -c "import rtmidi" 2>/dev/null; then
  echo "Instalo python-rtmidi (MIDI de macOS) ..."
  .venv/bin/python -m pip install -q python-rtmidi==1.5.8 || fail "No pude instalar python-rtmidi (¿sin internet?)."
fi
.venv/bin/python tools/traktor_live/sugerencias_servicio.py "${1:-instalar}"
read -r -p "Enter para cerrar..." _
