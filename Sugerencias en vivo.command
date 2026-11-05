#!/bin/bash
# Sugerencias en vivo en Traktor con un botón del controlador (tools/traktor_live/sugerencias_vivo.py), macOS.
# Doble clic en el Finder abre la Terminal. Primero abrir Traktor. La ventana tiene que quedar abierta.
# La primera vez crea el entorno de Python en .venv e instala python-rtmidi (el MIDI de macOS).
# Guía: docs/SUGERENCIAS_EN_VIVO.md. Acepta las mismas opciones, p. ej. --aprender o --midi K2.
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
echo "Sugerencias en vivo: Ctrl+C para salir."
.venv/bin/python tools/traktor_live/sugerencias_vivo.py "$@"
status=$?
if [ $status -ne 0 ]; then read -r -p "Enter para cerrar..." _; fi
exit $status
