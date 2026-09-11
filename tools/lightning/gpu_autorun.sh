#!/usr/bin/env bash
# PURPOSE: Ejecutar sin supervisión la extracción de la fase 1 en GPU y devolver el Studio
#          a CPU al terminar, para no gastar créditos si la sesión de Claude se corta.
#          Se invoca desde ~/.lightning_studio/on_start.sh en cada arranque del Studio:
#            - si no hay GPU o no existe el archivo bandera, no hace nada;
#            - si hay GPU y bandera: smoke de 3 temas -> extracción completa -> merge ->
#              borra la bandera -> vuelve la máquina a CPU (reinicia el Studio).
#          Crear la bandera:  touch artifacts/gpu_autorun.flag   (desde la raíz del repo)
#          Log:               artifacts/gpu_autorun.log
# CHANGELOG:
#   - 2026-09-11: Creación inicial.
set -u
REPO=/teamspace/studios/this_studio/traktor
FLAG="$REPO/artifacts/gpu_autorun.flag"
LOG="$REPO/artifacts/gpu_autorun.log"
PY="$REPO/.venv/bin/python"
SYSPY=/home/zeus/miniconda3/envs/cloudspace/bin/python
DATASET=test_20

cd "$REPO" || exit 0
[ -f "$FLAG" ] || exit 0
command -v nvidia-smi >/dev/null 2>&1 || { echo "$(date -Is) no GPU, flag kept" >> "$LOG"; exit 0; }
"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "$(date -Is) torch sees no CUDA; leaving flag, switching back to CPU to stop billing" >> "$LOG"
  "$SYSPY" tools/lightning/switch_machine.py CPU >> "$LOG" 2>&1
  exit 0
}

{
  echo "$(date -Is) === GPU autorun start: $(nvidia-smi -L)"
  rm -f artifacts/v4/datasets/$DATASET/embeddings/shards/*
  echo "--- smoke (3 tracks)"
  "$PY" src/v4/pipeline/phase1_extract.py --dataset-name $DATASET --device cuda --max-tracks 3 --checkpoint-every 1
  SMOKE=$?
  if [ $SMOKE -ne 0 ]; then
    echo "$(date -Is) smoke FAILED ($SMOKE); flag kept for diagnosis"
  else
    rm -f artifacts/v4/datasets/$DATASET/embeddings/shards/*
    echo "--- full extraction"
    "$PY" src/v4/pipeline/phase1_extract.py --dataset-name $DATASET --device cuda --checkpoint-every 20
    FULL=$?
    echo "--- merge (exit full=$FULL)"
    "$PY" src/v4/pipeline/phase1_merge_shards.py --dataset-name $DATASET
    echo "$(date -Is) merge exit $?"
    [ $FULL -eq 0 ] && rm -f "$FLAG"
  fi
  echo "$(date -Is) switching back to CPU"
  "$SYSPY" tools/lightning/switch_machine.py CPU
} >> "$LOG" 2>&1
