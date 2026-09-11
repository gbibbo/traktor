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
# Presupuesto duro de GPU (créditos ~ costo_hora * MAX_GPU_MINUTES/60). Se puede sobreescribir
# escribiendo un número de minutos en artifacts/gpu_autorun.max_minutes.
# Por tipo de GPU (créditos/h aprox: H100 4.5, L4 1.7, T4 0.55) para no pasar de ~4 créditos.
gpu_budget_minutes() {
  local name; name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  case "$name" in
    *H100*) echo 45 ;;
    *L4*)   echo 120 ;;
    *T4*)   echo 300 ;;
    *)      echo 45 ;;
  esac
}
MAX_GPU_MINUTES=45
[ -f "$REPO/artifacts/gpu_autorun.max_minutes" ] && MAX_GPU_MINUTES=$(cat "$REPO/artifacts/gpu_autorun.max_minutes")

cd "$REPO" || exit 0
[ -f "$FLAG" ] || exit 0
command -v nvidia-smi >/dev/null 2>&1 || { echo "$(date -Is) no GPU, flag kept" >> "$LOG"; exit 0; }
[ -f "$REPO/artifacts/gpu_autorun.max_minutes" ] || MAX_GPU_MINUTES=$(gpu_budget_minutes)
"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "$(date -Is) torch sees no CUDA; leaving flag, switching back to CPU to stop billing" >> "$LOG"
  "$SYSPY" tools/lightning/switch_machine.py CPU >> "$LOG" 2>&1
  exit 0
}

{
  echo "$(date -Is) === GPU autorun start: $(nvidia-smi -L) | budget ${MAX_GPU_MINUTES} min"
  rm -f artifacts/v4/datasets/$DATASET/embeddings/shards/*
  echo "--- smoke (3 tracks)"
  "$PY" src/v4/pipeline/phase1_extract.py --dataset-name $DATASET --device cuda --max-tracks 3 --checkpoint-every 1
  SMOKE=$?
  if [ $SMOKE -ne 0 ]; then
    echo "$(date -Is) smoke FAILED ($SMOKE); flag kept for diagnosis"
  else
    rm -f artifacts/v4/datasets/$DATASET/embeddings/shards/*
    echo "--- full extraction"
    # timeout: al agotar el presupuesto mata la extracción; los shards checkpointeados se conservan
    timeout --signal=INT --kill-after=60 $((MAX_GPU_MINUTES * 60)) \
      "$PY" src/v4/pipeline/phase1_extract.py --dataset-name $DATASET --device cuda --checkpoint-every 10
    FULL=$?
    [ $FULL -eq 124 ] && echo "$(date -Is) BUDGET HIT: extraction stopped after $MAX_GPU_MINUTES min; partial shards kept"
    echo "--- merge (exit full=$FULL)"
    "$PY" src/v4/pipeline/phase1_merge_shards.py --dataset-name $DATASET
    echo "$(date -Is) merge exit $?"
    [ $FULL -eq 0 ] && rm -f "$FLAG"
  fi
  echo "$(date -Is) switching back to CPU"
  "$SYSPY" tools/lightning/switch_machine.py CPU
} >> "$LOG" 2>&1
