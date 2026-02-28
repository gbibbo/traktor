# TRAKTOR ML V4 — Jobs Status

Actualizar este archivo después de cada submit o tras verificar resultados.
Monitorear con: `./slurm/tools/on_submit.sh squeue --me`
Logs en: `logs/v4_<jobname>_<jobid>.out`

---

## Secuencia de ejecución V4

```
Phase 0 (local)  ✅ DONE  → catalog.parquet (243 tracks)
                             artifacts/v4/datasets/test_20/catalog.parquet

Smoke test GPU   🔄 RUNNING/PENDING
Phase 1 GPU      ⏳ PENDING (enviar tras smoke test OK)
Phase 1 Merge    ⏳ PENDING (enviar tras Phase 1 completa, si hubo sharding)
Phases 2-5       ⏳ PENDING (enviar tras merge/embeddings validados)
```

---

## Jobs enviados

| Job ID  | Job Name          | Estado    | Fecha       | Descripción |
|---------|-------------------|-----------|-------------|-------------|
| 2068338 | v4_smoke_test_gpu | PD→RUN    | 2026-02-28  | Smoke test: 3 tracks, a100, 30min |

---

## Comandos útiles

```bash
# Ver queue
./slurm/tools/on_submit.sh squeue --me

# Ver estado detallado de un job
./slurm/tools/on_submit.sh scontrol show job <JOBID>

# Ver log del smoke test
tail -f logs/v4_smoke_test_gpu_2068338.out

# Cancelar job
./slurm/tools/on_submit.sh scancel <JOBID>

# Enviar Phase 1 completa (tras smoke test OK)
./slurm/tools/on_submit.sh sbatch /mnt/fast/nobackup/users/gb0048/traktor/slurm/jobs/v4/phase1_extract.job test_20

# Enviar Phase 1 Array (4 shards en paralelo, alternativa)
./slurm/tools/on_submit.sh sbatch /mnt/fast/nobackup/users/gb0048/traktor/slurm/jobs/v4/phase1_extract_array.job test_20

# Merge tras Phase 1
./slurm/tools/on_submit.sh sbatch /mnt/fast/nobackup/users/gb0048/traktor/slurm/jobs/v4/phase1_merge.job test_20
```

---

## Artefactos esperados tras Phase 1 completa

```
artifacts/v4/datasets/test_20/
  catalog.parquet              ✅ (243 tracks)
  ingest_report.json           ✅
  embeddings/
    mert_perc.npy              ⏳ (N, 1024) — embeddings percusivos (drums stem)
    mert_full.npy              ⏳ (N, 1024) — embeddings full mix
    track_uids.json            ⏳ lista de track_uids en mismo orden que .npy
    shards/                    ⏳ archivos temporales de shard
  features/
    bpm_key.parquet            ⏳ BPM, key, beat_confidence por track
  run_manifest.json            ⏳ metadata de corrida (git, config_hash, etc.)
  logs/
    phase0_ingest_*.jsonl      ✅
    phase1_extract_*.jsonl     ⏳
```

---

## Verificación post-Phase 1

```python
import numpy as np, json
from pathlib import Path

artifacts = Path('artifacts/v4/datasets/test_20')
perc = np.load(artifacts / 'embeddings/mert_perc.npy')
full = np.load(artifacts / 'embeddings/mert_full.npy')
with open(artifacts / 'embeddings/track_uids.json') as f:
    uids = json.load(f)

print(f'mert_perc: {perc.shape}, finite: {np.isfinite(perc).all()}')
print(f'mert_full: {full.shape}, finite: {np.isfinite(full).all()}')
print(f'BPMs:'); import pandas as pd; df = pd.read_parquet(artifacts / 'features/bpm_key.parquet'); print(df['bpm'].describe())
```

Checks:
- [ ] Shape (N, 1024) donde N = tracks OK (esperado ~242, 1 archivo malo conocido)
- [ ] `np.isfinite(perc).all()` y `np.isfinite(full).all()`
- [ ] BPMs en rango 100-160
- [ ] `mert_perc` y `mert_full` son distintos (cosine distance media > 0.05)

---

## Próxima tarea de código (Bloque 3)

Cuando los embeddings estén validados, continuar con:

**Tarea 3.1** — `src/v4/pipeline/phase2_cluster.py`
  - HDBSCAN L1 sobre mert_perc
  - HDBSCAN L2 sobre mert_full dentro de cada cluster L1
  - UMAP 2D para visualización
  - Output: `clustering/results_<hash>.parquet`

**Tarea 3.2** — `src/v4/evaluation/metrics.py` + `eval_runner.py`
  - ARI, NMI, Recall@k, noise_rate, transition_score

Referencia: tareas 3.1 y 3.2 en `v4_implementation_plan.md` (Bloque 3).
