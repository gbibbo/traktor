# TRAKTOR ML V4 — Jobs Status

Actualizar este archivo después de cada submit o tras verificar resultados.
Monitorear con: `./slurm/tools/on_submit.sh squeue --me`
Logs en: `logs/v4_<jobname>_<jobid>.out`

---

## Secuencia de ejecución V4

```
Phase 0 (local)  ✅ DONE  → catalog.parquet (243 tracks)
                             artifacts/v4/datasets/test_20/catalog.parquet

Smoke test GPU   ✅ PASSED (job 2068466, 2026-03-01)
                   shapes (3,1024) ✅ | finite ✅ | cos_dist=0.091 ✅ | 46.9s

Phase 1 GPU      ✅ DONE (job 2068468, 1h03m) — 239 OK, 1 failed (Lorenzo), 3 skipped*
Phase 1 Merge    ✅ DONE (job 2068566, 0.4s) — N=239 | shapes (239,1024) ✅ | cos=0.121 ✅
                   *3 tracks marcados como done por smoke test pero no en npy final (N=239 aceptable)
Phase 2 cluster  ✅ DONE (job 2068637, 29s) — 8 clusters L1, 53.1% noise, UMAP 2D ✅
                   hash=7b299510 | results_7b299510.parquet (239 rows, umap_x/umap_y ✅)
Phases 3-5       ✅ DONE (job 2068637, 44s total) — names, ordering, M3U export
                   playlists/V4_3/ (239 tracks, 8 grupos L1, 9 subclusters L2 + All_Noise)
Phase 2 cluster  ✅ DONE (job 2068899, 20s) — noise reassignment 1-NN activo
                   hash=f07e0de4 | 8 clusters L1, noise raw=51.0% → final=0.0% (122 reasignados)
                   label_l1_raw + label_l2_raw guardados para diagnóstico
Phases 3-5       ✅ DONE (job 2068899, 31s total) — names, ordering, M3U export
                   playlists/V4_4/ (239 tracks, 8 grupos L1, 10 subclusters L2)
Streamlit UI     ✅ RUNNING — localhost:8501 (datamove1, process background)
                   UI actualizada: muestra "Noise original: 51%" junto a "Noise rate: 0%"
```

### Infraestructura (resuelto 2026-03-01)

Los nodos `a100` (Rocky Linux 8.6) solo tienen Python 3.6 — se usa **Apptainer**:
- SIF: `/mnt/fast/nobackup/scratch4weeks/gb0048/apptainer/pytorch_2.5.1_cu124.sif`
- SIF v2 (preferido): `pytorch_2.7.0_cu128.sif` — actualizar job scripts cuando se migre
- `transformers==4.44.2` pinned (PyTorch 2.5 compat); con SIF v2 se puede usar latest
- pandas, pyarrow, soundfile, tqdm, demucs, essentia: instalados en `python_userbase_sif/`

### Artefactos Phase 1 validados (2026-03-01)

```
artifacts/v4/datasets/test_20/
  catalog.parquet              ✅ (243 tracks en catálogo)
  embeddings/
    mert_perc.npy              ✅ (239, 1024) — embeddings percusivos
    mert_full.npy              ✅ (239, 1024) — embeddings full mix
    track_uids.json            ✅ (239 UIDs, fuente de verdad de alineación)
  features/
    bpm_key.parquet            ✅ (239 rows, BPM 86-167, mediana 123.9)
  run_manifest.json            ✅
```

---

## Jobs enviados

| Job ID  | Job Name          | Estado    | Fecha       | Descripción |
|---------|-------------------|-----------|-------------|-------------|
| 2068338 | v4_smoke_test_gpu | FAILED    | 2026-02-28  | Python 3.11 no disponible en a100 |
| 2068466 | v4_smoke_test_gpu | ✅ PASSED  | 2026-03-01  | Smoke OK con Apptainer (46.9s, 3 tracks) |
| 2068468 | v4_phase1_extract | ✅ DONE    | 2026-03-01  | Phase 1: 239 OK, 1 failed, 1h03m |
| 2068566 | v4_phase1_merge   | ✅ DONE    | 2026-03-01  | Merge shards → mert_perc/full (239,1024) |
| 2068629 | v4_phases2to5     | ❌ FAILED  | 2026-03-01  | hdbscan pip build fail (Python.h missing on debug node) |
| 2068637 | v4_phases2to5     | ✅ DONE    | 2026-03-01  | Phases 2-5 OK (44s) — switched to sklearn.cluster.HDBSCAN |
| 2068899 | v4_phases2to5     | ✅ DONE    | 2026-03-02  | Phases 2-5 OK (31s) — noise reassignment 1-NN, hash=f07e0de4 |

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
