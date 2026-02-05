# TRAKTOR ML - Lecciones Aprendidas

Documento centralizado de lecciones aprendidas durante el desarrollo del proyecto.
Antes de agregar una nueva entrada, buscar si existe una similar y expandirla.

---

## Infraestructura HPC (Surrey)

### Python 3.11 solo disponible en nodos A100

**Fecha:** 2025-02-05
**Contexto:** Jobs de Phase 2 fallaban en partición `2080ti`

**Problema:**
- Nodos `2080ti` tienen Python 3.6.8 por defecto
- `/usr/bin/python3.11` no existe en esos nodos
- Dependencias modernas (tqdm, umap-learn) no funcionan con Python 3.6

**Solución:**
- Usar siempre partición `a100` con `--nodelist=aisurrey26` para jobs que requieren Python 3.11
- O usar Conda environment si se necesita otra partición

**Archivos afectados:**
- `slurm/jobs/v2/v2_phase2.job`

---

### Partición "cpu" no existe

**Fecha:** 2025-02-05
**Contexto:** Intento de enviar job CPU-only

**Problema:**
- El cluster Surrey HPC no tiene partición llamada "cpu"
- `sbatch: error: invalid partition specified: cpu`

**Solución:**
- Usar partición con GPU disponible (`a100`, `2080ti`, `3090`, etc.)
- Los jobs CPU-only funcionan en cualquier partición, solo no usan la GPU

**Particiones disponibles:** `debug`, `2080ti`, `3090`, `3090_risk`, `a100`, `rtx8000`, `rtx5000`

---

### Rutas absolutas requeridas para sbatch via SSH

**Fecha:** 2025-02-05
**Contexto:** `on_submit.sh` ejecuta comandos via SSH a `aisurrey-submit01`

**Problema:**
- `sbatch slurm/jobs/v2/v2_phase1.job` falla con "Unable to open file"
- El path relativo no funciona porque SSH aterriza en un directorio diferente

**Solución:**
- Siempre usar rutas absolutas:
  ```bash
  ./slurm/tools/on_submit.sh sbatch /mnt/fast/nobackup/users/gb0048/traktor/slurm/jobs/v2/v2_phase1.job
  ```

---

## Audio Processing

### soundfile + torchaudio.transforms.Resample como alternativa a FFmpeg

**Fecha:** 2025-02-04
**Contexto:** Smoke test de Demucs fallaba por falta de FFmpeg

**Problema:**
- `torchaudio.load()` requiere backend FFmpeg para MP3
- FFmpeg no está instalado en nodos HPC

**Solución:**
- Usar `soundfile` para cargar audio (soporta WAV, FLAC, OGG)
- Para MP3: primero convertir a WAV o usar `audioread` como fallback
- Implementado en `scripts/common/demucs_utils.py`

---

## Pipeline V2

### Velocidad real vs estimada

**Fecha:** 2025-02-05
**Contexto:** Phase 1 completó en 1 hora vs 5-6 horas estimadas

**Datos reales:**
- 242 tracks procesados en 1h 02min
- ~15 segundos por track (Demucs + 2x Essentia embeddings)
- GPU: NVIDIA A100-SXM4-80GB

**Notas:**
- La estimación original era muy conservadora
- El procesamiento on-the-fly en GPU es eficiente
- Checkpointing cada 10 tracks no añade overhead significativo

---

### TensorflowPredict2D requiere nodo input explícito para genre_discogs400

**Fecha:** 2025-02-05
**Contexto:** Clasificación de géneros fallaba con "not a valid node name"

**Problema:**
- `es.TensorflowPredict2D(graphFilename=..., output="PartitionedCall:0")` fallaba
- Error: `'model/Placeholder' is not a valid node name of this graph`
- El modelo `genre_discogs400` usa nombres de nodos diferentes al default

**Solución:**
- Especificar explícitamente el nodo de entrada:
  ```python
  genre_model = es.TensorflowPredict2D(
      graphFilename=str(genre_model_path),
      input="serving_default_model_Placeholder",  # <- REQUERIDO
      output="PartitionedCall:0"
  )
  ```

**Archivos afectados:**
- `scripts/common/embedding_utils.py` → función `extract_genre_predictions()`

---

## Visualización

### Separar generación de visualización del pipeline principal

**Fecha:** 2025-02-05
**Contexto:** Iterar sobre visualización requería re-ejecutar todo phase2_analysis.py

**Problema:**
- `phase2_analysis.py` generaba `visualization.html` como parte del pipeline
- Cualquier cambio a la visualización requería re-ejecutar clustering y clasificación
- Ciclo de iteración lento para ajustes de UI/UX

**Solución:**
- Crear script separado `scripts/local/generate_visualization.py`
- Lee CSVs intermedios ya generados (level1_clusters.csv, level2_clusters.csv, etc.)
- Permite regenerar visualización en segundos sin re-procesar datos
- Sin dependencias externas (solo stdlib de Python)

**Beneficios:**
- Iteración rápida sobre cambios visuales
- Puede ejecutarse localmente (Windows) o en HPC
- CSVs como "contrato" entre procesamiento y visualización

**Archivos:**
- `scripts/local/generate_visualization.py` - Nuevo script de visualización
- `results/v2_hierarchy/*.csv` - Datos intermedios (input)
- `results/v2_hierarchy/visualization.html` - Output regenerable

---

## Changelog

| Fecha | Cambio |
|-------|--------|
| 2025-02-05 | Añadida lección sobre separación de visualización |
| 2025-02-05 | Añadido fix TensorflowPredict2D para genre_discogs400 |
| 2025-02-05 | Creación inicial con lecciones de sesión V2 |
