# TRAKTOR ML — Lessons Learned

## Clustering (HDBSCAN / PCA)

### PCA pre-HDBSCAN es necesario para embeddings de alta dimensión
**Contexto:** MERT produce embeddings de 1024 dims. HDBSCAN no estima densidad correctamente en espacios de alta dimensión (curse of dimensionality). Con N=239 tracks y HDBSCAN directamente en 1024D, los mejores resultados eran 2 clusters con 31% noise, o 3 clusters con 80% noise.
**Solución:** Aplicar PCA (sklearn) antes de HDBSCAN. Con pca_dim=50 se retiene 93.7% de varianza y HDBSCAN produce 8 clusters en `test_20` (N=239).
**Parámetros óptimos para test_20 (N=239):** `pca_dim=50, l1_min_cluster_size=6, l1_min_samples=1` → 8 clusters L1, ~51% noise, tamaños [33,24,20,11,8,8,7,6].
**Nota:** Con N~239 no es posible obtener <30% noise con ≥3 clusters simultáneamente. El límite de ruido mejora con más tracks. `mert_full` como L1 no ofrece ventaja sobre `mert_perc`.

### Separabilidad del espacio MERT-v1-330M con test_20
Para N≤300 tracks de techno/tech house, esperar noise rates de 40-60% en L1. Esto es estructural del dataset pequeño, no un bug. Los tracks asignados (~116 de 239) sí tienen estructura musical real.

## Entorno HPC (ver también memory/MEMORY.md)

- El env `traktor_ml` de conda debe crearse con Python 3.11 desde `/user/HS300/gb0048/anaconda3/`.
- Login node (`datamove1`) sí tiene acceso a internet (puede hacer pip install, git pull).
- El comando `source /user/HS300/gb0048/anaconda3/etc/profile.d/conda.sh` es necesario antes de `conda activate`.
- Siempre correr scripts con `PYTHONPATH=/mnt/fast/.../traktor` desde el repo root.
