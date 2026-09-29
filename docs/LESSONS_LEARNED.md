# TRAKTOR ML — Lessons Learned

## Clustering (HDBSCAN / PCA)

### PCA pre-HDBSCAN es necesario para embeddings de alta dimensión
**Contexto:** MERT produce embeddings de 1024 dims. HDBSCAN no estima densidad correctamente en espacios de alta dimensión (curse of dimensionality). Con N=239 tracks y HDBSCAN directamente en 1024D, los mejores resultados eran 2 clusters con 31% noise, o 3 clusters con 80% noise.
**Solución:** Aplicar PCA (sklearn) antes de HDBSCAN. Con pca_dim=50 se retiene 93.7% de varianza y HDBSCAN produce 8 clusters en `test_20` (N=239).
[REEMPLAZADO 2026-09-29 por DECISIONS 2026-09-29: la organización vigente es CLAP + BPM (peso 0.3) con Ward; en CLAP, HDBSCAN deja casi todo como ruido (V4_USAGE, sección 0b). Estos parámetros valen solo para MERT + HDBSCAN.] **Parámetros óptimos para test_20 (N=239):** `pca_dim=50, l1_min_cluster_size=6, l1_min_samples=1` → 8 clusters L1, ~51% noise, tamaños [33,24,20,11,8,8,7,6].
**Nota:** Con N~239 no es posible obtener <30% noise con ≥3 clusters simultáneamente. El límite de ruido mejora con más tracks. `mert_full` como L1 no ofrece ventaja sobre `mert_perc`.

### Separabilidad del espacio MERT-v1-330M con test_20
Para N≤300 tracks de techno/tech house, esperar noise rates de 40-60% en L1. Esto es estructural del dataset pequeño, no un bug. Los tracks asignados (~116 de 239) sí tienen estructura musical real.

## Detección de voz

### Silero VAD no detecta canto sobre la mezcla completa
**Contexto (2026-09-29):** Silero VAD v5.1.2 (TorchScript, 16 kHz) sobre el tema entero, en 63 temas de
`musica`: en las carpetas «Vocal» de Gabriel (23 temas, todos con voz) encontró voz en >= 1 % del tema
solo en el 13 % y nunca llegó al 5 %; CLAP zero-shot marcó el 87 %. Es un detector de habla entrenado
para ignorar música, y el canto sobre la base cuenta como música.
**Regla:** no usarlo sobre la mezcla. Solo tendría sentido sobre la voz separada (Demucs).

### CLAP zero-shot sobre 90 s del medio pierde voces
**Contexto (2026-09-29):** con 3 tramos de 30 s del medio (≈ 20 % de un tema de 7 min) el detector marcó
55 % de 20 temas con «feat.»; con ventanas de 10 s cada 20 s sobre todo el tema, 75 %, y en 20 temas
Dub/Instrumental bajó de 30 % a 25 %. Muestras chicas (etiquetas débiles: «feat.» no garantiza voz
audible y un dub puede tener recortes de voz). Ningún umbral sobre los 90 s arreglaba las dos cosas.
**Regla:** para la voz, mirar el tema entero antes de ajustar el umbral.

## Entorno HPC (ver también memory/MEMORY.md)

[REEMPLAZADO 2026-09-12 por DECISIONS 2026-09-12, punto 4: Surrey HPC ya no existe; se trabaja en CPU local (AGENTS.md). Se conserva como registro.]

- El env `traktor_ml` de conda debe crearse con Python 3.11 desde `/user/HS300/gb0048/anaconda3/`.
- Login node (`datamove1`) sí tiene acceso a internet (puede hacer pip install, git pull).
- El comando `source /user/HS300/gb0048/anaconda3/etc/profile.d/conda.sh` es necesario antes de `conda activate`.
- Siempre correr scripts con `PYTHONPATH=/mnt/fast/.../traktor` desde el repo root.
