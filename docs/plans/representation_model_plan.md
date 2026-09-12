# Plan: modelo de representación para similitud de temas (2026-09-12)

Estado: aprobado por Gabriel el 2026-09-12 (ver `docs/DECISIONS.md`). Este documento es el
plan operativo vigente; prevalece sobre `v4_implementation_plan.md` y sobre el plan v6 de la
rama `feature/dj-clustering-v1` donde entren en conflicto. Se actualiza al cerrar cada fase.

## Objetivo

Un modelo de representación (embeddings) en el que la distancia entre dos temas refleje si un
DJ los consideraría similares o mezclables. Todo lo demás (carpetas, orden, recomendador, MVP)
se construye encima y se define después. Cada decisión de modelado se toma con una evaluación
fija y reproducible, nunca mirando la salida.

## Evidencia y evaluación (fija, no se toca para entrenar)

| Conjunto | Tamaño | Uso |
| :--- | ---: | :--- |
| Tripletas uniformes del anotador (2026-09-11) | 57 útiles | Accuracy principal; muestreo sin sesgo |
| Tripletas de la rama DJ (2026-05-28, recuperadas) | 37 útiles | Accuracy sobre casos difíciles; reportar por `selection_source` |
| Carpetas de v1 nombradas a mano (`legacy/v2/results/full_collection/clusters_by_group.txt`) | 16 carpetas, 200 temas sin ruido | Coherencia de vecindario: pureza kNN y precisión media de recuperación con "misma carpeta" como relevante |
| BPM de Essentia | 243 temas | Baseline a batir (0.636 sobre las 57 uniformes) |

Reglas: toda comparación entre representaciones se hace sobre exactamente las mismas tripletas,
con bootstrap pareado contra el baseline de BPM; los empates cuentan 0.5; se reporta n por
fuente. Estos conjuntos no se usan para ajustar pesos ni hiperparámetros. Si en el futuro se
ajusta algo contra las tripletas, primero se congela un split y se documenta.

Instrumento: `src/v4/evaluation/representation_eval.py`. Entrada: una o más matrices de
embeddings alineadas por `track_uids.json`. Salida: tabla en consola y JSON en
`artifacts/v4/datasets/<dataset>/evaluation/representations_<fecha>.json`.

## Fases

### Fase 1. Instrumento de evaluación (CPU) — en curso

1. Evaluador de representaciones con tripletas (por fuente y combinadas, bootstrap pareado
   contra BPM, clase mayoritaria como referencia) y coherencia contra las carpetas de v1.
2. Restaurar `features/bpm_key.parquet` de 243 filas con Essentia en WSL.
3. Test unitario con datos sintéticos.

Cierre: el evaluador corre sobre cualquier `.npy` y reproduce los baselines de BPM y clave.

### Fase 2. Tabla de representaciones congeladas (CPU, WSL)

Extraer para los 243 temas, con la misma segmentación (3 x 30 s del tramo central 35 %-65 %):

| Modelo | Origen | Datos de entrenamiento | Variantes |
| :--- | :--- | :--- | :--- |
| Discogs-EffNet bs64 | Essentia (`.pb`) | Discogs, supervisado por género | mezcla; percusivo HPSS |
| MAEST 30 s | Hugging Face `mtg-upf/discogs-maest-30s-pw-129e` | Discogs | mezcla |
| MERT-v1-330M | Hugging Face | Auto-supervisado | capas: última, media últimas 4, capa 7, capa 12; mezcla y HPSS |
| CLAP (laion, htsat-unfused) | Hugging Face | Pares audio-texto | mezcla; permite consultas por texto después |

Salida por modelo: `artifacts/v4/datasets/<dataset>/representations/<nombre>/embeddings.npy`
y `track_uids.json`, más `manifest.json` con modelo, capa, segmentación y versión. Script:
`src/v4/pipeline/extract_representations.py`. Si un modelo no puede correr en WSL con
Python 3.11 (Essentia con TensorFlow no tiene rueda para 3.11), se usa un venv de Python 3.10
solo para ese modelo.

Cierre: informe `docs/reports/representations_<fecha>.md` con la tabla y la elección del
encoder de partida para la fase 4.

### Fase 3. Motor de datos: 1001Tracklists

Requisito estricto de Gabriel. Decisiones ya tomadas el 2026-09-12:

- **Audio**: previews de proveedores (30 s de Deezer sin autenticación; 2 min de Beatport),
  no temas completos de YouTube. Coincide con la ventana de análisis de 30 s.
- **Almacenamiento**: todo en Kaggle como dataset privado. Estimación: 30 000 temas x 30 s ≈ 15 GB.
- **Uso**: pares de temas tocados consecutivos en un set como positivos para entrenamiento
  contrastivo; adyacencia a distancia 2-3 con peso menor. Las tripletas y las carpetas de v1
  quedan como evaluación intocable.

Pendiente de acordar con Gabriel al llegar a esta fase: fuente de los tracklists (scraper,
export, dataset publicado), cuentas y límites de los proveedores de previews, alcance
(DJs, sellos, años, número de sets), y política de matching artista/título/duración.

Componentes a construir: ingesta de tracklists a una tabla `set_id, position, artist, title`;
resolución de cada entrada a un preview con confianza; descarga y empaquetado como dataset
Kaggle; matching de la colección privada contra esa tabla; construcción de pares positivos
con peso por distancia en el set.

### Fase 4. Entrenamiento en Kaggle (GPU gratuita)

Cabeza de proyección o encoder pequeño sobre el mejor encoder congelado de la fase 2,
entrenado con los pares de la fase 3 (pérdida contrastiva, negativos en el batch más
pseudo-negativos difíciles con peso bajo). Selección de hiperparámetros con un split de
sets de 1001Tracklists, nunca con las tripletas. Evaluación final con el instrumento de la
fase 1. Cuaderno reproducible en `tools/kaggle/`.

Cierre: el modelo entrenado supera al mejor encoder congelado y al BPM en las tripletas
uniformes con bootstrap pareado, y no empeora la coherencia contra las carpetas de v1.

### Fase 5. MVP sobre el modelo

Con el espacio de embeddings validado: carpetas (clustering con etiquetas de métrica separadas
de las de export), orden dentro de carpeta, exportación a Traktor y panel de revisión. Se
reutiliza el pipeline V4 de Fases 2-5 corrigiendo los defectos listados en
`docs/reports/scientific_review_2026-09-12.md`, sección 5.

## Cómputo

CPU local y WSL por defecto. Kaggle solo para la fase 4 y, si hiciera falta, para Demucs.
Nada se lanza en Kaggle sin que Gabriel lo pida en la conversación.
