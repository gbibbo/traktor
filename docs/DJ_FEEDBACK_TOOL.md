# Anotador DJ — página mínima de escucha y tripletas

Página HTML autónoma, sin servidor. El DJ la abre desde su carpeta de música, responde una cola
fija de tripletas y descarga un CSV. Debajo tiene la lista completa de temas con botón de
reproducir, para escuchar mientras completa cualquier planilla propia (por ejemplo la agrupación
con semillas).

## Archivos

| Archivo | Descripción |
| :--- | :--- |
| `tools/dj_feedback/template.html` | Plantilla (UI + lógica, sin datos) |
| `tools/dj_feedback/build_feedback_page.py` | Genera `dj_feedback.html` con la lista de archivos y una cola fija de tripletas (semilla 42, 60 preguntas) |
| `tools/dj_feedback/dj_feedback.html` | Página generada para `test_20` |
| `playlists/feedback/all_tracks.m3u` | Toda la colección con rutas Windows, para Traktor |

Regenerar (solo si cambia el dataset o se quiere otra cola):

```bash
python tools/dj_feedback/build_feedback_page.py --dataset-name test_20 --seed 42 --n-questions 60
```

## Uso

1. Copiar `dj_feedback.html` dentro de `C:\Música\2020 new - copia` y abrirlo con doble clic.
   Si no está junto a los mp3, la página muestra un botón para elegir la carpeta.
2. Cada pregunta muestra un ancla y dos candidatos con botón de reproducir (arranca al 35 % del tema).
   Responder "B", "C" o saltar. El avance queda guardado en el navegador y se retoma donde quedó.
3. Al terminar, o en cualquier momento, "Descargar respuestas (CSV)".

CSV: `question_id, anchor, candidate_b, candidate_c, answer (B|C|skip), answered_at`.

La cola es fija: la misma pregunta tiene siempre el mismo id, así que las respuestas de varias
sesiones se pueden unir sin ambigüedad.

## Dónde van las respuestas y cómo se usan

Guardar cada CSV exportado en `tools/dj_feedback/answers/` con la fecha en el nombre
(`tripletas_respuestas_YYYY-MM-DD.csv`). Se versionan en Git: solo contienen nombres de archivo,
igual que las M3U ya publicadas. Si una pregunta aparece en varios archivos gana la respuesta más
reciente por `answered_at`.

Semántica real de la respuesta (Gabriel, 2026-09-11): "cuál de los dos tocaría a continuación del
ancla sin salto de estilo", es decir mezclabilidad. La compatibilidad armónica influye aunque se
intente evitar; por eso las carpetas no deben depender de la clave, pero el recomendador sí.

Archivos de respuestas existentes:

| Archivo | Tripletas | Origen y sesgo |
| :--- | ---: | :--- |
| `tripletas_respuestas_2026-09-11.csv` | 60 (57 útiles) | Anotador DJ, cola uniforme semilla 42. Sin sesgo de selección ni de posición (28 B / 29 C) |
| `tripletas_rama_dj_2026-05-28.csv` | 40 (37 útiles) | Recuperadas del archivo de salida de Surrey (rama `feature/dj-clustering-v1`, ids `DJB-Qnnn`). 20 elegidas por vecinos kNN de MERT y 20 por fronteras de las playlists V4_5; en las de frontera B es siempre el vecino del mismo cluster y C el de otro cluster, por eso salen 12 B / 25 C. Sirven como evidencia, pero al evaluar MERT o V4_5 sobre ellas hay circularidad; la columna `selection_source` permite separarlas |

Las 85 preguntas activas de la rama (Q041-Q125) nunca se respondieron: la plantilla en Descargas y la del archivo de Surrey están vacías.

Procesar la evidencia y medir los baselines de clave y BPM:

```bash
python src/v4/pipeline/phase1_extract.py --dataset-name test_20 --device cpu --essentia-only
python src/v4/evaluation/triplet_evidence.py --dataset-name test_20
```

## Requisito registrado: agrupación con semillas

El DJ marca un subconjunto (por ejemplo 200 temas), fija cuántos grupos quiere (por ejemplo 6) y
asigna a mano algunos temas a algunos grupos. El sistema debe completar la asignación. Cada par
dentro del mismo grupo semilla es un must-link, cada par entre grupos distintos es un cannot-link.
Se resuelve con clustering con restricciones sobre la métrica de similitud. Por ahora esa
asignación manual se registra en una planilla del DJ con columnas `filename, group`.
