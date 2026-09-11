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

## Requisito registrado: agrupación con semillas

El DJ marca un subconjunto (por ejemplo 200 temas), fija cuántos grupos quiere (por ejemplo 6) y
asigna a mano algunos temas a algunos grupos. El sistema debe completar la asignación. Cada par
dentro del mismo grupo semilla es un must-link, cada par entre grupos distintos es un cannot-link.
Se resuelve con clustering con restricciones sobre la métrica de similitud. Por ahora esa
asignación manual se registra en una planilla del DJ con columnas `filename, group`.
