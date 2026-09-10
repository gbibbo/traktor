# Anotador DJ — herramienta de escucha y anotación

Página HTML autónoma para que el DJ escuche la colección desde su disco local y registre
evidencia de similitud sin servidor ni instalación. Genera los CSV que alimentan la
selección y adaptación de la métrica de similitud, y M3U para cargar en Traktor.

## Archivos

| Archivo | Descripción |
| :--- | :--- |
| `tools/dj_feedback/template.html` | Plantilla de la página (UI + lógica, sin datos) |
| `tools/dj_feedback/build_feedback_page.py` | Genera `dj_feedback.html` embebiendo los nombres de archivo del dataset y la ruta Windows |
| `tools/dj_feedback/dj_feedback.html` | Página generada (243 temas de `test_20`) |
| `playlists/feedback/all_tracks.m3u` | Toda la colección con rutas Windows, para cargar en Traktor |

Regenerar:

```bash
python tools/dj_feedback/build_feedback_page.py --dataset-name test_20
```

## Uso en Windows

1. Copiar `dj_feedback.html` dentro de la carpeta de música (`C:\Música\2020 new - copia`).
   Al estar junto a los mp3, el navegador (Chrome, Edge o Firefox) reproduce el audio con rutas relativas.
   Si se abre desde otra carpeta, usar el botón "Elegir carpeta de música…".
2. Abrir el archivo con doble clic.
3. Las respuestas se guardan en el navegador (localStorage) y se descargan con los botones "Exportar".
4. Cualquier pregunta, grupo o subconjunto se puede exportar como M3U para mezclar en Traktor.

Atajos en la pestaña de tripletas: `1` `2` `3` reproducen ancla, B y C desde el 35 %;
`B`, `C`, `S` responden; espacio pausa. Los botones 10 / 35 / 60 / 85 % saltan a posiciones
típicas de un tema de club.

## Experimentos y CSV exportados

### 1. Tripletas — `tripletas_respuestas.csv`

Pregunta: dado un ancla, ¿cuál de B o C pondrías en la misma carpeta? Es la evidencia relativa
del plan v6 (D3.3 y D4.3). Acepta la cola del proyecto (`triplet_question_queue.csv` o la
plantilla activa) por "Cargar cola CSV…", emparejando por nombre de archivo, o genera una cola
aleatoria reproducible por semilla.

Columnas: `question_id, anchor, candidate_b, candidate_c, answer (B|C|skip), source,
answered_at, listened_anchor_s, listened_b_s, listened_c_s`.

### 2. Grupos con semillas — `grupos_semillas.csv`

Requisito funcional nuevo (2026-09-10): el DJ marca un subconjunto (por ejemplo 200 temas),
fija cuántos grupos quiere (por ejemplo 6) y asigna a mano algunos temas a algunos grupos
(4 al A, 2 al B, 1 al C). El sistema debe completar la asignación de todo el subconjunto.

Columnas: `filename, in_subset (0|1), seed_group, seed_group_name, target_k, exported_at`.

Interpretación para el modelo: cada par dentro del mismo grupo semilla es un must-link, cada par
entre grupos semilla distintos es un cannot-link, `target_k` fija el número de clusters. Esto se
resuelve con clustering con restricciones sobre la métrica de similitud (semillas como centroides
iniciales o propagación de etiquetas sobre el grafo kNN), sin entrenamiento. Un rechazo del
resultado ("estos dos no van juntos") se registra como nueva restricción y produce la siguiente
propuesta.

### 3. Transiciones — `transiciones.csv`

Para el modo en vivo: se elige el tema que suena y se califica cada candidato como buena, regular
o mala mezcla.

Columnas: `now_playing, candidate, rating (good|ok|bad), rated_at, listened_now_s,
listened_candidate_s`.

Cada calificación es un par dirigido con etiqueta; "good" frente a "bad" con el mismo `now_playing`
equivale a una tripleta y sirve para evaluar hit@10 del recomendador de siguiente tema.

## Privacidad

La página contiene solo nombres de archivo, igual que las M3U ya versionadas en `playlists/`.
Los CSV exportados quedan en el equipo del DJ; solo deben entrar al repositorio resúmenes agregados
o copias bajo rutas ignoradas por Git.
