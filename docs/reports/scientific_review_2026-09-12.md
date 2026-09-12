# Revisión científica de las versiones de TRAKTOR ML (2026-09-12)

Contexto: Surrey HPC y Lightning AI ya no existen como recursos. El trabajo sigue en CPU local
y, si hace falta GPU, en trabajos gratuitos de Kaggle. Antes de continuar había que decidir
sobre qué versión se construye y qué ideas de versiones anteriores merecen rescatarse.

Método: lectura completa de `legacy/v1`, `legacy/v2`, `legacy/v3`, `src/v4` (rama `main`,
HEAD `82f0951`) y de la rama `origin/feature/dj-clustering-v1` (plan v6, HEAD `d0f7122`),
más un cruce empírico nuevo entre las 60 tripletas del anotador DJ y los artefactos de
clustering versionados en el repo (`src/v4/evaluation/legacy_crosscheck.py`). Todas las
afirmaciones sobre código se verificaron en el fuente; los números se recalcularon aquí.

## 1. Veredicto

1. **La versión "más avanzada" (rama `feature/dj-clustering-v1`) no es la más robusta.** Es la
   mejor en ingeniería de proceso (plan preregistrado, 239 tests, trazabilidad), pero su
   resultado empírico es nulo: la mejor accuracy de tripletas (0.622 sobre 37) no supera a la
   clase mayoritaria (25/37 = 0.676), no es distinguible del azar (p ≈ 0.09 sin corregir por
   203 configuraciones evaluadas), y su evidencia está sesgada por construcción. Las fases
   D5-D7 (composite, splits, clustering solapado, export) no están implementadas. Además
   perdió BPM y tonalidad analíticos que `main` sí tiene.
2. **`main` (V4 + pivote de septiembre) es la base más sólida para continuar**: pipeline
   completo de extremo a extremo, exportación a Traktor funcionando, regla armónica formalizada,
   y sobre todo la única evidencia humana limpia del proyecto: 60 tripletas con muestreo
   uniforme (semilla 42), semántica declarada (mezclabilidad) e independiente de cualquier
   modelo. Pero su espacio de representación (MERT, última capa) nunca fue evaluado contra
   esa evidencia, y tiene varios errores de features y de evaluación (sección 5).
3. **La representación es el eslabón débil, no el clustering.** El cruce empírico (sección 3)
   muestra que el espacio de legacy v1 (Discogs-EffNet + MAEST, modelos supervisados con la
   taxonomía pública de Discogs) predice las elecciones del DJ mejor que MERT y mejor que el
   BPM, incluso a través de una proyección UMAP 2D muy degradada. Es la señal más fuerte que
   hay en el repositorio y no la produjo ninguna de las dos líneas activas.

Recomendación: continuar en `main`, tratar la rama DJ como cantera de ideas (sección 6), y
volver a poner Discogs-EffNet y MAEST como candidatos de representación al mismo nivel que
MERT, evaluados todos contra las mismas 57 tripletas con un test pareado contra el BPM.

## 2. Evidencia humana disponible

| Fuente | Ítems | Diseño | Estado |
| :--- | ---: | :--- | :--- |
| `tools/dj_feedback/answers/tripletas_respuestas_2026-09-11.csv` | 60 (57 útiles, 3 saltadas) | Muestreo uniforme de la colección, semilla 42, sin dependencia de ningún modelo | Limpia. 28 B / 29 C: sin sesgo de posición |
| Rama DJ, `manual_triplets.csv` (en el archivo de salida de Surrey) | 40 (37 útiles) | 20 por kNN de `mert_full` + 20 por fronteras de V4_5, con B siempre el vecino intra-cluster y C el cross-cluster | Sesgada: 12 B / 25 C; circular respecto al sistema evaluado |
| Rama DJ, 85 preguntas activas Q041-Q125 | 0 respondidas | Seleccionadas por desacuerdo entre los 3 mejores configs del propio sweep | Sin responder; si se respondieran, cada config acertaría ~50 % por construcción |
| `legacy/v2/results/full_collection/clusters_by_group.txt` | 16 clusters, 243 temas | Nombres de estilo escritos a mano por Gabriel tras escuchar cada cluster de v1 (feb 2025) | Gold parcial olvidado. 15/16 clusters resultaron nombrables |

Conclusión: las 57 tripletas de `main` son el conjunto de evaluación. Con n = 57 el intervalo
de confianza de una accuracy es de ±0.12, suficiente para descartar modelos claramente malos
y para comparaciones pareadas, no para elegir entre modelos cercanos. Se acepta como
limitación conocida: el plan es compensar con automatización, no con más anotación.

## 3. Cruce empírico: ¿qué artefacto predice las elecciones del DJ?

Predictor "distancia": el candidato más cercano al ancla en las coordenadas guardadas.
Predictor "misma carpeta": gana el candidato que comparte carpeta con el ancla; si ambos o
ninguno la comparten, empate (0.5). Intervalos por bootstrap de 2000 remuestreos.

| Artefacto | Representación | n | Accuracy | Nota |
| :--- | :--- | ---: | ---: | :--- |
| legacy v1, distancia UMAP-2D | Discogs-EffNet 1280 + MAEST 768, mezcla completa | 57 | **0.719** (0.60-0.82) | Proyección 2D de feb 2025; sin fuga posible |
| legacy v2, distancia UMAP-2D | Discogs-EffNet sobre stem de batería (Demucs) | 55 | 0.673 (0.55-0.80) | |
| BPM solo (informe 2026-09-11) | Essentia RhythmExtractor, equivalencia 0.5x/1x/2x | 55 | 0.636 (0.50-0.75) | Listón a batir |
| Rama DJ, mejor config del sweep | MERT concat full+perc, UMAP-15 | 37 | 0.622 | Otras tripletas; no comparable directamente |
| genre_discogs400, similitud top-3 | Clasificador de género público | 55 | 0.618 | 6 empates |
| Clave sola (informe 2026-09-11) | Regla armónica aprobada | 55 | 0.536 | 19 empates |
| Azar | | | 0.500 | |

Predictor "misma carpeta" (solo cuenta cuando exactamente un candidato comparte carpeta):

| Carpetas | Decisivas | Acierto en decisivas | Global con empates |
| :--- | ---: | ---: | ---: |
| v1 (EffNet+MAEST, nombradas a mano) | 10 / 57 | 0.900 | 0.570 |
| v2 L1 (EffNet drums) | 7 / 55 | 0.857 | 0.545 |
| V4_5 L1 (MERT perc, PCA50+HDBSCAN+1NN) | 11 / 54 | 0.636 | 0.528 |
| V4_5 L2 (MERT full) | 6 / 54 | 0.667 | 0.519 |

Lectura. Con las cautelas de n pequeño y de que las coordenadas 2D son un proxy pobre del
espacio original, hay una ordenación consistente: EffNet/MAEST > BPM ≈ MERT > género > clave.
Que una proyección 2D de EffNet+MAEST alcance 0.72 sugiere que el espacio completo podría ir
más alto. MERT en cambio, ya sea en la rama DJ o en las carpetas de V4_5, no supera el BPM.
Esto es coherente con lo que el plan v6 apuntaba como hipótesis ("V2 parece haber producido
una organización subjetivamente mejor que V4") y que nunca se comprobó porque la rama
prohibió recomputar V2.

Estos modelos de Essentia son exactamente una estrategia de datos públicos: están entrenados
con etiquetas de género de Discogs. Son pequeños (18 MB y 300 MB), corren en CPU en segundos
por tema, y no necesitan Kaggle.

## 4. Revisión por versión

### legacy v1 (feb 2025)
Discogs-EffNet (mean-pool, tema completo) y MAEST (primeros 30 s, CLS), 16 kHz, concatenados
tras L2; UMAP 2D y HDBSCAN sobre las 2 coordenadas; sin jerarquía, nombres, BPM ni clave.
Debilidad seria: clusterizar sobre UMAP 2D. Fortaleza: es la única corrida con veredicto
humano por cluster (`clusters_by_group.txt`), y su espacio es el que mejor predice las
tripletas (sección 3).

### legacy v2 (feb 2025)
Demucs en memoria a 44.1 kHz, EffNet sobre stem de batería (L1) y sobre mezcla (L2),
jerarquía drum-first, naming por votación de `genre_discogs400` con desambiguación.
Sin segmentación (promedia intro, breakdown y drop). 16 clusters L1 con 18 % de ruido y
cola larga de clusters de 2-5 temas. Los votos de género son de baja confianza (42.6 % de
los temas con confianza < 0.3) y 8 etiquetas cubren el 98 % de la colección. Velocidad medida:
15 s por tema en A100 para Demucs + 2 EffNet.

### legacy v3 (feb 2026)
MERT-v1-330M a 24 kHz, segmentos de 5 s descartando 45 s de intro y outro, **media de las
últimas 4 capas** (mejor fundamentado que la última capa sola), PCA-128 persistida con
varianza explicada, interfaz `BaseClusterer` con KMeans, aglomerativo y HDBSCAN, métricas
internas (silhouette, Calinski-Harabasz, Davies-Bouldin) con exclusión de ruido, UI con
sliders. Es la versión con mejor diseño de software para experimentar. Sin embargo no hay ni
un solo resultado guardado: cero números.

### V4 en `main` (marzo 2026 + pivote de septiembre)
Ver sección 5 para los defectos verificados. Fortalezas: pipeline completo con reentrancia y
manifiestos; contratos de sample rate; UMAP excluido del clustering; `label_l1_raw`
conservado; regla armónica única y testeada; baselines simples publicados antes que el
modelo y resultado negativo (la clave no explica las elecciones) documentado.

### Rama `feature/dj-clustering-v1` (mayo 2026)
Fortalezas: plan preregistrado con política anti-fuga, tres políticas de ruido separando
etiquetas de métrica y de export, `confidence_limited_1nn` (umbral coseno 0.30), identidad
por hash de contenido con duplicados restaurados solo en export, L2 por segmento antes de
promediar, `mert_concat` con L2 por componente, diagnósticos de cluster, diseño de
`bridge_score` en percentiles. Debilidades verificadas en el código:

- La accuracy de tripletas se calcula sobre la matriz transformada, no sobre las etiquetas
  (`scripts/dj_clustering/run_similarity_sweep.py:161-165`). El leaderboard ordena
  representaciones, y configs con 100 % de ruido pueden encabezarlo.
- El eje `normalization` del sweep es un no-op: los embeddings ya se guardan normalizados
  (`configs/dj_clustering/features.yaml:68`), así que la mitad de las 2304 configs son
  duplicados.
- En las tripletas de frontera B es siempre el vecino intra-cluster y C el cross-cluster
  (`src/dj_clustering/triplets.py:355-366`): sesgo de posición confundido con V4.
- BPM solo de tags ID3 (desviación 5.3 BPM) y tonalidad ausente por no ejecutar Essentia.
- Tabla de 30 135 pares y perfiles ponderados sin ningún consumidor; módulo 1001Tracklists
  completo con cero filas de entrada (`no_usable_source`).
- Régimen 2 requiere 200 tripletas o 100 más 1000 positivos de 1001Tracklists: inalcanzable
  con la estrategia de datos actual.

## 5. Defectos verificados en V4 (`main`) que hay que corregir antes de medir nada

1. `beat_confidence` no es una confianza. `src/v4/pipeline/phase1_extract.py:99-100` toma
   la quinta salida de `RhythmExtractor2013`, que son los intervalos entre beats en segundos,
   y guarda su media (~0.47 s) como confianza. La puerta beat-aware queda convertida en un
   test de tempo < 120 BPM.
2. Segmentación de temas cortos. En `src/v4/common/audio_utils.py:186` el paso entre
   segmentos es `(zone_len - seg_len) // (n - 1)`; con la zona central de 30 % y segmentos de
   30 s, cualquier tema de menos de 100 s produce pasos negativos y segmentos casi idénticos.
3. PCA en L2 nunca se aplica. `src/v4/pipeline/phase2_cluster.py:244` exige al menos
   `2 * pca_dim = 100` temas por cluster L1; los clusters reales tienen entre 6 y 33, así que
   todos los subclusters L2 corren HDBSCAN en 1024 dimensiones, el régimen que el propio
   addendum declara inservible.
4. El ruido reportado es 0 % por construcción. `src/v4/evaluation/eval_runner.py:58-64`
   lee `label_l1` (post 1-NN) y nunca `label_l1_raw`; `composite_score` es 0.0 sin dev set.
5. La evaluación del ordenamiento es circular: `test_transition_score_vs_random` compara el
   optimizador voraz contra su propia función objetivo.
6. Sub-scores del ordenamiento con escalas distintas: el coseno de MERT vive en un rango
   estrecho y el BPM en [0, 1] relativo al cluster, así que los pesos 0.5/0.3/0.2 no son los
   efectivos y los scores no son comparables entre clusters. La equivalencia 0.5x/2x de tempo
   existe en el baseline y no en el ordenador.
7. Estadística de las tripletas: intervalo de Wilson sobre un conteo redondeado, empates con
   varianza cero tratados como Bernoulli, comparación de baselines por solapamiento de
   intervalos en lugar de test pareado.
8. MERT usa solo `last_hidden_state` (`src/v4/common/embedding_utils.py:74-76`) pese a pedir
   todos los estados ocultos; v3 ya usaba la media de las últimas 4 capas.
9. El hash de configuración de Phase 2 no incluye modelo, capa, segmentación ni fuente de
   percusión (Demucs vs HPSS), así que dos parquets con el mismo hash pueden venir de
   features incompatibles.
10. Hiperparámetros de clustering ajustados sobre la salida con features de 4 x 5 s y nunca
    re-ajustados tras pasar a 3 x 30 s.

## 6. Qué rescatar de cada versión

| Idea | Origen | Coste | Por qué |
| :--- | :--- | :--- | :--- |
| Discogs-EffNet y MAEST como representaciones candidatas | v1, v2 | Bajo (CPU, modelos pequeños) | Mejor señal empírica del repo; datos públicos vía pesos |
| Evaluar el espacio de embeddings contra las tripletas con test pareado vs BPM | nuevo | Bajo | Hoy nada evalúa MERT; es lo que hace falsable el proyecto |
| Media de las últimas 4 capas de MERT o barrido de capas | v3 | Bajo | Última capa es la peor opción documentada |
| Etiquetas de métrica separadas de etiquetas de export; `confidence_limited_1nn` | rama DJ | Bajo | Deja de reportar 0 % de ruido |
| L2 por segmento antes de promediar; `mert_concat` con L2 por componente | rama DJ | Bajo | Evita que un segmento fuerte domine |
| Métricas internas silhouette / CH / DB con exclusión de ruido | v3 | Bajo | Permite comparar configs sin ground truth |
| Diagnósticos de cluster (mayor cluster, singletons, ruido crudo) | rama DJ | Bajo | `main` no reporta ninguno |
| Factory de clusterers (KMeans, aglomerativo, HDBSCAN) | v3, rama DJ | Medio | Convierte la elección de HDBSCAN en un experimento |
| Nombres de cluster por votación de `genre_discogs400` como fallback | v2 | Medio | Hoy sin metadata externa las carpetas se llaman A, B, C |
| `bridge_score` en percentiles y `weak_primary` | rama DJ (solo plan) | Medio | Buena idea, implementar en `main` donde el export existe |
| Identidad por hash de contenido con duplicados restaurados en export | rama DJ | Medio | Robustez ante cambios de ruta |
| Tabla de pares, perfiles ponderados, módulo 1001Tracklists, grid de 2304 | rama DJ | No portar | Sin consumidor, sin fuente, sobreajuste por selección |

## 7. Estado real de la estrategia con datos públicos

Lo que existe en el repositorio:

- Plan v6: 1001Tracklists como evidencia débil (nunca feature en Régimen 1, confianza ≥ 0.7,
  fingerprint preferido); datasets de género externos solo como comprobación; pseudo-negativos
  por kNN con peso bajo para Régimen 2. En la práctica no se encontró ninguna fuente utilizable
  de 1001Tracklists (`reports/dj_clustering/1001_matching_report.md` en la rama).
- V4: metadata opcional por CSV tipo Beatport, solo para catálogo y nombres. Nunca usada.
- legacy v1/v2: modelos entrenados con Discogs (EffNet, MAEST, genre_discogs400). Es la única
  forma de conocimiento público que produjo resultados, y quedó fuera de V4.
- El documento `docs/plans/dj_music_clustering_system_objectives_and_strategies_v9_1.md`,
  que el plan v6 obliga a leer y que debía contener la especificación de estrategia, **no
  existe** en ninguna rama, en ningún commit ni en el disco local. Si hay una versión fuera
  del repo hay que instalarla antes de decidir la estrategia de automatización.

Con la evidencia humana fijada en 57 tripletas, la vía de automatización viable en CPU es:
representaciones supervisadas con datos públicos (EffNet, MAEST, y opcionalmente CLAP para
descripciones textuales), pseudo-etiquetas de género de `genre_discogs400` como comprobación
de coherencia de carpetas, y métricas internas para comparar configuraciones. Régimen 2
(cabeza de proyección entrenada con tripletas) queda descartado por evidencia insuficiente,
como ya fijaba el propio plan v6.

## 8. Plan propuesto, en orden

1. Corregir en `main` los defectos 1 a 4 y 7 de la sección 5, y reportar ruido crudo.
2. Extender `triplet_evidence.py` para evaluar espacios de embeddings (coseno) con bootstrap
   pareado contra el BPM. Es el instrumento de medida de todo lo demás.
3. Restaurar `bpm_key.parquet` de 243 filas con Essentia en WSL (40 min de CPU).
4. Extraer Discogs-EffNet y MAEST para los 243 temas en CPU (Essentia TensorFlow en WSL) con
   la misma segmentación de 3 x 30 s central, y evaluarlos con (2). Si confirman el 0.72 o
   más, son la representación por defecto.
5. Extraer MERT en CPU con HPSS y barrido de capas (última, últimas 4, capa 7) y evaluarlo
   con el mismo instrumento. Solo si MERT o Demucs resultan necesarios se configura Kaggle.
6. Portar de la rama DJ las etiquetas de métrica/export y los diagnósticos; de v3 las
   métricas internas y la factory de clusterers. Re-ajustar hiperparámetros con estabilidad
   por remuestreo, no mirando la salida.
7. Reconstruir el gold parcial de v1 (`clusters_by_group.txt`) como conjunto de evaluación
   versionado de coherencia de carpetas, complementario a las tripletas.
8. Solo entonces: Fases 3-5, nombres por votación de género, y export.

Reproducir la sección 3:

```bash
python src/v4/evaluation/legacy_crosscheck.py
```
