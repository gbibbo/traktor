# MVP biblioteca completa → Rekordbox + Traktor (2026-09-27)

Qué se construyó y qué se midió al pasar de `test_20` a toda la biblioteca local (`Música/`,
dataset `musica`) en la laptop Windows, en CPU. Comandos: `docs/V4_USAGE.md`, sección 0b.
Decisiones de Gabriel: `docs/DECISIONS.md`, 2026-09-27.

## Biblioteca

| Medida | Valor |
| :--- | ---: |
| Archivos de audio encontrados (sin los `._*` de macOS) | 3086 |
| Temas únicos tras quitar duplicados exactos | 2303 |
| Duplicados descartados (gana la copia fuera de carpetas "copia"/"old N") | 782 |
| Ilegibles (ni libsndfile ni mutagen dan duración) | 1 |
| Temas entre 90 s y 15 min (entran a CLAP y a las playlists) | 2266 |
| BPM en tags / estimado del audio | 1979 / 324 |
| Tonalidad en tags | 1691 |
| Energía de Mixed In Key (comentario "8A - Energy 6") | 1424 |

El estimador de BPM (`src/v4/common/tempo.py`) coincidió con el BPM de los tags a ±1 BPM en
117 de 120 temas al azar; uno de los 3 desacuerdos es un tag a medio tempo (62).

## Representación: CLAP frente a BPM (test_20)

Evaluador `representation_eval.py`, 243 temas de 2020; tripletas uniformes n = 57; coherencia =
pureza de los 5 vecinos contra las 10 carpetas que Gabriel armó a mano en `2020 new` (147 temas;
al azar ≈ 0.11). BPM de los tags.

| Representación | Tripletas uniformes | Pureza kNN carpetas propias |
| :--- | ---: | ---: |
| BPM | 0.632 | 0.204 |
| Tonalidad | 0.491 | 0.109 |
| CLAP | 0.579 | 0.287 |
| CLAP + BPM (peso 0.3) | 0.684 | 0.301 |

Lectura: CLAP no supera al BPM en "qué tocaría a continuación" (diferencia dentro del ruido con
n = 57), pero agrupa por estilo mejor que el BPM. El peso 0.3 del BPM en el agrupamiento es una
decisión de diseño (cajones con tempo coherente); la fila CLAP + BPM se miró sobre los mismos datos
de evaluación, así que es exploratoria, no una mejora validada.

CLAP estaba mal extraído antes de hoy: el procesador recortaba 10 s al azar de cada segmento de
30 s (dos pasadas sobre el mismo segmento daban coseno 0.98). Ahora se usan las 3 ventanas de 10 s.

## Agrupamiento

HDBSCAN sobre CLAP (PCA 50), probado con los primeros 669 temas extraídos, deja ≈ 88 % de ruido y 2-3 grupos gigantes: el espacio no tiene
grupos densos. Se usa Ward con número de grupos fijado: 15 carpetas, 67 playlists (mediana 33
temas). Orden dentro de cada playlist: embedding + BPM + tonalidad + energía, arrancando por la
menor energía. Transiciones consecutivas con |ΔBPM| ≤ 3: 97 % ordenadas frente a 79 % en orden
aleatorio.

## Etiqueta Vocal

CLAP zero-shot (textos "con voces" frente a "instrumental") sobre las 9 ventanas de 10 s del tramo
central. El puntaje separa la carpeta "Vocal" de las otras 9 carpetas propias con AUC 0.906
(18 positivos) y los temas con "feat." en el nombre con AUC 0.73. Umbral 0.15 sobre la media,
calibrado con esa carpeta (exploratorio): en test_20 detecta 15 de 18 temas de "Vocal" y marca el
21 %; en la biblioteca marca 701 de 2266 (31 %), con reparto plausible por género (Tech House, House,
Afro House, Minimal/Tech House ≈ 44-58 %; Techno ≈ 7-8 %). Se escribió " - Vocal" en el comentario de 666 archivos;
no se escribieron 7 WAV (formato) ni 28 archivos de solo lectura. En el umbral hay errores en los
dos sentidos (p. ej. "Scissor Sisters - I Don't Feel Like Dancing" quedó justo por debajo):
`Chequeo Vocal.m3u8` trae 12 temas del borde para decidir el umbral escuchando.

## Export

`rekordbox.xml`, `traktor.nml` y 67 `.m3u8` con 2266 temas; todas las rutas verificadas contra el
disco. Rekordbox y Traktor no están instalados en esta laptop: la importación real queda por
validar (docs/v4/TODO.md 6.7).

## Actualización (03:45, mismo día)

Pedidos de Gabriel aplicados y re-exportado:
- Se borró `Música/Fede` (no es suya): 480 temas que no existían en otras carpetas. Quedan 1823 temas
  únicos, 1786 entre 90 s y 15 min; BPM estimado del audio solo en 37 (casi todos los temas sin BPM
  eran de Fede).
- Umbral Vocal 0.145: 548 de 1786 temas (30.7 %); 542 con la etiqueta escrita (6 WAV sin escribir).
  Se quitó el solo lectura a los 50 archivos del compilado "Best Of Toolroom 2015" (28 eran Vocal).
- Export: 1786 temas, 51 playlists (mediana 34), todas las rutas existen en los tres formatos;
  |ΔBPM| ≤ 3 en el 97 % de las transiciones ordenadas frente al 72 % en orden aleatorio.
- MAEST-HF verificado en "2022 sin clasificar" (9 temas, 6.1 s por tema, determinista, estilos
  coherentes con los tags); corrida completa lanzada con prioridad baja.

## MAEST frente a CLAP (04:10, test_20)

Mismo evaluador y mismas 57 tripletas uniformes; MAEST-HF capa 7, media de tokens y de 3 segmentos.

| Representación | Tripletas uniformes | Pureza kNN carpetas propias | Pureza kNN carpetas v1 |
| :--- | ---: | ---: | ---: |
| BPM | 0.632 | 0.204 | 0.202 |
| CLAP | 0.579 | 0.287 | 0.170 |
| MAEST capa 7 | 0.684 | 0.303 | 0.222 |
| MAEST estilos (400) | 0.684 | 0.229 | 0.258 |

MAEST queda a la par o algo por encima de CLAP; frente al BPM en tripletas la diferencia es +0.053
con IC pareado −0.105 a 0.219: no concluyente con n = 57.

La corrida sobre la biblioteca terminó (1786 temas, sin fallas). Con los mismos parámetros de
agrupamiento, Gabriel comparó en la página de revisión las dos organizaciones (A = MAEST, B = CLAP)
y eligió claramente B (2026-09-29). La comparación no fue del todo a ciegas: B ya la conocía. Queda
CLAP para el MVP; las tripletas no alcanzaban para decidirlo y el juicio escuchando playlists
enteras lo decidió.

## Pendiente

- Gabriel importa y valida en Rekordbox (pendrive) y Traktor.
- MAEST-HF sobre la biblioteca (≈ 6 s por tema en esta CPU: una noche) y comparación con CLAP.
- Ajustar el umbral Vocal con la lista de chequeo.
