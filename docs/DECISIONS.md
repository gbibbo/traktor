# Decisiones de proyecto

Registro fechado de decisiones de Gabriel que condicionan el trabajo. Cada entrada dice qué se
decidió y qué implica para el código. Las decisiones más recientes prevalecen sobre planes
anteriores (`v4_implementation_plan.md`, `dj_music_clustering_deterministic_implementation_plan_v6.md`).
Una decisión revertida o reemplazada no se borra: se marca en su lugar con
`[REEMPLAZADO <fecha> por <decisión>: <lo que rige ahora>]` (o `[REFUTADO …]` si era un error).

## 2026-09-29

1. **Organización elegida: CLAP.** En la página de revisión, Gabriel eligió "claramente" la
   organización B frente a la A. B = CLAP + BPM (peso 0.3), Ward 15 carpetas / ~35 temas por
   playlist (config `fb78f2f6`, la de `exports/V4_1/`); A = MAEST-HF capa 7 con los mismos
   parámetros (`9e5784c3`). Salvedad: la comparación no fue del todo a ciegas (B ya la había
   escuchado y se reconocía por su número de playlists). No se exportó el CSV de veredictos.

2. **Metadatos de Beatport en los tags** (Artist, Remixers, Label, Genre, Released), con su
   convención: el remixer sale de Artist; Remixers = nombre de la mezcla si es un remix, "Extended
   Mix" si el archivo es la Extended, si no "Original Mix". El género de Beatport sobrescribe el
   que tenga el archivo.
3. **Si Beatport solo tiene otro remix, el tema va al modelo de género** (no hereda el género de
   otra versión: Beatport lo asigna por versión y por lanzamiento).
4. **Nada se escribe en los archivos de la colección antes de que Gabriel vea números y
   resultados.** Primero una tabla aparte (`features/beatport.parquet`) y el modelo entrenado con
   esos géneros; la escritura de tags queda para después de su visto bueno.
5. **Género del modelo solo con confianza** (clasificador >= 0.6: 171 temas). Donde no hay género
   confiable (confianza baja, gate MAEST sin validar, sin representaciones), el género que ya tiene
   el archivo se conserva (50 casos); si no tiene, queda vacío.
6. **Los matches B sin confirmar (110) no reciben datos de Beatport** y conservan el género que ya
   tienen (35 casos): Gabriel vio los ejemplos y lo juzgó razonable. La escritura de tags la corre
   Gabriel con `write_beatport_tags.py` (primero `--folder`), cuando el agente le avisa.
7. **WAV y AIFF no se escriben**: su track_uid incluye los tags y cambiaría (31 archivos).
8. **Temas repetidos: una sola copia por tema en las playlists** (Gabriel, escuchando los 24 grupos
   de `Música/_duplicados_para_escuchar.html`). 21 grupos son el mismo tema (1 a 10, 12 a 21 y el
   caso A, Abotha); sus 23 copias sobrantes salieron de `biblioteca` (v10, `dedupe.py apply`) y de
   los exports. [REEMPLAZADO 2026-09-29 por el punto 13: los cuatro pendientes quedaron resueltos]
   Pendientes: grupo 11 (Joris Voorn – Goodbye Fly, sin responder), caso B (Boris
   Brejcha: según sus tags son la Vocal Mix y la Original Mix), caso C (Markus Homm, otro corte) y
   «Enjoy The Silence … Cotton Dub» contra su «(boosted and cutted)», que apareció después.
9. **Qué copia se queda: siempre la MP3 de 320 kbps; si no hay, la de mejor calidad** (sin pérdida
   antes que comprimida; entre comprimidas, más kbps). Coincide con lo que eligió en los 10 grupos
   donde la calidad decide. [REEMPLAZADO 2026-09-29 por el punto 13: el empate se resuelve por
   ubicación] Empate de calidad: se pregunta. Sin confirmar: en los 7 empates eligió
   carpetas de año (2019, 2020…) antes que Nuevitas y Nuevitas antes que Milo.
10. **El mismo tema en otro corte (edit contra original) es un duplicado: queda uno.** Cuál, se pregunta.
11. **Música nueva:** lo idéntico o casi idéntico con calidad distinta se resuelve solo; lo dudoso
    (empate de calidad, mismo nombre, parecido, otro corte) lo pregunta la app. [REEMPLAZADO
    2026-09-29 por el punto 13: se resuelven solos «sonido» y «nombre»; lo demás queda como temas
    separados, sin preguntar] La pregunta en la app todavía no existe; por ahora, `dedupe.py candidates`.
12. **Las copias que no se quedan, además, se mueven a una carpeta aparte (`_copias`).** Antes de
    mover archivos hay que confirmar el alcance: las 23 de hoy o también las ≈ 616 copias idénticas
    que Phase 0 ya ocultaba (`duplicates.csv`). [REEMPLAZADO 2026-09-29 por el punto 13: se mueven
    solo las copias decididas]
13. **Temas repetidos, cierre: Gabriel dio el tema por cerrado y delegó lo pendiente** («lo mejor,
    menos molesto y reversible»). Lo que dijo él: el empate de calidad se resuelve por ubicación,
    **nunca Milo** (un amigo que copiaba sus temas elegidos y los bajaba con peor calidad) y primero
    las carpetas mejor clasificadas. Regla en `duplicates.location_rank`: carpetas por año, después el
    resto, después «copia», Milo al final; coincide con sus 8 elecciones en empates. Lo que decidió
    el agente por delegación:
    - Grupo 11: resuelto por la regla (queda `2020 old/old 6`).
    - Caso B: son distintos, porque sus tags dicen Vocal Mix y Original Mix.
    - Casos C y D: otro corte, queda la MP3 320.
    - Techyon / Egorythmia (0.972, tramos de un set continuo): son distintos.
    - En música nueva se resuelven solas las capas «sonido» y «nombre» (los 5 grupos por nombre que
      escuchó eran el mismo tema); «parecido» y «corte» quedan como temas separados, sin preguntar.
    - Se movieron a `Música/_copias/` las copias decididas, con su ruta de carpetas: 25 de 26, en
      `moved_copies.csv`. `dedupe.py restore-copies` las devuelve. La otra («Hide U» en
      `2020 new - copia`) se quedó en su lugar porque esa carpeta es `test_20`: nunca se mueven
      archivos de la carpeta de otro dataset.
    - Las ≈ 616 copias idénticas que ya ocultaba Phase 0 no se tocan.
    Resultado: `biblioteca` v11, 1760 temas.

## 2026-09-27

1. **La copia fiel es `main` de GitHub.** Lo que quedó en la laptop sin subir (4 commits locales de
   `feature/dj-clustering-v1`) se aparca: no se construye sobre eso ni se borra.
2. **MVP primero.** La urgencia es tener el pendrive de Rekordbox y las playlists de Traktor para
   tocar con la biblioteca que ya existe; la investigación del modelo de representación sigue
   después, sobre el mismo pipeline.
3. **Destino: Rekordbox y Traktor**, los dos.
4. **El humano cura, el modelo ordena y sugiere.** No se depende de historiales de lo tocado: la
   idea es que el modelo sugiera mejor de lo que Gabriel tocó antes. Hay algunos DJ sets (solo
   audio) reutilizables más adelante.
5. **El audio se compra.** "Me gusta" de Spotify tiene unos 2200 temas; quizá se descarguen solo
   los últimos ~500.
6. **Etiqueta "Vocal" en la metadata** de cada tema con voces, detectada con un modelo
   preentrenado.
7. La biblioteca se copia a `Música/` en la raíz del repo (git-ignorada). Las carpetas "old N" no
   tienen un significado fijo (a veces son copias de respaldo desordenadas): no se usan como señal.
8. **La carpeta `Música/Fede` no es de Gabriel**: se borró de la copia local y sale de catálogo,
   playlists y exports.
9. **Umbral de la etiqueta Vocal: 0.145** (antes 0.15), para que "I Don't Feel Like Dancing" de
   Scissor Sisters quede dentro. Se quitó el atributo de solo lectura a los archivos que no se
   podían etiquetar (compilado "Best Of Toolroom 2015").
10. **El trabajo paralelo de Gabriel vive en `download_JIJIJI/`** y se incluye en los commits y
    pushes al repo, nunca los archivos de audio.
11. **Antes de cualquier corrida larga, una pasada por una carpeta chica** para verificar la salida
    (p. ej. `extract_representations.py --folder`); recién después la corrida completa.
12. Los pushes desde la laptop van con la cuenta `gbibbo` (credencial local del repo).

## 2026-09-12

1. **El corazón del proyecto es un modelo de representation learning** que permita encontrar
   temas similares en el espacio de embeddings. El producto final (carpetas, orden, recomendador)
   no está cerrado y se definirá con un MVP funcional; todo lo demás se construye encima del
   modelo. Prioridad: producir ese modelo con todas las fuentes de información disponibles y una
   estrategia de evaluación clara que permita transferirlo a distintas tareas.
2. **Evidencia humana: se usan los dos conjuntos de tripletas** (`tools/dj_feedback/answers/`):
   las 60 del anotador de septiembre (muestreo uniforme) y las 40 recuperadas de la rama
   `feature/dj-clustering-v1` (selección por kNN de MERT y por fronteras de V4_5, con sesgo
   documentado). No se prevé más anotación manual. Toda evaluación debe reportar el resultado
   por fuente y combinado.
3. **1001Tracklists es un requisito estricto**, fijado conscientemente: usar muchas playlists
   de 1001Tracklists como fuente de información para el modelo. Anula la decisión fija 7 del
   plan v6 ("evidencia débil, nunca feature en Régimen 1"). Implica resolver de dónde se
   descarga el audio de esos temas, cómo y dónde se guarda; esas cuestiones se consultan con
   Gabriel antes de implementar, no se descartan por dificultad.
4. **Cómputo**: CPU local por defecto; GPU solo mediante trabajos gratuitos de Kaggle, a
   configurar cuando haga falta. Surrey HPC y Lightning AI ya no existen.
5. **Base de código**: se continúa en `main`. La rama `feature/dj-clustering-v1` y `legacy/`
   son cantera de ideas (ver `docs/reports/scientific_review_2026-09-12.md`, sección 6).
