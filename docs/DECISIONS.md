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
