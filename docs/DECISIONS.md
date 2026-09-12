# Decisiones de proyecto

Registro fechado de decisiones de Gabriel que condicionan el trabajo. Cada entrada dice qué se
decidió y qué implica para el código. Las decisiones más recientes prevalecen sobre planes
anteriores (`v4_implementation_plan.md`, `dj_music_clustering_deterministic_implementation_plan_v6.md`).

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
