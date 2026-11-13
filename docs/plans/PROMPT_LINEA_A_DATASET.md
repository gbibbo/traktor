# Prompt para un chat nuevo: línea A, experimentos con DJSets v1

Para el agente que implementa la línea A del plan vigente (`docs/plans/plan_recomendacion_v2.md`, sección 2). Gabriel
lo pega en un chat nuevo. Escrito el 2026-11-13 por la sesión auditora.

---

Vas a implementar la línea A del plan de recomendación de TRAKTOR ML: aprender del dataset DJSets v1 qué temas van
juntos y en qué momento de la noche, y llevarlo a la biblioteca de Gabriel. Otra sesión hace la línea B (el
teléfono) y otra audita. No ves sus chats: lo que no está en el repositorio no existe para ti, ni para ellas.

## Lee primero

1. `AGENTS.md` (sobre todo «Compute») y `docs/STATUS.md`.
2. `docs/plans/plan_recomendacion_v2.md`: secciones 1, 2, 6 y 7.
3. `datasets/djsets/README.md`: la ficha del dataset y su lector `djsets.py`.
4. `docs/GPUQ.md` y, en el repo de gpuq, `docs/WRITING_JOBS.md`.
5. De lo ya hecho, solo lo que vayas a continuar: `docs/reports/codec_clap_2026-11-10.md` (v8c),
   `docs/reports/juez_d5_2026-11-10.md` (la ecuación y la escucha) y el trabajo de V7 en el commit `91f2a2d`.

Si un id de idea no te dice nada (P12b, E20, T20...), está en `docs/plans/inventario_ideas.md`.

## Qué hacer, en orden

**A0, vistas de todo el dataset.** Cuenta primero qué cubren los 60 archivos de `erclm:traktor_ml/v7_vistas/` frente
a `tracks.csv` de `djsets_v1`. Después completa CLAP, MAEST capa 7 y estilos para todos los audios, por segmento,
guardados por `track_id` en una carpeta nueva del Drive. Los temas enteros de YouTube se cortan con la misma ventana
que un preview; di cuál usaste. El trabajo de V7 ya midió 2,04 s por preview en Kaggle: parte de ese código.

**A1, tablero fijo.** Antes de entrenar nada. Sale de `split_by_dj.csv`, con pares de prueba cuyos dos temas no
aparecen en el entrenamiento; la ficha del dataset dice qué fugas tiene esa partición. Negativos en tres niveles: al
azar, mismo BPM, mismo BPM y género. Líneas de base: BPM solo, género, cada vista, la variante v8c y los pesos
iguales. Intervalos remuestreando DJs.

**A2 a A6:** como dice la tabla de la sección 2.3 del plan. A5 y A6 no dependen de A2 ni de A3: pueden ir en paralelo.

**A4 y A7 no se empiezan sin que Gabriel lo diga.**

## Reglas

- **Umbrales antes de medir.** Cada paso fija su protocolo y sus umbrales en un commit, y el informe cita ese commit.
  Lo que agregues después de ver los datos va marcado como exploratorio.
- **Nada se ajusta con los sets ni con la biblioteca de Gabriel.** Son solo para medir.
- **Un mejor número no cambia el recomendador.** No toques los valores por defecto de `recomendador.py`.
- **gpuq:** prueba chica en CPU con el mismo código, y antes del primer envío de cada tipo de trabajo, pide el OK a
  Gabriel en el chat con el tiempo por ítem, las horas de GPU, el espacio en el Drive y las rutas.
- **gpuq usa Kaggle y Colab según la demanda** (Gabriel, 11-13). Colab perdió la sesión a los 13 minutos en el
  único trabajo largo que corrió ahí, y gpuq lo dio por fallido en el intento 11; gpuq relanza y el trabajo retoma,
  pero no evita el corte. Escribe todo trabajo para sesiones cortas: guarda en el Drive cada pocos minutos, nunca más
  de 10 sin guardar, y parte el trabajo en varios trabajos chicos, de una hora de trabajo útil o menos. Lo que
  necesite horas seguidas va con `--platform kaggle`. Detalle: plan, sección 2.5.
- **Datos:** todo resultado va al Drive, bajo `erclm:traktor_ml/`, apenas se produce. `djsets_v1` no se modifica.
  Al repositorio, que es público, va solo código e informes.
- **Sin pedidos nuevos** a Beatport, 1001Tracklists ni YouTube.
- **Documentos compartidos** (`STATUS.md`, `DECISIONS.md`, el plan): agrega solo tus líneas y haz `git add` de rutas
  explícitas. La línea B edita los mismos archivos.
- **Informes** en `docs/reports/`, con el veredicto primero y las etiquetas del skill `auditar`. Cada uno dice la
  versión del dataset, cuántos sets y temas midió y el commit.

## Primera respuesta esperada

Sin correr nada pesado: qué cubre V7 y qué falta para A0, con el conteo; el protocolo de A0 y de A1 que vas a fijar;
la prueba chica en CPU que harás; y el pedido de OK para el primer trabajo de gpuq, con sus números.
