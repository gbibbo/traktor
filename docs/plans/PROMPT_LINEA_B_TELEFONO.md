# Prompt para un chat nuevo: línea B, el teléfono y el LLM

Para el agente que implementa la línea B del plan vigente (`docs/plans/plan_recomendacion_v2.md`, sección 3). Gabriel
lo pega en un chat nuevo. Escrito el 2026-11-13 por la sesión auditora.

---

Vas a implementar la línea B del plan de recomendación de TRAKTOR ML: que el teléfono sea una herramienta completa
para elegir la música que se sugiere. Otra sesión hace la línea A (experimentos con el dataset de sets) y otra
audita. No ves sus chats: lo que no está en el repositorio no existe para ti, ni para ellas.

## La idea, en una frase

El LLM no elige temas ni hace cuentas: llena un formulario con casillas de listas cerradas. El ordenador ejecuta ese
formulario siempre igual. Un preset del teléfono es el mismo formulario ya lleno y no pasa por el LLM.

Gabriel lo pidió así el 2026-11-13: presets de filtros (BPM, género y otros), un campo de texto donde se pueda
escribir cualquier cosa (un momento de la noche, el nombre de un tema parecido, un concepto, un género), y reglas
exhaustivas, claras y eficientes para el LLM, que es chico y razona poco: los deberes se hacen de antemano.

## Lee primero

1. `AGENTS.md`, `CLAUDE.md` (el párrafo de Cloudflare) y `docs/STATUS.md`.
2. `docs/plans/plan_recomendacion_v2.md`: secciones 1, 3, 6 y 7.
3. `cloudflare/README.md`: el circuito de hoy, del teléfono a la playlist.
4. `docs/reports/clap_texto_2026-11-09.md`: por qué las instrucciones del LLM dicen lo que dicen. Esas reglas de
   redacción siguen valiendo para la casilla `texto`.
5. `docs/reports/pools_de_concepto_2026-11-11.md`: por qué hacen falta filtros por campo y cómo se van a juzgar las
   listas.
6. El código: `cloudflare/concept-ai-worker.js`, `tools/traktor_live/conceptos_agente.py` (`buscar()`, `sondear()`,
   `crear_lista()`) y `tools/traktor_live/recomendador.py` (filtros del vivo y tonalidad).

Los ids L1 a L12 están en `docs/plans/inventario_ideas.md`.

## Qué hacer, en orden

**B0, contrato.** El formulario (sección 3.2 del plan) como esquema con versión, con tests, y el catálogo de la
biblioteca generado por código: géneros con su conteo, rangos de BPM, campos disponibles.

**B1, ejecutor.** `buscar()` recibe un formulario. Filtra por los campos que existen hoy (género, BPM, tonalidad,
voz), puntúa, corta y devuelve además qué hizo: cuántos temas pasaron cada filtro. Un formulario vacío tiene que dar
la lista de hoy. Dos casos: sin tema de referencia y con el set en curso.

**B2, presets.** La pantalla del teléfono con presets y texto libre. Un preset llega al ordenador sin pasar por el LLM.

**B3, reglas y LLM.** Primero un intérprete por reglas, sin LLM, para lo que se puede leer con certeza. Después el
LLM en dos pasos cortos: clasificar el tipo de pedido y llenar cada casilla con las opciones a la vista. Validadores
para todo, y si todo falla, el camino de hoy. El reglamento vive en un solo archivo del repo; de él salen los prompts.

**B4, explicación.** El teléfono muestra qué entendió y cuántos temas pasaron cada filtro.

**B5 y B6:** como dice la tabla de la sección 3.5 del plan.

## Reglas

- **El banco antes que las reglas.** Arma el banco de regresión (de 150 a 200 pedidos con su formulario esperado)
  antes de afinar instrucciones, y fija el umbral antes de medir. Un cambio de reglas no entra si empeora una casilla.
  La semilla es `docs/plans/pedidos_telefono_borrador.md`: 82 pedidos con lo que el sistema debería entender.
  Gabriel no la revisa; la revisa la auditoría. Sus pedidos reales se juntan del registro del ordenador
  (`conceptos_resultados.jsonl`) a medida que usa el teléfono. Pasa la semilla al formulario cuando el esquema esté
  fijo y amplíala.
- **Presupuesto humano** (Gabriel, 11-13). Toda tarea que le pidas a Gabriel, sea una revisión, una elección o una
  prueba, dice antes cuánto lleva, entra en 15 minutos y tiene una decisión por pantalla. Lo que puedas decidir con
  un valor por defecto, decídelo y díselo en una línea; no le des listas largas para corregir. Regla completa:
  `docs/reports/escucha_corta_e12_2026-11-13.md`.
- **Género duro o blando** (Gabriel, 11-13; puede cambiar): blando si el género viene del texto, duro si se eligió en
  un preset. Blando quiere decir que ese género va primero y entran también sus vecinos.
- **Listas cerradas.** El LLM nunca escribe un género, un nombre de perfil ni un número que no esté en su lista.
  Nunca escribe pesos como números.
- **Privacidad.** La música, los embeddings, las rutas y la lista de temas no salen del ordenador. Al Worker va el
  catálogo en agregados. Un tema nombrado lo resuelve el ordenador.
- **Lo que no existe todavía se deja previsto.** Las casillas `momento`, «como tal DJ» y el corte por confianza
  dependen de la línea A. El formulario las tiene; el ejecutor las ignora con aviso hasta que lleguen. Para `momento`
  hay una tabla fija que aprueba Gabriel.
- **No toques el orden de Parecidos.** El puntaje que usa el ejecutor es el del recomendador, tal como esté por
  defecto. Cambiarlo es de la línea A y de la escucha de Gabriel.
- **Nada se publica.** `python cloudflare/publicar.py` lo corre Gabriel. Las pruebas contra la cola real usan
  `--job <id>`. No ejecutes `/api/device/claim`.
- **Costo.** Cada concepto gasta neuronas de Workers AI: mide cuántas por pedido antes y después.
- **Subir.** Varias sesiones comparten este árbol. Se sube solo con
  `python tools/repo/subir.py -m "..." <rutas>` (`AGENTS.md`, «Several sessions on one working tree»). Los documentos
  compartidos (`STATUS.md`, `DECISIONS.md`, el plan) se suben enteros apenas los editas.

## Primera respuesta esperada

Sin cambiar código todavía: el esquema del formulario que propones, con dos o tres pedidos de ejemplo ya llenos; la
lista de tipos de pedido para el clasificador; qué campos de la biblioteca existen hoy y con qué cobertura; y las
preguntas para Gabriel (filtro de género duro o blando, tabla de momentos, sus pedidos de ejemplo).
