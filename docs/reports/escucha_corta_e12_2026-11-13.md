# La escucha E12 en versión corta, y la regla para todo lo que se le pide a Gabriel

**Fecha:** 2026-11-13. **Para:** Gabriel y las sesiones que diseñen una escucha, una revisión o cualquier tarea para
él. **Origen:** Gabriel rechazó la escucha E12 tal como estaba planteada y la revisión de 82 pedidos de ejemplo.
Autor: la sesión auditora, que había propuesto las dos cosas.

## Qué dijo Gabriel

La escucha le pedía oír 60 temas por tema principal, en 10 temas principales, y decidir de memoria cuál lista se
parecía más. «Eso no es solo una locura, ambicioso y poco considerado: es inviable.» Sobre los 82 pedidos: ni los
abrió, porque esperaba el mismo error. El pedido es adaptar los ejercicios a una persona ocupada, que se cansa.

Tiene razón. El error fue de la auditoría: multiplicó el formato del producto (listas de 30) por 10 temas y por 2
listas, calculó de una hora y media a tres horas y lo presentó como razonable.

## La regla: presupuesto humano

Vale para toda tarea que un agente le pida a Gabriel.

1. **Se dice cuánto lleva antes de pedirla,** medido o calculado. Si el diseño multiplica (temas × listas ×
   candidatos), se calcula el total.
2. **Entra en 15 minutos.** Si no entra, se rediseña con una muestra más chica. Partirla en diez sentadas no la
   arregla.
3. **Una decisión por pantalla.** Nada de llevar la cuenta de memoria ni de comparar listas largas.
4. **Lo que un agente puede decidir con un valor por defecto, no se pregunta.** Se elige, se le dice en una línea y
   él lo cambia si quiere.
5. **Sus datos reales se juntan solos** antes de pedirle que los escriba: el registro del teléfono, el historial de
   Traktor.
6. **Lo que un agente puede revisar, lo revisa un agente.** A Gabriel le llega lo que solo él puede juzgar: cómo suena.

## La escucha corta

Protocolo fijado antes de que exista ninguna respuesta. La clave de cuál lista es cuál no se abrió.

| | Antes | Ahora |
| :--- | :--- | :--- |
| Qué se oye por tema principal | El principal y las dos listas enteras: 60 temas | El principal y 6 fragmentos: los 3 primeros temas que cada lista tiene y la otra no |
| Cómo | A mano, tema por tema | La página los hace sonar sola, 10 s cada uno, desde el 35 % del tema |
| Qué decide Gabriel | Cuál lista prefiere, de memoria | Lo mismo, con 3 temas por lado a la vista |
| Cuánto lleva | De una hora y media a tres horas | Unos 12 minutos, con una decisión por pantalla |

- **Qué temas.** En el orden de cada lista, los 3 primeros que no están en la otra. En la ronda del 11-10 salen de
  los puestos 1 a 7. Los temas que las dos listas comparten no ayudan a decidir y no se oyen.
- **Orden.** El principal primero. Después, la lista A o la B según el número del tema sea impar o par.
- **Respuesta.** A, B o «no las distingo», un clic, y la página pasa al tema siguiente.
- **Cómo se lee.** Igual que antes: con 10 respuestas, 9 o 10 para el mismo lado deciden; 8 es un indicio.
- **Qué mide y qué no.** Mide el comienzo de cada lista, que es lo que primero se mira al tocar. No mide los 30 temas.
  Si no decide, no se vuelve a la versión larga: queda como «no se distinguen de oído en los primeros puestos».
- **Dónde.** `artifacts\v4\escucha\e12\escucha.html`, en este ordenador, fuera de Git. La arma
  `tools/playlist_review/pagina_escucha.py` con las listas de `erclm:traktor_ml/dj_sets/e12/listas/`. El audio sale de
  `C:\TRAKTOR ML\Musica`; sus 70 archivos existen. No depende del pendrive.
- **Probado.** La página se abrió en un navegador sin pantalla con dos respuestas simuladas: guarda, avanza y cuenta.
  El sonido no se pudo probar así.

## Los pedidos del teléfono

Gabriel no revisa los 82 (`docs/plans/pedidos_telefono_borrador.md`). Los revisa la auditoría. Sus pedidos reales se
juntan del registro del ordenador a medida que usa el teléfono. Las cuatro preguntas que tenía la lista pasaron a ser
valores por defecto del agente, que él cambia si quiere: BPM solo es ±2; si no hay, el teléfono lo dice y ofrece lo
cercano; «ignora la tonalidad» baja su peso y deja el filtro del vivo; ante un pedido ambiguo, el teléfono pregunta
con dos o tres opciones.

## Lo que cambia en el plan

- La escucha de A3 y la de A5 (E11) usan este formato.
- Para los pools de concepto (E18), los agentes proponen las playlists y Gabriel dice sí o no.
- La tabla de momentos de la línea B la propone el agente.
- De la sección 6 del plan, lo único que hace falta de Gabriel ahora es esta escucha, cuando pueda.
