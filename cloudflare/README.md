# Traktor Concept AI: del teléfono a una playlist nueva en Traktor

El DJ escribe en el teléfono lo que quiere escuchar, como le salga («quiero música de cumpleaños con cornetas»), y aparece
una playlist nueva en Traktor. En el medio, el LLM de Cloudflare traduce el pedido a textos que CLAP sabe buscar, prueba
varios en la música y decide mirando qué encontró cada uno. La música, los embeddings y las rutas nunca salen del
ordenador.

```
Teléfono --- texto ---> Worker (Cloudflare)
                          | LLM «interpretar»: 3 textos en inglés, qué no se oye, nombre
                          v
                      cola en D1 <--- reclama --- Ordenador (conceptos_agente.py)
                          |                         | sondeo: busca cada texto y describe lo que encontró
                          | <--- informe -----------+
                          | LLM «elegir»: se queda con los textos que encajan, o reescribe (una ronda más como máximo)
                          v
                      cola en D1 <--- reclama --- Ordenador
                                                    | final: busca con el promedio y escribe «Concepto <fecha> <nombre>.nml»
Teléfono <-- qué entendió, qué buscó, la lista -----+
```

Por qué las instrucciones del LLM dicen lo que dicen: `docs/reports/clap_texto_2026-11-09.md`. Antecedentes:
`docs/reports/traktor_concept_ai_cloudflare_2026-11-09.md`.

## v6 (2026-11-13): presets, formulario y lo que suena

Es la línea B del plan (`docs/plans/plan_recomendacion_v2.md`, § 3). El pedido es un formulario con casillas de listas
cerradas (`config/telefono/formulario.json`). El ordenador lo ejecuta siempre igual (`tools/traktor_live/ejecutor.py`).

- **Presets** (`config/telefono/presets.json`): seguir lo que suena, más lento, más rápido, sin voces, los 6 géneros con
  más temas (filtran duro) y los momentos, apagados hasta que Gabriel apruebe su tabla. Un pedido solo con presets va
  directo a la cola como lista final: no llama al LLM y no gasta neuronas (lo comprueba un test de punta a punta).
- **Texto libre, primero por reglas** (`reglas.js`, datos en `config/telefono/reglamento.json`). Lo que se lee con
  certeza llena el formulario sin LLM: «parecido a lo actual», BPM, géneros (también mal escritos), voz, tonalidad,
  momento, año, sello, un tema nombrado.
  - Si no queda nada sin leer, el pedido va directo a la lista: 122 de los 165 pedidos del banco.
  - Lo que queda sin leer pasa por el mismo circuito del LLM de la v5, con los filtros de las reglas y de los presets.
  - Un pedido ambiguo («bajar un cambio») se contesta con opciones de un toque. Uno de algo que no hay («reggaeton») y
    uno vago («hola») se contestan sin armar lista.
  - Banco y umbral: `config/telefono/banco_pedidos.jsonl`; `python tools/traktor_live/banco_telefono.py evaluar
    --interprete reglas --modo reglas`.
- **El resultado dice qué hizo**: cuántos temas quedaron después de cada filtro, lo aplicado en palabras, lo que no
  pudo aplicar y por qué. Si un tema nombrado tiene dudas, el resultado trae los candidatos para elegir.
- **Lo que suena, arriba de la pantalla.** El ordenador lo manda de paso en su pedido de trabajo, al cambiar y cada
  5 min, junto con el catálogo de la biblioteca en números (géneros con su conteo). Nada de rutas ni de la lista de
  temas. Se guarda en la tabla `estado` de D1. El teléfono lo pide (`GET /api/estado`) cada 20 s mientras la pantalla
  está a la vista, y deja de pedirlo con la pantalla apagada.
  - Costo de una noche de 6 h: unos 1100 pedidos del teléfono y unos 5400 del ordenador (uno cada 4 s, como en la v5),
    con unas 100 escrituras en D1.
  - Cupo gratuito: 100 000 pedidos y 100 000 escrituras por día. Sin LLM.
- **Contraseña una sola vez**: el teléfono la recuerda (`localStorage`). «Salir» la olvida.

Los trabajos guardados con la v5 (versión 2) siguen funcionando.

| Archivo | Qué es |
| --- | --- |
| `concept-ai-worker.js` | El Worker completo: instrucciones y validadores del LLM, la cola y la página del teléfono. Fuente única: no editar en el dashboard |
| `wrangler.jsonc` | Nombre y bindings de producción (`AI`, `DB`). Sin secretos |
| `schema.sql` | Tablas de D1: `jobs` (no cambió desde la v3) y `estado` (v6: lo que suena y el catálogo) |
| `publicar.py` | Publica y conecta este ordenador en un paso |

Los secretos `APP_PASSWORD` (teléfono) y `DEVICE_TOKEN` (ordenador) viven en Cloudflare. No se pueden leer, solo
reemplazar.

## Publicar

Publicar en producción lo decide Gabriel. Desde la raíz del repo, con el Python del pipeline:

```
python cloudflare/publicar.py
```

El script hace cinco cosas:

1. Aplica `schema.sql` en la D1 remota: crea la tabla `estado` si falta. No toca los trabajos.
2. Publica el Worker.
3. Deja el mismo `DEVICE_TOKEN` en Cloudflare y en `artifacts/v4/traktor_live/conceptos_agente.json`, que está fuera de
   Git.
4. Comprueba la conexión sin consumir trabajos.
5. Activa el agente para que arranque con Traktor.

Requiere haber hecho `npx wrangler@4.159.0 login` una vez. Si algo sale mal, desde `cloudflare/`:
`npx wrangler@4.159.0 rollback`. El despliegue automático desde GitHub sigue apagado a propósito.

## Probar sin publicar

- Código real del Worker con una D1 local y un LLM simulado (no hace falta cuenta):
  `$env:TRAKTOR_WORKER_E2E = "1"; python -m pytest tests/v4/test_concept_worker_e2e.py -q`
- Con el LLM real: `wrangler dev` con el binding `AI` (gasta neuronas; nada se publica) y `src/v4/evaluation/clap_texto_llm.py`.
- Unas frases contra la música, sin red: `python tools/traktor_live/conceptos_agente.py probar --frase "festive house with bright trumpets"`.
- Un formulario contra la música, sin red (cuántos temas pasan cada filtro y la lista):
  `python tools/traktor_live/conceptos_agente.py formulario '{"filtros": {"voz": "sin"}, "referencia": {"tipo": "sonando"}}' --sonando "Carl Craig The Melody"`.
  `--sesion` lee lo que suena del registro del vivo.

## Operar

- Agente: `conceptos_agente.py estado | instalar | quitar`. Registro: `artifacts/v4/traktor_live/conceptos_agente.log`. Cada
  sondeo y cada lista quedan en `conceptos_resultados.jsonl`, para evaluar después.
- Para una prueba que no consuma otros conceptos de la cola: `conceptos_agente.py correr --una-vez --job <id>`.
- Un trabajo reclamado y sin terminar vuelve a la cola a los 5 minutos. A los 3 reclamos queda `failed`.
- Mientras Traktor está cerrado el agente no consulta la cola: lo escrito espera.
- Los conceptos guardados con la versión anterior (formato 1) se procesan directamente como lista final.
- El buscador es `buscar()` en `conceptos_agente.py`. Cambiarlo no toca ni el Worker ni el LLM.
