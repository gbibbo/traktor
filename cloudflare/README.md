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

| Archivo | Qué es |
| --- | --- |
| `concept-ai-worker.js` | El Worker completo: instrucciones y validadores del LLM, la cola y la página del teléfono. Fuente única: no editar en el dashboard |
| `wrangler.jsonc` | Nombre y bindings de producción (`AI`, `DB`). Sin secretos |
| `schema.sql` | Tabla `jobs` de D1 (no cambió desde la v3) |
| `publicar.py` | Publica y conecta este ordenador en un paso |

Los secretos `APP_PASSWORD` (teléfono) y `DEVICE_TOKEN` (ordenador) viven en Cloudflare. No se pueden leer, solo
reemplazar.

## Publicar

Publicar en producción lo decide Gabriel. Desde la raíz del repo, con el Python del pipeline:

```
python cloudflare/publicar.py
```

El script hace cuatro cosas:

1. Publica el Worker.
2. Deja el mismo `DEVICE_TOKEN` en Cloudflare y en `artifacts/v4/traktor_live/conceptos_agente.json`, que está fuera de
   Git.
3. Comprueba la conexión sin consumir trabajos.
4. Activa el agente para que arranque con Traktor.

Requiere haber hecho `npx wrangler@4.159.0 login` una vez. Si algo sale mal, desde `cloudflare/`:
`npx wrangler@4.159.0 rollback`. El despliegue automático desde GitHub sigue apagado a propósito.

## Probar sin publicar

- Código real del Worker con una D1 local y un LLM simulado (no hace falta cuenta):
  `$env:TRAKTOR_WORKER_E2E = "1"; python -m pytest tests/v4/test_concept_worker_e2e.py -q`
- Con el LLM real: `wrangler dev` con el binding `AI` (gasta neuronas; nada se publica) y `src/v4/evaluation/clap_texto_llm.py`.
- Unas frases contra la música, sin red: `python tools/traktor_live/conceptos_agente.py probar --frase "festive house with bright trumpets"`.

## Operar

- Agente: `conceptos_agente.py estado | instalar | quitar`. Registro: `artifacts/v4/traktor_live/conceptos_agente.log`. Cada
  sondeo y cada lista quedan en `conceptos_resultados.jsonl`, para evaluar después.
- Para una prueba que no consuma otros conceptos de la cola: `conceptos_agente.py correr --una-vez --job <id>`.
- Un trabajo reclamado y sin terminar vuelve a la cola a los 5 minutos. A los 3 reclamos queda `failed`.
- Mientras Traktor está cerrado el agente no consulta la cola: lo escrito espera.
- Los conceptos guardados con la versión anterior (formato 1) se procesan directamente como lista final.
- El buscador es `buscar()` en `conceptos_agente.py`. Cambiarlo no toca ni el Worker ni el LLM.
