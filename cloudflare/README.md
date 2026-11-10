# Traktor Concept AI: Worker de Cloudflare

Desde el teléfono, el DJ describe qué quiere escuchar. Este Worker conversa con un LLM (Workers AI), arma un concepto y lo deja en una cola de D1. El ordenador con Traktor lo procesa con `tools/traktor_live/conceptos_agente.py` y devuelve los temas. La música, los embeddings y las rutas nunca salen del ordenador.

Antecedentes y contrato: `docs/reports/traktor_concept_ai_cloudflare_2026-11-09.md`.

| Archivo | Qué es |
| --- | --- |
| `concept-ai-worker.js` | El Worker completo, con la página del teléfono adentro. Fuente única: no editar en el dashboard |
| `wrangler.jsonc` | Nombre y bindings de producción (`AI`, `DB`). Sin secretos |
| `schema.sql` | Tabla `jobs` de D1 |

Los secretos `APP_PASSWORD` (teléfono) y `DEVICE_TOKEN` (ordenador) viven en Cloudflare. Cloudflare no los deja leer: se pueden reemplazar, no consultar.

## Probar sin tocar Cloudflare

Código real del Worker, con una D1 local. No hace falta cuenta.

```
$env:TRAKTOR_WORKER_E2E = "1"
python -m pytest tests/v4/test_concept_worker_e2e.py -q
```

Una frase contra la biblioteca real, sin red: `python tools/traktor_live/conceptos_agente.py probar --frase "dark hypnotic techno"`.

## Desplegar

Desde esta carpeta, con `npx wrangler@4.159.0`. Primero `login`, una sola vez por ordenador.

```
npx wrangler@4.159.0 deploy --dry-run
npx wrangler@4.159.0 deploy
```

Si algo sale mal: `npx wrangler@4.159.0 rollback`. El despliegue automático desde GitHub sigue apagado a propósito.

## Conectar el ordenador

`artifacts/v4/traktor_live/conceptos_agente.json`, fuera de Git:

```json
{ "url": "https://traktor-concept-ai.gabobibbo2.workers.dev", "device_token": "<el mismo valor que en Cloudflare>" }
```

Para arrancarlo solo con Traktor: `python tools/traktor_live/conceptos_agente.py instalar` (`estado` y `quitar` también existen). Registro: `artifacts/v4/traktor_live/conceptos_agente.log`.

Para una prueba que no consuma otros conceptos de la cola: `conceptos_agente.py correr --una-vez --job <id>`.

## Comportamiento de la cola

- Un trabajo reclamado y sin terminar vuelve a la cola a los 5 minutos. A los 3 reclamos queda `failed`.
- Guardar dos veces el mismo borrador no crea otro trabajo.
- Mientras Traktor está cerrado el agente no consulta la cola: lo escrito espera y se procesa al abrirlo.
