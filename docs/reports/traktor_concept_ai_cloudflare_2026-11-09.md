# Traktor Concept AI: handoff Cloudflare a TRAKTOR ML

**Fecha:** 2026-11-09. **Estado:** MVP remoto operativo para conversación, preparación de conceptos y encolado en D1. [REEMPLAZADO 2026-11-09 por el agente local y el Worker v4 (sección 8): probados de punta a punta en local] La integración con el ordenador de Traktor sigue pendiente.

> **Para Claude Code:** leer este documento, el archivo **cloudflare/concept-ai-worker.js**, **AGENTS.md**, **CLAUDE.md** y **docs/STATUS.md** antes de implementar el agente local. Este informe es un registro del trabajo realizado en la sesión de ChatGPT, no prueba de que se haya inspeccionado directamente el código binario o la versión activa de Cloudflare. El archivo JS es una copia de la versión v3 suministrada para pegar en el dashboard y que el usuario indicó haber desplegado; antes de cambios críticos, contrastarla con **Edit Code / Deployments**.

## 1. Objetivo y límites

Permitir que Gabriel, mientras pincha, escriba en el teléfono una intención musical en lenguaje natural. Un LLM remoto aclara esa intención, prepara un concepto para recuperación cruzada texto/audio mediante CLAP y solicita confirmación. El ordenador con TRAKTOR ML consultará por HTTPS la cola remota, calculará similitudes con su biblioteca local y enviará una lista de pistas para mostrar al usuario.

**Decisiones:** LLM exclusivamente en Cloudflare Workers AI, sin LLM en el ordenador de DJ; audio, embeddings, catálogo y rutas de archivos permanecen locales; sin abrir puertos entrantes ni exponer la biblioteca; infraestructura de coste bajo o gratuito dentro de las cuotas aplicables. No confundir similitud coseno CLAP con probabilidad ni porcentaje de precisión.

Flujo previsto:

~~~text
Teléfono (navegador)
    |
    | HTTPS: /api/chat, /api/draft, /api/jobs
    v
Cloudflare Worker traktor-concept-ai
    | AI: Gemma 4 26B A4B IT (conversación y JSON)
    | DB: D1 / traktor-concepts / jobs
    |
    | HTTPS autenticado: /api/device/claim y /api/device/complete
    v
[REEMPLAZADO 2026-11-09: tools/traktor_live/conceptos_agente.py] Agente Python local (AÚN NO IMPLEMENTADO)
    |
    | consulta embeddings CLAP ya existentes, metadatos y track_uid
    v
Biblioteca TRAKTOR ML local
    |
    | resultados JSON hacia D1
    v
Teléfono: GET /api/jobs/{id} ([REEMPLAZADO 2026-11-09: el Worker v4 muestra la lista] API lista; UI de resultados PENDIENTE)
~~~

**Entorno local:** repositorio público **gbibbo/traktor**, rama principal **main**. El repositorio documenta una biblioteca deduplicada de 1.760 temas y organización basada en CLAP + BPM. Verificar el estado de los artefactos actuales localmente; no asumir que están en Git. Respetar los límites de **AGENTS.md**: no subir audio, embeddings, bases locales, cachés ni credenciales. El usuario no tenía acceso al ordenador de Traktor el 09/11/2026; la única prueba desde otro ordenador fue una petición HTTPS de autenticación, no una instalación de agente.

## 2. Recursos Cloudflare creados

- **Cuenta/namespace:** gabobibbo2.
- **Worker:** traktor-concept-ai.
- **URL pública:** https://traktor-concept-ai.gabobibbo2.workers.dev/
- **D1:** base **traktor-concepts**, binding **DB**.
- **Workers AI:** binding **AI**, modelo **@cf/google/gemma-4-26b-a4b-it**.
- **Secret APP_PASSWORD:** clave para la interfaz y la API del DJ, cabecera **X-App-Password**.
- **Secret DEVICE_TOKEN:** clave independiente para el ordenador, cabecera **X-Device-Token**. Se probó su validez. No está ni debe estar en Git.
- **Despliegue:** efectuado manualmente desde el editor de Cloudflare; todavía no se ha conectado GitHub ni configurado Wrangler o despliegue automático.

**Tabla D1 creada:**

~~~sql
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  claimed_at INTEGER,
  attempts INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL CHECK (status IN ('queued', 'processing', 'done', 'failed')),
  concept_json TEXT NOT NULL,
  result_json TEXT,
  error TEXT
);
CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(status, created_at);
~~~

Timestamps en milisegundos Unix. Estados: **queued** (esperando), **processing** (reclamado), **done** (resultado), **failed** (error). No se configuraron colas de Cloudflare Queues: es una **cola implementada sobre D1**.

## 3. Código v3 y funcionamiento

**Fuente de referencia:** **cloudflare/concept-ai-worker.js**, 376 líneas, copia textual del archivo v3 entregado en ChatGPT. Contiene todo: backend Worker + HTML/CSS/JS móvil inline. No hay servidor FastAPI ni backend local operativo para esta característica, ni archivos estáticos separados, ni framework JS.

### Interfaz y LLM

1. El usuario introduce **APP_PASSWORD** en la página y su descripción musical en español.
2. **Enviar** manda historial a **POST /api/chat**. Se admiten hasta 12 mensajes, cada uno de hasta 1.500 caracteres; el historial vive en memoria del navegador, no en D1. El modelo responde de forma breve y, según el prompt, puede hacer una pregunta esencial. A partir del segundo mensaje del DJ debe resumir y pedir confirmación.
3. **Preparar concepto** llama **POST /api/draft**. El mismo modelo recibe instrucciones para devolver un objeto JSON (response_format json_object, máximo 650 tokens, temperatura 0,2), con hasta dos intentos. Se valida estructuralmente mediante **normalizeConcept**.
4. El Worker firma el objeto normalizado con HMAC-SHA256 basado en **APP_PASSWORD**, junto con una caducidad de 20 minutos. Devuelve **concept**, **expires** y **signature**.
5. El navegador muestra el JSON y **Confirmar y guardar concepto** exige confirmación explícita. **POST /api/jobs** vuelve a validar el JSON y la firma, e inserta la fila D1 con UUID y estado **queued**. La firma evita editar el borrador entre preparación y guardado; **no evita reenvíos repetidos** del mismo borrador durante su validez.

**Contrato JSON del concepto (ejemplo ilustrativo, no una nueva prueba):**

~~~json
{
  "version": 1,
  "name": "Techno industrial agresivo",
  "description": "Techno industrial pesado, con percusión intensa y carácter agresivo.",
  "positive_prompts": [
    "Heavy industrial techno with distorted hard kicks and metallic percussion",
    "Aggressive, driving industrial techno with rough mechanical textures"
  ],
  "negative_prompts": [],
  "constraints": {
    "energy": "unspecified",
    "vocals": "unspecified",
    "bpm": "unspecified"
  }
}
~~~

**Campos:** name (3-100 caracteres); description (15-700); positive_prompts (1-4 frases, 8-180 caracteres); negative_prompts (0-4, solo exclusiones explícitas pedidas al modelo); constraints.energy = maintain/increase/decrease/unspecified; vocals = present/absent/unspecified; bpm = near_current/unspecified. **Energy expresa cambio respecto al tema actual, no agresividad absoluta.** La validación es sintáctica y de enumeraciones: no demuestra semánticamente que el LLM no haya añadido atributos inventados.

### Endpoints v3

| Método / ruta | Autorización | Contrato y efecto |
| --- | --- | --- |
| GET / | pública | HTML móvil y JavaScript embebido. |
| GET /api/health | APP_PASSWORD | Responde database: connected y número total de filas jobs. |
| POST /api/chat | APP_PASSWORD | Body: messages; retorna reply del LLM. |
| POST /api/draft | APP_PASSWORD | Body: messages; retorna concept + expires + signature. |
| POST /api/jobs | APP_PASSWORD | Body: concept, expires, signature, confirmed:true; crea UUID queued; devuelve id y status (201). |
| GET /api/jobs/{uuid} | APP_PASSWORD | Retorna estado, concepto, resultado o error de una búsqueda. No está conectado todavía a la UI móvil. |
| POST /api/device/claim | DEVICE_TOKEN | Selecciona y reclama el primer queued mediante UPDATE ... RETURNING; cambia a processing, aumenta attempts y retorna job.id, job.attempts y job.concept, o job:null. **Hace una mutación**, no es una prueba inocua. |
| POST /api/device/complete | DEVICE_TOKEN | Body: id, attempts, status:done + result, o status:failed + error. Solo modifica una fila que continúa processing y coincide con attempts; 409 si ya no corresponde. |

**Contrato de resultado de POST /api/device/complete:**

~~~json
{
  "id": "<uuid_del_job>",
  "attempts": 1,
  "status": "done",
  "result": {
    "tracks": [
      {
        "track_id": "<track_uid_local>",
        "title": "Título local",
        "artist": "Artista local",
        "score": 0.63
      }
    ]
  }
}
~~~

**Restricciones:** 0 a 30 pistas; track_id requerido (hasta 200 caracteres); title requerido (hasta 250); artist texto hasta 200; score finito entre -1 y 1. Este intervalo admite similitud coseno, pero el Worker no calcula ni calibra el score. Para error: status=failed y error texto no vacío de hasta 400 caracteres. **attempts** funciona como comprobación de generación del reclamo. La lógica SQL evita completar una fila que dejó de estar processing.

**Observación:** la API devuelve un objeto mínimo de resultados, sin BPM, tonalidad, ruta, artwork, explicación, calibración ni ranking multimodal. La UI aún no presenta estos resultados.

## 4. Evidencia de las pruebas realizadas

- Workers AI respondió correctamente a una consulta musical en el editor Preview.
- La interfaz se probó en Android. Hubo inicialmente discrepancia entre Preview y versión pública; se resolvió entrando al Worker desde Cloudflare **Visit**. La UI pública mostró el botón de confirmación.
- D1 respondió a health y a inserciones de trabajos.
- En v2 se guardó erróneamente una pregunta del LLM como descripción. Job **3f16d5bd-212d-4e94-91f5-1b8940090b8f**: se marcó **failed** mediante UPDATE manual.
- En v2 se guardó el concepto **Techno Industrial Agresivo**, job **4810286b-c784-4634-b8df-b5c71342a2d4**. SQL verificó **queued** y el JSON. Es la última observación de ese job, no garantía de su estado actual.
- La versión **v3** se desplegó según confirmó el usuario. **GET /api/device/claim** devolvió HTTP **405** (método incorrecto), como corresponde; **POST /api/device/complete** con DEVICE_TOKEN correcto y cuerpo vacío devolvió HTTP **400** (validación), mostrando que la autenticación funciona.
- **NO probado:** reclamar de verdad un job mediante POST /api/device/claim; completar con resultados válidos; consulta móvil de un job terminado; ejecución CLAP remota; latencia extremo a extremo; resiliencia de interrupciones. Tampoco se verificó byte a byte la versión activa del Worker frente al archivo Git.

## 5. Pendientes y riesgos conocidos

1. [REEMPLAZADO 2026-11-09 por `tools/traktor_live/conceptos_agente.py`, sección 8] **Agente local Python** pendiente. Debe ejecutarse solo en la máquina de Traktor, autenticarse con DEVICE_TOKEN, hacer polling HTTPS, leer conceptos, consultar la biblioteca CLAP existente y responder con resultados. No trasladar música o embeddings a Cloudflare. Usar **track_uid canónico**, no índices posicionales ni rutas como identificadores externos. Ver **docs/STATUS.md** y **AGENTS.md** para la organización de datos.
2. [REEMPLAZADO 2026-11-09 por el Worker v4: lease de 5 min, 3 intentos, generación del reclamo por `attempts`] **Recovery / lease:** claim no reencola automáticamente trabajos processing tras caída del agente. Diseñar expiración de lease, heartbeat/reintentos, límites de attempts, manejo de duplicados e idempotencia. Evitar reclamar trabajos reales hasta contar con procesamiento completo y pruebas.
3. [REEMPLAZADO 2026-11-09 por el Worker v4: consulta el trabajo y muestra progreso, notas y lista; sigue sin corrección fina del concepto] **Interfaz:** aún no consulta GET /api/jobs/{id} después de guardar, no muestra progreso ni lista de pistas y no ofrece regeneración/corrección fina del concepto.
4. **Exactitud semántica:** Gemma a veces devolvió JSON inválido; v3 tiene un reintento. El prompt prohíbe negativos inferidos, pero no hay un verificador semántico. Revisar prompts CLAP y cómo ponderar positivos/negativos; no restar negativos sin validación empírica.
5. **Seguridad:** la página pública no tiene control de acceso previo, solo la API mediante contraseña compartida. Falta limitación de intentos/rate limiting, protección reforzada frente a abuso, autenticación individual, rotación de credenciales e idempotencia del guardado. No insertar credenciales en URL, logs, Git, HTML ni archivos .env versionados. La firma de draft usa APP_PASSWORD: rotarla invalida borradores previos.
6. **Costos y operación:** la meta es el plan gratuito, no se garantiza ausencia de costes ni cuotas futuras. Confirmar disponibilidad/consumo de Workers AI, D1 y Workers antes de uso intensivo.
7. **Despliegue y trazabilidad:** el Worker se editó manualmente en el dashboard; no hay hoy sincronización automática ni garantía de que Git refleje cambios futuros en producción.

## 6. Cómo continuar desde VS Code / Claude Code

**Primera tarea recomendada:** inspeccionar los módulos reales de CLAP/recomendación local y preparar un agente Python separado, testeable, sin ejecutarlo todavía sobre la cola real. Investigar la persistencia de embeddings, identificación canónica de temas, forma y normalización de vectores texto/audio y criterios de ranking. **No crear un segundo pipeline de CLAP si ya existe uno reutilizable.** Primero tests con servidor HTTP simulado y D1 simulado, luego una prueba controlada en Cloudflare.

**Secuencia de aceptación del flujo:** recuperar job de prueba -> processing -> calcular recomendaciones locales (CLAP) -> POST complete -> done -> GET jobs/{id} -> visualizar nombres y scores en móvil. Las pruebas que muten D1 deben usar un job de prueba identificado y no consumir por error el concepto de producción anterior.

**Configuración GitHub/Cloudflare, por ahora pendiente:**

- Cloudflare permite conectar un Worker existente mediante **Settings > Builds > Connect**. Es recomendable, pero **no activarlo todavía**, ya que no hay configuración Wrangler versionada ni pruebas de despliegue.
- Convertir el archivo JS en fuente única controlada por Git. Preparar **cloudflare/wrangler.jsonc** o **wrangler.toml** con name exactamente **traktor-concept-ai**, main apuntando a **concept-ai-worker.js**, binding AI y binding D1 con el **database_id real** consultado en Cloudflare. Mantener APP_PASSWORD y DEVICE_TOKEN como secretos de plataforma, nunca en el archivo.
- Configurar root directory **cloudflare/**. Si se conecta el monorepo gbibbo/traktor, limitar **Build watch paths** a **cloudflare/** para que cambios a Python, datos o documentación no desplieguen por accidente. El comando habitual de deploy es **npx wrangler deploy** desde esa raíz, tras revisión.
- Verificar equivalencia de una compilación/preview con el Worker manual actual y confirmar los bindings antes de habilitar despliegue de producción automático. Las ediciones posteriores deberían realizarse en Git, no mezclarse indistintamente con cambios manuales en el dashboard.
- Documentación Cloudflare: https://developers.cloudflare.com/workers/ci-cd/builds/ ; https://developers.cloudflare.com/workers/ci-cd/builds/advanced-setups/ ; https://developers.cloudflare.com/workers/ci-cd/builds/build-watch-paths/ .

## 7. Estado al cierre

**Listo:** cuenta y Worker, binding Workers AI, D1 y tabla, secretos separados, interfaz móvil, conversación, generación de JSON, confirmación, creación de trabajos y rutas de comunicación dispositivo. **Validado parcialmente:** rutas nuevas de dispositivo y autenticación; aún no su procesamiento real. **No listo:** [REEMPLAZADO 2026-11-09 por la sección 8: todo esto salvo CI/CD quedó hecho y probado en local] agente en PC Traktor, ejecución CLAP desde el agente, recuperación de trabajos, presentación de resultados y CI/CD. **Próximo entorno de trabajo:** máquina con Traktor + VS Code + Claude Code. La presente documentación y el archivo JS son la referencia inicial, sujetos a comparación con la versión activa de Cloudflare.

## 8. Adenda 2026-11-09 (Claude Code): agente local y Worker v4

**Veredicto.** El flujo completo funciona en local con el código real del Worker, una D1 local, CLAP real sobre la biblioteca real y la página real a tamaño de teléfono. Lo único simulado fueron las dos llamadas al LLM. Falta publicar el Worker v4 y darle el `DEVICE_TOKEN` a este ordenador; ambas cosas requieren la cuenta de Gabriel. Si las listas son buenas lo decide Gabriel escuchando.

| Afirmación | Etiqueta | Evidencia (2026-11-09) |
| --- | --- | --- |
| Crear, reclamar, completar y leer un trabajo funciona contra el Worker real | EVIDENCIA OBSERVADA | `tests/v4/test_concept_worker_e2e.py`, 18 pasan, `wrangler dev` 4.159.0 con D1 local |
| La página del teléfono muestra propuesta, progreso, notas y 30 temas, sin errores de consola ni desborde horizontal | EVIDENCIA OBSERVADA | Edge sin cabeza, 390×844, con el agente y CLAP reales; capturas revisadas |
| La playlist de Traktor es válida y coincide con la lista del teléfono | EVIDENCIA OBSERVADA | `TRAKTOR ML Concepto NML.nml`: 30 entradas, mismo orden, todos los archivos y claves en la colección. Sin abrir en Traktor |
| Primera búsqueda 8 s; las siguientes, 62 ms | EVIDENCIA OBSERVADA | `conceptos_agente.py probar`, 1717 candidatos, biblioteca `musica`, índice versión 3 del 2026-11-06 |
| Importar torch en frío puede tardar 76 a 121 s, y el proceso con CLAP ocupa unos 950 MB | EVIDENCIA OBSERVADA | Dos medidas con caché fría; memoria del proceso después de codificar |
| Conceptos distintos dan listas distintas | EVIDENCIA OBSERVADA | 10 conceptos × 30: 248 temas distintos, solapamiento medio 1,3 de 30, máximo 8 (industrial y peak time) |
| Restar frases negativas no sirve para excluir voz | EVIDENCIA OBSERVADA, exploratoria | Vocal@30: 0,51 sin restar, 0,46 a 0,49 restando (peso 0,25 a 1); hasta 19 de 30 temas cambian. Solo voz, el único atributo con etiqueta |
| La calidad musical de las listas | NO CONCLUYENTE | Los géneros de los 30 primeros son plausibles, pero no reemplazan escuchar. Sin medida contra sets; ver T18 en el plan |
| El Worker v3 publicado hace lo mismo que el snapshot de Git | INFERENCIA, parcial | El HTML público tiene la misma lógica con otro formato. El backend no se pudo comparar sin credenciales |

**Qué cambió en el Worker (v4).** Lease de 5 minutos con 3 intentos; `job_id` opcional en `/api/device/claim`, para que una prueba no consuma conceptos reales; guardado idempotente (el id sale de la firma del borrador); resultados con `bpm`, `key`, `vocal`, `notes`, `pool` y `passed`, todos opcionales, así que el formato v3 sigue valiendo; y la página muestra la lista. No cambia la tabla de D1.

**Lo que el teléfono avisa que no aplica.** Energía relativa, porque no hay coordenada validada, y exclusiones que no sean voz. Ver `docs/DECISIONS.md`, 2026-11-09.

**Sigue abierto.** Límite de intentos de contraseña (punto 5) y despliegue automático desde GitHub (punto 7, apagado a propósito). Recetas: `cloudflare/README.md`.
