// Traktor Concept AI, v5. Fuente unica en Git; despliegue en cloudflare/README.md.
// Del texto en el telefono a una playlist nueva en Traktor. El LLM (aca) reescribe lo que pide el DJ en textos que
// CLAP sabe buscar y decide mirando lo que encontro cada busqueda; el ordenador (conceptos_agente.py) solo busca.
// Por que las instrucciones dicen lo que dicen: docs/reports/clap_texto_2026-11-09.md.
const MODEL = "@cf/google/gemma-4-26b-a4b-it";
const LEASE_MS = 5 * 60 * 1000;   // un trabajo reclamado y sin terminar vuelve a la cola pasado este tiempo
const MAX_ATTEMPTS = 3;           // reclamos seguidos antes de darlo por fallido
const MAX_RONDAS = 2;             // reescrituras: una y, si no convence, otra
const IDS = ["A", "B", "C"];
const UUID_RE = /^[0-9a-f-]{36}$/i;

const PROMPT_INTERPRETAR = `You rewrite a DJ's rough request into search texts for CLAP, a model that matches text to the SOUND of music.
CLAP only hears audio. It does not know occasions, places, dates, artist names or song titles.

The collection is club music: techno, tech house, melodic house and techno, deep house, house, indie dance, afro house, minimal, progressive house. There is little of anything else.

Steps:
1. Find what the DJ wants to HEAR.
2. Anything that cannot be heard (birthday, beach, 6am, gym, an artist) must be replaced by how it sounds.
3. Keep every instrument or sound the DJ named. If the word is rare, use the common name too. corneta -> trumpet, brass.
4. Translate everything to English.
5. Never add an instrument or sound the DJ did not ask for.

Write exactly 3 texts. All 3 say the same idea with DIFFERENT words.
Each text:
- has 6 to 14 words and describes sound only;
- starts with a genre word if there is one, otherwise with the main sound;
- uses concrete words: instruments, voices, rhythm, how it feels;
- never contains: "this is", "the sound of", "in the style of", "track", "song", "music by", names of people or places,
  or words for moments and occasions (party, birthday, sunset, beach, night, summer, gym, vibes).
To exclude voice, or to have less voice, write "without vocals" (never "few vocals" or "sparse vocals": any other mention of vocals adds voice).
Never use "no", "not" or "without" for anything else: describe what IS there.

Example
Request: quiero musica de cumpleaños con cornetas
{"entendi":"Música de fiesta alegre con trompetas","no_se_oye":["cumpleaños"],"nombre":"Fiesta con trompetas","textos":["festive house with bright trumpets and a bouncy cheerful groove","joyful festive music with loud brass instruments and hand claps","happy dance rhythm with a brass section and horn melodies"]}

Example
Request: algo oscuro tipo after sin voces
{"entendi":"Techno oscuro, lento e hipnótico, sin voces","no_se_oye":["after"],"nombre":"After oscuro","textos":["dark hypnotic minimal techno without vocals, slow and sparse","deep dark techno with a rolling bassline, without vocals","slow gloomy trance-like electronic beat, minimal and repetitive, without vocals"]}

Reply with JSON only, with these keys:
"entendi": in Spanish, what the DJ wants to hear, max 12 words.
"no_se_oye": in Spanish, only the words for things that cannot be heard (occasions, places, times, names) that you replaced. Usually empty.
"nombre": in Spanish, a playlist title, 2 to 5 words.
"textos": the 3 English texts.`;

const PROMPT_ELEGIR = `You check whether searches in a music collection found what a DJ asked for. You cannot hear the music. For each search you get what it found: genres, BPM, share of tracks with voice, "sounds like" words, and example tracks.

Decide using mainly the "sounds like" words and the genres. A search FITS if they match what the DJ wants. Ignore the artist names.
The collection has little of some styles (trance, disco, ambient). If a search found the closest thing the collection has, it fits.

If at least one search fits, answer:
{"veredicto":"ok","quedan":["A","C"],"resumen":"..."}
"quedan" lists only the searches that fit. "resumen" is in Spanish, max 15 words: what the playlist will sound like.

If none fits, answer:
{"veredicto":"mal","problema":"...","textos":["...","...","..."]}
"problema" is in Spanish, max 12 words: what went wrong. "textos" are 3 new English search texts of 6 to 14 words that fix the problem.
Rules for new texts: describe sound only, with concrete words (instruments, voices, rhythm). Never write "this is", "the sound of", "in the style of", "track", "song", or names. Do not repeat the texts that failed.

Reply with JSON only.`;

function reply(data, status = 200) {
  return Response.json(data, {
    status,
    headers: { "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff" }
  });
}

function fail(message, status = 400) {
  return reply({ error: message }, status);
}

async function readJson(request) {
  if (Number(request.headers.get("Content-Length") || 0) > 24000) return null;
  const text = await request.text();
  if (text.length > 24000) return null;
  try {
    const value = JSON.parse(text);
    return value && typeof value === "object" && !Array.isArray(value) ? value : null;
  } catch {
    return null;
  }
}

async function readOptionalJson(request) {
  if (Number(request.headers.get("Content-Length") || 0) > 24000) return null;
  const text = await request.text();
  if (text.length > 24000) return null;
  if (!text.trim()) return {};
  try {
    const value = JSON.parse(text);
    return value && typeof value === "object" && !Array.isArray(value) ? value : null;
  } catch {
    return null;
  }
}

const PROHIBIDO = /\b(this is|sound of|in the style of|tracks?|songs?|music by)\b/i;
// Lo que CLAP no oye: si el modelo lo escribe igual, ese texto se descarta.
const NO_SE_OYE = /\b(birthday|wedding|beach|sunset|sunrise|gym|workout|christmas|funeral|dinner|summer|winter|night|morning|afternoon|vibes?|party)\b/i;

// «Pocas voces» escrito de cualquier forma nombra la voz y la suma: lo único medido que la resta es "without vocals".
const MENOS_VOZ = /\b(?:with\s+)?(?:very\s+)?(?:few|minimal|sparse|little|less|subtle|light|occasional|barely any|no)\s+(?:vocal\s+(?:snippets|chops|samples)|vocals?|voices?|singing)\b/gi;

function textoValido(texto) {
  if (typeof texto !== "string") return null;
  const t = texto.trim().replace(/^["']|["']$/g, "").replace(/\.$/, "").replace(MENOS_VOZ, "without vocals");
  const palabras = t.split(/\s+/).length;
  // CLAP no entiende la negación (salvo "without vocals", medido): un texto con otra negación busca lo contrario.
  const negacion = /\b(no|not|non|never|without)\b/i.test(t.replace(/\bwithout vocals\b/gi, ""));
  return palabras >= 3 && palabras <= 20 && t.length <= 180 && !/[^\x20-\x7e]/.test(t) && !PROHIBIDO.test(t) &&
    !NO_SE_OYE.test(t) && !negacion ? t : null;
}

function validarTextos(lista) {
  if (!Array.isArray(lista)) return null;
  const textos = [...new Set(lista.map(textoValido).filter(Boolean))].slice(0, IDS.length);
  return textos.length >= 2 ? textos.map((texto, i) => ({ id: IDS[i], texto })) : null;
}

function textoCorto(value, min, max) {
  if (typeof value !== "string") return null;
  const t = value.trim();
  return t.length >= min && t.length <= max ? t : null;
}

function validarInterpretacion(raw) {
  if (!raw || typeof raw !== "object") return null;
  const entendi = textoCorto(raw.entendi, 3, 160);
  const nombre = textoCorto(raw.nombre, 2, 60);
  const textos = validarTextos(raw.textos);
  if (!entendi || !nombre || !textos) return null;
  const noSeOye = Array.isArray(raw.no_se_oye) ? raw.no_se_oye.map(x => textoCorto(x, 1, 40)).filter(Boolean).slice(0, 6) : [];
  return { entendi, nombre, no_se_oye: noSeOye, textos };
}

function validarDecision(raw, ids) {
  if (!raw || typeof raw !== "object") return null;
  if (raw.veredicto === "ok" && Array.isArray(raw.quedan)) {
    const quedan = ids.filter(id => raw.quedan.includes(id));
    return quedan.length ? { veredicto: "ok", quedan, resumen: textoCorto(raw.resumen, 3, 160) || "" } : null;
  }
  if (raw.veredicto === "mal") {
    const textos = validarTextos(raw.textos);
    return textos ? { veredicto: "mal", textos, problema: textoCorto(raw.problema, 3, 120) || "" } : null;
  }
  return null;
}

async function llmJson(env, system, user, validar, maxTokens, temperature) {
  let nota = "";
  for (let intento = 0; intento < 2; intento++) {
    try {
      const answer = await env.AI.run(MODEL, {
        messages: [{ role: "system", content: system }, { role: "user", content: user + nota }],
        response_format: { type: "json_object" },
        max_tokens: maxTokens,
        temperature,
        chat_template_kwargs: { enable_thinking: false }
      });
      const content = answer.choices?.[0]?.message?.content ?? answer.response;
      const raw = typeof content === "string"
        ? JSON.parse(content.trim().replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, ""))
        : content;
      const valido = validar(raw);
      if (valido) return valido;
    } catch {
      // un reintento si el modelo no devuelve JSON o se cae
    }
    nota = "\n\nYour previous answer broke the rules. Follow every rule and return valid JSON only.";
  }
  return null;
}

function interpretar(env, texto, anterior) {
  let pedido = `Request: ${texto}`;
  if (anterior) {
    pedido = `The DJ first asked: ${anterior.original}\nYou searched: ${anterior.textos.map(t => t.texto).join(" | ")}\nNow the DJ says: ${texto}\nRewrite the 3 texts for the corrected request. The correction is about sound, so "no_se_oye" stays as before or empty.`;
  }
  return llmJson(env, PROMPT_INTERPRETAR, pedido, validarInterpretacion, 420, 0.3);
}

function describirBusquedas(concepto, busquedas) {
  return concepto.textos.map(t => {
    const b = busquedas[t.id];
    const generos = Object.entries(b.generos).map(([g, n]) => `${g} ${n}`).join(", ") || "unknown";
    const bpm = b.bpm ? `${b.bpm[0]}-${b.bpm[2]}, typical ${b.bpm[1]}` : "unknown";
    const voz = b.voz === null ? "unknown" : `${Math.round(b.voz * 100)}%`;
    return `Search ${t.id}: "${t.texto}"\nGenres found: ${generos}\nBPM: ${bpm}\nWith voice: ${voz}\nSounds like: ${b.suena_a.join(", ")}\nExamples: ${b.temas.join("; ")}`;
  }).join("\n\n");
}

function elegir(env, concepto, busquedas) {
  const user = `The DJ asked: ${concepto.original}\nMeaning: ${concepto.entendi}\n\n${describirBusquedas(concepto, busquedas)}`;
  return llmJson(env, PROMPT_ELEGIR, user, raw => validarDecision(raw, concepto.textos.map(t => t.id)), 360, 0.1);
}

function validateResults(result) {
  if (!result || typeof result !== "object" || Array.isArray(result) || !Array.isArray(result.tracks) ||
    result.tracks.length > 30) return null;
  const optional = {};
  if (result.playlist !== undefined && result.playlist !== null) {
    if (typeof result.playlist !== "string" || result.playlist.length > 120) return null;
    optional.playlist = result.playlist;
  }
  for (const field of ["pool", "passed"]) {
    if (result[field] === undefined || result[field] === null) continue;
    if (!Number.isSafeInteger(result[field]) || result[field] < 0) return null;
    optional[field] = result[field];
  }
  if (result.notes !== undefined && result.notes !== null) {
    if (!Array.isArray(result.notes) || result.notes.length > 8 ||
      !result.notes.every(n => typeof n === "string" && n.trim() && n.length <= 200)) return null;
    optional.notes = result.notes;
  }
  const tracks = [];
  for (const item of result.tracks) {
    if (!item || typeof item !== "object" || Array.isArray(item) ||
      typeof item.track_id !== "string" || !item.track_id.trim() || item.track_id.length > 200 ||
      typeof item.title !== "string" || !item.title.trim() || item.title.length > 250 ||
      typeof item.artist !== "string" || item.artist.length > 200 ||
      typeof item.score !== "number" || !Number.isFinite(item.score) ||
      item.score < -1 || item.score > 1) return null;
    const track = { track_id: item.track_id, title: item.title, artist: item.artist, score: item.score };
    if (item.bpm !== undefined && item.bpm !== null) {
      if (typeof item.bpm !== "number" || !Number.isFinite(item.bpm) || item.bpm <= 0 || item.bpm > 400) return null;
      track.bpm = item.bpm;
    }
    if (item.key !== undefined && item.key !== null) {
      if (typeof item.key !== "string" || item.key.length > 6) return null;
      track.key = item.key;
    }
    if (item.vocal !== undefined && item.vocal !== null) {
      if (typeof item.vocal !== "boolean") return null;
      track.vocal = item.vocal;
    }
    tracks.push(track);
  }
  return { tracks, ...optional };
}

function validarBusquedas(result, ids) {
  if (!result || typeof result !== "object" || !result.busquedas || typeof result.busquedas !== "object") return null;
  const salida = {};
  const cadenas = (lista, max, largo) => Array.isArray(lista) && lista.length <= max &&
    lista.every(x => typeof x === "string" && x.length > 0 && x.length <= largo);
  for (const id of ids) {
    const b = result.busquedas[id];
    if (!b || typeof b !== "object" || !cadenas(b.suena_a, 8, 40) || !cadenas(b.temas, 8, 140)) return null;
    if (!b.generos || typeof b.generos !== "object" || Array.isArray(b.generos) || Object.keys(b.generos).length > 6 ||
      !Object.entries(b.generos).every(([g, n]) => g.length <= 60 && Number.isSafeInteger(n) && n >= 0)) return null;
    if (b.bpm !== null && !(Array.isArray(b.bpm) && b.bpm.length === 3 && b.bpm.every(x => Number.isFinite(x)))) return null;
    if (b.voz !== null && !(typeof b.voz === "number" && b.voz >= 0 && b.voz <= 1)) return null;
    salida[id] = { generos: b.generos, bpm: b.bpm, voz: b.voz, suena_a: b.suena_a, temas: b.temas };
  }
  return salida;
}

function siguientePaso(concepto, busquedas, decision) {
  const ronda = { ronda: concepto.ronda, textos: concepto.textos, busquedas, decision };
  const rondas = [...(concepto.rondas || []), ronda];
  if (decision && decision.veredicto === "mal" && concepto.ronda < MAX_RONDAS) {
    return { ...concepto, rondas, fase: "sondeo", ronda: concepto.ronda + 1, textos: decision.textos, problema: decision.problema };
  }
  // Se termina: con las que quedaron o, si el modelo no respondió o nada convenció, con todas.
  const quedan = decision && decision.veredicto === "ok" ? decision.quedan : concepto.textos.map(t => t.id);
  const textos = concepto.textos.filter(t => quedan.includes(t.id));
  const nota = decision ? (decision.veredicto === "ok" ? decision.resumen : "No encontré nada que encaje del todo; es lo más cercano.") : "";
  return { ...concepto, rondas, fase: "final", textos, resumen: nota };
}

async function handleDevice(request, env, path) {
  if (request.method !== "POST") return fail("Método no permitido", 405);
  if (!env.DEVICE_TOKEN || request.headers.get("X-Device-Token") !== env.DEVICE_TOKEN) return fail("No autorizado", 401);

  if (path === "/api/device/claim") {
    const data = await readOptionalJson(request);
    if (!data || (data.job_id !== undefined && (typeof data.job_id !== "string" || !UUID_RE.test(data.job_id)))) {
      return fail("Solicitud inválida");
    }
    const now = Date.now();
    const cutoff = now - (Number(env.LEASE_MS) || LEASE_MS);
    try {
      // Primero se recuperan los trabajos que un ordenador reclamó y no terminó.
      await env.DB.batch([
        env.DB.prepare(`UPDATE jobs SET status = 'failed', error = ?, updated_at = ?
          WHERE status = 'processing' AND claimed_at < ? AND attempts >= ?`)
          .bind("El ordenador no terminó este trabajo", now, cutoff, MAX_ATTEMPTS),
        env.DB.prepare(`UPDATE jobs SET status = 'queued', updated_at = ?
          WHERE status = 'processing' AND claimed_at < ? AND attempts < ?`)
          .bind(now, cutoff, MAX_ATTEMPTS)
      ]);
      // La selección y el cambio de estado ocurren en una sola instrucción SQL.
      // Con job_id se reclama solo ese trabajo: las pruebas no consumen conceptos reales.
      const statement = data.job_id
        ? env.DB.prepare(`
            UPDATE jobs SET status = 'processing', claimed_at = ?, updated_at = ?, attempts = attempts + 1
            WHERE id = ? AND status = 'queued'
            RETURNING id, concept_json, attempts
          `).bind(now, now, data.job_id.toLowerCase())
        : env.DB.prepare(`
            UPDATE jobs SET status = 'processing', claimed_at = ?, updated_at = ?, attempts = attempts + 1
            WHERE id = (SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at, id LIMIT 1)
              AND status = 'queued'
            RETURNING id, concept_json, attempts
          `).bind(now, now);
      const job = await statement.first();
      if (!job) return reply({ job: null });
      return reply({ job: { id: job.id, attempts: job.attempts, concept: JSON.parse(job.concept_json) } });
    } catch {
      return fail("No se pudo reclamar un trabajo", 500);
    }
  }

  if (path === "/api/device/complete") {
    const data = await readJson(request);
    if (!data || typeof data.id !== "string" || !UUID_RE.test(data.id) ||
      !Number.isSafeInteger(data.attempts) || data.attempts < 1) return fail("Solicitud inválida");
    const status = data.status;
    if (status !== "done" && status !== "failed") return fail("Estado inválido");
    try {
      const row = await env.DB.prepare("SELECT concept_json FROM jobs WHERE id = ? AND status = 'processing' AND attempts = ?")
        .bind(data.id, data.attempts).first();
      if (!row) return fail("Trabajo inexistente, no reclamado o ya finalizado", 409);
      const concepto = JSON.parse(row.concept_json);

      // Una ronda de búsquedas de prueba: decide el LLM y el trabajo vuelve a la cola para el paso siguiente.
      if (status === "done" && concepto.version === 2 && concepto.fase === "sondeo") {
        const busquedas = validarBusquedas(data.result, concepto.textos.map(t => t.id));
        if (!busquedas) return fail("Resultados inválidos");
        const decision = await elegir(env, concepto, busquedas);
        const siguiente = siguientePaso(concepto, busquedas, decision);
        const update = await env.DB.prepare(`
          UPDATE jobs SET status = 'queued', attempts = 0, claimed_at = NULL, concept_json = ?, updated_at = ?
          WHERE id = ? AND status = 'processing' AND attempts = ?
          RETURNING id
        `).bind(JSON.stringify(siguiente), Date.now(), data.id, data.attempts).first();
        if (!update) return fail("Trabajo inexistente, no reclamado o ya finalizado", 409);
        return reply({ id: data.id, status: "queued", fase: siguiente.fase });
      }

      let resultJson = null;
      let message = null;
      if (status === "done") {
        const result = validateResults(data.result);
        if (!result) return fail("Resultados inválidos");
        resultJson = JSON.stringify(result);
      } else {
        if (typeof data.error !== "string" || !data.error.trim() || data.error.length > 400) {
          return fail("Descripción del error inválida");
        }
        message = data.error.trim();
      }
      const update = await env.DB.prepare(`
        UPDATE jobs SET status = ?, result_json = ?, error = ?, updated_at = ?
        WHERE id = ? AND status = 'processing' AND attempts = ?
        RETURNING id
      `).bind(status, resultJson, message, Date.now(), data.id, data.attempts).first();
      if (!update) return fail("Trabajo inexistente, no reclamado o ya finalizado", 409);
      return reply({ id: data.id, status });
    } catch {
      return fail("No se pudo completar el trabajo", 500);
    }
  }
  return fail("No encontrado", 404);
}

async function handleApp(request, env, path) {
  if (!env.APP_PASSWORD || request.headers.get("X-App-Password") !== env.APP_PASSWORD) return fail("No autorizado", 401);

  if (path === "/api/health" && request.method === "GET") {
    try {
      const row = await env.DB.prepare("SELECT COUNT(*) AS total FROM jobs").first();
      return reply({ database: "connected", jobs: row.total });
    } catch {
      return fail("Base de datos no disponible", 500);
    }
  }

  const match = path.match(/^\/api\/jobs\/([0-9a-f-]{36})$/i);
  if (match && request.method === "GET") {
    try {
      const row = await env.DB.prepare(
        "SELECT id, status, concept_json, result_json, error, updated_at FROM jobs WHERE id = ?"
      ).bind(match[1]).first();
      if (!row) return fail("Trabajo no encontrado", 404);
      return reply({ id: row.id, status: row.status, concept: JSON.parse(row.concept_json),
        result: row.result_json ? JSON.parse(row.result_json) : null,
        error: row.error, updated_at: row.updated_at });
    } catch {
      return fail("No se pudo consultar el trabajo", 500);
    }
  }

  if (path === "/api/conceptos" && request.method === "POST") {
    const data = await readJson(request);
    if (!data) return fail("JSON inválido o demasiado grande");
    const texto = textoCorto(data.texto, 2, 400);
    if (!texto) return fail("Escribe qué quieres escuchar");
    let anterior = null;
    if (data.anterior !== undefined) {
      if (typeof data.anterior !== "string" || !UUID_RE.test(data.anterior)) return fail("Solicitud inválida");
      try {
        const row = await env.DB.prepare("SELECT concept_json FROM jobs WHERE id = ?").bind(data.anterior).first();
        if (row) anterior = JSON.parse(row.concept_json);
        if (anterior && anterior.version !== 2) anterior = null;
      } catch {
        return fail("No se pudo leer el concepto anterior", 500);
      }
    }
    const interpretado = await interpretar(env, texto, anterior);
    if (!interpretado) return fail("El modelo no pudo interpretar el pedido. Prueba otra vez.", 502);
    const original = anterior ? `${anterior.original} / ${texto}` : texto;
    const concepto = { version: 2, fase: "sondeo", ronda: 1, original, ...interpretado };
    const id = crypto.randomUUID();
    const now = Date.now();
    try {
      await env.DB.prepare(
        "INSERT INTO jobs (id, created_at, updated_at, status, concept_json) VALUES (?, ?, ?, 'queued', ?)"
      ).bind(id, now, now, JSON.stringify(concepto)).run();
      return reply({ id, status: "queued", concept: concepto }, 201);
    } catch {
      return fail("No se pudo guardar en D1", 500);
    }
  }
  return fail("No encontrado", request.method === "POST" || request.method === "GET" ? 404 : 405);
}

const HTML = `<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Traktor Concept AI</title>
<style>
body{font:16px system-ui;background:#111827;color:white;max-width:650px;margin:auto;padding:18px}
h2{margin-bottom:8px}small{color:#9ca3af}
input,textarea,button{box-sizing:border-box;width:100%;padding:13px;margin-top:12px;border-radius:9px;font:inherit}
input,textarea{background:#1f2937;color:white;border:1px solid #4b5563}
button{background:#0d9488;color:white;border:0;cursor:pointer}
button:disabled{opacity:.4;cursor:default}
#adjust{background:#334155}#again{background:transparent;border:1px solid #475569}
#card{display:none;margin-top:18px;border:1px solid #475569;border-radius:10px;padding:12px}
#status{margin:8px 0 0;color:#cbd5e1}#detail{margin:10px 0 0;font-size:14px;color:#cbd5e1}#detail p{margin:6px 0}
#detail ul{margin:4px 0 0;padding-left:18px;color:#9ca3af}
#tracks{margin:12px 0 0;padding-left:30px}#tracks li{margin:0 0 11px;overflow-wrap:anywhere}
#tracks .meta{display:block;color:#9ca3af;font-size:13px;font-variant-numeric:tabular-nums}
#after{display:none}
</style>
</head>
<body>
<h2>Traktor Concept AI</h2>
<small>Escribe lo que quieres escuchar, como te salga. El asistente lo traduce a algo que tu música entienda y crea una playlist nueva en Traktor.</small>
<input id="password" type="password" placeholder="Contraseña de acceso" autocomplete="off">
<textarea id="message" rows="3" placeholder="Quiero música de cumpleaños con cornetas..."></textarea>
<button id="send">Buscar en mi música</button>
<div id="card" aria-live="polite"><strong id="name"></strong>
<p id="status"></p><div id="detail"></div><ol id="tracks"></ol></div>
<div id="after"><textarea id="correction" rows="2" placeholder="¿Algo que corregir? Más oscuro, sin tanta voz..."></textarea>
<button id="adjust">Ajustar</button><button id="again">Nuevo concepto</button></div>
<script>
let currentId=null,followId=0;
const $=id=>document.getElementById(id);
const sendButton=$('send'),adjustButton=$('adjust'),againButton=$('again'),input=$('message'),correction=$('correction'),password=$('password');
const card=$('card'),nameText=$('name'),statusText=$('status'),detail=$('detail'),tracksList=$('tracks'),after=$('after');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
async function call(path,options){
  const response=await fetch(path,options);
  const result=await response.json();
  if(!response.ok)throw new Error(response.status===401?'Contraseña incorrecta':result.error||'Error HTTP '+response.status);
  return result;
}
const post=(path,data)=>call(path,{method:'POST',headers:{'Content-Type':'application/json','X-App-Password':password.value},body:JSON.stringify(data)});
const get=path=>call(path,{headers:{'X-App-Password':password.value}});
function line(label,value){const p=document.createElement('p');const b=document.createElement('b');b.textContent=label+' ';p.append(b,value);return p}
function showConcept(c){
  nameText.textContent=c.nombre||c.name||'Concepto';
  const parts=[];
  if(c.entendi)parts.push(line('Entendí:',c.entendi));
  if(c.no_se_oye&&c.no_se_oye.length)parts.push(line('Eso no se oye, lo traduje a sonido:',c.no_se_oye.join(', ')));
  if(c.problema)parts.push(line('Primer intento:',c.problema));
  if(c.resumen)parts.push(line('Va a sonar a:',c.resumen));
  if(c.textos){const ul=document.createElement('ul');for(const t of c.textos){const li=document.createElement('li');li.textContent=t.texto;ul.append(li)}parts.push(line('Busqué:',''),ul)}
  detail.replaceChildren(...parts);
}
function showTracks(result){
  tracksList.replaceChildren(...result.tracks.map(t=>{
    const li=document.createElement('li');
    li.append((t.artist?t.artist+' — ':'')+t.title);
    const meta=document.createElement('span');meta.className='meta';
    meta.textContent=[t.bpm?Math.round(t.bpm)+' BPM':null,t.key||null,t.vocal===true?'Vocal':null].filter(Boolean).join(' · ');
    li.append(meta);return li;
  }));
}
function progress(job,waited){
  const c=job.concept;
  if(job.status==='queued'&&waited>30000)return 'Sigue esperando. ¿Está encendido el ordenador, con Traktor abierto?';
  if(c.fase==='final')return 'Creando la playlist...';
  if(c.ronda>1)return 'No convenció; lo reescribió y prueba otra vez...';
  return 'Probando '+c.textos.length+' maneras de decirlo en tu música... La primera vez de la noche puede tardar un minuto.';
}
async function follow(id){
  const mine=++followId,start=Date.now();
  while(mine===followId&&Date.now()-start<6*60*1000){
    await sleep(2000);
    if(mine!==followId)return;
    let job;
    try{job=await get('/api/jobs/'+id)}catch(e){statusText.textContent='Sin respuesta, reintentando... ('+e.message+')';continue}
    showConcept(job.concept);
    if(job.status==='done'){
      const r=job.result;
      statusText.textContent=r.tracks.length?(r.playlist?'Playlist nueva en Traktor: '+r.playlist:r.tracks.length+' temas'):'No se encontró ningún tema.';
      showTracks(r);finish();return;
    }
    if(job.status==='failed'){statusText.textContent='No se pudo completar: '+(job.error||'error desconocido');finish();return}
    statusText.textContent=progress(job,Date.now()-start);
  }
  if(mine===followId){statusText.textContent='El ordenador no respondió a tiempo. El pedido queda guardado y se hará cuando vuelva.';finish()}
}
function finish(){sendButton.disabled=false;adjustButton.disabled=false;after.style.display='block'}
async function start(texto,anterior){
  sendButton.disabled=true;adjustButton.disabled=true;after.style.display='none';followId++;
  card.style.display='block';nameText.textContent=texto;detail.replaceChildren();tracksList.replaceChildren();statusText.textContent='Pensando cómo buscarlo...';
  card.scrollIntoView({behavior:'smooth',block:'start'});
  try{
    const body={texto};if(anterior)body.anterior=anterior;
    const job=await post('/api/conceptos',body);
    currentId=job.id;showConcept(job.concept);statusText.textContent='En cola...';follow(job.id);
  }catch(e){statusText.textContent='Error: '+e.message;finish()}
}
sendButton.onclick=()=>{const t=input.value.trim();if(t)start(t,null)};
adjustButton.onclick=()=>{const t=correction.value.trim();if(t&&currentId){correction.value='';start(t,currentId)}};
againButton.onclick=()=>{followId++;currentId=null;card.style.display='none';after.style.display='none';tracksList.replaceChildren();input.value='';input.focus();sendButton.disabled=false};
</script>
</body>
</html>`;

export default {
  async fetch(request, env) {
    const path = new URL(request.url).pathname;
    if (path.startsWith("/api/device/")) return handleDevice(request, env, path);
    if (path.startsWith("/api/")) return handleApp(request, env, path);
    if (path !== "/" || request.method !== "GET") return new Response("No encontrado", { status: 404 });
    return new Response(HTML, {
      headers: {
        "Content-Type": "text/html; charset=utf-8",
        "Cache-Control": "no-store",
        "X-Frame-Options": "DENY",
        "X-Content-Type-Options": "nosniff",
        "Content-Security-Policy": "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; form-action 'self'"
      }
    });
  }
};