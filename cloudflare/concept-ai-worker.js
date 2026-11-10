// Traktor Concept AI, v4 (2026-11-09). Fuente unica en Git: cloudflare/README.md explica como se despliega.
// v4 sobre v3: los trabajos abandonados por el ordenador vuelven a la cola (lease), guardar dos veces el
// mismo borrador no crea otro trabajo, /api/device/claim acepta job_id y el telefono muestra los resultados.
const MODEL = "@cf/google/gemma-4-26b-a4b-it";
const MAX_MESSAGES = 12;
const LEASE_MS = 5 * 60 * 1000;   // un trabajo reclamado y sin terminar vuelve a la cola pasado este tiempo
const MAX_ATTEMPTS = 3;           // reclamos antes de darlo por fallido
const UUID_RE = /^[0-9a-f-]{36}$/i;

function reply(data, status = 200) {
  return Response.json(data, {
    status,
    headers: { "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff" }
  });
}

function fail(message, status = 400) {
  return reply({ error: message }, status);
}

function appAuthorized(request, env) {
  return !!env.APP_PASSWORD && request.headers.get("X-App-Password") === env.APP_PASSWORD;
}

function deviceAuthorized(request, env) {
  return !!env.DEVICE_TOKEN && request.headers.get("X-Device-Token") === env.DEVICE_TOKEN;
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

function cleanMessages(messages) {
  if (!Array.isArray(messages) || messages.length < 1 || messages.length > MAX_MESSAGES) return null;
  if (!messages.every(m => m && (m.role === "user" || m.role === "assistant") &&
    typeof m.content === "string" && m.content.trim().length > 0 && m.content.length <= 1500)) return null;
  if (!messages.some(m => m.role === "user")) return null;
  return messages.map(m => ({ role: m.role, content: m.content.trim() }));
}

function normalizeConcept(raw) {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const name = typeof raw.name === "string" ? raw.name.trim() : "";
  const description = typeof raw.description === "string" ? raw.description.trim() : "";
  const phrases = (value, min) => Array.isArray(value) && value.length >= min && value.length <= 4 &&
    value.every(s => typeof s === "string" && s.trim().length >= 8 && s.length <= 180);
  if (name.length < 3 || name.length > 100 || description.length < 15 || description.length > 700 ||
    !phrases(raw.positive_prompts, 1) || !phrases(raw.negative_prompts, 0)) return null;
  const c = raw.constraints;
  if (!c || typeof c !== "object" || Array.isArray(c)) return null;
  const energy = c.energy ?? "unspecified";
  const vocals = c.vocals ?? "unspecified";
  const bpm = c.bpm ?? "unspecified";
  if (!["maintain", "increase", "decrease", "unspecified"].includes(energy) ||
    !["present", "absent", "unspecified"].includes(vocals) ||
    !["near_current", "unspecified"].includes(bpm)) return null;
  return {
    version: 1,
    name,
    description,
    positive_prompts: raw.positive_prompts.map(s => s.trim()),
    negative_prompts: raw.negative_prompts.map(s => s.trim()),
    constraints: { energy, vocals, bpm }
  };
}

async function keyFromSecret(secret) {
  return crypto.subtle.importKey("raw", new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" }, false, ["sign", "verify"]);
}

async function signDraft(env, concept, expires) {
  const bytes = await crypto.subtle.sign("HMAC", await keyFromSecret(env.APP_PASSWORD),
    new TextEncoder().encode(JSON.stringify({ concept, expires })));
  return [...new Uint8Array(bytes)].map(x => x.toString(16).padStart(2, "0")).join("");
}

async function verifyDraft(env, concept, expires, signature) {
  if (!Number.isSafeInteger(expires) || expires < Date.now() ||
    expires > Date.now() + 21 * 60 * 1000 ||
    typeof signature !== "string" || !/^[0-9a-f]{64}$/i.test(signature)) return false;
  const signatureBytes = Uint8Array.from(signature.match(/.{2}/g), x => parseInt(x, 16));
  return crypto.subtle.verify("HMAC", await keyFromSecret(env.APP_PASSWORD), signatureBytes,
    new TextEncoder().encode(JSON.stringify({ concept, expires })));
}

function jobIdFromSignature(signature) {
  // La misma firma siempre da el mismo id: guardar dos veces un borrador no crea dos trabajos.
  const s = signature.toLowerCase();
  return `${s.slice(0, 8)}-${s.slice(8, 12)}-4${s.slice(13, 16)}-8${s.slice(17, 20)}-${s.slice(20, 32)}`;
}

function validateResults(result) {
  if (!result || typeof result !== "object" || Array.isArray(result) || !Array.isArray(result.tracks) ||
    result.tracks.length > 30) return null;
  const optional = {};
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

async function handleDevice(request, env, path) {
  if (request.method !== "POST") return fail("Método no permitido", 405);
  if (!deviceAuthorized(request, env)) return fail("No autorizado", 401);

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
    try {
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
  if (!appAuthorized(request, env)) return fail("No autorizado", 401);

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

  if (request.method !== "POST") return fail("Método no permitido", 405);
  const data = await readJson(request);
  if (!data) return fail("JSON inválido o demasiado grande");

  if (path === "/api/chat") {
    const messages = cleanMessages(data.messages);
    if (!messages) return fail("Mensajes inválidos");
    const userTurns = messages.filter(m => m.role === "user").length;
    const system = [
      "Eres el asistente musical de Traktor ML, utilizado mientras un DJ está pinchando.",
      "Responde en español, con frases muy breves.",
      "Interpreta la intención musical, sin inventar preferencias ni mencionar canciones concretas.",
      "Si es imprescindible, haz UNA pregunta de aclaración breve.",
      "No afirmes haber buscado pistas."
    ];
    if (userTurns >= 2) system.push("No formules más preguntas. Resume la intención y pregunta si es correcta.");
    try {
      const answer = await env.AI.run(MODEL, {
        messages: [{ role: "system", content: system.join("\n") }, ...messages],
        max_tokens: 250,
        chat_template_kwargs: { enable_thinking: false }
      });
      const content = answer.choices?.[0]?.message?.content;
      if (typeof content !== "string" || !content.trim()) return fail("Respuesta vacía del modelo", 502);
      return reply({ reply: content.trim() });
    } catch {
      return fail("No se pudo consultar el modelo", 502);
    }
  }

  if (path === "/api/draft") {
    const messages = cleanMessages(data.messages);
    if (!messages) return fail("Primero describe el concepto musical");
    const system = [
      "Extrae del diálogo las preferencias musicales expresas del DJ.",
      "Devuelve solamente un JSON válido con:",
      "name: título corto en español, de 3 a 100 caracteres.",
      "description: resumen específico afirmativo en español, de 15 a 700 caracteres.",
      "positive_prompts: array de 1 a 4 frases completas en inglés, cada una de 8 a 180 caracteres, que describan el sonido musical para CLAP.",
      "negative_prompts: array de 0 a 4 frases completas en inglés, cada una de 8 a 180 caracteres, SOLO para elementos que el DJ haya excluido expresamente.",
      "constraints: objeto con energy, vocals y bpm.",
      "energy: maintain, increase, decrease o unspecified. Agresivo no significa increase. Solo increase o decrease si pide un cambio respecto al tema actual.",
      "vocals: present, absent o unspecified. bpm: near_current o unspecified.",
      "No inventes preferencias ni negaciones y usa unspecified cuando no hay datos.",
      "No hagas preguntas ni introduzcas resultados de búsqueda."
    ].join("\n");
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const answer = await env.AI.run(MODEL, {
          messages: [{ role: "system", content: system }, ...messages,
            { role: "user", content: "Escribe ahora únicamente el objeto JSON del concepto musical." }],
          response_format: { type: "json_object" },
          max_tokens: 650,
          temperature: 0.2,
          chat_template_kwargs: { enable_thinking: false }
        });
        const content = answer.choices?.[0]?.message?.content;
        if (typeof content !== "string") continue;
        const raw = JSON.parse(content.trim().replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, ""));
        const concept = normalizeConcept(raw);
        if (!concept) continue;
        const expires = Date.now() + 20 * 60 * 1000;
        return reply({ concept, expires, signature: await signDraft(env, concept, expires) });
      } catch {
        // Un único reintento si el modelo no devuelve JSON válido.
      }
    }
    return fail("El modelo no generó un concepto válido. Vuelve a prepararlo.", 502);
  }

  if (path === "/api/jobs") {
    const concept = normalizeConcept(data.concept);
    if (!concept || data.confirmed !== true ||
      !(await verifyDraft(env, concept, data.expires, data.signature))) {
      return fail("Prepara y confirma el concepto antes de guardarlo");
    }
    const id = jobIdFromSignature(data.signature);
    const now = Date.now();
    try {
      await env.DB.prepare(
        "INSERT INTO jobs (id, created_at, updated_at, status, concept_json) VALUES (?, ?, ?, 'queued', ?) ON CONFLICT(id) DO NOTHING"
      ).bind(id, now, now, JSON.stringify(concept)).run();
      const row = await env.DB.prepare("SELECT status FROM jobs WHERE id = ?").bind(id).first();
      return reply({ id, status: row.status }, 201);
    } catch {
      return fail("No se pudo guardar en D1", 500);
    }
  }
  return fail("No encontrado", 404);
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
#prepare{background:#334155}#save{background:#2563eb}#again{background:#334155;display:none}#chat{min-height:120px;padding-top:15px}
.msg{padding:12px;margin:10px 0;border-radius:10px;background:#1f2937;white-space:pre-wrap;overflow-wrap:anywhere}
.user{background:#134e4a}#proposal,#results{display:none;margin-top:18px;border:1px solid #475569;border-radius:10px;padding:12px}
#proposal pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;font-size:14px;margin:10px 0 0}
#status{margin:8px 0 0;color:#cbd5e1}#notes{margin:10px 0 0;padding-left:18px;color:#fbbf24;font-size:14px}
#tracks{margin:12px 0 0;padding-left:30px}#tracks li{margin:0 0 11px;overflow-wrap:anywhere}
#tracks .meta{display:block;color:#9ca3af;font-size:13px;font-variant-numeric:tabular-nums}
#footnote{display:none;margin-top:6px}
</style>
</head>
<body>
<h2>Traktor Concept AI</h2>
<small>Describe lo que quieres escuchar y prepara el concepto. No hace falta responder a todas las preguntas.</small>
<input id="password" type="password" placeholder="Contraseña de acceso" autocomplete="off">
<div id="chat" aria-live="polite"></div>
<textarea id="message" rows="3" placeholder="Quiero techno oscuro e hipnótico, sin voces..."></textarea>
<button id="send">Enviar</button>
<button id="prepare" disabled>Preparar concepto</button>
<div id="proposal"><strong>Concepto para revisar</strong><pre id="proposalText"></pre>
<button id="save" disabled>Confirmar y buscar en mi música</button></div>
<div id="results" aria-live="polite"><strong id="resultName"></strong>
<p id="status"></p><ul id="notes"></ul><ol id="tracks"></ol>
<small id="footnote">El número es el parecido entre el texto y el audio (CLAP). Sirve para ordenar; no es un porcentaje.</small>
</div>
<button id="again">Nuevo concepto</button>
<script>
let history=[],draft=null,followId=0;
const $=id=>document.getElementById(id);
const chat=$('chat'),sendButton=$('send'),prepareButton=$('prepare'),saveButton=$('save'),againButton=$('again');
const input=$('message'),password=$('password'),proposal=$('proposal'),proposalText=$('proposalText');
const results=$('results'),resultName=$('resultName'),statusText=$('status'),notesList=$('notes'),tracksList=$('tracks'),footnote=$('footnote');
const ENERGY={maintain:'mantener',increase:'subir',decrease:'bajar'};
const VOCALS={present:'con voces',absent:'sin voces'};
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
function show(role,content){
  const div=document.createElement('div');div.className='msg '+role;div.textContent=content;
  chat.appendChild(div);div.scrollIntoView({behavior:'smooth',block:'nearest'});
}
function clearDraft(){draft=null;saveButton.disabled=true;proposal.style.display='none'}
function clearResults(){followId++;results.style.display='none';statusText.textContent='';notesList.replaceChildren();tracksList.replaceChildren();footnote.style.display='none';againButton.style.display='none'}
async function call(path,options){
  const response=await fetch(path,options);
  const result=await response.json();
  if(!response.ok)throw new Error(response.status===401?'Contraseña incorrecta':result.error||'Error HTTP '+response.status);
  return result;
}
const post=(path,data)=>call(path,{method:'POST',headers:{'Content-Type':'application/json','X-App-Password':password.value},body:JSON.stringify(data)});
const get=path=>call(path,{headers:{'X-App-Password':password.value}});
function describe(c){
  const lines=[c.name,'',c.description,'','Lo que se va a buscar:',...c.positive_prompts.map(p=>'• '+p)];
  if(c.negative_prompts.length)lines.push('','Excluir:',...c.negative_prompts.map(p=>'• '+p));
  const k=c.constraints,extra=[];
  if(ENERGY[k.energy])extra.push('energía: '+ENERGY[k.energy]);
  if(VOCALS[k.vocals])extra.push(VOCALS[k.vocals]);
  if(k.bpm==='near_current')extra.push('BPM cerca del tema actual');
  if(extra.length)lines.push('','Condiciones: '+extra.join(', '));
  return lines.join('\\n');
}
function showTracks(result){
  notesList.replaceChildren(...(result.notes||[]).map(text=>{const li=document.createElement('li');li.textContent=text;return li}));
  tracksList.replaceChildren(...result.tracks.map(t=>{
    const li=document.createElement('li');
    li.append((t.artist?t.artist+' — ':'')+t.title);
    const meta=document.createElement('span');meta.className='meta';
    meta.textContent=[t.bpm?Math.round(t.bpm)+' BPM':null,t.key||null,t.vocal===true?'Vocal':null,t.score.toFixed(2)].filter(Boolean).join(' · ');
    li.append(meta);return li;
  }));
  footnote.style.display=result.tracks.length?'block':'none';
}
async function follow(id,name){
  const mine=++followId,start=Date.now();
  results.style.display='block';resultName.textContent=name;statusText.textContent='En cola...';
  results.scrollIntoView({behavior:'smooth',block:'start'});
  while(mine===followId&&Date.now()-start<6*60*1000){
    await sleep(2500);
    if(mine!==followId)return;
    let job;
    try{job=await get('/api/jobs/'+id)}catch(e){statusText.textContent='Sin respuesta, reintentando... ('+e.message+')';continue}
    if(job.status==='done'){
      const n=job.result.tracks.length;
      statusText.textContent=n?n+' temas de tu biblioteca, del más parecido al menos parecido.':'Ningún tema pasó las condiciones.';
      showTracks(job.result);againButton.style.display='block';return;
    }
    if(job.status==='failed'){statusText.textContent='No se pudo completar: '+(job.error||'error desconocido');againButton.style.display='block';return}
    const waited=Date.now()-start;
    statusText.textContent=job.status==='processing'?'Buscando en tu biblioteca... La primera búsqueda de la noche puede tardar uno o dos minutos.':
      waited>30000?'Sigue en cola. ¿Está encendido el ordenador, con Traktor abierto?':'En cola...';
  }
  if(mine===followId){statusText.textContent='El ordenador no respondió a tiempo. El concepto queda guardado y se procesará cuando vuelva.';againButton.style.display='block'}
}
sendButton.onclick=async()=>{
  const content=input.value.trim();if(!content)return;
  clearDraft();show('user',content);input.value='';
  const messages=[...history,{role:'user',content}].slice(-12);
  sendButton.disabled=true;prepareButton.disabled=true;
  try{
    const result=await post('/api/chat',{messages});show('assistant',result.reply);
    history=[...messages,{role:'assistant',content:result.reply}].slice(-12);
  }catch(e){show('assistant','Error: '+e.message);history=messages}
  finally{sendButton.disabled=false;prepareButton.disabled=!history.some(m=>m.role==='user')}
};
prepareButton.onclick=async()=>{
  clearDraft();sendButton.disabled=true;prepareButton.disabled=true;prepareButton.textContent='Preparando...';
  try{
    draft=await post('/api/draft',{messages:history});
    proposalText.textContent=describe(draft.concept);
    proposal.style.display='block';saveButton.disabled=false;
    proposal.scrollIntoView({behavior:'smooth',block:'start'});
  }catch(e){show('assistant','Error al preparar concepto: '+e.message)}
  finally{sendButton.disabled=false;prepareButton.disabled=false;prepareButton.textContent='Preparar concepto'}
};
saveButton.onclick=async()=>{
  if(!draft||!confirm('¿Buscar este concepto en tu música?'))return;
  saveButton.disabled=true;
  try{
    const name=draft.concept.name;
    const result=await post('/api/jobs',{concept:draft.concept,expires:draft.expires,signature:draft.signature,confirmed:true});
    clearDraft();follow(result.id,name);
  }catch(e){show('assistant','Error al guardar: '+e.message);saveButton.disabled=false}
};
againButton.onclick=()=>{clearDraft();clearResults();history=[];chat.replaceChildren();prepareButton.disabled=true;input.focus()};
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