// Traktor Concept AI, v6. Fuente unica en Git; despliegue en cloudflare/README.md.
// Del telefono a una playlist nueva en Traktor. El pedido es un formulario (config/telefono/formulario.json): los presets
// lo llenan sin pasar por el LLM; el texto libre pasa por el LLM, que reescribe lo que pide el DJ en textos que CLAP sabe
// buscar y decide mirando lo que encontro cada busqueda. El ordenador (conceptos_agente.py) ejecuta el formulario.
// Por que las instrucciones dicen lo que dicen: docs/reports/clap_texto_2026-11-09.md.
// v6 (2026-11-13, linea B del plan v2): presets, formulario en el trabajo (version 3), el ordenador cuenta que suena y
// el catalogo de la biblioteca en numeros (tabla estado), y la pagina pide la contrasena una sola vez.
import FORMULARIO from "../config/telefono/formulario.json";
import PRESETS from "../config/telefono/presets.json";
import REGLAMENTO from "../config/telefono/reglamento.json";
import { interpretarReglas, normalizar } from "./reglas.js";
import { MODEL, IDS, PROMPT_INTERPRETAR, PROMPT_ELEGIR, PARAMS_INTERPRETAR, PARAMS_ELEGIR, REINTENTO, textoCorto,
  validarInterpretacion, validarDecision, pedidoInterpretar, leerJson } from "./llm.js";

const CAMELOT_RE = /^(1[0-2]|[1-9])[AB]$/;
const UID_TEMA_RE = /^[0-9A-Za-z_-]{1,80}$/;
const ORDENADOR_VIVO_MS = 12 * 60 * 1000;   // el ordenador manda que suena al cambiar y cada 5 min
const LEASE_MS = 5 * 60 * 1000;   // un trabajo reclamado y sin terminar vuelve a la cola pasado este tiempo
const MAX_ATTEMPTS = 3;           // reclamos seguidos antes de darlo por fallido
const MAX_RONDAS = 2;             // reescrituras: una y, si no convence, otra
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

// ---------------------------------------------------------------- formulario (el mismo contrato que formulario.py)

const esNumero = x => typeof x === "number" && Number.isFinite(x);
const esObjeto = x => x !== null && typeof x === "object" && !Array.isArray(x);

function formularioVacio() {
  return {
    version: 1,
    filtros: { bpm: null, generos: null, tonalidad: null, voz: "da_igual",
      otros: { anio: null, sellos: [], artistas: [], artistas_excluir: [], excluir_tocados: true } },
    referencia: { tipo: "ninguna" }, pesos: { perfil: "por_defecto", ajustes: [] }, momento: null, texto: [],
    orden: "puntaje", largo: FORMULARIO.largo.defecto
  };
}

// Normaliza un formulario: todo valor fuera de su lista se descarta y deja un problema. generosValidos: los del catalogo
// que mando el ordenador (null: no se comprueban; el ordenador vuelve a validar).
function validarFormulario(raw, generosValidos) {
  const C = FORMULARIO, p = [], f = formularioVacio();
  if (!esObjeto(raw)) return { form: f, problemas: ["el formulario no es un objeto"] };
  if (raw.version !== undefined && raw.version !== 1) return { form: f, problemas: ["version desconocida"] };
  for (const k of Object.keys(raw)) {
    if (!["version", "filtros", "referencia", "pesos", "momento", "texto", "orden", "largo"].includes(k)) p.push("casilla desconocida: " + k);
  }
  const fil = esObjeto(raw.filtros) ? raw.filtros : {};
  const b = fil.bpm;
  if (b != null) {
    if (esObjeto(b) && Object.keys(b).length === 1 && b.relativo !== undefined) {
      if (Object.hasOwn(C.bpm.relativo, b.relativo)) f.filtros.bpm = { relativo: b.relativo }; else p.push("bpm.relativo");
    } else if (esObjeto(b) && Object.keys(b).length === 1 && Array.isArray(b.rango) && b.rango.length === 2) {
      const [lo, hi] = b.rango, [L, H] = C.bpm.limites;
      const ok = x => x === null || (esNumero(x) && x >= L && x <= H);
      if (ok(lo) && ok(hi) && (lo !== null || hi !== null) && (lo === null || hi === null || lo <= hi)) f.filtros.bpm = { rango: [lo, hi] };
      else p.push("bpm.rango");
    } else p.push("bpm");
  }
  const g = fil.generos;
  if (g != null) {
    if (!esObjeto(g)) p.push("generos");
    else {
      const lista = k => {
        const out = [];
        for (const x of Array.isArray(g[k]) ? g[k] : []) {
          if (typeof x === "string" && (!generosValidos || generosValidos.includes(x)) && !out.includes(x)) out.push(x);
          else p.push("generos." + k);
        }
        return out.slice(0, C.generos.max);
      };
      const incluir = lista("incluir"), excluir = lista("excluir");
      const modo = g.modo === undefined ? "blando" : g.modo;
      if (!C.generos.modos.includes(modo)) p.push("generos.modo");
      if (incluir.length || excluir.length) f.filtros.generos = { incluir, excluir, modo: C.generos.modos.includes(modo) ? modo : "blando" };
    }
  }
  const t = fil.tonalidad;
  if (t != null) {
    if (C.tonalidad.includes(t)) f.filtros.tonalidad = t;
    else if (esObjeto(t) && typeof t.clave === "string" && CAMELOT_RE.test(t.clave.trim().toUpperCase())) f.filtros.tonalidad = { clave: t.clave.trim().toUpperCase() };
    else p.push("tonalidad");
  }
  if (fil.voz !== undefined) { if (C.voz.includes(fil.voz)) f.filtros.voz = fil.voz; else p.push("voz"); }
  const o = esObjeto(fil.otros) ? fil.otros : {};
  if (o.anio != null) {
    const [L, H] = C.otros.anio, ent = x => x === null || (Number.isInteger(x) && x >= L && x <= H);
    const [lo, hi] = Array.isArray(o.anio) ? o.anio : [undefined, undefined];
    if (Array.isArray(o.anio) && o.anio.length === 2 && ent(lo) && ent(hi) && (lo !== null || hi !== null) && (lo === null || hi === null || lo <= hi)) f.filtros.otros.anio = [lo, hi];
    else p.push("otros.anio");
  }
  for (const k of ["sellos", "artistas", "artistas_excluir"]) {
    if (o[k] == null) continue;
    if (!Array.isArray(o[k])) { p.push("otros." + k); continue; }
    for (const x of o[k]) {
      const s = typeof x === "string" ? x.trim() : "";
      if (s.length >= 1 && s.length <= C.otros.max_caracteres && !f.filtros.otros[k].includes(s)) f.filtros.otros[k].push(s); else p.push("otros." + k);
    }
    f.filtros.otros[k] = f.filtros.otros[k].slice(0, C.otros.max_nombres);
  }
  if (o.excluir_tocados !== undefined) { if (typeof o.excluir_tocados === "boolean") f.filtros.otros.excluir_tocados = o.excluir_tocados; else p.push("otros.excluir_tocados"); }
  const r = raw.referencia;
  if (r != null) {
    if (!esObjeto(r) || !C.referencia.includes(r.tipo)) p.push("referencia");
    else if (r.tipo === "ninguna" || r.tipo === "sonando") f.referencia = { tipo: r.tipo };
    else {
      const out = { tipo: r.tipo }, busca = typeof r.busca === "string" ? r.busca.trim() : "";
      if (busca.length >= 1 && busca.length <= C.otros.max_caracteres) out.busca = busca;
      if (r.tipo === "tema" && typeof r.uid === "string" && UID_TEMA_RE.test(r.uid)) out.uid = r.uid;
      if (out.busca || out.uid) f.referencia = out; else p.push("referencia." + r.tipo);
    }
  }
  const w = raw.pesos;
  if (w != null) {
    if (!esObjeto(w)) p.push("pesos");
    else {
      const perfil = w.perfil === undefined ? "por_defecto" : w.perfil;
      if (C.pesos.perfiles.includes(perfil)) f.pesos.perfil = perfil; else p.push("pesos.perfil");
      for (const a of Array.isArray(w.ajustes) ? w.ajustes : []) {
        if (esObjeto(a) && Object.keys(a).length === 2 && C.pesos.factores.includes(a.factor) && C.pesos.niveles.includes(a.nivel) &&
          !f.pesos.ajustes.some(x => x.factor === a.factor)) f.pesos.ajustes.push({ factor: a.factor, nivel: a.nivel });
        else p.push("pesos.ajustes");
      }
      f.pesos.ajustes = f.pesos.ajustes.slice(0, C.pesos.max_ajustes);
    }
  }
  if (raw.momento != null) { if (C.momentos.includes(raw.momento)) f.momento = raw.momento; else p.push("momento"); }
  if (raw.texto != null) {
    if (!Array.isArray(raw.texto)) p.push("texto");
    else for (const x of raw.texto) {
      const s = typeof x === "string" ? x.trim() : "";
      if (s.length >= 3 && s.length <= C.texto.max_caracteres && /^[\x20-\x7e]+$/.test(s) && !f.texto.includes(s)) f.texto.push(s); else p.push("texto");
    }
    f.texto = f.texto.slice(0, C.texto.max);
  }
  if (raw.orden !== undefined) { if (C.orden.includes(raw.orden)) f.orden = raw.orden; else p.push("orden"); }
  if (raw.largo !== undefined) { if (Number.isInteger(raw.largo) && raw.largo >= 1 && raw.largo <= C.largo.max) f.largo = raw.largo; else p.push("largo"); }
  return { form: f, problemas: p };
}

function fundir(a, b) {
  const out = { ...a };
  for (const [k, v] of Object.entries(b)) out[k] = esObjeto(v) && esObjeto(out[k]) ? fundir(out[k], v) : v;
  return out;
}

// Los presets elegidos, juntos (como formulario.componer): en "bpm" y "momento" vale uno solo, gana el ultimo.
// Un genero elegido como preset filtra duro.
function componer(ids, generos) {
  const porId = Object.fromEntries(PRESETS.presets.map(x => [x.id, x]));
  let form = {}, etiquetas = [];
  const desconocidos = [], grupos = {};
  for (const id of ids) {
    const preset = porId[id];
    if (!preset) { desconocidos.push(id); continue; }
    if ((preset.grupo === "bpm" || preset.grupo === "momento") && grupos[preset.grupo]) etiquetas = etiquetas.filter(e => e !== grupos[preset.grupo]);
    grupos[preset.grupo] = preset.etiqueta;
    form = fundir(form, preset.formulario);
    etiquetas.push(preset.etiqueta);
  }
  const gen = [...new Set(generos)];
  if (gen.length) {
    form = fundir(form, { filtros: { generos: { incluir: gen, modo: "duro" } } });
    etiquetas.push(...gen);
  }
  return { form, etiquetas, desconocidos };
}

// Un pedido que no arma lista: ambiguo (se pregunta con opciones de un toque), algo que no hay, o sin nada que buscar.
// Valores por defecto de la semilla del banco (pedidos_telefono_borrador.md): nunca se arma una lista de otra cosa.
function respuestaSinLista(salida, texto) {
  if (salida === "pregunta") {
    const t = normalizar(texto);
    const clave = Object.keys(REGLAMENTO.opciones_pregunta).find(k => t.includes(k));
    const ids = clave ? REGLAMENTO.opciones_pregunta[clave] : ["mas_rapido", "mas_lento", "pico"];
    const porId = Object.fromEntries(PRESETS.presets.map(p => [p.id, p]));
    return { salida, mensaje: "¿Hacia dónde quieres cambiar? Elige una opción o dilo de otra forma.",
      opciones: ids.filter(id => porId[id]).map(id => ({ id, etiqueta: porId[id].etiqueta })) };
  }
  if (salida === "no_hay") return { salida, mensaje: "Eso no está en tu biblioteca: no armo una lista de otra cosa.", opciones: [] };
  return { salida: "pide_detalle", mensaje: "Cuéntame un poco más: un género, un BPM, un momento o cómo quieres que suene.", opciones: [] };
}

// ---------------------------------------------------------------- estado que manda el ordenador (tabla estado)

function validarSonando(x) {
  if (x === null) return null;
  if (!esObjeto(x) || typeof x.titulo !== "string" || x.titulo.length > 160 || typeof x.artista !== "string" || x.artista.length > 120) return undefined;
  const bpm = esNumero(x.bpm) && x.bpm > 0 && x.bpm <= 400 ? x.bpm : null;
  const clave = typeof x.clave === "string" && CAMELOT_RE.test(x.clave) ? x.clave : null;
  const deck = typeof x.deck === "string" && /^[A-D]?$/.test(x.deck) ? x.deck : "";
  return { titulo: x.titulo, artista: x.artista, bpm, clave, deck };
}

// Solo numeros y los nombres de genero (etiquetas de Beatport): nada de temas, artistas ni rutas.
function validarCatalogo(x) {
  if (!esObjeto(x) || x.version !== 1 || !Number.isSafeInteger(x.temas) || !Array.isArray(x.generos) || x.generos.length > 80) return null;
  const generos = [];
  for (const g of x.generos) {
    if (!esObjeto(g) || typeof g.nombre !== "string" || !g.nombre || g.nombre.length > 60 || !Number.isSafeInteger(g.temas) || g.temas < 0) return null;
    generos.push({ nombre: g.nombre, temas: g.temas });
  }
  const numeros = v => esObjeto(v) ? Object.fromEntries(Object.entries(v).filter(([k, n]) => k.length <= 20 && (esNumero(n) || n === null)).slice(0, 12)) : null;
  return { version: 1, temas: x.temas, generos, sin_genero: Number.isSafeInteger(x.sin_genero) ? x.sin_genero : null,
    bpm: numeros(x.bpm), voz: numeros(x.voz), anio: numeros(x.anio), campos: numeros(x.campos) };
}

async function guardarEstado(env, clave, valor) {
  try {
    await env.DB.prepare(`INSERT INTO estado (clave, valor, updated_at) VALUES (?, ?, ?)
      ON CONFLICT(clave) DO UPDATE SET valor = excluded.valor, updated_at = excluded.updated_at`)
      .bind(clave, JSON.stringify(valor), Date.now()).run();
  } catch {
    // sin la tabla estado (schema viejo) el Worker sigue andando: solo no muestra que suena
  }
}

async function leerEstado(env) {
  try {
    const { results } = await env.DB.prepare("SELECT clave, valor, updated_at FROM estado").all();
    return Object.fromEntries(results.map(r => [r.clave, { valor: JSON.parse(r.valor), updated_at: r.updated_at }]));
  } catch {
    return {};
  }
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
      const valido = validar(leerJson(answer.choices?.[0]?.message?.content ?? answer.response));
      if (valido) return valido;
    } catch {
      // un reintento si el modelo no devuelve JSON o se cae
    }
    nota = REINTENTO;
  }
  return null;
}

function interpretar(env, texto, anterior) {
  return llmJson(env, PROMPT_INTERPRETAR, pedidoInterpretar(texto, anterior), validarInterpretacion,
    PARAMS_INTERPRETAR.max_tokens, PARAMS_INTERPRETAR.temperature);
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
  return llmJson(env, PROMPT_ELEGIR, user, raw => validarDecision(raw, concepto.textos.map(t => t.id)),
    PARAMS_ELEGIR.max_tokens, PARAMS_ELEGIR.temperature);
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
  // Lo que hizo el ordenador con el formulario (v6): cuantos temas quedaron en cada paso, lo aplicado en palabras, lo
  // que no pudo aplicar, el tema de referencia y, si un tema nombrado tiene dudas, los candidatos.
  const cad = (x, max) => typeof x === "string" && x.length > 0 && x.length <= max;
  if (result.pasos !== undefined && result.pasos !== null) {
    if (!Array.isArray(result.pasos) || result.pasos.length > 12 ||
      !result.pasos.every(x => esObjeto(x) && cad(x.paso, 120) && Number.isSafeInteger(x.quedan) && x.quedan >= 0)) return null;
    optional.pasos = result.pasos.map(x => ({ paso: x.paso, quedan: x.quedan }));
  }
  if (result.aplicado !== undefined && result.aplicado !== null) {
    if (!Array.isArray(result.aplicado) || result.aplicado.length > 12 || !result.aplicado.every(x => cad(x, 120))) return null;
    optional.aplicado = result.aplicado;
  }
  if (result.ignorado !== undefined && result.ignorado !== null) {
    if (!Array.isArray(result.ignorado) || result.ignorado.length > 8 ||
      !result.ignorado.every(x => esObjeto(x) && cad(x.casilla, 40) && cad(x.motivo, 200))) return null;
    optional.ignorado = result.ignorado.map(x => ({ casilla: x.casilla, motivo: x.motivo }));
  }
  if (result.referencia !== undefined && result.referencia !== null) {
    const r = result.referencia;
    if (!esObjeto(r) || typeof r.titulo !== "string" || r.titulo.length > 160 || typeof r.artista !== "string" || r.artista.length > 120) return null;
    optional.referencia = { titulo: r.titulo, artista: r.artista };
  }
  if (result.candidatos !== undefined && result.candidatos !== null) {
    if (!Array.isArray(result.candidatos) || result.candidatos.length > 5 ||
      !result.candidatos.every(x => esObjeto(x) && typeof x.uid === "string" && UID_TEMA_RE.test(x.uid) && cad(x.titulo, 250) &&
        typeof x.artista === "string" && x.artista.length <= 200)) return null;
    optional.candidatos = result.candidatos.map(x => ({ uid: x.uid, titulo: x.titulo, artista: x.artista }));
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
    // De paso, el ordenador cuenta que suena (al cambiar y cada 5 min) y el catalogo (una vez por proceso). Lo que no
    // valida se ignora: el reclamo sigue igual.
    if (data.sonando !== undefined) {
      const sonando = validarSonando(data.sonando);
      if (sonando !== undefined) await guardarEstado(env, "sonando", sonando);
    }
    if (data.catalogo !== undefined) {
      const catalogo = validarCatalogo(data.catalogo);
      if (catalogo) await guardarEstado(env, "catalogo", catalogo);
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
      if (status === "done" && (concepto.version === 2 || concepto.version === 3) && concepto.fase === "sondeo") {
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

  // Lo de arriba de la pantalla: que suena (y hace cuanto hablo el ordenador), los generos para los presets y los presets.
  // El telefono lo pide cada 20 s mientras esta a la vista: una lectura de D1, sin LLM.
  if (path === "/api/estado" && request.method === "GET") {
    const estado = await leerEstado(env);
    const now = Date.now();
    const s = estado.sonando, cat = estado.catalogo ? estado.catalogo.valor : null;
    const ordenadorHace = s ? Math.max(0, Math.round((now - s.updated_at) / 1000)) : null;
    return reply({
      sonando: s && s.valor && now - s.updated_at < ORDENADOR_VIVO_MS ? s.valor : null,
      ordenador_hace_s: ordenadorHace,
      ordenador_conectado: ordenadorHace !== null && ordenadorHace * 1000 < ORDENADOR_VIVO_MS,
      catalogo: cat ? { temas: cat.temas, sin_genero: cat.sin_genero, generos: cat.generos } : null,
      generos_preset: cat ? cat.generos.slice(0, PRESETS.generos.cuantos).map(g => g.nombre) : [],
      presets: PRESETS.presets.map(x => ({ id: x.id, grupo: x.grupo, etiqueta: x.etiqueta, pendiente: !!x.pendiente }))
    });
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

  // El pedido: presets (sin LLM), texto libre (LLM, como en la v5) o los dos. Va a la cola como version 3, con el
  // formulario de los presets; el texto llena la casilla texto cuando el LLM termina de elegir.
  if (path === "/api/conceptos" && request.method === "POST") {
    const data = await readJson(request);
    if (!data) return fail("JSON inválido o demasiado grande");
    const texto = textoCorto(data.texto, 2, 400);
    const ids = data.presets === undefined ? [] : data.presets;
    const generos = data.generos === undefined ? [] : data.generos;
    const lista = (v, max, largo) => Array.isArray(v) && v.length <= max && v.every(x => typeof x === "string" && x.length >= 1 && x.length <= largo);
    if (!lista(ids, 12, 40) || !lista(generos, FORMULARIO.generos.max, 60)) return fail("Presets inválidos");
    if (data.referencia_uid !== undefined && (typeof data.referencia_uid !== "string" || !UID_TEMA_RE.test(data.referencia_uid))) {
      return fail("Solicitud inválida");
    }
    let { form: parcial, etiquetas, desconocidos } = componer(ids, generos);
    if (desconocidos.length) return fail("Preset desconocido");
    if (data.referencia_uid) {
      parcial = fundir(parcial, { referencia: { tipo: "tema", uid: data.referencia_uid } });
      etiquetas.push(textoCorto(data.referencia_titulo, 1, 80) ? `Como «${data.referencia_titulo.trim()}»` : "Como el tema elegido");
    }
    const estado = generos.length ? await leerEstado(env) : {};
    const validos = estado.catalogo ? estado.catalogo.valor.generos.map(g => g.nombre) : null;
    const { form, problemas } = validarFormulario(parcial, validos);
    if (problemas.length) return fail("Ese género no está en la biblioteca");
    const conPresets = ids.length > 0 || generos.length > 0 || !!data.referencia_uid;
    if (!texto && !conPresets) return fail("Elige un preset o escribe qué quieres escuchar");
    let concepto;
    if (!texto) {
      const resumen = etiquetas.join(" · ");
      concepto = { version: 3, fase: "final", ronda: 0, original: resumen, entendi: resumen, nombre: etiquetas.join(", ").slice(0, 60),
        no_se_oye: [], textos: [], formulario: form, presets: etiquetas };
    } else {
      let anterior = null;
      if (data.anterior !== undefined) {
        if (typeof data.anterior !== "string" || !UUID_RE.test(data.anterior)) return fail("Solicitud inválida");
        try {
          const row = await env.DB.prepare("SELECT concept_json FROM jobs WHERE id = ?").bind(data.anterior).first();
          if (row) anterior = JSON.parse(row.concept_json);
          if (anterior && (anterior.version !== 2 && anterior.version !== 3)) anterior = null;
          if (anterior && !Array.isArray(anterior.textos)) anterior = null;
        } catch {
          return fail("No se pudo leer el concepto anterior", 500);
        }
      }
      // Primero las reglas (reglas.js): lo que se lee con certeza no pasa por el LLM. Si no queda nada sin leer, el LLM
      // ni se llama (cero neuronas).
      const estadoT = generos.length ? estado : await leerEstado(env);
      const s = estadoT.sonando;
      const haySet = !!(s && s.valor && Date.now() - s.updated_at < ORDENADOR_VIVO_MS);
      const reglas = interpretarReglas(texto, { haySet, anio: new Date().getFullYear() }, REGLAMENTO, FORMULARIO);
      if (reglas.salida !== "lista") return reply(respuestaSinLista(reglas.salida, texto));
      const validosT = estadoT.catalogo ? estadoT.catalogo.valor.generos.map(g => g.nombre) : null;
      // Al ajustar sin tocar los presets, lo nuevo del texto se pone sobre el pedido anterior; si no, los presets mandan
      // sobre lo que leyeron las reglas. Un género que no está en el catálogo se descarta.
      const ajusta = !conPresets && anterior && anterior.formulario;
      const base = ajusta ? fundir(anterior.formulario, reglas.formulario) : fundir(reglas.formulario, parcial);
      const formulario = validarFormulario(base, validosT).form;
      const presetsUsados = !conPresets && anterior && anterior.presets ? anterior.presets : etiquetas;
      const original = anterior ? `${anterior.original} / ${texto}` : texto;
      if (!reglas.resto) {
        const previos = anterior && Array.isArray(anterior.textos) ? anterior.textos : [];
        concepto = { version: 3, fase: "final", ronda: 0, original, entendi: `«${texto}», leído sin el LLM`, nombre: texto.slice(0, 60),
          no_se_oye: reglas.no_se_oye, textos: previos, formulario, presets: presetsUsados, reglas: reglas.tipos };
      } else {
        const interpretado = await interpretar(env, texto, anterior && anterior.textos.length ? anterior : null);
        if (!interpretado) return fail("El modelo no pudo interpretar el pedido. Prueba otra vez.", 502);
        concepto = { version: 3, fase: "sondeo", ronda: 1, original, ...interpretado, formulario, presets: presetsUsados, reglas: reglas.tipos };
      }
    }
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
:root{color-scheme:dark}
body{font:16px system-ui;background:#111827;color:#f9fafb;max-width:650px;margin:auto;padding:0 16px 24px}
#now{position:sticky;top:0;z-index:1;background:#0b1220;margin:0 -16px 6px;padding:9px 16px;font-size:13px;color:#e5e7eb;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;border-bottom:1px solid #1f2937}
#now b{color:#5eead4;font-weight:600}#now.off{color:#9ca3af}
h2{margin:14px 0 4px}small,.nota{color:#9ca3af;font-size:13px}
input,textarea,button{box-sizing:border-box;width:100%;padding:13px;margin-top:12px;border-radius:9px;font:inherit}
input,textarea{background:#1f2937;color:white;border:1px solid #4b5563}
button{background:#0d9488;color:white;border:0;cursor:pointer}
button:disabled{opacity:.4;cursor:default}
.label{margin:14px 0 0;font-size:13px;color:#9ca3af}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
.chip{width:auto;margin:0;padding:8px 13px;border-radius:999px;background:#1f2937;border:1px solid #4b5563;color:#e5e7eb;font-size:14px}
.chip[aria-pressed=true]{background:#0d9488;border-color:#0d9488;color:white}
#login,#main{display:none}
#adjust{background:#334155}#again{background:transparent;border:1px solid #475569}
#card{display:none;margin-top:18px;border:1px solid #475569;border-radius:10px;padding:12px}
#status{margin:8px 0 0;color:#cbd5e1}#detail,#extra{margin:10px 0 0;font-size:14px;color:#cbd5e1}#detail p,#extra p{margin:6px 0}
#detail ul,#extra ul{margin:4px 0 0;padding-left:18px;color:#9ca3af}
#extra .pasos{list-style:none;padding-left:0;font-variant-numeric:tabular-nums}#extra .pasos span{display:inline-block;min-width:3.2em;text-align:right;margin-right:8px;color:#e5e7eb}
.cand{background:#1f2937;border:1px solid #4b5563;text-align:left;margin-top:6px;padding:10px}
#tracks{margin:12px 0 0;padding-left:30px}#tracks li{margin:0 0 11px;overflow-wrap:anywhere}
#tracks .meta{display:block;color:#9ca3af;font-size:13px;font-variant-numeric:tabular-nums}
#after{display:none}
#salir{display:block;width:auto;margin:26px auto 0;padding:6px 10px;background:none;color:#6b7280;font-size:13px}
</style>
</head>
<body>
<div id="now" class="off" aria-live="polite">Traktor Concept AI</div>
<section id="login">
<h2>Traktor Concept AI</h2>
<small>La contraseña se pide una sola vez: este teléfono la recuerda.</small>
<input id="password" type="password" placeholder="Contraseña de acceso" autocomplete="current-password">
<button id="entrar">Entrar</button>
<p id="loginmsg" class="nota"></p>
</section>
<section id="main">
<div class="label">A partir de lo que suena</div><div class="chips" id="c-ref"></div>
<div class="label">Géneros (solo esos)</div><div class="chips" id="c-gen"></div>
<div class="label">Momento de la noche</div><div class="chips" id="c-mom"></div>
<textarea id="message" rows="2" placeholder="Y si quieres, escribe: con trompetas, algo oscuro, como tal tema..."></textarea>
<button id="send">Buscar en mi música</button>
<div id="card" aria-live="polite"><strong id="name"></strong>
<p id="status"></p><div id="detail"></div><div id="extra"></div><ol id="tracks"></ol></div>
<div id="after"><textarea id="correction" rows="2" placeholder="¿Algo que corregir? Más oscuro, sin tanta voz..."></textarea>
<button id="adjust">Ajustar</button><button id="again">Nuevo pedido</button></div>
<button id="salir">Salir (olvidar la contraseña en este teléfono)</button>
</section>
<script>
const KEY='traktor_concept_pw';
const $=id=>document.getElementById(id);
const sendButton=$('send'),adjustButton=$('adjust'),againButton=$('again'),input=$('message'),correction=$('correction');
const card=$('card'),nameText=$('name'),statusText=$('status'),detail=$('detail'),extra=$('extra'),tracksList=$('tracks'),after=$('after');
let pw='',currentId=null,followId=0,siguiendo=false,estado=null,presets=[];
const sel=new Set(),gens=new Set();
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
function guardar(v){try{if(v)localStorage.setItem(KEY,v);else localStorage.removeItem(KEY)}catch(e){}}
function recordada(){try{return localStorage.getItem(KEY)||''}catch(e){return ''}}
function mostrarLogin(msg){$('main').style.display='none';$('login').style.display='block';$('loginmsg').textContent=msg||'';$('now').className='off';$('now').textContent='Traktor Concept AI';$('password').focus()}
async function call(path,options){
  const response=await fetch(path,options);
  let result={};try{result=await response.json()}catch(e){}
  if(response.status===401){pw='';guardar('');mostrarLogin('La contraseña no es correcta.');throw new Error('Contraseña incorrecta')}
  if(!response.ok)throw new Error(result.error||'Error HTTP '+response.status);
  return result;
}
const post=(path,data)=>call(path,{method:'POST',headers:{'Content-Type':'application/json','X-App-Password':pw},body:JSON.stringify(data)});
const get=path=>call(path,{headers:{'X-App-Password':pw}});
async function entrar(clave){
  pw=clave;$('loginmsg').textContent='Entrando...';
  try{const e=await get('/api/estado');guardar(clave);$('login').style.display='none';$('main').style.display='block';$('password').value='';pintarEstado(e);armarChips(e)}
  catch(err){if(pw)$('loginmsg').textContent='Sin conexión: '+err.message}
}
function pintarEstado(e){
  estado=e;const now=$('now');
  if(!e.ordenador_conectado){now.className='off';now.textContent='Ordenador sin conexión: ¿está abierto Traktor?';return}
  if(!e.sonando){now.className='off';now.textContent='Nada sonando todavía';return}
  const s=e.sonando,b=document.createElement('b');
  b.textContent='▶ '+(s.deck?s.deck+'  ':'');
  now.className='';now.replaceChildren(b,(s.artista?s.artista+' — ':'')+s.titulo+(s.bpm?' · '+Math.round(s.bpm):'')+(s.clave?' · '+s.clave:''));
}
async function refrescar(){
  if(document.hidden||siguiendo||!pw)return;
  try{const e=await get('/api/estado');const antes=JSON.stringify(estado?estado.generos_preset:[]);pintarEstado(e);if(JSON.stringify(e.generos_preset)!==antes)armarChips(e)}catch(err){}
}
setInterval(refrescar,20000);
document.addEventListener('visibilitychange',()=>{if(!document.hidden)refrescar()});
function chip(texto,activo,onclick,apagado){
  const b=document.createElement('button');b.type='button';b.className='chip';b.textContent=texto;
  b.setAttribute('aria-pressed',activo?'true':'false');b.disabled=!!apagado;b.onclick=onclick;return b;
}
function nota(t){const s=document.createElement('span');s.className='nota';s.textContent=t;return s}
function alternar(p){
  if(sel.has(p.id))sel.delete(p.id);
  else{if(p.grupo==='bpm'||p.grupo==='momento')presets.filter(x=>x.grupo===p.grupo).forEach(x=>sel.delete(x.id));sel.add(p.id)}
  pintarChips();
}
function pintarChips(){
  $('c-ref').replaceChildren(...presets.filter(p=>p.grupo==='sonando'||p.grupo==='bpm'||p.grupo==='voz').map(p=>chip(p.etiqueta,sel.has(p.id),()=>alternar(p))));
  const g=estado&&estado.generos_preset?estado.generos_preset:[];
  $('c-gen').replaceChildren(...(g.length?g.map(n=>chip(n,gens.has(n),()=>{if(gens.has(n))gens.delete(n);else gens.add(n);pintarChips()})):[nota('Aparecen cuando el ordenador se conecta.')]));
  $('c-mom').replaceChildren(...presets.filter(p=>p.grupo==='momento').map(p=>chip(p.etiqueta+(p.pendiente?' (pronto)':''),sel.has(p.id),()=>alternar(p),p.pendiente)));
}
function armarChips(e){presets=e.presets||[];pintarChips()}
function line(label,value){const p=document.createElement('p');const b=document.createElement('b');b.textContent=label+' ';p.append(b,value);return p}
function showConcept(c){
  nameText.textContent=c.nombre||c.name||'Concepto';
  const parts=[];
  if(c.entendi)parts.push(line('Entendí:',c.entendi));
  if(c.presets&&c.presets.length&&c.textos&&c.textos.length)parts.push(line('Con:',c.presets.join(' · ')));
  if(c.no_se_oye&&c.no_se_oye.length)parts.push(line('Eso no se oye, lo traduje a sonido:',c.no_se_oye.join(', ')));
  if(c.problema)parts.push(line('Primer intento:',c.problema));
  if(c.resumen)parts.push(line('Va a sonar a:',c.resumen));
  if(c.textos&&c.textos.length){const ul=document.createElement('ul');for(const t of c.textos){const li=document.createElement('li');li.textContent=t.texto;ul.append(li)}parts.push(line('Busqué:',''),ul)}
  detail.replaceChildren(...parts);
}
function showResult(r){
  const parts=[];
  if(r.referencia&&r.referencia.titulo)parts.push(line('Referencia:',(r.referencia.artista?r.referencia.artista+' — ':'')+r.referencia.titulo));
  if(r.pasos&&r.pasos.length>1){
    const ul=document.createElement('ul');ul.className='pasos';
    for(const x of r.pasos){const li=document.createElement('li');const n=document.createElement('span');n.textContent=x.quedan;li.append(n,x.paso);ul.append(li)}
    parts.push(line('Cuántos quedaron:',''),ul);
  }
  for(const i of r.ignorado||[])parts.push(line('No aplicado:',i.motivo));
  for(const n of r.notes||[])parts.push(line('Nota:',n));
  if(r.candidatos&&r.candidatos.length){
    parts.push(line('¿Cuál de estos?',''));
    for(const c of r.candidatos){const b=document.createElement('button');b.className='cand';b.textContent=(c.artista?c.artista+' — ':'')+c.titulo;b.onclick=()=>start('',{referencia_uid:c.uid,referencia_titulo:c.titulo.slice(0,80)});parts.push(b)}
  }
  extra.replaceChildren(...parts);
  tracksList.replaceChildren(...r.tracks.map(t=>{
    const li=document.createElement('li');li.append((t.artist?t.artist+' — ':'')+t.title);
    const meta=document.createElement('span');meta.className='meta';
    meta.textContent=[t.bpm?Math.round(t.bpm)+' BPM':null,t.key||null,t.vocal===true?'Vocal':null].filter(Boolean).join(' · ');
    li.append(meta);return li;
  }));
}
function mostrarRespuesta(r){
  nameText.textContent='';statusText.textContent=r.mensaje||'';detail.replaceChildren();tracksList.replaceChildren();
  const botones=(r.opciones||[]).map(o=>chip(o.etiqueta,false,()=>{sel.clear();sel.add(o.id);if(estado&&estado.sonando)sel.add('sonando');pintarChips();start('',null,null)}));
  extra.replaceChildren(...(botones.length?[Object.assign(document.createElement('div'),{className:'chips'})]:[]));
  if(botones.length)extra.firstChild.append(...botones);
}
function progress(job,waited){
  const c=job.concept;
  if(job.status==='queued'&&waited>30000)return 'Sigue esperando. ¿Está encendido el ordenador, con Traktor abierto?';
  if(c.fase==='final'&&!(c.textos&&c.textos.length))return 'Armando la lista...';
  if(c.fase==='final')return 'Creando la playlist...';
  if(c.ronda>1)return 'No convenció; lo reescribió y prueba otra vez...';
  return 'Probando '+c.textos.length+' maneras de decirlo en tu música... La primera vez de la noche puede tardar un minuto.';
}
async function follow(id){
  const mine=++followId,start0=Date.now();siguiendo=true;
  while(mine===followId&&Date.now()-start0<6*60*1000){
    await sleep(2000);
    if(mine!==followId)return;
    let job;
    try{job=await get('/api/jobs/'+id)}catch(e){statusText.textContent='Sin respuesta, reintentando... ('+e.message+')';continue}
    showConcept(job.concept);
    if(job.status==='done'){
      const r=job.result;
      statusText.textContent=r.tracks.length?(r.playlist?'Playlist nueva en Traktor: '+r.playlist:r.tracks.length+' temas'):(r.candidatos&&r.candidatos.length?'Hay varios temas con ese nombre.':'No quedó ningún tema con ese pedido.');
      showResult(r);finish();return;
    }
    if(job.status==='failed'){statusText.textContent='No se pudo completar: '+(job.error||'error desconocido');finish();return}
    statusText.textContent=progress(job,Date.now()-start0);
  }
  if(mine===followId){statusText.textContent='El ordenador no respondió a tiempo. El pedido queda guardado y se hará cuando vuelva.';finish()}
}
function finish(){siguiendo=false;sendButton.disabled=false;adjustButton.disabled=false;after.style.display='block';refrescar()}
async function start(texto,extraBody,anterior){
  sendButton.disabled=true;adjustButton.disabled=true;after.style.display='none';followId++;
  card.style.display='block';nameText.textContent=texto||'Presets';detail.replaceChildren();extra.replaceChildren();tracksList.replaceChildren();statusText.textContent=texto?'Pensando cómo buscarlo...':'Enviando...';
  card.scrollIntoView({behavior:'smooth',block:'start'});
  try{
    const body={presets:[...sel],generos:[...gens]};
    if(texto)body.texto=texto;
    if(anterior)body.anterior=anterior;
    Object.assign(body,extraBody||{});
    const job=await post('/api/conceptos',body);
    if(!job.id){mostrarRespuesta(job);finish();return}
    currentId=job.id;showConcept(job.concept);statusText.textContent='En cola...';follow(job.id);
  }catch(e){statusText.textContent='Error: '+e.message;finish()}
}
sendButton.onclick=()=>{const t=input.value.trim();if(t||sel.size||gens.size)start(t,null,null);else{card.style.display='block';nameText.textContent='';detail.replaceChildren();extra.replaceChildren();tracksList.replaceChildren();statusText.textContent='Elige un preset o escribe qué quieres escuchar.'}};
adjustButton.onclick=()=>{const t=correction.value.trim();if(t&&currentId){correction.value='';start(t,null,currentId)}};
againButton.onclick=()=>{followId++;siguiendo=false;currentId=null;card.style.display='none';after.style.display='none';tracksList.replaceChildren();input.value='';input.focus();sendButton.disabled=false};
$('entrar').onclick=()=>{const v=$('password').value;if(v)entrar(v)};
$('password').onkeydown=e=>{if(e.key==='Enter')$('entrar').click()};
$('salir').onclick=()=>{guardar('');pw='';sel.clear();gens.clear();mostrarLogin('')};
const r0=recordada();if(r0)entrar(r0);else mostrarLogin('');
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