// Lo que el Worker le pide al LLM: el modelo, las instrucciones de «interpretar» y «elegir», cómo se arma el pedido
// y los validadores de sus respuestas. Separado de concept-ai-worker.js para que el banco del teléfono corra
// exactamente esto en otro servidor del mismo modelo (src/v4/evaluation/llm_gpu.py). Por qué las instrucciones dicen
// lo que dicen: docs/reports/clap_texto_2026-11-09.md.

export const MODEL = "@cf/google/gemma-4-26b-a4b-it";

export const IDS = ["A", "B", "C"];

export const PROMPT_INTERPRETAR = `You rewrite a DJ's rough request into search texts for CLAP, a model that matches text to the SOUND of music.
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

export const PROMPT_ELEGIR = `You check whether searches in a music collection found what a DJ asked for. You cannot hear the music. For each search you get what it found: genres, BPM, share of tracks with voice, "sounds like" words, and example tracks.

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

export const PROHIBIDO = /\b(this is|sound of|in the style of|tracks?|songs?|music by)\b/i;
// Lo que CLAP no oye: si el modelo lo escribe igual, ese texto se descarta.
export const NO_SE_OYE = /\b(birthday|wedding|beach|sunset|sunrise|gym|workout|christmas|funeral|dinner|summer|winter|night|morning|afternoon|vibes?|party)\b/i;

// «Pocas voces» escrito de cualquier forma nombra la voz y la suma: lo único medido que la resta es "without vocals".
export const MENOS_VOZ = /\b(?:with\s+)?(?:very\s+)?(?:few|minimal|sparse|little|less|subtle|light|occasional|barely any|no)\s+(?:vocal\s+(?:snippets|chops|samples)|vocals?|voices?|singing)\b/gi;

export function textoValido(texto) {
  if (typeof texto !== "string") return null;
  const t = texto.trim().replace(/^["']|["']$/g, "").replace(/\.$/, "").replace(MENOS_VOZ, "without vocals");
  const palabras = t.split(/\s+/).length;
  // CLAP no entiende la negación (salvo "without vocals", medido): un texto con otra negación busca lo contrario.
  const negacion = /\b(no|not|non|never|without)\b/i.test(t.replace(/\bwithout vocals\b/gi, ""));
  return palabras >= 3 && palabras <= 20 && t.length <= 180 && !/[^\x20-\x7e]/.test(t) && !PROHIBIDO.test(t) &&
    !NO_SE_OYE.test(t) && !negacion ? t : null;
}

export function validarTextos(lista) {
  if (!Array.isArray(lista)) return null;
  const textos = [...new Set(lista.map(textoValido).filter(Boolean))].slice(0, IDS.length);
  return textos.length >= 2 ? textos.map((texto, i) => ({ id: IDS[i], texto })) : null;
}

export function textoCorto(value, min, max) {
  if (typeof value !== "string") return null;
  const t = value.trim();
  return t.length >= min && t.length <= max ? t : null;
}

export function validarInterpretacion(raw) {
  if (!raw || typeof raw !== "object") return null;
  const entendi = textoCorto(raw.entendi, 3, 160);
  const nombre = textoCorto(raw.nombre, 2, 60);
  const textos = validarTextos(raw.textos);
  if (!entendi || !nombre || !textos) return null;
  const noSeOye = Array.isArray(raw.no_se_oye) ? raw.no_se_oye.map(x => textoCorto(x, 1, 40)).filter(Boolean).slice(0, 6) : [];
  return { entendi, nombre, no_se_oye: noSeOye, textos };
}

export function validarDecision(raw, ids) {
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

export const PARAMS_INTERPRETAR = { max_tokens: 420, temperature: 0.3 };
export const PARAMS_ELEGIR = { max_tokens: 360, temperature: 0.1 };
export const REINTENTO = "\n\nYour previous answer broke the rules. Follow every rule and return valid JSON only.";

// El pedido del usuario para «interpretar»: el texto, o la corrección con el intento anterior.
export function pedidoInterpretar(texto, anterior) {
  if (anterior) {
    return `The DJ first asked: ${anterior.original}\nYou searched: ${anterior.textos.map(t => t.texto).join(" | ")}\nNow the DJ says: ${texto}\nRewrite the 3 texts for the corrected request. The correction is about sound, so "no_se_oye" stays as before or empty.`;
  }
  return `Request: ${texto}`;
}

// Lo que devuelve el modelo, como objeto: quita las cercas de código si las pone.
export function leerJson(content) {
  return typeof content === "string"
    ? JSON.parse(content.trim().replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, ""))
    : content;
}
