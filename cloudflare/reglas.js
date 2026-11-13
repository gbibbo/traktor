// Interprete por reglas del telefono (plan v2, L6): lee con certeza lo que se puede leer sin LLM y devuelve un pedazo
// del formulario (config/telefono/formulario.json), la salida (lista, pregunta, no_hay, pide_detalle), lo que no se oye y
// el resto: las palabras que las reglas no consumieron, que van al LLM. Los datos viven en config/telefono/reglamento.json;
// este archivo solo los aplica, siempre en el mismo orden. Funcion pura: la usan el Worker y el banco
// (tools/traktor_live/banco_telefono.py, por cloudflare/banco_reglas.mjs).

const SEP = " | ";

export function normalizar(texto) {
  return String(texto || "").normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase()
    .replace(/[^a-z0-9]+/g, " ").replace(/\s+/g, " ").trim();
}

function escapar(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function frase(f) {
  return new RegExp("(?<![a-z0-9])" + escapar(f) + "(?![a-z0-9])");
}

// Busca la primera de las frases (de la mas larga a la mas corta) y la reemplaza por un separador.
function consumir(estado, frases) {
  const orden = [...frases].sort((a, b) => b.length - a.length);
  for (const f of orden) {
    const re = frase(f);
    if (re.test(estado.t)) {
      estado.t = estado.t.replace(re, SEP);
      return f;
    }
  }
  return null;
}

function contiene(t, frases) {
  return frases.some(f => frase(f).test(t));
}

function rango(lo, hi, C) {
  const [L, H] = C.bpm.limites;
  const ok = x => x === null || (x >= L && x <= H);
  if (!ok(lo) || !ok(hi) || (lo !== null && hi !== null && lo > hi)) return null;
  return { rango: [lo, hi] };
}

export function interpretarReglas(texto, contexto, R, C) {
  const haySet = !!(contexto && contexto.haySet);
  const anioActual = (contexto && contexto.anio) || new Date().getFullYear();
  const original = String(texto || "");
  const estado = { t: " " + normalizar(original) + " " };
  const form = {};
  const filtros = {};
  const otros = {};
  const tipos = new Set();
  const noSeOye = [];
  const salida = s => ({ formulario: {}, salida: s, no_se_oye: [], resto: "", tipos: [s] });
  const solo = estado.t.trim();

  // 0. saludos, pedidos vagos, ambiguos, lo que no hay: no se llena nada
  if (R.saludos.includes(solo)) return salida("pide_detalle");
  if (R.vagos.includes(solo)) return haySet ? { formulario: { referencia: { tipo: "sonando" } }, salida: "lista", no_se_oye: [], resto: "", tipos: ["sonando"] } : salida("pide_detalle");
  if (contiene(estado.t, R.ambiguos)) return salida("pregunta");
  if (contiene(estado.t, R.no_hay)) return salida("no_hay");

  // 1. un archivo nombrado
  const archivo = original.match(/([^\\/]+)\.(mp3|wav|flac|aiff?|m4a)\b/i);
  if (archivo) {
    return { formulario: { referencia: { tipo: "tema", busca: normalizar(archivo[1]).slice(0, C.otros.max_caracteres) } },
      salida: "lista", no_se_oye: [], resto: "", tipos: ["como_tema"] };
  }

  // 2. pesos, orden, largo, lo ya tocado
  for (const p of R.pesos) {
    if (consumir(estado, p.formas)) { form.pesos = { perfil: p.perfil, ajustes: p.ajustes }; tipos.add("pesos"); break; }
  }
  for (const [orden, frases] of Object.entries(R.orden)) {
    if (consumir(estado, frases)) { form.orden = orden; tipos.add("orden"); break; }
  }
  const largo = estado.t.match(/(?<![a-z0-9])solo (\d{1,2})(?: temas)?(?![a-z0-9])/);
  if (largo && +largo[1] >= 1 && +largo[1] <= C.largo.max) {
    form.largo = +largo[1]; estado.t = estado.t.replace(largo[0], SEP); tipos.add("orden");
  }
  if (consumir(estado, R.tocados)) { otros.excluir_tocados = true; tipos.add("datos"); }

  // 3. año (no si la fecha describe un sonido: «con reminiscencias a los 90»)
  const fechaEsSonido = contiene(estado.t, R.fecha_no_si);
  const decada = fechaEsSonido ? null : estado.t.match(/(?<![a-z0-9])(?:de )?los (\d{2}|\d{4})s?(?![a-z0-9])/);
  if (decada) {
    let d = +decada[1];
    if (d < 100) d += d >= 50 ? 1900 : 2000;
    if (d % 10 === 0 && d >= C.otros.anio[0] && d <= C.otros.anio[1]) {
      otros.anio = [d, d + 9]; estado.t = estado.t.replace(decada[0], SEP); tipos.add("datos");
    }
  }
  if (!otros.anio && !fechaEsSonido) {
    const anio = estado.t.match(/(?<![a-z0-9])(?:de |del )?((?:19|20)\d\d)(?![a-z0-9])/);
    if (anio) { otros.anio = [+anio[1], +anio[1]]; estado.t = estado.t.replace(anio[0], SEP); tipos.add("datos"); }
  }
  if (!otros.anio && consumir(estado, R.nuevos)) { otros.anio = [anioActual, null]; tipos.add("datos"); }

  // 4. tonalidad: frases, Camelot, nombres de nota
  for (const [modo, frases] of Object.entries(R.tonalidad)) {
    if (consumir(estado, frases)) { filtros.tonalidad = modo; tipos.add("ritmo_tono"); break; }
  }
  if (!filtros.tonalidad) {
    const cam = estado.t.match(/(?<![a-z0-9])(?:en )?(1[0-2]|[1-9]) ?([ab])(?![a-z0-9])/);
    if (cam) { filtros.tonalidad = { clave: cam[1] + cam[2].toUpperCase() }; estado.t = estado.t.replace(cam[0], SEP); tipos.add("ritmo_tono"); }
  }
  if (!filtros.tonalidad) {
    const nota = estado.t.match(/(?<![a-z0-9])(?:en )?(do|re|mi|fa|sol|la|si)( sostenido| bemol)? (menor|mayor)(?![a-z0-9])/);
    if (nota) {
      const semi = (R.notas[nota[1]] + (nota[2] === " sostenido" ? 1 : nota[2] === " bemol" ? 11 : 0)) % 12;
      const menor = nota[3] === "menor";
      filtros.tonalidad = { clave: (menor ? R.camelot_menor : R.camelot_mayor)[semi] + (menor ? "A" : "B") };
      estado.t = estado.t.replace(nota[0], SEP); tipos.add("ritmo_tono");
    }
  }

  // 5. voz (sin y con juntas, de la frase mas larga a la mas corta)
  const voces = [...R.voz.sin.map(f => [f, "sin"]), ...R.voz.con.map(f => [f, "con"])].sort((a, b) => b[0].length - a[0].length);
  for (const [f, v] of voces) {
    const re = frase(f);
    if (re.test(estado.t)) { filtros.voz = v; estado.t = estado.t.replace(re, SEP); tipos.add("voz"); break; }
  }

  // 6. BPM relativo y lo que suena
  for (const [r, frases] of Object.entries(R.relativo)) {
    if (consumir(estado, frases)) { filtros.bpm = { relativo: r }; tipos.add("ritmo_tono"); break; }
  }
  if (consumir(estado, R.sonando)) { form.referencia = { tipo: "sonando" }; tipos.add("sonando"); }

  // 7. momento
  for (const [mom, frases] of Object.entries(R.momentos)) {
    if (consumir(estado, frases)) { form.momento = mom; tipos.add("momento"); break; }
  }

  // 8. DJ y sello nombrados (el ordenador los busca)
  const dj = estado.t.match(/(?<![a-z0-9])(?:como un set de|como un dj set de|lo que pondria) ([a-z0-9 ]+?)(?= \||$| pero )/);
  if (dj && dj[1].trim()) { form.referencia = { tipo: "dj", busca: dj[1].trim().slice(0, C.otros.max_caracteres) }; estado.t = estado.t.replace(dj[0], SEP); tipos.add("como_dj"); }
  const sello = estado.t.match(/(?<![a-z0-9])(?:del )?sello ([a-z0-9]+(?: [a-z0-9]+)?)(?= \||\s*$| pero )/);
  if (sello) { otros.sellos = [sello[1].trim()]; estado.t = estado.t.replace(sello[0], SEP); tipos.add("datos"); }

  // 9. generos: lo que no es genero, los excluidos, los pedidos
  for (const f of R.no_es_genero) estado.t = estado.t.replace(frase(f), " _ ");
  const formas = R.generos.flatMap(g => g.formas.map(f => [f, g.generos])).sort((a, b) => b[0].length - a[0].length);
  const incluir = [], excluir = [];
  for (const [f, gs] of formas) {
    const antes = new RegExp("(?<![a-z0-9])(?:" + R.excluir_antes.map(escapar).join("|") + ") " + escapar(f) + "(?![a-z0-9])");
    while (antes.test(estado.t)) { estado.t = estado.t.replace(antes, SEP); excluir.push(...gs); }
    const re = frase(f);
    while (re.test(estado.t)) { estado.t = estado.t.replace(re, SEP); incluir.push(...gs); }
  }
  if (incluir.length || excluir.length) {
    filtros.generos = { incluir: [...new Set(incluir)].slice(0, C.generos.max), excluir: [...new Set(excluir)].slice(0, C.generos.max), modo: "blando" };
    tipos.add("genero");
  }

  // 10. BPM con numeros: rango, cotas, numero solo (±margen)
  const m = C.bpm.margen_numero;
  const num = "(\\d{2,3})";
  const pats = [
    [new RegExp("(?:entre|de) " + num + " (?:y|a) " + num + "(?![a-z0-9])"), x => rango(+x[1], +x[2], C)],
    [new RegExp("(?<![a-z0-9])" + num + " " + num + "(?: bpm)?(?![a-z0-9])"), x => rango(Math.min(+x[1], +x[2]), Math.max(+x[1], +x[2]), C)],
    [new RegExp("(?:arriba de|mas de|desde|que no bajen? de|no menos de|minimo) " + num + "(?: bpm)?(?![a-z0-9])"), x => rango(+x[1], null, C)],
    [new RegExp("(?<![a-z0-9])" + num + "(?: bpm)? (?:para arriba|o mas|en adelante)(?![a-z0-9])"), x => rango(+x[1], null, C)],
    [new RegExp("(?:menos de|hasta|debajo de|por debajo de|maximo) " + num + "(?: bpm)?(?![a-z0-9])"), x => rango(null, +x[1], C)],
    [new RegExp("(?<![a-z0-9])(?<!los )" + num + " ?(?:bpm)?(?![a-z0-9])"), x => rango(+x[1] - m, +x[1] + m, C)],   // «los 90» es una década
  ];
  if (!filtros.bpm) {
    for (const [re, hacer] of pats) {
      const x = estado.t.match(re);
      if (x) {
        const b = hacer(x);
        if (b) { filtros.bpm = b; estado.t = estado.t.replace(x[0], SEP); tipos.add("ritmo_tono"); }
        break;
      }
    }
  }

  // 11. lo que no se oye (queda en el resto: el LLM lo traduce a sonido)
  for (const o of R.ocasiones) if (frase(o).test(estado.t)) noSeOye.push(o);

  // 12. un tema nombrado, en lo que queda
  if (!form.referencia) {
    const tm = estado.t.match(/(?<![a-z0-9])(?:como|parecido a|tipo|estilo) ((?:(?! \| | pero ).)+)/);
    if (tm) {
      const busca = tm[1].replace(/^(?:ese|el) tema de /, "").replace(/ que tengo$/, "").trim();
      const palabras = busca.split(" ").filter(w => w && !R.relleno.includes(w));
      if (palabras.length) {
        form.referencia = { tipo: "tema", busca: busca.slice(0, C.otros.max_caracteres) };
        estado.t = estado.t.replace(tm[0], SEP); tipos.add("como_tema");
      }
    }
  }

  // 13. BPM relativo o tonalidad de lo que suena sin otra referencia: la referencia es lo que suena
  if (!form.referencia && ((filtros.bpm && filtros.bpm.relativo) || filtros.tonalidad === "compatible" || filtros.tonalidad === "igual")) {
    form.referencia = { tipo: "sonando" }; tipos.add("sonando");
  }

  if (Object.keys(otros).length) filtros.otros = otros;
  if (Object.keys(filtros).length) form.filtros = filtros;
  const resto = estado.t.split(" ").filter(w => w && w !== "|" && w !== "_" && !R.relleno.includes(w)).join(" ");
  if (resto) tipos.add("sonido");
  return { formulario: form, salida: "lista", no_se_oye: noSeOye, resto, tipos: [...tipos] };
}
