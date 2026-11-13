// Pasa las respuestas crudas de «interpretar» (de Workers AI o de gpuq) por el mismo validador del Worker (llm.js).
//   node cloudflare/validar_respuestas.mjs <respuestas.jsonl>   -> una linea por respuesta: {id, valido}
import { readFileSync } from "node:fs";
import { leerJson, validarInterpretacion } from "./llm.js";

const filas = readFileSync(process.argv[2], "utf8").split("\n").filter(l => l.trim()).map(l => JSON.parse(l));
for (const r of filas) {
  let valido = null;
  try {
    valido = r.content ? validarInterpretacion(leerJson(r.content)) : null;
  } catch {
    valido = null;
  }
  process.stdout.write(JSON.stringify({ id: r.id, valido }) + "\n");
}
