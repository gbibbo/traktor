// Arma los pedidos al LLM que haria el Worker para los pedidos del banco: primero las reglas (reglas.js) y, para lo que
// queda sin leer, la llamada «interpretar» con las mismas instrucciones (llm.js). Una linea JSON por llamada, lista para
// cualquier servidor compatible con OpenAI (llama.cpp en gpuq: src/v4/evaluation/llm_gpu.py) o para Workers AI.
//   node cloudflare/banco_llm.mjs <banco.jsonl> [--todos] [--temperatura 0] > pedidos.jsonl
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { interpretarReglas } from "./reglas.js";
import { MODEL, PROMPT_INTERPRETAR, PARAMS_INTERPRETAR, pedidoInterpretar } from "./llm.js";

const aqui = dirname(fileURLToPath(import.meta.url));
const R = JSON.parse(readFileSync(join(aqui, "..", "config", "telefono", "reglamento.json"), "utf8"));
const C = JSON.parse(readFileSync(join(aqui, "..", "config", "telefono", "formulario.json"), "utf8"));
const args = process.argv.slice(2);
const todos = args.includes("--todos");
const i = args.indexOf("--temperatura");
const temperatura = i >= 0 ? Number(args[i + 1]) : PARAMS_INTERPRETAR.temperature;
const filas = readFileSync(args[0], "utf8").split("\n").filter(l => l.trim()).map(l => JSON.parse(l));
for (const f of filas) {
  const reglas = interpretarReglas(f.pedido, { haySet: !!f.set, anio: 2026 }, R, C);
  if (!todos && (reglas.salida !== "lista" || !reglas.resto)) continue;
  process.stdout.write(JSON.stringify({
    id: f.id, tipo: "interpretar", modelo_worker: MODEL, pedido: f.pedido, resto: reglas.resto,
    messages: [{ role: "system", content: PROMPT_INTERPRETAR }, { role: "user", content: pedidoInterpretar(f.pedido, null) }],
    max_tokens: PARAMS_INTERPRETAR.max_tokens, temperature: temperatura,
    response_format: { type: "json_object" }, chat_template_kwargs: { enable_thinking: false }
  }) + "\n");
}
