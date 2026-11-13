// Corre el interprete por reglas (reglas.js) sobre un archivo de pedidos y escribe lo que entendio, una linea JSON por
// pedido. Lo usa tools/traktor_live/banco_telefono.py (interprete «reglas»). Uso:
//   node cloudflare/banco_reglas.mjs <pedidos.jsonl> [anio]
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { interpretarReglas } from "./reglas.js";

const aqui = dirname(fileURLToPath(import.meta.url));
const R = JSON.parse(readFileSync(join(aqui, "..", "config", "telefono", "reglamento.json"), "utf8"));
const C = JSON.parse(readFileSync(join(aqui, "..", "config", "telefono", "formulario.json"), "utf8"));
const anio = Number(process.argv[3]) || new Date().getFullYear();
const filas = readFileSync(process.argv[2], "utf8").split("\n").filter(l => l.trim()).map(l => JSON.parse(l));
for (const f of filas) {
  const pred = interpretarReglas(f.pedido, { haySet: !!f.set, anio }, R, C);
  process.stdout.write(JSON.stringify({ id: f.id, pred }) + "\n");
}
