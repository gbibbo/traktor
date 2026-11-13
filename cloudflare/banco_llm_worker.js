// Worker de prueba, solo para `wrangler dev` (nunca se publica): pasa un pedido compatible con OpenAI al mismo modelo de
// Workers AI que usa el Worker del telefono, con los mismos parametros, y devuelve la respuesta cruda. Lo usa
// src/v4/evaluation/llm_comparar.py para comparar Workers AI con el mismo modelo en gpuq. Gasta neuronas.
import { MODEL } from "./llm.js";

export default {
  async fetch(request, env) {
    if (request.method !== "POST" || new URL(request.url).pathname !== "/run") return new Response("No", { status: 404 });
    const p = await request.json();
    const t0 = Date.now();
    try {
      const answer = await env.AI.run(MODEL, {
        messages: p.messages, max_tokens: p.max_tokens, temperature: p.temperature,
        response_format: p.response_format, chat_template_kwargs: p.chat_template_kwargs
      });
      const content = answer.choices?.[0]?.message?.content ?? answer.response;
      return Response.json({ content: typeof content === "string" ? content : JSON.stringify(content),
        finish_reason: answer.choices?.[0]?.finish_reason ?? null, usage: answer.usage ?? null, ms: Date.now() - t0 });
    } catch (e) {
      return Response.json({ content: null, error: String(e).slice(0, 300), ms: Date.now() - t0 });
    }
  }
};
