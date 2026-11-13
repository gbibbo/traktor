# El LLM del teléfono en gpuq: ¿vale lo que se mida ahí para Workers AI?

**Fecha:** 2026-11-13. **Para:** quien mejore las instrucciones del LLM del teléfono (línea B, L7). **Pedido de Gabriel:**
correr el mismo modelo abierto en el cluster, para experimentar sin gastar neuronas.

## Veredicto

Sí. Gemma 4 26B A4B corrido con llama.cpp en dos T4 de Kaggle (GGUF Q8_0) responde casi igual que el mismo modelo en
Workers AI. Pasa el umbral fijado antes de medir en las tres medidas:

| Medida | Umbral | Resultado |
| :--- | ---: | ---: |
| Respuestas válidas para el validador del Worker | diferencia ≤ 0,05 | 1,00 y 1,00 |
| Mismo «no se oye» | ≥ 0,90 | 0,91 |
| Solapamiento de los 30 primeros temas | mediana ≥ 0,50 | mediana 0,70; mínimo 0,17 |

Desde ahora, las instrucciones se prueban en gpuq con el banco del teléfono. Antes de publicar un cambio que gane ahí, se
confirma una vez en Workers AI con los mismos pedidos, lo que cuesta unas 360 neuronas.

## Lo medido

- **Pedidos:** los 43 del banco (`config/telefono/banco_pedidos.jsonl`) que las reglas no leen enteros, con las
  instrucciones de producción (`cloudflare/llm.js`, «interpretar») y temperatura 0 en los dos servidores
  (`erclm:traktor_ml/llm_banco/pedidos/interpretar_t0_v1.jsonl`). Los arma `cloudflare/banco_llm.mjs`.
- **Workers AI:** `@cf/google/gemma-4-26b-a4b-it` con `wrangler dev`, que no publica nada. Fueron 28 893 tokens de
  entrada y 3409 de salida, unas 356 neuronas. Respuestas: `llm_banco/cloudflare_t0_v1/respuestas.jsonl`.
- **gpuq:**
  - Trabajo `traktor-llm-2`, en Kaggle, con dos Tesla T4. Código en `61a74ed`, el mismo de `llm_gpu.py` que en
    `f2e40d7`, el commit que fijó la cola.
  - llama.cpp `b12363`, compilado para CUDA 12.8, y `ggml-org/gemma-4-26B-A4B-it-GGUF`, Q8_0, de 26,9 GB.
  - Tiempos: bajar llama.cpp, 26 s; el modelo, 184 s; cargarlo, 243 s; 43 pedidos con una mediana de 2,7 s cada uno.
    En total, 11 minutos de GPU.
  - Respuestas: `llm_banco/gemma_q8_t0_v1/`. Antes, el trabajo corto `traktor-llm-1` contestó 5 pedidos en 9 minutos.
- **Comparación:** `src/v4/evaluation/llm_comparar.py comparar`. El umbral está en
  `config/telefono/llm_transferencia_umbral.json`, commiteado en `8d2077d` antes de las dos corridas. Salida:
  `llm_banco/transferencia_v1/comparacion.txt`. La lista de cada respuesta sale del coseno CLAP entre el promedio de
  sus textos y el índice de la biblioteca (1717 temas), igual que la lista de hoy.

| Afirmación | Etiqueta | Evidencia |
| :--- | :--- | :--- |
| Los dos servidores dan respuestas válidas en los 43 pedidos | EVIDENCIA OBSERVADA | Validador del Worker sobre cada respuesta (`validar_respuestas.mjs`) |
| El primer texto sale palabra por palabra igual en la mayoría de los pedidos | EVIDENCIA OBSERVADA | Comparación, columna de textos. Difieren, por ejemplo, s17, s34, s41, s44, s52, s57 y v75 |
| Las listas de 30 coinciden en un 0,70 de mediana, más que el 0,51 que coinciden dos promedios de tres redacciones del mismo pedido | EVIDENCIA OBSERVADA | Comparación; `docs/reports/clap_texto_2026-11-09.md` (ensamble) |
| Lo que se mida en gpuq vale para Workers AI | INFERENCIA | El modelo es el mismo y el acuerdo está por encima del umbral. La diferencia viene de la cuantización y del servidor, y no se sabe con qué precisión corre Workers AI |

## Revisión hostil

- **43 pedidos con temperatura 0.** En producción la temperatura es 0,3. Lo que se compara es el modelo, no el azar del
  muestreo.
- **«No se oye» quedó justo en el umbral** (0,91 contra 0,90). Un pedido más que no coincidiera lo dejaba debajo. Por
  eso, todo cambio que gane en gpuq se confirma en Workers AI antes de publicarse.
- **El peor caso** es «nada de boris brejcha»: 0,17. Ahí cada servidor escribió otro techno. Es un pedido que el LLM no
  debería convertir en texto: es un filtro de artista, y lo va a llenar el paso de casillas de L7.
- **Las instrucciones de L7 serán de listas cerradas.** El acuerdo entre servidores se va a poder medir como
  coincidencia exacta casilla por casilla, que es más estricto. Se vuelve a medir con la primera versión.

## Cómo reproducir

```
node cloudflare/banco_llm.mjs config/telefono/banco_pedidos.jsonl --temperatura 0 > pedidos.jsonl
python src/v4/evaluation/llm_comparar.py cloudflare --pedidos pedidos.jsonl --salida respuestas_cf.jsonl   (gasta neuronas)
python tools/gpuq/cola.py add <id> --cmd "python src/v4/evaluation/llm_gpu.py correr --pedidos llm_banco/pedidos/<n>.jsonl --salida llm_banco/<corrida>" --gpu T4 --platform kaggle --no-setup --timeout-h 2
python src/v4/evaluation/llm_comparar.py comparar --pedidos pedidos.jsonl --a respuestas_cf.jsonl --b respuestas_gpu.jsonl
```

Kaggle hace falta porque Q8_0 necesita las dos T4; Colab da una sola. Antes de encolar, se cuentan las horas de Kaggle de
la semana (`docs/GPUQ.md`, «Kaggle»).
