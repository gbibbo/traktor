# Compatibilidad armónica propia (TIV): protocolo de investigación

Rama `harmonic-tiv` (worktree aparte; nada se integra a `main` sin resultados positivos). Retoma la
tesis de máster de Gabriel: *Towards a new compatibility measure for harmonic EDM mixing* (UPF, 2021,
dir. Ángel Faraldo; PDF en Zenodo 5554688; código en github.com/gbibbo/harmonic_mix). Escrito el
2026-11-06 por el agente, **antes** de mirar resultados de evaluación, para que los criterios no se
acomoden a los números. Si un criterio cambia, se marca `[REEMPLAZADO fecha por motivo: ...]`, no se borra.

## 1. La decisión y la hipótesis

**Decisión:** si el recomendador en vivo usa un término armónico calculado del audio (TIV) en lugar de, o
además de, la regla Camelot de `src/v4/common/harmonic.py`.

**Hipótesis falsable:** con el mismo tema principal y el mismo conjunto de candidatos (BPM compatible y,
cuando exista, el mismo prefiltro de similitud), el sistema con TIV elige candidatos que Gabriel, **a
ciegas**, juzga armónicamente mejores que los que elige la regla Camelot con la mejor tonalidad disponible.

**No cuenta como éxito:** un Perf mayor (la métrica de la tesis), un AUC mayor contra la teoría de
tonalidades (ahí Camelot gana por construcción) ni un mejor percentil en los History por sí solo. Esos
números sirven para ajustar y para descartar lo claramente peor.

## 2. Qué dejó la tesis y qué se verificó (lectura completa del PDF y del código)

| Afirmación | Etiqueta | Evidencia |
| :--- | :--- | :--- |
| FEM#10 = HPSS de librosa (13, 31) → NNLSChroma de Essentia (frame 16384) → media de todos los frames → un TIV por tema → `small_scale_compatibility` de TIV.lib; T(st) = mejor de 12 transposiciones | EVIDENCIA OBSERVADA | tesis 4.2 y 4.3; `main.py` |
| El código publicado no es FEM#10: analiza el **30 % central** (`SONG_KEPT = 0.3`, el FEM#8) y llama a NNLSChroma con **`useNNLS=False`** (mapeo lineal, sin NNLS) | EVIDENCIA OBSERVADA | `main.py`, líneas 21 y 51 |
| [REFUTADO 2026-11-06 por la DFT de 12 valores reales: la energía (k = 0) y el coeficiente k = 6 son reales por definición; no hay bug] "un bug de TIV.lib deja en cero la parte imaginaria de dos de los siete números" (tesis 6.2) | EVIDENCIA OBSERVADA | `tests/v4/test_harmonic_tiv.py::test_k0_and_k6_are_real_not_a_bug`; 1000 cromas al azar dan imag = 0 exacto |
| `src/v4/harmonic_tiv/tiv.py` reproduce TIV.lib exacto | EVIDENCIA OBSERVADA | 2000 pares al azar: error máximo 3·10⁻¹⁶, 0 diferencias en la transposición sugerida |
| Perf = (100 − meanHC)/meanSTD premia la dispersión de una escala arbitraria (el reescalado 70–100 % de la ec. 4.2 la cambia) y no mide si el orden de los candidatos es correcto; se ajustó con 24 temas de un solo género | INFERENCIA | ecs. 3.3–3.10; tabla 5 |
| La prueba de escucha (15 personas × 10 ejemplos, TPRc 73,7 %, χ² = 36,7) trata 118 respuestas como independientes: la unidad que se replica son los 10 ejemplos, así que el χ² está inflado; descarta el 21 % de empates; los 5 FEM se compararon con las mismas respuestas | INFERENCIA | tesis 5.4–5.6, tabla 6 |
| La prueba comparó "candidato transpuesto contra original", nunca contra la regla Camelot: no hay evidencia de que el TIV mejore a la tonalidad | EVIDENCIA OBSERVADA | tesis 5.2 |
| El máximo sobre 12 transposiciones modela algo que los DJ casi no hacen: 2,5 % de 24 202 temas tocados estaban transpuestos (Kim et al., ISMIR 2020) | EVIDENCIA OBSERVADA (fuente externa, leída por un subagente) | arXiv 2008.10267 |
| Trabajo futuro de la tesis: croma de 36 bins (desafinación), ventana local de los últimos compases, TIV por beat sobre todo el tema, base de preferencias de usuarios, transposición automática | EVIDENCIA OBSERVADA | tesis 6.3 |

## 3. Datos (contados en esta sesión; `tools/harmonic_tiv/build_eval_sets.py`)

Todo fuera de Git, en `D:\TRAKTOR ML datos\harmonic_tiv\` (`eval_sets_report.json`).

- **thesis48**: los 48 temas de la tesis (24 Progressive House a 124 BPM, 24 Techno a 128; dos por cada
  tonalidad menor; tonalidad de Beatport verificada por Gabriel; switch points manuales). Tonalidad
  confiable pero solo menores y un solo BPM por género. Se usa para **reproducir la tesis** y como
  chequeo externo chico, no para elegir.
- **musica, tonalidades** (1760 temas): tag del archivo (Mixed In Key) 1472, Beatport confirmado 1321,
  Traktor 762. Acuerdo exacto MIK–Beatport **50 %** (n = 1187); Traktor–MIK 91 % (Traktor toma el tag:
  no es fuente independiente). El desacuerdo es casi todo de **modo**: 348 de los 593 desacuerdos son la
  paralela (misma tónica; MIK dice menor en el 90 % de los temas, Beatport en el 57 %); errores grandes,
  ~7 %. **Consenso** = MIK == Beatport: 594 temas.
- **musica, transiciones reales**: 36 sesiones de History, 665 temas tocados ≥ 60 s, 550 transiciones
  (B arranca mientras A sigue sonando en otro deck) con los dos temas en el catálogo, atados por ruta o
  por hash del audio. Señal armónica presente pero confundida con lo que Traktor muestra: con la tonalidad
  MIK (la que Gabriel ve) el tema que tocó cae en el percentil **0,603 [0,579; 0,627]** de los candidatos
  con BPM compatible; con la de Beatport (que no ve), **0,520 [0,495; 0,545]**.
- **Splits** (fijados antes de evaluar): temas dev/test por hash del artista principal (889/871; consenso
  355/318); sesiones dev/test por fecha (las 18 más viejas dev: 361 transiciones; test: 189).
- **Opcionales**: GiantSteps+ Key (600 extractos EDM con tonalidad anotada, CC BY-SA, Zenodo 1095691);
  las respuestas de la encuesta de la tesis, si Gabriel las conserva (10 ejemplos, P1/P2 en el repo).

## 4. Líneas base

- **K0 (la que hay que superar):** regla de `harmonic.py` con la tonalidad MIK (tag), sin transponer;
  desconocida = 0,5. Variante K0±2: con transposición hasta ±2 y −0,15 por semitono.
- **K1:** la misma regla con la tonalidad estimada del mismo croma que usa el TIV (aísla "TIV directo"
  contra "tonalidad estimada + Camelot" con la misma información de audio).
- **T0:** FEM#10 reimplementado en CPU sin Essentia (`src/v4/harmonic_tiv/`), y T0' = el código
  publicado (30 % central).

## 5. Variantes (una variable por vez respecto de T0)

| Hipótesis | Cambio | Prueba que la decide |
| :--- | :--- | :--- |
| H1 Sección | todo / 30 % / 50 % central / "core" por energía / intro y outro | proxies A1–A4; después, TIV del tramo que se superpone (outro del principal, intro del candidato) |
| H2 Preproceso | sin HPSS / HPSS / (Demucs sin batería, en un subconjunto) | A1–A4 |
| H3 Espectro | sin blanqueo / blanqueo / NNLS real | A1–A4 |
| H4 Registro | sin bajo (agudos) / completo / medio / solo bajo: en la mezcla el DJ cambia los bajos con el EQ, el choque está en medios y agudos | A1–A4 y escucha por estrato |
| H5 Ponderación | media cruda / cada bloque igual / bloques pesados por claridad tonal (los tramos de solo batería no votan) | A1–A4 y hubness |
| H6 Temas atonales | un tema sin centro tonal (‖T‖ chico) es neutro, no "compatible con todo" ni "incompatible" | hubness A3 y escucha del estrato S3 |
| H7 Fórmula | TIV.lib (distancia × disonancia) / solo distancia / disonancia de la mezcla con energías reales / choque ponderado por la claridad tonal de ambos | A1–A4 |
| H8 Transposición | sin transponer (default) / ±1 / ±2 con penalización | solo si Gabriel usa key shift (pregunta abierta) |
| H9 Afinación | corrección de afinación global (ya incluida) / sin corregir; medir cuántos temas están desafinados > 25 cents antes de pensar en 36 bins | conteo + A1 |

## 6. Métricas y criterios de éxito

**Etapa A — desarrollo (solo dev; ajusta y descarta).** Script: `tools/harmonic_tiv/eval_proxies.py`.
- A1 *teoría de tonalidades* sobre pares de consenso: Spearman del score contra la distancia en el
  círculo de quintas, AUC (distancia 0–1 contra 3–6), MRR del compañero de misma tonalidad.
- A2 *transposición*: % de pares donde la transposición sugerida lleva al candidato a la armadura del
  principal.
- A3 *hubness* (k = 10): asimetría de la k-ocurrencia, % de temas que nunca entran a un top-10, máximo de
  apariciones. Guardarraíl: no más concentrado que K0.
- A4 *transiciones reales* (sesiones dev): percentil medio del tema que Gabriel tocó entre los candidatos
  con BPM compatible, con IC bootstrap 95 %.
- A5 costo (s por tema en CPU) y cobertura (% de temas con score; K0 tiene ~84 %).
- **Regla de selección del finalista** (fijada ahora): entre las variantes que no empeoran A1 respecto de
  T0 más allá de su IC y cumplen A3, la de mayor A4; empate → la más simple. Combinar varios cambios solo
  si cada uno mejoró por separado, y volver a medir la combinación.

**Etapa B — prueba offline (test, una sola vez por finalista).** Las mismas métricas con IC.
- **Puerta B:** el finalista supera al azar en las transiciones reales de test (IC 95 % del percentil por
  encima de 0,5) y cumple A3. Si no supera al azar, no captura nada de lo que Gabriel elige: se detiene y
  se informa. No se le exige superar a K0 acá, porque K0 está favorecida (Gabriel ve esa tonalidad).

**Etapa C — escucha (decide la integración).**
- *Ítems:* temas principales del split test. Para cada uno, el mejor candidato según K0 y según el
  finalista, del mismo conjunto de candidatos; solo cuando difieren. Estratos (se informan, no deciden):
  S1 empate en K0 que el TIV desempata, S2 K0 compatible y TIV bajo, S3 TIV alto y K0 incompatible.
- *Audio:* transición de 32 beats con los tempos igualados (grilla de Traktor cuando exista), del outro del
  principal al intro del candidato, con el cambio de bajos por EQ como en la tesis, sin transponer. A y B
  difieren solo en el candidato.
- *Respuesta:* "A más armónico / B más armónico / igual", a ciegas y en orden al azar; 10 ítems repetidos
  para medir la consistencia de Gabriel consigo mismo.
- *Tamaño:* 60 ítems (dos sesiones de ~30 min) + 10 repetidos. Mirada intermedia a los 30: si el TIV gana
  ≤ 50 % de los no empates, se para por futilidad.
- **Éxito:** sin contar empates, el TIV gana ≥ 60 % y la prueba binomial exacta unilateral da p < 0,05;
  y la consistencia de Gabriel en los repetidos es ≥ 70 %. Si la consistencia es menor, el resultado es
  **NO CONCLUYENTE** (no se puede distinguir el método del ruido del oído). Los empates se informan.
- Si se suman otros DJ: modelo logístico mixto con efectos aleatorios de oyente y de ítem.

**Si sale positivo:** se integra al recomendador en vivo (TIV precalculado por tema; el score contra 1800
candidatos cuesta milisegundos), con tests, y se actualiza `docs/HARMONIC_COMPATIBILITY.md`. **Si sale
negativo o no concluyente:** se documenta en `docs/reports/` y la regla Camelot queda como está.

## 7. Estado

- [x] Tesis leída completa; código original revisado; TIV reimplementado y verificado contra TIV.lib.
- [x] Front-end CPU sin Essentia (`frontend.py`), validado con señales sintéticas (afinación ±0,02 st).
- [x] Conjuntos de evaluación y splits (`build_eval_sets.py`).
- [ ] Caché de espectros: thesis48 → musica completa (`extract_semitones.py`, ~25 s por tema).
- [ ] Etapa A sobre thesis48 (reproducir la tesis) y sobre musica dev.
- [ ] Etapa B, generador de transiciones para la escucha, Etapa C.
