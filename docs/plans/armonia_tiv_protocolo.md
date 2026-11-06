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
| [REEMPLAZADO 2026-11-06 por aclaración de Gabriel: no comparar con la tonalidad fue una decisión deliberada de la tesis, que no quería emular ni repetir las sugerencias de la clave; sigue siendo cierto que no hay medición TIV contra Camelot, y para integrarlo al recomendador hace falta, porque Camelot es lo que usaría si no] La prueba comparó "candidato transpuesto contra original", nunca contra la regla Camelot: no hay evidencia de que el TIV mejore a la tonalidad | EVIDENCIA OBSERVADA | tesis 5.2 |
| [REEMPLAZADO 2026-11-06 por decisión de Gabriel: él transpone, y se sigue trabajando con las 12 transposiciones porque el TIV lo permite; en la práctica se sugiere hasta ±2] El máximo sobre 12 transposiciones modela algo que los DJ casi no hacen: 2,5 % de 24 202 temas tocados estaban transpuestos (Kim et al., ISMIR 2020) | EVIDENCIA OBSERVADA (fuente externa, leída por un subagente) | arXiv 2008.10267 |
| Trabajo futuro de la tesis: croma de 36 bins (desafinación), ventana local de los últimos compases, TIV por beat sobre todo el tema, base de preferencias de usuarios, transposición automática | EVIDENCIA OBSERVADA | tesis 6.3 |

## 2b. Decisiones de Gabriel (2026-11-06)

1. Usa transposición. Se calculan las 12; el recomendador y la escucha sugieren hasta ±2.
2. Mezcla intro con outro y a veces partes centrales, con solapamientos de hasta 2 minutos.
3. La tesis no comparó con la tonalidad a propósito: el TIV no busca emular la clave. Por eso la
   concordancia con la teoría de tonalidades (A1, A2) queda como diagnóstico, no como objetivo.
4. Propuesta suya, adoptada como **objetivo principal de ajuste**: autosupervisado, sin etiquetas de
   tonalidad. Un tema suena bien consigo mismo, así que un fragmento y el que le sigue en el mismo tema
   tienen que salir compatibles (y con transposición 0) frente a fragmentos de otros temas.
5. La escucha (etapa C) la hace más adelante; hasta entonces todo es offline y la decisión queda abierta.
6. La rama `harmonic-tiv` se sube a GitHub (sin tocar `main`).

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
| H8 Transposición | se calculan las 12; se sugiere hasta ±2 con penalización por semitono | A2, S2 (abajo) y escucha con el candidato transpuesto |
| H10 Secciones de la mezcla | TIV de los últimos 2 min del principal contra los primeros 2 min del candidato (asimétrico), contra el tema entero | A4 con el orden real de las transiciones (A sale, B entra) |
| H9 Afinación | corrección de afinación global (ya incluida) / sin corregir; medir cuántos temas están desafinados > 25 cents antes de pensar en 36 bins | conteo + A1 |

## 6. Métricas y criterios de éxito

**Etapa A — desarrollo (solo dev; ajusta y descarta).** Script: `tools/harmonic_tiv/eval_proxies.py`.
- **S1 autosupervisado (objetivo principal, decisión de Gabriel):** para cada tema dev, pares de
  fragmentos consecutivos de 30 s; el siguiente fragmento del mismo tema tiene que quedar arriba entre
  él y 200 fragmentos de otros temas dev. Métricas: MRR y AUC con el score sin transponer.
- **S2 transposición autosupervisada:** entre fragmentos consecutivos del mismo tema, la mejor
  transposición (de las 12) tiene que ser 0; % de aciertos en ±6 y en ±2.
- **S3 intro contra outro:** los primeros 2 min de un tema contra los últimos 2 min del mismo tema, entre
  los últimos 2 min de 200 temas más (lo más parecido a una mezcla real que no necesita etiquetas).
- A1 *teoría de tonalidades* sobre pares de consenso: Spearman del score contra la distancia en el
  círculo de quintas, AUC (distancia 0–1 contra 3–6), MRR del compañero de misma tonalidad.
- A2 *transposición*: % de pares donde la transposición sugerida lleva al candidato a la armadura del
  principal.
- A3 *hubness* (k = 10): asimetría de la k-ocurrencia, % de temas que nunca entran a un top-10, máximo de
  apariciones. Guardarraíl: no más concentrado que K0.
- A4 *transiciones reales* (sesiones dev): percentil medio del tema que Gabriel tocó entre los candidatos
  con BPM compatible, con IC bootstrap 95 %.
- A4b *transiciones reales dentro de la clase Camelot* (agregado el 2026-11-06, antes de ver resultados
  de `musica`): mismo percentil, pero solo contra candidatos con el mismo score Camelot (tonalidad MIK)
  que el tema que tocó. Ahí K0 empata por construcción, así que A4b > 0,5 dice que el TIV sabe algo de lo
  que Gabriel elige que la tonalidad que ve no dice. Se informa con IC; no cambia la regla de selección.
- A5 costo (s por tema en CPU) y cobertura (% de temas con score; K0 tiene ~84 %).
- [REEMPLAZADO 2026-11-06 por decisión de Gabriel (punto 2b.4), antes de ver resultados de `musica`:
  el objetivo principal es S1] ~~Regla de selección: entre las variantes que no empeoran A1 respecto de
  T0 más allá de su IC y cumplen A3, la de mayor A4.~~
- **Regla de selección del finalista:** la de mayor S1 (MRR) entre las que cumplen A3 (hubness no peor
  que K0) y no empeoran A4 respecto de T0 más allá de su IC; empate → la más simple. Combinar cambios solo
  si cada uno mejoró por separado, y volver a medir la combinación. A1, A2, A4b y S2–S3 se informan.

**Etapa B — prueba offline (test, una sola vez por finalista).** Las mismas métricas con IC.
- **Puerta B:** el finalista supera al azar en las transiciones reales de test (IC 95 % del percentil por
  encima de 0,5) y cumple A3. Si no supera al azar, no captura nada de lo que Gabriel elige: se detiene y
  se informa. No se le exige superar a K0 acá, porque K0 está favorecida (Gabriel ve esa tonalidad).

**Etapa C — escucha (decide la integración).**
- *Ítems:* temas principales del split test. Para cada uno, el mejor candidato según K0 y según el
  finalista, del mismo conjunto de candidatos; solo cuando difieren. Estratos (se informan, no deciden):
  S1 empate en K0 que el TIV desempata, S2 K0 compatible y TIV bajo, S3 TIV alto y K0 incompatible.
- *Audio:* solapamiento de 64 beats (~30 s; las mezclas de Gabriel llegan a 2 min, pero 60 ítems de 2 min
  serían 4 horas) con los tempos igualados (grilla de Traktor cuando exista), de la segunda mitad del
  principal a la primera mitad del candidato, con el cambio de graves por EQ como en la tesis. Cada sistema
  transpone su candidato con lo que sugiere, hasta ±2. A y B difieren solo en el candidato y su
  transposición.
- *Respuesta:* "A más armónico / B más armónico / igual", a ciegas y en orden al azar; 10 ítems repetidos
  para medir la consistencia de Gabriel consigo mismo.
- *Tamaño:* 60 ítems (tres sesiones de ~30 min) + 10 repetidos. Mirada intermedia a los 30: si el TIV gana
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
- [x] Caché de espectros de thesis48 (48/48, 0 errores, valores finitos).
- [x] Etapa A sobre thesis48: ver la sección 8.
- [ ] Caché de musica completa (`extract_semitones.py`, ~6 s por tema con 8 procesos).
- [ ] Etapa A sobre musica dev → finalista → Etapa B.
- [x] Herramienta de la etapa C (`listening_test.py`), probada de punta a punta con 2 ítems.
- [ ] Etapa C con Gabriel.

## 8. Resultados de la etapa A en thesis48 (exploratorio: 24 temas por género)

Fuente: `D:\TRAKTOR ML datos\harmonic_tiv\proxies_thesis48.json`, caché `cache_semitones_v2`, commit
fddb3b1. IC bootstrap pareados sobre los temas objetivo (aproximados: las columnas no se remuestrean).

| Afirmación | Etiqueta | Evidencia |
| :--- | :--- | :--- |
| La reimplementación reproduce el orden de la tesis por Perf: FEM#10 13,3 > 30 % central 12,1 > sin blanqueo 9,7 (tesis: 11,26 / 10,44) | EVIDENCIA OBSERVADA | `proxies_thesis48.json` |
| Perf no sirve para comparar fórmulas: la disonancia de la mezcla da Perf 44,6 con peor orden de pares (Spearman 0,87 contra 0,91) | EVIDENCIA OBSERVADA | ídem |
| En Progressive House el TIV recupera las relaciones de tonalidad casi perfecto (Spearman 0,91, AUC 0,99, encuentra primero al compañero de misma tonalidad en 24 de 24, transposición correcta 99 %); en Techno mucho menos (0,41; 0,78; MRR 0,53; 61 %) | EVIDENCIA OBSERVADA | ídem |
| Con la misma información de audio, comparar TIV directo ordena mejor que estimar la tonalidad y aplicar Camelot (PH 0,91 contra 0,68; Techno 0,41 contra 0,28) | EVIDENCIA OBSERVADA | ídem; tonalidad por perfiles de Temperley sobre el mismo croma (83 % y 25 % de acierto) |
| El blanqueo espectral es el paso que más pesa (sin él, Spearman −0,25 y −0,23, IC lejos de 0) | EVIDENCIA OBSERVADA | bootstrap pareado |
| Agregar el bajo al croma, el NNLS, ponderar bloques por igual o por claridad tonal, y la distancia sola empeoran o no cambian | EVIDENCIA OBSERVADA (n chico) | bootstrap pareado |
| En Techno, 30 % o 50 % central mejora a todo el tema (+0,05 y +0,04 de Spearman) y la pérdida de coherencia mejora a TIV.lib (+0,04); en PH no cambian | EXPLORATORIO | IC excluyen 0 pero n = 24 y los IC son optimistas |
