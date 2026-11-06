# Auditoría: cómo elegir y validar la compatibilidad armónica por TIV antes de integrarla

## Consigna para el auditor (leer primero)

> Eres el auditor metodológico de este proyecto. Tienes que tomar una decisión, no dar una opinión
> general.
>
> 1. **Lee el informe completo antes de escribir nada:** las secciones 1 a 7, todas las tablas y los dos
>    anexos. El anexo A es la posición del agente que hizo el trabajo: léelo al final y atácalo; no lo
>    tomes como evidencia.
> 2. **Responde la pregunta de la sección 1** con este formato, y nada más:
>    - **Decisión (a):** una de las opciones P1 a P5 de la sección 5, o una combinación. Escribe en orden
>      qué se hace ahora y qué después.
>    - **Decisión (b):** contra qué se compara en la escucha a ciegas (Camelot, sin criterio armónico, o
>      los dos) y qué resultado aprueba la integración. Si cambias el criterio de aprobación vigente,
>      escribe el nuevo con números.
>    - **Justificación:** cada argumento apoyado en una fila concreta del informe (por ejemplo, "3.3,
>      tabla pareada, solo bajo"). Si un argumento no se apoya en el informe, márcalo como tuyo.
>    - **Qué te haría cambiar de decisión:** el resultado concreto y su umbral.
>    - **Fallas del informe:** errores, huecos o conclusiones más anchas que la evidencia. Si no
>      encuentras ninguna, dilo.
> 3. **Si falta información, decide igual** con lo que hay y escribe qué falta y cómo cambiaría la
>    decisión. No respondas solo con preguntas.
> 4. Responde en español, en menos de 600 palabras.

---

Pedido de Gabriel Bibbó (dueño del proyecto y autor de la tesis), 2026-11-06. Preparado por el agente que
lleva la investigación en la rama `harmonic-tiv` de github.com/gbibbo/traktor (los números salen del código
del commit 681c3f4).
Gabriel no opina sobre la pregunta de abajo: la decide el auditor.

## 1. La pregunta

> Los criterios offline disponibles se contradicen, y la escucha a ciegas de Gabriel (el único juez
> previsto) no estará disponible por un tiempo. **¿Qué procedimiento debe usarse ahora para (a) elegir qué
> configuración o configuraciones del TIV llegan a la prueba de escucha a ciegas y (b) definir contra qué
> se las compara en esa prueba y qué resultado aprueba la integración?**
>
> Respuesta pedida: una de las opciones P1 a P5 de la sección 5, o una combinación justificada, y qué
> evidencia haría cambiar esa decisión.

## 2. Contexto mínimo

- **Proyecto.** TRAKTOR ML organiza una biblioteca privada de techno y tech house (1760 temas, CPU local)
  y va a sugerir en vivo, dentro de Traktor, temas que "mezclen bien" con el que suena. El recomendador
  combinará similitud (embeddings CLAP), BPM y un término armónico.
- **Término armónico actual.** Regla Camelot de `src/v4/common/harmonic.py`, aprobada por Gabriel el
  2026-09-11: relaciones de tonalidad con fuerza de 0,6 a 1,0, transposición de hasta ±2 semitonos y
  −0,15 por semitono. Usa la tonalidad de los tags de los archivos (Mixed In Key).
- **Propuesta en estudio.** Retomar la tesis de máster de Gabriel (UPF 2021, *Towards a new compatibility
  measure for harmonic EDM mixing*, Zenodo 5554688). Tonal Interval Vector (TIV) por tema, con HPSS, croma
  y la media de todo el tema; compatibilidad *small-scale* de TIV.lib; la mejor de 12 transposiciones.
- **Regla del proyecto.** Se integra solo si los números muestran que mejora las sugerencias armónicas.
  Entre representaciones, decide Gabriel escuchando, no un puntaje.
- **Decisiones de Gabriel ya tomadas** (no están en discusión):
  1. Usa transposición. Se calculan las 12; en la práctica se sugiere hasta ±2.
  2. Mezcla intro con outro, a veces partes centrales, con solapamientos de hasta 2 minutos.
  3. La tesis no comparó con la tonalidad a propósito: el TIV no busca emular la clave.
  4. Propuso un objetivo autosupervisado: un tema suena bien consigo mismo, así que un fragmento y el
     siguiente del mismo tema tienen que salir compatibles. Hoy figura como criterio principal de ajuste
     en el protocolo; la evidencia de la sección 4 lo pone en duda.
  5. La escucha a ciegas la hace más adelante, no ahora.
- **Protocolo con criterios fijados antes de evaluar:** `docs/plans/armonia_tiv_protocolo.md`. Los cambios
  se marcan en el lugar con [REEMPLAZADO] y fecha.

## 3. Evidencia

Etiquetas del protocolo de auditoría del proyecto (`.claude/skills/auditar/SKILL.md`):
- **EVIDENCIA OBSERVADA:** medido o leído en esta sesión.
- **INFERENCIA:** deducido, con el paso explícito.
- **NO CONCLUYENTE:** falta algo para decidirlo.

### 3.1 La tesis

| Afirmación | Etiqueta | Evidencia |
| :--- | :--- | :--- |
| La reimplementación del TIV (`src/v4/harmonic_tiv/tiv.py`) da lo mismo que TIV.lib | EVIDENCIA OBSERVADA | 2000 pares al azar: error ≤ 3·10⁻¹⁶, 0 diferencias en la transposición sugerida |
| El "bug" de TIV.lib reportado en la tesis (6.2) no existe: la energía (k = 0) y el coeficiente k = 6 de la DFT de 12 valores reales son reales por definición | EVIDENCIA OBSERVADA | test `test_k0_and_k6_are_real_not_a_bug` |
| El código publicado no es el FEM#10 de la tesis: usa el 30 % central y NNLSChroma con `useNNLS=False` | EVIDENCIA OBSERVADA | `main.py` de github.com/gbibbo/harmonic_mix, líneas 21 y 51 |
| Las respuestas originales de la encuesta (15 personas × 10 ejemplos × 2 clips, puntajes de 1 a 5) reproducen el TPRc de 71,2 % que la tesis da para FEM#9 (84 de 118 respuestas sin empate; 32 empates) | EVIDENCIA OBSERVADA | Excel de respuestas; la fila "Respuesta esperada" es la de FEM#9 |
| Tomando el ejemplo como unidad (las 15 respuestas de un ejemplo no son independientes), el sistema acierta la mayoría en 8 de 10 ejemplos: binomial unilateral p = 0,055. Spearman entre la diferencia de HC y la diferencia media de los oyentes, sobre los 10 ejemplos: 0,65 (p = 0,04). El χ² de la tesis, que trata las 118 respuestas como independientes, da p ≈ 3·10⁻⁶ | EVIDENCIA OBSERVADA | reanálisis de esta sesión |
| La encuesta comparó "candidato transpuesto contra original"; nunca hubo una medición del TIV contra la regla Camelot (decisión deliberada de la tesis) | EVIDENCIA OBSERVADA | tesis 5.2 y aclaración de Gabriel |
| La encuesta podría servir para validar variantes nuevas, pero falta saber qué par de temas y qué transposición tiene cada ejemplo: el audio fuente está (carpeta `experiment`, P1 y P2), los clips mezclados no | NO CONCLUYENTE | se resuelve si Gabriel conserva los clips o la tabla de los ejemplos |

### 3.2 Datos de tonalidad y de comportamiento

| Afirmación | Etiqueta | Evidencia |
| :--- | :--- | :--- |
| Las tonalidades son ruidosas. MIK y Beatport coinciden exactamente en el 50 % de 1187 temas. El desacuerdo es casi todo de modo: 348 de 593 son la paralela (MIK dice menor en el 90 % de los temas, Beatport en el 57 %). Los errores grandes son ~7 % | EVIDENCIA OBSERVADA | `keys_musica.csv` |
| La tonalidad de Traktor coincide con MIK en el 91 %: no es una fuente independiente | EVIDENCIA OBSERVADA | ídem |
| De los History de Traktor salen 550 transiciones reales (B arranca mientras A suena en otro deck; los dos ≥ 60 s), atadas al catálogo por ruta o por hash del audio | EVIDENCIA OBSERVADA | `transitions_history.csv` |
| Con la tonalidad que Gabriel ve en Traktor (MIK), el tema que tocó cae en el percentil 0,603 [0,579; 0,627] de los candidatos con BPM compatible. Con la de Beatport, que no ve: 0,520 [0,495; 0,545]. [REEMPLAZADO 2026-11-06 por el bootstrap por sesión que pidió el auditor (27 y 26 sesiones): MIK 0,603 [0,576; 0,635]; Beatport 0,520 [0,495; 0,550]] | EVIDENCIA OBSERVADA | bootstrap 95 % |
| Las transiciones reales tienen señal armónica, pero confundida con la tonalidad que muestra Traktor | INFERENCIA | de las dos filas anteriores |

### 3.3 Criterios offline sobre variantes del TIV (resultado central)

Cada variante cambia una sola cosa respecto de FEM#10. Muestra preliminar: 243 temas del split de
desarrollo (split por artista), congelada, sobre 411 temas analizados de 1760 (la extracción sigue).
Definiciones:
- **S1 (autosupervisado):** MRR al buscar el fragmento siguiente de 30 s del mismo tema entre fragmentos
  de 200 temas.
- **S1-hard:** igual, pero con 50 negativos de temas con la misma tonalidad MIK.
- [REFUTADO 2026-11-06 por el auditor: S1 usa 200 fragmentos tomados del conjunto de fragmentos de otros
  temas, no necesariamente de 200 temas distintos (los temas largos aportan más); S1-hard, 50 fragmentos.]
- **S3:** los primeros 2 min de un tema contra sus últimos 2 min, entre los de 200 temas.
- **Teoría:** Spearman del score contra la distancia en el círculo de quintas, en los 66 temas de consenso
  (MIK = Beatport).
- **Hubness:** asimetría de cuántas veces aparece cada tema en el top-10 de los demás.

| Variante | S1 | S1-hard | S3 | Teoría | Hubness |
| :--- | ---: | ---: | ---: | ---: | ---: |
| FEM#10 (tesis) | 0,775 | 0,753 | 0,708 | 0,51 | 0,82 |
| 30 % central | 0,775 | 0,753 | 0,708 | 0,52 | 0,58 |
| sin HPSS | 0,771 | 0,758 | 0,685 | 0,51 | 0,91 |
| **sin blanqueo** | 0,780 | **0,787** | **0,787** | **0,30** | 0,90 |
| NNLS | 0,738 | 0,720 | 0,639 | 0,49 | 0,78 |
| con bajo | 0,785 | 0,773 | 0,717 | 0,46 | 0,95 |
| sin bajo ni agudos | 0,761 | 0,756 | 0,686 | 0,52 | 0,45 |
| **solo bajo** | 0,769 | **0,798** | 0,766 | **0,18** | 0,52 |
| pérdida de coherencia (fórmula nueva) | 0,743 | 0,731 | 0,424 | **0,55** | **7,86** |
| disonancia de la mezcla | 0,155 | 0,152 | 0,138 | 0,37 | 5,40 |
| Regla Camelot (referencia) | 0,188 | — | 0,179 | — | −0,37 |

Azar en S1: MRR 0,029. Comparación pareada con IC 95 % (mismas 659 consultas y mismos negativos;
bootstrap por tema):

| Contra FEM#10 | Δ S1-hard | Δ Teoría |
| :--- | :--- | :--- |
| sin blanqueo | +0,035 [+0,004; +0,064] | −0,21 [−0,27; −0,16] |
| solo bajo | +0,041 [+0,005; +0,074] | −0,33 [−0,41; −0,24] |
| con bajo | +0,015 [−0,003; +0,034] | −0,05 [−0,07; −0,04] |

[REEMPLAZADO 2026-11-06 por el bootstrap corregido que pidió el auditor: la versión anterior descartaba las
repeticiones del remuestreo (`dict.fromkeys`), así que era una submuestra, no un bootstrap. Los IC de teoría
de arriba ya están recalculados con multiplicidades, excluyendo solo los pares de un tema consigo mismo
(B = 1000). Antes decían: sin blanqueo [−0,26; −0,17], solo bajo [−0,39; −0,26]. Ningún signo cambió.]

| Afirmación | Etiqueta | Evidencia |
| :--- | :--- | :--- |
| [REFUTADO 2026-11-06 por el AUC del mismo JSON, señalado por el auditor: vale para el MRR de S1-hard y para S3; con AUC, FEM#10 queda por encima de las dos en S1 (0,980 > 0,968 > 0,958) y en S1-hard (0,946 > 0,941 > 0,934). Los criterios autosupervisados no ordenan igual según la métrica] Los objetivos autosupervisados (S1, S1-hard, S3) ponen arriba justo las variantes que peor representan las relaciones tonales (sin blanqueo, solo bajo), con IC que excluyen 0 en las dos direcciones | EVIDENCIA OBSERVADA | tablas de arriba |
| Lo mismo en los 48 temas de la tesis, con tonalidad verificada a mano (S1: FEM#10 0,821, sin blanqueo 0,829, solo bajo 0,816; Spearman Progressive House 0,91, 0,66 y 0,31) | EVIDENCIA OBSERVADA | `proxies_thesis48.json` |
| Encontrar la continuación del mismo tema premia la identidad del tema: cualquier contenido de alturas estable (una línea de bajo, el espectro sin blanquear) alcanza. La armonía en sí no queda medida | INFERENCIA | variantes que pierden información tonal mejoran S1; la regla Camelot, puramente armónica, saca 0,19 |
| Usar como positivos versiones de la misma canción (original y remix) evitaría esa fuga, pero hay 77 pares y solo el 40 % de los que tienen tonalidad comparte la de MIK | EVIDENCIA OBSERVADA | catálogo agrupado por título base |
| La fórmula nueva (pérdida de coherencia) es la mejor en teoría, pero crea "hubs": unos pocos temas casi sin centro tonal aparecen arriba para todos | EVIDENCIA OBSERVADA | hubness 7,86 contra 0,82 |
| En los 48 temas de la tesis, comparar TIV directo ordena mejor que estimar la tonalidad del mismo croma y aplicar Camelot (Spearman Progressive House 0,91 contra 0,68; Techno 0,41 contra 0,28). Muestra chica | EVIDENCIA OBSERVADA | ídem |
| Transiciones reales con TIV (percentil general y dentro de la misma clase Camelot, con el score del tema entero y con últimos 2 min → primeros 2 min) | NO CONCLUYENTE | necesitan la biblioteca completa como candidatos; la extracción termina en ~3,5 h |

### 3.4 Literatura

Revisión hecha por un subagente; las referencias de 2026 no las verifiqué yo.
- Pérez 2020 (UPF): en loops, el TIV por beat superó a otras medidas; contra el TIV entero, la diferencia
  no es significativa.
- Gebhardt et al. 2015 (28 oyentes): un desplazamiento que minimiza la rugosidad superó a la sugerencia de
  tonalidad de Traktor en 8 de 10 mezclas.
- Kim et al. 2020: solo el 2,5 % de 24 202 temas tocados en sets reales estaba transpuesto.
- No encontré ninguna prueba publicada de TIV contra Camelot en transiciones de temas completos.
- Diseño de escucha: el poder lo da el número de ítems distintos, no el de oyentes (Westfall et al. 2014);
  conviene modelar los empates en vez de descartarlos.

## 4. El conflicto, en una frase

El único criterio offline sin etiquetas de tonalidad (el autosupervisado) elige las variantes que el
único criterio de calidad tonal disponible (la teoría de tonalidades) marca como las peores. Gabriel no
quiere usar la teoría como objetivo, porque el TIV no busca emular la clave. El criterio de comportamiento
(transiciones reales) está pendiente, es chico (361 transiciones de desarrollo, 189 de prueba) y está
confundido con la tonalidad que Traktor muestra. Los juicios humanos que existen son 10 ejemplos de 2021,
con una configuración vieja y sin la descripción de los clips.

## 5. Opciones

**P1. Autosupervisado como criterio principal** (propuesta de Gabriel, vigente en el protocolo).
- Hoy elegiría "solo bajo" o "sin blanqueo".
- A favor: no usa etiquetas; no emula la clave; escala a toda la biblioteca.
- En contra: la evidencia de 3.3 indica que mide identidad, no armonía.
- Costo: ninguno extra.

**P2. Teoría de tonalidades como criterio principal**, en los temas de consenso.
- Hoy elegiría la pérdida de coherencia (con su problema de hubs) o el 30 % central.
- A favor: mide si el TIV captura relaciones tonales conocidas; detecta las variantes rotas.
- En contra:
  - contradice la decisión de diseño de Gabriel;
  - con esas etiquetas, Camelot gana por construcción;
  - el conjunto de consenso es chico (66 temas) y sesgado hacia temas con tonalidad clara.

**P3. Transiciones reales de Gabriel como criterio principal**, sobre todo el percentil dentro de la misma
clase Camelot, que no usa la tonalidad como objetivo.
- A favor: refleja elecciones reales de Gabriel; no emula la clave.
- En contra:
  - el efecto de base es chico (percentil 0,60 con la tonalidad que ve);
  - la armonía es uno de muchos factores;
  - probablemente no tenga poder para separar variantes.
- Costo: esperar la extracción (~3,5 h).

**P4. No ajustar con criterios offline.**
- Llevar FEM#10 tal cual a la escucha.
- Los criterios offline quedan solo como guardarraíles para descartar lo roto:
  - S1 detecta fallas de parecido (disonancia de la mezcla);
  - la teoría detecta fallas tonales (solo bajo, sin blanqueo);
  - hubness detecta donantes universales (pérdida de coherencia).
- A favor: no sobreajusta a criterios sin validar; barato.
- En contra: puede dejar mejoras sin probar; la escucha evalúa una sola configuración.

**P5. Calibrar primero con juicios humanos.** Dos variantes:
- **(a)** Reconstruir la encuesta de 2021 y ver qué criterio offline predice las preferencias de esos
  15 oyentes. Necesita la descripción de los clips; son 10 ítems.
- **(b)** Un piloto de escucha corto con Gabriel, con 2 o 3 configuraciones (por ejemplo FEM#10, la mejor
  por teoría y la mejor por autosupervisado), para elegir. Después, la prueba confirmatoria con ítems
  nuevos.
- A favor: calibra los criterios contra oídos.
- En contra: (a) es muy chica; (b) gasta tiempo de Gabriel, que hoy no está disponible.

**Sub-pregunta (b): contra qué comparar en la escucha.** El protocolo compara contra la regla Camelot de
`harmonic.py`: lo que usaría el recomendador sin TIV. Los ítems se eligen donde los dos sistemas eligen
candidatos distintos entre los mismos 30 vecinos CLAP con BPM compatible, y cada sistema transpone su
candidato hasta ±2. Hay dos alternativas:
- comparar contra "sin criterio armónico" (solo CLAP y BPM), más fiel a la idea de la tesis de no medirse
  contra la clave;
- usar los dos brazos.

Criterio de aprobación vigente: con 60 ítems más 10 repetidos, el TIV gana ≥ 60 % de los no empates con
binomial exacta unilateral p < 0,05, y Gabriel coincide consigo mismo en ≥ 70 % de los repetidos. Si no
coincide en ese 70 %, el resultado es no concluyente.

## 6. Ataques a esta evidencia (revisión hostil del propio informe)

- **La muestra de `musica` es preliminar:** 243 de 889 temas de desarrollo. Las conclusiones de 3.3 se
  repiten en los 48 temas de la tesis, con otras etiquetas, lo que reduce el riesgo, pero hay que
  confirmarlas con el split completo.
- **El front-end no es Essentia bit a bit:** es CPU, para Windows y Mac. Reproduce el orden de la tesis
  (Perf: FEM#10 13,3 > 30 % central 12,1 > sin blanqueo 9,7; tesis: 11,26 y 10,44), no los valores.
- **"Teoría" usa etiquetas de tonalidad**, que en este dominio fallan mucho. Por eso se usa solo el
  consenso MIK = Beatport, que es chico y sesgado hacia temas con tonalidad clara.
- **Los negativos de S1 son al azar.** S1-hard (misma tonalidad) apunta en la misma dirección, así que no
  parece un efecto de negativos fáciles.
- **Que una variante gane en teoría no la hace mejor para mezclar.** Esa es la razón por la que decide la
  escucha.

## 7. Reproducción

- Código: rama `harmonic-tiv` de github.com/gbibbo/traktor, `src/v4/harmonic_tiv/` y
  `tools/harmonic_tiv/`. Tests: `tests/v4/test_harmonic_tiv.py` (18 pasan).
- Datos (privados, fuera de Git, en la PC de Gabriel), en `D:\TRAKTOR ML datos\harmonic_tiv\`:
  - conjuntos de evaluación;
  - caché `cache_semitones_v2/`;
  - resultados `proxies_thesis48.json`, `proxies_musica_dev_preliminar_frozen.json` y
    `frozen_ids_preliminar.txt`;
  - encuesta en `encuesta_tesis/`.
- Comandos: `eval_proxies.py --set musica --split dev --no-history --ids-file frozen_ids_preliminar.txt`.
  Los scripts del bootstrap pareado y del reanálisis de la encuesta: anexo B.

## Anexo A. Posición del agente

Separada para que el auditor la ataque y no lo ancle; no es evidencia.

- **Selección:** P4 con los guardarraíles. Después, P5(b) cuando Gabriel pueda escuchar: un piloto chico
  con FEM#10 y una o dos variantes que pasen los guardarraíles (30 % central; sin bajo ni agudos), y la
  prueba confirmatoria con ítems nuevos.
- **Si Gabriel recupera los clips de 2021:** P5(a) como chequeo barato de cuál criterio offline se parece
  a oídos humanos.
- **P3** solo como desempate cuando termine la extracción.
- **Comparación:** contra Camelot como brazo principal, porque es lo que se reemplazaría, y "sin criterio
  armónico" como brazo secundario, para respetar la idea de la tesis.
- **Cambio en el protocolo:** retiraría S1 como criterio principal de ajuste y lo dejaría como
  guardarraíl.

## Anexo B. Scripts de esta sesión

- Reanálisis de la encuesta: `tools/harmonic_tiv/survey_reanalysis.py <csv de la hoja de respuestas>`.
- Bootstrap pareado S1-hard / teoría: `tools/harmonic_tiv/paired_selfsup_boot.py <repo> <eval_dir> <cache>
  <ids congelados>`.

## Anexo C. Respuesta del auditor (2026-11-06)

La respuesta está en `reports/Revisión científica de armonía TIV.md` (en `main`, commit b73afbf), con notas
en `research_notes/Revisión científica de armonía TIV/`. Decisión: **P4 modificada ahora → P5(b)
después; no integrar el TIV todavía.**
- Piloto de escucha con FEM#10 y «sin blanqueo»; se congela una sola configuración.
- Confirmación contra dos controles en orden: el recomendador actual y CLAP+BPM sin armonía.
- 60 contextos nuevos y representativos por comparación.
- Aprobación: U ≥ 0,60 contando los empates, y p < 0,05.
- 20 repetidos, con 16 acuerdos.

Las fallas que señaló están marcadas arriba en su lugar:
- bootstrap de teoría;
- negativos de S1;
- orden por AUC;
- IC por sesión.

Cómo se aplicó: `docs/plans/armonia_tiv_protocolo.md`, sección 9.
