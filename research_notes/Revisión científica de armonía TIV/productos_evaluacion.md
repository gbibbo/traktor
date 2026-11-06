# Productos de DJ y evaluación de compatibilidad armónica

Fecha de consulta: 2026-11-06. Informe revisado: `C:/VS projects/traktor_github/.claude/worktrees/harmonic-tiv/docs/reports/auditoria_armonia_tiv_2026-11-06.md`, completo. No se ejecutaron extracciones ni se modificó el protocolo. Fuentes primarias; las prestaciones documentadas de un producto no prueban eficacia perceptual ni superioridad del algoritmo.

## ¿Qué prácticas muestran los productos existentes?

### Takeaway
Los productos combinan características musicales, compatibilidad de claves y decisiones del DJ. La práctica trasladable es escuchar el solapamiento concreto y controlar frases, tempo y transposición; no convertir una etiqueta tonal o el éxito comercial en verdad perceptual.

### Cited Findings
- **EVIDENCIA OBSERVADA, documentación oficial:** Mixed In Key recomienda preseleccionar por Camelot y escuchar las secciones que se solaparán; advierte que las relaciones de la rueda no garantizan que cualquier pareja mezcle bien. También enumera BPM, energía, voces, bajos y fraseo como consideraciones. — [Mixed In Key, How to Use the Camelot Wheel](https://mixedinkey.com/workflows/how-to-use-the-camelot-wheel/), líneas web 138, 171, 230–237.
- **EVIDENCIA OBSERVADA, manual:** rekordbox 7.2.14, pp. 35–36, describe Collection Radar como recomendaciones por características musicales similares y permite condiciones BPM, Key y Vocal; pp. 164–165 documentan Key Sync y desplazamiento por semitonos. La documentación no publica un ensayo ciego que demuestre superioridad perceptual. — [Manual oficial rekordbox](https://cdn.rekordbox.com/files/20260409151936/rekordbox7.214_manual_EN.pdf).
- **EVIDENCIA OBSERVADA, documentación oficial de 7 octubre 2026:** DJ.Studio Harmonize pondera BPM/clave y permite definir estilo y longitud de transición; coloca transiciones intro/outro y recomienda revisar manualmente el resultado. — [DJ.Studio Harmonize](https://help.dj.studio/en/articles/7878402-harmonize-previously-automix).
- **EVIDENCIA OBSERVADA, documentación oficial de 17 septiembre 2026:** DJ.Studio 5.0 documenta análisis por frases y opciones de ubicación de mezcla por barras, solapamiento intro/outro, posición de frase y estructura. — [DJ.Studio Phrase Detection](https://help.dj.studio/en/articles/16985613-phrase-detection-in-dj-studio).
- **EVIDENCIA limitada al extracto indexado del manual oficial:** Traktor Pro 4 permite fijar la clave original y transponer por semitonos. La descarga textual del PDF completo fue rechazada por tamaño, y el HTML devolvió 403. No usar este resultado para afirmaciones más amplias. — [Manual oficial Traktor Pro 4](https://www.native-instruments.com/fileadmin/ni_media/downloads/manuals/traktor/Traktor-Pro-4-Manual-English-170724.pdf).

### Inferences
- **INFERENCIA PROPIA:** comparar TIV contra Camelot es una comparación de decisiones de producto, no exigir que TIV prediga etiquetas Camelot. Añadir CLAP+BPM sin armonía comprueba si añadir armonía aporta algo respecto del sistema más sencillo.
- **INFERENCIA PROPIA:** evaluar dos candidatos distintos identifica utilidad del recomendador completo; no aísla una causa puramente armónica, pues cambian también identidad y propiedades de las canciones. Para la conclusión más estrecha sobre armonía conviene un diagnóstico adicional de pares fijos/transposiciones, sin usarlo para seleccionar después sobre test.
- **RECOMENDACIÓN PROPIA:** mismo motor de audio, BPM final, ganancia percibida, ventana y política de transposición para los brazos; permitir ±2 semitonos en ambos sistemas que la optimizan, y fijar por anticipado la política del control sin armonía. Mantener ocultas claves, algoritmo y orden de preferencia.

### Gaps
- No se verificó ningún benchmark independiente de estos productos contra TIV. Sus manuales no establecen qué variante TIV elegir.
- Hay diferencias entre las páginas de DJ.Studio sobre opciones exactas del asistente; basta afirmar que dispone de control de estructura/transiciones, sin prometer una pantalla específica.

## ¿Qué exige un experimento interpretable y qué falla en el informe?

### Takeaway
Los 60 ítems y 10 repeticiones permiten una decisión personal si el efecto es grande, pero no prueban equivalencia cuando no hay significación ni garantizan buena potencia para una mejora real del 60 %. El análisis debe conservar empates, aislar selección de confirmación y limitar la población de inferencia a Gabriel.

### Cited Findings
- **EVIDENCIA OBSERVADA, artículo primario:** Westfall, Kenny y Judd (2014) estudian diseños cruzados de participantes y estímulos; más participantes pueden no compensar una muestra pequeña de estímulos. No dice que el número de participantes carezca de importancia. — [Artículo y abstract original, DOI 10.1037/xge0000014](https://pubmed.ncbi.nlm.nih.gov/25111580/).
- **EVIDENCIA OBSERVADA, artículo de los mismos autores (2015):** replicar con nuevos estímulos evita atribuir a una población lo que sólo ocurre en el pequeño conjunto original. — [Replicating Studies in Which Samples of Participants Respond to Samples of Stimuli](https://www2.psych.ubc.ca/~schaller/528Readings/WestfallJuddKenny2015.pdf), DOI 10.1177/1745691614564879.
- **EVIDENCIA OBSERVADA, artículo primario:** Davidson (1970) extiende las comparaciones Bradley–Terry para respuestas sin preferencia; en diseño equilibrado el ranking coincide con puntuar victoria/empate/derrota como 2/1/0. Sólo se consultó el abstract indexado del editor; apertura directa devolvió 403. — [Davidson, DOI 10.1080/01621459.1970.10481082](https://www.tandfonline.com/doi/abs/10.1080/01621459.1970.10481082).
- **EVIDENCIA OBSERVADA, artículo primario:** Schnabel et al. (2016) documentan sesgos de autoelección y de exposición en datos de recomendación. — [Recommendations as Treatments: Debiasing Learning and Evaluation](https://proceedings.mlr.press/v48/schnabel16.html).
- **EVIDENCIA OBSERVADA, artículo primario:** Cawley y Talbot (2010) muestran que optimizar y evaluar sobre muestras finitas puede producir sobreajuste del criterio de selección y evaluación optimista. — [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf).
- **EVIDENCIA OBSERVADA, referencia primaria de multiplicidad:** Holm (1979) ofrece control de error familiar con pruebas múltiples. Si se prueba una sola configuración confirmatoria frente a dos controles en orden fijo, también puede emplearse una jerarquía preespecificada: la segunda afirmación sólo se habilita si pasa la primera. — [Holm 1979](https://www.ime.usp.br/~abe/lista/pdf4R8xPVzCnX.pdf). La recomendación de jerarquía aquí es deducción estadística propia, no una cita del procedimiento Holm.

### Inferences
- **FALLA, INFERENCIA PROPIA:** sección 3.4 simplifica incorrectamente Westfall como “el poder lo da el número de ítems, no el de oyentes”. Ambos importan para generalizar; con un juez sólo se puede concluir sobre ese juez, sus circunstancias y el universo de canciones muestreado.
- **FALLA, INFERENCIA PROPIA:** elegir sólo desacuerdos entre algoritmos estima una mejora condicionada al desacuerdo. Hay que informar su frecuencia en el universo; no extrapolar el porcentaje a todas las recomendaciones. Una confirmación de utilidad global debería muestrear contextos representativos antes de saber las salidas; casos idénticos conservan aporte de empate, o emplear estratos con ponderaciones de población fijadas.
- **FALLA, INFERENCIA PROPIA:** 550 transiciones no son necesariamente 550 unidades independientes: pueden compartir pista, artista, sesión, exposición y contexto. Comparar dentro de una clase Camelot reduce una confusión, pero no identifica por sí solo el efecto causal armónico.
- **FALLA, INFERENCIA PROPIA:** la continuación de un mismo tema puede reflejar identidad, pero la evidencia no demuestra que no refleje ninguna armonía. Que una variante pierda correlación con quintas no prueba que esté “rota” para mezcla. Usar teoría como veto duro reinstala P2 indirectamente.
- **FALLA, INFERENCIA PROPIA:** S1-hard usa etiquetas MIK para elegir negativos, por tanto no es estrictamente “sin etiquetas”. Un remix tampoco garantiza independencia de identidad ni compatibilidad local.
- **RECOMENDACIÓN PROPIA:** separar piloto de selección y test por pistas/artistas; congelar una sola configuración, pesos, transposición y renderizado antes de test. Las repeticiones no aumentan el N de ítems. Si las pistas aparecen en varios ítems, usar unidad agrupada o un modelo apropiado y calcular potencia con esa dependencia, no binomial independiente.

### Gaps
- Faltan la tasa esperada de empates, dependencia por pista/artista y distribución del efecto. No se puede dar un N universal que garantice 80 % de potencia para el protocolo real.
- Un acuerdo observado de 7/10 tiene un IC Wilson 95 % aproximadamente [0.397, 0.892]. Por tanto el umbral 70 % no establece fiabilidad alta; es una decisión operativa. Preferencias débiles pueden producir desacuerdo sin fallo técnico del juez.

## ¿Qué decisión concreta y qué umbrales recomiendo?

### Takeaway
P4 modificada ahora, P5(b) después, P3 sólo diagnóstico. No usar las métricas indirectas para declarar un ganador perceptual ni descartar por teoría una variante que justamente pone a prueba esa teoría.

### Cited Findings
- **EVIDENCIA COMPUTADA EN ESTA SESIÓN, cálculo propio:** suma exacta binomial `sum(comb(n,j)*p^j*(1-p)^(n-j), j=k..n)`, ejecutada en JavaScript (sin Python ni imports del proyecto). Para 60 ítems independientes sin empates, el primer resultado con p unilateral <.05 es 37/60: p=.04623049. Bajo probabilidad verdadera de victoria .60, P(X≥37)=.45110639. Son aproximadamente 45.1 % de potencia.
- **EVIDENCIA COMPUTADA, cálculo propio:** para N=160, la sola prueba de superioridad requiere 91 victorias: alfa=.04828468; potencia bajo p=.60=.81290017. **Esto no justifica afirmar 81 % de potencia si también se exige un porcentaje observado ≥60 %**: ese requisito añade un umbral que se encuentra en el efecto verdadero.
- **EVIDENCIA COMPUTADA, cálculo propio:** IC Wilson95 de 7/10=[.396773,.892211]; de 16/20=[.583978,.919344]. Fórmula Wilson estándar, z=1.96, calculada directamente. Son incertidumbres del acuerdo, no validación del umbral.

### Inferences
1. **Decisión (a), propuesta propia:** congelar FEM#10 como referencia, completar comprobaciones de implementación y diagnóstico sobre desarrollo, conservar un candidato contrastante (por ejemplo sin blanqueo) para un piloto posterior en lugar de decidirlo por S1 o teoría. Cuando Gabriel pueda, P5(b) con 2–3 configuraciones y luego una sola configuración elegida para confirmación con ítems nuevos. P5(a) sería exploratorio; diez ejemplos históricos no bastan para calibrar múltiples variantes con validación fiable. P3 puede revelar fallos pero no desempatar como si fuera verdad de armonía.
2. **Decisión (b), propuesta concreta para presupuesto acotado:** 60 contextos nuevos por comparación, muestreados de la población prevista; Camelot primario, CLAP+BPM sin criterio armónico secundario, con orden fijo preespecificado. Para integrar exigir **en ambos** U=(victorias+0.5×empates)/60≥.60 y p unilateral<.05. El segundo contraste sólo tiene interpretación confirmatoria si pasa el primero. Es un umbral de producto propuesto, no recomendado literalmente por un paper. Si se desea demostrar sólo reemplazo de Camelot, aprobar únicamente con el primario sería otra política; exigir ambos sostiene además utilidad del componente frente a omitirlo.
3. **Prueba:** aleatorización de A/B y prueba pareada de signos/permutación bajo nulidad de simetría; los empates aportan 0 a diferencias y 0.5 a U. Que no contribuyan a la estadística de signos es matemático; no deben desaparecer de la estimación de utilidad ni del reporte. Bajo independencia y sin empates, esto recupera el umbral 37/60. Con pistas compartidas, emplear bloques apropiados en la aleatorización y el cálculo; no presentar el binomial simple como exacto.
4. **Repeticiones y fallos:** incluir repeticiones separadas para diagnóstico, excluidas del N confirmatorio. Si se necesita una regla operacional previa, proponer 20 repeticiones, acuerdo ≥16/20, y marcarla explícitamente como propia y sin validación externa; registrar respuestas empate/victoria/derrota por identidad, no por botón A/B. Si no hay consistencia o aparecen artefactos de reproducción, resultado no concluyente y revisar instrumento. No sostener que este corte vuelve fiable al juez con certeza.
5. **Qué cambiaría decisión:** una alternativa que supere a FEM#10 en piloto puede ser la única confirmatoria, pero debe comprobarse con ítems nuevos según lo anterior. Falta de aprobación no implica igualdad ni invalidez de TIV. Para buscar cambios verdaderos de .60 con 80 % de potencia, diseñar un estudio mayor con tasa de empates y margen útil fijados a partir del piloto; no ampliar N repetidamente hasta conseguir significación.
6. **Validez del instrumento:** misma duración/posición de mezcla, curva de ganancias, motor de pitch/time-stretch y BPM final entre brazos, y preregistro de las pocas condiciones que cambian. Escuchar fragmentos realmente superpuestos, no canciones aisladas ni sólo su croma promedio.

### Gaps
- No existe en las fuentes revisadas un umbral universal de U=.60, N=60 o acuerdo16/20 para mezcla armónica personalizada. Son reglas explícitas de decisión, no hechos científicos.
- La confirmación puede ser costosa si los contextos requieren largas mezclas; sesiones cortas con descansos y orden aleatorio son una decisión de diseño propia. La disponibilidad de Gabriel determina cuándo se puede obtener evidencia perceptual, no la validez de sustituirla por un proxy no validado.
