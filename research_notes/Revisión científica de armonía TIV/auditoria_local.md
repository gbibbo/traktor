# Auditoría local del informe TIV

## ¿Qué evidencia pude verificar?

### Takeaway

**EVIDENCIA OBSERVADA:** los resultados centrales transcritos en el informe existen en los artefactos congelados y los 18 tests de TIV pasan. Eso verifica cálculos y comportamiento de implementación; no demuestra que las variantes ganadoras de un proxy produzcan mezclas preferibles.

### Cited Findings

- Fecha de revisión: 2026-11-06. Worktree auditado: `C:/VS projects/traktor_github/.claude/worktrees/harmonic-tiv`; HEAD `b228858722805cd2e66e81261d6006a0e5f68da5`. `git diff 681c3f4 HEAD --stat` mostró exclusivamente 28 líneas cambiadas en el informe, agregando la consigna: no hay discrepancia de versión de código entre el commit indicado y el HEAD leído. `git status --short` del worktree quedó sin salida. [Informe](<C:/VS projects/traktor_github/.claude/worktrees/harmonic-tiv/docs/reports/auditoria_armonia_tiv_2026-11-06.md>)
- Se leyeron artefactos privados existentes en `D:/TRAKTOR ML datos/harmonic_tiv/` mediante acceso de lectura autorizado por la revisión automática. El intento sandbox inicial recibió acceso denegado; no era ausencia de datos. `keys_musica.csv`: 1760 filas, 1760 track_uid únicos; 1187 con ambos tags conocidos, 594 acuerdos exactos MIK–Beatport. `frozen_ids_preliminar.txt`: 411 IDs únicos, todos presentes en keys. El JSON congelado reporta para todas las variantes 411 tracks totales, 243 dev, 66 dev de consenso, 968 consultas S1 y 659 S1-hard. [Resultados congelados](<D:/TRAKTOR ML datos/harmonic_tiv/proxies_musica_dev_preliminar_frozen.json>)
- Valores verificados leyendo JSON: FEM#10 S1 .7748, S1-hard .7529, S3 .708, teoría .512, hubness .82; sin blanqueo .7800/.7870/.7869/.298/.90; solo bajo .7695/.7981/.7659/.184/.52; coherence_loss .7425/.7314/.4244/.550/7.86. Para coherence_loss, máximo 145 apariciones top10 frente a 35 FEM#10. [Resultados congelados](<D:/TRAKTOR ML datos/harmonic_tiv/proxies_musica_dev_preliminar_frozen.json>)
- Todos los campos históricos A4/A4b/asym del resultado congelado tienen `n: 0`; son resultados pendientes en ese artefacto, no resultados negativos. No inspeccioné si otra corrida terminó posteriormente. [Resultados congelados](<D:/TRAKTOR ML datos/harmonic_tiv/proxies_musica_dev_preliminar_frozen.json>)
- Entorno verificado antes de ejecutar proyecto: Python Store 3.11.9, numpy 2.3.2, scipy 1.16.1, pandas 2.3.1, pytest 8.4.2. Comando `python3.11 -m pytest tests/v4/test_harmonic_tiv.py -q --basetemp 'C:/VS projects/traktor_github/.pytest_tmp_tiv_audit' -p no:cacheprovider`: **18 passed in 23.02s**. [Tests](<C:/VS projects/traktor_github/.claude/worktrees/harmonic-tiv/tests/v4/test_harmonic_tiv.py>)
- Reejecuté `python3.11 tools/harmonic_tiv/survey_reanalysis.py 'D:/TRAKTOR ML datos/harmonic_tiv/encuesta_tesis/csv/Respuestas_de_formulario_1.csv'`: 15 oyentes, 20 clips, 32 empates/150, 84/118=.712, chi²21.78 p=3.1e-6; 8/10 ejemplos coinciden, binomial p=.055; Spearman .652 p=.041; TPR al retirar un ítem .683–.769. El código usa signo de diferencia de puntajes medios por ejemplo, aunque imprime «mayoría»; en estos diez ejemplos ambos criterios coinciden en dirección. [Reanálisis](<C:/VS projects/traktor_github/.claude/worktrees/harmonic-tiv/tools/harmonic_tiv/survey_reanalysis.py:43>)

### Inferences

- **INFERENCIA:** existe un desacuerdo real entre proxies; no es simplemente una tabla inventada o N diferente entre variantes. La muestra sigue siendo de desarrollo y parcial.
- **INFERENCIA:** los tests sustentan consistencia algebraica y casos sintéticos. No validan eficacia perceptual ni equivalencia de todo el front-end con Essentia/TIV.lib original. El test de TIV compara 50 pares contra fórmulas reescritas, no ejecuta 2000 pares contra la dependencia original. [Test de fórmulas](<C:/VS projects/traktor_github/.claude/worktrees/harmonic-tiv/tests/v4/test_harmonic_tiv.py:26>)

### Gaps

- No reejecuté extracción, bootstrap pareado ni experimento completo, ni modifiqué artefactos D:. No comprobé finitud de cada NPZ, ni la prueba original de 2000 pares, ni reconstruí audios de la encuesta. Los números del JSON están corroborados como artefactos, no todos regenerados.
- No reestimé el percentil histórico .603/.520 ni sus intervalos. Sí inspeccioné la implementación estadística usada para A4 y sus limitaciones abajo.

## ¿Qué fallas afectan la decisión?

### Takeaway

**EVIDENCIA OBSERVADA:** hay una implementación defectuosa del bootstrap de teoría, un control de escucha que difiere del recomendador actual, y una brecha entre proxies sin transposición y la política final con ±2. **INFERENCIA:** corresponde conservar candidatos y corregir el diseño antes de declarar exclusiones perceptuales o éxito de integración.

### Cited Findings

- **Bootstrap de teoría:** `paired_selfsup_boot.py` remuestrea índices de tracks y luego ejecuta `list(dict.fromkeys(cons_dev[k] for k in b))`. Esto elimina las multiplicidades del remuestreo. No es el bootstrap ordinario que describe el informe, sino una submuestra aleatoria de elementos distintos (aproximadamente 63 % de los originales en promedio). Los IC publicados de teoría requieren una reestimación apropiada por tracks/díadas, reteniendo multiplicidades y manejando explícitamente pares del mismo track original. No invalida automáticamente el signo de las diferencias, pero impide tratar esos IC como verificados correctamente. [Código: líneas 77–89](<C:/VS projects/traktor_github/.claude/worktrees/harmonic-tiv/tools/harmonic_tiv/paired_selfsup_boot.py:77>)
- **Control distinto del producto:** la escucha toma `k_tag` y elige máximo score Camelot con CLAP para desempatar. El main actual toma Traktor cuando está disponible, filtra candidatos con score≥.60 y por defecto ordena solamente por CLAP. También difiere el filtro BPM (escucha ±6 % admitiendo doble/mitad; producto ±20 BPM). Por tanto «superar a Camelot» en esa prueba no equivale automáticamente a superar al recomendador operativo. [Escucha](<C:/VS projects/traktor_github/.claude/worktrees/harmonic-tiv/tools/harmonic_tiv/listening_test.py:74>), [Prioridad Traktor](<C:/VS projects/traktor_github/tools/traktor_live/recomendador.py:210>), [Pesos por defecto](<C:/VS projects/traktor_github/tools/traktor_live/recomendador.py:60>), [Filtros](<C:/VS projects/traktor_github/tools/traktor_live/recomendador.py:261>), [BPM proxy](<C:/VS projects/traktor_github/.claude/worktrees/harmonic-tiv/src/v4/harmonic_tiv/evaluate.py:155>)
- **«Sin armonía» exige retirar el filtro:** poner peso clave=0 ya es el defecto actual, pero Camelot sigue filtrando. El brazo sin armonía debe quitar tanto ese filtro como el score y fijar qué hace con pitch, no limitarse a cambiar el peso. [Código](<C:/VS projects/traktor_github/tools/traktor_live/recomendador.py:264>)
- **Negativos incorrectamente descritos:** S1 extrae 200 fragmentos de un pool de fragmentos de otros tracks, no fragmentos de 200 tracks distintos. S1-hard análogamente usa 50 fragmentos y puede representar muchos menos que 50 tracks. Tracks más largos aportan más fragmentos al pool. [Implementación](<C:/VS projects/traktor_github/.claude/worktrees/harmonic-tiv/src/v4/harmonic_tiv/evaluate.py:198>)
- **MRR con empates:** `_rank_metrics` usa `1/(1+better+ties/2)`, recíproco del rango medio; no es el promedio del rango recíproco bajo desempate aleatorio. Es una convención reproducible, pero penaliza scores discretos como Camelot de modo distinto. Con tres empatados en posiciones 1–3 da .5, mientras desempate aleatorio da .6111. Su MRR .188 no prueba que Camelot carezca de señal: en el mismo JSON tiene AUC .8713. [Implementación](<C:/VS projects/traktor_github/.claude/worktrees/harmonic-tiv/src/v4/harmonic_tiv/evaluate.py:187>), [JSON](<D:/TRAKTOR ML datos/harmonic_tiv/proxies_musica_dev_preliminar_frozen.json>)
- **Objetivo sin transposición:** S1, S3 y correlación de teoría usan score sin transponer. Calcular además una matriz con 12 transposiciones no cambia ese score principal. Escucha permite ±2 y penaliza desplazamiento. Los resultados offline no identifican directamente la mejor política de recomendación con transposición. [Matriz](<C:/VS projects/traktor_github/.claude/worktrees/harmonic-tiv/tools/harmonic_tiv/eval_proxies.py:68>), [S1](<C:/VS projects/traktor_github/.claude/worktrees/harmonic-tiv/src/v4/harmonic_tiv/evaluate.py:202>), [Escucha](<C:/VS projects/traktor_github/.claude/worktrees/harmonic-tiv/tools/harmonic_tiv/listening_test.py:80>)
- **El número bajo de S1 no es contradicción de calidad armónica:** positivo=continuación propia; muchos negativos del mismo tono pueden ser perfectamente compatibles. En hard, Camelot empata por construcción; no identificar exactamente el positivo es limitación de la tarea de identidad, no necesariamente falla para mezcla. La hipótesis de atajos es razonable, pero falta un contraste que aísle identidad de compatibilidad. [Construcción de tarea](<C:/VS projects/traktor_github/.claude/worktrees/harmonic-tiv/src/v4/harmonic_tiv/evaluate.py:238>)
- **AUC y MRR ni siquiera ordenan igual las variantes:** en frozen, S1-hard AUC FEM#10=.9464; sin blanqueo=.9411; solo bajo=.9342, mientras los dos últimos ganan en MRR. Debe decirse «ganan en MRR de recuperación», no genéricamente que ganan en autosupervisión. [JSON](<D:/TRAKTOR ML datos/harmonic_tiv/proxies_musica_dev_preliminar_frozen.json>)
- **Dependencia histórica:** A4/A4b bootstrap remuestrea percentiles de transiciones individualmente, sin agrupar sesiones o pistas repetidas. Esos IC no corrigen esa dependencia. Condicionar por clase Camelot no remueve confusión por exposición, repertorio, género, energía o preferencia de timbre. [Bootstrap](<C:/VS projects/traktor_github/.claude/worktrees/harmonic-tiv/src/v4/harmonic_tiv/evaluate.py:178>)
- **Muestra de escucha por desacuerdo y cuotas:** S1/S2, 30+30 por defecto, solo candidatos diferentes. Además la exclusión `used` revisa los candidatos, pero no `t in used`; un objetivo nuevo puede ser un candidato de un ítem anterior. Ítems no necesariamente disjuntos por pista, menos aún artista. Inferencia sobre toda la biblioteca requiere cobertura y pesos de estratos, o muestra representativa aparte. [Selección](<C:/VS projects/traktor_github/.claude/worktrees/harmonic-tiv/tools/harmonic_tiv/listening_test.py:68>)
- **Guardarraíl vigente imposible si se interpreta literalmente por skew:** protocolo pide hubness no peor que K0, pero K0=.−365 en frozen y todas las variantes TIV tienen skew positivo, incluida FEM#10. Deben fijarse métrica, margen y población de candidatos antes de usarlo como veto; además Camelot rompe empates agregando ruido aleatorio mientras TIV no. [Regla](<C:/VS projects/traktor_github/.claude/worktrees/harmonic-tiv/docs/plans/armonia_tiv_protocolo.md:115>), [Empates K0](<C:/VS projects/traktor_github/.claude/worktrees/harmonic-tiv/tools/harmonic_tiv/eval_proxies.py:215>)

### Inferences

- **INFERENCIA:** «menor correlación con quintas» no autoriza «variante rota para mezclar». Usar teoría como veto reproduciría P2 encubierta. Usarla como diagnóstico y advertencia sí es defendible.
- **INFERENCIA:** acuerdo Traktor–MIK de 91 % no demuestra por sí solo dependencia de fuentes, ni desacuerdo MIK–Beatport establece quién está equivocado. En este sistema existe evidencia de prioridad/importación en código, que debe distinguirse del porcentaje.
- **INFERENCIA:** una baja concentración global puede ocultar hubs dentro del verdadero pool CLAP+BPM, y alta concentración puede reflejar tracks neutralmente mezclables. Hubness es indicador de diversidad y cobertura; no verdad auditiva autónoma.

### Gaps

- Los resultados no permiten fijar un umbral de correlación de teoría que garantice buena escucha. Cualquier corte nuevo sería una decisión operativa propia, no un umbral científico validado.
- No comprobé empíricamente el impacto de la duplicación entre ítems en una prueba de 60 ya renderizada; detecté la posibilidad en el código.

## ¿Qué decisión propongo con esta evidencia?

### Takeaway

**RECOMENDACIÓN PROPIA:** P4 ahora, pero sin convertir los proxies en vetos perceptuales; P5(b) después para elegir entre candidatos previamente congelados. P3 como evidencia secundaria, no criterio de aprobación ni desempate automático.

### Cited Findings

- Reservar FEM#10 como referencia implementada, y una alternativa que represente el conflicto (p. ej., sin blanqueo); para probar sección local contra global, conservar el resto idéntico. El anexo A preselecciona variantes favorecidas por teoría y excluye las ganadoras del proxy rival sin escucha; esto contradice su aparente neutralidad entre criterios. [Anexo A](<C:/VS projects/traktor_github/.claude/worktrees/harmonic-tiv/docs/reports/auditoria_armonia_tiv_2026-11-06.md>)
- Distinguir prueba de componente armónico y prueba de producto. Para aprobar integración, el comparador primario debe ser la política operativa congelada, con misma colección, pool, BPM, transposición permitida, renderizado y fuente de tonalidad. Un control CLAP+BPM sin filtro armónico sirve para probar valor añadido. [Control implementado](<C:/VS projects/traktor_github/.claude/worktrees/harmonic-tiv/tools/harmonic_tiv/listening_test.py:74>), [Producto](<C:/VS projects/traktor_github/tools/traktor_live/recomendador.py:261>)

### Inferences

- No gastaría la prueba confirmatoria eligiendo a la vez representación, sección, fórmula y penalización de pitch. Congelar finalista tras piloto con ítems separados y luego probar una sola hipótesis primaria.
- Mantener empates reportados, no equiparar ausencia de significación con equivalencia ni fracaso demostrado. El agente de estadística aporta potencia y propuesta numérica.
- Cambiaría de selección offline solo después de validar que un proxy predice preferencias humanas en ítems nuevos y en los casos de desacuerdo; hoy eso está NO CONCLUYENTE.

### Gaps

- No propongo integración hoy: aún falta escucha humana de la política final. Esta auditoría no edita el protocolo ni decide por Gabriel ninguna configuración en producción.
- No se modificó código ni documentación fuente del worktree ni datos de D:. Solo este archivo de notas, y la salida temporal de pytest en un directorio explícito del repositorio principal. No se hizo commit/push desde este subagente.
