# Decisiones de proyecto

Registro fechado de decisiones de Gabriel que condicionan el trabajo. Cada entrada dice qué se
decidió y qué implica para el código. Las decisiones más recientes prevalecen sobre planes
anteriores (`v4_implementation_plan.md`, `dj_music_clustering_deterministic_implementation_plan_v6.md`).
Una decisión revertida o reemplazada no se borra: se marca en su lugar con
`[REEMPLAZADO <fecha> por <decisión>: <lo que rige ahora>]` (o `[REFUTADO …]` si era un error).

## 2026-11-13 (plan v2: el dataset en la GPU y el teléfono completo)

Plan vigente: `docs/plans/plan_recomendacion_v2.md`. Inventario de ideas: `docs/plans/inventario_ideas.md`.

1. **Con el dataset completo empiezan los experimentos pesados en la GPU** (Gabriel: «quiero que comencemos a usarlo
   corriendo experimentos pesados en el GPU del cluster con el dataset completo»; confía en que DJSets, por tamaño y
   curación, es lo que le va a dar al sistema capacidad real de recomendar). Cómo (agente): línea A del plan, pasos A0
   a A8. Dos lecturas del agente que Gabriel tiene que confirmar: «la GPU del cluster» es gpuq, y el pedido cubre
   entrenar modelos con el dataset, de uso privado (C1; Beatport declara `ai-train=no`).
2. **El teléfono tiene que ser una herramienta completa para elegir la música** (Gabriel): presets de filtros por BPM,
   género y otros, y un campo de texto donde se pueda describir cualquier cosa: un momento de la noche, el nombre de
   un archivo de música parecido, un concepto, un género.
3. **El LLM es tonto y los deberes se hacen de antemano** (Gabriel): sus documentos de contexto llevan reglas
   exhaustivas, claras y eficientes de cómo filtrar, desde ajustar los pesos de la ecuación de similitud hasta
   filtrar exacto por BPM o por género. Cómo (agente): el LLM llena un formulario de listas cerradas y el ordenador
   lo ejecuta (plan, sección 3; ideas L1 a L12, que reemplazan a T31).
4. **Gabriel pidió reescribir el plan** con estos puntos. Cómo (agente): dos líneas en paralelo; el plan viejo queda
   como historia; sus 124 ideas y las 21 nuevas están en el inventario, cada una con su lugar.
5. **Sesiones** (recomendación del agente; espera a Gabriel): dos chats nuevos, uno por línea, que empiezan con
   `PROMPT_LINEA_A_DATASET.md` y `PROMPT_LINEA_B_TELEFONO.md`; la auditoría, en chats cortos por ronda.
6. **Estado encontrado ese día** (agente): la escucha E12 sigue sin respuestas; V7 dejó 60 archivos de vistas en el
   Drive y no se contó cuántos audios cubren; Kaggle hizo 10 266 previews a 2,04 s cada uno y Colab cortó las
   sesiones; el árbol de trabajo está en la rama `dataset-djsets`, que Gabriel tiene que unir a `main`.
7. **Respuestas de Gabriel, el mismo día.**
   - **La GPU es gpuq, con Kaggle y Colab según la demanda.** Preguntó si Colab se cortaría a mitad de camino; creía
     que gpuq era robusto frente a eso. Lo que muestra el registro (agente): gpuq hizo su parte, porque detectó cada
     corte, relanzó y el trabajo siguió desde su parcial; no puede evitar el corte. En `traktor-v7-4`, Colab perdió la
     sesión a los 13 minutos y gpuq dio el trabajo por fallido en el intento 11; el mismo trabajo en Kaggle corrió
     5 h 50 min sin cortes. Regla que sale de ahí (plan, 2.5): guardar cada pocos minutos, partir en trabajos chicos,
     y pedir Kaggle para lo que necesita horas seguidas. [REEMPLAZADO 2026-11-13 por
     `docs/reports/colab_corte_13min_2026-11-13.md`: del 2-nov a las 17 UTC al 10-nov Colab recuperó toda VM T4
     lanzada por el CLI a los 12-14 min de asignada, también en ERC-LM, que en octubre corría horas con el mismo CLI y
     el mismo diseño; no lo provocaba nuestro trabajo, y el 13-nov el patrón de gpuq volvió a durar 35 min. La causa
     del lado de Colab no se puede ver desde acá] Por qué Colab corta a los 13 minutos no se sabe.
   - **Se entrena con el dataset completo, de uso privado.** Cierra C1.
   - **Unir la rama:** hecho por el agente, con avance rápido; `main` quedó en `dcd93a3`.
   - **Pidió el HTML de la escucha con las indicaciones.** Cómo (agente): `tools/playlist_review/pagina_escucha.py`
     arma `e12\escucha.html` desde las listas; no lee `clave.json`.
   - **No puede escribir 50 pedidos de ejemplo; supervisa y agrega algunos.** Cómo (agente): borrador de 82 pedidos
     en `docs/plans/pedidos_telefono_borrador.md`, que Gabriel corrige editando el archivo.
   - **El filtro de género es blando cuando el género viene del texto,** porque las etiquetas son las de Beatport y
     es difícil que alguien las escriba tal cual, **y duro cuando se eligió en un preset del teléfono.** Puede cambiar
     más adelante.
8. **Respuestas de Gabriel a la línea B, el mismo día.**
   - **Género escrito en el texto: blando, y se aceptan errores de escritura** («tecno» por techno). Preguntó si el
     LLM debería leer «oscuro» como Techno (Raw / Deep / Hypnotic). Respuesta del agente: no. «Oscuro» describe un
     sonido y va a la casilla texto («dark»), donde CLAP lo busca en cualquier género. Deducir un género sacaría temas
     que el DJ no excluyó. La regla del reglamento será que solo una palabra de género, o una con un error de
     escritura, llena la casilla de géneros. Blando (agente): el género pedido va primero y entran los vecinos de la
     taxonomía con parecido de 0,75 o más; por ejemplo, Tech House trae House, Indie Tech y Minimal / Deep Tech, y no
     Dance / Pop.
   - **La tabla de momentos se mide en DJSets, no se escribe suponiendo.** Medida (agente):
     `docs/reports/momentos_djsets_2026-11-13.md` y `config/telefono/momentos.json`, en `"estado": "propuesta"`.
     [REEMPLAZADO 2026-11-13 por el punto 10: la forma la elige el agente y Gabriel la cambia si quiere] Espera que
     Gabriel elija su forma y qué hacer con after. Hasta entonces el momento no filtra.
   - **Los 50 pedidos los arma con el agente orquestador** (`pedidos_telefono_borrador.md`). [REEMPLAZADO 2026-11-13
     por el punto 9: Gabriel no revisa la semilla; sus pedidos reales salen del registro del teléfono]
   - **Los BPM a la mitad se corrigen en Traktor.** Hecho (agente): `traktor_collection.py bpm-doble`, con Traktor
     cerrado y respaldo en `artifacts/v4/traktor_backups/collection_20261113_120543.nml`. Corrige 20 temas. Solo toca
     un BPM de Traktor fuera de 90-180 que es la mitad o el doble del de Beatport o del tag original. En otros 9 temas
     la referencia era la que estaba a la mitad, y esos no se tocaron.
   - **Sus 15 carpetas no se usan como filtro** (Gabriel: tienen mucho ruido). El agente retira la propuesta: son
     grupos de CLAP y BPM, así que no agregan nada a la búsqueda por parecido.
   - **Presets aprobados:** seguir lo que suena, más rápido, más lento, sin voces, los 6 géneros grandes y los
     momentos cuando apruebe la tabla. **Pantalla:** la contraseña se pide una vez y queda guardada; lo que suena va
     arriba, en una línea, y se refresca sin gastar el cupo gratuito de Cloudflare. Hecho en el Worker v6, en local:
     el teléfono pide el estado cada 20 s solo con la pantalla a la vista, y los presets no llaman al LLM.
9. **Los ejercicios que se le piden a Gabriel se adaptan a una persona ocupada, que se cansa** (Gabriel, a la
   auditoría: la escucha E12 de 10 × 60 temas, con la cuenta hecha de memoria, «es inviable», y no abrió los 82 pedidos
   porque esperaba el mismo error). Error del agente auditor, que propuso las dos cosas. Cómo (agente), regla
   «presupuesto humano» en `docs/reports/escucha_corta_e12_2026-11-13.md` y en el plan, 2.4: toda tarea para Gabriel
   dice antes cuánto lleva, entra en 15 minutos y tiene una decisión por pantalla; lo que un agente puede decidir con
   un valor por defecto o revisar por su cuenta no se le pregunta; sus pedidos reales se juntan del registro del
   teléfono.
   - **E12 corta,** fijada antes de cualquier respuesta: por tema principal, los 3 primeros temas propios de cada
     lista, 10 s cada uno, y una elección; unos 12 minutos. Se lee igual: 9 o 10 de 10 deciden. Mide el comienzo de
     las listas, no los 30 temas. Página: `artifacts\v4\escucha\e12\escucha.html`.
   - **Los 82 pedidos** los revisa la auditoría. Sus cuatro preguntas quedaron como valores por defecto del agente.
10. **La línea B, con el presupuesto humano del punto 9** (agente; Gabriel cambia cualquiera cuando quiera, sin contestar).
   - **Tabla de momentos:** el momento ordena y no filtra, porque el estilo mueve el BPM más que el momento y una banda
     dura dejaría fuera el techno de 130 o más. Con lo que suena, la ventana de BPM se corre según lo medido: la
     apertura, unos 2,5 BPM por debajo; el cierre, unos 2 por encima. Los géneros que más pesan en ese momento van
     primero. After queda vacío: el teléfono dice que todavía no hay datos. [REEMPLAZADO 2026-11-13: implementado,
     ver el punto siguiente] Implementarlo es lo próximo de la línea B (L10); hasta entonces el momento se ignora con
     aviso.
   - **El momento ordena desde el 13-11** (agente; `ejecutor.py`, presets publicados). El puntaje del momento mezcla
     dos cosas, con el género el doble de peso:
     - el género: cuánto más aparece en ese momento que en toda la noche;
     - el BPM: la cercanía a lo que suena más el corrimiento medido. Sin tema sonando, la cercanía a la mediana de ese
       momento.

     Se promedia con el puntaje de siempre y no saca temas.
     - Con la biblioteca: la apertura trae 123 BPM melódico y orgánico, y el pico, techno a 125.
     - El cierre trae primero Dance / Pop, porque los DJs de DJSets lo ponen 1,8 veces más al final. Si a Gabriel no le
       gusta, se saca ese género del cierre.
   - **«Ignora la tonalidad» no saca el filtro del vivo** (valor por defecto 3 de la semilla). El formulario ya no
     tiene «tonalidad libre»: la casilla tonalidad acepta «compatible», «igual» o una tonalidad dada, y esta última sí
     reemplaza a la de lo que suena.
   - **Lo que se le pide a Gabriel en la línea B:** solo publicar, un paso de un minuto. El banco del LLM se arma con la
     semilla y con los pedidos reales del registro del teléfono. [REEMPLAZADO 2026-11-13 por Gabriel: «¿por qué no
     publicas tú mismo?». El agente publicó el Worker v6 con `publicar.py`: versión `27000c58`, tabla `estado` creada
     en la D1 remota, el mismo token del ordenador. Comprobado sin tocar la cola: la página es la v6 y `/api/estado`
     pide contraseña. Ya no queda nada para Gabriel en la línea B]
   - **Idea de Gabriel: la tonalidad según lo que se quiere conseguir** (L13 del inventario, para más adelante). Un
     músico no cambia de tonalidad igual para subir que para bajar la energía: desde Mi menor, Fa menor sube y Re menor
     baja, aunque la regla de compatibilidad las trate parecido. Cómo encaja (agente): el LLM no razona teoría. Elige
     una intención de una lista cerrada (subir energía, bajar, mantener, más luminoso, más oscuro), y el ordenador la
     traduce con una tabla de pasos de la rueda Camelot. La tabla se mide en los 36 327 pares consecutivos de DJSets y
     se contrasta con la teoría: quinta arriba, tono y semitono arriba, relativo mayor.

11. **Resultado de la escucha E12 corta: 5 a 5, no decide** (Gabriel escuchó el 11-13; la clave se abrió después).
    De 10 temas principales eligió la lista nueva en 5 y la de hoy en 5, y ninguna vez «no las distingo». Con la regla
    fijada, 9 o 10 de 10 deciden: de oído, en los primeros puestos, la ecuación de todos los factores con pesos iguales
    no se distingue de CLAP solo. **Parecidos sigue con CLAP por defecto.** No se pide otra escucha de estas dos
    listas (agente): la próxima es para lo que salga de la línea A (A2 o A3), contra la lista de hoy y con el mismo
    formato corto. Informe: `docs/reports/escucha_corta_e12_2026-11-13.md`.

12. **Primera prueba de Gabriel en el teléfono, y las reglas antes del LLM.**
   - **La prueba** (Gabriel, 11-13): sonaba «Samba» de Paul Ritch, eligió «Apertura» y escribió «Algo parecido a lo
     actual pero para apertura». El LLM lo tradujo como «Música house moderna y energética para comenzar un set». Para
     Gabriel, los primeros temas encajan con esa frase, pero la frase no es lo que pidió: Samba es un techno más oscuro
     y menos melódico. Preguntó cómo corregirlo de manera sistemática.
   - **Diagnóstico** (agente, en el registro del ordenador): la referencia quedó en «ninguna». En la v6 todo el texto
     iba al LLM, que describe y no sabe qué suena. El ordenador sí tenía la sesión: sacó de la lista el tema ya tocado.
   - **Arreglo** (agente): el intérprete por reglas corre antes del LLM (`cloudflare/reglas.js`, con los datos en
     `config/telefono/reglamento.json`). Lee con certeza «lo que suena» («parecido a lo actual»), el BPM, el BPM
     relativo, los géneros (también mal escritos), la voz, la tonalidad, el momento, el año, el sello, un tema nombrado,
     los pesos y el orden. Ese pedido ahora es Parecidos de lo que suena, ordenado para la apertura, sin LLM. Lo que las
     reglas no leen va al LLM junto con los filtros que ya pusieron. Un pedido ambiguo, vago o de algo que no hay se
     contesta sin armar lista.
   - **Medido en el banco**, con el umbral fijado antes de medir: 165 pedidos, y el de Gabriel entró como pedido real.
     Las reglas tienen precisión 1,0 en cada casilla que llenan, no inventan ningún género, y en 122 de los 165 pedidos
     no hace falta el LLM.
   - **Pendiente:** el LLM en dos pasos (L7) para lo que queda: artistas, sellos, ocasiones como «para una boda». Se mide
     con el banco completo.
13. **Las sesiones se coordinan solas para subir a GitHub** (Gabriel: son cuatro ventanas a la vez y todas tenían
    problemas para subir; quiere que se autogestionen, sin enterarse, sin choques y sin perder trabajo). Qué pasaba
    (agente): las cuatro comparten un solo árbol, un solo índice de Git y la rama `main`, así que un commit podía
    llevarse lo que otra dejó preparado, y cada una separaba a mano sus bloques de los documentos compartidos. Cómo
    (agente): un único camino, `python tools/repo/subir.py -m "..." <rutas>`. Espera su turno con un candado, hace el
    commit solo de las rutas nombradas con un índice propio, corre el chequeo pre-commit, sube y suelta el candado.
    El hook rechaza un `git commit` directo y manda a la herramienta. Los documentos compartidos se suben enteros
    apenas se editan, y ya no se separan bloques a mano. Reglas en `AGENTS.md`, «Several sessions on one working
    tree»; tests en `tests/test_subir.py`. No se usan árboles separados por sesión (`git worktree`) porque el código
    busca los datos dentro de la carpeta del repositorio y un árbol aparte no los tiene.

## 2026-11-10 (E16: la ecuación con D5 y la escucha E12 lista)

Informe: `docs/reports/juez_d5_2026-11-10.md`. Regla fijada en `77d4434` antes de medir.

1. **La lista nueva de la escucha usa pesos iguales** (por la regla de Gabriel del punto 8 de «mientras se baja el
   dataset»): con D3 y D5 × biblioteca (270 pares que pasan los filtros, de 42 DJs de D3 y 62 de D5, cada DJ fuera) la ecuación
   aprendida no se distingue de los pesos iguales (+0,007 [−0,008, +0,022]).
2. **V8 pasa entero** (D3 y E16): la lista nueva usa CLAP sin la dirección del MP3 (v8c). Los valores por defecto del
   recomendador no cambian hasta que Gabriel lo diga.
3. **La escucha E12 está lista** en `D:\TRAKTOR ML datos\dj_sets\e12\listas\` (10 temas principales, A y B a ciegas).
   GABRIEL: por cada tema, A, B o «no las distingo»; `e12\clave.json` no se abre antes. Decide 9 o 10 de 10.
4. **Dato para Gabriel, sin propuesta** (agente): dentro del filtro de compatibilidad la tonalidad no ordena (peso
   0,01 [−0,19, 0,18]); con la tonalidad de Beatport, la «paralela» de los DJs es casi siempre la «misma» de Traktor.
5. **V7 corre en gpuq** con el OK de Gabriel (trabajo corto `traktor-v7-prueba-1` igual a E15 y a la CPU; completos
   `traktor-v7-1` en Colab, `lote_04` [REEMPLAZADO 2026-11-10 por `traktor-v7-4`: Colab cortó la sesión a los 12-13 min
   antes de terminar una tanda; el trabajo guarda ahora un parcial cada 96 previews y la interrupción a propósito siguió
   desde él], y `traktor-v7-2` en Kaggle, el resto). Al terminar: pesos con los pares de
   previews y cuántos de los 30 cambian; otra escucha solo si cambian.

## 2026-11-10 (G13 y P12a: los DJs de D5, solo texto)

Informe: `docs/reports/pares_texto_2026-11-10.md`. 1549 sets, 13 559 pares consecutivos con metadatos de Beatport.

1. **La taxonomía de géneros ordena el estilo de la noche, no la transición** (agente): frente a un tema de otro set
   sube con lo que hacen los DJs; dentro del set no distingue el tema siguiente de uno lejano.
2. **La coaparición de géneros aprendida de los sets le gana a la taxonomía** con los DJs de prueba fuera (AUC 0,72
   frente a 0,68). **Pregunta para Gabriel**: si el factor `genero` pasa a ser esa coaparición. Hasta que conteste,
   E16 y la escucha usan la taxonomía.
3. **Tonalidad de Beatport** (sin peso, por decisión de Gabriel): los DJs repiten la tonalidad (2,0 veces el azar) y
   la tónica (1,8); la paralela 1,7, la vecina 1,3, la relativa 0,9. La regla de compatibilidad no se toca.

## 2026-11-10 (E17 y V8: el códec en CLAP)

Informe: `docs/reports/codec_clap_2026-11-10.md`. Protocolo y umbrales fijados en `f863a9e` antes de medir.

1. **El corrimiento de CLAP entre un preview y su archivo es de la compresión MP3** (agente, E17): el mismo audio de
   20 archivos de 320 kbps recodificado a 96 kbps se corre sobre la dirección de E15 tanto como el preview (+3,20
   frente a +2,53 desvíos), con dosis y respuesta (192 kbps +0,43; 128 +2,32), y su dirección tiene coseno 0,991
   con la de E15. Con 6 dB menos, CLAP también cambia, pero en otra dirección.
2. **V8 (CLAP sin esa dirección, estimada sin previews) pasa los umbrales fijados con D3** (agente): previews
   primero en el 96,6 % sin centrarlos aparte; el agrupamiento por bitrate baja por debajo del de MAEST; en D3 no
   pierde AUC. Es provisorio hasta medirlo con E16. Si pasa, v8c es el CLAP de la lista nueva de E12; los valores
   por defecto del recomendador no cambian hasta que Gabriel lo diga.
3. **E15, un metadato por vez** (agente, arreglo pedido por la auditoría): la tonalidad de Beatport baja el puntaje
   de todos los factores de 0,947 a 0,815 y el BPM a 0,909; la tónica coincide en el 73 % y el modo en el 64 %.

## 2026-11-10 (después de E15: respuestas de Gabriel)

Gabriel leyó el informe de E15 y la auditoría (`docs/reports/auditoria_e15_2026-11-10.md`, commit `9c28acf`).

1. **V7 sigue, con las tres vistas (CLAP incluido) y la ventana V4, guardando cada segmento.** Antes del primer
   envío a gpuq: prueba chica en CPU y pedido de OK a Gabriel con los números (`AGENTS.md`, «Compute»). «CLAP
   centrado aparte» no es la solución: la dan E17 y V8.
2. **Tonalidad.** El peso que va al recomendador en vivo sale solo de pares con la tonalidad de Traktor (D3 y
   D5 × biblioteca, E16). Con las tonalidades de Beatport, P12a no aprende un peso para la regla actual: informa
   cuántas veces sobre el azar aparece cada relación (P7) y lo mismo mirando solo la tónica. En los pares de E16, la
   misma tabla con la tonalidad de Traktor y con la de Beatport (`bp_key` de `beatport.parquet`), que es la única
   variable que cambia. **La regla de compatibilidad es de Gabriel**: el agente no propone cambiarla.
3. **G13, P12a y E16 empiezan ya.**
4. **Unidad nueva, corta y en CPU, antes de aprender pesos con CLAP: E17 y V8** (plan 12.9). E17: 20 archivos de la
   biblioteca de 300 kbps o más, al azar con semilla fija, recodificados a 192, 128 y 96 kbps y aparte con 6 dB menos
   sin recodificar; dice si la causa es la compresión o el nivel. V8: CLAP sin la dirección de E17, una dirección y
   más solo si no llega, con los umbrales de la fila V8 fijados en un commit antes de medir. Si V8 pasa, es el CLAP de
   la lista nueva de E12. MAEST y estilos, lo mismo, solo informado. Los valores por defecto del recomendador no
   cambian hasta que Gabriel lo diga.
5. **Orden**: E17 y V8; en paralelo, la prueba chica de V7 y su pedido de OK; G13 y P12a; E16 y la ecuación ajustada
   de nuevo; E12.
6. **Dos arreglos al informe de E15** cuando se vuelva a correr `medir`: cambiar un metadato por vez (solo tonalidad,
   solo BPM) para decir cuánto baja cada uno, y separar la tonalidad en «misma tónica» y «mismo modo».
7. Siguen las reglas de antes: nada de pedidos a Beatport ni a 1001Tracklists, no se editan los archivos de la sesión
   de la bajada, nada se ajusta con los sets de Gabriel y cada informe dice con cuántos sets y previews midió.

## 2026-11-10 (E15: los previews frente a la biblioteca)

Informe: `docs/reports/dominio_previews_2026-11-10.md`. Protocolo fijado en el commit `269af4c` antes de medir.

1. **Los previews no pasan E15 con los umbrales fijados de antemano** (agente): el Spearman de todos los factores da
   0,78 (umbral 0,90), el códec de CLAP 0,92 (umbral 0,97) y CLAP pone primero al tema verdadero en el 32 % (umbral
   90 %). Por el protocolo, V7 no arranca hasta que Gabriel decida.
2. **Por partes** (agente): MAEST pasa todo (primero en el 100 % de 146, códec 0,988); los estilos casi no se mueven
   con el códec (0,993); CLAP se corre como bloque con el preview de 96 kbps y, centrado aparte, queda primero en el
   92 %. Lo que baja el puntaje de todos los factores es la tonalidad: la de Beatport coincide con la de Traktor o
   los tags en el 47 % de los temas; con los metadatos de la biblioteca, ese puntaje da 0,947.
3. **La ventana de los previews es el preview entero en 4 segmentos de 30 s** (V4, por la regla fijada); la regla
   del extractor tal cual es mucho peor y no se usa.
4. [REEMPLAZADO 2026-11-10 por las respuestas de Gabriel (sección «después de E15», puntos 1 y 2): V7 sigue con
   las tres vistas; el peso de la tonalidad para el vivo sale de pares con la tonalidad de Traktor] **Pregunta para Gabriel** (agente): si V7 sigue con MAEST y estilos como vistas del juez y CLAP solo entre
   previews y centrado aparte, y cómo se trata la tonalidad de los pares de previews, que en vivo es la de Traktor.
5. **Auditoría de E15** (agente auditor; `docs/reports/auditoria_e15_2026-11-10.md`): los números se reproducen. El
   umbral del Spearman era de la auditoría y medía también la fuente de los metadatos: el propio archivo de la
   biblioteca con los metadatos de Beatport da 0,81. CLAP registra la calidad del archivo: 420 de los 1760 temas son
   MP3 de menos de 200 kbps y sus vecinos en CLAP se agrupan por bitrate. La tonalidad de Beatport difiere de la de
   Traktor sobre todo en el modo. **Recomendaciones** [REEMPLAZADO 2026-11-10: Gabriel las aprobó (sección «después de E15»)]: V7 sigue con las tres vistas; antes
   de aprender pesos con CLAP, E17 y V8 (plan 12.9); el peso de la tonalidad para el vivo sale de pares con la
   tonalidad de Traktor; G13, P12a y E16 empiezan ya.

## 2026-11-10 (recomendación: mientras se baja el dataset)

Plan, sección 12.9. Las medidas de la auditoría de este día son exploratorias (scripts fuera de Git).

1. **El implementador sigue mientras bajan D5 y D6, con lo que no necesita el dataset completo.** Gabriel lo había
   pausado hasta tenerlo entero, le pidió opinión a la auditoría y después pidió el mensaje para esta etapa. [REEMPLAZADO
   2026-11-10 por el punto 8: los pesos de la escucha se aprenden con los pares de E16 y los definitivos con los
   de previews; el valor por defecto del recomendador sigue esperando a que Gabriel lo diga] Fijar
   los pesos de la ecuación y cambiar valores por defecto espera al dataset completo. Cómo (agente): el orden es
   E15, V7, G13 con P12a, E16, y después la escucha E12 y las unidades 3 y 4.
2. **Se verifica si el embedding de un preview cae cerca del embedding del tema completo** (Gabriel: «agrega esa
   verificación»). Cómo (agente): E15, lo primero de la etapa, con umbrales escritos antes de medir. 166 temas de
   la biblioteca ya tienen su preview bajado (el mismo id de Beatport), así que no hace falta bajar nada. El
   preview dura 120 s y empieza en el 40 % del tema; la biblioteca usa 3 segmentos de 30 s del tramo 35-65 %. El
   extractor tal cual usaría solo 36 s del preview: la ventana se decide con E15.
3. **La taxonomía de géneros se valida con los sets bajados y el peso del factor se ajusta según salgan los
   números** (Gabriel). Cómo (agente): G13 y P12a, solo texto, con el género, el BPM y la tonalidad que ya traen
   los manifiestos de los previews. El peso que salga es el que se propone para la ecuación; el valor por defecto
   del recomendador sigue cambiando solo cuando Gabriel lo diga, después de escuchar.
4. **La auditoría no recomienda invertir ahora en el clasificador de género** (agente; plan, filas G1 y G3). En
   75 pares de D3 × biblioteca, la taxonomía separa «mismo set» de azar del mismo BPM mejor que «mismo género o
   no» (AUC 0,69 frente a 0,60) y peor que los estilos Discogs sacados del audio (0,75), que cubren los 1760 temas
   sin depender de la etiqueta. En los sets de Gabriel, sumarla a CLAP no cambia nada distinguible.
5. **La pregunta del implementador sobre `ai-train=no` ya estaba contestada** (Gabriel, 2026-11-09, sets de
   1001Tracklists, punto 5: «Sí, bajarlos igual», y el dataset es para evaluar y entrenar). Lectura del agente:
   cubre aprender los pesos de BPM, tonalidad y género con esos datos, que ya están bajados; se lo dijo a Gabriel
   el 11-10 para que lo corrija si no es así. Gabriel lo confirmó el mismo día: «bajamos toda la info disponible
   desde Beatport».
6. **Los 10 temas principales de la escucha E12 salen al azar** (Gabriel, sobre la propuesta de la auditoría).
   Cómo (agente): entre los temas que tocó en sus sets, repartidos por estilo y BPM, con semilla fija.
7. **Gabriel no etiqueta nada por ahora: consume mucho tiempo** (Gabriel). En la escucha solo dice qué lista
   prefiere. Con 10 preferencias decide un resultado de 9 o 10 de 10; 8 de 10 es un indicio (agente).
8. **La lista nueva usa todos los factores, y el peso relativo de cada uno se decide con datos** (Gabriel). Cómo
   (agente): para la escucha, los pesos se aprenden con los pares de E16 (D3 y D5 × biblioteca), con cada DJ
   fuera del entrenamiento; si los datos no los distinguen de los pesos iguales, se usan iguales y se dice así.
   Se vuelven a estimar con los pares de previews (V7); hay otra escucha solo si las listas cambian.
9. **Siempre habrá temas «basura» en la biblioteca, que no deberían entrar en las recomendaciones ni en el
   análisis** (Gabriel). Queda como idea P14 del plan, sin diseñar: una lista de exclusión armada al paso, sin
   etiquetado previo.

## 2026-11-13 (D5/D6 terminados)

1. **La descarga terminó** a las 05:09 (`descarga_continua.py`: FIN, copia final al Drive, `descarga_terminada.txt`).
   [REFUTADO 2026-11-13 por `build_djsets.py`, que cuenta temas distintos y no archivos: 33 321 temas con audio, 30 255
   de Beatport y 3066 de YouTube, y 3290 sin audio; 33 323 sumaba archivos, contaba dos veces 7 temas bajados de las dos
   fuentes y no contaba 5 que comparten el audio de otro tema de su tanda] De 36 611 temas distintos de los 1549 sets,
   33 323 tienen audio (91 %): 30 251 previews de Beatport (43,1 GB en
   `erclm:traktor_ml/beatport_previews/`) y 3072 audios completos de YouTube (19,0 GB en `youtube_temas/`). Los 3288
   restantes no están en Beatport ni en YouTube en la misma versión. Las listas, los manifiestos, los sets y D7 están
   copiados en `erclm:traktor_ml/dj_sets/`. YouTube a 20,2 s entre pedidos no volvió a pedir la verificación desde
   el 11-11 05:50 (unas 47 h); la sesión de Chrome de Gabriel no se pudo usar (cifrado app-bound, yt-dlp #10927).
   Un apagado normal de Windows el 11-11 a las 22:19 la paró 9 h (el guardián no sobrevive a un apagado).
2. **Dataset DJSets v1 en una sola carpeta** (Gabriel: una única ubicación fácil de usar, con el código en una rama nueva
   y un README; el formato lo eligió el agente siguiendo FMA, MTG-Jamendo, mirdata y las fichas de datasets).
   `erclm:traktor_ml/djsets_v1/` tiene:
   - las tablas CSV `sets`, `set_tracks`, `tracks`, `shards` y `split_by_dj`;
   - las 132 tandas como shards `.tar`, copiadas dentro del Drive con nombre nuevo, sin bajar el audio;
   - `raw/` con lo original y `checksums.md5`;
   - el lector `djsets.py`, sin dependencias, que lee del `.tar` o desempaqueta.
   El código y la ficha están en `datasets/djsets/`. Verificado: el md5 de cada archivo publicado coincide, y 541 audios
   reales de dos shards se leyeron y compararon por md5 contra `tracks.csv`. Las carpetas originales de la descarga
   quedan como estaban. Gabriel lo copia a su NAS y a su pendrive.

## 2026-11-11 (pendrive D: con errores)

1. **Windows marca el pendrive `D:` (FAT32) como «Full Repair Needed»** (agente, 13:30, `Get-Volume`); no se sabe
   desde cuándo. Lo de D5/D6 que solo vivía ahí (sets de 1001Tracklists, listas y manifiestos de cada corrida, el
   registro) quedó respaldado en `erclm:traktor_ml/respaldo_d5_d6/respaldo_d5_d6_20261111_1329.tar` (3558 archivos,
   174 MB, verificado por tamaño y MD5). [REEMPLAZADO 2026-11-11 por la revisión de Gabriel: `chkdsk D:` de solo
   lectura (13:40, con el pendrive en uso) no encontró problemas en archivos ni carpetas; no se repara y la descarga
   sigue. «Full Repair Needed» sería la marca de sucio de FAT32 mientras está montado y en uso (inferencia, no
   comprobada); se borraría al expulsarlo] La reparación (`chkdsk D: /f`) pide administrador: la decide y la corre
   Gabriel, con la descarga parada (`PARAR`); después se relanza y retoma.
2. **Nada del dataset solo en el pendrive** (Gabriel: «obviamente que los datos del dataset no pueden estar
   únicamente en el pendrive; si hiciste eso, estuvo mal»). Error del agente: las tandas de Beatport y YouTube iban al
   Drive, pero los 1549 sets de 1001Tracklists, las listas de cada corrida y D7 (518 compilados de YouTube, CLAP,
   MAEST, MusicBrainz) vivían solo en `D:`. Ahora `descarga_continua.py` copia `D:\TRAKTOR ML datos\dj_sets` entero a
   `erclm:traktor_ml/dj_sets/` al arrancar y cada 2 h (`rclone copy`, nunca borra allá), menos la caché de Beatport y
   lo temporal.
3. **YouTube desde Colab o Kaggle: no** (Gabriel: si YouTube va a bloquear, seguir como está). Las IP de los
   servidores de Google suelen recibir la verificación anti-robots enseguida; Beatport ya rechazó a Kaggle.
4. **`correr` murió a las 13:28 sin dejar rastro** y la descarga quedó parada 19 min (agente). Coincide con señales de
   que D: dejó de responder un momento (inferencia, no comprobada). Ahora un `guardian` (cada 5 min, registro en C:)
   la relanza si se murió sin terminar, salvo que exista `PARAR`; escribir el registro o mirar el disco ya no pueden
   tumbarla.

## 2026-11-10 (sets y audio de los temas, de día)

1. **Gabriel autoriza** cambiar el filtro de la búsqueda en Beatport, con la condición de que el agente supervise
   exhaustivamente que encuentre la versión correcta; bajar de YouTube los audios que falten; probar de nuevo
   1001Tracklists (y si no entra, darle otra cuenta). El PC queda encendido.
2. **Filtro de Beatport** (agente, `48485bf`): `match(soft_mix=True)` acepta la mezcla escrita de otra forma
   (posesivo, «X Mix» por «X Remix»), solo si ninguna consulta trae la exacta, y lo marca (`mezcla_suave`); más
   palabras de versión (Remake, Reshape, Interpretation…); «X aka Y» después del nombre como viene. Verificación:
   repetidos sin red los 11 139 temas de los lotes, el código viejo reproduce lo grabado y el nuevo no cambia
   ninguno de los que ya tenían preview; 185 rescatados, leídos uno por uno por el agente y por un revisor aparte
   sin su juicio: 185 la misma versión. En el piloto, contra el enlace de 1001Tracklists, uno más y ninguno menos.
   La biblioteca no lo usa: sus 1602 temas repetidos no cambian. [REEMPLAZADO 2026-11-10 por el control hecho a las
   17:30 (`docs/reports/control_filtro_suave_2026-11-10.md`): de 158 temas flexibles con enlace a Beatport, 127 con el
   mismo id; de los 31 distintos, ninguno es otra versión (14 la misma grabación en otro lanzamiento, 10 enlaces sin
   ficha en Beatport, 4 donde la búsqueda acierta igual o mejor, 1 sin decidir), y 2 son un corte corto de la misma
   mezcla] El enlace de 1001Tracklists de los rescatados con botón queda como control cuando el sitio deje.
3. **YouTube** (agente, `youtube_temas.py`): audio completo en m4a sin recodificar, tandas de 200 en
   `erclm:traktor_ml/youtube_temas/<corrida>/`; prefiere el corte largo, el canal «- Topic» y el audio al videoclip.
   En 100 temas que Beatport sí identificó, eligió 89: 73 con la misma duración (±3 s) y 16 la misma producción en
   otro corte. YouTube pidió la verificación anti-robots tras unos 200 temas a 2 s: no se esquiva; va a 6 s y espera
   1 h si vuelve. Usar la sesión del navegador de Gabriel (`--cookies-from-browser`) lo evitaría, pero es su cuenta:
   no se usa sin su OK.
4. **1001Tracklists**: a las 08:10 entraron la página del 403 y otra: el bloqueo de la noche era de todo el sitio y
   pasó en unas 5 h. A las 08:40 cortó la conexión tras 150 páginas; a las 10:40 dejó entrar y bajaron 200 seguidas.
   Todos los sets de trayectoria (3 h o más) están bajados; quedan 358 de pares. El agente no usa otra cuenta para
   seguir pasado el límite del sitio (sería saltear su control y arriesga las dos cuentas): espera y sigue.
5. **El pendrive `D:` es FAT32**: la caché plana de búsquedas de Beatport se llenó a los 13 106 archivos y cortó la
   bajada de `lote_04` (`PermissionError`). Va en subcarpetas desde ese día (`docs/LESSONS_LEARNED.md`). Error del
   agente en el mismo rato: su script de verificación pidió a Beatport unas mil búsquedas que no hacían falta
   (08:29-08:58); Beatport no cortó.
6. **La descarga no se detiene en seco** (Gabriel, 13:50: «si debemos esperar más entre descarga y descarga para que
   no se bloquee el control anti robots, entonces esperamos», pero sin parar días). El agente encadena todo en
   `tools/dj_sets/descarga_continua.py`: tres carriles a la vez (1001Tracklists, Beatport, YouTube) que eligen el paso
   siguiente por lo que hay en disco (rescate, buscar, enlaces, bajar, lote nuevo, corrida de YouTube) y nunca se
   rinden: tras un corte esperan más y vuelven a probar; cada verificación anti-robots de YouTube alarga la pausa
   entre pedidos (×1,5, tope 30 s). Se para con el archivo `PARAR` en la carpeta de datos. Motivos: a las 13:00 un
   corte de conexión de Beatport tumbó `lote_04` con una excepción (ahora se reintenta), los bucles de la mañana se
   rendían tras 3 o 4 intentos, los rescates y YouTube esperaban a que el agente los lanzara, y una tanda de YouTube
   cortada volvía a buscar desde el primer tema (ahora sigue donde quedó: cada pedido repetido acerca el bloqueo).
   A las 16:30, dos ajustes: Beatport baja de a una tanda y entre tanda y tanda atiende los rescates (el de
   `rescate_01` esperaba a que terminaran las 34 de `lote_04`), y YouTube toma enseguida los temas sin botón de compra
   que un lote no resolvió (1442 en las 22 tandas hechas de `lote_04`), sin esperar al rescate ni a los enlaces.
   YouTube pidió la verificación anti-robots a las 16:28, tras unas 3 h 30 min a 6 s, y de nuevo a las 17:30, un minuto
   después de la hora de espera: la pausa sube solo si la corrida bajó al menos 10 min antes del bloqueo (si no, es el
   mismo bloqueo y solo se espera más), y las esperas quedan en disco para que un reinicio no las acorte.

## 2026-11-09 (unidad 1: fusión de vistas y ecuación de Parecidos)

Unidad 1 de `docs/plans/PROMPT_FUSION_Y_CONCEPTOS.md`. Informe: `docs/reports/fusion_vistas_2026-11-09.md`.

1. **El Drive tiene alojamiento infinito y gpuq iba a quedar operativo desde otra ventana** (Gabriel, al pedir la
   unidad). Cómo (agente): el tope de unos 10 GB de disco nuevo era por el disco de este escritorio; el espacio del
   Drive no limita. gpuq quedó probado de punta a punta el mismo día (`docs/GPUQ.md`).
2. **El MAEST de los 179 originales de Bedrock se calculó en local** (agente): 11 s por tema, unos 30 minutos con
   prioridad baja, debajo del umbral de pesado. `tools/dj_sets/maest_originales.py`, con el modelo y la segmentación
   de la biblioteca (al recalcular 4 temas de la biblioteca, coseno 1,0000).
3. **Lo que dejó la unidad 1** (agente; exploratorio, sin escucha): los números de la auditoría se reproducen;
   Bedrock no confirma la fusión ni la tumba (las dos medidas posibles dicen cosas opuestas); la ecuación aprendida
   no supera a los pesos iguales, porque el juez son 63 pares de D3 y un solo DJ con originales.
4. **Para la escucha E12 se propone todos los factores con pesos iguales frente a la lista de hoy** (agente;
   decide Gabriel escuchando). Son CLAP centrado, MAEST centrado y estilos en z, más BPM, tonalidad y género con las
   fórmulas del recomendador, cada uno en desvíos de los pares al azar dentro de los filtros en vivo. No sale de
   ajustar pesos con estos datos. Los pesos por defecto del recomendador siguen en solo CLAP.
5. **Cómo se mide la ecuación** (agente): pares de un mismo set que pasan los filtros en vivo frente a 20 candidatos
   al azar por par que también los pasan; cada set pesa 1; cada DJ se predice con un modelo entrenado sin él; un
   factor desconocido no cuenta (vale la media de los pares al azar); Bedrock no aporta a tonalidad ni a género
   porque sus originales no los tienen.

## 2026-11-09 (algoritmo de recomendación)

1. [REFUTADO 2026-11-09 por la aclaración de Gabriel del mismo día (punto 8): el agente leyó mal. BPM y tonalidad sí
   entran al puntaje final, como factores propios de la ecuación; lo que no hacen es usarse dentro de CLAP y de los
   demás modelos] **El parecido de la playlist de D0 no cuenta BPM ni tonalidad** (Gabriel: «esos dos parámetros se pueden ajustar a
   posteriori»). La playlist son temas parecidos al que suena y que con seguridad van a sonar bien mezclados con él.
   - Cómo (agente auditor): BPM y tonalidad no entran al puntaje de parecido y toda medida de parecido lleva control de
     tempo. [REEMPLAZADO 2026-11-09 por el punto 6: Gabriel contestó que los filtros siguen como están] Los filtros en
     vivo (BPM ±20 y tonalidad a ±2 semitonos; 2026-11-06, punto 7) quedan como están hasta que Gabriel diga otra cosa:
     es una pregunta abierta para él.
2. **Las otras playlists redirigen el set hacia un concepto escrito en lenguaje natural; cómo se eligen esos temas es
   una pregunta abierta a optimizar** (Gabriel), con el paper «Guided Music Retrieval» (Brandão et al., enviado a IEEE
   TASLP en 2026) como inspiración.
3. **La sesión auditora propone la estrategia y entrega el prompt para un agente implementador** (Gabriel lo pidió).
   - [REEMPLAZADO 2026-11-09 por el punto 4: aprobada, con los dos casos de uso del punto 5] Propuesta (agente,
     pendiente de que Gabriel la apruebe): Parecidos como zona segura con varias vistas (CLAP, MAEST
     capa 7 y estilos Discogs sumados; probabilidad de «van juntos» calibrada con sets profesionales), y cada concepto
     como un paso dentro de esa zona, con el puntaje de concepto elegido en un banco local construido como el del paper.
   - Informe con lo medido (exploratorio, sin escucha): `docs/reports/auditoria_recomendacion_2026-11-09.md`. Prompt:
     `docs/plans/PROMPT_FUSION_Y_CONCEPTOS.md`.
   - La candidata a energía del punto 2 de «conceptos desde el teléfono» (Energy de Mixed In Key) choca con el punto 3
     del 2026-11-07: Gabriel la descartó. La coordenada sale de T19 y T20 del plan, con E11 de juez.
4. **Gabriel aprueba la estrategia** («Por lo demás, apruebo tu estrategia»), con la precisión del punto 5. El detalle
   queda en la sección 12 de `docs/plans/recomendacion_playlists_plan.md`.
5. **Las playlists de concepto tienen dos casos de uso** (Gabriel).
   - **Con el set en curso**: hay un historial de lo que se viene tocando, y conviene usarlo como fuente de información
     para la trayectoria del tema siguiente. Si se filtra por concepto, primero se eligen los temas parecidos que van
     bien con el principal y sobre esa lista se filtra por el concepto.
   - **Antes de empezar a tocar**: no hay tema de referencia. El DJ pide una playlist con «vibras africanas» o «música
     de muertos», y bajo esa etiqueta se agrupan los temas que transmiten ese concepto.
   - Si empieza por el segundo caso, elige un tema de esa lista y la iteración siguiente ya es el primero: hay tema
     principal y los que suenen a continuación tienen que ser parecidos.
   - Cómo (agente): el caso lo decide el estado. Sin tema principal, la lista se ordena solo por el puntaje del concepto
     sobre toda la biblioteca, con un piso de confianza y largo variable. Con tema principal, zona de Parecidos (con el
     historial del set) y, dentro, orden por el concepto. El banco de conceptos mide los dos modos.
6. **Los filtros en vivo siguen como están** (Gabriel): BPM ±20 y tonalidad a ±2 semitonos no se quitan.
7. **Todo lo pesado en procesamiento o en disco se procesa con las GPU de gpuq y con el Google Drive de Gabriel**
   (Gabriel), el de su cuenta de trabajo. La dirección no se escribe acá: el repositorio es público.
   - Qué es (agente, leído de github.com/gbibbo/gpuq el 2026-11-09): una cola de trabajos sobre los niveles gratuitos de
     Colab (una T4, un trabajo por vez) y Kaggle (unas 30 horas de GPU por semana, sesiones de hasta 12 horas). Un
     ordenador siempre encendido los lanza y los relanza; el código se toma del repositorio en la rama o commit pedido;
     los datos y resultados van por `rclone` a un remoto, por defecto `gdrive:gpuq`; los trabajos tienen que poder
     retomarse desde un punto de control.
   - Cómo (agente): lo liviano sigue en local (tests, bancos sobre embeddings ya calculados, texto de CLAP, extracciones
     de pocos temas). Va a gpuq lo que pase de alrededor de una hora de CPU en este escritorio o de unos 10 GB de disco
     nuevo; el umbral es del agente y Gabriel lo corrige. Antes del primer envío de cada tipo de trabajo: prueba chica
     en CPU, qué necesita la máquina remota, horas de GPU estimadas frente a la cuota, espacio en Drive y el OK de
     Gabriel en el chat. Las credenciales las pone Gabriel con `gpuq setup`; no pasan por el agente. Nada de lo subido
     es público: ni datasets ni notebooks públicos en Kaggle, ni enlaces compartidos de Drive.
   - [REEMPLAZADO 2026-11-09 por la puesta en marcha de gpuq: el despachador corre en la Raspberry `asr-rp5`, la cola es `gbibbo/gpuq-personal`, el Drive es el
     remoto `erclm` con los datos de TRAKTOR en `erclm:traktor_ml/`, y este escritorio tiene el clon de la cola y
     rclone; probado de punta a punta en una T4, docs/GPUQ.md] Estado (agente, 2026-11-09): en este escritorio no encontré `gpuq` ni `rclone` (no están en el PATH, el Python de
     la Store no tiene el módulo y no hay una carpeta `gpuq` en `C:\VS projects`, en la carpeta del usuario, en
     Documentos ni en `D:\`). No miré otros ordenadores ni dónde corre el despachador.
   - Reemplaza al punto 4 del 2026-09-12 (GPU solo por trabajos gratuitos de Kaggle).
8. **El puntaje de Parecidos es una ecuación que pondera varios factores** (Gabriel, aclarando el punto 1: «quizás fui
   confuso»). En el listado de parecidos influyen modelos como CLAP y los demás, el género, la tonalidad, el BPM y
   otros; una ecuación pondera el peso relativo de cada variable y da el puntaje final con el que se ordenan las
   sugerencias. BPM y tonalidad se usan después, como factores propios: no se usan dentro de CLAP y de los demás
   modelos.
   - Cómo (agente): es la arquitectura que ya tiene el recomendador (componentes `clap`, `bpm`, `clave`, `genero` y
     `tiv`, cada uno con su peso; 2026-11-06, punto 9). Las vistas nuevas (MAEST capa 7, estilos Discogs) son factores
     nuevos de esa ecuación.
   - Los modelos se miden con el tempo controlado, para ver qué aportan aparte del BPM: los estilos Discogs llevan
     tempo (−0,37 con la diferencia de BPM) y contarían dos veces lo que ya cuenta el factor BPM.
   - Los pesos de todos los factores se fijan juntos, con pares de sets profesionales fuera de muestra y con la escucha
     de Gabriel (plan, P12). Hoy el peso por defecto es solo CLAP (2026-11-06) y no cambia hasta que él lo diga.

## 2026-11-09 (conceptos desde el teléfono)

Gabriel armó el Worker de Cloudflare en otra sesión (LLM solo en Cloudflare; audio, embeddings y rutas solo en el
ordenador) y pidió completarlo con la mayor autonomía posible. Lo que sigue lo decidió el agente y Gabriel puede
cambiarlo. Evidencia: `docs/reports/traktor_concept_ai_cloudflare_2026-11-09.md`, secciones 8 y 9.

1. [REEMPLAZADO 2026-11-09 por el punto 6: sin filtros; menos voz se pide con «without vocals» en el texto, que sí la baja (medido en `clap_texto_2026-11-09.md`)] **Las frases negativas no se restan en CLAP** (agente; exploratorio, solo voz, el único atributo con etiqueta). Restar
   una negativa de voz en 10 conceptos sobre los 1760 temas dejó la voz en los 30 primeros casi igual (0,51 sin restar;
   0,46 a 0,49 con peso 0,25 a 1) y cambió hasta 19 de 30 temas. La voz se filtra con la etiqueta Vocal; las demás
   exclusiones se informan en el teléfono como sin aplicar.
2. [REEMPLAZADO 2026-11-09 por el punto 6: el circuito no tiene condiciones; la candidata de energía la descartó Gabriel (algoritmo, punto 3)] **La energía relativa no se aplica todavía** (agente): no hay coordenada validada (plan, sección 11), y el teléfono lo
   dice. Candidata que decide Gabriel: el Energy de Mixed In Key de los tags (1408 de 1770 temas, valores 2 a 8).
3. [REEMPLAZADO 2026-11-09 por el punto 6: en el circuito el BPM no filtra; los filtros en vivo de Parecidos siguen (algoritmo, punto 6)] **BPM cerca del actual es ±20, la regla de Parecidos; sin esa condición el BPM no filtra** (agente: se respeta lo que
   pide el concepto). El BPM se muestra en la lista.
4. [REEMPLAZADO 2026-11-09 por el punto 9: una playlist nueva por concepto] **Cada lista se escribe también en `TRAKTOR ML Concepto NML.nml`, en la carpeta de sugerencias** (agente): es una
   playlist más, como las de dirección (2026-11-07, punto 8). El botón D#0+D0 no cambia. Sin probar dentro de Traktor.
5. **CLAP se carga con el primer concepto y se libera a los 20 min sin uso** (agente): ocupa cerca de 950 MB y este PC
   tenía 2 GB libres. Prioridad baja y dos hilos, para no competir con el audio.
6. [REEMPLAZADO 2026-11-13 por Gabriel (sección «plan v2», puntos 2 y 3): el teléfono lleva presets y filtros por BPM, género y otros campos, y el LLM llena un formulario que el ordenador ejecuta; sigue en pie que las instrucciones tienen que ser claras porque el LLM razona poco] **El trabajo de esta etapa es la arquitectura, no los filtros** (Gabriel). El punto de partida es un concepto escrito en
   el teléfono y el resultado, una playlist nueva en Traktor. Cómo se eligen los temas no entra en esta etapa (es el punto
   5 de «algoritmo de recomendación»). Sí entra la parte en que el LLM de Cloudflare encuentra la mejor forma de definir
   el concepto: una persona escribe mal, y dos textos que para nosotros dicen lo mismo dan listas muy distintas en CLAP.
   El LLM se ocupa de todo lo necesario, puede lanzar varias búsquedas y decidir con lo que devuelven. Las instrucciones
   tienen que ser claras, porque el LLM gratuito razona poco.
7. **El circuito** (agente). El Worker piensa (LLM, decisiones, cola) y el ordenador oye: busca y describe lo que
   encontró, sin puntajes. Interpretar → tres textos → sondeo → elegir, o reescribir una vez como máximo → lista final.
   Si el LLM no responde, sigue con todos los textos. `buscar()` es la única pieza que decide qué temas salen.
   Detalle: `cloudflare/README.md`.
8. **Reglas para escribir los textos, medidas sobre la biblioteca** (agente; exploratorio):
   - inglés;
   - sin plantillas;
   - lo que no se oye se traduce a sonido, y el teléfono lo dice;
   - género más sonido, de 6 a 14 palabras;
   - tres redacciones, promediadas;
   - la única negación es «without vocals»;
   - no agregar sonidos que nadie pidió.

   El validador del Worker hace cumplir lo que el modelo no respeta. Con estas reglas, el texto curado ordena mejor
   que el texto de la persona en 14 de 16 pedidos (+0,078 AP contra estilos de MAEST, IC95 [0,029; 0,125]).
   Evidencia: `docs/reports/clap_texto_2026-11-09.md`.
9. **Cada concepto crea una playlist nueva, `Concepto <MMDD-HHMM> <nombre>.nml`, en la carpeta de sugerencias** (agente,
   por «una playlist nueva en Traktor»). No se pisan ni se borran solas. Sin probar dentro de Traktor.
10. **Publicar en producción lo hace Gabriel** (agente). [REEMPLAZADO 2026-11-13 por el pedido de Gabriel («¿por qué no
    publicas tú mismo?»): el agente publica cuando Gabriel se lo pide en la conversación; DECISIONS 2026-11-13,
    punto 10] El control de permisos de la sesión bloqueó el despliegue y no
    se buscó otra vía. `python cloudflare/publicar.py` publica, deja un `DEVICE_TOKEN` nuevo en Cloudflare y en el
    ordenador y activa el agente con Traktor.

## 2026-11-09 (sets de 1001Tracklists)

1. **Los sets comerciales quedan fuera de D5** (Gabriel). Comercial, en sus palabras, es «David Guetta y Martin Garrix»;
   ARTBAT, Argy, Boris Brejcha, Solomun, Tale Of Us, Charlotte de Witte, Amelie Lens y Nina Kraviz van incluidos. Rige
   (agente): afuera solo el EDM de ese tipo, por nombre, y lo que el sitio etiqueta Mainstage o similar si el DJ no es
   semilla; ante la duda, adentro; un festival no vuelve comercial a nadie. Quedan 12 sets comerciales.
   [REFUTADO 2026-11-09 por Gabriel: esa lista era criterio del agente, no el suyo] Cómo (agente): ante la duda, afuera. Se descarta por género
   (Mainstage, Dance / Electro Pop y otros), por DJ (una lista de comerciales y otra de dudosos, como ARTBAT o Charlotte
   de Witte) y por evento o escenario (Tomorrowland, Ultra, EDC y otros). Trance, Hard Dance y Drum & Bass salen por
   estilo. Las listas están en `config/dj_sets/tracklists_1001.json`; son criterio del agente y Gabriel las corrige.
2. **Los sets de 3 h o más sirven para aprender la trayectoria de la elección de música a lo largo de la noche; para saber
   qué temas van juntos alcanzan los temas consecutivos de sets más cortos** (Gabriel). Cómo (agente): cada set lleva su
   uso, trayectoria o pares, o los dos (plan, sección 6).
3. **La estrategia de selección de D5 queda aprobada** (Gabriel: «Puedes proceder»; los detalles son del agente, en el
   plan, sección 6, ideas S1 a S15).
   - Se lee la página de cada set en lugar del botón de exportar: trae lo mismo con un pedido, y además sello, enlaces y
     grabación. Herramienta: `tools/dj_sets/tracklists_1001.py`, con la sesión iniciada a mano por Gabriel; 8 s entre
     pedidos, sin reintentos, y parada en la primera respuesta anormal. El perfil queda en `~/.traktor_ml/`, fuera del repo.
   - [REEMPLAZADO 2026-11-09 por las 158 búsquedas: 1549 candidatos, más abajo] [REEMPLAZADO 2026-11-09 por el criterio de comercial de Gabriel: 831 candidatos (142 para trayectoria, 689 solo para pares, 156 DJs); afuera, 12 comerciales, 3 por estilo, 217 por el tope por DJ y 206 por datos] Resultado: 84 de 147 búsquedas, 1269 sets distintos y 740 candidatos (135 para trayectoria, 605 solo para pares, 139
     DJs). Quedaron fuera 128 comerciales, 43 por estilo, 169 por el tope de 15 por DJ y 189 por datos insuficientes o
     género no aceptado. La tabla: `1001tracklists/candidatos.csv`, con columna `veto`.
   - **El sitio cortó la conexión en la búsqueda 81 de esa tanda** (`ERR_CONNECTION_RESET`, a unos 11 s por pedido; el
     resto de internet funcionaba). Puede ser un rechazo o un corte casual; una sola vez no lo distingue. No se reintentó
     y el piloto de 30 sets no se corrió. `buscar` quedó con un tope de 40 por corrida. [REEMPLAZADO 2026-11-09 por las búsquedas completas, punto siguiente: no falta ninguna] [REEMPLAZADO 2026-11-09 al sumar a los DJs que nombró Gabriel y a tres de su Spotify que se habían sacado: faltan 74, de 158: 45 DJs, 15 términos y 14 eventos] Faltan 63 búsquedas: 34 DJs, 15
     términos y 14 eventos.
   - Qué dicen los datos guardados del corte (agente; sin volver a tocar el sitio): el ritmo fue parejo (12 s) y sin
     lentitud antes del corte; la sesión siguió abierta y no hay avisos en la última página; el corte fue de conexión, no
     una respuesta de límite ni un captcha. No distingue un límite de un corte casual: es HIPÓTESIS.
   - **Búsquedas completas y piloto** (Gabriel pidió seguir y quitar la pausa entre tandas). La página principal, 17 min
     después del corte, respondió HTTP 200 con la sesión abierta. Después corrieron 40 y 34 búsquedas, con 7 min entre las
     dos, y las 30 páginas del piloto, sin ningún corte. Tabla final: 2540 sets distintos y 1549 candidatos (385 para
     trayectoria, 1164 solo para pares, 354 DJs, 1489 con grabación). Afuera: 89 comerciales (44 por nombre, casi todos
     del programa de radio de Armin van Buuren; 45 por etiqueta, de DJs que no son semillas), 108 por estilo, 436 por el
     tope de 15 por DJ y 358 por datos.
   - Piloto: 30 sets de arriba de la tabla, uno por DJ (no es una muestra al azar). 1206 temas y 1092 pares consecutivos
     identificados; mediana de 100 % identificado; 19 de 30 con tiempos en el 80 % de los temas o más; 29 con grabación;
     enlace a Beatport en una mediana del 80 % de los identificados. **En la biblioteca: 11 de 1206 temas (0,9 %), ningún
     par consecutivo y 5 pares dentro de un mismo set**, con la comparación de `zona_sets.py` (artista y título con la
     versión). Para cruzar estos sets con la biblioteca hace falta el puente S15 del plan.
4. **El audio de los temas de los sets (D6) son los previews de Beatport, guardados en el Drive de Gabriel** (Gabriel:
   «está planificado descargar el audio de la preview de Beatport. La idea es subir todos los audios de los temas a
   Drive»). Destino `erclm:traktor_ml/beatport_previews/<corrida>/`, en tandas de hasta 500 previews por `.tar` con su
   manifiesto; sin el conector de Drive ni Drive para escritorio (indicación pegada por Gabriel). Herramienta:
   `tools/dj_sets/beatport_previews.py`. [REEMPLAZADO 2026-11-09 por las respuestas de Gabriel del punto 5: dio el OK a la corrida grande y al trabajo de gpuq, y decidió bajar los previews pese al `ai-train=no`] La corrida grande y el primer trabajo de gpuq de este tipo esperan su OK, y
   el `ai-train=no` que declara Beatport sigue siendo decisión suya.
   - Piloto de 100 temas (agente, desde este PC, 2026-11-09): 95 previews en `beatport_previews/piloto/tanda_00000.tar`
     (134 MB, subida verificada por tamaño y MD5). Cada preview es un MP3 de 120 s, de unos 1,4 MB; Beatport respondió
     200 a todo. Tiempo por tema: 1,8 s de búsqueda y 1,7 s de página y MP3, pausas incluidas.
   - El enlace del botón de compra en 1001Tracklists (un pedido por tema, 8,4 s) llevó a Beatport en 96 de 100 temas;
     4 llevaban a otras tiendas. La búsqueda por nombre en Beatport dio la misma versión en 95 de 100.
   - En los 92 temas con las dos vías, coinciden 81. De los 11 restantes: 3 son el mismo ISRC en otro disco y 4 el
     mismo título y versión; en 3 el enlace de 1001Tracklists apunta a una página de Beatport que ya no existe y la
     búsqueda encontró una que sí; y en 2 la búsqueda se equivocó: eligió el «Love Is Gonna Save Us» de Benny Benassi
     de 2003 en lugar del de ARTBAT y Benny Benassi de 2025, y otro «One Day». INFERENCIA: la búsqueda alcanza como vía
     principal, y el enlace de 1001Tracklists sirve de respaldo para los que la búsqueda no resuelve.
5. **Se bajan todos los sets candidatos y el preview de todos sus temas, para que otro agente evalúe y entrene con ese
   dataset** (Gabriel: «descargues todos los sets y todos los temas que habíamos definido en nuestro dataset»). Antes de
   irse a dormir, sin poder supervisar la noche, respondió:
   - Previews: buscar primero en Beatport y usar el enlace del tracklist si el tema no sale con la búsqueda.
   - `ai-train=no` de Beatport: «Sí, bajarlos igual».
   - Si 1001Tracklists corta: esperar 30 min, probar con una página y seguir; como máximo 3 veces.
   - Encolar en gpuq; si Beatport no responde desde ahí, los previews se bajan en este PC, encendido unas 8 h.
   Resultado (agente, noche del 9 al 10):
   - **Kaggle no llega a Beatport**: la página le pide la verificación de Cloudflare (dos intentos con salida 2; trabajo
     cancelado). No se intenta saltarla. Los previews salen de este PC, en paralelo con las páginas de sets.
   - **Páginas de sets: 838 de 1549** en `1001tracklists/sets/`. A las 02:48, tras unas 830 páginas en 2 h 40 min a 8 s,
     el sitio respondió HTTP 403, y la prueba de las 03:18 dio 403 en la misma página. Los otros dos reintentos se habían
     ido en un timeout (23:11) y en un tracklist movido (302; error del agente, corregido en `e6c566a`). No se guardó lo
     que mostró la página: no se sabe si era un bloqueo del sitio o de esa página. Desde ahora `bajar` la guarda en
     `1001tracklists/paros/`. La próxima prueba conviene hacerla con otra página. Faltan 711.
   - Medidas de los 838 (`medidas.csv`): 453 solo de pares, 360 de trayectoria y pares, 25 solo de trayectoria; 33 304
     filas de temas; 1870 h en los sets de trayectoria; medianas de 97 % identificado y 79 % con enlace a Beatport.
     20 542 temas identificados distintos. Con la biblioteca: 359 apariciones, 10 pares consecutivos y 234 pares en un
     mismo set (S15 sigue haciendo falta).
   - **Previews, por lotes** en `erclm:traktor_ml/beatport_previews/lote_NN/`: cada lote toma los temas de los sets ya
     bajados que no estaban en lotes anteriores. Hasta las 03:30, `lote_01` a `lote_03`: 3703 previews en 11 tandas
     (5,3 GB), 899 temas sin id y 2 sin preview. Los 3703 salieron por la búsqueda: los lotes no pasaron por `enlaces`,
     así que los 899 esperan el respaldo de 1001Tracklists, a un pedido por tema (8 s). Solo 211 de los 899 tienen
     botón de compra en 1001Tracklists; los otros 688 no tienen enlace que seguir. [REEMPLAZADO 2026-11-10 por
     `docs/reports/beatport_busqueda_sin_id_2026-11-10.md`: en 10 de 30 al azar el tema estaba entre los resultados
     (posesivo, «Remix» frente a «Mix», versión pegada al título, alias); sin la «s» suelta ni «remix» salen 66 de
     los 338 de nivel C; el cambio en `match()` espera, porque lo usa también la biblioteca] Entre los de nivel C hay temas que
     parecen estar en Beatport con otra forma de nombrar la versión («Hyenah Raw Beat RMX», «20ms (Cristoph Remix)»):
     HIPÓTESIS, se revisa con una muestra de la búsqueda antes de cambiarla. [REEMPLAZADO 2026-11-10 por el final de
     `lote_03` a las 05:47, con 12 de 12 tandas] `lote_03` seguía (6 de 12 tandas) y [REFUTADO 2026-11-10 por los
     `temas.csv` de los lotes: ese conteo dejaba fuera los 4087 temas identificados solo por nombre; quedaban 16 700,
     que forman `lote_04`] 14 132 temas de los 838 sets no están en ningún lote: a unos 1200 temas por hora, unas 14 h
     más, más los de los 711 sets que faltan.
   - **Final de la noche** (05:47): `lote_01` a `lote_03` cerrados, 7539 temas: 5937 previews en 17 tandas (8,5 GB en el
     Drive, subidas verificadas), 1598 sin id (378 con botón de compra), 3 sin preview y 1 repetido. Los lotes cubren
     24 339 temas distintos de los 838 sets: 20 252 con id de 1001Tracklists y 4087 solo por nombre, sin nombres
     repetidos entre los dos grupos. `lote_04` (los 16 700 restantes, 34 tandas, unas 13 h) arrancó a las 05:47 y sigue
     mientras el PC esté encendido; se retoma por tandas (plan § 6, «Cómo se retoma la bajada»).

## 2026-11-08

1. **Los tracklists de 1001Tracklists se bajan directamente de la página, con la cuenta de Gabriel**
   (Gabriel). El correo ya se mandó, tardan en contestar y no se depende de eso. Reemplaza la ruta del
   punto 9 del 2026-11-07. [REEMPLAZADO 2026-11-08 por decisión de Gabriel: se bajan muchos, con su cuenta] Bajar muchos queda sin decidir.
   - [REEMPLAZADO 2026-11-09 por el punto 3 de «sets de 1001Tracklists»: en volumen se lee la página, con `tools/dj_sets/tracklists_1001.py`; trae lo mismo con un solo pedido por set] Cómo (agente, probado con un solo tracklist, Maddix `1ch990z1`): `export_1001tracklists.py`
     (Playwright con Chromium visible; está en Descargas, fuera de Git) pulsa el botón oficial de exportar
     y guarda el texto que devuelve `/ajax/export_data.php`. Gabriel inicia sesión a mano en esa ventana:
     la contraseña no pasa por el agente.
   - La página y el botón se ven sin sesión, pero sin cuenta el servidor rechaza la exportación: «will be
     enabled after your account has been active for one week».
   - Resultado: 35 temas, igual al texto que pegó Gabriel. `Fēlēs` llega descompuesto (NFD): normalizar a
     NFC antes de comparar nombres.
   - Aviso (agente): la ruta anterior evitaba justo esto. El `robots.txt` bloquea al agente «CPython» y
     pide `Crawl-delay: 8` al resto.
   - [REEMPLAZADO 2026-11-09 por `tools/dj_sets/tracklists_1001.py`: 8 s, parada al primer rechazo, salta lo ya bajado y el perfil queda fuera del repo; el script de Descargas no se usa para volumen] Pendiente en el script antes de darle una lista: en Windows `write_text` deja `\r\r\n` (falta
     `newline=""`); la pausa es de 3 s y el sitio pide 8; sigue con la lista después de un rechazo; no
     salta lo ya bajado; y crea su perfil con cookies en la carpeta desde donde se corre.
   - Enlaces al audio (agente, probado con un tracklist y un tema). El texto exportado no los trae. La
     página sí da, sin pedidos extra, el reproductor del set (en este caso Hearthis) y, por tema, el
     segundo exacto, el ID y la URL en 1001Tracklists y el sello; duración y género, no en todos. YouTube y
     Beatport salen de `/ajax/get_medialink.php`, un pedido por tema: `playerId` es el ID y el botón de
     compra trae la URL de Beatport. El video de YouTube puede ser más corto que el tema (157 s contra
     271 s en el probado): comparar con la duración.
   - Límites (agente): en el HTML y el JS no hay contador de descargas, y las respuestas no traen
     cabeceras de límite. Eso no prueba que no exista: el servidor sabe qué cuenta exporta, y hay captcha
     en `get_data.php` y Turnstile en `add_count.php`. Al bajar: 8 s entre pedidos y parar al primer
     rechazo o captcha.
2. **Parecidos (la playlist de D0) es la lista de temas de los que Gabriel tiene «la seguridad de que
   van a sonar bien» con lo que está tocando** (Gabriel). Es precisión de la lista, no acertar el tema que
   él habría puesto después. «Sonar bien» puede ser CLAP tal cual o CLAP mejorado con lo que se aprenda de
   los sets de DJs profesionales.
   - Cómo (agente): el acierto en 30 sobre los sets de Gabriel (E1) queda como línea de descarte; las
     medidas nuevas son la compacidad de los sets profesionales en CLAP (radio de zona), la fracción de
     los 30 dentro de ese radio y la escucha a ciegas de listas enteras de a dos (plan, sección 11, E9 a
     E12). Ninguna variante de la etapa 1 ni de líneas D1 había pasado por la escucha.
3. **El material de DJs profesionales ya bajado (D7, D8) se reutiliza para Parecidos**, no solo para las
   playlists de dirección (Gabriel). Cómo (agente): radio de zona calibrado (P8) y proyección liviana de
   CLAP aprendida de sets (P9), con discos fuera de la muestra.
4. **Los modelos de Essentia (danceability y otras «energías») son candidatos a coordenada de las
   playlists de dirección** (Gabriel: deberían ser los que juzguen las trayectorias). Cómo (agente): el
   juez es la forma de los sets largos reales y los pares de Gabriel (E11); una coordenada que no se mueve
   con la noche no sirve, por buena que suene su etiqueta (como la energía de Mixed In Key). Hace falta
   la pasada de EffNet o MusiCNN en WSL (V6, prueba chica antes).
5. **Texto en CLAP como dirección**: parecidos al tema principal y a una palabra o concepto a la vez
   (Gabriel). Cómo (agente): puntaje = peso por el coseno con el principal más peso por el coseno con el
   concepto; dirección como diferencia de frases opuestas o percentil, varias frases por concepto; se
   valida escuchando los extremos de la biblioteca y proyectando D7, D8 y los sets largos de Gabriel
   (plan, T18 y E11). El codificador de texto ya está en el repo (`tag_vocals.py`).
6. **«Fine-tuning de Essentia» para esta aplicación** (Gabriel preguntó si se puede). Cómo (agente): lo
   realista es entrenar cabezas sobre los embeddings congelados (MAEST, CLAP), como ya hacen
   `genre_model.py` y `clap_probe`, con etiquetas de posición o fase en los sets profesionales y partición
   por DJ (plan, T20). Afinar el tronco de MAEST en torch es M4 (GPU) y espera.
7. **Referencia para trayectorias**: Vande Veire y De Bie 2018 (DJ automático de Drum and Bass). Se toman
   las ideas, no se copia (Gabriel): zona de estilo con centro y paso chico, validación escuchando dos
   mezclas, tonalidad con ±1 semitono transpuesto, tipos de transición (plan, sección 11.1).
8. **Un agente nuevo implementa la sección 11 desde un chat nuevo** (Gabriel), con el prompt
   `docs/plans/PROMPT_SEGURIDAD_Y_DIRECCION.md`. La sesión auditora sigue en paralelo.
9. **Los conceptos de las direcciones de texto son provisionales; si el método funciona, se escriben desde el
   teléfono** (Gabriel: «los conceptos van a cambiar»; si es efectivo, una interfaz en el móvil para escribir
   conceptos y que esos conceptos filtren la música). Por ahora, 4 conceptos ortogonales para probar la idea.
   - Cómo (agente, propuesta): energía, luz (luminoso u oscuro), melodía (melódico o percusivo) y voz, medidos sobre
     los 1760 temas con 5 frases por extremo; la correlación de Spearman más alta entre ellos es 0,30. Voz es un control
     contra la etiqueta Vocal (AUC 0,76). Energía se mueve en parte con el BPM (0,37). Profundidad, la tercera dimensión
     de la literatura junto con energía y valencia (Greenberg et al. 2016), no quedó independiente: −0,52 con energía
     y −0,59 con luz, y se prueba aparte en A3. En A3 se mide también el caso de una sola frase, como se escribiría en
     el móvil.
10. **Para las medidas se usa todo lo disponible ahora, curado** (Gabriel: «no queremos usar basura»). Los tracklists de
    1001Tracklists están en importación y entran cuando lleguen.
    - Cómo (agente): `docs/reports/zona_sets_2026-11-08.md` dice qué entró y qué quedó afuera. Además de D7 y D1 entran dos
      fuentes que el plan no listaba: los temas de la biblioteca que aparecen juntos en sets de D3 y los originales de
      Bedrock, porque el radio se usa entre originales y D7 es audio mezclado. Se toma su respuesta como que sus 4 sets
      largos entran en E11, como referencia con ruido.

## 2026-11-07

1. **En Traktor habrá varias playlists que se actualizan solas** (Gabriel).
   - Una playlist de parecidos, ordenada por parecido: el primer tema es el más parecido de la
     biblioteca al tema principal que suena.
   - Otras de dirección: mantener la energía, más energía, menos energía, más introspectiva. Estas
     tienen en cuenta la trayectoria de todo el set.
   - Plan con el inventario completo de ideas: `docs/plans/recomendacion_playlists_plan.md`.
2. **Introspectivo**, en palabras de Gabriel:
   - suele pasar en sets de 3 horas o más, en general después de la segunda hora;
   - no depende necesariamente de la energía;
   - en general no tiene grandes saltos.
3. **La energía de Mixed In Key no se usa** (Gabriel: prácticamente inútil). En los datos, el 83 % de
   los 1424 temas que la tienen está en 5 o 6. La fase 4 del pipeline, que arranca cada playlist por el
   tema de menor energía de Mixed In Key, no se toca por ahora (Gabriel, 2026-11-07): quitarlo cambiaría
   el orden de las playlists ya exportadas.
4. **Pueden usarse las listas de sets de Kim et al.**, solo texto y sin audio (Gabriel).
   - Cómo (agente): el dataset público de ese grupo es djmix-dataset, en
     `artifacts/v4/dj_sets/djmix/`, fuera de Git. Viene de MixesDB, no de 1001Tracklists: 5040 sets,
     63 038 temas y 64 764 transiciones.
   - No tiene sets de más de 150 minutos: para la fase introspectiva cuentan los sets largos de Gabriel
     y, si se decide, 1001Tracklists propiamente dicho.
5. **Los sets pasados de Gabriel en este PC sirven de referencia, con ruido** (Gabriel: «no me
   considero muy buen DJ»). Están en la historia de Traktor: 21 sets reales en 4.4.1 (592 temas al
   aire) y 12 sesiones en 4.2.0 [REFUTADO 2026-11-07 por el md5 de los 12 archivos (auditoría del plan):
   la carpeta 4.2.0 es copia byte a byte de 12 archivos de 4.4.1; D1 son los 21 sets de 4.4.1, 562 de
   sus 592 temas al aire están en el índice]. Toda medida se informa también sobre sets de otros DJs.
6. **GPU y disco los consigue Gabriel cuando hagan falta**: en la etapa con audio, con una estimación
   medida antes. Mientras se trabaje solo con texto, no hacen falta.
7. **El género de Beatport es una decisión editorial por lanzamiento** (agente, sobre una observación
   de Gabriel).
   - Gabriel advirtió que usar esas etiquetas es aprender lo que decidió Beatport.
   - El centro de ayuda de Beatport dice que su equipo de curación revisa y asigna el género, y que el
     del sello es un pedido. No se encontró una fuente sobre un clasificador automático.
   - La advertencia vale igual: es una señal entre otras.
8. **El botón de sugerencias mantiene su gesto y refresca todas las playlists** (Gabriel). D0 en su
   descripción; en la mezcla externa, D#0 + D0, punto 9 de 2026-11-06.
   - Al apretarlo, refresca las playlists respecto del tema principal y abre la de parecidos.
   - Las de trayectoria quedan al lado, en Favoritos de Traktor, y Gabriel las elige con el mouse.
   - Al apretarlo otra vez, vuelve a la carpeta del navegador que estaba elegida antes de la primera
     pulsación.
   - Cómo (agente): todas las playlists de sugerencias tienen que contar como «dentro» para la vuelta.
     [REEMPLAZADO 2026-11-07 por `Toggle.inside` y la prueba V1: cualquier playlist de la carpeta de
     sugerencias cuenta como dentro, y Traktor relee desde Favoritos un `.nml` reescrito al llegar
     desde otra playlist, no al volver a tocar el que está abierto] Hoy `Toggle` reconoce una sola.
     Falta probar que Traktor relea desde Favoritos un `.nml` reescrito.
9. **Se usan sets largos de 1001Tracklists** (Gabriel). Preguntó de dónde sacarlos y si tiene que hacer
   algo.
   - Cómo (agente), 2026-11-07: 1001Tracklists no tiene API pública, y su `robots.txt` bloquea a los
     scripts de Python. [REEMPLAZADO 2026-11-08 por decisión de Gabriel (punto 1 de ese día): se exporta
     desde la página con su cuenta, sin depender del correo] La ruta propuesta es que Gabriel pida acceso de investigación a
     info@1001tracklists.com, como hizo el grupo de Kim. Mientras tanto, sets largos de MixesDB y los
     suyos.
   - Plan e informe para auditoría: `docs/plans/recomendacion_playlists_plan.md`.
10. **Al grabar, Spotify va solo a Voicemeeter y después vuelve a su salida, sin que Gabriel lo note**
    (Gabriel). Cómo (agente):
    - antes de darle play, el reproductor manda Spotify a la salida elegida para grabar («Solo Spotify,
      sin que suene nada», en el cuadro que se abre al darle Descargar o Continuar);
    - al terminar la descarga (bien, pausada, detenida o con error), Spotify recupera la salida que
      tenía. Si se la cambió a mano mientras tanto, queda la suya;
    - `spotify_output.py` y `app/audio_devices.ps1 -App/-SetApp`. Es la interfaz no documentada que
      usa EarTrumpet (`docs/LESSONS_LEARNED.md`).
    - Reemplaza el paso manual del Mezclador de volumen del punto 4 de 2026-11-06.
11. **La playlist de cada tanda suena en repetición** (agente, por la primera prueba de Gabriel con
    Voicemeeter, que grabó 0/1). Al final de una playlist, Spotify se detiene y vuelve al primer tema
    con el progreso en 0, y el vigilante lo tomaba por una pausa. En repetición, el último tema se
    corta como los demás y Spotify se pausa apenas la tanda queda grabada. El vigilante también
    reconoce ese final.
12. **Para aprender trayectorias de DJs se puede bajar audio de YouTube** (Gabriel: «para esta
    aplicación está totalmente perfecto descargar de YouTube»).
    - La propuesta es de Gabriel: sets de DJs que él elige, analizados en bloques de unos 30 s con CLAP,
      para aprender las trayectorias de DJs profesionales sin entrenar otro modelo de representaciones.
    - Cómo (agente): el audio queda fuera de Git y se borra después de sacar los embeddings. Antes de
      escalar, un piloto en CPU con unos 5 sets (`docs/plans/recomendacion_playlists_plan.md`,
      sección 10).
13. **Las compilaciones mezcladas son una fuente de sets** (Gabriel). Por ejemplo, discos de John
    Digweed: los temas están subidos uno por uno como «Mixed» y, escuchados en orden, son un set
    mezclado con transiciones suaves en el que cada tema conserva su identidad.
    - Cómo (agente): en las tiendas el álbum lleva «(DJ Mix)» y cada tema «(Mixed)». MusicBrainz las
      marca con el tipo `DJ-mix`: 26 438 que no son emisiones de radio, 4817 con etiquetas de techno o
      house, 49 de Digweed.
    - Dan el orden, el límite y la identidad de cada tema. Con su audio, los componentes de audio se
      miden sobre sets de otros DJs desde la etapa 2 del plan.
14. **Los temas descargados llevan la metadata de la biblioteca, sola, al terminar cada descarga**
    [REEMPLAZADO 2026-11-07 por el punto 16: tema por tema mientras la descarga sigue, y con el embedding]
    (Gabriel: salían solo con la de Spotify). Se usan las reglas de Beatport (2026-09-29) y de Vocal
    (2026-10-05). **Sin BPM ni tonalidad** (Gabriel).
    - Cómo (agente): `src/v4/pipeline/tag_downloads.py` reutiliza `beatport_lookup`,
      `write_beatport_tags` y `tag_vocals`, sin reescribirlos (Gabriel lo pidió así). Corre con un
      Python que tiene el pipeline (`TRAKTOR_PIPELINE_PYTHON`).
    - Si Beatport pide la verificación anti-robots de Cloudflare, esos temas quedan en espera para la
      próxima descarga.
15. **Los temas grabados de Spotify suben al nivel de los bajados** (Gabriel: en Traktor se veían más
    chicos y había que subirles la ganancia).
    - Medido: un grabado quedó en −17 LUFS (pico −8,9 dBFS); los de Soulseek de la misma playlist
      están entre −6 y −10 LUFS.
    - Cómo (agente): el grabador sube cada tema hasta un pico de −1 dBFS. Solo sube, nunca baja: la
      normalización de Spotify baja el tema entero por igual, así que esto le devuelve el nivel sin
      cambiar el sonido. [REFUTADO 2026-11-07 por la medición de la carpeta (Gabriel: «no lograste que
      todas las pistas tengan el mismo volumen»): con la normalización apagada, los grabados quedaron en
      −8,4, −9,6 y −11,9 LUFS contra −6,3 a −8,3 de Soulseek; la versión de Spotify tiene más rango
      dinámico y llega al techo de pico con menos sonoridad. Ahora se sube la sonoridad a −8 LUFS con un
      limitador bajo −1 dBFS (solo sube, hasta +10 dB); quedaron en −8,4 y −8,5.]
16. **Que todo el proceso dure lo menos posible, en paralelo donde se pueda, y que la música quede lista
    para la colección y las playlists** (Gabriel). Cómo (agente):
    - la metadata (Beatport, Vocal) y el **embedding CLAP** (`clap_full`, el de la biblioteca) se hacen
      tema por tema apenas cada uno está bajado o grabado, mientras sigue el resto
      (`tag_downloads.py --watch`, con prioridad baja). El embedding queda dentro del archivo, como en
      el pendrive, y en la caché del dataset `musica`. Antes iba todo al final: casi 5 minutos para 10
      temas;
    - la tanda de grabación siguiente empieza en el último corte de la anterior, sin esperar a que se
      guarde su MP3; ese MP3 se codifica con LAME `-q 2` (14 s contra 45 s por tema de 5 minutos).
    - [REEMPLAZADO 2026-11-07 por el punto 18: lo que Beatport no confirma lleva el género del
      clasificador de la biblioteca, y MAEST queda en caché] Sigue sin modelo de género: lo que Beatport
      no encuentra queda sin género. El clasificador de la biblioteca necesita además MAEST; se puede
      sumar si Gabriel lo pide.

17. **La tonalidad se prueba como componente del puntaje, no solo como filtro, y lo tocado registra la
    transposición** (Gabriel).
    - Gabriel cambia a menudo la tonalidad en sus sets; a veces toca 3 o 4 temas seguidos transpuestos
      un semitono.
    - Lo compatible sin transponer puntúa más (como ya decía el punto 9 de 2026-11-06). Transponer 1 o
      2 semitonos no debe penalizar mucho, pero no transponer es deseable.
    - El registro de lo tocado anota si el tema se transpuso y cuánto.
    - Cómo (agente):
      - opciones del recomendador `filtro_clave`, `clave_penalizacion` y `clap_escala`, que por
        defecto dejan lo de antes;
      - «principal» y «tocado» anotan la tonalidad original, la resultante, los semitonos, el tempo y
        el key lock;
      - en D1, apagar el filtro no cambia el acierto (8,9 % frente a 9,3 %) y deja disponible el tema
        siguiente en el 97 % de las transiciones, contra el 74 %;
      - [REEMPLAZADO 2026-11-07 por la auditoría de E1: la causa son los valores de la regla armónica,
        que ordenan al revés de D1 la vecina sin transponer y la misma tonalidad transpuesta. Los valores
        tienen que salir de datos, evaluados fuera de muestra (plan, P7)] sumar la tonalidad como
        componente empeora, probablemente porque D1 solo tiene tonalidades originales;
      - se repite con D2. Informe: `docs/reports/e1_sets_gabriel_2026-11-07.md`, adenda.

18. **Los temas nuevos quedan con la misma metadata que la colección, para entrar en playlists con los
    viejos; la parte que no va en el archivo se completa al agregarlos a la colección** (Gabriel: «la
    metadata debe estar completa, sino no se podrán utilizar los temas nuevos para formar playlists con
    los temas viejos»).
    - BPM y tonalidad los calcula Traktor cuando se cargan los temas (Gabriel). Traktor los guarda en
      `collection.nml`, no en el archivo. HIPÓTESIS: por eso 249 de 300 archivos de la biblioteca
      tienen TBPM=0 (todos llevan PRIV:TRAKTOR4).
    - Lo que se hace al descargar (agente, aprobado por Gabriel):
      - lo que Beatport no confirma (C o D) lleva el género del clasificador de la biblioteca
        (`genre_model.py`), con la regla de 2026-09-29: confianza ≥ 0,6 y sin el gate de MAEST;
      - MAEST y el subgénero de Beatport quedan en caché.
    - Lo que se hace en «Agregar música nueva», el paso siguiente: Gabriel tiene que usarlo después de
      cargar y analizar los temas en Traktor.
      - BPM y tonalidad: los tags; si no, Traktor (`phase1_tags.py --traktor`, por nombre de archivo y
        duración, solo con una coincidencia); y el BPM que siga faltando se estima.
      - El paso «Género y subgénero»: MAEST de la carpeta, Beatport de la carpeta sumado a
        `features/beatport`, `beatport_subgenres` y `genre_model --predict`. Lo que falla queda como
        aviso y no corta el agregado.
      - Una carpeta que ya se agregó se completa con «Completar BPM, tonalidad y género», sin tocar las
        playlists.
    - Verificado:
      - Traktor y los tags coinciden en el BPM en el 98 % de los 102 temas comparables (±1 BPM).
      - En la tonalidad coinciden en 81 de esos 102. Las diferencias son casi todas de un paso en el
        círculo de quintas.
      - El clasificador de la descarga da el mismo género que `genre_pred.parquet` en 330 de los 331
        temas C/D de la biblioteca.

19. **«Agregar a la colección» trabaja solo con lo nuevo, no se queda trabado y no duplica carpetas**
    (Gabriel: «para agregar 10 temas se revisa toda la biblioteca desde cero… es poco práctico»; «que no se
    vuelva a trabar o, si lo hace, que no se generen dos carpetas duplicadas»).
    - Medido el 2026-11-07, agregando 10 temas que ya venían analizados de la descarga:
      - BPM y tonalidad: 24 s;
      - embeddings: 70 s, rearmando la tabla de los 1760 temas;
      - voces: 3,5 min, leyendo los comentarios de toda la biblioteca;
      - MAEST: 35 s;
      - subgénero y modelo de género: 44 s;
      - el pendrive iba a reescribir los 1760 archivos, porque la versión de la organización va en cada uno.
    - Cómo (agente):
      - voces de la carpeta (`tag_vocals --scope`);
      - ensamblado incremental (`extract_representations --incremental`);
      - en el pendrive, solo lo nuevo o lo que se movió, y la versión ya no cuenta como cambio;
      - tags leídos con caché por tamaño y fecha (`tag_cache.py`) en el pendrive y en la búsqueda de repetidos.
    - Traba (EVIDENCIA OBSERVADA):
      - El primer intento quedó en «Leer los temas»: el script no arrancó (13 MB, 0 s de CPU a los 30 min).
      - El segundo se congeló entero al importar `dedupe` a mitad de la tarea.
      - En una prueba aislada, con un hilo bloqueado leyendo la entrada estándar, un proceso hijo no arranca e
        importar la app queda colgado más de 200 s.
      - INFERENCIA: la causa es la lectura pendiente de esa entrada, con la que la app sabía cuándo se cerraba la
        de descargas. Con la cadena exacta de la app, un hijo sí arrancó: el mecanismo exacto no está aislado.
    - Lo que se hizo contra la traba:
      - la entrada se mira sin bloquear (`PeekNamedPipe`), y también si sigue vivo el proceso de la app de
        descargas (`--parent-pid`);
      - los scripts arrancan sin heredar la entrada;
      - un script que no escribe ni usa CPU durante 10 min se corta y se lanza una vez más;
      - «Cancelar» corta el paso en curso.
    - Lo que se hizo contra las carpetas duplicadas:
      - una copia cortada se completa en vez de hacer «nombre (2)»;
      - un tema que ya está en la biblioteca no se copia de nuevo, y lo que falta va junto a él;
      - cada archivo se copia con otro nombre y se renombra al terminar.
    - Lo que se hizo para que el proceso sea más claro:
      - en Descargas, «Siguiente paso: sumarla a tu colección» (Traktor y Analyze; después «Agregar a la
        colección»);
      - en Organización, cuántos temas ya analizó Traktor;
      - pasos con nombres que dicen lo que hacen.

## 2026-11-06

1. **Descargas de Soulseek tema a tema, grabando mientras sigue la búsqueda** (Gabriel). Con muchos
   temas, el preflight en lote se trancaba: abría 20 búsquedas juntas, gastaba las 34 permitidas en
   menos de 2 minutos y quedaba otros 2 parado, una y otra vez (log del 2026-11-02, 152 temas: 4
   pausas de 104-119 s). A mano, Gabriel busca un tema, lo elige y lo baja, y no pasa. Grabar en
   Spotify puede empezar con el primer tema que haya que grabar, sin esperar a buscar toda la
   playlist. Cómo (agente):
   - el descargador busca cada tema en un proceso de sockseek y lo baja en el momento; el límite de
     34 búsquedas cada 220 s lo lleva un archivo del ordenador, compartido por todas las descargas;
   - lo que no se consigue entra en `recording_queue.jsonl` apenas se sabe;
   - el orquestador graba la cola en tandas de hasta 10 temas, cada una con su playlist privada de
     Spotify y su carpeta `recordings/NN/` con checkpoint;
   - con la cola vacía espera el tema siguiente y pausa Spotify (solo en este ordenador);
   - Spotify detenido, el reproductor que no arranca o una playlist que no se pudo crear detienen la
     grabación pero no la descarga: el run termina «En pausa» con lo bajado ya en la carpeta;
   - «Posponer la grabación» corta la tanda en curso al instante;
   - 3 búsquedas seguidas que fallan cortan la descarga sin mandar esos temas a grabar;
   - los runs anteriores no se retoman: un run nuevo reutiliza lo que bajaron o grabaron.
2. **Un punto medio: 2 temas a la vez, y vigilar las descargas que no avanzan** (Gabriel, el mismo
   día). Propuso fijar 2 en paralelo o ajustarlo solo, y detectar una descarga que se pausa para
   relanzarla o probar otra. Cómo (agente):
   - dos procesos de sockseek con la misma cuenta se cortan, así que todo va por una sola sesión:
     el daemon de sockseek de la descarga;
   - 2 a la vez por defecto (`--parallel`, de 1 a 4), sin ajuste automático: el log del 2026-11-02
     muestra que el tiempo se va en los usuarios que no mandan el archivo, no en el límite de
     búsquedas, que con 2 temas a la vez queda lejos;
   - una transferencia que no recibe nada en 45 s se deja y se prueba otro archivo;
   - primero se piden los archivos de usuarios con lugar libre para subir;
   - un tema del que no llegó ningún archivo se busca de nuevo al final antes de grabarlo.
3. **Las playlists temporales de Spotify se borran al terminar cada tanda** (Gabriel lo pidió). Cómo
   (agente): `DELETE /me/library`, el reemplazo de «dejar de seguir» desde febrero de 2026; un fallo
   queda anotado y no corta la descarga.
4. **Grabar Spotify por Voicemeeter, sin que suene nada y sin el DJM** (Gabriel). El DJM servía para
   grabar horas con los parlantes apagados, pero con Traktor abierto el DJM queda en exclusiva y no se
   puede grabar (la prueba del 2026-11-06 falló así). Gabriel eligió Voicemeeter (ya tenía el
   instalador) entre las opciones del agente: un cable virtual, o la salida del K2. Cómo (agente):
   Spotify sale solo por «Voicemeeter Input» (Mezclador de volumen de Windows) [REEMPLAZADO 2026-11-07
   por el punto 10 de 2026-11-07: la descarga manda Spotify ahí al grabar y se lo devuelve al terminar]
   y el grabador graba esa salida («Grabar desde» en la app, `SPOTIFY_RECORDING_OUTPUT`). Pasos en
   `download_JIJIJI/CONFIGURAR_OTRO_ORDENADOR.md`, paso 7b. Falta que Gabriel lo instale y lo pruebe.
   [REEMPLAZADO 2026-11-07 por la instalación: Voicemeeter está instalado en el escritorio; falta una
   grabación completa con el punto 11 de 2026-11-07.]

## 2026-11-05

1. **Sugerencias en vivo dentro de Traktor con un botón del K2, sin clics en la pantalla** (Gabriel).
   Gabriel descartó los clics automáticos: no son robustos si él navega el árbol o pliega el panel de
   temas. Tampoco hace falta rehacer Traktor. Eligió un solo botón, Ch15.Note.D0, cuya preescucha no
   usaba (no C#0, que despliega el panel): la primera pulsación lleva a 10 sugerencias nuevas y la
   segunda vuelve a la carpeta donde estaba. Cómo (agente): `sugerencias_vivo.py` lee el K2 junto con
   Traktor. Escribe el `.nml` en `C:\TRAKTOR ML Sugerencias` y mueve el árbol con el navegador de la
   pantalla del D2 virtual (`ApiBrowserProbe.qml`). Probado por Gabriel el 11-05 desde `#1 BIBO` y
   `#1 BIBO › PRO`: cada ida muestra la lista nueva (0,5-0,65 s) y la vuelta llega a la carpeta con el
   mismo tema elegido (0,6-0,8 s; el tema lo conserva Traktor). Siguiente: el cálculo de las
   sugerencias con la trayectoria del set (ideas de Gabriel, sin empezar).
2. **Las sugerencias en vivo quedan registradas en GitHub y tienen que funcionar en Windows y macOS, y
   adaptarse a otro equipo** (Gabriel). Cómo (agente):
   - la versión probada tiene la etiqueta `sugerencias-vivo-v1`;
   - la guía es `docs/SUGERENCIAS_EN_VIVO.md`;
   - el MIDI va por `midi_io.py`: winmm en Windows, python-rtmidi en macOS;
   - el instalador encuentra Traktor en los dos sistemas;
   - las rutas de la colección de Mac se leen (VOLUME es el nombre del disco);
   - el controlador, el botón (nota o CC), la cantidad y la carpeta se eligen y quedan guardados;
   - el camino del árbol se aprende si Traktor lo muestra distinto;
   - hay lanzadores `.bat` y `.command`.

   macOS está cubierto por tests (corren en Windows y Linux), pero falta probarlo en un Mac con
   Traktor. Gabriel probó esta versión en Windows el mismo día y funcionó (etiqueta
   `sugerencias-vivo-v2`).
3. **Los mapeos de Gabriel van al repo, y otro LLM puede guiar la réplica** (Gabriel). Cómo (agente):
   - `tools/traktor_live/mapeos/` tiene su Xone K2 y su teclado, exportados de su Traktor 4.4.1 con
     `mapeo_tsi.py`.
   - Al K2 se le quitó la preescucha, que no usaba (punto 1), así que queda listo para las sugerencias.
     Son tres asignaciones:
     - D0: "Load Selected (Preview Player)", con M1 = 0;
     - D0: "Modifier #2" mientras está apretado;
     - el encoder derecho con M2 = 1: "Seek Position (Preview Player)".

     El 11-06 Gabriel preguntó por el modificador y se vio que solo servía para eso. En su Traktor
     actual las tres siguen.
   - No se sube su exportación completa de la configuración (`XONE K2 + TECLADO (TEMPO).tsi`), porque
     trae audio, ruteo y campos de contraseña.
   - `PROMPT_LLM.md`, en la raíz, es el prompt mínimo para un LLM que acompaña a una persona.
4. **Las sugerencias arrancan solas con Traktor, sin correr nada a mano** (Gabriel, 2026-11-06). Cómo
   (agente):
   - `sugerencias_servicio.py` corre sin ventana desde que se inicia sesión: en Windows, un acceso
     directo a `pythonw` en la carpeta Inicio, que no pide administrador; en macOS, un LaunchAgent;
   - cuando Traktor se abre, empieza una sesión; cuando se cierra, suelta el K2 y el puerto;
   - avisa con notificaciones del sistema sin sonido, para que nada salga por la DJM;
   - en Windows detecta Traktor por la API de procesos, porque `tasklist` desde `pythonw` abriría una
     consola cada vez.

   Queda activo en este PC desde el 2026-11-06. La ventana a mano (`Sugerencias en vivo.bat`) sigue
   para ver lo que pasa. Al cerrar esta etapa se entrega un prompt para un chat nuevo sobre el cálculo
   de las sugerencias: tema principal con mezclador interno o la DJM, generación y registro de lo que
   suena.
5. **Cierre de la etapa de las sugerencias en vivo** (Gabriel, 2026-11-06: «considera que sí funciona
   tocando un rato largo y pasemos a la siguiente etapa»). Queda con la etiqueta
   `sugerencias-vivo-v3`. La siguiente etapa, la recomendación real, empieza en un chat nuevo con
   `docs/plans/PROMPT_ETAPA_RECOMENDACION.md`: tema principal con mezclador interno o la DJM,
   registro de lo que suena y cómo se generan las sugerencias.
6. **Dos modos para elegir el tema principal, según el MASTER de Traktor** (Gabriel). En Traktor el
   MASTER puede ser el deck 1, 2, 3 o 4, o el master tempo: cinco opciones.
   - **Si el MASTER es un deck,** el tema principal es el de ese deck, esté o no sonando y tenga o no
     el volumen arriba.
   - **Si es el master tempo,** el tema principal se calcula con el algoritmo de las señales: faders
     de la DJM o volumen interno, play, tiempo al aire. Es el modo difícil.
   - El receptor ya lo distingue: `updateMasterClock.deck` vale "A" a "D", o `null` para el master
     tempo.

   Se agregó al prompt de la etapa de recomendación.
7. **Reglas del tema principal y de las sugerencias** (Gabriel, 2026-11-06, al plan de la etapa de
   recomendación).
   - Usa las dos mezclas: la interna con `XONE K2 + TECLADO (TEMPO).tsi` y la externa con
     `DJM + K2 (external mix) (1).tsi`, donde los decks A a D van a los canales 1 a 4 de la DJM.
   - Hace mezclas largas con los dos temas arriba. Un tema tiene intro (el primer 30 %), main (del 30
     al 70 %) y outro (desde el 70 %): main gana a intro e intro gana a outro; el outro siempre pierde.
     El tema principal casi seguro tiene el fader por encima del 90 %.
   - Un fader sin dato al arrancar cuenta como arriba (él los va a mover al abrir Traktor).
   - Sugerencias por ahora solo por parecido CLAP, las 10 o 30 más cercanas; la trayectoria, después.
     Filtros: BPM ±20 y tonalidad compatible transponiendo hasta 2 semitonos
     (`docs/HARMONIC_COMPATIBILITY.md`). Se excluye lo tocado en la sesión, comprobando que sonó: cargar
     no alcanza.
   - El pendrive está siempre conectado; el agente elige una sola ubicación de los temas.

   Cómo (agente), en `tools/traktor_live/tema_principal.py`:
   - la mezcla se lee de `Routing.Mixer.External` en `Traktor Settings.tsi`, porque la API de Traktor
     no la informa;
   - con la mezcla externa los faders salen de la DJM (CC 17 a 20) y con la interna, de Traktor;
   - un cambio con el tema anterior todavía al aire se sostiene 3 s; si el anterior sale del aire,
     cambia enseguida;
   - tocado = 30 s en play con el fader al 50 % o más.
8. **Tonalidad de Traktor, 30 sugerencias y cambio de mezcla con Traktor abierto** (Gabriel, 2026-11-06).
   - Toca con la tonalidad que estima Traktor: se ve en el deck, con un color que le dice si el cambio
     de pitch deja los temas compatibles. La recomendación usa esa; la de los tags, solo si Traktor
     todavía no analizó el tema. Traktor analizó 231 de los 1717 temas del paquete que tiene en su
     colección; en los 194 que tienen las dos, coinciden 160.
   - 30 sugerencias por lista.
   - Cambia entre la mezcla interna y la externa con Traktor abierto, cargando el otro `.tsi`; el K2
     está siempre conectado. [REFUTADO 2026-11-06 por la sesión de las 17:12: Traktor escribió sus
     ajustes estando abierto (17:14:00, al cargar la configuración externa, y 17:20:13), y el programa
     anotó «mezcla externa» a las 17:14:02. `MixerWatch` ve el cambio a los 2 s.] Traktor escribe
     `Traktor Settings.tsi` solo al cerrarse, así que el modo
     leído al abrir no ve ese cambio. La DJM conectada tampoco indica mezcla externa: desde el 11-04,
     Traktor estuvo en mezcla interna con la DJM como placa de sonido (respaldos de `Backup/Settings`).
     Lo que sí cambia: con `XONE K2 + TECLADO (TEMPO).tsi` los faders del K2 mueven el volumen de canal
     de Traktor; con `DJM + K2 (external mix) (1).tsi`, no.

   Cómo (agente), en `tools/traktor_live/recomendador.py`:
   - el índice de la biblioteca (`indice_recomendacion.py`, con el Python del pipeline) da embedding
     CLAP, BPM y tonalidad de tags por track_uid; en vivo se lee con la biblioteca estándar;
   - se sugiere solo la copia de `C:\TRAKTOR ML`, la que Traktor tiene en su colección (1717 temas);
   - un tema sin tonalidad de Traktor ni de tags no se sugiere (227 de 1717 hoy);
   - si el tema principal no está en la biblioteca, quedan los compatibles por BPM y tonalidad, del BPM
     más cercano al más lejano;
   - la lista se recalcula cuando cambia el tema principal, cuando un tema pasa a tocado y cuando el
     pitch cambia el BPM redondeado o la tonalidad del tema principal (unos 50 ms).
9. **El orden de las sugerencias será una combinación pesada de variables, y el botón en la mezcla
   externa es D#0 + D0** (Gabriel, 2026-11-06).
   - Las variables son CLAP, BPM, tonalidad, género de Beatport y un vector TIV que Gabriel va a sumar
     para estimar la compatibilidad armónica. Los pesos se ajustan más adelante.
   - El mismo BPM puntúa más que uno a -20. La misma tonalidad original puntúa más que una que pide
     mover el pitch.
   - [REEMPLAZADO 2026-11-06 por el punto 11: la taxonomía existe y da la similitud del componente
     `genero`] Para el género hace falta una taxonomía de géneros de Beatport que traduzca dos
     etiquetas a un número de similitud: queda pendiente.
   - En su configuración de mezcla externa, D0 pliega o despliega la biblioteca. Gabriel la condiciona en
     Traktor a M6 = 0, y las sugerencias se abren manteniendo D#0 (M6 = 1 mientras está apretado) y
     apretando D0.

   Cómo (agente):
   - `recomendador.py` da a cada candidato componentes de 0 a 1 (`clap`, `bpm` lineal hasta ±20, `clave`
     con la regla armónica, `genero` 1 igual o 0 distinto hasta que exista la taxonomía, `tiv`
     reservado); el puntaje es su promedio pesado;
   - los pesos van en `"pesos"` de `sugerencias_vivo.json`; por defecto solo CLAP, que mantiene el
     orden del punto 7;
   - cada lista queda en el registro con sus componentes, para ajustar los pesos después;
   - el género sale de `genre_pred.parquet`: el de Beatport en 1429 temas y, si falta, el del
     clasificador;
   - el botón por defecto pasa a ser D#0 + D0 también en la mezcla interna, donde D#0 es Modifier #1 y
     D0 no tiene función: el mismo gesto en las dos.
10. **El género se resuelve de forma automática, y la taxonomía la construye una red de agentes**
    (Gabriel, 2026-11-06).
    - Revisión manual, como mucho unas 20 tripletas.
    - Para los temas que no están en Beatport puede usarse un clasificador público, con un mapeo de sus
      etiquetas a las de Beatport. Un tema sin género puede quedar fuera del componente de género.
    - Beatport no publica similitudes entre géneros. La taxonomía propia de etiquetas de Beatport sale
      de taxonomías existentes (la de su tesis sobre mezcla armónica u otras más nuevas). Un grupo de
      agentes la modifica y otro la evalúa, en rondas, hasta que la den por terminada. La base es el
      texto de las etiquetas; también, propiedades musicales claras medidas en temas de cada género.

    Cómo (agente), informe `docs/reports/generos_beatport_2026-11-06.md`:
    - en el índice, el género es el de Beatport (también los 107 sin confirmar). Si no hay, el del
      clasificador con confianza >= 0.6 (173 temas). Si tampoco, ninguno (158): el componente no cuenta;
    - `beatport_subgenres.py` baja categoría, género y subgénero de la página de cada tema;
    - `genre_evidence.py` mide BPM, estilos de Discogs de MAEST, vecinos CLAP y géneros mezclados entre
      versiones, y arma un borrador de similitud, que es la entrada de la red de agentes.
11. **La taxonomía de la red de agentes da el componente `genero`** (agente, con el OK de Gabriel
    para lanzar la red, 2026-11-06). Informe: `docs/reports/taxonomia_generos_2026-11-06.md`.
    - La taxonomía de la tesis de Gabriel no apareció. Gabriel pidió usar otras conocidas, completas y
      al día: se usaron Wikidata, MusicBrainz, Discogs y la evidencia de la biblioteca.
    - La red tuvo un constructor, dos evaluadores por ronda (texto y audio) y un editor. Aprobó la
      taxonomía en la ronda 2 sin objeciones mayores; 17 menores quedan sin aplicar.
    - Resultado, en `config/genre_taxonomy/taxonomia.json`:
      - 34 familias y 211 relaciones;
      - regla: 0.75 (mismo género, otro subgénero), 0.7 (misma familia), 0.2 (familias hermanas) y 0
        (sin relación).
    - En `recomendador.py`, `genero` = `Taxonomy.compare`, con subgénero cuando Beatport lo da (30
      temas, índice versión 3).
      - «DJ Tools / Acapellas» cuenta como sin género, y «DJ Edits > X» como el género X.
      - Sin la taxonomía, el componente vuelve a igual o distinto.
    - El mapeo de Discogs (MAEST) a Beatport acierta el 24 % del género, contra el 38 % del
      clasificador propio. No se usa: los temas sin Beatport siguen con el clasificador (>= 0.6).
    - Los pesos siguen en solo CLAP. Antes de darle peso al género hay que llevar CLAP a una escala
      comparable: entre los 30 primeros, CLAP varía unos 0.05, y con un peso de género de 0.5 las 30
      sugerencias son del mismo género.

## 2026-11-04

1. **La Organización en el PC de escritorio, sin copiar la música** (Gabriel): del ordenador que
   tiene la biblioteca se traen solo los datos de la organización (`artifacts/v4/datasets/musica/`,
   0,4 GB, por el pendrive), y los temas salen del paquete que ya está en el escritorio
   (`C:\TRAKTOR ML`, o el pendrive), buscados por track_uid en su `_organizacion.csv`. No hacía falta
   copiar `Música/` (37,8 GB): el paquete ya tiene los 1760 temas de `biblioteca` v11. Cómo
   (agente): `review_app.py` usa el paquete cuando falta `Música/` y queda en solo lectura (ver y
   escuchar). Agregar música, fusionar, ordenar y preparar para Rekordbox/Traktor se hacen en la
   laptop, que manda, para no tener dos organizaciones que se separan. Falta que Gabriel diga si
   quiere hacer cambios también en el escritorio.
2. **En el escritorio, todo en el pendrive D:** (Gabriel: las descargas ya van a D:, y C: casi no tiene
   espacio). Los datos de la organización están en `D:\TRAKTOR ML datos\musica` (la copia verificada
   que trajo la laptop), y `artifacts/v4/datasets/musica` del repo es un junction a esa carpeta; la
   copia en C: se borró. La música sale de `D:\TRAKTOR ML`: la app busca el paquete primero en la
   unidad de los datos. Sin el pendrive, la pestaña Organización no abre. Temas nuevos: pueden
   quedarse en D:, porque analizar cada tema lo frena el cálculo y no la lectura del pendrive: el
   pendrive lee unos 28 MB/s (un MP3 de 10 MB, ~0,4 s), y solo la etiqueta Vocal tarda unos 13 s
   por tema. Pero analizarlos en el escritorio cambia la organización
   de ahí: antes hay que decidir qué copia manda (punto 1).
3. **Embeddings dentro de los archivos: se prueba antes de escribir** (Gabriel). Con su visto bueno se
   probó Traktor (conserva el embedding; `docs/reports/embeddings_en_los_archivos_2026-11-04.md`) y
   se hizo un respaldo de la música del paquete en el propio pendrive:
   `D:\TRAKTOR ML respaldo 2026-11-04` (1760 archivos, verificado). Todavía no se escribió nada en
   los archivos de verdad; [REEMPLAZADO 2026-11-04 por la prueba con Rekordbox 7.2.19 en la laptop:
   también conserva el embedding; falta que Gabriel decida si se escribe y qué va en cada archivo]
   falta probar Rekordbox y decidir qué se guarda en cada uno. [REEMPLAZADO 2026-11-04 por el punto 4:
   escrito en los 1752 MP3/FLAC del paquete.]
4. **La laptop deja de usarse; se trabaja solo en el escritorio, con el pendrive, y una copia en el NAS**
   (Gabriel). La copia del pendrive pasa a ser la principal [REEMPLAZA el «manda la laptop» del punto 1].
   El NAS (Synology, por Synology Drive) ya tiene la biblioteca completa en `SynologyDrive\Música`
   (4100 temas, 135 GB, agosto de 2025): están los 1823 temas del catálogo (1785 iguales en ruta y
   tamaño, 13 con unos KB de diferencia, seguramente tags, y los 25 de `_copias` en su ruta
   original), así que no hace falta copiar la `Música/` de la laptop. Gabriel configuró una tarea de
   copia de seguridad manual de Synology Drive para `D:\TRAKTOR ML` y `D:\TRAKTOR ML datos`: después de
   cada cambio grande hay que pulsar «Back up now». Con su visto bueno se
   escribió en cada MP3/FLAC del paquete su embedding `clap_full` y su lugar en `biblioteca` v11
   (`file_embeddings.py`): 1752 escritos y verificados (hash del audio igual antes y después, relectura
   idéntica), 0 errores; los 8 WAV/AIFF no se tocaron.
5. **La Organización se puede cambiar desde el escritorio, con la biblioteca en el pendrive** (Gabriel:
   «adelante»). Cómo (agente): `package_library.py adopt` pasó el catálogo a los archivos del paquete
   por track_uid (1760 temas; las 63 filas sin archivo, copias descartadas y temas sin embedding, quedaron
   en el respaldo `respaldo_package_library_*` junto con el catálogo anterior) y `Música/` es un junction a
   `D:\TRAKTOR ML\Musica`. «Preparar para Rekordbox y Traktor» y «Agregar música nueva» rehacen además
   las playlists del pendrive (`usb_package.py --refresh --from-org`) y el lugar guardado en cada archivo.
   Reglas: **un tema que cambia de playlist no se mueve de carpeta** (se actualizan las playlists; moverlo
   podría perder sus cue points), la música nueva entra en su propia carpeta dentro de `Musica\`, ante un
   empate de calidad entre copias repetidas se queda la que ya está organizada, y el origen de cada tema
   en `_organizacion.csv` se conserva (Preparar Traktor reconoce con él los temas ya analizados).
   Verificado: regenerar el pendrive con la organización sin cambios dejó `_organizacion.csv`, NML, m3u8
   y LEEME idénticos (en `rekordbox.xml` solo cambian los tamaños de los archivos que crecieron al guardar
   el embedding). Falta probar «Agregar música» con temas reales.

## 2026-11-02

1. **Temas del paquete que quedaron incompletos** (Gabriel pidió revisarlos y corregirlos; reglas del
   agente, escritas en las dos copias con respaldo): segunda búsqueda `package_tags.py rescue`
   (Deezer, iTunes, Beatport por título) solo para lo que falta, y el plan completa solo campos
   vacíos (Artist, Album, Genre, carátula). Beatport B sin confirmar (mismo tema, otra edición)
   completa Genre y Album, no Label ni Released. Sin artista en el archivo, el tema se acepta solo
   con título y duración exacta (+-0.5 s; Deezer, segundos enteros): con +-3 s un título común
   ("Magnetic") daba otro tema. Resultado: 81 de 236 encontrados; con carátula 1630/1760.
   Siguen sin fuente 123 temas sin carátula (bootlegs, rips, ediciones de DJ) y los 7 WAV (no se
   escriben: Traktor no lee sus tags).
2. **Una sola app con tres pestañas arriba a la izquierda** (Gabriel): «Descargas» (la app de
   descargas de siempre), «Colección» (nueva y chica: qué música forma la colección y cuál se elige
   para generar playlists) y «Organización» (la app web de carpetas y playlists por estilo,
   `src/v4/ui/review_app.py`). Cómo (agente): la ventana es la de `descargas_app.py`; la
   Organización corre en su propio proceso, lanzado al abrir la pestaña y mostrado dentro de ella,
   y se cierra con la app (después de la tarea en curso); `Abrir TRAKTOR ML.bat` sigue igual. La
   Colección son las carpetas de descargas, la biblioteca organizada (si está en ese ordenador) y
   las que se agregan a mano; se elige por subcarpeta y lo elegido queda en
   `%APPDATA%\traktor_descargas\collection.json`. Por ahora la elección solo se lleva a la
   organización carpeta por carpeta («Organizar» abre «Agregar música nueva» con ella); falta que
   Gabriel diga si la generación de playlists debe usar solo lo elegido.

## 2026-10-30

1. **El pendrive es para Rekordbox con controladora** (no para CDJ): el paquete portátil lleva
   `rekordbox.xml` y `Preparar Rekordbox.bat`, que lo rearma con la letra del pendrive en cada PC.
   Traktor sigue en el paquete. Motivo: las playlists de Traktor del pendrive no funcionaron en otra
   PC (rutas a `E:` y a `D:\MÚSICA` de la laptop) y Gabriel tuvo que tocar desde las carpetas.
2. **Metadata y carátula de todos los temas del paquete** (`package_tags.py`), en el pendrive y en
   `C:\TRAKTOR ML`, no en `Música/`. Gabriel vio los números del plan, eligió **Title = "Título
   (Mezcla)"** (sin "Original Mix"; la mezcla queda también en Remixers) y aprobó escribir en las dos
   copias. Reglas del agente, mostradas con el plan:
   - El título sale del archivo (limpio), no de Beatport (trae errores de tipeo).
   - Artist, Label y Released de Beatport solo donde faltan o tienen basura: un valor válido no se
     pisa (Beatport devolvía otras ediciones: compilados, otro sello).
   - Carátula donde falta o está rota: Beatport confirmado, Deezer verificado, o Beatport B sin
     confirmar (mismo artista, título y mezcla).
3. **Traktor en esta PC: las playlists se escriben en `collection.nml`** (`traktor_collection.py`,
   Traktor cerrado, respaldo), porque "Import Playlist" tomó una sola playlist y sin carátulas.
   Gabriel aprobó el plan, agregó `C:\TRAKTOR ML\Musica` a Track Collection desde Traktor y autorizó
   terminar el proceso de Traktor que quedó colgado al cerrar (la colección ya estaba guardada).

## 2026-10-05

1. **Etiqueta Vocal: clasificador lineal sobre CLAP del tema entero** (validado con Electrobyte y
   MTG-Jamendo; `docs/plans/20260929_deteccion_voz.md`), sin sobreingeniería: producto bueno y
   eficiente antes que el máximo rendimiento. Vocal si al menos el **5 %** del tema tiene voz (Gabriel;
   descartó el 25 % que salía de los datos).

## 2026-09-29

1. **Organización elegida: CLAP.** En la página de revisión, Gabriel eligió "claramente" la
   organización B frente a la A. B = CLAP + BPM (peso 0.3), Ward 15 carpetas / ~35 temas por
   playlist (config `fb78f2f6`, la de `exports/V4_1/`); A = MAEST-HF capa 7 con los mismos
   parámetros (`9e5784c3`). Salvedad: la comparación no fue del todo a ciegas (B ya la había
   escuchado y se reconocía por su número de playlists). No se exportó el CSV de veredictos.

2. **Metadatos de Beatport en los tags** (Artist, Remixers, Label, Genre, Released), con su
   convención: el remixer sale de Artist; Remixers = nombre de la mezcla si es un remix, "Extended
   Mix" si el archivo es la Extended, si no "Original Mix". El género de Beatport sobrescribe el
   que tenga el archivo.
3. **Si Beatport solo tiene otro remix, el tema va al modelo de género** (no hereda el género de
   otra versión: Beatport lo asigna por versión y por lanzamiento).
4. **Nada se escribe en los archivos de la colección antes de que Gabriel vea números y
   resultados.** Primero una tabla aparte (`features/beatport.parquet`) y el modelo entrenado con
   esos géneros; la escritura de tags queda para después de su visto bueno.
5. **Género del modelo solo con confianza** (clasificador >= 0.6: 171 temas). Donde no hay género
   confiable (confianza baja, gate MAEST sin validar, sin representaciones), el género que ya tiene
   el archivo se conserva (50 casos); si no tiene, queda vacío.
6. **Los matches B sin confirmar (110) no reciben datos de Beatport** y conservan el género que ya
   tienen (35 casos): Gabriel vio los ejemplos y lo juzgó razonable. La escritura de tags la corre
   Gabriel con `write_beatport_tags.py` (primero `--folder`), cuando el agente le avisa.
7. **WAV y AIFF no se escriben**: su track_uid incluye los tags y cambiaría (31 archivos).
8. **Temas repetidos: una sola copia por tema en las playlists** (Gabriel, escuchando los 24 grupos
   de `Música/_duplicados_para_escuchar.html`). 21 grupos son el mismo tema (1 a 10, 12 a 21 y el
   caso A, Abotha); sus 23 copias sobrantes salieron de `biblioteca` (v10, `dedupe.py apply`) y de
   los exports. [REEMPLAZADO 2026-09-29 por el punto 13: los cuatro pendientes quedaron resueltos]
   Pendientes: grupo 11 (Joris Voorn – Goodbye Fly, sin responder), caso B (Boris
   Brejcha: según sus tags son la Vocal Mix y la Original Mix), caso C (Markus Homm, otro corte) y
   «Enjoy The Silence … Cotton Dub» contra su «(boosted and cutted)», que apareció después.
9. **Qué copia se queda: siempre la MP3 de 320 kbps; si no hay, la de mejor calidad** (sin pérdida
   antes que comprimida; entre comprimidas, más kbps). Coincide con lo que eligió en los 10 grupos
   donde la calidad decide. [REEMPLAZADO 2026-09-29 por el punto 13: el empate se resuelve por
   ubicación] Empate de calidad: se pregunta. Sin confirmar: en los 7 empates eligió
   carpetas de año (2019, 2020…) antes que Nuevitas y Nuevitas antes que Milo.
10. **El mismo tema en otro corte (edit contra original) es un duplicado: queda uno.** Cuál, se pregunta.
11. **Música nueva:** lo idéntico o casi idéntico con calidad distinta se resuelve solo; lo dudoso
    (empate de calidad, mismo nombre, parecido, otro corte) lo pregunta la app. [REEMPLAZADO
    2026-09-29 por el punto 13: se resuelven solos «sonido» y «nombre»; lo demás queda como temas
    separados, sin preguntar] La pregunta en la app todavía no existe; por ahora, `dedupe.py candidates`.
12. **Las copias que no se quedan, además, se mueven a una carpeta aparte (`_copias`).** Antes de
    mover archivos hay que confirmar el alcance: las 23 de hoy o también las ≈ 616 copias idénticas
    que Phase 0 ya ocultaba (`duplicates.csv`). [REEMPLAZADO 2026-09-29 por el punto 13: se mueven
    solo las copias decididas]
13. **Temas repetidos, cierre: Gabriel dio el tema por cerrado y delegó lo pendiente** («lo mejor,
    menos molesto y reversible»). Lo que dijo él: el empate de calidad se resuelve por ubicación,
    **nunca Milo** (un amigo que copiaba sus temas elegidos y los bajaba con peor calidad) y primero
    las carpetas mejor clasificadas. Regla en `duplicates.location_rank`: carpetas por año, después el
    resto, después «copia», Milo al final; coincide con sus 8 elecciones en empates. Lo que decidió
    el agente por delegación:
    - Grupo 11: resuelto por la regla (queda `2020 old/old 6`).
    - Caso B: son distintos, porque sus tags dicen Vocal Mix y Original Mix.
    - Casos C y D: otro corte, queda la MP3 320.
    - Techyon / Egorythmia (0.972, tramos de un set continuo): son distintos.
    - En música nueva se resuelven solas las capas «sonido» y «nombre» (los 5 grupos por nombre que
      escuchó eran el mismo tema); «parecido» y «corte» quedan como temas separados, sin preguntar.
    - Se movieron a `Música/_copias/` las copias decididas, con su ruta de carpetas: 25 de 26, en
      `moved_copies.csv`. `dedupe.py restore-copies` las devuelve. La otra («Hide U» en
      `2020 new - copia`) se quedó en su lugar porque esa carpeta es `test_20`: nunca se mueven
      archivos de la carpeta de otro dataset.
    - Las ≈ 616 copias idénticas que ya ocultaba Phase 0 no se tocan.
    Resultado: `biblioteca` v11, 1760 temas.

## 2026-09-27

1. **La copia fiel es `main` de GitHub.** Lo que quedó en la laptop sin subir (4 commits locales de
   `feature/dj-clustering-v1`) se aparca: no se construye sobre eso ni se borra.
2. **MVP primero.** La urgencia es tener el pendrive de Rekordbox y las playlists de Traktor para
   tocar con la biblioteca que ya existe; la investigación del modelo de representación sigue
   después, sobre el mismo pipeline.
3. **Destino: Rekordbox y Traktor**, los dos.
4. **El humano cura, el modelo ordena y sugiere.** No se depende de historiales de lo tocado: la
   idea es que el modelo sugiera mejor de lo que Gabriel tocó antes. Hay algunos DJ sets (solo
   audio) reutilizables más adelante.
5. **El audio se compra.** "Me gusta" de Spotify tiene unos 2200 temas; quizá se descarguen solo
   los últimos ~500.
6. **Etiqueta "Vocal" en la metadata** de cada tema con voces, detectada con un modelo
   preentrenado.
7. La biblioteca se copia a `Música/` en la raíz del repo (git-ignorada). Las carpetas "old N" no
   tienen un significado fijo (a veces son copias de respaldo desordenadas): no se usan como señal.
8. **La carpeta `Música/Fede` no es de Gabriel**: se borró de la copia local y sale de catálogo,
   playlists y exports.
9. [REEMPLAZADO 2026-10-05 por la decisión 2026-10-05, punto 1: Vocal = clap_probe, al menos 5 % del
   tema con voz; desde 2026-11-04 también al agregar música con la app y `organize.py ingest`]
   **Umbral de la etiqueta Vocal: 0.145** (antes 0.15), para que "I Don't Feel Like Dancing" de
   Scissor Sisters quede dentro. Se quitó el atributo de solo lectura a los archivos que no se
   podían etiquetar (compilado "Best Of Toolroom 2015").
10. **El trabajo paralelo de Gabriel vive en `download_JIJIJI/`** y se incluye en los commits y
    pushes al repo, nunca los archivos de audio.
11. **Antes de cualquier corrida larga, una pasada por una carpeta chica** para verificar la salida
    (p. ej. `extract_representations.py --folder`); recién después la corrida completa.
12. Los pushes desde la laptop van con la cuenta `gbibbo` (credencial local del repo).

## 2026-09-12

1. **El corazón del proyecto es un modelo de representation learning** que permita encontrar
   temas similares en el espacio de embeddings. El producto final (carpetas, orden, recomendador)
   no está cerrado y se definirá con un MVP funcional; todo lo demás se construye encima del
   modelo. Prioridad: producir ese modelo con todas las fuentes de información disponibles y una
   estrategia de evaluación clara que permita transferirlo a distintas tareas.
2. **Evidencia humana: se usan los dos conjuntos de tripletas** (`tools/dj_feedback/answers/`):
   las 60 del anotador de septiembre (muestreo uniforme) y las 40 recuperadas de la rama
   `feature/dj-clustering-v1` (selección por kNN de MERT y por fronteras de V4_5, con sesgo
   documentado). No se prevé más anotación manual. Toda evaluación debe reportar el resultado
   por fuente y combinado.
3. **1001Tracklists es un requisito estricto**, fijado conscientemente: usar muchas playlists
   de 1001Tracklists como fuente de información para el modelo. Anula la decisión fija 7 del
   plan v6 ("evidencia débil, nunca feature en Régimen 1"). Implica resolver de dónde se
   descarga el audio de esos temas, cómo y dónde se guarda; esas cuestiones se consultan con
   Gabriel antes de implementar, no se descartan por dificultad.
4. [REEMPLAZADO 2026-11-09 por la decisión de Gabriel de esa fecha (algoritmo de recomendación, punto 7): lo
   liviano sigue en CPU local; lo pesado en procesamiento o en disco va a gpuq (Colab y Kaggle gratuitos) y a
   su Google Drive] **Cómputo**: CPU local por defecto; GPU solo mediante trabajos gratuitos de Kaggle, a
   configurar cuando haga falta. Surrey HPC y Lightning AI ya no existen.
5. **Base de código**: se continúa en `main`. La rama `feature/dj-clustering-v1` y `legacy/`
   son cantera de ideas (ver `docs/reports/scientific_review_2026-09-12.md`, sección 6).
