# Decisiones de proyecto

Registro fechado de decisiones de Gabriel que condicionan el trabajo. Cada entrada dice qué se
decidió y qué implica para el código. Las decisiones más recientes prevalecen sobre planes
anteriores (`v4_implementation_plan.md`, `dj_music_clustering_deterministic_implementation_plan_v6.md`).
Una decisión revertida o reemplazada no se borra: se marca en su lugar con
`[REEMPLAZADO <fecha> por <decisión>: <lo que rige ahora>]` (o `[REFUTADO …]` si era un error).

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
     scripts de Python. La ruta propuesta es que Gabriel pida acceso de investigación a
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
4. **Cómputo**: CPU local por defecto; GPU solo mediante trabajos gratuitos de Kaggle, a
   configurar cuando haga falta. Surrey HPC y Lightning AI ya no existen.
5. **Base de código**: se continúa en `main`. La rama `feature/dj-clustering-v1` y `legacy/`
   son cantera de ideas (ver `docs/reports/scientific_review_2026-09-12.md`, sección 6).
