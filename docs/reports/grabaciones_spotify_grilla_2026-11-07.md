# Temas grabados de Spotify sin grilla fija en Traktor, 2026-11-07

Pedido de Gabriel (2026-11-07, tarde): los temas grabados de Spotify no tienen grilla fija en Traktor,
que calcula un BPM que no parece constante a lo largo del tema. Además, los grabados tienen que quedar
recortados como los de Soulseek (los mismos milisegundos antes del primer beat), para que Traktor ponga
solo el primer beat sin errores. Diagnóstico con la skill `auditar`, sobre la playlist «Workout Delft»
(`7hZfN5NxLTgWNAC2NAfRcM`, 10 temas). Los que se graban son Unsociable Hours, Oye, Amor y That Civic Has A
Laptop. Gabriel se fue y pidió que el agente siguiera solo: el agente lanzó las descargas y decidió los
arreglos. Falta la verificación en Traktor, que es de Gabriel.

## Veredicto

1. **La causa: el grabador perdía audio.** Leía el loopback de Voicemeeter con lecturas bloqueantes y un
   buffer de 42,7 ms. Cada vez que el bucle de lectura se atrasaba más que eso, porque la PC estaba ocupada
   con el embedding, Soulseek o el MP3 anterior, se perdía audio sin ningún aviso. Cada pérdida adelanta
   todos los beats que siguen: una grilla constante calza en un tramo del tema y queda corrida en el
   siguiente. En la corrida A (el grabador de antes, instrumentado) se perdieron 0,15 a 1,1 s por tema, en
   escalones. Spotify, mientras tanto, avanzaba a velocidad normal. Ese faltante explica el BPM
   «no entero» que Traktor dio a los grabados: con el audio perdido repuesto, el tempo vuelve a ser el
   entero del tema.
2. **El arreglo: capturar en modo callback.** El hilo de PortAudio entrega cada bloque a una cola, y el
   bucle puede atrasarse sin perder nada. En la corrida B, con el arreglo y con pausas del bucle de hasta
   94 ms, el audio recibido quedó a ±3 ms del reloj en 5 a 6 minutos. El tempo salió entero sin corregir
   nada y la fase de los beats contra una grilla fija quedó plana.
3. **Tres errores más en el corte, arreglados.**
   - El comienzo del primer tema se tomaba de audio que ya sonaba al abrir el grabador (`Started: 0.000 s`).
   - En tandas de varios temas, cada tema siguiente empezaba en «comienzo del anterior + duración».
   - Una falla del detector dejaba de 0 a 21 ms antes del primer sonido (o cortaba el arranque) en vez de 20.
4. **El criterio de Soulseek no es un número fijo.** Es el silencio que trae cada master: en 70 descargas,
   el 31 % suena antes de 5 ms y la mediana es 30 ms, pero un cuarto trae más de 0,2 s. Una grabación no
   puede saberlo, porque se mezcla con el silencio antes de que Spotify arranque. Ahora cada tema grabado
   empieza 20 ms antes de su primer sonido, como la mayoría de los archivos de Soulseek que arrancan con
   sonido.
5. **NO CONCLUYENTE hasta que Gabriel cargue los temas en Traktor:** las mediciones de abajo son de tempo
   y fase. Lo que dice el análisis de Traktor sobre los archivos nuevos todavía no se vio.

## Afirmaciones

| Afirmación | Etiqueta | Evidencia |
| :--- | :--- | :--- |
| Traktor dio BPM enteros a los 7 de Soulseek y no enteros a los 3 grabados de la corrida de Gabriel | EVIDENCIA OBSERVADA | `collection.nml` de Traktor 4.4.1, carpeta `nueva descarga/Workout Delft`: Soulseek 142,9998, 144,99997, 148,0001, 140,00005, 142,99997, 143,9999 y 146,0002; grabados 142,6867, 144,1297 y 147,1989 |
| La lectura bloqueante tiene un buffer de 42,7 ms y una pausa más larga pierde audio | EVIDENCIA OBSERVADA | Sonda sobre Voicemeeter Input. Pausa de 60 ms: se pierden 17 a 40 ms. 100 ms: 56 a 78. 250 ms: 206 a 235. 1 s: 957 ms |
| PortAudio no avisa esa pérdida en modo bloqueante | EVIDENCIA OBSERVADA | Con `exception_on_overflow=True`, 0 errores con pausas de hasta 2,5 s |
| Un buffer bloqueante de 2 s no sirve | EVIDENCIA OBSERVADA | Con `frames_per_buffer=48000` no pierde con pausas de 1 s. Casi lleno (pausa de 1,9 s) entrega 280.576 frames en 4,01 s, unos 70.000 por segundo de un stream de 48 kHz, y no se recupera |
| El modo callback no pierde audio con el bucle atrasado | EVIDENCIA OBSERVADA | Con el bucle dormido 0,1, 1, 3 y 6 s, o con Python ocupado 1 y 3 s, el audio recibido queda a −13/+9 ms del reloj (un bloque) y sin avisos de pérdida |
| En la corrida A el grabador perdió audio en escalones, durante las pausas del bucle | EVIDENCIA OBSERVADA | `capture_timing.csv`. Unsociable Hours: +351 ms, el mayor escalón (+166 ms) en la pausa de 187 ms. Oye: +1.132 ms, con +318 ms en una pausa de 343 ms |
| Spotify no se frenó: lo perdido es de la captura | EVIDENCIA OBSERVADA | `spotify_progress.csv`. Unsociable Hours: Spotify avanzó 297,44 s en 297,49 s de reloj y se recibieron 297,09 s. Oye: 339,01 s contra 338,01 s. That Civic: 288,95 s contra 288,78 s |
| La fase de los beats baja en los mismos escalones que la pérdida | EVIDENCIA OBSERVADA | Unsociable Hours de la corrida A contra una grilla de 142 BPM, por tramos de 15 s: −34 contra +28 ms, −101 contra +107 ms, −156 contra +166 ms. En total −350 ms de fase y +351 ms perdidos |
| Medido contra el original de Soulseek, Amor de la corrida A perdió 146 ms en 9 escalones | EVIDENCIA OBSERVADA | `align.py`, 336 de 346 ventanas alineadas. Escalones de 7, 27, 7, 17, 27, 27, 7, 17 y 7 ms. El registro de captura daba unos 155 ms |
| That Civic de la corrida A perdió unos 200 ms | EVIDENCIA OBSERVADA, más débil | `align.py`: +203 ms en 7 escalones, pero solo 142 de 248 ventanas alineadas (el loop se repite y la alineación se engancha en una repetición). El registro de captura daba de 257 a 273 ms |
| Con el audio perdido repuesto, el tempo vuelve al del original | EVIDENCIA OBSERVADA | `explain.py`, grabación contra repuesta. Unsociable 142,163 → 142,003. Oye 147,091 → 147,003. That Civic 146,084 → 145,994 (el original de Soulseek da 145,977). Amor 144,026 → 143,962 (el original da 143,960). Error del estimador sobre archivos limpios: unos ±0,05 BPM |
| Con el arreglo (corrida B) no se pierde audio, con las mismas pausas | EVIDENCIA OBSERVADA | Unsociable: +3 ms en 308 s, 10 s con pausas de más de 42,7 ms (máximo 94 ms), 0 avisos. Oye: −2 ms en 347 s, 7 s con pausas (máximo 94 ms), 0 avisos. Con lectura bloqueante, una pausa de 60 a 100 ms pierde de 17 a 78 ms (sonda) |
| Con el arreglo, el tempo sale entero y la fase queda plana | EVIDENCIA OBSERVADA | Corrida B. Unsociable: 142,003 BPM y fase de −125,5 a −125,9 ms a lo largo de 5 minutos. Oye: 147,003 BPM, 9 ms de deriva y coherencia 0,319, igual a la de A con lo perdido repuesto (0,317) |
| En las corridas de Gabriel, la pérdida fue probablemente mayor que en A | INFERENCIA | En sus logs, el reloj avanzó de 4 a 21 s más que el audio recibido por tema. La resolución es de 1 s y no hay `spotify_progress.csv` que separe una pérdida de una pausa de la salida. Había más carga a la vez (Soulseek y embedding) |
| El comienzo se tomaba de audio que ya sonaba | EVIDENCIA OBSERVADA | Corrida A, tercer intento: 2,3 s de otro audio, 2,9 s de silencio y recién el tema a los 5,2 s. El grabador tomó 0,000 s, el MP3 empezó 5 s antes y el corte final cayó 4,5 s antes del final. En los logs de Gabriel aparecen `Started: 0.000 s` y `0.277 s` |
| En una tanda de varios temas, el comienzo del tema siguiente queda corrido | EVIDENCIA OBSERVADA | Amor de la corrida A, tercero de su tanda, empieza 232 ms antes que el original de Soulseek (`align.py`) |
| El detector de comienzo caía al borde del bloque | EVIDENCIA OBSERVADA | Oye de la corrida B: comienzo 5,226666 s = 245 × 1024 frames, primer sonido a 6,8 ms del MP3. Reproducido: pedía 60 ms de ventanas RMS cuando había 64 ms de sonido, devolvía `None` y caía al borde. Con el arreglo, 20,6 ms en 5 posiciones distintas dentro del bloque |
| Con el detector final, el tema grabado empieza como uno de Soulseek que arranca con sonido | EVIDENCIA OBSERVADA | Corrida C, Oye: comienzo 4,649 s (fuera del borde de bloque) y primer sonido a 24,9 ms del MP3, es decir 20 ms más la decodificación, como los 22,9 ms de Unsociable Hours. 147,003 BPM, 6 ms de deriva, +0 ms de pérdida en 346 s y 0 avisos |
| Soulseek no tiene un adelanto fijo | EVIDENCIA OBSERVADA | Primer sonido (−60 dBFS) en 70 descargas: percentil 10 a 0 ms, percentil 25 a 0,4, mediana 30,5, percentil 75 a 222,6 y percentil 90 a 390,8. El AutoGrid de Traktor cae de 21 a 62 ms en los que arrancan con bombo, y a 257 y 525 ms en los que traen silencio propio |
| El ecualizador de Spotify está activado | EVIDENCIA OBSERVADA | `audio.equalizer_v2=true` en el `prefs` de Spotify. Las bandas no figuran ahí |
| Las grabaciones salen con ecualización | HIPÓTESIS | Solo si alguna banda no está en 0 dB. Se decide mirando Spotify → Configuración → Reproducción → Ecualizador |
| Otro dispositivo de la cuenta corta la grabación | INFERENCIA | A las 18:43:48 la cuenta pasó a un tema que no está en la playlist (`3togPmlAC31YEN0HUZ60Hl`) y se pausó: 2 intentos cortados. La otra sesión del agente no llamó a Spotify en ese momento. Hay un televisor «Comedor» en `/me/player/devices` |
| No quedan más grabaciones en la colección | EVIDENCIA OBSERVADA | MP3 con `Spotify Track ID` y 48 kHz en `D:\nueva descarga`, `D:\TRAKTOR ML\Musica`, `C:\TRAKTOR ML`, `D:\TRAKTOR ML datos`, `Música/` y `download_JIJIJI/runs`: además de los de hoy, solo 2 de Jorge Ben Jor (run del 30-10) |

## Qué se cambió

- `playlist_loopback_recorder.py`:
  - `LoopbackCapture`: el loopback se abre con `stream_callback` y los bloques van a una cola. Una
    pérdida que PortAudio avise igual se cuenta por tema (`capture_overruns` en el checkpoint) y el log
    dice «AVISO: se perdió audio al grabar…».
  - `StartDetector`: el comienzo es 20 ms (`START_LEAD_SEC`) antes de la primera muestra que suena después
    de 0,3 s de silencio (o sin datos), buscada desde el bloque en que terminó el silencio.
  - `capture_timing.csv` en cada tanda: una fila por segundo, con el reloj, el audio recibido, la cola y
    la pausa más larga del bucle. `JIJIJI_KEEP_WAV=1` conserva el WAV de la sesión.
- `spotify_soulseek_orchestrator.py`: `SESSION_SIZE = 1`, un tema por tanda. Cada tema empieza después de
  un silencio, y el final se corta en «comienzo + duración», dentro del silencio de la repetición.
- `spotify_playlist.py`, en el commit 8b12ab7 de la otra sesión: el watchdog escribe `spotify_progress.csv`.
- Tests: `CaptureTests` (callback, pérdidas, comienzo después de silencio, adelanto fijo dentro del
  bloque, sin datos) y una tanda por tema. Los 4 tests de flujo que suponían tandas de 10 lo fijan.
- Herramientas de auditoría en `tools/recording_audit/`: `beat_phase.py`, `staircase.py`, `explain.py`,
  `align.py` y `lead_in.py`.

## Resultado en la carpeta

`D:\nueva descarga\Workout Delft` tiene los 10 temas: 8 de Soulseek (esta vez bajaron Amor y That Civic)
y 2 grabados con el arreglo, Unsociable Hours (corrida B) y Oye (corrida C, con el detector final). Las
grabaciones de las corridas A y B y los registros quedan en
`download_JIJIJI/runs/diagnostico_grilla_2026-11-07/`, que Git ignora.

## Para Gabriel

1. **Traktor guarda la grilla de los grabados que borraste** (mismas rutas: Unsociable 142,687 BPM, Oye
   147,199, Amor 144,130 con la grilla en 22 ms). Seleccioná los 10 temas de la carpeta, botón derecho →
   Analyze (Async), con la grilla y el BPM sin bloquear. Mirá si la grilla queda fija en Unsociable Hours
   y Oye.
2. Mirá el ecualizador de Spotify (Configuración → Reproducción → Ecualizador): para grabar, apagado.
3. Mientras graba, no uses Spotify en otro dispositivo de la misma cuenta (el teléfono, el televisor del
   comedor): corta la grabación.
