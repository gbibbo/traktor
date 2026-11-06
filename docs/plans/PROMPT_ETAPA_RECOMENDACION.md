# Prompt: etapa de recomendación musical en vivo

Para empezar esta etapa en un chat nuevo de Claude Code (o de otro agente que lea el repo). Lo escribió el
agente al cerrar la etapa de las sugerencias en vivo, el 2026-11-06, a pedido de Gabriel.

---

Vas a trabajar con Gabriel en la integración definitiva de la **recomendación musical en vivo** de
TRAKTOR ML. Respondé en español, con «tú», corto y con lo que sirve para avanzar.

## Qué ya existe y funciona

Lo verificó Gabriel tocando, en Windows, con Traktor Pro 4.4.1, el Xone K2 y la DJM-750MK2.

- **El botón D0 del K2:** lleva el navegador de Traktor a una playlist de 10 sugerencias y lo devuelve a
  la carpeta donde estaba, sin clics. Tarda 0,3 a 0,8 s de ida y 0,6 a 1,3 s de vuelta.
- **Arranque automático:** con Traktor (`tools/traktor_live/sugerencias_servicio.py`), sin ventanas y
  con notificaciones.
- **Las sugerencias son de prueba:** temas al azar de la colección (`Suggestions` en
  `tools/traktor_live/sugerencias_vivo.py`). Esta etapa las reemplaza.
- **Guía y mecanismo:** `docs/SUGERENCIAS_EN_VIVO.md`. Decisiones: `docs/DECISIONS.md` 2026-11-05 y
  2026-11-06. Estado: `docs/STATUS.md`.

## Lo que hay que definir con Gabriel ANTES de programar

Estas definiciones son de Gabriel. Proponé, mostrá datos y preguntá; no las decidas solo.

1. **Cuál es el tema principal que está sonando.**
   - **Con el mezclador interno de Traktor:** `updateChannel/<n>` trae `isOnAir` (volumen > 0) y
     `onAirLevel` (volumen por crossfader).
   - **Con la DJM (mezcla externa):** Traktor no ve los faders. La DJM manda sus faders de canal por
     MIDI, canal 1, CC 17 a 20 (0 a 127). Gabriel nunca usa el crossfader.
   - **Qué canal de la DJM corresponde a cada deck:** sale del ruteo externo de Traktor
     (`Routing.Out.External.*` en `Traktor Settings.tsi`). Confirmarlo con Gabriel.
   - **Valor de los faders al arrancar:** la DJM manda un fader solo cuando se mueve.
   - **Señales que se combinan:** `isPlaying` y `elapsedTime` de cada deck (`updateDeck`), nivel del
     fader, cuánto tiempo lleva al aire.
   - **Reglas a fijar:** histéresis, y qué pasa en una transición con dos temas al aire.
   - **El deck master de `updateMasterClock`** es el del tempo, no necesariamente el que suena.
2. **Registro riguroso de lo que suena:** todo lo que importa para recomendar.
   - Distinguir **cargado**, **preescuchado**, **sonando** (deck en play) y **al aire** (fader arriba).
   - Por tema: inicio y fin de reproducción, curva o resumen del volumen, tiempo al aire, tempo
     ajustado, tonalidad resultante, deck, si vino de una sugerencia (cuál lista, qué puesto).
   - Uno por sesión, en `artifacts/v4/traktor_live/` (ignorado por Git).
   - Hoy ya se anotan `deckLoaded` (con `filePath`), `updateDeck`, `updateChannel`,
     `updateMasterClock` y las pulsaciones `boton` con los temas sugeridos
     (`sugerencias_vivo_*.jsonl`).
   - Los History de Traktor (`Documentos/Native Instruments/Traktor 4.4.1/History/*.nml`) anotan lo
     **cargado**. `EXTENDEDDATA DURATION` dice cuánto corrió.
3. **Cómo se generan las sugerencias respecto del tema principal.**
   - La representación elegida es CLAP + BPM (peso 0.3), organización `biblioteca` v11; ver STATUS.
     El embedding `clap_full` y el lugar en la organización están escritos dentro de los MP3/FLAC del
     paquete (`file_embeddings.py`).
   - Compatibilidad de BPM y tonalidad (Camelot, como en la Phase 4).
   - La trayectoria del set: las ideas de Gabriel; preguntarle.
   - No repetir lo ya tocado. Cómo ordenar las 10.
   - Si se muestra un porcentaje o un motivo, y dónde. Traktor muestra los datos de su colección; ver
     `sugerencias_itunes.py` y `LESSONS_LEARNED`.
4. **Qué copia de cada tema se sugiere.** La colección de Traktor de este PC tiene temas en
   `C:\Música` y en el paquete (`C:\TRAKTOR ML`, `D:\TRAKTOR ML`). El paquete trae `_organizacion.csv`
   con archivo y track_uid. La sugerencia tiene que apuntar a un archivo que Traktor tenga en su
   colección.
5. **Cómo se evalúa.**
   - Offline con los History y los registros: qué tocó después de qué.
   - Ninguna representación supera al BPM de forma concluyente en las tripletas. Entre alternativas
     decide Gabriel escuchando (skill `auditar`), no un puntaje.

## Restricciones

- `AGENTS.md` manda: CPU local, nada de GPU paga ni Lightning sin permiso, nada de música ni
  embeddings en Git, commits verificados con tests y el CHECKPOINT al final.
- **No romper lo que anda:** el botón, la sonda, el servicio y los tests
  `tests/v4/test_traktor_live_*.py`, `test_sugerencias_servicio.py` y `test_mapeo_tsi.py`.
- **Latencia del botón:** que D0 siga por debajo de ~1 s. Calcular las sugerencias cuando cambia el
  tema principal, no al apretar.
- **Funcionar en Windows y macOS,** como el resto de `tools/traktor_live/`.
- **1001Tracklists:** si la recomendación lo necesita, preguntarle a Gabriel antes (DECISIONS
  2026-09-12).

## Tu primera respuesta

Un plan corto, sin código:
- tu propuesta para cada uno de los puntos 1 a 5;
- qué datos vas a mirar para fundamentarla (por ejemplo, los registros `.jsonl` existentes y los
  History);
- las preguntas concretas que Gabriel tiene que contestar.

Después, de a una unidad verificada por vez.
