# traktor-api-client (copia)

`Api/` es la carpeta `D2/Api` de [traktor-api-client](https://github.com/ErikMinekus/traktor-api-client)
de Erik Minekus (licencia MIT, `LICENSE`), sin cambios, del commit `1caf11f` (2025-07-26).

Es un módulo QML del controlador Kontrol D2: Traktor lo carga con el mapeo del D2 (un D2 virtual
agregado en Controller Manager si no hay uno conectado) y manda por HTTP a `http://localhost:8080`:

- `deckLoaded/<A-D>`: tema cargado (ruta, título, artista, BPM, tonalidad, tiempo, cues);
- `updateDeck/<A-D>`: play/pausa, tiempo transcurrido cada 1 s mientras suena, tempo, sync;
- `updateChannel/<1-4>`: al aire y nivel, según el volumen y el crossfader del mezclador **interno**;
- `updateMasterClock`: deck master y BPM.

No lee nada de la red: solo envía. La recibe `tools/traktor_live/traktor_estado.py`.

El original reemplaza `CSI/D2/D2.qml` por uno propio. Ese archivo es el de Traktor con dos líneas
más (`import "./Api"` y `ApiModule {}`), así que `instalar_api_client.py` agrega esas dos líneas al
`D2.qml` de la versión instalada de Traktor en vez de reemplazarlo, y guarda antes una copia. Funciona
en Windows (`Resources64/qml`) y macOS (`Traktor*.app/Contents/Resources/qml`).

`Screens/ApiBrowserProbe.qml` es nuestro (no viene del mod): la sonda que mueve el navegador de la
pantalla del D2, que es el árbol de Traktor, para las sugerencias en vivo
(`docs/SUGERENCIAS_EN_VIVO.md`). `instalar_api_client.py --sonda` la agrega al final de
`Screens/S8/Views/Screen.qml`.
