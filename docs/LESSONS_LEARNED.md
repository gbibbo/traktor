# TRAKTOR ML — Lessons Learned

## Clustering (HDBSCAN / PCA)

### PCA pre-HDBSCAN es necesario para embeddings de alta dimensión
**Contexto:** MERT produce embeddings de 1024 dims. HDBSCAN no estima densidad correctamente en espacios de alta dimensión (curse of dimensionality). Con N=239 tracks y HDBSCAN directamente en 1024D, los mejores resultados eran 2 clusters con 31% noise, o 3 clusters con 80% noise.
**Solución:** Aplicar PCA (sklearn) antes de HDBSCAN. Con pca_dim=50 se retiene 93.7% de varianza y HDBSCAN produce 8 clusters en `test_20` (N=239).
[REEMPLAZADO 2026-09-29 por DECISIONS 2026-09-29: la organización vigente es CLAP + BPM (peso 0.3) con Ward; en CLAP, HDBSCAN deja casi todo como ruido (V4_USAGE, sección 0b). Estos parámetros valen solo para MERT + HDBSCAN.] **Parámetros óptimos para test_20 (N=239):** `pca_dim=50, l1_min_cluster_size=6, l1_min_samples=1` → 8 clusters L1, ~51% noise, tamaños [33,24,20,11,8,8,7,6].
**Nota:** Con N~239 no es posible obtener <30% noise con ≥3 clusters simultáneamente. El límite de ruido mejora con más tracks. `mert_full` como L1 no ofrece ventaja sobre `mert_perc`.

### Separabilidad del espacio MERT-v1-330M con test_20
Para N≤300 tracks de techno/tech house, esperar noise rates de 40-60% en L1. Esto es estructural del dataset pequeño, no un bug. Los tracks asignados (~116 de 239) sí tienen estructura musical real.

## Tags de audio

### Guardar un ID3 v2.3 une con «/» los campos que traían varios valores
**Contexto (2026-09-29):** al escribir los tags de Beatport, mutagen conserva la versión ID3 del archivo.
Un archivo v2.3 («Len Faki - BX 3») traía un título con dos valores ("BX 3", "BX 3 (original mix)"), algo
que v2.3 no admite; al guardarlo quedó "BX 3/BX 3 (original mix)". Fue 1 de 1087 archivos escritos: los
demás campos no cambiaron (verificado contra el catálogo).
**Regla:** después de escribir tags, comparar todos los campos contra el catálogo, no solo los escritos.

## Detección de voz

### Silero VAD no detecta canto sobre la mezcla completa
**Contexto (2026-09-29):** Silero VAD v5.1.2 (TorchScript, 16 kHz) sobre el tema entero, en 63 temas de
`musica`: en las carpetas «Vocal» de Gabriel (23 temas, todos con voz) encontró voz en >= 1 % del tema
solo en el 13 % y nunca llegó al 5 %; CLAP zero-shot marcó el 87 %. Es un detector de habla entrenado
para ignorar música, y el canto sobre la base cuenta como música.
**Regla:** no usarlo sobre la mezcla. Solo tendría sentido sobre la voz separada (Demucs).

### CLAP zero-shot sobre 90 s del medio pierde voces
**Contexto (2026-09-29):** con 3 tramos de 30 s del medio (≈ 20 % de un tema de 7 min) el detector marcó
55 % de 20 temas con «feat.»; con ventanas de 10 s cada 20 s sobre todo el tema, 75 %, y en 20 temas
Dub/Instrumental bajó de 30 % a 25 %. Muestras chicas (etiquetas débiles: «feat.» no garantiza voz
audible y un dub puede tener recortes de voz). Ningún umbral sobre los 90 s arreglaba las dos cosas.
**Regla:** para la voz, mirar el tema entero antes de ajustar el umbral.

### Al cambiar un método, cambiar también a quien lo llama
**Contexto (2026-11-04):** la etiqueta Vocal pasó a `clap_probe` el 2026-10-05, pero «Agregar música
nueva» de la app y `organize.py ingest` seguían llamando `tag_vocals.py --method clap --write-tags`,
que recalcula y escribe toda la biblioteca: el primer agregado le habría quitado la marca a 473 temas
y puesto a 94. **Regla:** al reemplazar un método, buscar todas sus llamadas (`grep -rn` en `src/`,
`tools/`, `docs/`) y fijar el método vigente en un test.

### «2020 new - copia» no es un respaldo: es la carpeta de test_20
**Contexto (2026-09-29):** `data/raw_audio/test_20` es un enlace (junction) a `Música/2020 new - copia`.
Las reglas que ven «copia» como respaldo (Phase 0 al elegir entre archivos idénticos, `location_rank`)
la ponen última. Al mover a `_copias` una copia repetida de esa carpeta, test_20 perdió un tema
(`test_block1_common` esperaba 243). **Regla:** nunca mover ni borrar archivos de la carpeta de otro
dataset. `dedupe.move_copies` las protege con `protected_roots`.

## Descargas (`download_JIJIJI/`)

### Sockseek limita las búsquedas: un tope fijo de preflight rompe las playlists largas

Sockseek hace como máximo 34 búsquedas cada 220 s (Soulseek banea 30 minutos si se excede;
`--searches-per-time`, `--searches-renew-time`). El preflight lanza una búsqueda por tema en un
único proceso, así que 152 temas necesitan al menos 4 esperas de 220 s (880 s). Con el tope fijo
de 600 s, los dos intentos del 2026-10-30 agotaron el tiempo, perdieron todo el lote y los 152
temas quedaron como error. El tope ahora se calcula con `preflight_timeout_for(n)`. Un preflight
que falla para todos los temas es un problema de la búsqueda: no se manda la playlist a grabar.

### `--strict-title` de sockseek pide el título de Spotify al pie de la letra

En una playlist de 152 temas (2026-11-02), 29 de los 46 «no disponibles» estaban en Soulseek con la
versión correcta: `--strict-title` exigía «Título - Versión» (como lo escribe Spotify) dentro del
nombre del archivo y perdía «Título (Versión)», los nombres scene con guiones bajos, etc. Contraste
de una sola variable sobre 17 de esos temas: con la configuración real, 2 con resultados; sin
`--strict-title`, 16; con 15 s de búsqueda en vez de 5, 1; los 3 de control aparecieron siempre.
Buscar por el título base no agregó ninguno. El título ahora lo juzga el descargador con las reglas
de la verificación (`_availability_title_error`): la búsqueda nueva dio 132 disponibles contra 106.
Ojo al leer el log de sockseek: «N results found» cuenta lo que pasa sus filtros, no lo que llegó.

### Probar en Soulseek mientras baja una descarga: otra cuenta y `--no-config`

Una segunda sesión con la misma cuenta corta la otra, y el descargador abre una por transferencia.
Para buscar en paralelo hace falta otra cuenta (se crea sola al conectarse con un nombre nuevo),
pasada con `--no-config --user=… --pass=…` y otro `--listen-port` (el 49998 lo usa la descarga).
Con `--config=<archivo>`, sockseek usó igual la cuenta de la configuración global; no se sabe por
qué, así que la configuración «segura» del descargador puede no aislarlo de esa configuración. Una
contraseña que empieza con «-» tiene que ir como `--pass=-…`.

## Windows local

### El Python de la Microsoft Store virtualiza %APPDATA%

Lo que ese Python escribe en `%APPDATA%` (token `traktor_spotify_token.json`, ajustes de la app de
descargas) queda en `%LOCALAPPDATA%\Packages\PythonSoftwareFoundation.Python.3.11_*\LocalCache\Roaming\`.
Los procesos de ese Python lo ven en `%APPDATA%`; PowerShell y el Explorador, no. Al buscar esos
archivos fuera de Python, hay que mirar en la ruta virtualizada. Lo que se copia a `%APPDATA%` desde
fuera (`sockseek.conf`) sí se lee, porque la lectura cae a la carpeta real.

### Un Job Object no agrupa a los hijos del Python de la Store

`AssignProcessToJobObject` no captura a los procesos que lanza ese Python (ya corre dentro de su
propio contenedor), y al detener el orquestador quedó un `sockseek.exe` huérfano con la sesión de
Soulseek abierta. La app de descargas sigue el árbol por PID padre y guarda la hora de creación de
cada proceso para no cerrar uno que reutilizó el PID (`ProcessTree` en `descargas_app.py`).

### Las playlists de Rekordbox y Traktor guardan la letra de la unidad

`traktor.nml` (VOLUME="E:") y `rekordbox.xml` (file://localhost/E:/...) llevan la ruta absoluta con la
letra. El pendrive cambia de letra entre computadoras y un NML preparado en la laptop (pendrive en
`E:`, temas locales en `D:\MÚSICA`) dio todos los temas como faltantes en otra PC (2026-10-30). Las
playlists del paquete se rearman en cada PC con su `.bat` (`usb_package.py`), o se copia el paquete a
`C:\TRAKTOR ML` y se prepara desde ahí.

### Traktor 4.4: "Import Playlist" toma una sola playlist y no lee carátulas

Un NML con una carpeta y 51 playlists entró como una sola playlist (la primera), y sus temas sin
carátula: Traktor usa los datos del NML y solo lee la carátula del archivo (a su caché propia,
`Coverart/`) al agregarlo desde el disco. Lo que funciona: arrastrar la carpeta de música a Track
Collection y escribir las playlists en `collection.nml` con Traktor cerrado (`traktor_collection.py`).
Traktor a veces queda abierto sin ventana al cerrarse y sin "LOG CLOSE" en `Logs/Traktor.log`: una
vez no guardó la colección; otra la guardó y siguió colgado. Mirar la hora de `collection.nml`.

### Preparar Traktor reconoce los temas por su nombre y carpeta originales

`traktor_prepare.ps1` busca en la colección local de Traktor cada tema del paquete por `nombre_original` y
`carpeta_original` de `_organizacion.csv`. Al regenerar el paquete hay que conservar esas columnas (y `origen`):
si pasan a la ruta del paquete, Traktor deja de reconocer los temas que ya analizó.

### Python no abre rutas de más de 260 caracteres en esta PC

Sin LongPathsEnabled, `Path.exists()` da False en silencio para rutas largas: un paquete probado
desde una carpeta de prueba profunda "perdió" 705 temas. Probar con la misma longitud de ruta que el
destino real (`subst R: <carpeta>` deja `R:\TRAKTOR ML`, como `D:\TRAKTOR ML`).

### El nodo iTunes de Traktor puede duplicar entradas de la colección

Traktor busca cada tema del XML de iTunes en su colección por la ruta. Los que reconoce los muestra
con los datos de la colección (cues, grid, BPM). Los que no reconoce los muestra con los datos del
XML (BPM 0.00, sin carátula), y al cargarlos en un deck agrega al final de `collection.nml` una
**segunda entrada con la misma ruta**, con lo que trae el XML (comentario, duración, contador de
reproducciones vacío). La entrada original queda intacta. Un "#" en la ruta (%23 en el XML) basta
para que no lo reconozca (prueba del 2026-11-04). Para revisar después de una prueba, contar las
entradas por ruta (`entry_key`), no solo las rutas distintas: un dict por ruta esconde el duplicado.

### Traktor relee un .nml del Explorer solo al entrar desde otra carpeta; el D2 mueve el árbol

El Explorer de Traktor 4.4.1 muestra como playlist los `.nml` de cualquier carpeta (los `.m3u8` no).
Si el archivo cambia, sigue mostrando lo que leyó hasta que se entra a la playlist viniendo de otra
carpeta: volver a elegir el mismo favorito no alcanza (prueba de Gabriel, 2026-11-04). Traktor no
tiene comando para volver a la posición anterior del árbol, pero el navegador de la pantalla del
Kontrol D2 (QML `Traktor.Browser`, también con un D2 virtual) es el de la ventana: `enterNode` y
`exitNode` mueven el árbol que ve el DJ y `currentPath` dice dónde está (`ApiBrowserProbe.qml`).
Entrar a un **tema** desde esa pantalla lo carga en un deck: la sonda solo entra a carpetas.
Esa pantalla vio vacía `C:\Users\bibbo` durante 6 s y tardó más de 2 s en dar los nombres de
`C:\Users` (2026-11-05): las sugerencias van en la raíz de la unidad y el "goto" espera cada nivel.

### Si Traktor arranca sin la DJM, cambia solo el audio y el K2 deja de responder

El 2026-11-05 la DJM-750MK2 no estaba conectada al abrir Traktor: Traktor pasó solo de
`DJM-750MK2 ASIO` a la tarjeta interna (`Senary Audio`), borró el ruteo (Master 3/4, Monitor 1/2) y el
mapeo del K2 no hacía nada, aunque el K2 mandaba MIDI y el puerto no estaba tomado. Primero conectar y
encender la DJM, después abrir Traktor. Para volver: Preferences > Audio Setup, o copiar las entradas
`Audio.*` y `Routing.*Out.Internal.*` de un respaldo en `Backup\Settings\` con Traktor cerrado.

## Entorno HPC (ver también memory/MEMORY.md)

[REEMPLAZADO 2026-09-12 por DECISIONS 2026-09-12, punto 4: Surrey HPC ya no existe; se trabaja en CPU local (AGENTS.md). Se conserva como registro.]

- El env `traktor_ml` de conda debe crearse con Python 3.11 desde `/user/HS300/gb0048/anaconda3/`.
- Login node (`datamove1`) sí tiene acceso a internet (puede hacer pip install, git pull).
- El comando `source /user/HS300/gb0048/anaconda3/etc/profile.d/conda.sh` es necesario antes de `conda activate`.
- Siempre correr scripts con `PYTHONPATH=/mnt/fast/.../traktor` desde el repo root.
