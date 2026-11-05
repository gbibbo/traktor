# Sugerencias en vivo en Traktor

Un botón del controlador MIDI muestra, dentro de Traktor, una playlist con temas sugeridos, sin tocar la
pantalla. Otro apretón devuelve el navegador a la carpeta donde estaba el DJ, con el mismo tema elegido.
Funciona mientras se toca: Traktor sigue usando el controlador como siempre.

**Estado (2026-11-05).**

- **Windows:** probado por Gabriel en Traktor Pro 4.4.1 con un Xone K2 (botón Ch15.Note.D0) y la
  DJM-750MK2 como placa de sonido.
  - Desde `#1 BIBO` y `#1 BIBO › PRO`, cada ida mostró la lista recién escrita, en 0,5 a 0,65 s.
  - Cada vuelta llegó a la carpeta con el mismo tema elegido, en 0,6 a 0,8 s.
  - La versión probada tiene la etiqueta `sugerencias-vivo-v1` en GitHub.
  - La versión para Windows y macOS (`sugerencias-vivo-v2`) se probó en Windows el mismo día, con el
    lanzador `.bat`: idas de 0,4 a 0,75 s, también desde el pendrive `D:`, y vueltas de 0,6 a 1,1 s,
    también a una carpeta cuatro niveles adentro (`#1 BIBO › PRO › #2 VA - Best Of Toolroom 2015`).
- **macOS:** el código y los tests ya cubren las rutas, el MIDI (python-rtmidi), el instalador y el
  permiso de administrador, pero todavía no se probó en un Mac con Traktor.
- **Los temas son de prueba:** salen al azar de la colección. El cálculo de las sugerencias, con la
  trayectoria del set, viene después.

## Cómo funciona

```mermaid
flowchart LR
    C[Controlador MIDI<br/>p. ej. Xone K2] -->|MIDI, compartido| T[Traktor]
    C -->|MIDI, compartido| P[sugerencias_vivo.py]
    P -->|escribe 10 temas| N[(TRAKTOR ML Sugerencias NML.nml)]
    N -->|Explorer lo lee al entrar| T
    P <-->|HTTP localhost:8080| S[Sonda QML en la pantalla<br/>del Kontrol D2 virtual]
    S -->|entra y sale de carpetas| T
```

1. **El sistema comparte el controlador.** En Windows (Windows MIDI Services) y en macOS (CoreMIDI),
   Traktor y `sugerencias_vivo.py` leen el mismo controlador a la vez. El programa solo reacciona al
   botón elegido, así que conviene que ese botón no tenga otra función en Traktor.
2. **La playlist es un archivo `.nml`.** El nodo Explorer de Traktor muestra como playlist los `.nml` de
   cualquier carpeta. Si el archivo cambia, Traktor lo vuelve a leer solo cuando se entra a la playlist
   **viniendo de otra carpeta**.
3. **La pantalla del Kontrol D2 mueve el mismo árbol que ve el DJ.** Traktor dibuja la pantalla del D2
   con QML, aunque el D2 sea virtual (agregado en el Controller Manager, sin el aparato). Su navegador
   (`Traktor.Browser`) es el mismo árbol de la ventana: entrar y salir ahí lo mueve en la ventana.
   Traktor no tiene un comando de "volver atrás", y esto lo reemplaza.
4. **El mod `traktor-api-client` conecta Traktor con el programa.** Es de ErikMinekus, con licencia MIT,
   y se copió en `tools/traktor_live/traktor_qml/`. Le agrega a esa pantalla un módulo que avisa por
   HTTP qué hay en cada deck. Nuestra sonda, `ApiBrowserProbe.qml`, recibe del programa órdenes para el
   navegador:
   - `report`: dónde está;
   - `goto`: ir a un camino, saliendo hasta la parte común y entrando carpeta por carpeta.
5. **Al apretar el botón:**
   - el programa pregunta dónde está el árbol y lo guarda;
   - escribe 10 temas nuevos en el `.nml`;
   - lleva el árbol a la playlist, pasando por su carpeta para que Traktor relea el archivo.

   Al apretarlo otra vez, vuelve a la carpeta guardada, y Traktor conserva el tema que estaba elegido
   en ella. Si una ida o una vuelta falla, el siguiente apretón la repite.

## Requisitos

- **Traktor Pro 4**, probado en 4.4.1. El instalador también encuentra Traktor Pro 3, pero no se probó.
- **Un Kontrol D2 agregado en Traktor:** Preferences > Controller Manager > Add… > Traktor > Kontrol D2.
  No hace falta tenerlo.
- **Un controlador MIDI con un botón libre** (nota o CC). Por defecto es el D0 del canal 15 del Xone K2
  de Gabriel, la nota 14, que antes cargaba la preescucha. En el Controller Manager hay que borrarle la
  función que tenga.
- **Python 3.11** en el `.venv` del repo, sin paquetes extra en Windows. En macOS hace falta
  `python-rtmidi`:

  ```
  .venv/bin/python -m pip install python-rtmidi==1.5.8
  ```

  Ese paquete también está en `requirements_v4.txt`, solo para macOS y Linux.

## Instalación (una vez, con Traktor cerrado)

**Windows:**

```
.\.venv\Scripts\python.exe tools\traktor_live\instalar_api_client.py --sonda
```

- Windows pide permiso de administrador (aviso de UAC).
- El resultado sale en la ventana nueva, que se cierra con Enter, y en
  `artifacts/v4/traktor_backups/instalar_api_client.log`.

**macOS:**

```
.venv/bin/python tools/traktor_live/instalar_api_client.py --sonda
```

- Si la carpeta de Traktor no es escribible, macOS pide la contraseña de administrador.
- Traktor se busca en `/Applications/Native Instruments/Traktor*/Traktor*.app/Contents/Resources/qml`.
- Modificar archivos dentro de la app es lo que indica el mod original para Mac, pero puede afectar la
  firma de la app. Si macOS no deja abrir Traktor, desinstalar con `--uninstall`.

**Qué hace el instalador en los dos sistemas:**

- Busca el Traktor más nuevo instalado. Con `--d2 <carpeta qml/CSI/D2>` se elige otro.
- Agrega dos líneas al `D2.qml` instalado, copia `Api/` y agrega la sonda a la pantalla del D2.
- Antes guarda una copia de lo que toca en `artifacts/v4/traktor_backups/`.
- Con Traktor abierto no cambia nada.
- `--status` muestra el estado y `--uninstall` deja todo como estaba.

Al actualizar Traktor, el instalador de NI puede reemplazar esos archivos: volver a correr esto.

## Uso

1. Encender la placa de sonido (la DJM). Ver *Problemas* abajo.
2. Abrir Traktor.
3. Abrir `Sugerencias en vivo.bat` (Windows) o `Sugerencias en vivo.command` (macOS) con doble clic, o
   correr `tools/traktor_live/sugerencias_vivo.py`.
4. La ventana tiene que quedar abierta mientras se toca.

Opciones; lo que se elige queda guardado para la próxima vez:

| Opción | Para qué |
| --- | --- |
| `--midi <palabra>` | Otro controlador: una palabra de su nombre (`--listar` muestra las entradas). Por defecto `K2`. |
| `--aprender` | Elegir otro botón apretándolo. Sirve una nota o un CC (valor >= 64). |
| `--n <N>` | Temas por lista (10). |
| `--dir <carpeta>` | Carpeta del `.nml`. |
| `--backend winmm\|rtmidi` | Motor MIDI. Por defecto `winmm` en Windows y `rtmidi` en el resto. |
| `--listar` | Mostrar las entradas MIDI y salir. |
| `--collection <archivo>` | Otra `collection.nml`. Por defecto la de `Documentos/Native Instruments/Traktor*`. |

**Si Traktor muestra la carpeta por otro camino** (por ejemplo en macOS, si el disco tiene otro
nombre), el programa avisa que no pudo llegar. En ese caso:

1. Navegar una vez a mano hasta la playlist `TRAKTOR ML Sugerencias NML`.
2. Apretar el botón.

El programa aprende ese camino y lo guarda.

## Dónde quedan las cosas

| Qué | Windows | macOS |
| --- | --- | --- |
| Playlist de sugerencias | `C:\TRAKTOR ML Sugerencias\` (raíz de la unidad del usuario) | `/Users/Shared/TRAKTOR ML Sugerencias/` |
| Ajustes (botón, controlador, camino aprendido) | `artifacts/v4/traktor_live/sugerencias_vivo.json` | igual |
| Registro de cada sesión (`.jsonl`) | `artifacts/v4/traktor_live/` | igual |
| Copias de los QML de Traktor | `artifacts/v4/traktor_backups/` | igual |

En Windows la playlist no va en la carpeta personal porque la pantalla del D2 vio vacía
`C:\Users\bibbo` (2026-11-05). En macOS va en `/Users/Shared` por la misma razón: así no pasa por la
carpeta personal.

## Problemas frecuentes

- **El controlador no hace nada en Traktor.** Si la placa de sonido no estaba encendida al abrir
  Traktor, Traktor cambia solo a la tarjeta interna, borra el ruteo y el mapeo deja de responder
  (2026-11-05). Pasos:
  1. Cerrar Traktor y encender la placa.
  2. Abrir Traktor.
  3. Revisar Preferences > Audio Setup: DJM-750MK2 ASIO, Master 3/4, Monitor 1/2.
- **«La pantalla del D2 no responde».** Falta algo de esto: Traktor abierto, el Kontrol D2 agregado en
  el Controller Manager, o la sonda instalada (`instalar_api_client.py --status`).
- **«No pude llegar».** El siguiente apretón lo reintenta. Si se repite, enseñarle el camino (ver *Uso*).
- **«El puerto 8080 ya lo usa otro programa».** Hay otra ventana de este programa o de
  `traktor_estado.py` abierta: cerrarla con Ctrl+C.
- **macOS: «falta python-rtmidi».** Instalarlo con el comando de *Requisitos*.

## Archivos

| Archivo | Qué hace |
| --- | --- |
| `tools/traktor_live/sugerencias_vivo.py` | El programa: botón, ida y vuelta, ajustes. |
| `tools/traktor_live/midi_io.py` | Entrada MIDI: winmm (Windows) o python-rtmidi (macOS). |
| `tools/traktor_live/traktor_estado.py` | Receptor HTTP del mod y de la sonda (también se usa solo para ver los decks). |
| `tools/traktor_live/traktor_qml/` | El mod `traktor-api-client` (MIT) y `Screens/ApiBrowserProbe.qml`, la sonda. |
| `tools/traktor_live/instalar_api_client.py` | Instala, revisa y desinstala el mod y la sonda en Windows y macOS. |
| `tools/traktor_live/midi_monitor.py` | Monitor MIDI para probar controladores. |
| `Sugerencias en vivo.bat` / `.command` | Lanzadores de doble clic. |
| `tests/v4/test_traktor_live_api.py`, `test_traktor_live_platforms.py` | Tests. Corren en Windows, macOS y Linux. |

La historia está en `docs/DECISIONS.md` (2026-11-05) y `docs/LESSONS_LEARNED.md`: cómo Traktor relee
un `.nml`, que el D2 mueve el árbol, y qué pasa si Traktor arranca sin la DJM.
