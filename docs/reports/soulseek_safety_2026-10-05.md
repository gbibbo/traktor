# Revisión de tamaño y seguridad de Soulseek — 2026-10-05

**Veredicto:** el descargador original registraba `Size` pero no lo filtraba.
La descarga tampoco exigía los formatos del preflight: `--pref-format` era una
preferencia. Se agregaron controles antes de transferir y antes de incorporar
el archivo. Ninguno permite garantizar ausencia absoluta de malware.

| Afirmación | Etiqueta | Evidencia de esta sesión |
| --- | --- | --- |
| Antes no se comparaba tamaño con duración | EVIDENCIA OBSERVADA | `spotify_soulseek_download.py` en HEAD inicial `5b28780186f06e02e8d067402617876f50864f17`: `_availability_candidate_summary` conserva `Size`; `_classify_availability_raw` solo evalúa formato/bitrate; `download_with_sockseek` no evalúa tamaño |
| Las preferencias no rechazan otros formatos | EVIDENCIA OBSERVADA | Ayuda del `tools/sockseek/sockseek.exe` local y [documentación de Sockseek](https://github.com/fiso64/sockseek#file-conditions): `--pref-format` ordena; `--format` exige |
| El filtro nuevo rechaza 9 de 120 candidatos guardados | EVIDENCIA OBSERVADA | `_audio_size_error` sobre todos los `top_candidates` de `download_JIJIJI/soulseek_prueba_descarga/soulseek_availability.json`: 111 aceptados, 2 tamaños incompatibles, 7 tamaños/duraciones inválidos o ausentes; los 16 temas con candidatos conservan alguno |
| Defender funciona con el gate local | EVIDENCIA OBSERVADA | WAV estéreo generado de 1 s, 44100 Hz/16 bits: `_validate_download` completa el análisis con `MpCmdRun.exe` plataforma `4.18.26080.4-0`, sin detecciones, y valida duración real de 1 s |
| Los controles bloquean las amenazas que detectan | EVIDENCIA OBSERVADA | Tests offline: ejecutable disfrazado, tamaño recibido distinto, errores/detección/timeout del antivirus, falta de scanner, limpieza de payload rechazado y rechazo del MP3 final antes de promoción |
| Los archivos rechazados por tamaño eran virus | NO CONCLUYENTE | Solo se inspeccionaron metadatos guardados, no esos archivos; el tamaño puede deberse a carátulas, corrupción o metadata incorrecta |
| Todo archivo que pasa está libre de virus | NO CONCLUYENTE | El peer controla los metadatos y un antivirus puede omitir amenazas desconocidas; no se promete esta propiedad |

## Implementación y verificación

MP3: tamaño esperado = duración × bitrate / 8, bitrate de 315 a 325 kbps,
±15 % y 2 MiB extra para tags. Sin bitrate anunciado se usa 320 para este
filtro y se exige calidad real después. WAV/FLAC usan parámetros PCM, con un
rango más amplio para FLAC y para mono/estéreo no anunciado por Soulseek.
Tamaño máximo: 512 MiB. Sin tamaño/duración válida o parámetros PCM necesarios,
el candidato se rechaza. Los umbrales son heurísticos, no detectores de virus.

La transferencia solicita un archivo concreto mediante `slsk://`, aprobado en
el batch inicial. La CLI local confirmó, usando `--print jobs-full` sin red,
la extracción de archivos individuales y nombres escapados con espacios/`#`.
Cada intento tiene su carpeta temporal. Se comprueban nombre, cabecera, tamaño
anunciado/real, antivirus y datos reales de audio. Se conserva el nombre original
para mantener el matching posterior de artista, versión y duración.

La configuración temporal de Sockseek conserva solo el login global; elimina
hooks, perfiles y fallbacks heredados. Defender usa análisis personalizado con
`-DisableRemediation`, para que el código cero no represente un archivo que
había sido detectado y reparado. Véase [documentación de Microsoft](https://learn.microsoft.com/en-us/defender-endpoint/command-line-arguments-microsoft-defender-antivirus).
Linux admite ClamAV con límites explícitos y alerta si se exceden; esa ruta está
cubierta por construcción del comando, pero no se ejecutó ClamAV en esta máquina.

Validación de la primera corrección (`4c085bb`): `.venv/Scripts/python.exe` (Python 3.12.10, discrepancia ya registrada
en STATUS frente al 3.11 declarado), imports existentes correctos, **70 tests
passed, 1 skipped** en `test_soulseek_safety.py`,
`test_spotify_soulseek_orchestrator.py` y `test_status_doc.py`.
No hubo descargas reales, reproducción ni escritura en la colección.

El resultado sobre 20 temas utiliza un reporte previo, no una consulta actual
de disponibilidad. No se extrapola al resto de playlists. Los archivos pueden
llegar a la recepción temporal con contenido malicioso antes de ser analizados;
el objetivo implementado es impedir que los detectados lleguen a la biblioteca.

## Corrección de las cuatro vulnerabilidades adicionales

**Veredicto:** se cerraron las omisiones de consolidación, control durante la
transferencia, límites de conversión y validación de carátulas. La comparación
parte del código posterior al primer gate (`4c085bb`); estas correcciones se
verificaron sobre fixtures sintéticos, sin descargar música ni tocar la colección.

| Afirmación | Etiqueta | Evidencia de esta corrección |
| --- | --- | --- |
| Una etiqueta Spotify sola ya no permite consolidar un MP3 inválido | EVIDENCIA OBSERVADA | `_validate_final_mp3` reutiliza `_validate_download` antes de leer la identidad; tests con ID3 sin audio y fallo de antivirus bloquean el archivo y conservan el origen |
| Un duplicado inválido no provoca borrar la copia válida del run | EVIDENCIA OBSERVADA | `_collision_destination` revalida el destino; test de colisión con ID3 sin audio conserva ambos archivos |
| Un archivo fuente inválido bloquea los movimientos del lote | EVIDENCIA OBSERVADA | `consolidate_audio` valida todas las fuentes antes de crear la biblioteca; test con la segunda fuente inválida mantiene ambas en el run. No se afirma atomicidad ante errores posteriores en destinos |
| Se cancela una transferencia que supera los bytes anunciados mientras sigue activa | EVIDENCIA OBSERVADA | `_run_sockseek_transfer`; tests con procesos Python locales que escriben 2048 bytes frente a un límite de 1024, con proceso vivo y ya terminado; se espera su terminación antes de limpiar |
| Timeout y falta de disco impiden continuar la transferencia | EVIDENCIA OBSERVADA | Test con proceso local dormido y deadline reducido; test de espacio insuficiente comprueba que no se lanza el proceso |
| ffmpeg tiene límites y un MP3 truncado no se promociona | EVIDENCIA OBSERVADA | Timeout de 180 s, dos hilos, `-max_alloc`, formato explícito, protocolos locales, `-t` y `-fs`; tests de timeout/truncamiento mantienen vacía la salida |
| MIME falso, contenido truncado, animación y dimensiones excesivas de carátulas se rechazan | EVIDENCIA OBSERVADA | Tests con ejecutable disfrazado, PNG etiquetado JPEG, imágenes truncadas, APNG y cabecera PNG con dimensiones excesivas; antivirus precede al decoder |
| La imagen incorporada se reconstruye sin datos añadidos al final | EVIDENCIA OBSERVADA | `_validated_cover` verifica, carga, limita dimensiones y genera un JPEG nuevo sin metadata; test de payload añadido y dos análisis antivirus |
| La ruta válida completa funciona con herramientas reales | EVIDENCIA OBSERVADA | WAV estéreo sintético de 2 s + PNG 16×16; Pillow, ffmpeg 9.0.2 y Microsoft Defender reales; MP3 de 2.038 s a 320 kbps consolidado en directorio temporal, ID conservado |

La recepción se sondea cada 100 ms y puede exceder brevemente el límite antes
de cancelarse; no es una cuota de disco del sistema operativo. Se reservan 64 MiB
de espacio libre y se limita el log del descargador a 2 MiB. La asignación de
64 MiB de ffmpeg es **por bloque**, no un límite total de memoria; `-fs` puede
tener un pequeño exceso, y se comprueba además el límite final de 512 MiB.
Véase la [documentación de ffmpeg](https://ffmpeg.org/ffmpeg.html).
El protocolo permitido es `file`, según la
[documentación de protocolos](https://ffmpeg.org/ffmpeg-protocols.html).

Carátulas: máximo 15 MiB, 16 millones de píxeles y un fotograma; JPEG reconstruido
de hasta 2048×2048 y 2 MiB. `verify()` se complementa con una reapertura y `load()`;
la [documentación de Pillow](https://pillow.readthedocs.io/en/stable/reference/Image.html)
explica que abrir una imagen no decodifica sus píxeles. Se añadió `Pillow==12.3.0`
a las dependencias y se instaló únicamente ese paquete en el entorno existente.

Verificación final: **92 passed, sin omisiones**, en `test_soulseek_safety.py`,
`test_spotify_soulseek_orchestrator.py` y `test_status_doc.py`, con Python 3.12.10.
La ejecución dentro del sandbox dio 91 passed/1 skipped por no ver la instalación
WinGet de ffmpeg; al permitir acceso a esa instalación, también pasó esa prueba.
Las pruebas adversariales simulan detecciones del antivirus; el smoke válido
ejecutó Defender real. No se descargó ni ejecutó malware para verificar el gate.

Estos cambios reducen las vías observadas; no demuestran ausencia de todas las
vulnerabilidades de Sockseek, los decoders, sus dependencias o el antivirus.
