# Colab corta a los 13 minutos: diagnóstico y cómo usarlo (2026-11-13)

Para los agentes que van a mandar trabajo a Colab por gpuq, y para Gabriel, que lo pidió el 2026-11-13 después de
que `traktor-v7-4` se cortara 11 veces en Colab. Sigue el protocolo `auditar`. Cada afirmación lleva su etiqueta:
EVIDENCIA OBSERVADA, INFERENCIA, HIPÓTESIS o NO CONCLUYENTE.

## Veredicto

El corte no lo provoca la falta de GPU ni el esquema de consultas de gpuq, y no es cosa de un trabajo nuestro en
particular. Empezó del lado de Colab el 2-nov a las 17 UTC: desde entonces Colab recupera, unos 12 a 14 min después
de asignada, toda VM T4 que corre un trabajo real de gpuq. Pasa en los dos despachadores de la Raspberry (gpuq, el de
este repo, y el de ERC-LM), que en octubre corrían en Colab trabajos de 1,5 a 4,3 h con el mismo CLI, el mismo diseño
y desde la misma Raspberry (con la cuenta de Colab de Gabriel según las notas de ERC-LM; la de gpuq no la miré, para
no leer credenciales). El 13-nov sigue: `traktor-a0-1` se cortó a los 12,2 y a los 12,6 min.

Ese mismo día, en cambio, tres pruebas que solo usan la GPU vivieron más de 30 min, una de ellas con el esquema exacto
de consultas de gpuq. Lo que dispara el corte está en algo que hacen los trabajos reales y las pruebas no: el arranque
de gpuq (subir las credenciales, instalar rclone, clonar el repo) o el trabajo en sí (bajar del Drive y de Hugging
Face, decodificar audio en la CPU). [REEMPLAZADO 2026-11-13 por E3, que vivió 35 min con el arranque completo de
gpuq: no es el arranque; es algo que hacen los trabajos, y E4 prueba por separado su tráfico al Drive y a Hugging
Face] La prueba E3, encolada con el OK de Gabriel, separa las dos cosas. Por qué Colab
lo hace no se puede saber desde acá, ni si es para todos o solo para esta cuenta. Mientras tanto, en Colab cada
intento rinde unos 10 min útiles: sirve para trabajos que guardan cada pocos minutos, y lo que necesita horas va a
Kaggle (sección «Cómo usar Colab»).

[REFUTADO 2026-11-13 por `traktor-a0-1`, cortado a los 12,2 y 12,6 min a las 18:37 y 18:50 UTC: ese día vivían las
pruebas sintéticas, no los trabajos reales] El 13-nov, tres pruebas con T4 sobrevivieron más de 30 min, incluida una
con el patrón exacto de gpuq: Colab volvió a servir tal como está.

De paso apareció un error de gpuq que cortaba intentos sanos: un corte de red pasajero en una consulta se leía como
«VM perdida» y gpuq liberaba una VM viva (sección «Un error de gpuq»). Corregido el 13-nov con el OK de Gabriel.

| Afirmación | Etiqueta | Evidencia |
| --- | --- | --- |
| Del 2-nov 17 UTC al 10-nov se cortaron 56 intentos T4 (mediana 13,4 min, máximo 21,6) y ninguno de más de 5 min terminó; los otros 4 fueron dos cancelados a mano y dos pruebas de 2 y 4 min | EVIDENCIA OBSERVADA | `docker logs` de `gpuq` y `erclm-colab`, `tools/gpuq/diag_colab/duraciones.py` |
| Del 19-oct al 2-nov, 14 de 21 intentos T4 terminaron tras más de 15 min (mediana 89 min, máximo 4 h 21 min) | EVIDENCIA OBSERVADA | ídem |
| Los dos despachadores usan `google-colab-cli` 0.7.4 y el mismo diseño (trabajo desprendido, una consulta corta por minuto) | EVIDENCIA OBSERVADA | `pip show` en los dos contenedores; `gpuq/dispatcher.py`; commit `5211e52` de `erclm-2027-dev` |
| El 13-nov el patrón de consultas de gpuq, con un trabajo que solo usa la GPU, sobrevivió 35 min | EVIDENCIA OBSERVADA | prueba E0, sección «Las pruebas del 13-nov» |
| El 13-nov los trabajos reales de gpuq se siguen cortando: `traktor-a0-1` a los 12,2 y 12,6 min, con el mensaje propio del CLI | EVIDENCIA OBSERVADA | `docker logs gpuq`; `cola.py logs traktor-a0-1` |
| El corte empezó del lado de Colab, no de nuestro código | INFERENCIA | mismo cliente con resultados opuestos antes y después del 2-nov; trabajos distintos de dos proyectos cortados igual |
| Lo dispara algo que hacen los trabajos reales y las pruebas no | INFERENCIA | el mismo día, 3 de 3 pruebas sintéticas vivieron más de 30 min y 2 de 2 intentos reales se cortaron a los 12 min; muestra chica |
| No es el arranque de gpuq (credenciales, rclone, clonado, sesión `gpuq`, consulta por minuto) | EVIDENCIA OBSERVADA | E3 (`traktor-diag-e3-1`) vivió 35 min por ese camino con la GPU al 100 % |
| Lo dispara el tráfico con el Drive por rclone, no las descargas de Hugging Face | EVIDENCIA OBSERVADA | E4-drive (4 tars reales, 1,3 GB en 9 min, y un parcial subido cada 3 min) se cortó a los 12,2 min; E4-hf (1,25 GB de Hugging Face) vivió 20 min. Un solo cambio frente a E3 en cada una |
| Qué parte del tráfico con el Drive: las bajadas grandes (rclone abre 4 conexiones para archivos de más de 256 MB) o las subidas chicas | NO CONCLUYENTE | lo deciden E5 (`traktor-diag-e5-bajadas-1`, `-1hilo-1`, `-subidas-1`) |
| Es una poda por inactividad | HIPÓTESIS, debilitada | coincide con los 12-13 min de las sesiones sin keep-alive de junio, pero E0 vivió con el kernel quieto y una consulta por minuto |
| Fue para todos o solo para esta cuenta | NO CONCLUYENTE | no hay reportes públicos; haría falta otra cuenta, y usar varias cuentas para esquivar límites va contra las reglas de Colab |

## Qué se observa

Las duraciones salen de `docker logs` de los dos despachadores, del par «starting X attempt N» → «X -> resultado».
Incluyen el minuto de limpieza posterior a la consulta fallida, así que la VM murió un poco antes.

| Período | Intentos T4 de Colab | Duración |
| --- | --- | --- |
| 19-oct al 2-nov a las 17 UTC (casi todo ERC-LM) | 21: 17 terminados (14 de más de 15 min) y 4 cortados | terminados de más de 15 min: mediana 89 min, máximo 4 h 21 min (`full-base-1`); cortados a los 13,3, 13,4, 49 y 58 min |
| Desde el 2-nov a las 17 UTC (ERC-LM el 2 y 3-nov, gpuq el 10-nov) | 60: 56 cortados, 2 cancelados a mano, 2 cortos terminados (2 y 4 min) | cortados: mediana 13,4 min, máximo 21,6 min; ninguno de más de 5 min terminó |

- El último trabajo largo que terminó en Colab antes del corte fue `prompt-hand_csl-1` de ERC-LM, el 30-oct a las
  12:19 UTC (91 min). El 2-nov, `gen-opro_llm_base-1` empezó a las 16:32 y se cortó a los 49 min; los 43 intentos
  siguientes de ERC-LM se cortaron todos, casi siempre entre 12 y 14,5 min (uno a los 2, uno a los 9 y dos a los
  21,6). El 10-nov, los 13 intentos largos de traktor (`traktor-v7-1` y `traktor-v7-4`) duraron entre 12,6 y
  15,7 min. Entre el 3 y el 10-nov nadie usó Colab con trabajos largos. El 13-nov, `traktor-a0-1` se cortó a los
  12,2 y 12,6 min (sección «Las pruebas del 13-nov»).
- En octubre también hubo cortes sueltos a los 13 min (`split-audio-2` el 22-oct; `pilot-base-1` el 19-oct, que
  ERC-LM atribuyó a haber liberado la VM a mano), pero eran la excepción: 2 de 21.
- En traktor cada intento tenía la T4: el trabajo dice «dispositivo cuda», y los latidos de gpuq muestran la GPU al
  100 % y 4,9 GB ocupados, 9 GB de RAM libres de 13 y 110 GB de disco libres hasta el minuto anterior al corte.
- El CLI informó siempre su propio mensaje, `[colab] Session 'gpuq' appears to be lost (404/401)`, y no un error de
  red. Los intentos siguientes recibieron una T4 nueva en menos de un minuto, sin `TooManyAssignmentsError`, y
  retomaron desde el último parcial del Drive sin avances de una VM huérfana: la VM anterior ya no existía.

## Las pruebas del 13-nov

Con el OK de Gabriel en el chat, desde el contenedor `gpuq` de la Raspberry, con sesiones de nombre propio
(`diag-e1`, `diag-e0`) y la cola de gpuq vacía. Mismo trabajo en todas: multiplicar matrices de 4096 × 4096 en la
T4 durante 35 min. Scripts en `tools/gpuq/diag_colab/`.

| Prueba | Qué cambia | VM asignada (UTC) | Resultado |
| --- | --- | --- | --- |
| E1 | El trabajo corre dentro de un solo `colab exec` largo, con la conexión abierta | 15:23:00 | Viva hasta el final: 35,4 min, 70 marcas sin huecos; seguía viva al leerla (38,4 min de encendida) |
| E0, primer intento | Iba a ser el control, pero un error mío en el script dejó el trabajo sin arrancar: VM con la GPU ociosa y una consulta corta por minuto | 15:58:38 | Viva 43 min, hasta que la liberé |
| E0 | El patrón de gpuq: trabajo desprendido con `Popen(start_new_session=True)` y una consulta corta por minuto | 17:38:12 | Viva hasta el final: el trabajo terminó sus 35 min con la GPU al 100 % y la VM siguió contestando hasta los 42,2 min de asignada; la liberé a las 18:21 |

- E0 tuvo tres consultas fallidas por red, con la VM en `asia-southeast1`: un `ReadTimeout` a los 2 min y dos
  `Connection was lost` en consultas que empezaron a los 25 y a los 33 min. La consulta siguiente contestó las tres
  veces, con el trabajo al día, y al final `colab sessions` no mostró ninguna sesión asignada. Con el primero, mi
  script dio la VM por perdida por el mismo error que tiene gpuq (sección siguiente); retomé las consultas sobre la
  misma sesión, sin liberarla, y el reloj de la prueba siguió.
- Colab gratis da una sola T4 por cuenta. Mientras E0 la tenía, `traktor-a0-prueba-1` de la línea A recibió `no_gpu`
  en Colab a las 17:45 y corrió en Kaggle; la cola estaba vacía cuando empecé, a las 17:38. Antes de una prueba fuera
  de gpuq, avisar a las otras sesiones.
- Lo que dicen: con un trabajo que solo usa la GPU, ninguno de los tres patrones dispara el corte, así que E1 no
  distingue nada.
- El mismo día, un trabajo real sí se cortó. `traktor-a0-1` (línea A: baja previews del Drive y los pasa por CLAP y
  MAEST, encolado por otra sesión) empezó a las 18:24, 4 min después de que liberé E0, y recibió
  `appears to be lost (404/401)` a las 18:36:39 y, en el intento 2, a las 18:49:12: 12,2 y 11,9 min después de
  empezar. Entre las pruebas y ese trabajo cambian dos cosas a la vez, el camino de arranque y lo que hace el trabajo.
- E3 (`tools/gpuq/diag_colab/carga_gpu.py`, trabajo `traktor-diag-e3-1`) deja fijo el trabajo sintético y lo manda
  por el camino completo de gpuq: subir credenciales, instalar rclone, clonar el repo, desprenderlo y consultarlo cada
  minuto. Si se corta a los 13 min, el disparador está en el arranque de gpuq; si vive, en lo que hacen los trabajos.
  Encolado el 13-nov a las 18:55 UTC con `--platform colab`; corre cuando le toque el turno. [REEMPLAZADO 2026-11-13
  por su resultado: corrió de 19:50:51 a 20:27:32, con la sesión `gpuq`, las credenciales subidas, rclone instalado
  y el repo clonado en `79cf199`; la GPU al 100 % y 11,4 GB de RAM libres en cada latido; terminó «E3 DONE» a los
  35 min. El arranque de gpuq no es]
- E4 (`carga_gpu.py --drive 6` y `--hf`, encolados el 13-nov a las 20:30 UTC) suma al trabajo sintético una cosa por
  vez de lo que hacen los trabajos reales de A0: bajar tars reales de `djsets_v1/audio/beatport` (unos 550 MB cada
  uno, uno cada 3 min) y subir un parcial cada 3 min, o bajar CLAP y MAEST de Hugging Face. CPU sola ya vivió más de
  15 min en las pruebas de ERC-LM de octubre.
- Resultado de E4, el 13-nov. `traktor-diag-e4-drive-1` empezó a las 21:08:15, bajó 4 tars (128, 530, 543 y 92 MB,
  de 19 a 31 s cada uno), subió los parciales y recibió `appears to be lost (404/401)` a las 21:20:25: 12,2 min, como
  los trabajos reales. `traktor-diag-e4-hf-1` bajó CLAP (589 MB en 5 s) y MAEST (663 MB en 7 s) y vivió los 20 min.
  El volumen es parecido; lo que cambia es el Drive por rclone. En E4-drive hubo además un `ReadTimeout` de red a los
  7 min y gpuq lo contó como una consulta fallida, sin liberar la VM: la corrección de los marcadores funcionó.
- E5 parte el tráfico con el Drive en tres pruebas de 20 min, encoladas a las 21:55 UTC: solo las bajadas
  (`--sin-subidas`), las mismas bajadas en un solo hilo (`RCLONE_MULTI_THREAD_STREAMS=0`) y solo las subidas
  (`--sin-bajadas`).
- Mientras la causa no esté, E2 (`run_diag.sh e2`, consultas largas con la conexión abierta) queda para después de E3.

## Un error de gpuq: un corte de red se lee como VM perdida

- `gpuq/dispatcher.py` da el intento por cortado en cuanto la salida de una consulta contiene `"appears to be lost"` o
  `"not found"` (`LOST_MARKERS`), y entonces corre `colab stop`. El despachador de ERC-LM tiene los mismos marcadores.
- Cuando el CLI falla por red, imprime un traceback que muestra el código fuente de las librerías, y una de esas
  líneas es `self.log.warning("Kernel not found at: %s", ...)` de `jupyter_kernel_client`. Pasó el 13-nov en E0, con
  un `ReadTimeout` de 10 s: la salida contenía «not found» y la VM estaba viva. EVIDENCIA OBSERVADA
  (`~/colab_diag/run_e0b.out` en la Raspberry).
- Con gpuq, ese intento se habría cortado y gpuq habría liberado una VM sana. Un trabajo que se retoma pierde
  solo el arranque (1 a 2 min y la carga del modelo), pero cada caso gasta uno de los 10 cortes que gpuq tolera por
  trabajo. INFERENCIA, por la lectura del código.
- En los registros de gpuq no hay ningún caso: todos los cortes de noviembre traen el mensaje propio del CLI.
- Corrección propuesta: dar la VM por perdida solo con la línea propia del CLI,
  `^\[colab\] Session '<nombre>' (appears to be lost|not found)`, y contar cualquier otra falla como una consulta
  fallida (6 seguidas cortan el intento, como ya hace gpuq). [REEMPLAZADO 2026-11-13 por el OK de Gabriel: hecho en
  gpuq `332546b` (`LOST_LINE` en `gpuq/dispatcher.py`, con un test que reproduce el traceback), traído a
  `gpuq-personal` en `a60c5ee`; el despachador de la Raspberry lo corre desde las 18:37 UTC] Es un cambio en el
  despachador, que sirve también a otros proyectos: espera el OK de Gabriel.
- El despachador de ERC-LM (`tools/pi_operator/dispatcher.py` de `erclm-2027-dev`) tiene el mismo error y no se tocó:
  es de otro proyecto.

## Qué no es

| Explicación | Por qué se descarta | Etiqueta |
| --- | --- | --- |
| Una VM sin GPU (la nota vieja de gpuq: «las VM de CPU duran poco, las T4 horas») | Todas las VM cortadas en noviembre tenían T4 y la usaban | EVIDENCIA OBSERVADA |
| Algo de nuestro trabajo (`vistas_previews.py`) | Los trabajos de ERC-LM, con otros modelos y otros datos, se cortaron igual; el corte cae a un tiempo fijo desde la asignación, haga lo que haga el trabajo en ese momento | EVIDENCIA OBSERVADA |
| Memoria, disco o CPU | Latidos normales hasta el minuto anterior; ERC-LM ya lo había descartado en octubre con una sonda cada 5 s | EVIDENCIA OBSERVADA |
| Un cambio del CLI | 0.7.4 fija en los dos contenedores desde octubre; el CLI solo avisa de versiones nuevas, no se actualiza solo (`auto_update.py`) | EVIDENCIA OBSERVADA |
| El vencimiento del token del proxy (issues 106 y 147 del CLI) | Vence a los 60 min y el CLI lo renueva desde la 0.7.3 | EVIDENCIA OBSERVADA |
| El diseño desprendido con una consulta por minuto | Corrió horas en octubre y 35 min el 13-nov | EVIDENCIA OBSERVADA |
| El error de los marcadores de gpuq | Los cortes de noviembre traen el mensaje propio del CLI, y el intento siguiente no vio avances de una VM viva | EVIDENCIA OBSERVADA |

## Lo que se sabe de cómo Colab decide

- Hasta la versión 0.7.2, el CLI mantenía un proceso que cada 60 s llamaba a `GET /tun/m/<endpoint>/keep-alive/`
  para «refrescar el temporizador de inactividad de la asignación». En junio, cuando ese ping fallaba, las sesiones
  del CLI «se podaban por inactividad en minutos» (issue 14 del CLI; un usuario midió que «un hueco de 10 a 15 min
  alcanza para perder el runtime», con nuestro mismo mensaje). El 25-sep, la 0.7.3 retiró el ping: «la vida de la VM
  la mantiene el backend de Colab según la actividad del kernel» (`docs/01_session_management.md` del CLI). Un
  mantenedor aclaró el 1-oct que Colab «depende sobre todo de ver conexiones vivas a la VM» y recomendó mantener la
  conexión abierta durante todo el trabajo (issue 160). EVIDENCIA OBSERVADA: código del CLI 0.6.0 a 0.8.0 y issues,
  leídos en esta sesión.
- Colab no publica sus límites y avisa que cambian (FAQ de Colab; `docs/HOW_IT_WORKS.md` de gpuq).
- El corte de noviembre se parece a la poda por inactividad de junio: mismo tiempo y mismo mensaje. [REEMPLAZADO
  2026-11-13 por E0, que vivió 42 min con el kernel quieto y una consulta por minuto, y `traktor-a0-1`, cortado el
  mismo día: si es una poda, Colab la aplica según algo que hacen los trabajos reales] HIPÓTESIS: durante esos días,
  Colab dejó de contar como actividad algo que el cliente sí hace en el resto del tiempo.
- Reinstalar el ping de keep-alive por nuestra cuenta queda descartado sin una decisión de Gabriel: Google lo sacó del
  CLI oficial, y la guía de gpuq pide averiguar por qué una plataforma corta en vez de buscarle la vuelta («Do not
  work around limits», `docs/HOW_IT_WORKS.md`).

## Cómo usar Colab

1. **Hoy Colab corta los trabajos reales de gpuq a los 12-13 min** (13-nov, `traktor-a0-1`). Cada intento rinde unos
   10 min útiles y gasta uno de los 10 cortes que gpuq tolera por trabajo (INFERENCIA, de los intentos de
   `traktor-v7-4`: arranque y carga de modelos de 1 a 2 min, más lo que quedó sin guardar).
2. **Colab primero** (Gabriel, 13-nov; `docs/GPUQ.md`, sección «Kaggle»): todo trabajo de GPU que guarda un parcial
   al Drive cada pocos minutos, nunca más de 10 sin guardar, va con `--platform colab`, aunque sea largo; así el corte
   cuesta minutos y no el trabajo. `auto` gasta Kaggle cada vez que Colab está ocupado. Kaggle queda para lo que
   Colab no puede hacer. [REEMPLAZADO 2026-11-13 por esa regla de Gabriel] **Lo que necesita horas va a Kaggle**
   (`--platform kaggle`). Con `--platform auto`, un trabajo cae en Colab cada vez que el carril se libera.
3. **Cómo reconocer el corte**: `python tools/gpuq/cola.py logs <id>` muestra, en cada intento,
   `poll failed 1/6: [colab] Session 'gpuq' appears to be lost (404/401)` cerca de los 12-13 min de empezado. Un
   `RuntimeError`, un `ReadTimeout` o un traceback son cortes de red: gpuq ya no los toma por VM perdida.
4. **Cómo saber que terminó**: un intento real en Colab que pasa de 20 min. Cuando pase, anotarlo en este informe y
   volver a `--platform auto` para todo.
5. **Para volver a medir** (pedirle el OK a Gabriel: usa su cuenta de Colab y hasta 1,5 h de T4 gratis, y le quita
   el carril de Colab a la cola mientras dura): `carga_gpu.py` por gpuq (E3) separa el arranque de gpuq del trabajo;
   `run_diag.sh e0` y `e1` miden fuera de gpuq; `run_diag.sh e2` prueba consultas largas con la conexión abierta.
6. **Para mirar el historial** de cortes de los dos despachadores: `tools/gpuq/diag_colab/duraciones.py` (instrucciones
   en su docstring).

## Pendiente

- [REEMPLAZADO 2026-11-13 por el OK de Gabriel: gpuq corregido y desplegado, y «Known issues» de
  `docs/HOW_IT_WORKS.md` de gpuq reescrito] GABRIEL: OK para corregir en gpuq los marcadores de VM perdida (sección
  «Un error de gpuq»); el mismo cambio sirve para el despachador de ERC-LM. Con eso también se corrige la nota de
  «Known issues» de `docs/HOW_IT_WORKS.md` de gpuq, que dice que las T4 duran horas y las VM de CPU poco: en noviembre
  las T4 duraron lo mismo que esas CPU.
- [REEMPLAZADO 2026-11-13 por E3, con el OK de Gabriel para usar el carril de Colab cuando quede libre] LOCAL: si el
  corte vuelve, correr E0 y E2 con su OK y completar este informe.
- [REEMPLAZADO 2026-11-13 por el resultado de E3: vivió, así que siguen las E4 con tráfico] LOCAL: leer el resultado
  de `traktor-diag-e3-1` (`cola.py logs traktor-diag-e3-1`) y completar este informe. Si se corta, partir el arranque
  de gpuq (credenciales, rclone, clonado) para ver cuál lo dispara; si vive, probar el trabajo sintético con tráfico
  al Drive.
- LOCAL: leer `traktor-diag-e4-drive-1` y `traktor-diag-e4-hf-1`. La que se corte a los 12-14 min señala el
  disparador; entonces se prueba una forma de hacer eso mismo que Colab no corte. Si viven las dos, probar las dos
  juntas y con decodificación en la CPU, como hace A0.
- [REEMPLAZADO 2026-11-13 por la regla de Gabriel «Colab primero»: la línea A pasó todo A0 a `--platform colab`, en
  trabajos de unos 25 min útiles] LÍNEA A: decidir si sus `traktor-a0` pendientes van a Kaggle mientras dure el corte
  (avisado el 13-nov).
- Hecho el 13-nov, con el OK de Gabriel: gpuq `6394eb5`, traído a `gpuq-personal` en `5684481` y en marcha desde las
  20:27 UTC. Un trabajo `auto` de GPU espera a Colab mientras está ocupado y toma Kaggle solo si Colab no puede
  correrlo (`kaggle.auto_gpu`).

## Reproducir

```
# En la Raspberry (asr-rp5): los cortes de cada despachador
docker logs gpuq 2>&1 | grep '] colab: ' > gpuq.txt
docker logs erclm-colab 2>&1 | grep '] colab: ' > erclm.txt
# En esta PC, con esos dos archivos copiados:
python tools/gpuq/diag_colab/duraciones.py gpuq.txt erclm.txt --cpu lane-test-colab,banks-v2-1,pilot-joint-1,diag-cpu-1,diag-up-cpu-1,diag-idle-cpu-1
# Las pruebas (con el OK de Gabriel), en la Raspberry, con la carpeta copiada a ~/colab_diag:
cd ~/colab_diag && nohup sh run_diag.sh all > run.out 2>&1 < /dev/null &
```

Fuentes externas: issues 14, 106, 147, 160 y 220 de `github.com/googlecolab/google-colab-cli` y su
`docs/01_session_management.md`; el código de `google-colab-cli` 0.6.0, 0.7.2, 0.7.3, 0.7.4, 0.7.5, 0.7.6 y 0.8.0
bajado de PyPI. Del lado de ERC-LM: `docs/NOTAS_infra.md` y el historial de `tools/pi_operator/dispatcher.py` en
`gbibbo/erclm-2027-dev`.
