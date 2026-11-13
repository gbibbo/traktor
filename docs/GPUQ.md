# gpuq y Drive en TRAKTOR ML

Cómo corre el trabajo pesado de este repositorio en las GPU gratuitas de Colab y Kaggle (gpuq) con los datos
en el Google Drive de trabajo de Gabriel. La regla y el procedimiento obligatorio están en `AGENTS.md`, sección
«Compute»; acá están las piezas, los comandos y la prueba que mostró que todo funciona.

## Las piezas (verificadas el 2026-11-09)

| Pieza | Dónde | Qué hace |
| --- | --- | --- |
| Despachador | Raspberry Pi `asr-rp5` de Gabriel, contenedor Docker `gpuq` | Lanza los trabajos en Colab (una T4 por vez) y Kaggle (unas 30 h de GPU por semana, 12 h por sesión), los relanza si se pierde la máquina y avisa al teléfono (ntfy) |
| Cola | repo privado `gbibbo/gpuq-personal` (rama `jobs`; estado y logs en `runs`) | `gpuq.toml` del despachador; clon en esta PC en `../gpuq-personal` (o `GPUQ_HOME`) |
| Código de los trabajos | este repo (público), en un commit ya subido | La VM lo baja con el token de GitHub del despachador |
| Almacenamiento | remoto de rclone `erclm` = Drive de trabajo de Gabriel, alcance `drive.file` | rclone solo ve lo que creó este mismo cliente de rclone, no el resto del Drive |
| Carpeta de TRAKTOR | `erclm:traktor_ml/` | Raíz que resuelve `src/v4/common/remote_storage.py` (en la VM y en esta PC) |
| Otras carpetas del remoto | `ERC-LM-2027/`, `gpuq-test/` | De otros proyectos: no se tocan |

`gpuq doctor` en la Raspberry dio `[ok]` en todo el 2026-11-09: token de GitHub, sesión de Colab, token de
Kaggle, dataset privado de credenciales en Kaggle, rclone, Hugging Face y ntfy. No hizo falta crear credenciales
nuevas: el token de GitHub del despachador lee este repo porque es público, y el Drive ya era el de Gabriel
(dueño de las carpetas según rclone y según el conector de Drive, por separado).

## Colab: leer antes de mandarle trabajo

Informe completo: `docs/reports/colab_corte_13min_2026-11-13.md`.

- Desde el 2-nov a las 17 UTC, Colab corta a los 12-14 min de asignada toda VM T4 que corre un trabajo real de gpuq,
  también en el despachador de ERC-LM, que en octubre corría horas con el mismo cliente. El 13-nov seguía:
  `traktor-a0-1` se cortó a los 12,2 y 12,6 min. No es la falta de GPU ni el esquema de consultas de gpuq: ese mismo
  día, un trabajo que solo usa la GPU vivió 35 min con ese esquema. Qué lo dispara, el arranque de gpuq o lo que hace
  el trabajo, lo dice la prueba E3 (`traktor-diag-e3-1`).
- Mientras tanto, en Colab cada intento rinde unos 10 min útiles. Entra lo que guarda un parcial al Drive cada pocos
  minutos, nunca más de 10 sin guardar. [REEMPLAZADO 2026-11-13 por Gabriel, «Colab primero» (sección «Kaggle»):
  un trabajo largo que guarda cada pocos minutos también va a Colab] Lo que necesita horas va con `--platform kaggle`.
- Se reconoce en `cola.py logs <id>`: `appears to be lost (404/401)` a los 12-13 min de cada intento. Un
  `RuntimeError`, un `ReadTimeout` o un traceback son cortes de red, y gpuq ya no los toma por VM perdida (corregido
  el 13-nov, despachador `a60c5ee`). El corte terminó cuando un intento real en Colab pase de 20 min: anotarlo en el
  informe.
- Para volver a medir: `tools/gpuq/diag_colab/` (con el OK de Gabriel: usa su cuenta de Colab y el único carril de
  Colab de la cola).
- [REFUTADO 2026-11-13 por `traktor-a0-1`, cortado a los 12 min ese mismo día: lo que duró 35 min fue un trabajo que
  solo usa la GPU] Del 2 al 10-nov Colab cortó las VM T4 a los 12-14 min, y el 13-nov el patrón de gpuq volvió a
  durar 35 min: Colab sirve con gpuq tal como está.

## Kaggle: la cuota, y Colab primero

Regla de Gabriel (2026-11-13): Colab primero, Kaggle de reserva, para no quedarse sin GPU gratis hasta que Kaggle
renueve la cuota.

- Kaggle da unas 30 h de GPU por semana para todos los proyectos de Gabriel: ERC-LM usa la misma cuenta. Colab no
  tiene cuota semanal; corta cada intento a los 12-13 min mientras dure el corte (sección «Colab»), y un trabajo
  que guarda cada pocos minutos igual avanza.
- Todo trabajo de GPU que guarda cada pocos minutos va con `--platform colab`. `auto` no prefiere Colab: toma el
  carril que se libere primero, así que gasta Kaggle cada vez que Colab está ocupado.
- `--platform kaggle` solo para lo que Colab no puede hacer: un paso que no puede guardar en menos de unos 10 min,
  más memoria de GPU que una T4 (Kaggle da dos) o Colab sin dar GPU durante horas. Los trabajos solo de CPU van a
  Kaggle y no gastan cuota de GPU.
- Antes de mandar trabajo de GPU a Kaggle, contar las horas de la semana, usadas y encoladas, y dejar libres al menos
  10 de las 30 para lo que solo Kaggle puede correr. Las usadas salen de los logs de los dos despachadores:
  ```
  # En la Raspberry:
  docker logs gpuq 2>&1 | grep -E '] (colab|kaggle): ' > gpuq.txt
  docker logs erclm-colab 2>&1 | grep -E '] (colab|kaggle): ' > erclm.txt
  # En esta PC, con los dos archivos copiados (la fecha es el comienzo de la semana de Kaggle):
  python tools/gpuq/diag_colab/duraciones.py gpuq.txt erclm.txt --lane kaggle --horas-desde "2026-11-08 00:00"
  ```
  Las encoladas: `cola.py status` y la duración medida de cada tipo de trabajo. El 13-nov a las 19:41 UTC iban 11,3 h
  desde el sábado 8-nov, más un trabajo corriendo. El día exacto en que Kaggle renueva la cuota lo muestra kaggle.com.

## Puesta en marcha en una PC nueva

1. `git clone https://github.com/gbibbo/gpuq-personal.git` al lado de este repo (o definir `GPUQ_HOME`).
   Con eso funcionan `tools/gpuq/cola.py` y `open_remote()`; gpuq no tiene dependencias.
2. `winget install Rclone.Rclone` (1.75 o más nueva; la Raspberry usa 1.75.2).
3. Gabriel, en su propia ventana de PowerShell, copia el remoto del despachador sin mostrarlo:
   `powershell -ExecutionPolicy Bypass -File tools\gpuq\rclone_desde_raspberry.ps1 -Pi gbibbo@<ip> -Key $HOME\.ssh\id_ed25519_rp5`.
   Desde el escritorio el nombre `asr-rp5.local` no resuelve: va la IP de la Raspberry en la red de la casa.
   El script con `-Prueba` muestra solo los nombres de las claves y no escribe nada.
4. Comprobar: `rclone lsf --max-depth 1 erclm:` lista `ERC-LM-2027/`, `gpuq-test/` y `traktor_ml/`.

## Comandos

```
python tools/gpuq/cola.py status
python tools/gpuq/cola.py add <id> --cmd "python <script> ..." [--gpu T4|CPU] [--platform colab|kaggle] [--no-setup] [--timeout-h N]
python tools/gpuq/cola.py logs <id>
python tools/gpuq/cola.py cancel <id>          (en Kaggle además «Cancel run» en kaggle.com)
rclone about erclm:                            (espacio del Drive)
rclone copy <carpeta local> erclm:traktor_ml/<tarea>/
```

`cola.py add` fija `--repo` en este repo y `--ref` en el commit actual, y se niega si ese commit no está en
GitHub. Los ids de trabajo no se reusan nunca. Sin `--setup` la VM no instala nada (no hay
`requirements.txt` en la raíz); Colab y Kaggle ya traen torch, numpy y PyYAML.

En el código de un trabajo:

```python
from src.v4.common.remote_storage import open_remote
remote = open_remote()                 # erclm:traktor_ml en la VM y en esta PC; TRAKTOR_REMOTE lo cambia
hechos = set(remote.ls("features/run1"))
remote.put("shard_00000.npz", "features/run1/shard_00000.npz")   # atómico, verificado por tamaño y MD5
```

Para probar en CPU sin tocar el Drive: `TRAKTOR_REMOTE=<carpeta local>`. La guía de gpuq para escribir
trabajos que se retoman es `docs/WRITING_JOBS.md` de su repo.

## Prueba de punta a punta (2026-11-09)

`tools/gpuq/smoke_drive.py`: la PC sube a `erclm:traktor_ml/smoke/<nombre>/input.json` un número al azar; el
trabajo de gpuq lo lee desde la VM, hace un producto de matrices en la GPU y escribe `result.json`; la PC lo
baja y comprueba que la VM devolvió el mismo número.

```
python tools/gpuq/smoke_drive.py prepare --run <nombre>
python tools/gpuq/cola.py add <id> --cmd "python tools/gpuq/smoke_drive.py run --run <nombre>" --gpu T4 --no-setup --timeout-h 1
python tools/gpuq/smoke_drive.py check --run <nombre>
```

Resultado del 2026-11-09, trabajo `traktor-smoke-1`, prueba `primera-prueba`, código en el commit `1f318fa`:

- La entrada subió desde esta PC a `erclm:traktor_ml/smoke/primera-prueba/input.json` (120 bytes).
- gpuq la tomó en Colab al primer intento: VM con Tesla T4 (15 GB), Python 3.13, torch 2.14.1+cu130; instaló
  rclone 1.75.2, bajó este repo en `1f318fa` y corrió el trabajo. De la cola a `done` tardó unos 3 minutos.
- La VM leyó el número de la PC, hizo un producto de matrices de 4096 × 4096 en la GPU (40,7 ms), escribió
  `result.json` (640 bytes) y lo releyó.
- `check` en esta PC: el número devuelto coincide, CUDA verdadero, la máquina es la T4. Por separado, el
  conector de Google Drive ve `result.json` con dueño la cuenta de trabajo de Gabriel y visibilidad privada.
- Drive marca ese archivo como `application/x-partial-download` porque gpuq lo sube con extensión `.part` y
  lo renombra; a rclone y a la verificación por MD5 no les afecta.
