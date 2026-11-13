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

- Del 2-nov a las 17 UTC al 10-nov, Colab cortó toda VM T4 lanzada por el CLI a los 12-14 min de asignada, en gpuq y
  en el despachador de ERC-LM. No lo provocaba nuestro trabajo, ni la falta de GPU, ni el diseño de gpuq: el mismo
  cliente corría horas en octubre, y el 13-nov el patrón de gpuq volvió a durar 35 min con la GPU al 100 %.
- Colab sirve con gpuq tal como está. Igual, todo trabajo se retoma: parcial al Drive cada pocos minutos, nunca más
  de 10 sin guardar.
- El corte puede volver sin aviso. Se reconoce en `cola.py logs <id>`: dos intentos seguidos con
  `appears to be lost (404/401)` a los 12-15 min de empezados. Si pasa, cancelar y mandar con `--platform kaggle`, y
  anotar la fecha en el informe. Un `RuntimeError`, un `ReadTimeout` o un traceback son cortes de red, no ese corte.
- gpuq da por perdida una VM viva cuando una consulta falla por red (su marcador `"not found"` aparece en el
  traceback del CLI); la corrección espera el OK de Gabriel. Hasta entonces, un intento cortado así se relanza solo.
- Para volver a medir: `tools/gpuq/diag_colab/` (con el OK de Gabriel: usa su cuenta de Colab).

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
