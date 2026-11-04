# La pestaña Organización en el PC de escritorio (2026-11-04)

Medido en la laptop (`C:\VS\traktor`) el 2026-11-04. El escritorio tiene 20 GB libres en C: y 89 GB
en el pendrive D:; su Python del sistema tiene pandas, PyYAML, scikit-learn y umap (dato del agente
del escritorio, no verificado desde acá). La decisión es de Gabriel: este informe no copia nada.

**Verificado en el escritorio (2026-11-04, agente del escritorio):**

- El repo está en `C:\VS projects\traktor_github`, no en `C:\VS\traktor` (esa carpeta no existe aquí).
- El Python del sistema (3.11.9) tiene todo lo de abrir y de agregar música: pandas 2.3.1,
  pyarrow 21, PyYAML 6.0.3, scikit-learn 1.7.2, umap 0.5.11, mutagen, joblib, torch 2.8.0+cpu,
  torchaudio, soundfile, transformers 4.57.0 y scipy 1.16.1. El `.venv` no tiene ninguno; la app
  usa el del sistema para la Organización (7b268ab).
- No están el modelo CLAP (no aparece en la caché de Hugging Face) ni `models/vocal_probe/`.
- El pendrive D: es FAT32 (114,6 GB, 89,3 GB libres). Un junction desde C: hacia una carpeta del
  pendrive funciona para leer (probado con `D:\TRAKTOR ML\Musica` y quitado después). Escribir a
  través del junction no se probó.

## Qué hay en la laptop

`Música/`: **37,8 GB**, 2440 archivos de audio (2415 sin contar `_copias/`), 1823 temas en el catálogo
(616 archivos son copias idénticas que el catálogo no repite) y 1760 en `biblioteca` v11.

| Subcarpeta | Tamaño | Subcarpeta | Tamaño |
|---|---|---|---|
| `#1 BIBO` | 16,8 GB | `2021` | 1,3 GB |
| `2019` | 6,8 GB | `2024` | 1,1 GB |
| `2020 new - copia` (= test_20) | 4,1 GB | `2025` | 0,6 GB |
| `2020 new` | 2,6 GB | `_copias` | 0,5 GB |
| `2020 old` | 1,9 GB | `Temp Descarga` | 0,3 GB |
| `2021 sin clasificar (…)` | 1,5 GB | `2022 sin clasificar`, `Temp Descarga 2` | 0,3 GB |

`artifacts/v4/datasets/musica/`: **0,40 GB**.

| Carpeta o archivo | Tamaño | Abrir | Agregar música |
|---|---|---|---|
| `representations/clap_wholetrack/` (caché) | 318 MB | no | sí (etiqueta Vocal) |
| `representations/clap_full/` (embeddings + caché) | 58 MB | fusiones | sí |
| `representations/maesthf_*` | 23 MB | no | no |
| `exports/` (listas y XML; las rutas apuntan a `Música/`) | 7 MB | no (se escribe) | no |
| `orgs/` | 5,5 MB | sí | sí |
| `features/` (`bpm_key.parquet`, `vocals_clap_probe.parquet`) | 5,1 MB | `bpm_key` | sí |
| `catalog.parquet`, `duplicates.csv`, `duplicate_decisions.json`, `moved_copies.csv` | 0,6 MB | sí | sí |
| `clustering/`, `evaluation/`, `logs/`, `*.log`, `catalog_before_*` | 1 MB | no | no |

Fuera de `artifacts/`, para agregar música: `models/vocal_probe/clap_probe.npz` (12 KB, ignorado por
Git) y el modelo CLAP `laion/clap-htsat-unfused` (1,2 GB en la caché de Hugging Face; se baja solo si
hay internet). Sin la caché de `clap_wholetrack`, la etiqueta Vocal recalcula los 1823 temas (unos
13 s por tema: 6-7 h).

## Qué necesita `review_app.py`

- **Abrir** (ver, escuchar, fusiones, orden a mano, deshacer, preparar para Rekordbox/Traktor):
  `Música/` (tiene que existir: sin ella la pestaña muestra el aviso) y lo marcado «sí» arriba.
  Python: pandas, pyarrow, PyYAML, scikit-learn, umap-learn, mutagen.
- **Agregar música nueva**, además: las dos cachés CLAP, `features/`, la sonda de voz y el modelo
  CLAP. Python: además torch, torchaudio, soundfile, transformers y scipy.
  [REFUTADO 2026-11-04 por la verificación en el escritorio: el Python del sistema del escritorio
  los tiene todos (versiones arriba).] No consta que el Python del escritorio los tenga.
- Lo simple es copiar `artifacts/v4/datasets/musica/` entero (0,4 GB, entra en C:).
- `catalog.parquet` guarda rutas absolutas (`C:\VS\traktor\Música\...`) en `source_path`. Las usan
  la etiqueta Vocal, `phase1_tags`, los exports (m3u8, Rekordbox, Traktor) y `beatport_lookup`; la
  app lee y reproduce con `Música/` del repo y `rel_path`.
  [REFUTADO 2026-11-04 por la verificación en el escritorio: allí el repo está en
  `C:\VS projects\traktor_github` y `C:\VS\traktor` no existe. Hacen falta las dos rutas:
  `<repo>\Música` para la app y `C:\VS\traktor\Música` para `source_path`, como carpetas o
  junctions.] En el escritorio el repo está en la misma ruta (`CONFIGURAR_OTRO_ORDENADOR.md`):
  sirve si `C:\VS\traktor\Música` existe, como carpeta o como junction.

## Opciones

[REFUTADO 2026-11-04 por un recuento en el escritorio: las opciones 2 y 3 parten de que en el
escritorio falta la música, y no falta. Los 1760 temas de `biblioteca` ya están en
`D:\TRAKTOR ML\Musica` y en `C:\TRAKTOR ML\Musica`, ordenados por playlist, y
`_organizacion.csv` da el archivo y el track_uid de cada uno (los 1760 existen en las dos). No hay
que copiar `Música/` (37,8 GB): basta con traer `artifacts/v4/datasets/musica/` (0,4 GB) y hacer que
la app lea los temas del paquete con ese índice. Ese cambio en la app falta hacerlo [REEMPLAZADO
2026-11-04 por el modo paquete de `review_app.py`: hecho, y en el escritorio abre `biblioteca` v11 con
1760 de 1760 temas, en solo lectura (DECISIONS 2026-11-04)]. Agregar música
nueva en el escritorio sigue necesitando decidir dónde se guarda y qué copia manda.]

1. **Solo en la laptop.** El escritorio descarga; la música nueva se lleva a la laptop y se agrega
   allí. No se copia nada; en el escritorio la pestaña Organización muestra el aviso. Hay una sola
   organización, así que no hay que decidir cuál manda.
2. **Una copia en el escritorio.** `Música/` en `D:\Música` (37,8 GB de 89 libres, junto al paquete
   de 27 GB) con un junction `C:\VS\traktor\Música` → `D:\Música`; `artifacts/v4/datasets/musica/` y
   `models/vocal_probe/` en C:. Funciona mientras el pendrive esté conectado con la misma letra.
   [REFUTADO 2026-11-04 por la verificación en el escritorio: un solo junction no alcanza, porque allí
   el repo está en `C:\VS projects\traktor_github`. Hacen falta dos hacia `D:\Música`:
   `C:\VS projects\traktor_github\Música` y `C:\VS\traktor\Música`. Lo demás sigue igual. El modelo
   CLAP (1,2 GB) se baja la primera vez que se agrega música.]
3. **Una sola copia en el pendrive para los dos.** `Música/` y `artifacts/v4/datasets/musica/` en D:,
   con junctions en los dos ordenadores. No hay dos versiones, pero la laptop pasa a depender del
   pendrive (velocidad USB, y si se pierde, se pierde la organización: hace falta un respaldo).

**Con dos copias (opción 2), agregar música, fusionar u ordenar en un ordenador no lo hace en el
otro.** Las dos organizaciones se separan y no hay forma de juntarlas. Hay que decidir cuál manda y
copiar siempre en ese sentido (`artifacts/v4/datasets/musica/` y las carpetas nuevas de `Música/`).
En el escritorio, `data/raw_audio/test_20` tiene que seguir apuntando a su copia de test_20.
