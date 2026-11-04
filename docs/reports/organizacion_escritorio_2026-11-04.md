# La pestaña Organización en el PC de escritorio (2026-11-04)

Medido en la laptop (`C:\VS\traktor`) el 2026-11-04. El escritorio tiene 20 GB libres en C: y 89 GB
en el pendrive D:; su Python del sistema tiene pandas, PyYAML, scikit-learn y umap (dato del agente
del escritorio, no verificado desde acá). La decisión es de Gabriel: este informe no copia nada.

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
  CLAP. Python: además torch, torchaudio, soundfile, transformers y scipy. No consta que el Python
  del escritorio los tenga.
- Lo simple es copiar `artifacts/v4/datasets/musica/` entero (0,4 GB, entra en C:).
- `catalog.parquet` guarda rutas absolutas (`C:\VS\traktor\Música\...`), y la etiqueta Vocal y los
  exports las usan. En el escritorio el repo está en la misma ruta (`CONFIGURAR_OTRO_ORDENADOR.md`):
  sirve si `C:\VS\traktor\Música` existe, como carpeta o como junction.

## Opciones

1. **Solo en la laptop.** El escritorio descarga; la música nueva se lleva a la laptop y se agrega
   allí. No se copia nada; en el escritorio la pestaña Organización muestra el aviso. Hay una sola
   organización, así que no hay que decidir cuál manda.
2. **Una copia en el escritorio.** `Música/` en `D:\Música` (37,8 GB de 89 libres, junto al paquete
   de 27 GB) con un junction `C:\VS\traktor\Música` → `D:\Música`; `artifacts/v4/datasets/musica/` y
   `models/vocal_probe/` en C:. Funciona mientras el pendrive esté conectado con la misma letra.
3. **Una sola copia en el pendrive para los dos.** `Música/` y `artifacts/v4/datasets/musica/` en D:,
   con junctions en los dos ordenadores. No hay dos versiones, pero la laptop pasa a depender del
   pendrive (velocidad USB, y si se pierde, se pierde la organización: hace falta un respaldo).

**Con dos copias (opción 2), agregar música, fusionar u ordenar en un ordenador no lo hace en el
otro.** Las dos organizaciones se separan y no hay forma de juntarlas. Hay que decidir cuál manda y
copiar siempre en ese sentido (`artifacts/v4/datasets/musica/` y las carpetas nuevas de `Música/`).
En el escritorio, `data/raw_audio/test_20` tiene que seguir apuntando a su copia de test_20.
