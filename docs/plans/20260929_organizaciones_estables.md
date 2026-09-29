# Plan: organizaciones estables, por carpeta, incrementales y con semillas (2026-09-29)

Pedido de Gabriel (2026-09-29): (1) organizar solo una carpeta de la biblioteca, con su propio mapa;
(2) agregar música nueva a una organización existente sin tocar lo ya organizado, o rehacerla desde
cero; (3) poder exigir que dos o más temas vayan en la misma playlist, congelando el resto (cambio
mínimo) o rehaciendo todo con esa semilla. Motivo: las playlists tienen que ser estables para poder
tocarlas con confianza.

## Concepto: organización con nombre y versiones

`artifacts/v4/datasets/<dataset>/orgs/<nombre>/`

| Archivo | Contenido |
| :--- | :--- |
| `org.json` | parámetros (representación, peso BPM, PCA, n carpetas, temas por playlist, alcance) e historial de versiones |
| `constraints.json` | semillas: grupos de temas que deben ir juntos (persisten entre versiones) |
| `models/` | PCA, UMAP y media/desvío del BPM con que se construyó la versión base (para congelar) |
| `v<N>/assignments.parquet` | por tema: carpeta, playlist, posición, x/y del mapa y origen (`build`, `add`, `add-far`, `add-new`, `link`) |
| `v<N>/names.json` | nombres de carpetas y playlists |

Cada operación escribe una versión nueva; las anteriores quedan para volver atrás y los exports
llevan el número de versión.

## Operaciones (`src/v4/pipeline/organize.py`)

- `ingest --scope <carpeta>`: catálogo, tags y embeddings solo de esa carpeta (el resto del catálogo
  se conserva). Es la pasada chica que exige AGENTS.md antes de algo largo.
- `build --name X [--scope <carpeta>]`: organización desde cero sobre el alcance (toda la biblioteca
  si no hay alcance). Ward en dos niveles igual que Phase 2 y mapa UMAP propio. Las semillas se
  respetan colapsando cada grupo en un solo punto (su centroide) antes de Ward: así quedan en la misma
  carpeta y playlist por construcción.
- `add --name X [--scope <carpeta>]` (congelado): los temas nuevos se proyectan con la PCA y el UMAP
  guardados; cada uno va a la playlist de centroide más cercano si cae dentro de su radio (percentil
  90 de las distancias de sus miembros). Los que quedan lejos de todas: si son al menos 12 forman
  playlists nuevas (Ward entre ellos) en la carpeta más cercana; si son menos, van igual a la más
  cercana, marcados `add-far`. Se insertan en el orden existente en la posición más barata
  (inserción de menor costo, con el mismo puntaje de transición de Phase 4). Nada de lo existente
  cambia de playlist, de posición relativa ni de lugar en el mapa.
- `link --name X --track A --track B [--rebuild]`: agrega la semilla. Congelado (default): se mueve
  el grupo completo (incluidas semillas anteriores que lo toquen) a la playlist, entre las que ya
  contienen algún miembro, que minimiza la distancia total de los que se mueven; nadie más se mueve.
  `--rebuild`: `build` desde cero con todas las semillas.
- `import --name X --from-hash <config>`: convierte una corrida de Phase 2-4 en organización v1 sin
  cambiar nada (se reajustan PCA y UMAP, deterministas, y se verifica que el mapa sea idéntico).
  Se usa para la organización elegida el 2026-09-29 (`fb78f2f6`).
- Export (`phase5_export.py --org-name X`) y página de revisión (`--org-name X`) leen la versión
  actual; la página marca los temas nuevos y los vinculados.

Phase 2-4 por hash quedan como camino de experimentos; las playlists para tocar salen de `orgs/`.

## Arreglo previo necesario

`extract_representations.py --folder` ensamblaba la variante solo con los temas de la carpeta:
después de extraer una carpeta nueva, `embeddings.npy` quedaba con esos temas nada más. El
ensamblado tiene que usar siempre el catálogo completo.

## Implementado (2026-09-29)

- Una organización guarda una lista de alcances: el de `build` más los que se suman con
  `add --scope`; un `build` posterior cubre todos.
- `add` usa la PCA/UMAP y las estadísticas de BPM de la versión base (`models_v<N>/`); el mapa de
  los temas nuevos sale de `UMAP.transform`.
- Con datos reales, en una organización descartable (`#1 BIBO`, 1008 temas) se agregó `2019`
  (251): 220 a playlists existentes y 31 en una playlist nueva; los 1008 existentes no cambiaron de
  playlist, de orden relativo ni de lugar en el mapa. `link` congelado movió solo el tema necesario.
  Rehacer con una sola semilla cambia mucho la organización (ARI 0.345 frente a la congelada).

## Verificación

Tests: Ward en dos niveles idéntico a Phase 2 sin semillas; semillas siempre juntas; `add` no cambia
nada de lo existente; `link` congelado solo mueve el grupo; inserción conserva el orden relativo;
`import` reproduce el mapa; catálogo por alcance conserva el resto. Con datos reales: importar
`fb78f2f6`, construir una carpeta chica, y en una organización de prueba descartable construir sin
una carpeta y luego agregarla.
