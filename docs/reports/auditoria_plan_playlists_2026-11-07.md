# Auditoría del plan de playlists en vivo (2026-11-07)

Objeto: `docs/plans/recomendacion_playlists_plan.md` (commit `faf321b`), con el protocolo de la skill
`auditar`. Pedido de Gabriel: robustecer el plan, no romperlo. Las correcciones y los refuerzos quedaron
en el plan, en su lugar, marcados `[REFUTADO …]`, `[REEMPLAZADO …]` o «(auditoría 11-07)».

## Veredicto

El plan es coherente y ejecutable en el orden que propone. Los cinco contratos de la sección 3 resuelven
el riesgo de compartimentos estancos: cada idea del inventario tiene un contrato y ninguna llega a Traktor
por un camino propio. Las etiquetas del plan son honestas: lo que marca como EVIDENCIA OBSERVADA se
reprodujo en esta sesión, salvo tres números, que se corrigieron.

Lo que había que corregir:

1. **D1 son 21 sets, no 29.** Los 12 archivos de `Traktor 4.2.0/History` son copias byte a byte (md5) de
   12 archivos de `Traktor 4.4.1/History`. Las «sesiones de 4.2.0» no suman nada. La misma lectura, con
   `STARTDATE` además de `STARTTIME`, no deja ningún set de «24 h»: V5 queda resuelto.
2. **T3 con Essentia no está listo para correr.** No hay embeddings EffNet-Discogs ni MusiCNN guardados
   de `musica` (solo `clap_full`, `maesthf_full_l7` y `maesthf_full_styles`), las cabezas de Essentia no
   están bajadas, y las de arousal/valencia no existen sobre EffNet (solo sobre MusiCNN y VGGish). El
   orden de T3 se cambió: primero las direcciones de texto de CLAP y los estilos de MAEST, que ya están.
3. **El «13 728 temas, 66 GB» de la etapa 5 no tenía origen.** Se reemplazó por conteos medidos: 63 038
   temas en D3, 40 790 en House, Techno y Tech House, 4186 con 3 o más apariciones.

Lo que había que reforzar (sin cambiar decisiones de Gabriel):

4. **E1 estaba subespecificado.** Faltaban la medida (acierto en los 30 primeros, que es lo que Gabriel
   ve), qué pasa cuando los filtros dejan fuera el tema que el DJ tocó (se informa aparte: es la medida de
   P3), el pool de candidatos por fuente (en D3 el tema siguiente no está en la biblioteca), la partición
   por DJ entre entrenamiento y medida, y la popularidad como línea de base en sets ajenos.
5. **M3 y T9 no tenían su etiqueta en D2.** El registro anota los componentes de los 30 primeros; si el
   tema que Gabriel toca después no estaba entre ellos, no queda su vector. Nuevo I5, etapa 1.
6. **El estado de la trayectoria no definía la sesión.** `fase` necesita saber cuándo empezó el set; se
   agregó una regla (primer `tocado`, reinicio tras 20 minutos sin nada al aire) que también corta las
   sesiones de D1 dejadas abiertas.
7. **La etapa 1 mide solo sobre D1**, los sets de Gabriel, que por E6 no pueden fijar pesos solos. Se
   dice explícito: lo de la etapa 1 orienta, los pesos esperan a la etapa 2.
8. **C1 es largo**: unas 27 h para los tres géneros y 42 h para todo D3, al ritmo del cliente de Beatport.
   Se añadió el orden de prioridad (temas con 3 o más apariciones, sets de 2010 en adelante) y la señal
   `ai-train=no` del `robots.txt` de Beatport, que decide Gabriel antes de que M3 o M4 entrenen con esas
   etiquetas.
9. **M1 (set2vec) es ralo**: 50 696 de los 63 038 temas de D3 aparecen una sola vez. Quedó condicionado a
   los conteos, con M2 evaluado por validación cruzada contra una línea de base trivial.
10. **Introspectiva** tiene pocos sets largos hasta D5 (4 de Gabriel; 13 de D3 pasan de 120 min). El
    perfil arranca con `suavidad` y `clap`; `fase` y `texto_clap` toman peso cuando haya con qué medirlos.
11. **T4 puede tumbar una coordenada, no elegir** entre dos razonables (20 pares). Está dicho junto a la
    circularidad de etiquetar la dirección por el cambio de la coordenada (3.8).

Lo que se confirmó tal cual: los conteos de D3; que `Toggle` reconoce una sola playlist (y que tiene 3
tests); que el lado en vivo usa solo la biblioteca estándar; que el índice se carga en 0,02 s; que
1001Tracklists bloquea a «CPython» y pide 8 s; que MixesDB está activo (373 756 páginas de mezclas) y
pide 4 s; que D2 no tiene todavía eventos `principal` ni `tocado`; la energía de Mixed In Key (1424
temas, 83 % en 5 o 6); que `genre_pred.parquet` trae `top1`-`top3` con probabilidad (G3 es viable).

Un dato nuevo a favor del plan: **562 de los 592 temas al aire de D1 (95 %) están en el índice y 535 de
las 571 transiciones (94 %) tienen los dos temas**. E1 sobre D1 es viable desde la etapa 1 sin
resolver nada.

## Afirmaciones

| Afirmación | Etiqueta | Evidencia |
| :--- | :--- | :--- |
| Los 12 archivos de `4.2.0/History` son copias byte a byte de 12 de `4.4.1/History` | EVIDENCIA OBSERVADA | md5 de los 12 pares, idénticos; 4.4.1 tiene 36 archivos |
| D1 = 21 sets con 8 o más temas al aire por 2 min o más; 592 temas; 571 transiciones | EVIDENCIA OBSERVADA | Lectura de `PLAYLIST//ENTRY` con `EXTENDEDDATA` (`PLAYEDPUBLIC`, `DURATION`, `STARTDATE`, `STARTTIME`) unida a `COLLECTION/ENTRY` por `PRIMARYKEY` |
| 4 sets de 3 h o más: 3,40, 3,76, 3,37 h y uno de 130 temas en 8,5 h; ninguno de «24 h» | EVIDENCIA OBSERVADA | Misma lectura; el plan decía 3,5 a 3,9 h contando hasta el fin del último tema |
| 562 de 592 temas al aire y 535 de 571 transiciones de D1 están en el índice | EVIDENCIA OBSERVADA | Cruce por nombre de archivo (`archivo`, `original`) o artista-título normalizados contra `indice_recomendacion.json` (1760) |
| D3: 5040 sets, 102 932 entradas, 63 038 ids, 64 764 transiciones; House 1555, Techno 1547, Tech House 1410 | EVIDENCIA OBSERVADA | Conteo de `djmix-dataset.json`; coincide con el plan |
| D3: 1 set con una marca de 404 min; 13 sets con la última marca ≥ 120 min, 312 ≥ 90 | EVIDENCIA OBSERVADA | Mismo conteo; el plan decía «ninguno pasa de 150» |
| Cruce D3-biblioteca: 316 temas (19 %), 777 apariciones, 636 sets, 17 transiciones con los dos | EVIDENCIA OBSERVADA | Normalización propia (sin paréntesis ni corchetes). El plan da 355/851/696/25 con otra: el orden de magnitud y la inferencia de 3.6 se sostienen |
| En D3, 50 696 ids aparecen 1 vez; 12 342 ≥ 2; 4186 ≥ 3; 749 ≥ 5; 33 ≥ 10 | EVIDENCIA OBSERVADA | Conteo de apariciones por id (82 723 con id) |
| 40 790 ids distintos en sets de House, Techno o Tech House; 4741 de 5040 sets son de 2010 en adelante | EVIDENCIA OBSERVADA | Mismo conteo, por etiquetas `Category:` y por el año del título |
| C1 tarda unas 27 h (tres géneros) o 42 h (todo) | INFERENCIA | 2 pedidos por tema × 1,2 s (`BeatportClient(min_interval=1.2)`, `src/v4/common/beatport.py:213`) × ids; sin contar fallos ni la tasa de resolución, que mide la muestra chica |
| `Toggle.at_target` reconoce un solo destino | EVIDENCIA OBSERVADA | `tools/traktor_live/sugerencias_vivo.py:212-221`; tests en `tests/v4/test_traktor_live_api.py:286,303` y `test_traktor_live_platforms.py:63` |
| El lado en vivo usa solo la biblioteca estándar | EVIDENCIA OBSERVADA | Importaciones de `recomendador.py`, `sugerencias_vivo.py`, `tema_principal.py`, `traktor_estado.py`, `sugerencias_itunes.py`, `sugerencias_explorer.py`, `midi_io.py` y de `src/v4/common/{harmonic,genre_taxonomy,traktor_nml,__init__}.py` |
| El índice (versión 3, 1760 temas, 5,8 MB) se carga en 0,02 s y se decodifica en 0,01 s | EVIDENCIA OBSERVADA | Medido con `json` y `array("f")` en esta sesión |
| D2 no tiene eventos `principal` ni `tocado` | EVIDENCIA OBSERVADA | `kind` de los 9 registros `sugerencias_vivo_*.jsonl`: browserProbe, updateDeck, updateChannel, updateMasterClock, deckLoaded, boton, mezcla |
| `genre_pred.parquet` trae `top1`-`top3` con probabilidad; `final_source`: beatport_B 1336, classifier 324, beatport_A 115, gate_discogs 11 | EVIDENCIA OBSERVADA | Lectura del parquet (1786 filas) |
| Energía de Mixed In Key: 1424 temas; 5 y 6 suman 1189 (83,5 %) | EVIDENCIA OBSERVADA | `features/bpm_key.parquet`, columna `energy` |
| No hay EffNet ni MusiCNN guardados de `musica`; `models/essentia/` solo tiene los troncos y `genre_discogs400`; en el home de WSL no hay otros `.pb` (profundidad 4) | EVIDENCIA OBSERVADA | `ls` de `representations/` (3 con `embeddings.npy`, 1786 uids) y de `models/essentia/`; `find` en WSL. `~/traktor-artifacts/effnet_full_run.log` es de `test_20` |
| Arousal/valencia (DEAM, emoMusic, MuSe) existe solo sobre MusiCNN y VGGish; engagement, approachability, danceability y mood_* existen sobre discogs-effnet; licencia CC BY-NC-SA 4.0 | EVIDENCIA OBSERVADA | essentia.upf.edu/models, 2026-11-07 (primeros 100 000 caracteres de la página) |
| 1001Tracklists: `CPython` bloqueado, `Crawl-delay: 8` para `*` | EVIDENCIA OBSERVADA | `robots.txt`, 2026-11-07 |
| MixesDB está activo (373 756 páginas de mezclas, novedades del 2026-11-07), `Crawl-delay: 4`, bloquea `/db/`, `/tools/` y muchas categorías | EVIDENCIA OBSERVADA | Portada y `robots.txt`, 2026-11-07 |
| Beatport: `Allow: /`, sin `Crawl-delay`, `Content-Signal: search=yes, ai-train=no`, unos 35 robots de entrenamiento bloqueados | EVIDENCIA OBSERVADA | `robots.txt`, 2026-11-07 |
| Raveform: 38 % con marcas de tiempo; 1423 temas con estructura anotada a mano y género «referencing Beatport»; fuente MixesDB; sin licencia | EVIDENCIA OBSERVADA | mir-aidj.github.io/raveform, 2026-11-07. El total de 4902 sets del plan no está en la página: NO CONCLUYENTE |
| Traktor relee desde Favoritos un `.nml` reescrito (V1) | HIPÓTESIS | Sin Traktor en esta sesión; la decide Gabriel |
| Que el género mejore las listas; que una coordenada de CLAP o MAEST sea «energía» | NO CONCLUYENTE | Faltan E1 y T4; el plan ya lo dice |

## Revisión hostil de esta auditoría

- *¿La carpeta 4.2.0 podría ser la original y 4.4.1 la copia?* Da igual: contenido idéntico, se cuenta una
  vez.
- *¿El 95 % de cobertura de D1 está inflado por emparejar por título?* La mayoría empareja por nombre de
  archivo (471 temas tocados desde `Música/` y 141 desde `TRAKTOR ML/`, mismos nombres). Es cota alta en
  unos pocos casos de versiones distintas con el mismo título; no cambia que E1 sobre D1 sea viable.
- *¿Mi cruce D3-biblioteca (316) refuta el del plan (355)?* No: normalizaciones distintas. Lo que importa,
  que las transiciones con los dos temas son unas pocas decenas, sale igual.
- *¿La lectura de essentia.upf.edu fue completa?* No: 100 000 de 173 849 caracteres. La sección de
  arousal/valencia sí quedó dentro. Si una cabeza de arousal sobre EffNet apareciera en el resto, T3
  gana una opción; el orden propuesto no cambia.
- *¿Las 27-42 h de C1 son medidas?* No, son aritmética sobre el intervalo del cliente; la muestra chica de
  la etapa 2 las mide, y mide la tasa de resolución, que puede ser baja para sets viejos o títulos sucios.
- *¿Toqué decisiones de Gabriel?* No. Botón, Favoritos, filtros, Mixed In Key, fuentes y la regla de
  escucha siguen como están. Lo que se agregó son medidas, condiciones y orden dentro de las etapas.

## Pendiente de Gabriel

- V1 (Favoritos) y el correo a 1001Tracklists, como estaban.
- Decidir, antes de la etapa 4, si M3 y M4 pueden entrenar con etiquetas de Beatport dada la señal
  `ai-train=no`.
