# Avance desde la auditoría del plan de playlists (2026-11-07, tarde)

Para el auditor que revisó `docs/plans/recomendacion_playlists_plan.md` en el commit `4a07779`
(`docs/reports/auditoria_plan_playlists_2026-11-07.md`). Cubre lo hecho después, hasta el commit
`5a02e1b`. Autor: agente de Claude Code, sesión traktor-github-3a. Las etiquetas son las de la skill
`auditar`.

Fuera de alcance: los commits de descargas de Spotify y Soulseek (`download_JIJIJI/`) de otra sesión,
traktor-github-22. No tocan el plan.

## 1. Qué se pide auditar

1. **Si las fuentes nuevas entran por los contratos del plan.** Son D7 (compilaciones mezcladas) y D8
   (sets crudos de YouTube), y van en la sección 10 del plan. La pregunta es si alguna abre un camino
   propio o un compartimento estanco.
2. **Si las medidas del piloto con John Digweed sostienen lo que dice el informe**
   (`docs/reports/piloto_trayectorias_digweed_2026-11-07.md`). Sobre todo, la inferencia «en CLAP, el
   orden dentro de la noche pesa poco».
3. **Si el orden de las etapas se respetó.** El piloto es la etapa 2, punto 4, y se hizo antes de la
   etapa 1, a pedido de Gabriel. Ver la sección 5.
4. **Si el código nuevo cumple `AGENTS.md`:**
   - `tools/dj_sets/`;
   - `Toggle.inside`;
   - las configuraciones `.tsi` sin credenciales.
5. **Los riesgos nuevos de disco y cómputo de la sección 6**, antes de escalar.

## 2. Decisiones de Gabriel desde la auditoría

| Decisión | Dónde quedó |
| :--- | :--- |
| Bajar audio de YouTube está «totalmente perfecto» para esta aplicación | DECISIONS 2026-11-07, punto 12 |
| Las compilaciones mezcladas (los temas subidos uno por uno como «Mixed», por ejemplo las de John Digweed) son una fuente de sets | DECISIONS 2026-11-07, punto 13 |
| No simplificar los experimentos por falta de recursos: hay Google Drive (`gabriel@edgeaudiolabs.com`) y GPU por `github.com/gbibbo/gpuq` | Este informe, sección 6 |
| Lista de los 92 artistas que sigue en Spotify, guardada de forma segura | `config/dj_sets/artistas_spotify.txt` (commit `2fe4037`): repetidos quitados y «Peter Kruderv» leído como «Peter Kruder» |
| Subir al repo las dos configuraciones `.tsi` que usa | `tools/traktor_live/mapeos/` (commit `c4258b3`), sin los datos del servidor de transmisión |
| Orden siguiente: buscar en YouTube las compilaciones completas de sus artistas y, en paralelo, la etapa 1 | Sección 7 |
| La fase 4 sigue con la energía de Mixed In Key; el gesto del botón no cambia; se usan sets largos de 1001Tracklists | DECISIONS 2026-11-07, puntos 3, 8 y 9 (anteriores a la auditoría) |

## 3. Qué cambió en el plan

| Cambio | Commit | Lugar en el plan |
| :--- | :--- | :--- |
| Adenda: D7 y D8, ideas T11 a T15, componente `siguiente`, piloto de audio en la etapa 2 | `88674db` | Secciones 3.3, 4, 5 (etapa 2, puntos 4 y 5), 6, 8 y 10 |
| V1 resuelta (Favoritos) | `c4258b3` | Etapa 0, V1; sección 8 |
| Piloto hecho y sus consecuencias | `5a02e1b` | Etapa 2, puntos 4 y 5; sección 8 |

Lo que la adenda cambia de la arquitectura:
- Con D7, los componentes de audio se pueden medir sobre sets de otros DJs desde la etapa 2. Antes
  esperaban a la etapa 5.
- El puente 3 de la sección 3.7, el audio de los sets, pasa de la etapa 5 a la 2.
- D6 y M4 quedan para lo que D7 y D8 no cubran.

## 4. Lo hecho y verificado

### 4.1 V1: Favoritos de Traktor

Prueba de Gabriel con dos playlists de prueba en `C:\TRAKTOR ML Sugerencias`, reescritas en tres
rondas (el script quedó en el scratchpad de la sesión, fuera del repo).

| Caso | Resultado |
| :--- | :--- |
| Abrir A desde el favorito, viniendo de B | Muestra la versión nueva |
| Volver a B desde A | Muestra la versión nueva |
| Ir a otra carpeta y volver a B por el favorito | Muestra la versión nueva |
| Estar en B y volver a tocar el favorito de B | **No** la relee. Coincide con la lección del 2026-11-04 de `LESSONS_LEARNED.md` |

Consecuencia: las playlists de dirección se abren con el mouse desde Parecidos y siempre se ven al día.
Para ver una lista nueva estando ya en una de dirección, hay que pasar por otra.

### 4.2 I2: el botón vuelve desde cualquier playlist de sugerencias

`Toggle.inside` (commit `4308931`): cualquier playlist de la misma carpeta que el destino cuenta como
«dentro».
- Desde una de dirección, la pulsación vuelve al lugar guardado antes de la primera.
- Sin lugar guardado, refresca y abre Parecidos.
- Una carpeta de otro lado con el mismo nombre de playlist no cuenta.

Hay un test nuevo en `tests/v4/test_traktor_live_api.py`. Pasaron 64 tests del lado en vivo y del
recomendador. El servicio en vivo que corre ahora (PID 22768) tiene el código anterior y toma el nuevo
al reiniciarse.

### 4.3 Configuraciones de Traktor y Favoritos

- **Copias limpias.** Las copias de los «NUEVO» de Gabriel dejan en blanco la dirección, el punto de
  montaje, la URL y la contraseña de Icecast. Se comparó entrada por entrada contra el original: en la
  externa cambian solo esas 4, y en la interna ninguna.
- **Kontrol D2.** Las dos traen el dispositivo `Traktor.Kontrol D2`, leído con `mapeo_tsi.devices`.
  Deja de estar pendiente agregarlo.
- **Favoritos.** Viven en la entrada `Browser.Shortcuts`. La configuración externa trae unos sin
  «TRAKTOR ML Sugerencias» y con `C:\Users\usuario\...`, de otra PC: importarla reemplazó los de
  Gabriel. La lección quedó en `LESSONS_LEARNED.md`, junto a la de que importar un `.tsi` completo
  reemplaza el Controller Manager.

### 4.4 Piloto con John Digweed

Informe completo: `docs/reports/piloto_trayectorias_digweed_2026-11-07.md`. Herramientas:
`tools/dj_sets/` y `tests/v4/test_dj_sets.py` (5 tests). Los datos están en
`D:\TRAKTOR ML datos\dj_sets`, fuera de Git.

| Medida | Resultado | Etiqueta |
| :--- | :--- | :--- |
| CLAP de esta PC frente a `clap_full` (6 temas de la biblioteca) | Coseno 1,0000 (5) y 0,9973 (1) | EVIDENCIA OBSERVADA |
| Duración de las 16 mezclas continuas frente a la suma de sus pistas en los CD de MusicBrainz | Iguales al décimo de minuto | EVIDENCIA OBSERVADA |
| Pistas de los CD con su original en YouTube | 179 de 198; todos los originales subidos tienen pareja | EVIDENCIA OBSERVADA |
| Original identificado, por tema (mayoría de bloques) | 79-86 % | EVIDENCIA OBSERVADA |
| Original 1.º, bloque a bloque, entre los del disco y contra toda la biblioteca | 57-62 % y 26-34 % | EVIDENCIA OBSERVADA |
| Separación por picos de novedad frente a cortes fijos (F a ±60 s) | 0,38-0,43 frente a 0,35-0,56 | EVIDENCIA OBSERVADA |
| Distancia CLAP entre temas consecutivos, al azar del disco y al azar de la biblioteca | 0,16-0,17, 0,17-0,19 y 0,24 | EVIDENCIA OBSERVADA |
| «En CLAP, el orden dentro de la noche pesa poco frente a la elección del repertorio» | De la fila anterior; vale solo para CLAP | INFERENCIA |
| Pasos más chicos después de la segunda hora | 4 a 10 transiciones por hora y por disco; por bloque no aparece | NO CONCLUYENTE |
| Costo de CLAP | 1,3 a 1,9 min de CPU por hora de audio | EVIDENCIA OBSERVADA |

Dos errores propios que se corrigieron antes del commit:
1. **Pasos entre temas que no eran vecinos.** El primer cálculo de la trayectoria salteaba las pistas sin
   original y medía pasos entre temas no consecutivos. Ahora solo cuenta pistas vecinas (`seq_pos`).
2. **Dos parejas reales perdidas.** El emparejamiento por «artista título» perdía a Stelios Vassiloudis
   (en griego en MusicBrainz) y a «Fanfare». Se agregó un criterio por título, con test, y se verificó
   que no crea parejas falsas: «Verche - Rapid Eye Movement» sigue sin pareja, como debe.

### 4.5 Material para escalar D7

`musicbrainz_mixes.py artistas` sobre los 92 artistas encontró 345 grupos `DJ-mix` que no son radio, de
37 artistas. Los que más tienen: Sasha 69, John Digweed 49, Carl Cox 43, Nick Warren 20, Hernán
Cattáneo 13, M.A.N.D.Y. 11, Soulwax 10 y Dubfire 8.
- Los otros 55 no tienen ninguno en MusicBrainz.
- HIPÓTESIS: en parte porque figuran con la relación «mezclado por» y no como artista del disco.
- Falta ver cuántas compilaciones están completas en YouTube: la mezcla continua más los originales.

## 5. Desvío del orden de etapas

El piloto (etapa 2, punto 4) se hizo antes de la etapa 1. Lo pidió Gabriel («puedes arrancar»), no
consume salidas de la etapa 1 y su resultado cambia cómo se arma la etapa 2. De la etapa 1 solo se hizo
I2. Pedido: confirmar que ninguna dependencia de la sección 5 quedó rota. Según la lista de la auditoría,
E1 sobre D1 sigue siendo el primer banco, y el piloto no fijó pesos.

## 6. Riesgos nuevos

| Riesgo | Estado | Qué se propone |
| :--- | :--- | :--- |
| **Disco del sistema** | C: está al 97 % (19 GB libres; tenía 25 GB al empezar la tarde). Google Drive está en modo de transmisión en `G:`, y su caché local va a C: (`AppData\Local\Google\DriveFS`). Escribir en `G:` pasa por esa caché | No escribir audio en `G:` hasta mover la caché de Drive a D: (Preferencias de Drive). Mientras, el audio sigue en D: (58 GB libres) |
| **Escala de D7** | Una compilación con originales ocupa unas 2,3 veces el set. Las 345 serían unas 2000 h de audio: unas 55 h de CPU y unos 120 GB | Lotes en prioridad baja; el audio a Drive cuando la caché esté en D:, o se borra después de los embeddings |
| **GPU** | `gpuq` usa las GPU gratis de Colab y Kaggle; necesita su propia puesta en marcha (cuentas, una máquina siempre encendida). `AGENTS.md` pide estimar y pedir el OK antes de usar GPU | Seguir en CPU. Antes de usar `gpuq`, la estimación medida y el OK de Gabriel |
| **Condiciones de las fuentes** | YouTube: autorizado por Gabriel; los videos son subidas oficiales de Bedrock. MusicBrainz: API con 1,5 s entre pedidos. `yt-dlp` quedó instalado en el Python de la Store con `pip --user` | Auditor: confirmar que nada de esto choca con `AGENTS.md` (nada de audio en Git, nada de credenciales) |
| **Generalización** | Un solo DJ, de progressive y techno melódico | Repetir las medidas con otros artistas antes de fijar algo |

## 7. Próximos pasos acordados con Gabriel

1. Buscar en YouTube cuáles de las 345 compilaciones están completas, empezando por Sasha, Carl Cox,
   Nick Warren y Hernán Cattáneo. Bajarlas en lotes y sacar CLAP con las mismas herramientas.
2. En paralelo, la etapa 1:
   - D1 como tabla de sets (los 21 sets de Gabriel);
   - el banco E1 con las medidas que fijó la auditoría;
   - P2 (escala de CLAP);
   - G3 y G4 (género blando);
   - la estructura de varias playlists (I1, I3, I5).
3. Con D7 a escala: T11 por tema, E1 sobre sets de otros DJs, y la separación en temas aprendida de los
   límites reales de D7 (T12).

## 8. Pendiente de Gabriel

- El correo a 1001Tracklists. No se encontró ningún límite publicado de descargas manuales por cuenta.
- Mover la caché local de Google Drive a D: antes de usarlo para audio.
- De la auditoría: si M3 y M4 pueden entrenar con etiquetas de Beatport (señal `ai-train=no`). D7 no
  usa etiquetas de Beatport.

## 9. Afirmaciones

| Afirmación | Etiqueta | Evidencia |
| :--- | :--- | :--- |
| V1: Traktor relee desde Favoritos al llegar desde otra playlist y no al volver a tocar la abierta | EVIDENCIA OBSERVADA | Prueba de Gabriel, 3 rondas, 2026-11-07 |
| `Toggle.inside` vuelve desde cualquier playlist de la carpeta de sugerencias | EVIDENCIA OBSERVADA | `test_toggle_returns_from_any_playlist_of_the_suggestions_folder`; 64 tests en verde |
| Las copias `.tsi` del repo no tienen credenciales | EVIDENCIA OBSERVADA | Comparación entrada por entrada (457 entradas); `check_staged.py` en verde; `grep` de la contraseña y del servidor: 0 |
| Importar la configuración externa reemplazó los Favoritos | INFERENCIA | Su `Browser.Shortcuts` es la lista que tenía Traktor a las 15:43, sin TRAKTOR ML; Gabriel dice que faltaba desde que abrió Traktor |
| D7 da trayectorias por tema limpias en el espacio de la biblioteca | EVIDENCIA OBSERVADA para 3 compilaciones de Digweed | Informe del piloto |
| Hay 345 compilaciones `DJ-mix` de 37 de los 92 artistas | EVIDENCIA OBSERVADA | `D:\TRAKTOR ML datos\dj_sets\musicbrainz\artistas.json` |
| Escribir audio en `G:` llenaría C: | HIPÓTESIS | La caché de DriveFS está en C:; se decide mirando cuánto crece con un lote chico |
