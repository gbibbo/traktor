# Taxonomía de géneros de Beatport (red de agentes), 2026-11-06

Continúa `generos_beatport_2026-11-06.md` (auditoría del género y evidencia). Decisión: DECISIONS
2026-11-06, puntos 10 y 11.

## Veredicto

Una red de agentes construyó una taxonomía propia de los 47 géneros de Beatport, y sus dos evaluadores
(texto y audio) la aprobaron en la ronda 2 sin objeciones mayores. Está en
`config/genre_taxonomy/taxonomia.json` y ya es el componente `genero` del recomendador en vivo. El peso
por defecto del género sigue en 0 (por ahora solo CLAP, decisión de Gabriel), así que lo que suena hoy no
cambia.

En los 16 géneros con al menos 10 temas, la matriz sigue a la evidencia medida en la biblioteca
(Spearman 0.915). Esto mide coherencia, no valida la taxonomía: los agentes armaron la matriz mirando esa
misma evidencia. La validación queda para la escucha o para 20 tripletas como mucho.

El mapeo de los estilos de Discogs (MAEST) a Beatport acierta el género en el 24 % de los temas, contra
el 38 % del clasificador propio. No se usa.

Hay que tenerlo en cuenta al ajustar los pesos: entre los 30 sugeridos, CLAP varía unos 0.05 y el género
salta de 0.2 a 0.4. Con un peso de género de 0.5, las 30 sugerencias tras un Tech House pasan a ser
todas Tech House.

## Cómo se hizo

- **Entradas** (`artifacts/v4/genre_taxonomy/inputs/LEEME.md`, fuera de Git):
  - Beatport: 47 géneros y 136 subgéneros. Copia sin ids en `config/genre_taxonomy/beatport_generos.json`.
  - Wikidata: 619 géneros bajo «electronic music».
  - MusicBrainz: 358 géneros electrónicos con sus relaciones.
  - Los 400 estilos de Discogs de MAEST.
  - La evidencia de `genre_evidence.py`: BPM, estilos de Discogs, vecinos y centroides CLAP, versiones
    con dos géneros y un borrador de similitud.
  - La taxonomía de la tesis de Gabriel no apareció. Gabriel pidió usar otras conocidas, completas y al
    día.
- **Red** (Workflow `taxonomia-generos-beatport`, corrida `wf_8d4fd8ed-513`):
  - Un constructor arma la taxonomía.
  - En cada ronda la revisan dos evaluadores en paralelo:
    - la lente de texto juzga contra Wikidata, MusicBrainz y los nombres de Beatport;
    - la lente de audio, contra la evidencia de la biblioteca.
  - Si hay objeciones mayores o problemas de forma (`genre_taxonomy.validate`), un editor corrige.
  - Hubo como mucho 3 rondas. Al final, un agente mapeó Discogs a Beatport y otro revisó el mapeo.
  - En total, 8 agentes, unos 100 minutos y 1,5 M de tokens.
- **Rondas**:
  - Ronda 1: la lente de texto planteó 3 objeciones mayores y 6 menores; la de audio, 1 mayor y 5 menores.
  - El editor aceptó 11 de esas 15 objeciones y aceptó en parte las otras 4.
  - Ronda 2: las dos lentes aprobaron, con 9 y 8 objeciones menores.

## Resultado

**Regla de cálculo** (`genre_taxonomy.py`):
- mismo género con otro subgénero: 0.75;
- misma familia: 0.7;
- familias hermanas: 0.2;
- sin relación: 0.

Encima van 211 relaciones explícitas, de género o de subgénero, que solo pueden subir el valor.

**Árbol**: 34 familias.

| Rama | Familias (géneros) |
| :--- | :--- |
| baile > pista (4/4, 115-135 BPM) | house (House, Funky, Jackin) · tech (Tech House, Minimal / Deep Tech) · indie_tech · bass_house · deep (Deep, Organic) · afro (Afro House, Amapiano) · melodico (MH&T, Progressive) · indie_disco (Indie Dance, Nu Disco) · techno (Peak Time, Raw) · comercial (Dance / Pop, Mainstage) |
| baile > rave | trance (dos Trance) · psy · hard_techno · duro (Hard Dance / Hardcore / Neo Rave) |
| baile > bass | breaks_electro · uk_garage (UK Garage, 140, Bass / Club) · festival_bass (Dubstep, Trap) · dnb |
| escucha | electronica · chill (Downtempo, Ambient) |
| abierto (Open Format) | urbano · latino · afro_caribe · pop · rock · country · edits |
| utilitario | DJ Tools / Acapellas |

**Matriz de los 16 géneros con evidencia** (sin subgénero):

| | TH | PT | MHT | DH | H | ID | AH | MDT | E | PH | DP | Raw | OH | Psy | IT | MS |
| :--- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| Tech House | 1 | .60 | .40 | .55 | .85 | .40 | .50 | .80 | .35 | .20 | .70 | .35 | .30 | 0 | .80 | .45 |
| Techno PT | .60 | 1 | .60 | .40 | .40 | .50 | .20 | .55 | .30 | .60 | .30 | .85 | .20 | .45 | .30 | .20 |
| Melodic H&T | .40 | .60 | 1 | .65 | .40 | .70 | .50 | .20 | .55 | .85 | .40 | .35 | .60 | 0 | .20 | .20 |
| Deep House | .55 | .40 | .65 | 1 | .60 | .50 | .60 | .55 | .65 | .50 | .45 | .20 | .85 | 0 | .20 | .20 |
| House | .85 | .40 | .40 | .60 | 1 | .45 | .65 | .45 | .50 | .20 | .65 | .20 | .20 | 0 | .60 | .50 |
| Techno Raw | .35 | .85 | .35 | .20 | .20 | .40 | .20 | .40 | 0 | .50 | .20 | 1 | .20 | .30 | .20 | .20 |

Las demás filas están en `taxonomia.json` (`Taxonomy.matrix`). Abreviaturas de las columnas:
- TH Tech House, PT Techno (Peak Time / Driving), MHT Melodic House & Techno;
- DH Deep House, H House, ID Indie Dance, AH Afro House, MDT Minimal / Deep Tech;
- E Electronica, PH Progressive House, DP Dance / Pop;
- Raw Techno (Raw / Deep / Hypnotic), OH Organic House, Psy Psy-Trance, IT Indie Tech, MS Mainstage.

**Decisiones discutibles de la red**, con el detalle en `notas` del JSON:
- **Hard Techno**: tiene familia propia en «rave»; vale 0.65 con los dos techno y 0.5 con Hard Dance.
- **Electronica**: va en «escucha» por el texto, aunque en la biblioteca suena a deep house. Por eso
  tiene relaciones de 0.55 a 0.65 con Deep House, Organic House y MH&T.
- **Indie Tech y Bass House**: tienen familia propia de un solo género, porque la familia compartida los
  acercaba demasiado a Minimal y a Funky House.
- **Psy-Trance**: vale 0 con casi toda la pista, salvo 0.45 con Techno PT.

**Propuestas del editor, ya aplicadas** en `compare()`:
- «DJ Edits > X» se compara como el género X. Hoy ningún tema de la biblioteca es DJ Edits.
- «DJ Tools / Acapellas» cuenta como tema sin género: el componente no cuenta. Son 3 temas.

## Mapeo de Discogs a Beatport

`config/genre_taxonomy/discogs_a_beatport.json` asigna de 1 a 3 géneros de Beatport, con pesos, a cada
uno de los 400 estilos. El revisor corrigió 40 estilos.

La medición (esta sesión) se hizo sobre los 1451 temas con género de Beatport. Se toman las
probabilidades MAEST de cada estilo (`maesthf_full_styles`), se multiplican por el mapeo y se elige el
género con más peso:

| | Mapeo de Discogs | Clasificador propio (`genre_model_eval.json`) |
| :--- | --: | --: |
| Acierta el género | 23,8 % | 38,1 % (50,2 % con confianza >= 0.6) |
| Entre los 3 primeros | 54,0 % | 71,6 % |

Además:
- la similitud media entre el género que da el mapeo y el real es 0.631; decir siempre Techno PT da 0.577;
- en los 175 temas sin Beatport donde el clasificador tiene confianza >= 0.6, el mapeo coincide con él en
  el 33 %.

Por eso los temas sin género de Beatport siguen con el clasificador. Las cifras no son del todo
comparables: el clasificador elige entre 15 géneros, con validación cruzada por artista sobre 1307
confirmados, y el mapeo elige entre los 47.

## En el recomendador

- **Índice versión 3**: agrega el subgénero de Beatport (`beatport_genres.parquet`) cuando su género es
  el del tema. Hay 30 temas con subgénero; 1602 de 1760 tienen género.
- **`recomendador.py`**: el componente `genero` es `Taxonomy.compare((género, subgénero), ...)`. Si falta
  la taxonomía, vuelve a valer 1 el mismo género y 0 otro, y el aviso lo dice.
- **Prueba con la colección real de Traktor** (1717 temas sugeribles): tema principal de Tech House con
  subgénero Latin Tech, 968 candidatos, 35 ms.
  - Solo con CLAP, las 30 sugerencias se reparten en 8 Techno PT, 6 MH&T, 5 Tech House, 4 Indie Dance y
    otros. El componente de género va de 0.4 a 1.
  - Con CLAP 1 y género 0.5, las 30 son Tech House.
- **Escala de CLAP**: sobre 40 temas principales al azar, la mediana de CLAP entre los candidatos es 0.76.
  El 1.º sugerido tiene 0.93 y el 30.º 0.88.
  - CLAP separa poco entre los primeros, y el género salta de a 0.2-0.4.
  - Antes de ajustar pesos conviene llevar CLAP a una escala comparable, por ejemplo el percentil entre
    los candidatos. Queda pendiente.

## Pendiente

- **Objeciones menores de la ronda 2**: 17, no aplicadas. Entre ellas:
  - bajar Tech House–Dance / Pop de 0.7 a 0.6;
  - agregar Tech House–Progressive House 0.35;
  - subir House–Dance / Pop a 0.75;
  - subir Indie Dance con Techno PT y con Deep House a 0.6;
  - agregar Psy-Trance–Trance (Raw) 0.4 y Mainstage–Trance (Main Floor) 0.35;
  - pasar Amapiano y Bass / Club a familias propias;
  - bajar Hard Techno–Raw a 0.55;
  - agregar puentes de Open Format (Latin Pop–Pop, Latin Hip-Hop–Hip-Hop);
  - corregir dos citas: el homónimo de «electro» en Wikidata y la evidencia de Rock.

  Se aplican en una ronda más de la red o a mano con un evaluador.
- **Validar con Gabriel**: escucha o 20 tripletas como mucho (DECISIONS 2026-11-06, punto 10).
- **Escala de los componentes**: ver «En el recomendador».

## Afirmaciones

| Afirmación | Etiqueta | Evidencia |
| :--- | :--- | :--- |
| Los dos evaluadores aprobaron en la ronda 2, sin objeciones mayores | EVIDENCIA OBSERVADA | Salida del Workflow `wf_8d4fd8ed-513`: `rondas[1]`, `objeciones_abiertas` vacío |
| La taxonomía cubre los 47 géneros y los nombres son los de Beatport | EVIDENCIA OBSERVADA | `validate(...) == []`; `test_the_taxonomy_in_use_is_valid_and_keeps_the_main_relations` |
| Spearman 0.915 y Pearson 0.909 con el borrador (120 pares) | EVIDENCIA OBSERVADA | Cálculo de esta sesión contra `genre_evidence/borrador_similitud.csv` |
| Esa correlación no valida la taxonomía | INFERENCIA | La red tenía el borrador entre sus entradas; mide coherencia con lo que vio |
| El mapeo de Discogs acierta menos que el clasificador | EVIDENCIA OBSERVADA | 23,8 % contra 38,1 % (`genre_model_eval.json`); conjuntos y número de clases distintos |
| Con un peso de género de 0.5, el género manda sobre CLAP | EVIDENCIA OBSERVADA | Prueba con la colección real: 30 de 30 Tech House; CLAP entre el 1.º y el 30.º, 0.93 a 0.88 |
| Las objeciones menores no cambian el orden de las filas principales | HIPÓTESIS | Se decide aplicándolas y comparando la matriz y las listas |
| La taxonomía mejora las sugerencias | NO CONCLUYENTE | Falta escuchar con un peso de género mayor que 0 (y la escala de CLAP) |
