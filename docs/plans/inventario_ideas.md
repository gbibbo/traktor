# Inventario de ideas de la recomendación

Todas las ideas con id, para que ninguna se pierda al pasar del plan viejo
(`recomendacion_playlists_plan.md`) al vigente (`plan_recomendacion_v2.md`). Armado el 2026-11-13.

Cómo leerlo:

- **Estado al 11-13.** Donde dice «(v)», se verificó ese día. El resto está transcripto del plan viejo, con la marca
  más reciente de cada fila, y revisado por muestreo. Si hay duda, manda la fila del plan viejo, que tiene el detalle.
- **En el plan v2.** El paso donde la idea se trabaja (A0 a A8, B0 a B6), la sección, o por qué queda afuera.
- Una idea nueva lleva el número siguiente de su letra. Las de la línea del teléfono usan la letra L.

## Ideas nuevas del 2026-11-13

| Id | Idea | Origen | Estado al 11-13 | En el plan v2 |
| :--- | :--- | :--- | :--- | :--- |
| D10 | DJSets v1: los sets de D5 y el audio de D6 en una sola carpeta, con tablas, lector y partición por DJ | Gabriel 11-13 | Hecho (v): 1549 sets, 36 611 temas, 33 321 con audio | 2.1; base de la línea A |
| V9 | Vistas de todo DJSets v1 por `track_id`: completa V7 e incluye los temas enteros de YouTube con la ventana de un preview | Plan v2 | Pendiente | A0 |
| E19 | Tablero fijo de la línea A: pares de prueba con temas no vistos, negativos en tres niveles, líneas de base | Plan v2; E7, E14 | Pendiente | A1 |
| E20 | Control de dominio de todo modelo aprendido: archivos de la biblioteca recodificados a 96 kbps y recortados como un preview | Plan v2; E15, E17 | Pendiente | A3; regla de 2.4 |
| P15 | Probabilidad calibrada de «van juntos»: da el campo `seguro` y dónde cortar la lista | Plan v2; P12, P13 | Pendiente | A2 |
| G14 | Escenas: grupos de DJs y sellos que comparten música; un campo `escena` por tema | Plan v2 | Pendiente | A6 |
| T32 | «Como tal DJ»: el centro de lo que toca cada uno de los 573 DJs, para buscar por su nombre | Plan v2; M6 | Pendiente | A6; casilla `referencia` |
| V10 | Comparar codificadores de texto y audio para la búsqueda por texto (un CLAP entrenado con música frente al actual) | Plan v2 | Pendiente | A7 |
| V11 | Otra vista de audio (por ejemplo MERT o las cabezas de Essentia), solo si A3 muestra que falta | Plan v2; T19 | Condicional | A7 |
| L1 | Formulario de consulta con casillas de listas cerradas: filtros, referencia, pesos, momento, texto, orden y largo | Gabriel 11-13; T31 | Hecho 11-13 (`config/telefono/formulario.json`, `formulario.py`) | 3.2; B0 |
| L2 | Ejecutor en el ordenador: recibe el formulario, filtra, puntúa, corta y dice qué hizo | Gabriel 11-13 | Hecho 11-13 (`ejecutor.py`) | B1 |
| L3 | Presets en el teléfono: formularios ya llenos que no pasan por el LLM | Gabriel 11-13 | Hecho en local 11-13 (Worker v6); sin publicar | B2 |
| L4 | Catálogo de la biblioteca generado por código: géneros con conteo, rangos de BPM, campos disponibles | Plan v2 | Hecho 11-13 (`formulario.catalogo()`) | B0 |
| L5 | Reglamento del LLM: tabla de decisión por tipo de pedido, en un solo archivo, del que salen los prompts | Gabriel 11-13 | Pendiente | B3 |
| L6 | Intérprete por reglas antes del LLM: números de BPM, nombres exactos de género, tonalidades, «como <tema>» | Plan v2 | Pendiente | B3 |
| L7 | LLM en dos pasos cortos: clasificar el tipo de pedido y llenar cada casilla; validadores y salida por defecto | Plan v2 | Pendiente | B3 |
| L8 | Un tema o archivo nombrado lo resuelve el ordenador por coincidencia aproximada; el teléfono muestra candidatos | Gabriel 11-13 | En parte 11-13: el ordenador resuelve y el teléfono muestra candidatos; falta que el texto llene `referencia` (B3) | B3 |
| L9 | Perfiles de pesos con nombre y tres niveles por factor; el LLM nunca escribe números | Gabriel 11-13; P6 | Pendiente | B1, B3 |
| L10 | «Momento de la noche» como preset y como texto; tabla fija hasta que exista A5 | Gabriel 11-13 | Tabla medida en DJSets 11-13, propuesta (`momentos_djsets_2026-11-13.md`); espera a Gabriel | B3, B6 |
| L11 | Banco de regresión del LLM: de 150 a 200 pedidos con su formulario esperado | Plan v2 | Semilla de 82 pedidos en `pedidos_telefono_borrador.md`; la revisa la auditoría, no Gabriel | B3 |
| L12 | El teléfono muestra qué entendió y cuántos temas pasaron cada filtro, con un toque para corregir | Plan v2 | Hecho en local 11-13 (Worker v6), salvo el toque para corregir: hoy se corrige con «Ajustar» | B4 |

## Ideas del plan viejo

### C, D y S: datos

| Id | Idea | Origen | Estado al 11-13 | En el plan v2 |
| :--- | :--- | :--- | :--- | :--- |
| C1 | Metadatos de Beatport para los temas de los sets | Auditoría 11-07 | Bajados para 30 655 temas (v). Entrenar con ellos: Gabriel confirmó el 11-13 que se usa el dataset completo, de uso privado | Sección 6, punto 2 |
| D0 | Botón: la primera pulsación refresca las playlists y abre Parecidos | Gabriel 11-07 | En uso | Fuera de las dos líneas |
| D1 | Historia de Traktor de Gabriel como sets | Local | Hecho 11-07: 21 sets, 592 temas | Solo para medir (2.4) |
| D2 | Registro en vivo de sesiones | Local | Crece por sesión; falta el lector | A8 |
| D3 | djmix-dataset: 5040 sets de MixesDB, solo texto | Kim et al. 2020 | Bajado; 355 temas en común con la biblioteca | A1 y A3, pares de la biblioteca |
| D4 | Raveform, 4902 sets | Propuesta | Sin bajar | No hace falta con D10 |
| D5 | Sets de 1001Tracklists | Gabriel 11-09 | Hecho (v): 1549 sets | D10 |
| D6 | Audio de los temas de los sets | Gabriel 11-09 | Hecho (v): 30 255 previews y 3066 de YouTube | D10 |
| D7 | Compilaciones mezcladas con un tema por pista | Gabriel 11-07 | 83 procesadas; la revalidación encontró discos contaminados | No se usa |
| D8 | Sets crudos de YouTube de DJs que elige Gabriel | Gabriel 11-07 | Sin estado | No hace falta por ahora |
| D9 | MTIQ y MTSQ: tripletas y audio de Jamendo | Paper GMR | Sin bajar | Condicional, con T27 |
| S1 | Dos usos de los sets: trayectoria y pares | Gabriel 11-09 | Hecho; columna `use` del dataset | A5 usa trayectoria |
| S2 | Sin sets comerciales | Gabriel 11-09 | Corregido: comercial es Guetta y Garrix; ante la duda, adentro | Cerrado con D10 |
| S3 | Porcentaje de géneros comerciales como segunda señal | Agente | Calculado, sin umbral | Cerrado con D10 |
| S4 | Curar con el listado, sin abrir el set | Agente | Hecho | Cerrado con D10 |
| S5 | Filtros del buscador en vez de scroll | Agente | Hecho | Cerrado con D10 |
| S6 | Semillas: Spotify, artistas de la biblioteca, DJs que nombró Gabriel | Agente; Gabriel | Hecho | Cerrado con D10; sesgo anotado en 2.1 |
| S7 | Términos y eventos como semillas | Agente | Hecho | Cerrado con D10 |
| S8 | Bola de nieve por afinidad de artistas | Agente | No se hizo | Para una v2 del dataset |
| S9 | Topes: 15 sets por DJ y 30 por programa | Agente | Hecho | Cerrado con D10 |
| S10 | Columna `veto` que la bajada respeta | Agente | Hecho; Gabriel no vetó todavía | Sección 6, punto 8 |
| S11 | Leer la página en vez del botón de exportar | Agente | Hecho | Cerrado con D10 |
| S12 | Piloto y después todo | Gabriel 11-10 | Hecho (v): bajada completa | Cerrado con D10 |
| S13 | Unir noches publicadas por partes | Agente | Cubre «Part N»; faltan episodios seguidos | Columna `night_group`; A5 |
| S14 | Enlaces por tema solo de lo usado | Gabriel 11-08 | Hecho con la bajada | Cerrado con D10 |
| S15 | Puente con la biblioteca por Beatport o por audio | Agente | Pocos temas en común (0,9 % en el piloto) | El puente es el audio: A3 y E20 |

### E: evaluación

| Id | Idea | Origen | Estado al 11-13 | En el plan v2 |
| :--- | :--- | :--- | :--- | :--- |
| E1 | Banco único: repite sets reales y mide el puesto del tema siguiente | Plan 11-07 | Hecho sobre D1; solo sirve para descartar | Medida secundaria en A1 |
| E2 | Los números descartan; entre lo que queda, escucha a ciegas | Plan 11-07 | Regla vigente | 2.4 |
| E3 | Como mucho unas 20 preguntas a Gabriel | Plan 11-07 | Regla vigente | A5 (E11) |
| E4 | Informar aparte los géneros chicos | Plan 11-07 | Regla vigente | A1 |
| E5 | Una sola variable por comparación | Plan 11-07 | Regla vigente | 2.4 |
| E6 | Resultados aparte para los sets de otros DJs y los de Gabriel | Plan 11-07 | Regla vigente | 2.4 |
| E7 | Partición por DJ; popularidad como línea de base | Auditoría 11-07 | `split_by_dj.csv` propuesta en el dataset (v) | A1 (E19) |
| E8 | Medidas: acierto en 30, rango recíproco, percentil; desempate al azar | Auditoría 11-07 | Vigente | A1 |
| E9 | Compacidad de los sets en CLAP | Adenda 11-08 | Medido 11-08 | Referencia |
| E10 | Precisión de zona y pureza fuera de muestra | Adenda 11-08 | Medido 11-08 | Referencia |
| E11 | Validar una coordenada en tres pasos: ordena, se mueve con la noche, acuerda con Gabriel | Adenda 11-08 | Regla vigente | A5 |
| E12 | Escucha a ciegas de listas enteras, de a dos | Adenda 11-08 | Lista desde el 11-10; sin respuestas (v). Versión corta hecha el 11-13: 5 a 5, no decide; Parecidos sigue con CLAP (`escucha_corta_e12_2026-11-13.md`) | Sección 6, punto 4 |
| E13 | Banco de conceptos local, con y sin referencia | Adenda 11-09 | Versión exploratoria medida | B5 |
| E14 | Control de tempo al medir los modelos | Adenda 11-09 | Medido 11-09 | A1: negativos del mismo BPM |
| E15 | Control de dominio: preview frente a biblioteca | Auditoría 11-09 | Medido 11-10: CLAP sufre el códec, MAEST pasa | Lo continúa E20 |
| E16 | Juez «mismo set» con sets cruzados con la biblioteca | Auditoría 11-10 | Hecho 11-10: pesos iguales y aprendidos no se distinguen | A1 y A3, pares de la biblioteca |
| E17 | El códec en CLAP, recodificando archivos | Auditoría 11-10 | Hecho 11-10: la causa es la compresión | Base de E20 |
| E18 | Pools de concepto con ajenos, en cuatro escalones | Gabriel 11-11 | Propuesta; faltan los conceptos y las playlists | B5; sección 6, punto 6 |

### G: género

| Id | Idea | Origen | Estado al 11-13 | En el plan v2 |
| :--- | :--- | :--- | :--- | :--- |
| G1 | Taxonomía propia de Beatport: similitud entre géneros | DECISIONS 11-06 | Hecha; es el factor de género de hoy | Línea de base en A1 |
| G2 | Aplicar las 17 objeciones menores a la taxonomía | Informe | Pendiente | Solo si G1 sigue siendo el factor |
| G3 | Género blando: probabilidades; cubre los 158 temas sin género | Propuesta 11-07 | Exploratorio: el clasificador acierta 39 % | A6, con las 30 655 etiquetas |
| G4 | Versiones del mismo tema con géneros distintos dan una distribución | `versiones.csv` | Pendiente | A6 |
| G5 | El género de Beatport es una decisión editorial | Gabriel | Vigente | 3.1: el filtro devuelve la etiqueta |
| G6 | Estilos de Discogs de MAEST como vector de género | Alonso-Jiménez 2023 | En uso como vista | A0, A1 |
| G7 | Similitud entre géneros por aparición en los mismos sets | Knees y Schedl 2013 | Medida en G13: AUC 0,72 | Sección 6, punto 7 |
| G8 | Una sola coordenada de género pierde estructura | Literatura | Vigente | Criterio |
| G9 | Embeddings de géneros aprendidos | Hennequin 2018 | Exploratorio | A6, si G3 no alcanza |
| G10 | Embeddings hiperbólicos de la jerarquía | Nickel y Kiela 2017 | Condicional a G9 | Sin fecha |
| G11 | Mapeo de estilos de Discogs a Beatport | Agentes | Descartado 11-06: acierta 24 % | No se hace |
| G12 | Tech House y Minimal, y Progressive y Melodic, se funden en el audio | Literatura | Vigente | B1: modo blando del filtro |
| G13 | Validar la taxonomía con transiciones de los sets | Gabriel 11-10 | Hecho 11-10 | Sección 6, punto 7 |

### H, I y V: infraestructura, Traktor y verificaciones

| Id | Idea | Origen | Estado al 11-13 | En el plan v2 |
| :--- | :--- | :--- | :--- | :--- |
| H1 | Cómputo pesado en gpuq y datos en el Drive | Gabriel 11-09 | En uso; Kaggle confiable, Colab corta las sesiones (v) | 2.5 |
| I1 | Un `.nml` por perfil en la carpeta de sugerencias | Plan 11-07 | Sin estado | Fuera de las dos líneas |
| I2 | Todas las playlists de sugerencias cuentan como «dentro» | Plan 11-07 | Sin estado | Fuera de las dos líneas |
| I3 | Cada lista anota sus componentes y su perfil en el registro | Plan 11-07 | Existe para Parecidos | B1: el ejecutor dice qué hizo |
| I4 | Costo en vivo con 5 perfiles por debajo de 300 ms | Plan 11-07 | Sin estado | B1 |
| I5 | Anotar los componentes y el rango de cada tema tocado | Auditoría 11-07 | Sin estado | A8, con D2 |
| V1 | Traktor acepta un `.nml` en Favoritos y lo relee | Plan 11-07 | Hecho | Cerrado |
| V2 | Test del cambio de `Toggle` | Plan 11-07 | Sin estado | Fuera de las dos líneas |
| V3 | Medir el costo en vivo con 5 perfiles | Plan 11-07 | Sin estado | B1 |
| V4 | Acceso a los sets largos de 1001Tracklists | Plan 11-07 | Resuelto con D5 | Cerrado |
| V5 | Leer la historia de Traktor y sus repetidos | Plan 11-07 | Hecho | Cerrado |
| V6 | Costo de la pasada de EffNet para la energía | Auditoría 11-07 | Sin estado | A7, si se elige esa vista |
| V7 | Vistas de cada preview en gpuq | Auditoría 11-10 | Dos trabajos terminados en Kaggle, 60 archivos en el Drive; cobertura sin contar (v) | A0 (V9) |
| V8 | CLAP sin la dirección del códec | Auditoría 11-10 | Pasó los umbrales; es la variante v8c | La vista CLAP de la línea A |

### M: representación aprendida

| Id | Idea | Origen | Estado al 11-13 | En el plan v2 |
| :--- | :--- | :--- | :--- | :--- |
| M1 | Vector por tema aprendido de las listas, sin audio | Deej-AI | Pendiente | No se hace: 30 261 de 36 611 temas están en un solo set (2.1) |
| M2 | Predecir ese vector desde CLAP | Vall et al. | Pendiente | No se hace, por M1 |
| M3 | Ranker aprendido sobre los componentes, por perfil | Propuesta 11-07 | Pendiente | A3 |
| M4 | Modelo de audio contrastivo con pares «tema y el siguiente» | Alonso-Jiménez 2023 | Pendiente | A4, condicional |
| M5 | Afinar el tronco de MAEST con pares de sets | Gabriel 11-08 | Lo cubre M4 | A4 |
| M6 | Vector de sets por artista | Auditoría 11-09 | Pendiente | A6 (T32) |

### P: parecido

| Id | Idea | Origen | Estado al 11-13 | En el plan v2 |
| :--- | :--- | :--- | :--- | :--- |
| P1 | CLAP como base del parecido | Gabriel 09-29 | Vigente; valor por defecto | Línea de base en A1 |
| P2 | CLAP en escala comparable | Medido 11-06 | Lo cubre P11 | A1 |
| P3 | Filtros de BPM ±20 y tonalidad ±2 semitonos | DECISIONS 11-06 | Vigentes; dejan fuera el 22 % de las transiciones reales | 3.2: casillas de filtro; la regla es de Gabriel |
| P4 | La misma tonalidad original puntúa más que la que pide pitch | DECISIONS 11-06 | Vigente, con peso 0 | A2 |
| P5 | Vector TIV de compatibilidad armónica | Rama `harmonic-tiv` | En curso en otra sesión | Aparte |
| P6 | Pesos por componente, con registro por lista | DECISIONS 11-06 | Vigente | L9 |
| P7 | Valores de tonalidad aprendidos de los sets | Gabriel 11-07 | Medido en P12a: misma tonalidad 2,0 veces el azar | A6: tablas con v1 |
| P8 | Radio de zona para el campo `seguro` | Gabriel 11-08 | Reemplazado por P12 y P13 | P15 en A2 |
| P9 | Proyección liviana aprendida de los sets | Gabriel 11-08 | Pendiente | A3, sobre las tres vistas |
| P10 | Consulta compuesta para Mantener: centro más paso | Vande Veire 2018 | Pendiente | A8 |
| P11 | Fusión de vistas: CLAP, MAEST y estilos | Auditoría 11-09 | Reproducida; falta la escucha | Línea de base en A1 |
| P12 | Ecuación de Parecidos por regresión logística | Gabriel 11-09 | Medida: no supera a los pesos iguales con 270 pares | A2, con los pares del dataset |
| P12a | Pesos de BPM, tonalidad y género con pares consecutivos | Auditoría 11-09 | Hecho 11-10 | A6: se repite con v1 |
| P12b | Pesos de los modelos con pares con audio | Auditoría 11-09 | Medido con pocos pares | A2 |
| P13 | Consenso entre vistas como marca de seguridad | Auditoría 11-09 | Medido: 99 % de acierto en el 3,5 % de los pares | A2, junto a P15 |
| P14 | Temas basura: lista de exclusión | Gabriel 11-10 | Idea sin diseñar | A8 |

### T: dirección, trayectoria y conceptos

| Id | Idea | Origen | Estado al 11-13 | En el plan v2 |
| :--- | :--- | :--- | :--- | :--- |
| T1 | Transiciones dirigidas entre géneros | Propuesta 11-07 | Pendiente | A6 |
| T2 | Fase del set por tema y género | Propuesta 11-07 | Pendiente | A5 |
| T3 | Coordenada de energía sin Mixed In Key | Essentia | Pendiente | A5; V11 si hace falta |
| T4 | Validar coordenadas con unos 20 juicios de Gabriel | MusAV | Pendiente | A5 (E11) |
| T5 | Trayectoria en vivo: nivel y pendiente | Kell 2013 | Pendiente | A8 |
| T6 | Paso parejo medido en sets reales | Kell 2013 | Pendiente | A8 |
| T7 | Introspectiva: saltos chicos y fase tardía | Gabriel 11-07 | Pendiente | A5; momento «after» en L10 |
| T8 | Más y menos energía dentro del vecindario | Propuesta 11-07 | Pendiente | A8 |
| T9 | Aprender en la sesión qué transiciones gustan | DJ-MC 2015 | Pendiente | A8 |
| T10 | La dirección se aprende de sets de DJ | Vall 2019 | Vigente | Criterio |
| T11 | Trayectorias en el espacio CLAP con D7 y D8 | Gabriel 11-07 | Números de D7 contaminados | A5 las mide con D10 |
| T12 | Separar en temas los sets crudos | Kim 2020 | El piloto no superó cortes fijos | No hace falta con D10 |
| T13 | Predictor del paso siguiente | Gabriel 11-07 | No superó a CLAP; la medida quedó refutada | A8 |
| T14 | Fases del set sin etiquetas | Adenda 11-07 | Pendiente | A5, como alternativa |
| T15 | Direcciones de energía aprendidas de las trayectorias | Adenda 11-07 | Pendiente | A5 |
| T16 | Final del tema actual frente al comienzo del candidato | Revalidación 11-07 | En curso | Solo con los 3066 temas enteros; sin fecha |
| T17 | La consulta suma el centro de lo ya tocado | Revalidación 11-07 | No mejora el siguiente; se mide como zona | A8 (T30) |
| T18 | Direcciones de texto en CLAP | Gabriel 11-08 | Exploratorio; es la línea de base | B5 |
| T19 | Cabezas de Essentia como coordenadas | Gabriel 11-08 | Pendiente | A7 (V11) |
| T20 | Cabeza propia sobre embeddings congelados: minuto o fase | Gabriel 11-08 | Pendiente | A5 |
| T21 | Variar la energía con tipos de transición | Vande Veire 2018 | No aplica por ahora | Sin fecha |
| T22 | Representar el tema por sus tramos altos | Vande Veire 2018 | Pendiente | No se puede con previews; sin fecha |
| T23 | Con el set en curso: el concepto sobre la zona de Parecidos | Gabriel 11-09 | Exploratorio medido | B1 |
| T24 | Puntaje de texto con varias frases y CLAP centrado | Auditoría 11-09 | Exploratorio: AUC 0,64 a 0,68 | B5 |
| T25 | Sondas supervisadas para conceptos con etiqueta | Auditoría 11-09 | Techo medido para género: 0,76 | A6 y B5 |
| T26 | Mapa aprendido de texto a audio | Paper GMR | Pendiente | A7 |
| T27 | Fusión tipo Combiner entrenada con MTIQ | Paper GMR | Condicional | Sin fecha |
| T28 | «Menos X» como eje de dos polos, nunca como «sin X» | Paper GMR | Pendiente | B3: por casillas; en texto solo «without vocals» |
| T29 | Sin referencia: playlist por concepto sobre toda la biblioteca | Gabriel 11-09 | En uso con texto crudo | B1 |
| T30 | Con el set en curso, la zona usa el historial | Gabriel 11-09 | Pendiente | A8 |
| T31 | Consulta con campos: el LLM devuelve géneros, BPM y voz | Gabriel 11-11 | Reemplazada el 11-13 por L1 a L12 | Sección 3 |
