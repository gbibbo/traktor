# Pedidos del teléfono: semilla del banco

**Para qué sirve.** Son los ejemplos y el banco de regresión de la línea B del plan (`plan_recomendacion_v2.md`,
sección 3; idea L11). Cada fila es un pedido como se escribiría en el teléfono y lo que el sistema debería entender.
Los escribió el agente auditor el 2026-11-13, salvo los tres que dicen «tuyo», que Gabriel escribió de verdad.

**Gabriel no tiene que revisar esta lista.** [REEMPLAZADO 2026-11-13 por Gabriel: revisar 82 filas es el mismo error
que la escucha de 600 temas; los ejercicios se adaptan a una persona ocupada] La primera versión le pedía corregirla
fila por fila. Ahora:

- La revisa la sesión auditora, no él.
- Sus pedidos reales se juntan solos: el ordenador ya guarda cada pedido que llega del teléfono
  (`conceptos_resultados.jsonl`). Cada pedido real entra al banco con lo que el sistema entendió, y se corrige solo
  si él dice que la lista salió mal.
- Si quiere agregar alguno, lo escribe al final, sin la segunda columna.

**Reglas** (Gabriel, 11-13; pueden cambiar):

- Un género escrito en el texto es **blando**: ese género va primero y entran también los géneros vecinos.
- Un género elegido en un preset del teléfono es **duro**: solo ese género. Los presets no están en esta lista, porque
  no pasan por el LLM.

**Cuatro valores por defecto que eligió el agente.** Gabriel los cambia cuando quiera; no hace falta que conteste:

1. Un número de BPM solo es ese número ±2.
2. Si se pide algo que no hay, el teléfono dice que no hay. Si existe algo cercano, lo ofrece con un toque; nunca arma
   por su cuenta una lista de otra cosa.
3. «Ignora la tonalidad» baja el peso de la tonalidad en el orden. El filtro del vivo queda, porque es regla de Gabriel.
4. Ante un pedido ambiguo («que contraste», «bajar un cambio»), el teléfono pregunta con dos o tres opciones de un
   toque, en vez de adivinar.

Lo que hoy existe como dato por tema: género (1602 de 1760 temas), BPM, tonalidad, voz y artista. Sello y fecha de
lanzamiento existen para 1461 temas en los datos de Beatport, pero no están en el índice del recomendador. El momento
de la noche y «como tal DJ» llegan con la línea A.

## 1. Un género

| # | Pedido | Lo que debería entender |
| --: | :--- | :--- |
| 1 | tech house | Género Tech House, blando |
| 2 | algo de melodic techno | Género Melodic House & Techno, blando |
| 3 | techno | Géneros Techno (Peak Time / Driving) y Techno (Raw / Deep / Hypnotic), blando |
| 4 | afro | Género Afro House, blando |
| 5 | progressive | Género Progressive House, blando |
| 6 | minimal | Género Minimal / Deep Tech, blando |
| 7 | house clásico | Género House, blando. Texto: «classic house» |
| 8 | indie dance con aire disco | Género Indie Dance, blando. Texto: «disco-influenced» |

## 2. BPM

| # | Pedido | Lo que debería entender |
| --: | :--- | :--- |
| 9 | 126 bpm | BPM de 124 a 128 |
| 10 | entre 120 y 124 | BPM de 120 a 124 |
| 11 | más rápido que esto | BPM más rápido que lo que suena |
| 12 | bajemos un poco las revoluciones | BPM más lento que lo que suena |
| 13 | mismo tempo | BPM igual al de lo que suena |
| 14 | arriba de 130 | BPM de 130 para arriba |

## 3. Género y BPM juntos

| # | Pedido | Lo que debería entender |
| --: | :--- | :--- |
| 15 | tech house a 126 | Género Tech House, blando. BPM de 124 a 128 |
| 16 | deep house lento, 118 o por ahí | Género Deep House, blando. BPM de 116 a 120 |
| 17 | techno rápido, 132 para arriba | Los dos géneros de techno, blando. BPM de 132 para arriba |
| 18 | melodic entre 122 y 125 | Género Melodic House & Techno, blando. BPM de 122 a 125 |

## 4. Tonalidad

| # | Pedido | Lo que debería entender |
| --: | :--- | :--- |
| 19 | que combine en tonalidad con lo que suena | Tonalidad compatible con lo que suena |
| 20 | algo en 8A | Tonalidad 8A y sus compatibles |
| 21 | misma tonalidad que este | Tonalidad igual a la de lo que suena |
| 22 | tech house que no choque armónicamente | Género Tech House, blando. Tonalidad compatible con lo que suena |

## 5. Voz

| # | Pedido | Lo que debería entender |
| --: | :--- | :--- |
| 23 | sin voces | Voz: sin |
| 24 | algo con voz de mujer | Voz: con. Texto: «female vocals» |
| 25 | instrumental, nada cantado | Voz: sin |
| 26 | un vocal house | Género House, blando. Voz: con |

## 6. Parecido a un tema nombrado

| # | Pedido | Lo que debería entender |
| --: | :--- | :--- |
| 27 | algo como losing it de fisher | Referencia: el tema «Losing It» de FISHER |
| 28 | parecido a miau on acid | Referencia: «Miau On Acid» de Nicone |
| 29 | Kolombo - Nos Fuimos (Extended Mix).mp3 | Referencia: ese archivo |
| 30 | temas como the curve pero más rápidos | Referencia: «The Curve» de Adana Twins. BPM más rápido que el de la referencia |
| 31 | algo tipo aria de vsy | Referencia: «Aria» de VSY |
| 32 | como ese tema de artbat que tengo | Referencia sin decidir: el teléfono muestra los 4 temas de ARTBAT para elegir |

## 7. A partir de lo que suena

| # | Pedido | Lo que debería entender |
| --: | :--- | :--- |
| 33 | más de esto | Referencia: lo que suena. Es la lista de Parecidos |
| 34 | como este pero más oscuro | Referencia: lo que suena. Texto: «darker» |
| 35 | lo mismo pero sin voz | Referencia: lo que suena. Voz: sin |
| 36 | algo parecido pero más melódico | Referencia: lo que suena. Texto: «more melodic» |
| 37 | seguir en esta línea pero subiendo | Referencia: lo que suena. Momento: subida |
| 38 | algo que contraste con esto, cambiar de clima | Ambiguo: el teléfono pregunta con opciones (otro género, otro clima, otra energía) |

## 8. Momento de la noche

| # | Pedido | Lo que debería entender |
| --: | :--- | :--- |
| 39 | para abrir la noche | Momento: apertura |
| 40 | estamos en el pico | Momento: pico |
| 41 | ya es tarde, estoy cerrando | Momento: cierre |
| 42 | after, seis de la mañana | Momento: after |
| 43 | warm up tranquilo sin voces | Momento: apertura. Voz: sin |
| 44 | recién se está llenando la pista | Momento: subida |
| 45 | la gente está cansada, bajar un cambio | Ambiguo: el teléfono pregunta con opciones (más lento, más tranquilo, momento de cierre) |

## 9. Cómo suena

| # | Pedido | Lo que debería entender |
| --: | :--- | :--- |
| 46 | techno oscuro e hipnótico | Los dos géneros de techno, blando. Texto: «dark hypnotic techno» |
| 47 | algo con trompetas | Texto: «brass, trumpets» |
| 48 | percusión tribal con tambores | Texto: «tribal percussion, hand drums» |
| 49 | acid bien ácido | Texto: «acid bassline, squelchy synth» |
| 50 | piano house alegre | Género House, blando. Texto: «uplifting piano chords» |
| 51 | bajo gordo y rodante | Texto: «fat rolling bassline» |
| 52 | algo espacial con pads largos | Texto: «spacey, long atmospheric pads» |
| 53 | Musica electrónica con reminisencias a música italiana de los 90 (tuyo) | Texto: «90s italian house, piano, euro dance» |

## 10. Ocasiones y cosas que no se oyen

| # | Pedido | Lo que debería entender |
| --: | :--- | :--- |
| 54 | Quiero música de cumpleaños con cornetas (tuyo) | «Cumpleaños» no se oye: el teléfono lo dice. Texto: «festive, bright brass and horns» |
| 55 | para el atardecer en la playa | «Atardecer en la playa» no se oye. Texto: «warm, mellow, balearic» |
| 56 | para una boda, que baile todo el mundo | «Boda» no se oye. Voz: con. Texto: «uplifting, catchy» |
| 57 | música para manejar de noche | «Manejar de noche» no se oye. Texto: «driving, hypnotic, steady» |

## 11. Artistas, DJs y sellos

| # | Pedido | Lo que debería entender |
| --: | :--- | :--- |
| 58 | temas de solomun | Artista: Solomun. Lista corta: hay 2 temas |
| 59 | algo del sello afterlife | Sello: Afterlife, si el campo está; si no, el teléfono dice que no puede filtrar por sello |
| 60 | como un set de adriatique | Referencia: el DJ Adriatique, cuando exista «como tal DJ». Hoy no hay temas suyos en la biblioteca |
| 61 | nada de fisher | Excluir al artista FISHER |

## 12. Época

| # | Pedido | Lo que debería entender |
| --: | :--- | :--- |
| 62 | clásicos de los 90 | Fecha hasta 1999, si el campo está. Texto: «90s» |
| 63 | temas nuevos, de este año | Fecha de 2026, si el campo está |

## 13. Pesos

| # | Pedido | Lo que debería entender |
| --: | :--- | :--- |
| 64 | que pese más el género | Perfil de pesos: género |
| 65 | me importa más que suene parecido que el bpm | Perfil de pesos: sonido. BPM: nivel bajo |
| 66 | que sea bien armónico | Perfil de pesos: armónico |
| 67 | ignora la tonalidad | Tonalidad: nivel bajo. El filtro del vivo queda |

## 14. Excluir, limitar y ordenar

| # | Pedido | Lo que debería entender |
| --: | :--- | :--- |
| 68 | sin repetir lo que ya puse hoy | Excluir lo tocado esta noche |
| 69 | solo 10 temas | Largo: 10 |
| 70 | nada de vocales ni de tech house | Voz: sin. Excluir el género Tech House |
| 71 | ordénalos por bpm | Orden: por BPM |

## 15. Lo que no está en la biblioteca

| # | Pedido | Lo que debería entender |
| --: | :--- | :--- |
| 72 | bossa nova brasilera (tuyo) | No hay: el teléfono lo dice y no arma una lista de otra cosa |
| 73 | reggaeton | No hay |
| 74 | drum and bass | No hay. El teléfono ofrece lo más cercano con un toque: Breaks / Breakbeat / UK Bass, 7 temas |
| 75 | rock | Género Rock: lista corta, hay 2 temas |

## 16. Vagos, largos o mal escritos

| # | Pedido | Lo que debería entender |
| --: | :--- | :--- |
| 76 | algo bueno | Nada que llenar. Si hay algo sonando, la lista de Parecidos; si no, el teléfono pide más detalle |
| 77 | tec haus 125 | Género Tech House, blando. BPM de 123 a 127 |
| 78 | melodic tecno sin vos | Género Melodic House & Techno, blando. Voz: sin |
| 79 | a ver quiero algo que arranque tranquilo tipo deep house pero que no sea aburrido que tenga algo de voz y que ande por los 120 122 más o menos y que combine con lo que está sonando ahora | Género Deep House, blando. Voz: con. BPM de 120 a 122. Tonalidad compatible con lo que suena. Texto: «groovy» |
| 80 | estoy cerrando la noche y quiero ir bajando de a poco con cosas melódicas sin voz que no bajen de 120 | Momento: cierre. Voz: sin. BPM de 120 para arriba. Texto: «melodic» |
| 81 | hola | Nada que llenar: el teléfono pide un pedido |
| 82 | poneme lo de siempre | Nada que llenar: el teléfono pide más detalle |

## Los tuyos

| # | Pedido | Lo que debería entender |
| --: | :--- | :--- |
| | | |
