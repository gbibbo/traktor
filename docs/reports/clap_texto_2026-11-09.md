# Cómo escribir un concepto para que CLAP lo encuentre en la biblioteca

**Fecha:** 2026-11-09. **Para:** quien cambie las instrucciones del LLM de `cloudflare/concept-ai-worker.js` o el buscador
de `tools/traktor_live/conceptos_agente.py`. **Pedido de Gabriel:** entender cómo se pasa de lo que escribe una persona a un
texto que sirva para buscar en su música, y armar el circuito donde el LLM de Cloudflare hace ese trabajo.

## Veredicto

Con CLAP, cómo se escribe pesa tanto como lo que se dice. Dos frases que una persona leería igual comparten un tercio de
la lista de 30. Envolver el concepto en plantillas («This is a sound of…», «music in the style of…») empeora la búsqueda.
Promediar tres redacciones estabiliza la lista. Ningún número de la búsqueda distingue un concepto real de uno sin
sentido: una frase absurda da el mismo contraste que una buena. Por eso el LLM no puede elegir mirando puntajes; tiene que
ver qué trajo cada búsqueda. El circuito se armó así: el LLM escribe tres textos, el ordenador los busca y describe lo que
encontró, el LLM decide y se crea una playlist nueva en Traktor. Con las instrucciones finales, el texto curado ordena
mejor que el texto escrito por la persona en 14 de 16 pedidos, medidos contra estilos de MAEST. Si las listas suenan
bien lo decide Gabriel escuchando.

## Lo medido

Biblioteca: índice `indice_recomendacion.json` (1760 temas, `clap_full`, modelo `laion/clap-htsat-unfused`). Código:
`src/v4/evaluation/clap_texto.py` (frases en `clap_texto_frases.py`) y `clap_texto_llm.py`. Salidas:
`artifacts/v4/datasets/musica/evaluation/clap_texto/*.json`, fuera de Git. Commit de partida: `73d042d`.

Las referencias son independientes de CLAP pero ninguna es la verdad:

- 20 estilos Discogs de MAEST presentes en la biblioteca (positivo = el 10 % con más activación, 176 temas por estilo);
- género confirmado de Beatport (10 géneros con al menos 40 temas);
- etiqueta Vocal (`clap_probe`);
- clases de AudioSet de AST, positivo = el 5 % superior, sobre 129 temas. La corrida local
  (`extract_representations.py --models ast --shard 4:0`) se detuvo en 140 temas: completarla lleva horas de CPU, y por
  la regla de cómputo de AGENTS.md eso va a gpuq, con el visto bueno de Gabriel para el primer trabajo de ese tipo.

| Afirmación | Etiqueta | Evidencia |
| --- | --- | --- |
| Frases equivalentes dan listas distintas: comparten 34 % de los 30 primeros; dos conceptos distintos, 9 % | EVIDENCIA OBSERVADA | `parafrasis.json`: 12 conceptos × 6 frases; mínimo 0 % en «dark hypnotic» y «vocal soulful» |
| Las plantillas empeoran frente a la palabra sola: «This is a sound of X» −0,039 AP [−0,080; −0,003], «music in the style of X» −0,051 [−0,101; −0,005], «X electronic dance music» −0,047 [−0,089; −0,011]; «X music» y «X song» sin diferencia clara | EVIDENCIA OBSERVADA | `formas.json`, 20 estilos; Beatport va en el mismo sentido (AP de la palabra sola 0,150, la mejor) |
| Cambiar la forma cambia la lista: «This is a sound of X» comparte el 15 % con «X» | EVIDENCIA OBSERVADA | `formas.json`, columna de solapamiento |
| Nombre del género + descripción del sonido: +0,039 AP [−0,003; 0,079] frente al nombre solo; ayuda donde el nombre es débil (disco +0,20, tribal house +0,16, downtempo +0,15) y daña donde el nombre es fuerte (minimal −0,15, house −0,12) | EVIDENCIA OBSERVADA, no concluyente en promedio | `formas.json`, 18 estilos con descripción escrita a mano |
| Promediar 3 redacciones sube el acuerdo entre listas de 34 % a 51 % (+0,17 [0,12; 0,23]); no sube el AP frente a una forma sola (+0,003 [−0,018; 0,021]) | EVIDENCIA OBSERVADA | `ensamble.json` |
| «Without vocals» sí baja la voz: Vocal en los 30 primeros 0,30, contra 0,44 con el concepto solo, 0,47 con «instrumental» y 0,72 con «with vocals» (biblioteca 0,52) | EVIDENCIA OBSERVADA, solo voz | `negacion.json`, 8 conceptos. La literatura dice que la negación no funciona en CLAP; acá, para la voz, sí. Otras negaciones no se midieron |
| Restar una frase negativa no saca lo excluido | EVIDENCIA OBSERVADA, exploratoria | Medido antes, en 10 conceptos (`DECISIONS.md`, 2026-11-09) |
| Escribir en español no sirve: comparte 29 % con el inglés, y a veces nada («bajo profundo» y «deep bass», ρ 0,12) | EVIDENCIA OBSERVADA | `idioma.json`, 16 pares |
| Ningún puntaje dice si el concepto existe: 8 frases sin sentido («tax invoice…») dan un contraste de 2,55 y los conceptos reales, 2,2 a 2,9 | EVIDENCIA OBSERVADA | `no_sonoro.json` |
| Entre redacciones de un mismo estilo, el contraste apenas sirve para elegir (ρ 0,28 [0,11; 0,43] con el AP); coherencia, acuerdo y máximo no | EVIDENCIA OBSERVADA | `senales.json`, 20 estilos |
| Nombrar al artista no encuentra sus temas (AUC medio 0,52) | EVIDENCIA OBSERVADA, n chico | `artistas.json`: 4 artistas con al menos 4 temas |
| Las palabras de instrumento son las más estables entre formas (cuerdas 0,59, viento 0,58); las de ritmo y textura, las menos (0,18 y 0,20) | EVIDENCIA OBSERVADA | `vocabulario.json`, 150 palabras × 3 formas |
| CLAP reconoce: coro gospel 0,95, voz de mujer 0,94, piano eléctrico 0,93, piano 0,90, órgano 0,88, saxo 0,87, «brass» 0,85, synth lead 0,85, cuerdas 0,84; «trumpet» 0,75, «corneta» 0,59 | EVIDENCIA OBSERVADA, exploratoria | `instrumentos.json`: 129 temas con AST, unos 6 positivos por clase (AUC contra AST) |
| No sirven: kick 0,47, claps 0,51, rap 0,49, sirena 0,46, pájaros 0,35, «bass guitar» 0,28 | EVIDENCIA OBSERVADA, exploratoria | Ídem. Kick y claps los tiene todo tema de club |
| Lo que encontró una búsqueda se puede leer con palabras: «birthday party music» → latin house, saxophone, percussion; «cheerful … brass» → brass instruments, wind instruments, horns | EVIDENCIA OBSERVADA | Centro de los 30 primeros frente al de la biblioteca, contra 150 palabras del vocabulario |

## Las reglas, y dónde se aplican

| Regla | Por qué | Dónde |
| --- | --- | --- |
| Inglés siempre | Español: 29 % de acuerdo | Instrucciones y validador (rechaza lo que no es ASCII) |
| Sin envolturas («this is», «sound of», «in the style of», «track», «song», «music by») | Empeoran el AP | Instrucciones y validador |
| Lo que no se oye (cumpleaños, playa, gym, un artista) se reemplaza por cómo suena, y el teléfono lo dice | Artistas: AUC 0,52; las ocasiones dan listas sin relación con su descripción sonora | Instrucciones; validador de palabras de ocasión |
| Género + sonido, 6 a 14 palabras | Ayuda donde el nombre del género es débil | Instrucciones |
| Tres textos con palabras distintas, y la lista final sale de su promedio | Estabiliza la lista | Instrucciones y `crear_lista()` |
| Menos voz se escribe «without vocals»; cualquier otra negación se descarta | Es la única negación medida que funciona; «few vocals» nombra la voz | Instrucciones; el validador reescribe «few/minimal/no vocals» y descarta «no», «not», «without» en lo demás |
| No agregar sonidos que el DJ no pidió | Con la lista de palabras reconocidas en las instrucciones, el modelo agregaba piano, órgano o acid sin que nadie los pidiera, sin ganar AP (rondas 3 y 4 del banco) | Instrucciones |
| El juez decide con lo que se oye («suena a», géneros) y no con puntajes | Ningún puntaje distingue un concepto real | Instrucciones del juez e informe de `sondear()` |

## El circuito

1. El DJ escribe en el teléfono. El Worker llama al LLM («interpretar») y guarda el trabajo con tres textos.
2. El ordenador reclama el trabajo y busca cada texto por separado (`sondear()`). Devuelve, sin puntajes, los géneros, el
   BPM, la proporción con voz, seis ejemplos y seis palabras de «suena a».
3. El Worker llama al LLM («elegir»): se queda con los textos que encajan o, si ninguno encaja, escribe otros tres. Hay
   una ronda más como máximo. Si el LLM no responde, sigue con todos los textos.
4. El ordenador busca con el promedio de los textos elegidos (`buscar()`, la única pieza que decide qué temas salen) y
   escribe una playlist nueva, `Concepto <fecha> <nombre>.nml`, en la carpeta de sugerencias.
5. El teléfono muestra qué entendió, qué no se oye, qué buscó, cómo va a sonar y la lista. «Ajustar» vuelve al paso 1 con
   la corrección y el intento anterior.

Tiempos medidos en local con el LLM real: interpretar 1 a 3 s, sondeo y decisión 2 a 5 s. La primera búsqueda de la
noche tarda unos 30 s más: carga CLAP y el vocabulario. Costo: 9.091 neuronas por millón de tokens de entrada y 27.273
por millón de salida (página de precios de Workers AI, 2026-11-09). Unas 35 a 50 por concepto, contra 10.000 gratis por
día.

## Banco del LLM

18 pedidos escritos como los escribiría un DJ apurado (`clap_texto_llm.py`), por el circuito real: Worker en `wrangler dev`
con D1 local, LLM real, agente con CLAP. Se compara el texto tal como lo escribió la persona contra el promedio de los
textos curados. 16 pedidos tienen un estilo de MAEST como referencia; 2 tienen la etiqueta Vocal.

| Ronda | Cambio en las instrucciones | AP crudo → curado (16 estilos) | Mejoran | Vocal en 30 (2 pedidos) |
| --- | --- | --- | --- | --- |
| 1 | Primera versión | 0,168 → 0,250, +0,082 [0,038; 0,123] | 14 | 0,53 → 0,85 |
| 2 | Ejemplo sin «party», aceptar lo más cercano si un estilo escasea, validador de ocasiones | +0,080 [0,035; 0,124] | 14 | → 0,77 |
| 3 | Lista de palabras que CLAP reconoce | +0,077 [0,028; 0,123] | 14 | → 0,68 |
| 4 | La lista, solo para nombrar lo pedido | +0,082 [0,039; 0,125] | 14 | → 0,75 |
| 5 | Sin lista; «nunca agregar un sonido no pedido» (vigente) | +0,078 [0,029; 0,125] | 14 | → 0,78 |

El validador que descarta las negaciones (salvo «without vocals») se agregó después de la ronda 5. Lo prueban los tests
de punta a punta, no el banco. Las rondas no se distinguen entre sí: el LLM corre con temperatura 0,3 y n = 16. La ronda
5 se eligió por fidelidad al pedido, no por número. Los dos pedidos que empeoran son «tribal» (0,33 → 0,25) y «atardecer
en la playa» (contra *downtempo*, una referencia elegida por el agente).

## Revisión hostil

- **Circularidad.** Las reglas salieron de medidas con MAEST sobre esta biblioteca, y el banco también usa MAEST. Los
  pedidos y sus referencias se fijaron antes de correr el banco y no se cambiaron, pero el banco no es una prueba fuera
  de muestra.
- **MAEST mide estilo, no si la lista sirve en una pista.** Un AP mejor no es una playlist mejor.
- **AST cubre 129 temas.** Con unos 6 positivos por clase, un AUC tiene un error de ±0,1 o más. Las clases raras
  (trompeta) casi no las detecta ni AST: el umbral del 5 % superior es una probabilidad de 0,0009.
- **La negación se midió solo para la voz.** La regla de no negar nada más sigue a la literatura (Kuan y Lee 2026,
  arXiv 2607.12290) y no está medida acá.
- **No concluyente:** si la lista curada suena mejor que la cruda. Lo que la decidiría es que Gabriel escuche a ciegas
  pares de listas (cruda y curada) para 10 pedidos suyos.

## Cómo reproducir

```
python src/v4/evaluation/clap_texto.py --pruebas F,P,E,N,L,A,S,R,Q,V,I --low-priority
python src/v4/evaluation/clap_texto_llm.py --url <Worker> --salida banco.json   (gasta neuronas)
```

Para terminar AST sobre la biblioteca hace falta un trabajo de gpuq (`docs/GPUQ.md`), que es pesado y el primero de su
tipo, así que espera el visto bueno de Gabriel. La caché local, con 140 temas, se puede subir para no repetirlos. Después
se vuelve a correr la prueba I.
