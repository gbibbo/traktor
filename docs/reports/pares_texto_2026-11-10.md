# G13 y P12a: género, BPM y tonalidad en las transiciones de los DJs de D5 (solo texto), 2026-11-10

Plan de playlists en vivo, sección 12.9 (G13, P12a); DECISIONS 2026-11-10 («después de E15», puntos 2 y 3).
Protocolo: skill `auditar`; decisiones fijadas en el encabezado de `src/v4/evaluation/pares_texto.py` antes de
correrlo. Tests: `tests/v4/test_pares_texto.py`. Resultado, fuera de Git:
`D:\TRAKTOR ML datos\dj_sets\sets\pares_texto.json` (semilla 0, 327 s). No se le pidió nada a Beatport ni a
1001Tracklists: se leyeron los sets y los manifiestos que dejó la sesión de la bajada.

**Con qué se midió** (la bajada sigue; lo que falta bajar son sets de pares, así que esto no es una muestra al azar
del total, DECISIONS 2026-11-10, punto 4):
- 1549 sets de 354 DJs, con 53 650 filas; 1329 filas tocadas encima de otra quedan fuera.
- 15 655 previews con `detalles`, de los que 13 975 son temas de estos sets.
- 13 559 pares consecutivos con los dos temas identificados, de 282 DJs. Se sacaron 7 pares del mismo tema y quedan
  317 pares repetidos entre sets.
- Dos nulos: «mismo set» (otro tema del set a 5 filas o más; 67 635 pares) y «otro set» (un tema de un set de otro
  DJ; 67 795). Cada set pesa 1 en cada clase. Intervalos: remuestreo de DJs.
- Género, BPM y tonalidad son los de Beatport.

## Veredicto

1. **G13: la taxonomía ordena el estilo de la noche, no la transición.**
   - Contra «otro set», cuanto más alto el valor de la taxonomía, más aparece entre consecutivos (Spearman 0,91
     entre el valor y el log del cociente). De 0 a 0,4 aparece la mitad que al azar (0,43 a 0,58 veces); de 0,7 a
     0,85, 1,35 a 1,47 veces; el mismo género, 3,4 veces.
   - Separa mejor que «mismo género o no»: AUC 0,682 frente a 0,627, diferencia [+0,043, +0,066].
   - Contra «mismo set» (la transición, con la noche fija), nada que mire el género distingue el tema siguiente de uno
     lejano del mismo set: AUC 0,526 a 0,529. Solo el mismo género sube un poco (1,18 veces).
2. **Lo aprendido de los sets le gana a la taxonomía, con los DJs de prueba fuera.** Contra «otro set»:
   - coaparición sin dirección (G7): 0,718, diferencia con la taxonomía [+0,028, +0,047];
   - transición dirigida (T1): 0,711, diferencia [+0,020, +0,041].
   El plan dice que el peso del género sale «con el factor que gane», y gana G7. Pero usarlo cambia la fórmula del
   factor `genero` del recomendador, y esa decisión es de Gabriel. Hasta que la tome, E16 y la escucha siguen con la
   taxonomía.
3. **P12a, BPM**: cuanto más cerca, más aparece. Contra «otro set»:

   | Diferencia de BPM | Veces sobre el azar |
   | :--- | --: |
   | menos de 1 | 2,17 |
   | 1 a 2 | 1,59 |
   | 2 a 3 | 1,42 |
   | 3 a 5 | 1,07 |
   | 5 a 8 | 0,58 |
   | 8 a 12 | 0,41 |
   | 12 a 20 | 0,18 |

   Contra «mismo set» la caída es más suave, de 1,17 a 0,55. La fórmula del recomendador, que baja en línea recta
   hasta 20 BPM, va en el mismo sentido.
4. **P12a, tonalidad, con la de Beatport** (no se aprende un peso: lo decidió Gabriel; la regla de compatibilidad no
   se toca). Las cifras son contra «otro set»; contra «mismo set» dan casi lo mismo.

   | Relación | Veces sobre el azar | Intervalo |
   | :--- | --: | :--- |
   | misma | 1,98 | [1,64, 2,32] |
   | paralela | 1,66 | [1,33, 1,98] |
   | vecina | 1,28 | [1,14, 1,45] |
   | dos semitonos | 1,09 | — |
   | un semitono | 0,92 | — |
   | diagonal | 0,89 | — |
   | relativa | 0,87 | [0,74, 1,03] |
   | otra | 0,83 | — |

   Solo la tónica: la misma tónica, 1,83 veces; a 5 semitonos (cuarta o quinta), 1,17; las demás distancias, de 0,81
   a 0,93. Los pares del mismo modo aparecen apenas más que los de modo distinto (1,05 a 1,12 frente a 0,90 a
   0,93).
5. **Ecuación de solo texto** (BPM y género, con los filtros en vivo y la tonalidad de Beatport en el filtro):
   - pares: 9556 positivos de 275 DJs; 4003 consecutivos quedan fuera por los filtros o por no tener candidatos;
   - AUC fuera de muestra: BPM solo 0,666, género solo 0,688, pesos iguales 0,731, aprendida 0,732. La aprendida gana
     por +0,0016 [+0,0004, +0,0027]: se distingue, pero no importa;
   - pesos en desvíos: BPM 0,68 [0,59, 0,76], género 0,54 [0,48, 0,60].
   No va al recomendador: los pesos del vivo salen de E16, con audio y con la tonalidad de Traktor.
6. **El género de Beatport y el de 1001Tracklists** coinciden en el 38,9 % de 12 618 temas, y en el 56,2 % sin el
   paréntesis («Techno (Peak Time / Driving)» = «Techno»).

**Discrepancias entre la taxonomía y los DJs** (contra «otro set», pares de géneros con 30 consecutivos o más; para
G2 y para Gabriel):

| Géneros | Taxonomía | Veces en los DJs | Consecutivos |
| :--- | --: | --: | --: |
| Mainstage, Techno (Peak Time / Driving) | 0,20 | 1,94 | 92 |
| Mainstage, Melodic House & Techno | 0,20 | 1,47 | 51 |
| Progressive House, Techno (Peak Time / Driving) | 0,60 | 0,33 | 132 |
| Indie Dance, Indie Tech | 0,40 | 2,01 | 76 |
| Indie Tech, Melodic House & Techno | 0,20 | 1,15 | 124 |
| Deep House, Melodic House & Techno | 0,65 | 0,59 | 171 |
| Deep House, Organic House | 0,85 | 0,82 | 42 |
| Minimal / Deep Tech, Techno (Peak Time / Driving) | 0,55 | 0,38 | 33 |
| Melodic House & Techno, Techno (Peak Time / Driving) | 0,60 | 0,58 | 393 |
| Dance / Pop, Progressive House | 0,20 | 0,90 | 39 |

## Afirmaciones

| Afirmación | Etiqueta | Evidencia |
| :--- | :--- | :--- |
| La taxonomía sube con lo que hacen los DJs cuando el azar es otro set | EVIDENCIA OBSERVADA | `pares_texto.json`, `g13.otro set`: Spearman 0,908; AUC 0,682 frente a 0,627 |
| No distingue el tema siguiente dentro del mismo set | EVIDENCIA OBSERVADA | `g13.mismo set`: AUC 0,526-0,529 en las cuatro medidas |
| La coaparición aprendida de los sets ordena mejor que la taxonomía | EVIDENCIA OBSERVADA, DJs de prueba fuera | G7 0,718, diferencia [+0,028, +0,047] |
| Cuanto más cerca el BPM, más aparece entre consecutivos | EVIDENCIA OBSERVADA | Cociente de 2,17 a 0,18 contra «otro set» |
| Con la tonalidad de Beatport, los DJs repiten la tonalidad y la tónica | EVIDENCIA OBSERVADA | Misma 1,98; misma tónica 1,83; relativa 0,87 |
| Parte de la «paralela» es la misma tonalidad con el modo cambiado por Beatport | INFERENCIA | E15: Beatport dice «menor» en el 64 % y la biblioteca en el 91 %; la paralela es la relación que más aparece entre los dos (38 de 146). Se decide en E16, con la tonalidad de Traktor de los mismos pares |
| La auditoría daba 2,3, 1,9 y 1,3 para misma, paralela y vecina | EVIDENCIA OBSERVADA (otro conteo) | Aquí 2,03, 1,73 y 1,31 contra «mismo set», sin las filas tocadas encima y con más sets; el orden es el mismo |
| Los pesos de BPM y género de solo texto sirven para el vivo | NO CONCLUYENTE | Sin audio ni tonalidad de Traktor; los del vivo salen de E16 |

## Revisión hostil

- **¿El género es fiable?** Es el de Beatport por lanzamiento (G5). Con el de 1001Tracklists coincide en el 39 %; no
  se midió G13 con ese otro género.
- **¿El nulo «mismo set» es justo?** Un set cambia de BPM y de estilo con las horas, así que un tema a 5 filas o más
  ya es de otro momento. Por eso el BPM pesa contra ese nulo y el género casi no.
- **¿Hay DJs que pesan de más?** Cada set pesa 1 y los intervalos remuestrean DJs. Los 317 pares repetidos entre sets
  cuentan una vez por set.
- **¿La identificación es buena?** Cada tema se identificó por el id de Beatport de la búsqueda de la otra sesión. Un
  error de identificación agrega ruido y no inventa orden. No se midió.
- **¿Lo aprendido gana por haber visto los datos?** Se aprende sin el pliegue del DJ, en 10 grupos de DJs.
