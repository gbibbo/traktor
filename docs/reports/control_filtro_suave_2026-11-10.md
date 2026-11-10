# Control del filtro flexible de Beatport contra el enlace de 1001Tracklists (2026-11-10)

Pedido de Gabriel (DECISIONS 2026-11-10, punto 1): supervisar exhaustivamente que el filtro nuevo de la búsqueda en
Beatport encuentre la versión correcta. Este control es el que quedó pendiente en el punto 2: el enlace del botón de
compra de 1001Tracklists de los temas que resolvió la comparación flexible de la mezcla (`mezcla_suave = si`).

## Veredicto

En ningún tema la búsqueda flexible eligió otra versión que la del set. Pedidos los 161 temas con `mezcla_suave = si`,
estado `ok` y botón de compra (90 de `rescate_01`, 71 de las 22 tandas hechas de `lote_04`): 3 no tienen enlace a
Beatport; de los 158 con enlace, 127 dan el mismo id y 31 otro. De esos 31, leídos uno por uno con la ficha de
Beatport de los dos ids: 10 enlaces no tienen ficha en Beatport (el tema quitado o movido), 14 son la misma grabación
en otro lanzamiento, en 4 la búsqueda coincide con la versión nombrada igual o mejor que el enlace, 1 es la misma
mezcla en otro largo sin saber cuál sonó, y en 2 la búsqueda tomó un corte corto de la misma mezcla (Yolanda Be Cool,
159 s frente a 283 s; MOGUAI, 187 s frente al Extended de 275 s).

| afirmación | etiqueta | evidencia |
|---|---|---|
| 127 de 158 ids iguales al del enlace | EVIDENCIA OBSERVADA | `beatport_previews/control_suave/temas.csv` (en D:), `enlaces` del 2026-11-10 17:05-17:30 |
| 14 de los 31 distintos son la misma grabación | EVIDENCIA OBSERVADA | 9 con el mismo ISRC; 5 con la misma mezcla y la misma duración (±1 s); tabla de abajo |
| 10 enlaces apuntan a ids sin ficha en Beatport | EVIDENCIA OBSERVADA | `track_details` vacío para el id del enlace |
| en esos 10 la búsqueda tiene la versión del set | INFERENCIA | el nombre de la mezcla coincide con el de 1001Tracklists; no hay segundo id con qué comparar |
| ninguna versión equivocada | INFERENCIA | sale de las cuatro filas anteriores; vale para los temas con botón, no para los 41 sin botón (28 de `rescate_01` y 13 de `lote_04`), que no tienen enlace |
| 2 de 158 (1,3 %) son un corte corto | EVIDENCIA OBSERVADA | filas 21 y 28: mismo nombre de mezcla y otra duración; `match()` no desempata por largo |

## Idea abierta (sin hacer)

Entre candidatos con la misma mezcla, preferir el corte más largo, como ya hace `youtube_temas.py`. Toca `match()`,
que también usa la biblioteca: antes de cambiarlo, medir cuántos temas de los lotes y de la biblioteca cambian de id.

## Los 31 pares con otro id

Columnas: nombre en 1001Tracklists, ficha del id de la búsqueda, ficha del id del enlace, lectura del agente.

| tema | búsqueda | enlace de 1001Tracklists | lectura |
|---|---|---|---|
| Phonique ft. Ian Whitelaw & Meitz - Worked It Out (Charles Webster First Remix) | 384783: Worked It Out feat. Volker Meitz, Ian Whitelaw (Charles Webster's First Remix), 491 s, Good Idea Remixes, ISRC DEL020780047 | 5788697: Worked It Out feat. Volker Meitz, Ian Whitelaw (Charles Webster's First Remix), 491 s, Dessous #Beatportdecade Deep House, ISRC DEL020780047 | misma grabación, otro lanzamiento |
| Tunnelvisions - Guava (Aera Sunset Remix) | 31019737: Guava (Aera's Sunset Remix), 393 s, Guava, ISRC NLC7F1800012 | 10227698: Beatport no da ficha | enlace sin ficha; la búsqueda tiene el nombre del set |
| Pablo Bolivar - Seconds Of Light (Simon Garcia Poltergeist Dub) | 9002967: Seconds of Light (Simon Garcia's Poltergeist Dub), 494 s, Seconds of Light, ISRC DEAR41712835 | 9657983: Beatport no da ficha | enlace sin ficha; la búsqueda tiene el nombre del set |
| Mike Dunn pres. Mr. 69 - Phreaky MF (Mike Dunn Original Phreak Mixx) | 13812141: Phreaky MF (Mike Dunn's Phreak MixX), 729 s, Phreaky MF, ISRC GBCPZ0817666 | 2015059: Beatport no da ficha | enlace sin ficha; la búsqueda tiene el nombre del set |
| DJ Hell - Car Car Car (Phil Kieran Drum's Mix) | 19656724: Car Car Car  (Phil Kieran's Drum Remix), 434 s, Car Car Car, Pt. 1, ISRC DEAR41716422 | 9343227: Beatport no da ficha | enlace sin ficha; la búsqueda tiene el nombre del set |
| Above & Beyond - Believer (Marsh Guatape Remix) | 16078202: Believer (Marsh's Guatape Remix), 256 s, Believer (Marsh's Guatape Remix), ISRC GBEWA2200010 | 16078204: Believer (Marsh's Guatape Remix [Above & Beyond's Cercle Respray]), 409 s, Believer (Marsh's Guatape Remix), ISRC GBEWA2200012 | búsqueda = la versión nombrada; el enlace es un «Cercle Respray» |
| Monkey Safari - Walls (Guy Gerber Office Sex Remix) | 6869396: Walls (Guy Gerber's Office Sex Remix), 585 s, Walls, ISRC DEQ121534017 | 7515815: Beatport no da ficha | enlace sin ficha; la búsqueda tiene el nombre del set |
| Joe T Vannelli ft. Rochelle Fleming - Get It On (Joe T Vannelli Dubby Remix) | 28108381: Get It On (Joe T Vannelli Dubby Mix), 680 s, Get It On (feat. Rochelle Fleming) (Summer Love), ISRC ITA160501026 | 4742629: Beatport no da ficha | enlace sin ficha; la búsqueda tiene el nombre del set |
| GoldFish & Der Dulz - Privacy (Patrick Chardronnet Remix) | 392780: Privacy (Patrick Chardronnet Mix), 481 s, Boxer 50, ISRC None | 8441951: Beatport no da ficha | enlace sin ficha; la búsqueda tiene el nombre del set |
| SFire - SFire3 (John Talabot Tribalist Rework) | 14082889: Sfire3 (John Talabot's Tribalist Rework), 500 s, Sfire 3 (John Talabot Remixes), ISRC GBKQU2086761 | 8342758: Sfire 3 (John Talabot's Tribalist Rework), 500 s, Sfire 3 (John Talabot Remixes), ISRC DEEC31690001 | mismo lanzamiento y duración: la misma grabación cargada dos veces |
| Uffie ft. Pharrell Williams - Add Suv (Armand van Helden Club Mix) | 1205183: Add Suv feat. Pharrell Williams (Armand Van Helden Club Remix), 285 s, ADD SUV (Armand Van Helden Remix), ISRC FR0NT1000320 | 1216049: ADD SUV (feat. Pharrell Williams) (Armand Van Helden Club Remix), 285 s, ADD SUV (feat. Pharrell Williams), ISRC FR0NT1000320 | misma grabación, otro lanzamiento |
| Above & Beyond - Sun In Your Eyes (Marsh Remix) | 22006157: Sun In Your Eyes (Marsh Extended Mix), 460 s, Sun In Your Eyes (Marsh Remix), ISRC GBEWA2505371 | 22384919: Sun In Your Eyes (Marsh Extended Mix), 460 s, 25 Years Of Anjuna Mixed By Marsh, ISRC GBEWA2505371 | misma grabación, otro lanzamiento |
| Starsailor - Four To The Floor (Thin White Duke Remix) | 27041535: Four to the Floor (Thin White Duke Mix), 492 s, Good Souls: The Greatest Hits, ISRC GBAYE0302593 | 22077286: Four to the Floor (Thin White Duke Mix), 491 s, Silence Is Easy (20th Anniversary Edition), ISRC GBAYE0302697 | misma ficha, 1 s de diferencia (reedición) |
| Afefe Iku - Mirror Dance (Yoruba Soul Mix) | 1265958: Mirror Dance feat. Oveous Maximus (Yoruba Soul Remix), 629 s, Mirror Dance (Yoruba Soul Remix), ISRC ITK140800045 | 1165950: Mirror Dance (Yoruba Soul Remix feat. Oveous Maximus), 629 s, Osunlade Presents Yoruba Gold, ISRC ITK140800045 | misma grabación, otro lanzamiento |
| BBE - Seven Days And One Week (YOTTO Remix) | 11393250: Seven Days And One Week (Yotto Extended Mix), 480 s, Seven Days And One Week (Yotto Remix), ISRC BEN581800624 | 11056532: Seven Days And One Week (Yotto Extended Mix), 480 s, Seven Days And One Week (Yotto Extended Mix), ISRC BEN581800624 | misma grabación, otro lanzamiento |
| Blu Peter - Funky Suite (Oliver Lieb Remix) | 9126694: Funky Suite (Oliver Lieb Mix), 410 s, React Singles Club, ISRC GBHCD1604111 | 4765780: Funky Suite (Oliver Leib Remix), 311 s, Funky Suite, ISRC GBBEV0000293 | misma mezcla, otro largo; no se sabe cuál sonó |
| Chris Liebing & Ralf Hildenbeutel ft. Polly Scattergood - And All Went Dark (Mark Broom Vision Remix) | 13029370: And All Went Dark (feat. Polly Scattergood & Ralf Hildenbeutel) (Mark Broom's Vision Remix), 436 s, BURN SLOW REMIXES PT. I, ISRC GBR8R1900351 | 13144371: Beatport no da ficha | enlace sin ficha; la búsqueda tiene el nombre del set |
| Sasha ft. Polica - Out Of Time (Patrice Bäumel Remix) | 18281323: Out Of Time (Patrice Bäumel Mix), 509 s, Van Czar Series, Vol. 10, ISRC DEU671700065 | 9210528: Out Of Time (Patrice Baumel Mix), 509 s, Out Of Time (feat. Polica), ISRC DEU671700065 | misma grabación, otro lanzamiento |
| Audion ft. Troels Abrahamsen - Dem Howl (Joris Voorn Remix) | 6064718: Dem Howl Feat. Troels Abrahamsen (Joris Voorn Mix), 455 s, Balance Presents Kölsch, ISRC AUXN21423211 | 5935963: Dem Howl Feat. Troels Abrahamsen (Joris Voorn Mix), 455 s, Dem Howl feat. Troels Abrahamsen, ISRC DEU671400277 | misma mezcla y duración (compilado) |
| Cajmere - Percolator (Jamie Jones Vault Remix) | 4390720: Percolator (Jamie Jones Vault Mix), 302 s, Too Underground For The Main Stage, ISRC USCEI1135313 | 5802990: Percolator (Jamie Jones Remix), 302 s, Cajual Records #BeatportDecade House, ISRC USCEI1137301 | misma duración; el compilado le quita «Vault» |
| Yolanda Be Cool - Space Jam (Redondo Disco Jam Remix) | 27317831: Space Jam (Redondo's Disco Jam Remix), 159 s, Space Jam (Remixes), ISRC FR59R1948225 | 12253559: Space Jam (Redondo's Disco Jam Remix), 283 s, Space Jam (Remixes), ISRC FR59R1948223 | misma mezcla; la búsqueda tomó el corte corto |
| Brian Eno - An Ending (Leama & Moor Remix) | 179024: An Ending (Leama & Moor Mix), 626 s, An Ending, ISRC None | 311571: Ascent (An Ending) (Leama and Moor Mix), 626 s, Ascent (An Ending), ISRC GBHCD0407075 | misma mezcla y duración; el título varía |
| Shredder SA & ADJUMA ft. Lyrical Ivy - We Belong (Hyenah So Much Love Remix) | 20038278: We Belong (Hyenah's So Much Love Remix), 482 s, "Echoes From the Moon": Ultimate Hits Collection, ISRC ITBVO2200187 | 16606591: We Belong (Hyenah‘s So Much Love Remix), 482 s, We Belong, ISRC ITBVO2200187 | misma grabación, otro lanzamiento |
| RÜFÜS DU SOL - Underwater (YOTTO Dusk Mix) | 12057549: Underwater (Yotto's Dusk Remix Extended), 411 s, SOLACE REMIXES VOL. 6, ISRC USRE11900264 | 12057550: Underwater (Yotto's Dusk Remix), 302 s, SOLACE REMIXES VOL. 6, ISRC USRE11900263 | misma mezcla; búsqueda el corte extendido, enlace el corto |
| Robbie Rivera - Feel This (Robbie Rivera Tribal Sessions Mix) | 23628916: Feel This (Robbie Rivera's Tribal Sessions Mix), 475 s, Feel This, ISRC USSR30161101 | 9812908: Beatport no da ficha | enlace sin ficha; la búsqueda tiene el nombre del set |
| Grum ft. Natalie Shay - Stay (Genix Remix) | 13637766: Stay (Genix Extended Mix), 361 s, Deep State Deluxe, ISRC GBEWA1903321 | 12626793: Stay (feat. Natalie Shay) (Genix Extended Mix), 361 s, Stay (feat. Natalie Shay) [Genix Remix], ISRC GBEWA1903321 | misma grabación, otro lanzamiento |
| Nina Simone - See-Line Woman (Mochakk Remix) | 28914842: See-Line Woman (Mochakk Mix), 288 s, See-Line Woman (Mochakk Mix), ISRC USUM72602636 | 30861821: See-Line Woman (Mochakk Mix), 288 s, Verve Remixed // Nina Simone, ISRC USUM72602636 | misma grabación, otro lanzamiento |
| MOGUAI & Nora Van Elken & Lohrasp Kansara ft. Norman Alexander - New Day (Lohrasp Kansara Bright Lights Remix) | 18585855: New Day (Lohrasp Kansara's Bright Lights Remix), 187 s, New Day, ISRC FR10S2480296 | 18585859: New Day (Lohrasp Kansara's Bright Lights Extended Remix), 275 s, New Day, ISRC FR10S2480297 | misma mezcla; la búsqueda tomó el corte corto (el enlace, Extended) |
| JohNick - Magic (Armand van Helden Remix) | 1946229: Magic (Armand Van Helden Mix), 284 s, Magic (Armand Van Helden Continuous DJ Mix), ISRC USRK31100008 | 1946235: Magic (Armand Van Helden), 3433 s, Magic (Armand Van Helden Continuous DJ Mix), ISRC USRK31100015 | el enlace es un DJ mix continuo de 57 min; búsqueda correcta |
| Banco De Gaia - Obsidian (Framewerk Full On Remix) | 16896031: Obsidian (Framewerk Full On MIx), 565 s, Capital Heaven Record Box, Vol. 7, ISRC FRS182124930 | 16053146: Beatport no da ficha | enlace sin ficha; la búsqueda tiene el nombre del set |
| Carl Cox - Dr. Funk (Riva Starr Mo' Disco Mix) | 12030789: Dr. Funk (Riva Starr Mo' Disco Remix), 409 s, Dr. Funk (Riva Starr Remixes), ISRC AUXN21916248 | 12850584: Dr Funk (Riva Starr Mo' Funk Mix), 360 s, Defected presents Most Rated 2020, ISRC AUXN21916248 | mismo ISRC pero otra duración y otro nombre; la búsqueda coincide con el nombre del set |

Cómo se hizo: `beatport_previews.py --corrida control_suave enlaces` (161 pedidos, 8 s entre pedidos, con 1001Tracklists
libre) y la ficha de Beatport de los dos ids de cada par distinto (31 pedidos). Los scripts quedaron en la carpeta
temporal de la sesión; los datos, en `D:\TRAKTOR ML datos\dj_setseatport_previews\control_suave\`.
