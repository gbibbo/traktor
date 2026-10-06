# Detección de voz: protocolo de evaluación

Fecha: 2026-09-29. Pedido de Gabriel: reemplazar la etiqueta Vocal actual por un método validado.
Código: `src/v4/evaluation/vocal_eval.py`. Resultados: `artifacts/v4/vocal_eval/report_<detector>.json`.

## Pregunta

¿Qué temas tienen voz (canto, rap o habla como parte del tema)? La etiqueta actual (CLAP zero-shot sobre
90 s del medio, umbral 0.145 elegido con 23 temas) no está validada: marca 31 % de la colección y no
separa los instrumentales del promedio.

## Definición operativa

- **Por segundo:** hay actividad vocal (canto, rap o habla) en ese segundo.
- **Por tema:** segundos con voz >= T → «Vocal». Se guardan los segundos y la fracción, así T se
  cambia sin recalcular. T se calibra con etiquetas humanas por tema (fase 2), no a ojo.

## Datos de evaluación (públicos, etiquetados por personas)

1. **Electrobyte** (Romero-Arenas et al. 2022; Zenodo 6757945, CC BY 4.0; en
   `data/public/electrobyte/`, MD5 aebbfa243dacbbd53852a4c417f7b815). 90 temas de música electrónica
   con voz / sin voz por tramos, partición 60/15/15. El umbral se elige en valid y se mide en test.
   Límite: es electrónica con canción (estilo NCS/Monstercat) y todos los temas tienen voz (≈ 50 % de
   los segundos). Mide la detección dentro del tema, incluidos los drops con sintes, pero no temas
   instrumentales enteros.
2. **Fase 2, MTG-Jamendo `voice_instrumental`**: 2070 temas con 3 anotadores (87 % de acuerdo),
   filtrados a géneros electrónicos. Es una etiqueta por tema e incluye instrumentales, así que sirve
   para calibrar T y medir por tema. Hay que bajar el audio de esos temas.
3. Opcional: corpus Jamendo (Ramona et al. 2008), para comparar con la literatura (exactitud 82 %
   en 2008, 92 % en 2015).

## Detectores (preentrenados, sin entrenar con la colección)

| Nombre | Qué es | Puntaje |
|---|---|---|
| `clap` | El detector actual (zero-shot), pero sobre todo el tema | P(voz) por ventana de 10 s, paso de 5 s |
| `ast` | Etiquetador AudioSet (Gong et al. 2021) | Máximo de las clases de voz (Singing, Rapping, Speech, Vocal music…) por ventana de 10 s, paso de 5 s |
| `hdemucs` | Separación de fuentes (Défossez 2021; torchaudio `HDEMUCS_HIGH_MUSDB_PLUS`) | Energía del stem de voz relativa a la mezcla (dB) por segundo |
| `clap_probe` | Clasificador lineal (regresión logística) sobre los embeddings CLAP, entrenado con Electrobyte train | P(voz) por ventana de 10 s, paso de 5 s; C elegido por AUC en valid |

Si hace falta, después: AST sobre el stem de voz (menos falsos positivos por sintes), PANNs, o los
clasificadores `voice_instrumental` de Essentia (hay ONNX; requiere conversión).
Descartado: Silero VAD (es un detector de habla; encontró voz solo en el 13 % de 23 temas con canto).

## Métricas y reglas

- **Por segundo:** AUC (sin umbral). Con el umbral de máxima exactitud balanceada elegido en valid:
  exactitud, exactitud balanceada, precisión, recall y F1 en test, con IC del 95 % por bootstrap
  sobre temas.
- Nada se ajusta mirando test. Una sola variable por comparación.
- **Riesgos conocidos** (Lee et al. 2018; Schlüter y Lehner 2018; Stoller et al. 2018):
  - sintes que imitan la voz dan falsos positivos;
  - el volumen puede funcionar como atajo;
  - el stem separado puede tener fugas en los tramos instrumentales.

## Costo medido (piloto, CPU de la laptop)

| Detector | Tiempo por tema de Electrobyte (3-4 min) |
|---|---|
| clap | 5-10 s |
| ast | 60-75 s |
| hdemucs | ≈ 60 s (≈ 0.3 veces la duración) |

Para 1786 temas de unos 7 min, hdemucs serían ≈ 50 h de CPU. Si gana, antes de correrlo se mide en
Electrobyte cuánto se pierde procesando solo algunos tramos de cada tema.

## Referencias

- Ramona, Richard y David, «Vocal detection in music with support vector machines», ICASSP 2008.
- Schlüter y Grill, «Exploring data augmentation for improved singing voice detection with neural
  networks», ISMIR 2015.
- Lee, Choi y Nam, «Revisiting singing voice detection: a quantitative review and the future
  outlook», ISMIR 2018.
- Schlüter y Lehner, «Zero-mean convolutions for level-invariant singing voice detection», ISMIR 2018.
- Stoller, Ewert y Dixon, «Jointly detecting and separating singing voice: a multi-task approach»,
  LVA/ICA 2018.
- Romero-Arenas, Gómez-Espinosa y Valdés-Aguirre, «Singing voice detection in electronic music with a
  long-term recurrent convolutional network», Applied Sciences 12(15):7405, 2022 (Electrobyte).
- Gemmeke et al., «Audio Set», ICASSP 2017. Gong, Chung y Glass, «AST», Interspeech 2021.
- Défossez, «Hybrid spectrogram and waveform source separation», MDX @ ISMIR 2021.
- Monir, Kostrzewa y Mrozek, «Singing voice detection: a survey», Entropy 24:114, 2022.

## Resultados

### Electrobyte, por segundo (umbral elegido en valid, métricas en test; IC 95 % por bootstrap sobre temas)

Test: 15 temas, 3245 segundos, 49.5 % con voz.

| Detector | AUC | Exactitud balanceada | Precisión | Recall | F1 | Umbral (de valid) |
|---|---|---|---|---|---|---|
| hdemucs | 0.913 (0.872–0.947) | 0.842 (0.792–0.885) | 0.850 | 0.824 | 0.837 | −10.4 dB |
| clap_probe | 0.876 (0.838–0.909) | 0.816 (0.765–0.855) | 0.790 | 0.854 | 0.821 | 0.49 |
| ast | 0.824 (0.760–0.874) | 0.771 (0.712–0.813) | 0.732 | 0.847 | 0.785 | 0.022 |
| clap (zero-shot, tema entero) | 0.807 (0.734–0.868) | 0.735 (0.662–0.797) | 0.734 | 0.729 | 0.731 | 0.27 |

Lectura: HDemucs, sin entrenar con estos datos, es el mejor y su intervalo no se superpone con el de
CLAP zero-shot. El clasificador lineal sobre CLAP mejora mucho al zero-shot (AUC 0.95 por ventana en
valid). Las ventanas de 10 s tienen menos resolución en los bordes que la grilla de 1 s de HDemucs, y
eso favorece a HDemucs en esta métrica por segundo.

### MTG-Jamendo, por tema (en curso)

238 temas de baile (119 con voz y 119 instrumentales, etiqueta unánime de 3 anotadores), mitad dev
para elegir T y mitad test. Un tema es «con voz» si sus segundos con voz (puntaje por segundo >= umbral de
Electrobyte valid) superan T; T se elige en dev (máxima exactitud balanceada) y se mide en test
(118 temas). IC 95 % por bootstrap sobre temas.

| Detector | Regla por tema (T de dev) | AUC test | Exactitud balanceada test | Precisión | Recall |
|---|---|---|---|---|---|
| clap_probe | fracción del tema con voz >= 25.5 % | 0.979 (0.954–0.997) | 0.941 (0.895–0.981) | 0.933 | 0.949 |
| clap_probe | segundos con voz >= 31.7 | 0.951 (0.899–0.990) | 0.932 (0.887–0.975) | 0.892 | 0.983 |
| clap (zero-shot, tema entero) | fracción >= 1.7 % | 0.887 (0.827–0.938) | 0.805 (0.738–0.870) | 0.750 | 0.915 |
| hdemucs | (en curso) | | | | |

Lectura parcial: el clasificador lineal sobre CLAP, entrenado solo con Electrobyte, generaliza a otra
fuente (Jamendo, otros géneros de baile y temas instrumentales enteros): 94 % de exactitud balanceada
por tema, contra 80 % del zero-shot. Cuesta ≈ 10-15 s por tema en CPU.
