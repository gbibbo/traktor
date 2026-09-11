# Baselines de clave y BPM sobre las tripletas del DJ (2026-09-11)

Evidencia: 60 tripletas respondidas por Gabriel con el Anotador DJ (28 B, 29 C, 3 saltadas).
Semántica: "cuál tocaría a continuación del ancla sin salto de estilo" (mezclabilidad).
Features: BPM y tonalidad de Essentia (`phase1_extract.py --essentia-only`, CPU, 2040 s para
243 temas; 1 fallo: un mp3 con error interno de decodificación). 55 tripletas evaluables.

## Acierto de predictores simples

| Predictor | n | Acierto | IC 95 % | Empates |
| :--- | ---: | ---: | :--- | ---: |
| Clave sola (regla armónica aprobada) | 55 | 0.536 | 0.42 – 0.67 | 19 |
| BPM solo (equivalencia 0.5x/1x/2x) | 55 | 0.636 | 0.50 – 0.75 | 0 |
| Clave + BPM (0.5 / 0.5) | 55 | 0.636 | 0.50 – 0.75 | 0 |
| Azar | 55 | 0.500 | | |

Empate: ambos candidatos tienen la misma compatibilidad con el ancla; cuenta 0.5.

Desglose de la clave:

| Medida | Elegido | No elegido |
| :--- | ---: | ---: |
| Acierto de la clave en las 36 tripletas sin empate | 0.556 (IC 0.40 – 0.70) | |
| Compatibilidad armónica media con el ancla | 0.661 | 0.637 |
| Fracción con compatibilidad ≥ 0.60 (lista aprobada) | 0.71 | 0.75 |
| Mediana de la diferencia de BPM con el ancla | 2.0 | 2.9 |

## Lectura

1. **La clave no explica las elecciones.** El candidato elegido no es más compatible
   armónicamente que el descartado. Con esta muestra, el esfuerzo de Gabriel por no dejarse
   guiar por la clave funciona: la contaminación armónica en la evidencia es pequeña. Esto
   simplifica el diseño: el espacio de estilo puede aprenderse de estas tripletas casi sin
   descontar la clave, aunque igual se mantiene la clave como entrada explícita por seguridad.
2. **El BPM sí explica algo.** 0.636 con BPM solo es comparable al mejor resultado del sweep
   de la rama `feature/dj-clustering-v1` con MERT congelado (0.622 sobre 37 tripletas). Todo
   modelo futuro debe superar claramente el baseline de BPM, no el azar.
3. **Los intervalos son anchos.** Con 55 tripletas nada es concluyente. Hacen falta más
   respuestas; el objetivo del plan v6 era 120 antes de seleccionar un modelo.
4. **Posible ruido en la tonalidad.** La confianza mediana de KeyExtractor es 0.87, pero no se
   ha validado contra tags ni contra el oído de Gabriel. Vale la pena una revisión de 10 temas.

## Reproducir

```bash
python src/v4/pipeline/phase0_ingest.py --dataset-name test_20
python src/v4/pipeline/phase1_extract.py --dataset-name test_20 --device cpu --essentia-only
python src/v4/evaluation/triplet_evidence.py --dataset-name test_20
```
