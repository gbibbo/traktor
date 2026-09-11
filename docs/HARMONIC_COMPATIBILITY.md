# Compatibilidad armónica (regla aprobada 2026-09-11)

Implementación: `src/v4/common/harmonic.py`. Reemplaza la regla Camelot básica de V4
(1.0 mismo número, 0.5 vecinas, 0.0 resto) en `phase4_order.py` y `evaluation/metrics.py`.

## Relaciones entre el ancla y un candidato sin transponer

Para un ancla `nA` (menor) o `nB` (mayor):

| Relación | Candidato | Fuerza |
| :--- | :--- | ---: |
| Misma tonalidad | `nA` | 1.00 |
| Relativa | `nB` | 0.90 |
| Vecinas en la rueda | `(n±1)A` | 0.70 |
| Diagonales (mismas notas que las vecinas) | `(n±1)B` | 0.70 |
| Paralela (misma tónica) | `(n+3)B` para menor, `(n−3)A` para mayor | 0.60 |
| Relativa de la paralela | `(n+3)A` para menor, `(n−3)B` para mayor | 0.60 |

## Transposición del candidato

El candidato puede transponerse con key lock hasta 2 semitonos (máximo absoluto, aprobado como
simplificación; en la práctica depende del material). Un semitono equivale a +7 posiciones en la
rueda Camelot. Cada semitono transpuesto resta 0.15.

```text
score(ancla, candidato) = max over s in {-2..2} of  fuerza(relación(ancla, candidato + s)) - 0.15·|s|
```

Tonalidad desconocida en cualquiera de los dos: 0.5 (neutro).

## Ejemplo: ancla 12A (Do# menor)

| Candidato | Relación tras transponer | Semitonos | Score |
| :--- | :--- | ---: | ---: |
| 12A | misma | 0 | 1.00 |
| 5A, 7A | misma | +1, −1 | 0.85 |
| 10A, 2A | misma | +2, −2 | 0.70 |
| 12B | relativa | 0 | 0.90 |
| 5B, 7B | relativa | +1, −1 | 0.75 |
| 10B, 2B | relativa | +2, −2 | 0.60 |
| 11A, 1A | vecinas | 0 | 0.70 |
| 11B, 1B | diagonales | 0 | 0.70 |
| 3B | paralela (Do# mayor) | 0 | 0.60 |
| 3A | relativa de la paralela (Si♭ menor) | 0 | 0.60 |
| 4A, 6A, 8A, 4B, 6B, 8B | vecina o diagonal | ±1 | 0.55 |
| 9A, 9B | vecina o diagonal | ±2 | 0.40 |

El máximo elige siempre la mejor combinación relación + transposición. Con este esquema las 24
tonalidades reciben un score mayor que cero (mínimo 0.40, para 9A y 9B). El ordenamiento y el
recomendador usan el valor continuo, no un umbral; si hace falta un corte duro, la lista aprobada
por Gabriel corresponde a score ≥ 0.60.

## Uso en el pipeline

* Fase 4 (ordenamiento): `key_compatibility(camelot_actual, camelot_candidato)` dentro del score
  mixto con pesos `ordering.weights`.
* Métricas: `transition_score` usa la misma función.
* Futuro recomendador de siguiente tema: `best_key_shift` devuelve además cuántos semitonos
  transponer el candidato y qué relación se usó, para mostrarlo al DJ.

Requisito: la tonalidad de cada tema debe venir de análisis (Essentia KeyExtractor), no de tags.
