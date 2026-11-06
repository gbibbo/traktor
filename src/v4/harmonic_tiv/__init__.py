"""
PURPOSE: Compatibilidad armónica calculada del audio con Tonal Interval Vectors (retoma la tesis de
         máster de Gabriel, github.com/gbibbo/harmonic_mix). Investigación en la rama harmonic-tiv:
         nada de esto alimenta el pipeline ni las sugerencias en vivo hasta que la escucha lo apruebe
         (docs/plans/armonia_tiv_protocolo.md).
           tiv.py       TIV y métricas de compatibilidad (numpy, vectorizado)
           frontend.py  audio -> espectro por semitono (CPU, sin Essentia) y cromas
           features.py  caché -> TIV por tema según una configuración
           evaluate.py  métricas proxy (teoría de tonalidades, hubness, transiciones reales)
CHANGELOG:
  - 2026-11-06: Creación.
"""
