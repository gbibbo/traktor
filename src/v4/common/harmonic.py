"""
PURPOSE: Compatibilidad armónica entre tonalidades según la regla aprobada por Gabriel
         (2026-09-11), reemplazando la regla Camelot básica de V4.
         Relaciones consideradas para un ancla nA (menor) o nB (mayor), con fuerza:
           misma tonalidad ............ 1.0
           relativa (mismo número) .... 0.9
           vecinas en la rueda (n±1) .. 0.7   (mismo modo)
           diagonales (n±1, otro modo)  0.7   (mismo conjunto de notas que las vecinas)
           paralela (misma tónica) .... 0.6   (nA -> (n+3)B ; nB -> (n-3)A)
           relativa de la paralela .... 0.6   (nA -> (n+3)A ; nB -> (n-3)B)
         El candidato puede transponerse hasta MAX_SHIFT semitonos (key lock); cada
         semitono resta SHIFT_PENALTY. Score = max sobre transposiciones, acotado a [0, 1].
         Un semitono equivale a +7 posiciones en la rueda Camelot.
         Tonalidad desconocida -> 0.5 (neutro).
CHANGELOG:
  - 2026-09-11: Creación. Sustituye key_compatibility de phase4_order y _key_compatibility
                de evaluation/metrics (que daban 1.0 / 0.5 / 0.0 sin transposición ni paralela).
"""
from typing import Dict, Optional, Tuple

MAX_SHIFT = 2            # semitonos máximos de transposición (aprobado como máximo absoluto)
SHIFT_PENALTY = 0.15     # penalización por semitono transpuesto
UNKNOWN_SCORE = 0.5      # tonalidad desconocida: neutro

RELATION_STRENGTH: Dict[str, float] = {
    "same": 1.0,
    "relative": 0.9,
    "adjacent": 0.7,
    "diagonal": 0.7,
    "parallel": 0.6,
    "parallel_relative": 0.6,
}

# --- Parsing a Camelot -----------------------------------------------------

_SEMITONE: Dict[str, int] = {
    "C": 0, "B#": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "Fb": 4,
    "F": 5, "E#": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
    "A#": 10, "Bb": 10, "B": 11, "Cb": 11,
}
_MINOR_CAMELOT = [5, 12, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10]   # índice = semitono desde C
_MAJOR_CAMELOT = [8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6, 1]


def to_camelot(key_str: Optional[str]) -> str:
    """Normaliza una tonalidad a Camelot ('5A', '12B'). Acepta 'C minor', 'Cm', 'Eb',
    'F# major', 'Abm', '5A'. Devuelve '?' si no se puede interpretar."""
    if key_str is None:
        return "?"
    s = str(key_str).strip()
    if not s or s.lower() in ("?", "nan", "none", "unknown"):
        return "?"
    # Ya en Camelot
    if len(s) in (2, 3) and s[:-1].isdigit() and s[-1].upper() in ("A", "B") and 1 <= int(s[:-1]) <= 12:
        return f"{int(s[:-1])}{s[-1].upper()}"
    # 'C minor' / 'G major' / 'F# min'
    parts = s.split()
    if len(parts) == 2:
        tonic, mode = parts[0], parts[1].lower()
        semitone = _SEMITONE.get(tonic)
        if semitone is not None:
            if mode.startswith("min"):
                return f"{_MINOR_CAMELOT[semitone]}A"
            if mode.startswith("maj"):
                return f"{_MAJOR_CAMELOT[semitone]}B"
        return "?"
    # 'Cm', 'Abm', 'Ebmaj', 'G'
    low = s.lower()
    if low.endswith("min") or low.endswith("m") and not low.endswith("maj"):
        tonic = s[:-3] if low.endswith("min") else s[:-1]
        semitone = _SEMITONE.get(tonic)
        return f"{_MINOR_CAMELOT[semitone]}A" if semitone is not None else "?"
    if low.endswith("maj"):
        s = s[:-3]
    semitone = _SEMITONE.get(s)
    return f"{_MAJOR_CAMELOT[semitone]}B" if semitone is not None else "?"


def _split(camelot: str) -> Optional[Tuple[int, str]]:
    c = to_camelot(camelot)
    if c == "?":
        return None
    return int(c[:-1]), c[-1]


def _wrap(n: int) -> int:
    return (n - 1) % 12 + 1


def shift_camelot(camelot: str, semitones: int) -> str:
    """Tonalidad resultante de transponer `semitones` (+1 semitono = +7 en la rueda)."""
    parsed = _split(camelot)
    if parsed is None:
        return "?"
    num, mode = parsed
    return f"{_wrap(num + 7 * semitones)}{mode}"


def key_relation(anchor: str, candidate: str) -> Optional[str]:
    """Nombre de la relación armónica entre dos tonalidades sin transponer, o None."""
    a, b = _split(anchor), _split(candidate)
    if a is None or b is None:
        return None
    (na, ma), (nb, mb) = a, b
    diff = (nb - na) % 12
    same_mode = ma == mb
    if diff == 0:
        return "same" if same_mode else "relative"
    if diff in (1, 11):
        return "adjacent" if same_mode else "diagonal"
    # Paralela: menor nA -> (n+3)B ; mayor nB -> (n-3)A
    parallel_diff = 3 if ma == "A" else 9
    if diff == parallel_diff:
        return "parallel_relative" if same_mode else "parallel"
    return None


def best_key_shift(anchor: str, candidate: str,
                   max_shift: int = MAX_SHIFT,
                   shift_penalty: float = SHIFT_PENALTY) -> Tuple[float, int, Optional[str]]:
    """Mejor (score, semitonos a transponer el candidato, relación) sobre transposiciones
    en [-max_shift, max_shift]. Sin transponer gana ante empate. Desconocida -> (0.5, 0, None)."""
    if _split(anchor) is None or _split(candidate) is None:
        return UNKNOWN_SCORE, 0, None
    best = (0.0, 0, None)
    for s in sorted(range(-max_shift, max_shift + 1), key=abs):
        rel = key_relation(anchor, shift_camelot(candidate, s))
        if rel is None:
            continue
        score = max(0.0, RELATION_STRENGTH[rel] - shift_penalty * abs(s))
        if score > best[0]:
            best = (score, s, rel)
    return best


def key_compatibility(anchor: str, candidate: str,
                      max_shift: int = MAX_SHIFT,
                      shift_penalty: float = SHIFT_PENALTY) -> float:
    """Score de compatibilidad armónica en [0, 1] (ver docstring del módulo)."""
    return best_key_shift(anchor, candidate, max_shift, shift_penalty)[0]
