"""
PURPOSE: Constantes centrales de TRAKTOR ML V4.
         Los sample rates son contratos de los modelos upstream. NO modificar sin revisar
         la model card correspondiente.
CHANGELOG:
  - 2026-02-28: Creación inicial V4.
"""
from pathlib import Path

# === Rutas base ===
REPO_ROOT = Path(__file__).resolve().parents[2]  # traktor/
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "v4.yaml"

# === Sample rates (contratos de modelos, NO cambiar) ===
MERT_SAMPLE_RATE = 24000       # MERT-v1-330M espera 24kHz
ESSENTIA_SAMPLE_RATE = 44100   # RhythmExtractor2013 y KeyExtractor requieren 44.1kHz
DEMUCS_SAMPLE_RATE = 44100     # htdemucs opera a 44.1kHz

# === Segmentación (defaults; overrideable via config/v4.yaml > segmentation.*) ===
SEGMENT_DURATION_S = 5.0       # Duración de cada ventana en segundos
SEGMENT_DURATION_BARS = 16     # Alternativa: duración en barras musicales (si hay beat tracking)
N_INTRO_SEGMENTS = 1
N_MID_SEGMENTS = 2
N_OUTRO_SEGMENTS = 1

# === Modelos ===
MERT_MODEL_NAME = "m-a-p/MERT-v1-330M"
MERT_EMBEDDING_DIM = 1024
DEMUCS_MODEL_NAME = "htdemucs"

# === Clustering defaults ===
L1_MIN_CLUSTER_SIZE = 10
L1_MIN_SAMPLES = 3
L2_MIN_CLUSTER_SIZE = 4
L2_MIN_SAMPLES = 2

# === Evaluación ===
RETRIEVAL_K_VALUES = [5, 10, 20]

# === Track ID ===
TRACK_UID_BYTES_TO_READ = 1_048_576  # 1MB para hash parcial (modo fast)

# === Ordering weights (defaults, override desde v4.yaml) ===
ORDERING_WEIGHTS = {"embedding": 0.5, "bpm": 0.3, "key": 0.2}

# === CLAP naming vocabulary (techno/tech house oriented) ===
CLAP_DESCRIPTORS = [
    "dark rolling techno", "melodic progressive techno", "acid techno",
    "minimal deep techno", "hard industrial techno", "tribal percussive techno",
    "atmospheric ambient techno", "peak time techno", "deep hypnotic techno",
    "raw warehouse techno", "dub techno", "Detroit techno",
    "groovy tech house", "funky tech house", "deep tech house",
    "minimal tech house", "vocal tech house", "jackin tech house",
    "afro tech house", "organic house", "progressive house",
    "breaks and electro", "downtempo electronica",
]
