"""
PURPOSE: Métricas de evaluación puras para clustering y retrieval de tracks.
         Funciones sin side effects. Compatibles con numpy arrays.
         ARI/NMI requieren ground truth; noise_rate y cluster stats no lo requieren.
CHANGELOG:
  - 2026-02-28: Creación inicial V4 (Block 3).
"""
import numpy as np
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Clustering metrics (requieren ground truth)
# ---------------------------------------------------------------------------

def clustering_ari(labels_pred: np.ndarray, labels_true: np.ndarray) -> float:
    """Adjusted Rand Index entre labels predichos y verdaderos.

    Labels -1 (ruido) se excluyen de ambas máscaras antes del cálculo
    (solo se evalúa sobre los puntos asignados).

    Returns:
        ARI en [-1, 1]. 1.0 = perfecto, 0.0 = aleatorio.
    """
    from sklearn.metrics import adjusted_rand_score
    mask = (labels_pred != -1) & (labels_true != -1)
    if mask.sum() < 2:
        return 0.0
    return float(adjusted_rand_score(labels_true[mask], labels_pred[mask]))


def clustering_nmi(labels_pred: np.ndarray, labels_true: np.ndarray) -> float:
    """Normalized Mutual Information entre labels predichos y verdaderos.

    Labels -1 (ruido) se excluyen antes del cálculo.

    Returns:
        NMI en [0, 1]. 1.0 = perfecto.
    """
    from sklearn.metrics import normalized_mutual_info_score
    mask = (labels_pred != -1) & (labels_true != -1)
    if mask.sum() < 2:
        return 0.0
    return float(normalized_mutual_info_score(labels_true[mask], labels_pred[mask],
                                               average_method="arithmetic"))


# ---------------------------------------------------------------------------
# Retrieval metrics (requieren positive_pairs)
# ---------------------------------------------------------------------------

def _pairwise_cosine_distances(embeddings: np.ndarray) -> np.ndarray:
    """Matrix de distancias coseno (N, N). Diagonal = 0."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    normalized = embeddings / norms
    sim = normalized @ normalized.T
    return 1.0 - np.clip(sim, -1.0, 1.0)


def retrieval_recall_at_k(
    embeddings: np.ndarray,
    positive_pairs: List[Tuple[int, int]],
    k: int = 10,
) -> float:
    """Recall@k: fracción de pares positivos cuyos vecinos más cercanos incluyen al par.

    Args:
        embeddings: (N, D) array normalizado o no.
        positive_pairs: lista de (i, j) donde j debería aparecer en top-k de i.
        k: número de vecinos a considerar.

    Returns:
        Recall@k en [0, 1].
    """
    if not positive_pairs:
        return 0.0
    dist_matrix = _pairwise_cosine_distances(embeddings)
    hits = 0
    for i, j in positive_pairs:
        row = dist_matrix[i].copy()
        row[i] = np.inf  # excluir self
        topk = np.argpartition(row, min(k, len(row) - 1))[:k]
        if j in topk:
            hits += 1
    return hits / len(positive_pairs)


def retrieval_mrr(
    embeddings: np.ndarray,
    positive_pairs: List[Tuple[int, int]],
) -> float:
    """Mean Reciprocal Rank para recuperación por similitud.

    Returns:
        MRR en [0, 1].
    """
    if not positive_pairs:
        return 0.0
    dist_matrix = _pairwise_cosine_distances(embeddings)
    reciprocal_ranks = []
    for i, j in positive_pairs:
        row = dist_matrix[i].copy()
        row[i] = np.inf
        rank = int(np.sum(row <= row[j]))  # cuántos elementos tienen distancia <= dist(i,j)
        rank = max(rank, 1)
        reciprocal_ranks.append(1.0 / rank)
    return float(np.mean(reciprocal_ranks))


def retrieval_ndcg_at_k(
    embeddings: np.ndarray,
    positive_pairs: List[Tuple[int, int]],
    k: int = 10,
) -> float:
    """NDCG@k para retrieval de pares positivos.

    Asume relevancia binaria: 1 si j está en top-k, 0 en caso contrario.

    Returns:
        NDCG@k en [0, 1].
    """
    if not positive_pairs:
        return 0.0
    dist_matrix = _pairwise_cosine_distances(embeddings)
    ndcg_scores = []
    for i, j in positive_pairs:
        row = dist_matrix[i].copy()
        row[i] = np.inf
        sorted_indices = np.argsort(row)[:k]
        dcg = 0.0
        for rank_idx, idx in enumerate(sorted_indices, start=1):
            if idx == j:
                dcg = 1.0 / np.log2(rank_idx + 1)
                break
        idcg = 1.0  # ideal: j en posición 1
        ndcg_scores.append(dcg / idcg)
    return float(np.mean(ndcg_scores))


def pairwise_auc(
    embeddings: np.ndarray,
    positive_pairs: List[Tuple[int, int]],
    n_negatives: int = 1000,
) -> float:
    """AUC por pares: probabilidad de que un par positivo tenga menor distancia que un negativo.

    Para cada par positivo (i, j), muestrea n_negatives índices aleatorios k != i, k != j
    y calcula P(dist(i,j) < dist(i,k)).

    Returns:
        AUC en [0, 1]. 0.5 = aleatorio.
    """
    if not positive_pairs:
        return 0.5
    N = len(embeddings)
    rng = np.random.default_rng(42)
    dist_matrix = _pairwise_cosine_distances(embeddings)
    wins = []
    for i, j in positive_pairs:
        d_pos = dist_matrix[i, j]
        neg_indices = rng.integers(0, N, size=n_negatives * 2)
        neg_indices = neg_indices[(neg_indices != i) & (neg_indices != j)][:n_negatives]
        if len(neg_indices) == 0:
            continue
        d_neg = dist_matrix[i, neg_indices]
        wins.append(float(np.mean(d_pos < d_neg)))
    return float(np.mean(wins)) if wins else 0.5


# ---------------------------------------------------------------------------
# Ordering / transition metrics
# ---------------------------------------------------------------------------

def transition_score(
    ordering: List[int],
    embeddings: np.ndarray,
    bpm: np.ndarray,
    keys: List[str],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Score de calidad de transición en un ordering de tracks.

    Combina tres sub-scores ponderados:
      - embedding_score: similitud coseno media entre tracks consecutivos (↑ mejor)
      - bpm_score: 1 - |Δbpm| / max_bpm_range — penaliza saltos grandes de BPM
      - key_score: compatibilidad Camelot wheel entre tracks consecutivos

    Args:
        ordering: lista de índices de tracks en orden de reproducción.
        embeddings: (N, D) array de embeddings (usados con ordenación).
        bpm: (N,) array de BPMs.
        keys: lista de N strings de tonalidades (ej. "Cm", "G#m", "3A").
        weights: dict con keys 'embedding', 'bpm', 'key'. Default: ORDERING_WEIGHTS.

    Returns:
        Score combinado en [0, 1]. Mayor = mejor.
    """
    from src.v4.config import ORDERING_WEIGHTS
    if weights is None:
        weights = ORDERING_WEIGHTS

    if len(ordering) < 2:
        return 1.0

    # Normalizar embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    emb_norm = embeddings / norms

    emb_scores, bpm_scores, key_scores = [], [], []
    bpm_vals = np.array(bpm, dtype=float)
    bpm_range = float(bpm_vals.max() - bpm_vals.min()) if bpm_vals.max() > bpm_vals.min() else 1.0

    for a, b in zip(ordering[:-1], ordering[1:]):
        # Embedding similarity
        cos_sim = float(np.dot(emb_norm[a], emb_norm[b]))
        emb_scores.append((cos_sim + 1.0) / 2.0)  # remap [-1,1] → [0,1]

        # BPM smoothness
        delta_bpm = abs(bpm_vals[a] - bpm_vals[b])
        bpm_scores.append(max(0.0, 1.0 - delta_bpm / bpm_range))

        # Key compatibility (Camelot wheel)
        key_scores.append(_key_compatibility(keys[a], keys[b]))

    total = (
        weights.get("embedding", 0.5) * float(np.mean(emb_scores))
        + weights.get("bpm", 0.3) * float(np.mean(bpm_scores))
        + weights.get("key", 0.2) * float(np.mean(key_scores))
    )
    return float(np.clip(total, 0.0, 1.0))


# Camelot wheel: lista de 24 tonalidades en orden de rueda (1A-12A minor, 1B-12B major)
_CAMELOT_ORDER = [
    "Am", "Em", "Bm", "F#m", "C#m", "G#m", "D#m", "A#m", "Fm", "Cm", "Gm", "Dm",  # A (minor)
    "C",  "G",  "D",  "A",   "E",   "B",   "F#",  "C#",  "Ab", "Eb", "Bb", "F",   # B (major)
]

# Mapas alternativos de notación
_CAMELOT_ALIASES: Dict[str, str] = {
    "Abm": "G#m", "Ebm": "D#m", "Bbm": "A#m",
    "Db": "C#", "Gb": "F#", "Cb": "B",
    # numeric Camelot notation: "1A" -> "Am", etc.
    **{f"{i+1}A": _CAMELOT_ORDER[i] for i in range(12)},
    **{f"{i+1}B": _CAMELOT_ORDER[i + 12] for i in range(12)},
}


def _key_compatibility(key_a: str, key_b: str) -> float:
    """Compatibilidad Camelot wheel entre dos tonalidades.

    Returns:
        1.0 = misma tonalidad o vecinos directos en la rueda
        0.5 = un paso en cualquier dirección
        0.0 = tonalidades incompatibles o desconocidas
    """
    def _resolve(k: str) -> Optional[int]:
        k = k.strip()
        k = _CAMELOT_ALIASES.get(k, k)
        try:
            return _CAMELOT_ORDER.index(k)
        except ValueError:
            return None

    idx_a = _resolve(key_a)
    idx_b = _resolve(key_b)
    if idx_a is None or idx_b is None:
        return 0.0

    # Índices dentro de su anillo (A o B, cada uno de 12 posiciones)
    ring_a, pos_a = divmod(idx_a, 12)
    ring_b, pos_b = divmod(idx_b, 12)

    if ring_a == ring_b:
        # Mismo anillo: distancia circular
        diff = min(abs(pos_a - pos_b), 12 - abs(pos_a - pos_b))
    else:
        # Anillos distintos: solo compatible si mismo número (relativa mayor/menor)
        if pos_a == pos_b:
            return 1.0
        diff = min(abs(pos_a - pos_b), 12 - abs(pos_a - pos_b))

    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Noise / cluster stats
# ---------------------------------------------------------------------------

def noise_rate(labels: np.ndarray) -> float:
    """Fracción de labels == -1 (puntos de ruido HDBSCAN).

    Args:
        labels: array de enteros con -1 indicando ruido.

    Returns:
        Proporción en [0, 1]. 0 = ningún punto de ruido.
    """
    labels = np.asarray(labels)
    if len(labels) == 0:
        return 0.0
    return float(np.sum(labels == -1) / len(labels))


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

def composite_score(
    metrics_dict: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Score compuesto ponderado de un dict de métricas.

    Args:
        metrics_dict: dict métrica → valor numérico.
        weights: dict métrica → peso. Si None, promedio simple.

    Returns:
        Score ponderado en el rango de las métricas (típicamente [0, 1]).
    """
    if not metrics_dict:
        return 0.0
    if weights is None:
        return float(np.mean(list(metrics_dict.values())))
    total_weight = sum(weights.get(k, 0.0) for k in metrics_dict)
    if total_weight == 0:
        return 0.0
    score = sum(metrics_dict[k] * weights.get(k, 0.0) for k in metrics_dict)
    return float(score / total_weight)
