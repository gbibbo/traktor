"""
PURPOSE: Extracción de embeddings musicales con MERT-v1-330M para TRAKTOR ML V4.
         El modelo espera audio a 24kHz. NO cargar en el nodo de login — usar Slurm GPU.
CHANGELOG:
  - 2026-02-28: Creación inicial V4.
"""
from typing import List, Optional

import numpy as np

from src.v4.config import MERT_MODEL_NAME, MERT_SAMPLE_RATE, MERT_EMBEDDING_DIM


class MERTEmbedder:
    """
    Extractor de embeddings usando MERT-v1-330M.

    Audio de entrada debe ser a 24kHz (MERT_SAMPLE_RATE).
    Usar en GPU (Slurm job con partición a100).
    """

    def __init__(
        self,
        model_name: str = MERT_MODEL_NAME,
        device: str = "cuda",
        hf_cache: Optional[str] = None,
    ):
        """
        Cargar Wav2Vec2FeatureExtractor y AutoModel.

        Args:
            model_name: Nombre del modelo HuggingFace.
            device: 'cuda' o 'cpu'.
            hf_cache: Ruta de cache HuggingFace (opcional; si None usa HF_HOME env).
        """
        import torch
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        self.device = device
        self.model_name = model_name
        self.sample_rate = MERT_SAMPLE_RATE

        kwargs = {"trust_remote_code": True}
        if hf_cache:
            kwargs["cache_dir"] = hf_cache

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, **kwargs)
        self.model = AutoModel.from_pretrained(model_name, **kwargs)
        self.model.to(device)
        self.model.eval()

    def embed_audio(self, audio_24k: np.ndarray) -> np.ndarray:
        """
        Extraer embedding de un array de audio.

        Args:
            audio_24k: numpy float32 array mono a 24kHz.

        Returns: numpy array (1024,) — mean pool del last hidden state.
        """
        import torch

        inputs = self.processor(
            audio_24k,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Last hidden state: (batch=1, time_steps, hidden_dim=1024)
        hidden = outputs.last_hidden_state.squeeze(0)  # (time_steps, 1024)
        embedding = hidden.mean(dim=0).cpu().numpy().astype(np.float32)
        return embedding  # (1024,)

    def embed_segments(self, segments: List[np.ndarray]) -> np.ndarray:
        """
        Extraer embeddings de múltiples segmentos.

        Args:
            segments: Lista de arrays numpy float32 mono a 24kHz.

        Returns: numpy array (n_segments, 1024).
        """
        embeddings = [self.embed_audio(seg) for seg in segments]
        return np.stack(embeddings, axis=0)  # (n_segments, 1024)

    def aggregate_segments(
        self,
        segment_embeddings: np.ndarray,
        method: str = "mean",
    ) -> np.ndarray:
        """
        Agregar embeddings de segmentos a un vector por track.

        Args:
            segment_embeddings: (n_segments, 1024)
            method: "mean" → (1024,) | "mean_std" → (2048,)

        Returns: numpy array agregado.
        """
        if method == "mean":
            return segment_embeddings.mean(axis=0)  # (1024,)
        elif method == "mean_std":
            mean = segment_embeddings.mean(axis=0)
            std = segment_embeddings.std(axis=0)
            return np.concatenate([mean, std])  # (2048,)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
