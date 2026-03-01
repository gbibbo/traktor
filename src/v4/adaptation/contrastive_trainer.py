"""
PURPOSE: Stub para entrenador contrastivo sobre embeddings MERT.
         Interfaz definida para futura implementación. train_epoch() levanta
         NotImplementedError — sirve como referencia de la API esperada.
CHANGELOG:
  - 2026-03-01: Creación inicial V4 (stub con interfaz definida).
"""
from typing import Optional

import torch
import torch.nn as nn


class ContrastiveTrainer:
    """
    Entrenador contrastivo para adaptar el ProjectionHead a la colección de música.

    El flujo esperado cuando se implemente:
      1. Pares positivos: tracks del mismo cluster L2 (semánticamente similares).
      2. Pares negativos: tracks de clusters L1 distintos (semánticamente distintos).
      3. Loss: NT-Xent (InfoNCE) o Triplet Loss sobre los embeddings proyectados.
      4. Solo el ProjectionHead es trainable; el backbone MERT permanece frozen.

    Uso futuro:
        trainer = ContrastiveTrainer(model=head, optimizer=optim, device="cuda")
        for epoch in range(n_epochs):
            metrics = trainer.train_epoch(dataloader)
            print(metrics)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
        temperature: float = 0.07,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.temperature = temperature

    def train_epoch(self, dataloader) -> dict:
        """
        Ejecuta una época de entrenamiento contrastivo.

        Args:
            dataloader: DataLoader que produce pares (anchor, positive, negative)
                        de embeddings MERT pre-computados.

        Returns:
            dict con métricas: {'loss': float, 'n_pairs': int}

        Raises:
            NotImplementedError: Este método aún no está implementado.
        """
        raise NotImplementedError(
            "Contrastive training not yet implemented. "
            "Esta función es un stub para futura implementación. "
            "Ver docs/V4_USAGE.md para el roadmap de adaptación."
        )

    def evaluate(self, dataloader) -> dict:
        """
        Evalúa el modelo sobre un conjunto de validación.

        Raises:
            NotImplementedError: Este método aún no está implementado.
        """
        raise NotImplementedError("Evaluation not yet implemented.")
