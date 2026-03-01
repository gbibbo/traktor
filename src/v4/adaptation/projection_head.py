"""
PURPOSE: Projection head MLP para fine-tuning contrastivo de embeddings MERT.
         Stub para futura adaptación: backbone MERT (frozen) + MLP trainable.
         Arquitectura: 1024 → Linear+ReLU → 512 → Linear+ReLU → 256 → L2-normalize.
CHANGELOG:
  - 2026-03-01: Creación inicial V4 (stub).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    MLP projection head para embeddings MERT de dimensión 1024.

    Se usa con el backbone MERT frozen: los embeddings pre-computados se pasan
    a este head, que aprende a proyectarlos a un espacio más discriminativo.

    Architecture: 1024 → 512 → 256 (con ReLU y L2-normalize en la salida).
    """

    def __init__(
        self,
        in_dim: int = 1024,
        hidden_dim: int = 512,
        out_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_dim) — embeddings MERT pre-computados

        Returns:
            (batch, out_dim) — embeddings L2-normalizados
        """
        projected = self.net(x)
        return F.normalize(projected, p=2, dim=-1)
