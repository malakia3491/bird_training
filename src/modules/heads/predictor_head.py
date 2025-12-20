"""
Predictor Head для BYOL/SimSiam.

В BYOL/SimSiam архитектура:
    view1 -> encoder -> projector -> predictor -> output1
    view2 -> encoder -> projector ────────────> output2 (target, stop gradient)

Predictor - дополнительный MLP который применяется только к online branch.
"""

import torch
import torch.nn as nn


class PredictorHead(nn.Module):
    """
    Predictor MLP для BYOL/SimSiam.

    Архитектура: Linear -> BN -> ReLU -> Linear
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 512,
        out_dim: int = 128
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
