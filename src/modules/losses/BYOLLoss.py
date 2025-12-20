"""
BYOL и SimSiam Loss функции.

BYOL (Bootstrap Your Own Latent):
    loss = MSE(predictor(online_proj), target_proj.detach())

SimSiam:
    loss = -cosine_similarity(predictor(proj1), proj2.detach())
         + -cosine_similarity(predictor(proj2), proj1.detach())  # симметричный

Главное отличие от contrastive:
- Нет negative samples
- Нет температуры
- Работает через предсказание одного view из другого
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BYOLLoss(nn.Module):
    """
    BYOL Loss: normalized MSE между prediction и target.

    loss = 2 - 2 * cosine_similarity(pred, target)
         = MSE(normalize(pred), normalize(target))
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, D) - выход predictor (online branch)
            target: (B, D) - выход projector (target branch), detached

        Returns:
            loss: скаляр
        """
        # L2 normalize
        pred = F.normalize(pred, dim=-1, p=2)
        target = F.normalize(target, dim=-1, p=2)

        # MSE после нормализации эквивалентен 2 - 2*cosine_sim
        loss = 2 - 2 * (pred * target).sum(dim=-1)
        return loss.mean()


class SimSiamLoss(nn.Module):
    """
    SimSiam Loss: отрицательная cosine similarity.

    Симметричный loss:
        loss = -0.5 * (cos_sim(p1, z2) + cos_sim(p2, z1))

    где p = predictor(z), z = projector(encoder(x))
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        p1: torch.Tensor,
        z1: torch.Tensor,
        p2: torch.Tensor,
        z2: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            p1: predictor(z1) - предсказание из первого view
            z1: projection первого view (для target второго)
            p2: predictor(z2) - предсказание из второго view
            z2: projection второго view (для target первого)

        Returns:
            loss: скаляр

        Note:
            z1, z2 должны быть detached перед вызовом
            или detach внутри этой функции
        """
        # Negative cosine similarity
        loss1 = -F.cosine_similarity(p1, z2.detach(), dim=-1).mean()
        loss2 = -F.cosine_similarity(p2, z1.detach(), dim=-1).mean()

        return 0.5 * (loss1 + loss2)


class SimSiamLossSimple(nn.Module):
    """
    Упрощённая версия SimSiam Loss для совместимости с текущим pipeline.
    Принимает p1, z2 (один direction).
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        p: torch.Tensor,
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            p: (B, D) - выход predictor
            z: (B, D) - target projection (должен быть detached!)

        Returns:
            loss: скаляр
        """
        # z должен быть detached!
        return -F.cosine_similarity(p, z.detach(), dim=-1).mean()
