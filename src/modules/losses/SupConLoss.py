"""
Supervised Contrastive Loss (SupCon)

Отличие от SimCLR/NTXent:
- SimCLR: позитивная пара = две аугментации ОДНОГО файла
- SupCon: позитивные пары = ВСЕ записи одного КЛАССА

Это решает проблему SimCLR для классификации птиц:
модель учится группировать по классу, а не по файлу.

Paper: "Supervised Contrastive Learning" (Khosla et al., 2020)
https://arxiv.org/abs/2004.11362
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    Args:
        temperature: температура для softmax (default: 0.07)
        base_temperature: базовая температура для нормализации (default: 0.07)
        contrast_mode: 'all' или 'one' (default: 'all')
            - 'all': все views как anchor
            - 'one': только первый view как anchor
    """

    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        contrast_mode: str = 'all'
    ):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.contrast_mode = contrast_mode

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch_size, n_views, feature_dim) или (batch_size, feature_dim)
                      Эмбеддинги должны быть L2-нормализованы!
            labels: (batch_size,) метки классов

        Returns:
            loss: скаляр
        """
        device = features.device

        # Если features 2D (batch, dim) — добавляем view dimension
        if features.ndim == 2:
            features = features.unsqueeze(1)  # (B, 1, D)

        batch_size = features.shape[0]
        n_views = features.shape[1]

        # Проверка labels
        if labels.shape[0] != batch_size:
            raise ValueError(f"Num of labels ({labels.shape[0]}) != batch size ({batch_size})")

        labels = labels.contiguous().view(-1, 1)  # (B, 1)

        # Маска позитивных пар: mask[i,j] = 1 если labels[i] == labels[j]
        mask = torch.eq(labels, labels.T).float().to(device)  # (B, B)

        # Разворачиваем features: (B, n_views, D) -> (B * n_views, D)
        contrast_features = features.view(batch_size * n_views, -1)  # (B*V, D)

        # Выбираем anchor features
        if self.contrast_mode == 'one':
            anchor_features = features[:, 0]  # (B, D)
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_features = contrast_features  # (B*V, D)
            anchor_count = n_views
        else:
            raise ValueError(f"Unknown contrast_mode: {self.contrast_mode}")

        # Расширяем маску для всех views
        # mask: (B, B) -> (B*V, B*V) если mode='all'
        mask = mask.repeat(anchor_count, n_views)  # (B*anchor_count, B*n_views)

        # Cosine similarity: (B*anchor_count, B*n_views)
        anchor_dot_contrast = torch.matmul(anchor_features, contrast_features.T) / self.temperature

        # Numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Маска для исключения самих себя (диагональ)
        # Создаём identity маску размером (B*anchor_count, B*n_views)
        if self.contrast_mode == 'all':
            # Диагональ в развёрнутом виде
            logits_mask = torch.ones_like(mask)
            # Убираем диагональ (сам с собой)
            self_mask = torch.eye(batch_size * n_views, device=device)
            logits_mask = logits_mask - self_mask
        else:
            # mode='one': anchor (B,) vs contrast (B*V,)
            logits_mask = torch.ones(batch_size, batch_size * n_views, device=device)
            # Убираем первый view каждого anchor
            for i in range(batch_size):
                logits_mask[i, i * n_views] = 0

        mask = mask * logits_mask

        # Log-sum-exp для знаменателя
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Среднее по позитивным парам
        # Избегаем деления на 0, если нет позитивных пар
        mask_sum = mask.sum(1)
        mask_sum = torch.clamp(mask_sum, min=1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # Loss с температурной нормализацией
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


class SupConLossSimple(nn.Module):
    """
    Упрощённая версия SupCon для случая с двумя views (как в SimCLR).

    Принимает z1, z2 и labels отдельно (совместимо с текущим SSL pipeline).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z1: (B, D) - первый view, L2-нормализован
            z2: (B, D) - второй view, L2-нормализован
            labels: (B,) - метки классов

        Returns:
            loss: скаляр
        """
        device = z1.device
        batch_size = z1.shape[0]

        # Конкатенируем views: (2B, D)
        features = torch.cat([z1, z2], dim=0)

        # Дублируем labels для обоих views: (2B,)
        labels = torch.cat([labels, labels], dim=0)
        labels = labels.contiguous().view(-1, 1)

        # Маска позитивных пар (одинаковый класс)
        mask = torch.eq(labels, labels.T).float().to(device)  # (2B, 2B)

        # Cosine similarity
        similarity = torch.matmul(features, features.T) / self.temperature  # (2B, 2B)

        # Убираем диагональ (сам с собой)
        logits_mask = 1.0 - torch.eye(2 * batch_size, device=device)
        mask = mask * logits_mask

        # Numerical stability
        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Среднее по позитивным парам
        mask_sum = mask.sum(1).clamp(min=1e-12)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        loss = -mean_log_prob_pos.mean()

        return loss
