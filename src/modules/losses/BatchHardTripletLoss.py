import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchHardTripletLoss(nn.Module):
    """
    Считает Triplet Loss с самой сложной (hardest) тройкой для каждого якоря.
    Не требует формирования троек в датасете, работает с (embeddings, labels).
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        # embeddings: (B, D)
        # labels: (B)
        
        # 1. Матрица попарных расстояний (Euclidean)
        # |x-y|^2 = |x|^2 - 2xy + |y|^2
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances = torch.sqrt(distances.clamp(min=1e-16)) # (B, B)

        # 2. Маски для позитивов и негативов
        # labels_equal: матрица (B, B), где True если labels[i] == labels[j]
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_anchor_positive = labels_equal.float()
        mask_anchor_negative = 1 - mask_anchor_positive

        # Убираем диагональ (расстояние до самого себя = 0) из позитивов
        mask_self = torch.eye(labels.size(0), device=labels.device).bool()
        mask_anchor_positive[mask_self] = 0

        # 3. Ищем Hardest Positive (самый далекий друг)
        # Чтобы max не взял нули (от негативов), заменяем негативы на -inf?
        # Нет, просто умножаем distance * mask. Негативы станут 0.
        # Но если дистанция до позитива маленькая, 0 может стать максимумом.
        # Лучше: ставим -1 для негативов.
        # Реализация:
        hardest_positive_dist, _ = (distances * mask_anchor_positive).max(dim=1)

        # 4. Ищем Hardest Negative (самый близкий враг)
        # Чтобы min не взял нули (от позитивов), добавляем к позитивам +inf
        max_dist = distances.max()
        distances_for_neg = distances + max_dist * mask_anchor_positive
        hardest_negative_dist, _ = distances_for_neg.min(dim=1)

        # 5. Лосс: max(0, pos - neg + margin)
        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        
        return triplet_loss.mean()