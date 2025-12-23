import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceHead(nn.Module):
    """
    Голова для Metric Learning (ArcFace).
    Принимает (features, labels) во время обучения.
    """
    def __init__(self, in_features, num_classes, s=30.0, m=0.50, dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        
        # Веса W (центроиды классов)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Константы ArcFace
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
        # Слои предобработки
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Добавляем Dropout. Если 0.0, он просто ничего не будет делать.
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, labels=None):
        # 1. Получаем вектор признаков
        if x.ndim == 4:
            embeddings = self.pool(x).flatten(1)
        elif x.ndim == 3:
            embeddings = x.mean(dim=1)
        else:
            embeddings = x
        
        # 2. Применяем Dropout ПЕРЕД нормализацией!
        # Это помогает регуляризации, не ломая геометрию на сфере
        embeddings = self.dropout(embeddings)

        # 3. Нормализация (L2 Norm)
        embeddings = F.normalize(embeddings, dim=1)
        weights = F.normalize(self.weight, dim=1)

        # 4. Вычисляем косинус (Logits)
        cosine = F.linear(embeddings, weights)

        # --- Режим Inference / Validation (или если labels не переданы) ---
        if labels is None:
            return cosine * self.s

        # --- Режим Training (добавляем Margin) ---
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output