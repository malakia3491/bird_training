import torch
import torch.nn as nn

class LinearHead(nn.Module):
    def __init__(self, in_features, num_classes, dropout=0.0):
        super().__init__()
        # Адаптивный пулинг нужен только для CNN (4D вход)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x может быть:
        # 1. (B, C, F, T) - от CNN
        # 2. (B, C) - от Transformer (с global_pool='avg')
        
        if x.ndim == 4:
            # Если это CNN (картинка), делаем пулинг и выпрямляем
            x = self.pool(x).flatten(1)
        elif x.ndim == 3:
            # Если это Transformer без пулинга (B, Tokens, Dim)
            # Усредняем по токенам (Global Average Pooling)
            x = x.mean(dim=1)
        
        # Если x.ndim == 2, значит он уже (B, C), ничего делать не надо
        
        x = self.dropout(x)
        return self.fc(x)