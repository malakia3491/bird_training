import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    """
    MLP голова для SSL (SimCLR/BYOL).
    """
    def __init__(self, in_features, hidden_dim=2048, out_dim=128):
        super().__init__()
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        # x может быть:
        # 4D: (B, C, F, T) -> CNN
        # 2D: (B, Dim) -> PANNs
        
        # 1. Приводим к плоскому вектору (Batch, Features)
        if x.ndim == 4:
            x = self.pool(x).flatten(1)
        elif x.ndim == 3:
            x = x.mean(dim=1)
        
        # ВАЖНО: Если x.ndim == 2 (как у PANNs), мы пропускаем if/elif
        # и сразу идем в self.net(x).
        
        # 2. Прогоняем через MLP
        return self.net(x)