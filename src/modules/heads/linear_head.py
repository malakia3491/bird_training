import torch
import torch.nn as nn

class LinearHead(nn.Module):
    def __init__(self, in_features, num_classes, dropout=0.0):
        super().__init__()
        # Мы принимаем (Batch, Channels, Freq, Time)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        # x: (Batch, Channels, Freq, Time)
        x = self.pool(x).flatten(1) # -> (Batch, Channels)
        return self.head(x)