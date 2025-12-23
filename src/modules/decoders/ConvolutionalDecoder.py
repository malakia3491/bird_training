import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalDecoder(nn.Module):
    def __init__(self, in_features=2048, output_channels=1, output_size=(128, 313)):
        super().__init__()
        # --- ИСПРАВЛЕНИЕ: Конвертация в tuple ---
        if hasattr(output_size, 'cuda'): # Если вдруг пришел Tensor
            output_size = tuple(output_size.tolist())
        elif isinstance(output_size, (list, tuple)): # Если list или ListConfig
            output_size = tuple(output_size)
        else:
            # Если это что-то странное (например ListConfig), пробуем превратить в tuple
            output_size = tuple(output_size)
            
        self.output_size = output_size
        
        # Разворачиваем вектор в карту 4x10 с 256 каналами
        self.init_h, self.init_w = 4, 10
        self.map_channels = 256
        
        # Адаптивный пулинг, на случай если пришел 4D тензор
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.projection = nn.Linear(in_features, self.map_channels * self.init_h * self.init_w)
        
        self.net = nn.Sequential(
            # 4x10 -> 8x20
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 8x20 -> 16x40
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 16x40 -> 32x80
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # 32x80 -> 64x160
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            # 64x160 -> 128x320
            nn.ConvTranspose2d(16, output_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        # x: (B, C, F, T) или (B, Dim)
        
        # 1. Приводим к вектору (Batch, Dim)
        if x.ndim == 4:
            x = self.pool(x).flatten(1)
        elif x.ndim == 3:
            x = x.mean(dim=1)
            
        # 2. Проекция
        x = self.projection(x)
        x = x.view(-1, self.map_channels, self.init_h, self.init_w)
        
        # 3. Апскейлинг
        x = self.net(x)
        
        # 4. Кроп/Паддинг до целевого размера
        if x.shape[-2:] != self.output_size:
            x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
            
        return x