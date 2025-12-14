import torch
import torch.nn as nn
import timm

class AudioBackbone(nn.Module):
    def __init__(self, name, pretrained, in_chans, num_classes, global_pool):
        super().__init__()
        
        self.model = timm.create_model(
            name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,       # <--- ЖЕСТКО ЗАДАЕМ 0
            global_pool=global_pool 
        )
        
        # Вычисляем размерность эмбеддинга
        with torch.no_grad():
            # (Batch, Chans, Freq, Time)
            # Прогоняем через модель, чтобы узнать выходной размер
            dummy = torch.randn(1, in_chans, 128, 128)
            out = self.model(dummy)
            self.embed_dim = out.shape[1] 

    def forward(self, x):
        return self.model(x)