import torch
import torch.nn as nn
import torchvision.transforms as T
import torchaudio.transforms as AT

# 1. Выносим функцию шума отдельно (чтобы Windows мог её "подхватить")
def add_gaussian_noise(x):
    return x + 0.01 * torch.randn_like(x)

class ContrastiveTransform:
    """
    Генерирует две разные версии (view) одного и того же входного тензора.
    """
    def __init__(self, input_size=(128, 313)):
        # input_size: (Freq, Time)
        
        self.transform = T.Compose([
            # Random Crop
            T.RandomResizedCrop(size=input_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
            
            # Masking
            AT.FrequencyMasking(freq_mask_param=input_size[0] // 4),
            AT.TimeMasking(time_mask_param=input_size[1] // 4),
            
            # Шум (теперь через обычную функцию)
            T.Lambda(add_gaussian_noise)
        ])

    def __call__(self, x):
        # Генерируем два "взгляда"
        x1 = self.transform(x)
        x2 = self.transform(x)
        return [x1, x2]