import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from src.modules.models.official_panns import Cnn14

class PANNsBackbone(nn.Module):
    def __init__(self, pretrained=True, in_chans=1, num_classes=0, **kwargs):
        super().__init__()
        
        # Инициализируем ОРИГИНАЛЬНУЮ модель (64 mels)
        self.model = Cnn14(classes_num=527, mel_bins=64) # <--- ВАЖНО: 64
        self.embed_dim = 2048

        if pretrained:
            self._load_official_weights()

    def _load_official_weights(self):
        url = 'https://zenodo.org/record/3987831/files/Cnn14_mAP=0.431.pth?download=1'
        print(f"Loading PANNs weights from {url}...")
        
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = load_state_dict_from_url(url, map_location=device, progress=True)
        except Exception as e:
            print(f"Error downloading weights: {e}")
            return
        
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

        # Удаляем только экстракторы (они нам не нужны, так как вход уже мел)
        keys_to_remove = []
        for key in state_dict.keys():
            if key.startswith('logmel_extractor.') or key.startswith('spectrogram_extractor.'):
                keys_to_remove.append(key)
        
        # BN0 НЕ УДАЛЯЕМ! Он совпадает (64 канала).
                
        for key in keys_to_remove:
            del state_dict[key]
        
        # Загружаем (strict=False, т.к. удалили экстракторы)
        self.model.load_state_dict(state_dict, strict=False)
        print(f"PANNs weights loaded successfully.")

    def forward(self, x):
        # x: (Batch, 1, 128, T)
        if x.ndim == 3:
            x = x.unsqueeze(1) # (B, F, T) -> (B, 1, F, T)
        # 1. Адаптация разрешения (128 -> 64)
        # Это критически важно, чтобы использовать оригинальные веса!
        if x.shape[2] != 64:
            x = F.interpolate(x, size=(64, x.shape[-1]), mode='bilinear', align_corners=False)
        
        # Теперь x: (Batch, 1, 64, T)
        
        # 2. Подготовка для PANNs (Batch, 1, Time, Freq)
        # PANNs (Cnn14) ожидает Freq в последнем измерении для сверток
        # (Смотрим код ConvBlock: он обычный Conv2d, но PANNs подает ему транспонированный вход)
        
        # В оригинале PANNs:
        # x = x.transpose(1, 3) -> (Batch, 1, Time, Freq)
        # x = self.bn0(x) -> BN по каналу 1? Нет.
        
        # Давай сделаем точно как в оригинале, пропустив STFT часть.
        # Вход в свертки (после STFT и LogMel): (Batch, 1, Time, Mel=64)
        
        x = x.transpose(2, 3) # (Batch, 1, T, 64)
        
        # BN0 в PANNs: self.bn0 = nn.BatchNorm2d(64)
        # BatchNorm2d ожидает (N, C, H, W).
        # Чтобы нормировать по частотам (64), нам нужно засунуть их в канал C.
        
        x = x.transpose(1, 3) # (Batch, 64, T, 1)
        x = self.model.bn0(x)
        x = x.transpose(1, 3) # (Batch, 1, T, 64) -> Возвращаем обратно
        
        # Дальше свертки
        x = self.model.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.model.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.model.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.model.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = self.model.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = self.model.conv_block6(x, pool_size=(1, 1), pool_type='avg')

        # Global Pooling
        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        
        # FC
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.model.fc1(x))
        
        return x