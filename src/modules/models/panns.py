import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from src.modules.models.official_panns import Cnn14

class PANNsBackbone(nn.Module):
    def __init__(self, pretrained=True, in_chans=1, num_classes=0, **kwargs):
        super().__init__()
        
        self.model = Cnn14(classes_num=527, mel_bins=128)
        self.embed_dim = 2048

        # --- ПАТЧ BN0 ---
        if self.model.bn0.num_features != 128:
            print(f"Patching PANNs BN0 layer: {self.model.bn0.num_features} -> 128 channels")
            self.model.bn0 = nn.BatchNorm2d(128)
            self.model.bn0.bias.data.fill_(0.)
            self.model.bn0.weight.data.fill_(1.)

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

        keys_to_remove = []
        for key in state_dict.keys():
            if key.startswith('bn0.') or key.startswith('logmel_extractor.') or key.startswith('spectrogram_extractor.'):
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            del state_dict[key]
        
        self.model.load_state_dict(state_dict, strict=False)
        print(f"PANNs weights loaded. Ignored {len(keys_to_remove)} layers.")

    def forward(self, x):
        # x приходит в формате: (Batch, 1, Freq=128, Time=313)
        
        # 1. Нормализация по частотам (BN0)
        # BN0(128) ожидает, что 128 будет в 1-м измерении (Channels).
        # Нам нужно превратить (B, 1, F, T) -> (B, F, T, 1)
        # Permute индексы: 0(B), 2(F), 3(T), 1(1)
        x = x.permute(0, 2, 3, 1) 
        x = self.model.bn0(x)
        
        # 2. Подготовка для сверток (Cnn14)
        # Cnn14 ожидает вход (Batch, 1, Time, Freq)
        # Сейчас у нас (B, F, T, 1) после BN0.
        # Нам нужно (B, 1, T, F).
        # Permute индексы из текущего состояния: 0(B), 3(1), 2(T), 1(F)
        x = x.permute(0, 3, 2, 1)
        
        # Дальше стандартный PANNs пайплайн
        x = self.model.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.model.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.model.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.model.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = self.model.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = self.model.conv_block6(x, pool_size=(1, 1), pool_type='avg')

        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.model.fc1(x))
        
        return x