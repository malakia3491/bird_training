import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os
import torchvision.transforms as T # <--- ИМПОРТ

class PrecomputedBirdDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_encoder, data_root=None, augmentation=None, resize_shape=None): # <--- НОВЫЙ АРГУМЕНТ
        self.df = df
        self.label_encoder = label_encoder
        self.data_root = data_root
        self.augmentation = augmentation
        
        # Если задан размер (H, W), создаем трансформ ресайза
        self.resize = T.Resize(resize_shape, antialias=True) if resize_shape else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel_path = str(row['mel_path'])
        
        if self.data_root:
            mel_path_clean = mel_path.replace('\\', '/')
            if 'mel/' in mel_path_clean:
                rel_path = mel_path_clean.split('mel/')[-1]
                full_path = os.path.join(self.data_root, 'mel', rel_path)
            else:
                full_path = os.path.join(self.data_root, os.path.basename(mel_path))
            full_path = os.path.normpath(full_path)
        else:
            full_path = mel_path

        try:
            mel_spec = np.load(full_path) 
        except Exception:
            # Возвращаем заглушку нужного размера, если файл битый
            # Лучше вернуть что-то похожее на правду, чтобы батч не упал
            return torch.zeros(1, 128, 313), torch.tensor(0, dtype=torch.long)

        mel_tensor = torch.from_numpy(mel_spec).float()
        
        # Гарантируем (C, F, T)
        if mel_tensor.ndim == 2:
            mel_tensor = mel_tensor.unsqueeze(0)
        
        # Применяем resize_shape из конфига DataModule (если есть)
        if self.resize:
            mel_tensor = self.resize(mel_tensor)

        # Применяем Transform (SSLEvalTransform или ContrastiveTransform)
        # Именно здесь SSLEvalTransform исправит размер, если self.resize был None
        if self.augmentation:
            mel_tensor = self.augmentation(mel_tensor)

        # Получение метки (без изменений)
        species = row['species']
        try:
            label_idx = self.label_encoder.transform([species])[0]
            label = torch.tensor(label_idx, dtype=torch.long)
        except ValueError:
            label = torch.tensor(0, dtype=torch.long)

        return mel_tensor, label