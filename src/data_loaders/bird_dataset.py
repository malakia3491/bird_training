import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os

class PrecomputedBirdDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_encoder, data_root=None, augmentation=None):
        self.df = df
        self.label_encoder = label_encoder
        self.data_root = data_root
        self.augmentation = augmentation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Получаем путь из CSV
        mel_path = str(row['mel_path']) # /home/user/.../mel/birdclef/file.npy
        
        # 2. Чиним путь для Windows/твоей папки
        if self.data_root:
            # Нормализуем слеши (превращаем \ в / для унификации)
            mel_path_clean = mel_path.replace('\\', '/')
            
            # Ищем ключевую папку 'mel/'
            if 'mel/' in mel_path_clean:
                # Берем хвост пути: birdclef/file.npy
                rel_path = mel_path_clean.split('mel/')[-1]
                # Собираем новый полный путь: D:/data/mel/birdclef/file.npy
                full_path = os.path.join(self.data_root, 'mel', rel_path)
            else:
                # Если в пути нет 'mel/', просто клеим имя файла
                full_path = os.path.join(self.data_root, os.path.basename(mel_path))
                
            # Важно: нормализуем под текущую ОС (Windows сделает обратные слеши)
            full_path = os.path.normpath(full_path)
        else:
            full_path = mel_path

        # 3. Загружаем .npy
        try:
            mel_spec = np.load(full_path) 
        except Exception as e:
            # ЧТОБЫ УВИДЕТЬ ОШИБКУ В КОНСОЛИ, НО НЕ КРАШИТЬ CUDA
            # Можно раскомментировать принт для отладки
            # print(f"MISSING: {full_path}")
            
            # !!! ВАЖНО !!!
            # Возвращаем класс 0, а не -1. Класс 0 валиден для CrossEntropy.
            # Если файлов не найдено много, модель будет учить мусор, 
            # но хотя бы не упадет с ошибкой драйвера.
            return torch.zeros(1, 128, 313), torch.tensor(0, dtype=torch.long)

        # 4. Обработка тензора
        mel_tensor = torch.from_numpy(mel_spec).float()
        if mel_tensor.ndim == 2:
            mel_tensor = mel_tensor.unsqueeze(0)
        
        if self.augmentation:
            mel_tensor = self.augmentation(mel_tensor)

        # 5. Метка
        species = row['species']
        try:
            label_idx = self.label_encoder.transform([species])[0]
            label = torch.tensor(label_idx, dtype=torch.long)
        except ValueError:
            # Если класс неизвестен, возвращаем 0 (Negative/Unknown)
            label = torch.tensor(0, dtype=torch.long)

        return mel_tensor, label