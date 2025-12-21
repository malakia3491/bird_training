"""
SSL Dataset с кэшированными mel-спектрограммами.

Загружает предварительно рассчитанные mel из .pt файлов.
Значительно ускоряет обучение (в 2-3 раза).

Использование:
    1. Сначала запустите: python scripts/precompute_mels.py --manifest ... --output data/mels_cache
    2. Затем используйте CachedSSLDataset
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Callable, List, Tuple
import random


class CachedSSLDataset(Dataset):
    """
    Dataset для SSL с предварительно рассчитанными mel-спектрограммами.

    Загружает готовые .pt файлы вместо конвертации аудио на лету.
    """

    def __init__(
        self,
        cache_dir: str,
        transform: Optional[Callable] = None,
        index_file: Optional[str] = None,
    ):
        """
        Args:
            cache_dir: Директория с .pt файлами mel-спектрограмм
            transform: SSL трансформ (ContrastiveTransform, MultiCropTransform, etc.)
            index_file: Опционально - файл с путями к .pt файлам
        """
        self.cache_dir = Path(cache_dir)
        self.transform = transform

        # Загружаем список файлов
        if index_file:
            with open(index_file, 'r') as f:
                self.mel_files = [line.strip() for line in f if line.strip()]
        else:
            # Ищем все .pt файлы
            self.mel_files = sorted([str(p) for p in self.cache_dir.glob("*.pt")])

        if not self.mel_files:
            raise ValueError(f"Не найдено .pt файлов в {cache_dir}")

        print(f"[CachedSSLDataset] Загружено {len(self.mel_files)} mel-спектрограмм")

    def __len__(self) -> int:
        return len(self.mel_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Возвращает:
            - mel спектрограмма (или список для contrastive)
            - dummy label (0)
        """
        mel_path = self.mel_files[idx]

        # Загружаем mel
        data = torch.load(mel_path, weights_only=True)
        mel = data['mel']

        # Применяем SSL трансформ
        if self.transform is not None:
            mel = self.transform(mel)

        label = torch.tensor(0, dtype=torch.long)
        return mel, label


class CachedSSLDatasetWithLabels(Dataset):
    """
    Cached dataset с метками классов для SupCon.

    Ожидает структуру: cache_dir/{class_name}_{filename}.pt
    """

    def __init__(
        self,
        cache_dir: str,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[dict] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.transform = transform

        # Загружаем файлы
        self.mel_files = sorted([str(p) for p in self.cache_dir.glob("*.pt")])

        if not self.mel_files:
            raise ValueError(f"Не найдено .pt файлов в {cache_dir}")

        # Извлекаем классы из имён файлов (class_filename.pt)
        if class_to_idx is None:
            classes = set()
            for f in self.mel_files:
                # Имя файла: classname_originalname.pt
                name = Path(f).stem
                class_name = name.rsplit('_', 1)[0] if '_' in name else name
                classes.add(class_name)
            self.class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
        else:
            self.class_to_idx = class_to_idx

        # Создаём маппинг файл -> класс
        self.labels = []
        for f in self.mel_files:
            name = Path(f).stem
            class_name = name.rsplit('_', 1)[0] if '_' in name else name
            self.labels.append(self.class_to_idx.get(class_name, 0))

        print(f"[CachedSSLDatasetWithLabels] {len(self.mel_files)} файлов, {len(self.class_to_idx)} классов")

    def __len__(self) -> int:
        return len(self.mel_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mel_path = self.mel_files[idx]
        label = self.labels[idx]

        data = torch.load(mel_path, weights_only=True)
        mel = data['mel']

        if self.transform is not None:
            mel = self.transform(mel)

        return mel, torch.tensor(label, dtype=torch.long)


class CachedSSLDataModule:
    """
    DataModule для работы с кэшированными mel-спектрограммами.
    """

    def __init__(
        self,
        cache_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        val_split: float = 0.1,
        with_labels: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.val_split = val_split
        self.with_labels = with_labels

        # Загружаем все файлы
        all_files = sorted([str(p) for p in self.cache_dir.glob("*.pt")])

        # Разбиваем на train/val
        random.seed(42)
        random.shuffle(all_files)
        val_size = int(len(all_files) * val_split)
        self.val_files = all_files[:val_size]
        self.train_files = all_files[val_size:]

        print(f"[CachedSSLDataModule] Train: {len(self.train_files)}, Val: {len(self.val_files)}")

    def train_dataloader(self):
        from torch.utils.data import DataLoader

        dataset = CachedSSLDataset(
            cache_dir=str(self.cache_dir),
            transform=self.train_transform,
        )
        # Фильтруем только train файлы
        dataset.mel_files = self.train_files

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        from torch.utils.data import DataLoader

        dataset = CachedSSLDataset(
            cache_dir=str(self.cache_dir),
            transform=self.val_transform,
        )
        dataset.mel_files = self.val_files

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
