"""
DataModule для Supervised Contrastive Learning.
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Callable
import random

from src.data_loaders.supcon_dataset import SupConAudioDataset


class SupConDataModule(pl.LightningDataModule):
    """
    DataModule для SupCon обучения.
    Загружает данные из директории с структурой class/audio.ogg
    """

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        max_duration: float = 10.0,
        val_split: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()

        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.max_duration = max_duration
        self.val_split = val_split
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None
        self.class_names = None
        self.num_classes = None

    def setup(self, stage: Optional[str] = None):
        """Настройка датасетов с разбиением по файлам внутри классов."""
        root = Path(self.root_dir)
        extensions = ('.ogg', '.wav', '.mp3', '.flac')

        # Собираем все данные
        class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
        self.class_names = [d.name for d in class_dirs]
        self.num_classes = len(self.class_names)

        train_paths, train_labels = [], []
        val_paths, val_labels = [], []

        random.seed(self.seed)

        for idx, class_dir in enumerate(class_dirs):
            # Собираем файлы этого класса
            class_files = []
            for ext in extensions:
                class_files.extend(list(class_dir.glob(f'*{ext}')))

            # Перемешиваем и разбиваем
            random.shuffle(class_files)
            val_size = max(1, int(len(class_files) * self.val_split))

            val_files = class_files[:val_size]
            train_files = class_files[val_size:]

            for f in train_files:
                train_paths.append(str(f))
                train_labels.append(idx)

            for f in val_files:
                val_paths.append(str(f))
                val_labels.append(idx)

        print(f"[SupConDataModule] Classes: {self.num_classes}")
        print(f"[SupConDataModule] Train: {len(train_paths)}, Val: {len(val_paths)}")

        self.train_dataset = SupConAudioDataset(
            audio_paths=train_paths,
            labels=train_labels,
            class_names=self.class_names,
            transform=self.train_transform,
            max_duration=self.max_duration,
            random_crop=True
        )

        self.val_dataset = SupConAudioDataset(
            audio_paths=val_paths,
            labels=val_labels,
            class_names=self.class_names,
            transform=self.val_transform,
            max_duration=self.max_duration,
            random_crop=False
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0
        )
