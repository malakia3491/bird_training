"""
SSL DataModule для PyTorch Lightning.
Работает напрямую с аудио файлами без предварительной конвертации.
"""

import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Callable, Tuple
import random

from src.data_loaders.ssl_dataset import SSLAudioDataset, SSLAudioDatasetFromManifest


class SSLDataModule(pl.LightningDataModule):
    """
    DataModule для SSL обучения.

    Поддерживает три режима загрузки данных:
    1. Из директории с аудио файлами (root_dir)
    2. Из TXT манифеста (manifest_txt)
    3. Из CSV манифеста (manifest_csv)
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        manifest_txt: Optional[str] = None,
        manifest_csv: Optional[str] = None,
        audio_column: str = 'audio_path',
        batch_size: int = 32,
        num_workers: int = 4,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        max_duration: float = 10.0,
        val_split: float = 0.1,  # Доля данных для валидации
        seed: int = 42,
    ):
        """
        Args:
            root_dir: Путь к директории с аудио файлами
            manifest_txt: Путь к TXT манифесту (один путь на строку)
            manifest_csv: Путь к CSV манифесту
            audio_column: Название колонки с путями в CSV
            batch_size: Размер батча
            num_workers: Количество воркеров для DataLoader
            train_transform: SSL трансформ для обучения
            val_transform: SSL трансформ для валидации (обычно None или тот же)
            max_duration: Максимальная длительность сегмента в секундах
            val_split: Доля данных для валидации
            seed: Seed для воспроизводимости разбиения
        """
        super().__init__()

        self.root_dir = root_dir
        self.manifest_txt = manifest_txt
        self.manifest_csv = manifest_csv
        self.audio_column = audio_column
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.max_duration = max_duration
        self.val_split = val_split
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None

    def _load_audio_paths(self) -> list:
        """Загружает список путей к аудио файлам."""
        audio_paths = []

        if self.manifest_txt:
            # Из TXT манифеста
            with open(self.manifest_txt, 'r', encoding='utf-8') as f:
                audio_paths = [line.strip() for line in f if line.strip()]
            print(f"[SSLDataModule] Загружено {len(audio_paths)} путей из {self.manifest_txt}")

        elif self.manifest_csv:
            # Из CSV манифеста
            import pandas as pd
            df = pd.read_csv(self.manifest_csv)
            audio_paths = df[self.audio_column].tolist()
            print(f"[SSLDataModule] Загружено {len(audio_paths)} путей из {self.manifest_csv}")

        elif self.root_dir:
            # Из директории
            root = Path(self.root_dir)
            extensions = ('.ogg', '.wav', '.mp3', '.flac')
            for ext in extensions:
                audio_paths.extend([str(p) for p in root.rglob(f'*{ext}')])
            print(f"[SSLDataModule] Найдено {len(audio_paths)} аудио файлов в {self.root_dir}")

        else:
            raise ValueError("Необходимо указать root_dir, manifest_txt или manifest_csv")

        return audio_paths

    def setup(self, stage: Optional[str] = None):
        """Настройка датасетов."""
        audio_paths = self._load_audio_paths()

        if not audio_paths:
            raise ValueError("Не найдено аудио файлов!")

        # Разбиение на train/val
        random.seed(self.seed)
        random.shuffle(audio_paths)

        val_size = int(len(audio_paths) * self.val_split)
        train_paths = audio_paths[val_size:]
        val_paths = audio_paths[:val_size]

        print(f"[SSLDataModule] Train: {len(train_paths)}, Val: {len(val_paths)}")

        # Создаём датасеты
        self.train_dataset = SSLAudioDataset(
            audio_paths=train_paths,
            transform=self.train_transform,
            max_duration=self.max_duration,
            random_crop=True
        )

        self.val_dataset = SSLAudioDataset(
            audio_paths=val_paths,
            transform=self.val_transform,
            max_duration=self.max_duration,
            random_crop=False  # Для валидации берём с начала
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,  # Важно для contrastive learning
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
