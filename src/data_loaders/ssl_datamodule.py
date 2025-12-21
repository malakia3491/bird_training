"""
SSL DataModule для PyTorch Lightning.

Поддерживает два режима:
1. On-the-fly: конвертация аудио в mel на лету (медленнее, но экономит место)
2. Cached: загрузка предварительно рассчитанных mel (быстрее в 2-3 раза)

Для cached режима сначала запустите:
    python scripts/precompute_mels.py --manifest ... --output data/mels_cache
"""

import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Callable, Tuple
import random

from src.data_loaders.ssl_dataset import SSLAudioDataset, SSLAudioDatasetFromManifest
from src.data_loaders.cached_ssl_dataset import CachedSSLDataset


class SSLDataModule(pl.LightningDataModule):
    """
    DataModule для SSL обучения.

    Поддерживает режимы загрузки данных:
    1. Из директории с аудио файлами (root_dir)
    2. Из TXT манифеста (manifest_txt)
    3. Из CSV манифеста (manifest_csv)
    4. Из кэша mel-спектрограмм (cache_dir) - БЫСТРЕЕ!
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        manifest_txt: Optional[str] = None,
        manifest_csv: Optional[str] = None,
        cache_dir: Optional[str] = None,  # NEW: директория с .pt файлами
        use_cache: bool = False,  # NEW: флаг использования кэша
        audio_column: str = 'audio_path',
        batch_size: int = 32,
        num_workers: int = 4,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        max_duration: float = 10.0,
        val_split: float = 0.1,
        seed: int = 42,
    ):
        """
        Args:
            root_dir: Путь к директории с аудио файлами
            manifest_txt: Путь к TXT манифесту (один путь на строку)
            manifest_csv: Путь к CSV манифесту
            cache_dir: Путь к директории с кэшированными .pt файлами
            use_cache: Использовать кэшированные mel-спектрограммы
            audio_column: Название колонки с путями в CSV
            batch_size: Размер батча
            num_workers: Количество воркеров для DataLoader
            train_transform: SSL трансформ для обучения
            val_transform: SSL трансформ для валидации
            max_duration: Максимальная длительность сегмента в секундах
            val_split: Доля данных для валидации
            seed: Seed для воспроизводимости разбиения
        """
        super().__init__()

        self.root_dir = root_dir
        self.manifest_txt = manifest_txt
        self.manifest_csv = manifest_csv
        self.cache_dir = cache_dir
        self.use_cache = use_cache or (cache_dir is not None)
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

        if self.use_cache:
            # Cached режим - загружаем готовые mel
            self._setup_cached()
        else:
            # On-the-fly режим - конвертируем аудио в mel на лету
            self._setup_onthefly()

    def _setup_cached(self):
        """Настройка датасетов из кэша."""
        cache_path = Path(self.cache_dir)

        if not cache_path.exists():
            raise ValueError(
                f"Директория кэша не найдена: {self.cache_dir}\n"
                f"Сначала запустите: python scripts/precompute_mels.py --manifest ... --output {self.cache_dir}"
            )

        # Загружаем все .pt файлы
        all_files = sorted([str(p) for p in cache_path.glob("*.pt")])

        if not all_files:
            raise ValueError(f"Не найдено .pt файлов в {self.cache_dir}")

        # Разбиение на train/val
        random.seed(self.seed)
        random.shuffle(all_files)

        val_size = int(len(all_files) * self.val_split)
        train_files = all_files[val_size:]
        val_files = all_files[:val_size]

        print(f"[SSLDataModule] CACHED MODE")
        print(f"[SSLDataModule] Train: {len(train_files)}, Val: {len(val_files)}")

        # Создаём датасеты
        self.train_dataset = CachedSSLDataset(
            cache_dir=self.cache_dir,
            transform=self.train_transform,
        )
        self.train_dataset.mel_files = train_files

        self.val_dataset = CachedSSLDataset(
            cache_dir=self.cache_dir,
            transform=self.val_transform,
        )
        self.val_dataset.mel_files = val_files

    def _setup_onthefly(self):
        """Настройка датасетов с on-the-fly конвертацией."""
        audio_paths = self._load_audio_paths()

        if not audio_paths:
            raise ValueError("Не найдено аудио файлов!")

        # Разбиение на train/val
        random.seed(self.seed)
        random.shuffle(audio_paths)

        val_size = int(len(audio_paths) * self.val_split)
        train_paths = audio_paths[val_size:]
        val_paths = audio_paths[:val_size]

        print(f"[SSLDataModule] ON-THE-FLY MODE")
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
            random_crop=False
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
