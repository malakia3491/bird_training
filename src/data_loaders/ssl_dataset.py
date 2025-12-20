"""
SSL Dataset для загрузки аудио файлов и конвертации в mel спектрограммы на лету.
Экономит дисковое пространство - не требует предварительной конвертации.
"""

import torch
import torchaudio
import torchaudio.transforms as T
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, Callable
import random


class SSLAudioDataset(Dataset):
    """
    Dataset для SSL обучения на аудио файлах.
    Загружает OGG/WAV файлы и конвертирует в mel спектрограммы на лету.
    """

    # Параметры mel спектрограммы (должны совпадать с конфигом)
    SAMPLE_RATE = 32000
    N_FFT = 1024
    WIN_LENGTH = 1024
    HOP_LENGTH = 320
    N_MELS = 128
    F_MIN = 50
    F_MAX = 14000
    TOP_DB = 80.0

    def __init__(
        self,
        audio_paths: list,
        transform: Optional[Callable] = None,
        max_duration: float = 10.0,  # Максимальная длительность в секундах
        random_crop: bool = True,    # Случайный кроп или с начала
    ):
        """
        Args:
            audio_paths: Список путей к аудио файлам
            transform: SSL трансформ (например, ContrastiveTransform)
            max_duration: Максимальная длительность сегмента в секундах
            random_crop: Если True - случайный кроп, иначе с начала файла
        """
        self.audio_paths = audio_paths
        self.transform = transform
        self.max_duration = max_duration
        self.random_crop = random_crop

        # Максимальное количество сэмплов
        self.max_samples = int(self.SAMPLE_RATE * max_duration)

        # Инициализируем трансформации для mel спектрограммы
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.SAMPLE_RATE,
            n_fft=self.N_FFT,
            win_length=self.WIN_LENGTH,
            hop_length=self.HOP_LENGTH,
            n_mels=self.N_MELS,
            f_min=self.F_MIN,
            f_max=self.F_MAX,
            power=2.0,
            normalized=True
        )
        self.db_transform = T.AmplitudeToDB(top_db=self.TOP_DB)

        # Кэш для ресемплеров (разные исходные sample rate)
        self._resamplers = {}

    def _get_resampler(self, orig_sr: int) -> T.Resample:
        """Получает или создаёт ресемплер для данного sample rate."""
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = T.Resample(orig_sr, self.SAMPLE_RATE)
        return self._resamplers[orig_sr]

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Загружает и предобрабатывает аудио файл."""
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            # Возвращаем тишину при ошибке
            print(f"[SSLAudioDataset] Ошибка загрузки {audio_path}: {e}")
            return torch.zeros(1, self.max_samples)

        # Конвертируем в моно
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Ресемплинг если нужно
        if sr != self.SAMPLE_RATE:
            resampler = self._get_resampler(sr)
            waveform = resampler(waveform)

        # Кроп или паддинг до нужной длины
        num_samples = waveform.shape[1]

        if num_samples > self.max_samples:
            # Кроп
            if self.random_crop:
                start = random.randint(0, num_samples - self.max_samples)
            else:
                start = 0
            waveform = waveform[:, start:start + self.max_samples]
        elif num_samples < self.max_samples:
            # Паддинг нулями
            pad_size = self.max_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))

        return waveform

    def _to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Конвертирует waveform в log-mel спектрограмму."""
        mel = self.mel_transform(waveform)
        log_mel = self.db_transform(mel)
        return log_mel  # Shape: (1, n_mels, time)

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Возвращает:
            - mel спектрограмму (или список из двух для contrastive)
            - dummy label (0) - для SSL метки не нужны
        """
        audio_path = self.audio_paths[idx]

        # Загружаем аудио
        waveform = self._load_audio(audio_path)

        # Конвертируем в mel
        mel = self._to_mel(waveform)

        # Применяем SSL трансформ (создаёт два view для contrastive)
        if self.transform is not None:
            mel = self.transform(mel)

        # Dummy label для SSL
        label = torch.tensor(0, dtype=torch.long)

        return mel, label


class SSLAudioDatasetFromManifest(SSLAudioDataset):
    """
    SSL Dataset из манифеста (CSV или TXT файла).
    """

    @classmethod
    def from_manifest_csv(
        cls,
        manifest_path: str,
        audio_column: str = 'audio_path',
        transform: Optional[Callable] = None,
        **kwargs
    ) -> 'SSLAudioDatasetFromManifest':
        """Создаёт dataset из CSV манифеста."""
        df = pd.read_csv(manifest_path)
        audio_paths = df[audio_column].tolist()
        return cls(audio_paths=audio_paths, transform=transform, **kwargs)

    @classmethod
    def from_manifest_txt(
        cls,
        manifest_path: str,
        transform: Optional[Callable] = None,
        **kwargs
    ) -> 'SSLAudioDatasetFromManifest':
        """Создаёт dataset из TXT манифеста (один путь на строку)."""
        with open(manifest_path, 'r', encoding='utf-8') as f:
            audio_paths = [line.strip() for line in f if line.strip()]
        return cls(audio_paths=audio_paths, transform=transform, **kwargs)

    @classmethod
    def from_directory(
        cls,
        root_dir: str,
        extensions: Tuple[str, ...] = ('.ogg', '.wav', '.mp3', '.flac'),
        transform: Optional[Callable] = None,
        **kwargs
    ) -> 'SSLAudioDatasetFromManifest':
        """Создаёт dataset из директории, рекурсивно находя все аудио файлы."""
        root = Path(root_dir)
        audio_paths = []
        for ext in extensions:
            audio_paths.extend([str(p) for p in root.rglob(f'*{ext}')])

        if not audio_paths:
            raise ValueError(f"Не найдено аудио файлов в {root_dir}")

        print(f"[SSLAudioDataset] Найдено {len(audio_paths)} аудио файлов")
        return cls(audio_paths=audio_paths, transform=transform, **kwargs)
