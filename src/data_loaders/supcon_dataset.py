"""
Dataset для Supervised Contrastive Learning.
Загружает аудио с метками классов из структуры папок.
"""

import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, Callable, List, Dict
import random


class SupConAudioDataset(Dataset):
    """
    Dataset для SupCon - возвращает (mel, label) где label - реальный класс.

    Структура данных:
        root_dir/
            class_1/
                audio1.ogg
                audio2.ogg
            class_2/
                audio3.ogg
            ...
    """

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
        audio_paths: List[str],
        labels: List[int],
        class_names: List[str],
        transform: Optional[Callable] = None,
        max_duration: float = 10.0,
        random_crop: bool = True,
    ):
        """
        Args:
            audio_paths: Список путей к аудио файлам
            labels: Список меток (int) для каждого файла
            class_names: Список имён классов (для reference)
            transform: SSL трансформ
            max_duration: Максимальная длительность в секундах
            random_crop: Случайный кроп или с начала
        """
        self.audio_paths = audio_paths
        self.labels = labels
        self.class_names = class_names
        self.transform = transform
        self.max_duration = max_duration
        self.random_crop = random_crop

        self.max_samples = int(self.SAMPLE_RATE * max_duration)
        self.num_classes = len(class_names)

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
        self._resamplers = {}

    def _get_resampler(self, orig_sr: int) -> T.Resample:
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = T.Resample(orig_sr, self.SAMPLE_RATE)
        return self._resamplers[orig_sr]

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"[SupConDataset] Ошибка загрузки {audio_path}: {e}")
            return torch.zeros(1, self.max_samples)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != self.SAMPLE_RATE:
            waveform = self._get_resampler(sr)(waveform)

        num_samples = waveform.shape[1]
        if num_samples > self.max_samples:
            if self.random_crop:
                start = random.randint(0, num_samples - self.max_samples)
            else:
                start = 0
            waveform = waveform[:, start:start + self.max_samples]
        elif num_samples < self.max_samples:
            pad_size = self.max_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))

        return waveform

    def _to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.mel_transform(waveform)
        log_mel = self.db_transform(mel)
        return log_mel

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]

        waveform = self._load_audio(audio_path)
        mel = self._to_mel(waveform)

        if self.transform is not None:
            mel = self.transform(mel)

        label_tensor = torch.tensor(label, dtype=torch.long)

        return mel, label_tensor

    @classmethod
    def from_directory(
        cls,
        root_dir: str,
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = ('.ogg', '.wav', '.mp3', '.flac'),
        **kwargs
    ) -> 'SupConAudioDataset':
        """
        Создаёт dataset из директории со структурой class/audio.ogg
        """
        root = Path(root_dir)
        audio_paths = []
        labels = []
        class_to_idx: Dict[str, int] = {}

        # Собираем классы
        class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])

        for idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            class_to_idx[class_name] = idx

            for ext in extensions:
                for audio_file in class_dir.glob(f'*{ext}'):
                    audio_paths.append(str(audio_file))
                    labels.append(idx)

        class_names = list(class_to_idx.keys())

        print(f"[SupConDataset] Найдено {len(audio_paths)} файлов в {len(class_names)} классах")

        return cls(
            audio_paths=audio_paths,
            labels=labels,
            class_names=class_names,
            transform=transform,
            **kwargs
        )
