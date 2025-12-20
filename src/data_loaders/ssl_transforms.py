"""
SSL аугментации для аудио-спектрограмм.

Режимы:
- contrastive: SimCLR (два разных взгляда на один файл)
- reconstruction: MAE/Denoising (испорченный -> оригинал)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchaudio.transforms as AT
import numpy as np
import os
import random
from typing import List, Tuple, Optional, Literal
from pathlib import Path


# Вынесены отдельно для совместимости с Windows multiprocessing (pickle)

def add_gaussian_noise(x: torch.Tensor, std: float = 0.05) -> torch.Tensor:
    return x + std * torch.randn_like(x)


def time_roll(x: torch.Tensor, max_shift_ratio: float = 0.5) -> torch.Tensor:
    """Циклический сдвиг по времени."""
    time_dim = x.shape[-1]
    shift = random.randint(0, int(time_dim * max_shift_ratio))
    return torch.roll(x, shifts=shift, dims=-1)


def freq_shift(x: torch.Tensor, max_shift: int = 10) -> torch.Tensor:
    """Сдвиг по частоте (имитация pitch shift)."""
    shift = random.randint(-max_shift, max_shift)
    if shift == 0:
        return x
    return torch.roll(x, shifts=shift, dims=-2)


def random_gain(x: torch.Tensor, min_gain: float = 0.5, max_gain: float = 1.5) -> torch.Tensor:
    return x * random.uniform(min_gain, max_gain)


def time_stretch_simple(x: torch.Tensor, rate_range: Tuple[float, float] = (0.8, 1.2)) -> torch.Tensor:
    """Растяжение/сжатие по времени через интерполяцию."""
    rate = random.uniform(*rate_range)
    if abs(rate - 1.0) < 0.01:
        return x

    orig_time = x.shape[-1]
    new_time = int(orig_time * rate)

    x_stretched = F.interpolate(
        x.unsqueeze(0),
        size=(x.shape[-2], new_time),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)

    if new_time > orig_time:
        start = random.randint(0, new_time - orig_time)
        x_stretched = x_stretched[..., start:start + orig_time]
    else:
        pad_total = orig_time - new_time
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        x_stretched = F.pad(x_stretched, (pad_left, pad_right), mode='constant', value=0)

    return x_stretched


def freq_stretch_simple(x: torch.Tensor, rate_range: Tuple[float, float] = (0.9, 1.1)) -> torch.Tensor:
    """Растяжение/сжатие по частоте (имитация pitch shift)."""
    rate = random.uniform(*rate_range)
    if abs(rate - 1.0) < 0.01:
        return x

    orig_freq = x.shape[-2]
    new_freq = int(orig_freq * rate)

    x_stretched = F.interpolate(
        x.unsqueeze(0),
        size=(new_freq, x.shape[-1]),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)

    if new_freq > orig_freq:
        start = random.randint(0, new_freq - orig_freq)
        x_stretched = x_stretched[..., start:start + orig_freq, :]
    else:
        pad_total = orig_freq - new_freq
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        x_stretched = F.pad(x_stretched, (0, 0, pad_top, pad_bottom), mode='constant', value=0)

    return x_stretched


class NoiseBank:
    """Банк фоновых шумов (.npy). Если папки нет - возвращает None."""

    def __init__(self, noise_dir: Optional[str] = None, cache_size: int = 50):
        self.noise_files: List[Path] = []
        self.cache: dict = {}
        self.cache_size = cache_size

        if noise_dir and os.path.exists(noise_dir):
            noise_path = Path(noise_dir)
            self.noise_files = list(noise_path.glob("**/*.npy"))
            print(f"[NoiseBank] Загружено {len(self.noise_files)} файлов шумов")

    def get_random_noise(self, target_shape: Tuple[int, int]) -> Optional[torch.Tensor]:
        if not self.noise_files:
            return None

        noise_path = random.choice(self.noise_files)
        cache_key = str(noise_path)

        if cache_key in self.cache:
            noise = self.cache[cache_key]
        else:
            try:
                noise = torch.from_numpy(np.load(noise_path)).float()
                if len(self.cache) < self.cache_size:
                    self.cache[cache_key] = noise
            except Exception as e:
                print(f"[NoiseBank] Ошибка загрузки {noise_path}: {e}")
                return None

        if noise.ndim == 2:
            noise = noise.unsqueeze(0)

        return F.interpolate(
            noise.unsqueeze(0),
            size=target_shape,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)


class ContrastiveTransform:
    """
    SimCLR-стиль: генерирует два сильно разных взгляда на один файл.

    strength: 'weak', 'medium', 'strong'
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (128, 313),
        strength: Literal['weak', 'medium', 'strong'] = 'strong',
        noise_dir: Optional[str] = None
    ):
        self.input_size = input_size
        self.strength = strength
        self.noise_bank = NoiseBank(noise_dir)

        params = self._get_params(strength)
        self.crop_scale = params['crop_scale']
        self.freq_mask_param = params['freq_mask_param']
        self.time_mask_param = params['time_mask_param']
        self.num_masks = params['num_masks']
        self.noise_prob = params['noise_prob']
        self.noise_alpha_range = params['noise_alpha_range']
        self.gaussian_std = params['gaussian_std']
        self.blur_prob = params['blur_prob']
        self.time_roll_prob = params['time_roll_prob']
        self.freq_shift_prob = params['freq_shift_prob']
        self.stretch_prob = params['stretch_prob']
        self.gain_prob = params['gain_prob']

        self.random_crop = T.RandomResizedCrop(
            size=input_size,
            scale=self.crop_scale,
            ratio=(0.8, 1.2),
            antialias=True
        )
        self.gaussian_blur = T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        self.freq_masking = AT.FrequencyMasking(freq_mask_param=self.freq_mask_param)
        self.time_masking = AT.TimeMasking(time_mask_param=self.time_mask_param)

    def _get_params(self, strength: str) -> dict:
        if strength == 'weak':
            return {
                'crop_scale': (0.7, 1.0),
                'freq_mask_param': 15,
                'time_mask_param': 30,
                'num_masks': 1,
                'noise_prob': 0.3,
                'noise_alpha_range': (0.05, 0.15),
                'gaussian_std': 0.02,
                'blur_prob': 0.2,
                'time_roll_prob': 0.2,
                'freq_shift_prob': 0.1,
                'stretch_prob': 0.1,
                'gain_prob': 0.3,
            }
        elif strength == 'medium':
            return {
                'crop_scale': (0.5, 1.0),
                'freq_mask_param': 25,
                'time_mask_param': 50,
                'num_masks': 2,
                'noise_prob': 0.5,
                'noise_alpha_range': (0.1, 0.3),
                'gaussian_std': 0.05,
                'blur_prob': 0.3,
                'time_roll_prob': 0.3,
                'freq_shift_prob': 0.2,
                'stretch_prob': 0.2,
                'gain_prob': 0.5,
            }
        else:  # strong
            return {
                'crop_scale': (0.3, 1.0),
                'freq_mask_param': 40,
                'time_mask_param': 80,
                'num_masks': 3,
                'noise_prob': 0.8,
                'noise_alpha_range': (0.15, 0.4),
                'gaussian_std': 0.08,
                'blur_prob': 0.5,
                'time_roll_prob': 0.5,
                'freq_shift_prob': 0.3,
                'stretch_prob': 0.3,
                'gain_prob': 0.7,
            }

    def _apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.gain_prob:
            x = random_gain(x, 0.5, 1.5)

        if random.random() < self.stretch_prob:
            x = time_stretch_simple(x, (0.8, 1.2))
        if random.random() < self.stretch_prob:
            x = freq_stretch_simple(x, (0.9, 1.1))

        x = self.random_crop(x)

        if random.random() < self.time_roll_prob:
            x = time_roll(x, 0.5)

        if random.random() < self.freq_shift_prob:
            x = freq_shift(x, max_shift=15)

        if random.random() < self.blur_prob:
            x = self.gaussian_blur(x)

        if random.random() < self.noise_prob:
            x = self._add_background_noise(x)

        for _ in range(self.num_masks):
            x = self.freq_masking(x)
            x = self.time_masking(x)

        x = add_gaussian_noise(x, self.gaussian_std * random.uniform(0.5, 1.5))

        return x

    def _add_background_noise(self, x: torch.Tensor) -> torch.Tensor:
        target_shape = (x.shape[-2], x.shape[-1])
        noise = self.noise_bank.get_random_noise(target_shape)

        if noise is None:
            noise = self._generate_colored_noise(target_shape)

        noise = noise * (x.std() / (noise.std() + 1e-8))
        alpha = random.uniform(*self.noise_alpha_range)
        return x * (1 - alpha) + noise * alpha

    def _generate_colored_noise(self, shape: Tuple[int, int]) -> torch.Tensor:
        """Розовый шум (1/f) - fallback если нет банка шумов."""
        f, t = shape
        noise = torch.randn(1, f, t)
        freq_weights = torch.linspace(1, 0.1, f).view(1, f, 1)
        return noise * freq_weights

    def __call__(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(0)

        x1 = self._apply_augmentation(x.clone())
        x2 = self._apply_augmentation(x.clone())

        return [x1, x2]


class ReconstructionTransform:
    """
    MAE/Denoising: генерирует (испорченный, оригинал, маска).

    corruption_type: 'mask', 'noise', 'both'
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (128, 313),
        mask_ratio: float = 0.75,
        corruption_type: Literal['mask', 'noise', 'both'] = 'both',
        noise_dir: Optional[str] = None
    ):
        self.input_size = input_size
        self.mask_ratio = mask_ratio
        self.corruption_type = corruption_type
        self.noise_bank = NoiseBank(noise_dir)
        self.patch_size = (16, 16)
        self.resize = T.Resize(input_size, antialias=True)

    def _mask_patches(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c, f, t = x.shape
        pf, pt = self.patch_size

        num_patches_f = f // pf
        num_patches_t = t // pt
        total_patches = num_patches_f * num_patches_t
        num_masked = int(total_patches * self.mask_ratio)

        mask_indices = random.sample(range(total_patches), num_masked)

        mask = torch.ones(c, f, t)
        masked_x = x.clone()

        for idx in mask_indices:
            pi = idx // num_patches_t
            pj = idx % num_patches_t

            f_start = pi * pf
            f_end = min(f_start + pf, f)
            t_start = pj * pt
            t_end = min(t_start + pt, t)

            masked_x[:, f_start:f_end, t_start:t_end] = 0
            mask[:, f_start:f_end, t_start:t_end] = 0

        return masked_x, mask

    def _add_corruption_noise(self, x: torch.Tensor) -> torch.Tensor:
        corrupted = x.clone()

        noise_level = random.uniform(0.1, 0.3)
        corrupted = corrupted + noise_level * torch.randn_like(corrupted)

        target_shape = (x.shape[-2], x.shape[-1])
        bg_noise = self.noise_bank.get_random_noise(target_shape)
        if bg_noise is not None:
            alpha = random.uniform(0.2, 0.5)
            corrupted = corrupted * (1 - alpha) + bg_noise * alpha

        # Random dropout - имитация потери сигнала
        dropout_mask = torch.bernoulli(torch.full_like(corrupted, 0.9))
        corrupted = corrupted * dropout_mask

        return corrupted

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if x.ndim == 2:
            x = x.unsqueeze(0)

        original = self.resize(x)
        mask = None

        if self.corruption_type == 'mask':
            corrupted, mask = self._mask_patches(original)
        elif self.corruption_type == 'noise':
            corrupted = self._add_corruption_noise(original)
        else:
            corrupted, mask = self._mask_patches(original)
            corrupted = self._add_corruption_noise(corrupted)

        return corrupted, original, mask


class MixupTransform:
    """Смешивает два сэмпла с весом из Beta-распределения."""

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, float]:
        lam = np.random.beta(self.alpha, self.alpha)
        return lam * x1 + (1 - lam) * x2, lam


class CutmixTransform:
    """Вырезает прямоугольник из x2 и вставляет в x1."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, float]:
        lam = np.random.beta(self.alpha, self.alpha)

        _, f, t = x1.shape

        cut_ratio = np.sqrt(1 - lam)
        cut_f = int(f * cut_ratio)
        cut_t = int(t * cut_ratio)

        cf = random.randint(0, f - cut_f) if cut_f < f else 0
        ct = random.randint(0, t - cut_t) if cut_t < t else 0

        mixed = x1.clone()
        mixed[:, cf:cf + cut_f, ct:ct + cut_t] = x2[:, cf:cf + cut_f, ct:ct + cut_t]

        lam = 1 - (cut_f * cut_t) / (f * t)
        return mixed, lam


class MultiCropTransform:
    """
    Multi-Crop аугментация для SimCLR/DINO.

    Создаёт несколько кропов разного размера:
    - n_global_crops больших кропов (например, 224x224)
    - n_local_crops маленьких кропов (например, 96x96)

    Это даёт больше positive pairs без увеличения batch size.
    """

    def __init__(
        self,
        global_size: Tuple[int, int] = (128, 313),
        local_size: Tuple[int, int] = (64, 128),
        n_global_crops: int = 2,
        n_local_crops: int = 4,
        global_scale: Tuple[float, float] = (0.5, 1.0),
        local_scale: Tuple[float, float] = (0.2, 0.5),
        strength: Literal['weak', 'medium', 'strong'] = 'strong',
        noise_dir: Optional[str] = None
    ):
        self.n_global_crops = n_global_crops
        self.n_local_crops = n_local_crops

        # Базовые аугментации
        params = self._get_params(strength)
        self.noise_bank = NoiseBank(noise_dir)
        self.gaussian_std = params['gaussian_std']

        # Global crops
        self.global_crop = T.RandomResizedCrop(
            size=global_size,
            scale=global_scale,
            ratio=(0.8, 1.2),
            antialias=True
        )

        # Local crops
        self.local_crop = T.RandomResizedCrop(
            size=local_size,
            scale=local_scale,
            ratio=(0.8, 1.2),
            antialias=True
        )

        # Общие аугментации
        self.freq_masking = AT.FrequencyMasking(freq_mask_param=params['freq_mask_param'])
        self.time_masking = AT.TimeMasking(time_mask_param=params['time_mask_param'])
        self.gaussian_blur = T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))

    def _get_params(self, strength: str) -> dict:
        if strength == 'weak':
            return {'freq_mask_param': 15, 'time_mask_param': 30, 'gaussian_std': 0.02}
        elif strength == 'medium':
            return {'freq_mask_param': 25, 'time_mask_param': 50, 'gaussian_std': 0.05}
        else:
            return {'freq_mask_param': 40, 'time_mask_param': 80, 'gaussian_std': 0.08}

    def _apply_augmentation(self, x: torch.Tensor, is_local: bool = False) -> torch.Tensor:
        # Crop
        if is_local:
            x = self.local_crop(x)
        else:
            x = self.global_crop(x)

        # Random augmentations
        if random.random() < 0.5:
            x = self.gaussian_blur(x)

        if random.random() < 0.8:
            x = self.freq_masking(x)
            x = self.time_masking(x)

        # Noise
        x = add_gaussian_noise(x, self.gaussian_std * random.uniform(0.5, 1.5))

        return x

    def __call__(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(0)

        crops = []

        # Global crops
        for _ in range(self.n_global_crops):
            crops.append(self._apply_augmentation(x.clone(), is_local=False))

        # Local crops
        for _ in range(self.n_local_crops):
            crops.append(self._apply_augmentation(x.clone(), is_local=True))

        return crops


def create_ssl_transform(
    mode: Literal['contrastive', 'reconstruction', 'multicrop'] = 'contrastive',
    input_size: Tuple[int, int] = (128, 313),
    strength: Literal['weak', 'medium', 'strong'] = 'strong',
    noise_dir: Optional[str] = None,
    **kwargs
):
    """Фабрика для создания SSL трансформов."""
    if mode == 'contrastive':
        return ContrastiveTransform(
            input_size=input_size,
            strength=strength,
            noise_dir=noise_dir
        )
    elif mode == 'reconstruction':
        return ReconstructionTransform(
            input_size=input_size,
            noise_dir=noise_dir,
            **kwargs
        )
    elif mode == 'multicrop':
        return MultiCropTransform(
            global_size=input_size,
            strength=strength,
            noise_dir=noise_dir,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'contrastive', 'reconstruction', or 'multicrop'")
