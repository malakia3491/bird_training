"""
Скрипт для создания банка шумов из датасета ESC-50.

ESC-50 содержит 50 классов звуков, включая:
- rain, thunderstorm, wind, sea_waves (природа)
- helicopter, chainsaw, engine, train (техника)
- и другие

Использование:
    python scripts/prepare_noise_bank.py --output D:/data/noise_bank

Или если ESC-50 уже скачан:
    python scripts/prepare_noise_bank.py --esc50_path D:/datasets/ESC-50 --output D:/data/noise_bank
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path
import urllib.request

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm


ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"

# Категории шумов, полезные для аугментации птиц
NOISE_CATEGORIES = [
    "rain",
    "thunderstorm",
    "wind",
    "sea_waves",
    "crackling_fire",
    "water_drops",
    "helicopter",
    "chainsaw",
    "engine",
    "train",
    "airplane",
    "church_bells",
    "clock_tick",
    "insects",  # может пересекаться с птицами, но добавляет реализма
]


def download_esc50(download_dir: Path) -> Path:
    """Скачивает ESC-50 если ещё не скачан."""
    zip_path = download_dir / "ESC-50-master.zip"
    extract_path = download_dir / "ESC-50-master"

    if extract_path.exists():
        print(f"ESC-50 уже скачан: {extract_path}")
        return extract_path

    download_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        print(f"Скачиваю ESC-50 (~600MB)...")
        urllib.request.urlretrieve(ESC50_URL, zip_path, reporthook=_download_progress)
        print()

    print("Распаковываю...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(download_dir)

    zip_path.unlink()  # удаляем архив
    print(f"ESC-50 готов: {extract_path}")
    return extract_path


def _download_progress(count, block_size, total_size):
    percent = count * block_size * 100 // total_size
    print(f"\r  {percent}%", end="", flush=True)


def load_esc50_metadata(esc50_path: Path) -> dict:
    """Загружает метаданные ESC-50."""
    import csv

    meta_path = esc50_path / "meta" / "esc50.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Не найден файл метаданных: {meta_path}")

    categories = {}
    with open(meta_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(row['filename'])

    return categories


def wav_to_melspec(
    wav_path: Path,
    sample_rate: int = 32000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 320,
) -> np.ndarray:
    """Конвертирует wav в mel-спектрограмму."""
    waveform, sr = torchaudio.load(wav_path)

    # Ресемплим если нужно
    if sr != sample_rate:
        resampler = T.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    # Моно
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Mel-спектрограмма
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=0,
        f_max=sample_rate // 2,
    )

    mel = mel_transform(waveform)

    # В децибелы
    mel_db = T.AmplitudeToDB(stype='power', top_db=80)(mel)

    return mel_db.squeeze(0).numpy()  # (n_mels, time)


def prepare_noise_bank(
    esc50_path: Path,
    output_dir: Path,
    categories: list = None,
    sample_rate: int = 32000,
    n_mels: int = 128,
):
    """Создаёт банк шумов из ESC-50."""
    if categories is None:
        categories = NOISE_CATEGORIES

    output_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем метаданные
    all_categories = load_esc50_metadata(esc50_path)
    audio_dir = esc50_path / "audio"

    available = [c for c in categories if c in all_categories]
    missing = [c for c in categories if c not in all_categories]

    if missing:
        print(f"Категории не найдены в ESC-50: {missing}")

    print(f"Обрабатываю категории: {available}")

    total_files = 0
    for category in available:
        cat_dir = output_dir / category
        cat_dir.mkdir(exist_ok=True)

        files = all_categories[category]
        print(f"\n{category}: {len(files)} файлов")

        for filename in tqdm(files, desc=f"  {category}"):
            wav_path = audio_dir / filename
            if not wav_path.exists():
                continue

            try:
                mel = wav_to_melspec(wav_path, sample_rate, n_mels)

                npy_name = filename.replace('.wav', '.npy')
                np.save(cat_dir / npy_name, mel)
                total_files += 1
            except Exception as e:
                print(f"  Ошибка {filename}: {e}")

    print(f"\nГотово! Создано {total_files} файлов в {output_dir}")
    return total_files


def generate_synthetic_noise(output_dir: Path, num_files: int = 50, duration_sec: float = 5.0):
    """
    Генерирует синтетические шумы (fallback если нет ESC-50).
    Не так хорошо как реальные, но лучше чем ничего.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 32000
    n_mels = 128
    n_samples = int(duration_sec * sample_rate)

    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=320,
        n_mels=n_mels,
    )
    to_db = T.AmplitudeToDB(stype='power', top_db=80)

    print(f"Генерирую {num_files} синтетических шумов...")

    for i in tqdm(range(num_files)):
        noise_type = i % 4

        if noise_type == 0:
            # Белый шум
            audio = torch.randn(1, n_samples) * 0.3
            name = f"white_noise_{i:03d}.npy"
        elif noise_type == 1:
            # Розовый шум (1/f)
            white = torch.randn(1, n_samples)
            # Простая аппроксимация розового шума через фильтрацию
            b = torch.tensor([0.049922035, -0.095993537, 0.050612699, -0.004408786])
            a = torch.tensor([1, -2.494956002, 2.017265875, -0.522189400])
            # Используем простой RC фильтр вместо полного IIR
            pink = torch.zeros_like(white)
            pink[0, 0] = white[0, 0]
            for j in range(1, n_samples):
                pink[0, j] = 0.99 * pink[0, j-1] + 0.01 * white[0, j]
            audio = pink * 0.5
            name = f"pink_noise_{i:03d}.npy"
        elif noise_type == 2:
            # Броуновский шум (случайное блуждание)
            white = torch.randn(1, n_samples) * 0.01
            brown = torch.cumsum(white, dim=1)
            brown = brown - brown.mean()
            brown = brown / (brown.abs().max() + 1e-8) * 0.5
            audio = brown
            name = f"brown_noise_{i:03d}.npy"
        else:
            # Импульсный шум (капли дождя)
            audio = torch.zeros(1, n_samples)
            num_impulses = torch.randint(50, 200, (1,)).item()
            positions = torch.randint(0, n_samples, (num_impulses,))
            amplitudes = torch.rand(num_impulses) * 0.5
            for pos, amp in zip(positions, amplitudes):
                decay = torch.exp(-torch.arange(1000).float() / 100) * amp
                end = min(pos + 1000, n_samples)
                audio[0, pos:end] += decay[:end-pos]
            name = f"impulse_noise_{i:03d}.npy"

        mel = mel_transform(audio)
        mel_db = to_db(mel)
        np.save(output_dir / name, mel_db.squeeze(0).numpy())

    print(f"Готово! Синтетические шумы в {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Создание банка шумов для SSL аугментаций")
    parser.add_argument("--output", type=str, required=True, help="Путь для сохранения .npy файлов")
    parser.add_argument("--esc50_path", type=str, default=None, help="Путь к уже скачанному ESC-50")
    parser.add_argument("--download_dir", type=str, default="./downloads", help="Куда скачать ESC-50")
    parser.add_argument("--synthetic", action="store_true", help="Сгенерировать синтетические шумы вместо ESC-50")
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--n_mels", type=int, default=128)

    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.synthetic:
        generate_synthetic_noise(output_dir, num_files=100)
    else:
        if args.esc50_path:
            esc50_path = Path(args.esc50_path)
        else:
            esc50_path = download_esc50(Path(args.download_dir))

        prepare_noise_bank(
            esc50_path=esc50_path,
            output_dir=output_dir,
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
        )

    print(f"\nТеперь можно использовать:")
    print(f"  uv run train_ssl.py augmentation.noise_dir=\"{output_dir}\"")


if __name__ == "__main__":
    main()
