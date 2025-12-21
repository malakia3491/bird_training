"""
Предварительный расчёт mel-спектрограмм для ускорения обучения.

Запуск:
    python scripts/precompute_mels.py --input data/train --output data/mels_cache

С манифестом:
    python scripts/precompute_mels.py --manifest birdclef_manifest.txt --output data/mels_cache
"""

import argparse
import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import os


# Параметры mel (должны совпадать с ssl_dataset.py)
SAMPLE_RATE = 32000
N_FFT = 1024
WIN_LENGTH = 1024
HOP_LENGTH = 320
N_MELS = 128
F_MIN = 50
F_MAX = 14000
TOP_DB = 80.0
MAX_DURATION = 10.0


def process_audio(audio_path: str, output_dir: Path, max_samples: int) -> dict:
    """Обрабатывает один аудио файл."""
    try:
        # Загружаем аудио
        waveform, sr = torchaudio.load(audio_path)

        # Моно
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Ресемплинг
        if sr != SAMPLE_RATE:
            resampler = T.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        # Обрезаем до max_duration (без random crop - сохраняем всё)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        elif waveform.shape[1] < max_samples:
            pad_size = max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))

        # Mel спектрограмма
        mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX,
            power=2.0,
            normalized=True
        )
        db_transform = T.AmplitudeToDB(top_db=TOP_DB)

        mel = mel_transform(waveform)
        log_mel = db_transform(mel)

        # Сохраняем
        # Создаём уникальное имя из пути
        path = Path(audio_path)
        # Используем parent folder + filename как ключ
        rel_name = f"{path.parent.name}_{path.stem}"
        output_path = output_dir / f"{rel_name}.pt"

        torch.save({
            'mel': log_mel,
            'original_path': audio_path
        }, output_path)

        return {'success': True, 'path': audio_path}

    except Exception as e:
        return {'success': False, 'path': audio_path, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Предварительный расчёт mel-спектрограмм')
    parser.add_argument('--input', type=str, help='Директория с аудио файлами')
    parser.add_argument('--manifest', type=str, help='Путь к манифесту (txt, один путь на строку)')
    parser.add_argument('--output', type=str, required=True, help='Директория для сохранения mel')
    parser.add_argument('--workers', type=int, default=4, help='Количество worker процессов')
    parser.add_argument('--max-duration', type=float, default=MAX_DURATION, help='Макс. длительность (сек)')
    args = parser.parse_args()

    # Получаем список файлов
    if args.manifest:
        with open(args.manifest, 'r', encoding='utf-8') as f:
            audio_paths = [line.strip() for line in f if line.strip()]
    elif args.input:
        input_dir = Path(args.input)
        audio_paths = []
        for ext in ['.ogg', '.wav', '.mp3', '.flac']:
            audio_paths.extend([str(p) for p in input_dir.rglob(f'*{ext}')])
    else:
        raise ValueError("Укажите --input или --manifest")

    print(f"Найдено {len(audio_paths)} аудио файлов")

    # Создаём выходную директорию
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Параметры
    max_samples = int(SAMPLE_RATE * args.max_duration)

    # Обрабатываем
    process_fn = partial(process_audio, output_dir=output_dir, max_samples=max_samples)

    success_count = 0
    error_count = 0

    if args.workers > 1:
        # Многопоточная обработка
        with mp.Pool(args.workers) as pool:
            results = list(tqdm(
                pool.imap(process_fn, audio_paths),
                total=len(audio_paths),
                desc="Генерация mel"
            ))
    else:
        # Однопоточная обработка
        results = []
        for path in tqdm(audio_paths, desc="Генерация mel"):
            results.append(process_fn(path))

    # Статистика
    for r in results:
        if r['success']:
            success_count += 1
        else:
            error_count += 1
            print(f"Ошибка: {r['path']} - {r.get('error', 'unknown')}")

    print(f"\nГотово!")
    print(f"Успешно: {success_count}")
    print(f"Ошибок: {error_count}")
    print(f"Сохранено в: {output_dir}")

    # Сохраняем индекс
    index_path = output_dir / "index.txt"
    mel_files = list(output_dir.glob("*.pt"))
    with open(index_path, 'w') as f:
        for mel_file in mel_files:
            f.write(f"{mel_file}\n")
    print(f"Индекс сохранён: {index_path}")


if __name__ == "__main__":
    main()
