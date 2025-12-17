"""
Генерирует фейковые данные для тестирования пайплайна.

Использование:
    python scripts/generate_test_data.py
    python train_ssl.py data.root_dir="data/test_dataset" trainer.max_epochs=3
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_test_dataset(
    output_dir: str = "data/test_dataset",
    num_train: int = 100,
    num_val: int = 20,
    n_mels: int = 128,
    time_frames: int = 313,
    num_classes: int = 5,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "splits").mkdir(exist_ok=True)

    species = [f"test_bird_{i}" for i in range(num_classes)]
    all_records = []

    # Train
    train_records = []
    for i in range(num_train):
        spec = np.random.randn(n_mels, time_frames).astype(np.float32)
        class_idx = i % num_classes
        spec[class_idx * 20:(class_idx + 1) * 20, :] += 2.0

        filename = f"train_{i:04d}.npy"
        np.save(output_path / filename, spec)
        train_records.append({"mel_path": filename, "species": species[class_idx]})
        all_records.append({"mel_path": filename, "species": species[class_idx]})

    # Val
    val_records = []
    for i in range(num_val):
        spec = np.random.randn(n_mels, time_frames).astype(np.float32)
        class_idx = i % num_classes
        spec[class_idx * 20:(class_idx + 1) * 20, :] += 2.0

        filename = f"val_{i:04d}.npy"
        np.save(output_path / filename, spec)
        val_records.append({"mel_path": filename, "species": species[class_idx]})
        all_records.append({"mel_path": filename, "species": species[class_idx]})

    # manifest.csv (все файлы)
    pd.DataFrame(all_records).to_csv(output_path / "manifest.csv", index=False)

    # splits/train.csv и splits/val.csv
    pd.DataFrame(train_records).to_csv(output_path / "splits" / "train.csv", index=False)
    pd.DataFrame(val_records).to_csv(output_path / "splits" / "val.csv", index=False)

    print(f"Создано {num_train} train + {num_val} val в {output_path}")
    print(f"\nЗапуск:")
    print(f'  python train_ssl.py data.root_dir="{output_dir}" trainer.max_epochs=3')


if __name__ == "__main__":
    generate_test_dataset()
