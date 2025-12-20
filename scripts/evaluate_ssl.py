"""
Скрипт для оценки качества SSL представлений через Linear Probing и kNN.
Использование:
    python scripts/evaluate_ssl.py --ckpt outputs/2025-12-19/22-53-45/checkpoints/last.ckpt
"""

import argparse
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


# Параметры mel спектрограммы (как в SSL датасете)
SAMPLE_RATE = 32000
N_FFT = 1024
WIN_LENGTH = 1024
HOP_LENGTH = 320
N_MELS = 128
F_MIN = 50
F_MAX = 14000
TOP_DB = 80.0
MAX_DURATION = 10.0


class EvalDataset(Dataset):
    """Dataset для оценки - загружает аудио с метками классов из структуры папок."""

    def __init__(self, audio_paths, labels, max_samples):
        self.audio_paths = audio_paths
        self.labels = labels
        self.max_samples = max_samples

        self.mel_transform = T.MelSpectrogram(
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
        self.db_transform = T.AmplitudeToDB(top_db=TOP_DB)
        self._resamplers = {}

    def _get_resampler(self, orig_sr):
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = T.Resample(orig_sr, SAMPLE_RATE)
        return self._resamplers[orig_sr]

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]

        try:
            waveform, sr = torchaudio.load(audio_path)
        except:
            waveform = torch.zeros(1, self.max_samples)
            sr = SAMPLE_RATE

        # Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample
        if sr != SAMPLE_RATE:
            waveform = self._get_resampler(sr)(waveform)

        # Pad/crop
        num_samples = waveform.shape[1]
        if num_samples > self.max_samples:
            waveform = waveform[:, :self.max_samples]
        elif num_samples < self.max_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.max_samples - num_samples))

        # To mel
        mel = self.mel_transform(waveform)
        mel = self.db_transform(mel)

        return mel, label


def load_ssl_backbone(ckpt_path, device):
    """Загружает backbone из SSL чекпойнта."""
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Извлекаем state_dict
    state_dict = checkpoint.get('state_dict', checkpoint)

    # Фильтруем только backbone веса
    backbone_state = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            new_key = k.replace('backbone.', '')
            backbone_state[new_key] = v

    # Создаём backbone
    import timm
    # Определяем модель из конфига или используем default
    model_name = 'tf_efficientnet_b0_ns'

    backbone = timm.create_model(model_name, pretrained=False, num_classes=0, in_chans=1)
    backbone.load_state_dict(backbone_state, strict=False)
    backbone = backbone.to(device)
    backbone.eval()

    print(f"Loaded backbone: {model_name}")
    return backbone


def collect_labeled_data(root_dir, max_per_class=100):
    """Собирает пути к файлам и метки из структуры папок."""
    root = Path(root_dir)
    audio_paths = []
    labels = []

    # Каждая подпапка = класс
    class_dirs = [d for d in root.iterdir() if d.is_dir()]

    print(f"Found {len(class_dirs)} classes")

    for class_dir in class_dirs:
        class_name = class_dir.name
        files = list(class_dir.glob('*.ogg')) + list(class_dir.glob('*.wav'))

        # Ограничиваем количество на класс
        if len(files) > max_per_class:
            files = np.random.choice(files, max_per_class, replace=False).tolist()

        for f in files:
            audio_paths.append(str(f))
            labels.append(class_name)

    print(f"Total samples: {len(audio_paths)}")
    return audio_paths, labels


@torch.no_grad()
def extract_embeddings(backbone, dataloader, device):
    """Извлекает эмбеддинги для всех данных."""
    embeddings = []
    all_labels = []

    for batch_mel, batch_labels in tqdm(dataloader, desc="Extracting embeddings"):
        batch_mel = batch_mel.to(device)
        emb = backbone(batch_mel)

        # Global average pooling если нужно
        if len(emb.shape) > 2:
            emb = emb.mean(dim=[2, 3]) if len(emb.shape) == 4 else emb.mean(dim=1)

        embeddings.append(emb.cpu().numpy())
        all_labels.extend(batch_labels.numpy())

    embeddings = np.vstack(embeddings)
    return embeddings, np.array(all_labels)


def evaluate_linear_probe(X_train, y_train, X_test, y_test):
    """Обучает линейный классификатор и оценивает accuracy."""
    print("\n=== Linear Probe ===")
    clf = LogisticRegression(max_iter=1000, multi_class='multinomial', n_jobs=-1)
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")

    return test_acc


def evaluate_knn(X_train, y_train, X_test, y_test, k=20):
    """Оценивает через kNN."""
    print(f"\n=== kNN (k={k}) ===")
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_train, y_train)

    train_acc = knn.score(X_train, y_train)
    test_acc = knn.score(X_test, y_test)

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")

    return test_acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate SSL model quality")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to SSL checkpoint")
    parser.add_argument("--data_dir", type=str,
                       default="D:/Downloads/pipeline_russian-20251214T110835Z-1-001/pipeline_russian/birdclef/train_short_audio",
                       help="Path to labeled audio data")
    parser.add_argument("--max_per_class", type=int, default=50, help="Max samples per class")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Загружаем backbone
    backbone = load_ssl_backbone(args.ckpt, device)

    # 2. Собираем данные с метками
    audio_paths, labels = collect_labeled_data(args.data_dir, args.max_per_class)

    # 3. Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    num_classes = len(le.classes_)
    print(f"Number of classes: {num_classes}")

    # 4. Split
    paths_train, paths_test, y_train, y_test = train_test_split(
        audio_paths, labels_encoded, test_size=args.test_size,
        stratify=labels_encoded, random_state=42
    )

    print(f"Train: {len(paths_train)}, Test: {len(paths_test)}")

    # 5. Create dataloaders
    max_samples = int(SAMPLE_RATE * MAX_DURATION)

    train_ds = EvalDataset(paths_train, y_train, max_samples)
    test_ds = EvalDataset(paths_test, y_test, max_samples)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 6. Extract embeddings
    print("\nExtracting train embeddings...")
    X_train, y_train = extract_embeddings(backbone, train_loader, device)

    print("Extracting test embeddings...")
    X_test, y_test = extract_embeddings(backbone, test_loader, device)

    print(f"\nEmbedding shape: {X_train.shape[1]}")

    # 7. Evaluate
    linear_acc = evaluate_linear_probe(X_train, y_train, X_test, y_test)
    knn_acc = evaluate_knn(X_train, y_train, X_test, y_test, k=20)

    print("\n" + "="*50)
    print(f"FINAL RESULTS")
    print(f"="*50)
    print(f"Linear Probe Accuracy: {linear_acc:.4f} ({linear_acc*100:.2f}%)")
    print(f"kNN (k=20) Accuracy:   {knn_acc:.4f} ({knn_acc*100:.2f}%)")
    print(f"="*50)


if __name__ == "__main__":
    main()
