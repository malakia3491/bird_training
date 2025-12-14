import os
import argparse
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    SPLITS_DIR, CHECKPOINTS_DIR, RESULTS_DIR,
    CNN_LR, CNN_EPOCHS, CNN_BATCH_SIZE,
    MLP_LR, MLP_EPOCHS, MLP_BATCH_SIZE, MLP_HIDDEN_DIM
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class MelDataset(Dataset):
    def __init__(self, df, label_encoder=None, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(df['species'])
        else:
            self.label_encoder = label_encoder
            self.labels = self.label_encoder.transform(df['species'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel = np.load(row['mel_path'])
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)

        if self.augment:
            if np.random.rand() > 0.5:
                t = np.random.randint(0, mel.shape[1] - 10)
                mel[:, t:t+10] = 0
            if np.random.rand() > 0.5:
                f = np.random.randint(0, mel.shape[0] - 10)
                mel[f:f+10, :] = 0

        mel = mel[np.newaxis, :, :]
        return torch.FloatTensor(mel), self.labels[idx]


class SimpleCNN(nn.Module):
    def __init__(self, num_classes, input_shape=(128, 313)):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_embeddings(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


def train_model(model_type, num_classes, train_loader, val_loader, input_dim=None):
    print(f"\n{'='*60}")
    print(f"TRAINING: {model_type.upper()}")
    print(f"{'='*60}")

    if model_type == 'cnn':
        model = SimpleCNN(num_classes).to(DEVICE)
        lr = CNN_LR
        epochs = CNN_EPOCHS
    else:
        model = SimpleMLP(input_dim, MLP_HIDDEN_DIM, num_classes).to(DEVICE)
        lr = MLP_LR
        epochs = MLP_EPOCHS

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    print(f"Device: {DEVICE}")
    print(f"Epochs: {epochs}, LR: {lr}")

    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Train: {train_loss:.4f} / {train_acc:.4f} | "
              f"Val: {val_loss:.4f} / {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(CHECKPOINTS_DIR, f'{model_type}_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'num_classes': num_classes
            }, checkpoint_path)

    print(f"\nBest Val Acc: {best_val_acc:.4f}")

    history_path = os.path.join(CHECKPOINTS_DIR, f'{model_type}_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)

    return model, history


def final_evaluation(model, test_loader, label_encoder, model_name):
    print(f"\n{'='*60}")
    print(f"TEST: {model_name}")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, DEVICE)

    print(f"Test Accuracy: {test_acc:.4f}")

    report = classification_report(
        labels, preds,
        target_names=label_encoder.classes_,
        output_dict=True
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results_df = pd.DataFrame(report).transpose()
    results_df.to_csv(os.path.join(RESULTS_DIR, f'{model_name}_report.csv'))

    cm = confusion_matrix(labels, preds)
    np.save(os.path.join(RESULTS_DIR, f'{model_name}_confusion_matrix.npy'), cm)

    print(f"Results saved: {RESULTS_DIR}")
    return test_acc, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'mlp', 'all'])
    args = parser.parse_args()

    print("=" * 60)
    print("TRAIN CLASSIFIER")
    print("=" * 60)

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\nLoading data...")
    train_df = pd.read_csv(os.path.join(SPLITS_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(SPLITS_DIR, 'val.csv'))
    test_df = pd.read_csv(os.path.join(SPLITS_DIR, 'test.csv'))

    print(f"Train: {len(train_df)}")
    print(f"Val: {len(val_df)}")
    print(f"Test: {len(test_df)}")

    train_dataset = MelDataset(train_df, augment=True)
    val_dataset = MelDataset(val_df, label_encoder=train_dataset.label_encoder)
    test_dataset = MelDataset(test_df, label_encoder=train_dataset.label_encoder)

    num_classes = len(train_dataset.label_encoder.classes_)
    print(f"Classes: {num_classes}")

    le_path = os.path.join(CHECKPOINTS_DIR, 'label_encoder.pkl')
    with open(le_path, 'wb') as f:
        pickle.dump(train_dataset.label_encoder, f)

    train_loader = DataLoader(train_dataset, batch_size=CNN_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CNN_BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=CNN_BATCH_SIZE, shuffle=False, num_workers=0)

    sample_mel = np.load(train_df.iloc[0]['mel_path'])
    input_dim = sample_mel.shape[0] * sample_mel.shape[1]

    results = {}

    if args.model in ['cnn', 'all']:
        model, _ = train_model('cnn', num_classes, train_loader, val_loader)

        checkpoint = torch.load(os.path.join(CHECKPOINTS_DIR, 'cnn_best.pt'))
        model = SimpleCNN(num_classes).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_acc, _ = final_evaluation(model, test_loader, train_dataset.label_encoder, 'cnn')
        results['cnn'] = test_acc

    if args.model in ['mlp', 'all']:
        model, _ = train_model('mlp', num_classes, train_loader, val_loader, input_dim)

        checkpoint = torch.load(os.path.join(CHECKPOINTS_DIR, 'mlp_best.pt'))
        model = SimpleMLP(input_dim, MLP_HIDDEN_DIM, num_classes).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_acc, _ = final_evaluation(model, test_loader, train_dataset.label_encoder, 'mlp')
        results['mlp'] = test_acc

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for name, acc in results.items():
        print(f"{name.upper()}: {acc:.4f}")


if __name__ == '__main__':
    main()
