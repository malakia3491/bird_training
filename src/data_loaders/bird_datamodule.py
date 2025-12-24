import os
import glob
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from hydra.utils import to_absolute_path
import numpy as np

from src.data_loaders.bird_dataset import PrecomputedBirdDataset
from src.utils.serialization import load_pickle, save_pickle
from src.data_loaders.samplers import MPerClassSampler

class BirdDataModule(pl.LightningDataModule):
    # --- ИСПРАВЛЕНИЕ: Добавлен sampler и **kwargs ---
    def __init__(self, root_dir, batch_size, num_workers, m_per_class=0, resize_shape=None, train_transform=None, sampler=None, **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize_shape = resize_shape
        self.train_transform = train_transform
        self.m_per_class = m_per_class 
        self.sampler_factory = sampler 
        # Мы пока сохраняем конфиг сэмплера, если он придет, 
        # но основную логику оставляем на m_per_class для совместимости
        self.sampler_cfg = sampler 
        
        self.label_encoder = None

    def setup(self, stage=None):
        abs_root = to_absolute_path(self.root_dir) if not os.path.isabs(self.root_dir) else self.root_dir
        
        # 1. Train DataFrame
        train_split_dir = os.path.join(abs_root, 'splits', 'train')
        train_dfs = []
        train_files = glob.glob(os.path.join(train_split_dir, "*.csv"))
        
        if not train_files:
             fallback = os.path.join(abs_root, 'splits', 'train.csv')
             if os.path.exists(fallback):
                 train_files = [fallback]
             else:
                 manifest_path = os.path.join(abs_root, 'manifest.csv')
                 if os.path.exists(manifest_path):
                     print("⚠️ No train splits found. Using full manifest as train (SSL mode?).")
                     train_files = [manifest_path]
                 else:
                     raise FileNotFoundError(f"No train csv files found in {train_split_dir}")

        for f in train_files:
            train_dfs.append(pd.read_csv(f))
        
        self.train_df = pd.concat(train_dfs, ignore_index=True)

        # 2. Val DataFrame
        val_split_dir = os.path.join(abs_root, 'splits', 'val')
        val_dfs = []
        val_files = glob.glob(os.path.join(val_split_dir, "*.csv"))
        
        if not val_files:
            fallback_val = os.path.join(abs_root, 'splits', 'val.csv')
            if os.path.exists(fallback_val):
                val_files = [fallback_val]

        if val_files:
            for f in val_files:
                val_dfs.append(pd.read_csv(f))
            self.val_df = pd.concat(val_dfs, ignore_index=True)
        else:
            self.val_df = pd.DataFrame(columns=self.train_df.columns)

        # 3. Test DataFrame
        test_split_dir = os.path.join(abs_root, 'splits', 'test')
        test_dfs = []
        test_files = glob.glob(os.path.join(test_split_dir, "*.csv"))
        if not test_files:
            fallback_test = os.path.join(abs_root, 'splits', 'test.csv')
            if os.path.exists(fallback_test):
                test_files = [fallback_test]
        
        if test_files:
            for f in test_files:
                test_dfs.append(pd.read_csv(f))
            self.test_df = pd.concat(test_dfs, ignore_index=True)
        else:
            self.test_df = pd.DataFrame(columns=self.train_df.columns)

        print(f"Dataset Loaded. Train: {len(self.train_df)}, Val: {len(self.val_df)}, Test: {len(self.test_df)}")

        # 4. LabelEncoder (Строго по Train+Val)
        le_path = os.path.join(abs_root, 'checkpoints', 'label_encoder.pkl')
        if os.path.exists(le_path):
            print(f"Loading LabelEncoder from {le_path}")
            self.label_encoder = load_pickle(le_path)
        else:
            print("Creating LabelEncoder from TRAIN/VAL splits...")
            all_species = pd.concat([self.train_df['species'], self.val_df['species']]).dropna().unique()
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(all_species)
            save_pickle(self.label_encoder, le_path)
            print(f"LabelEncoder saved. Classes: {len(self.label_encoder.classes_)}")

        self.num_classes = len(self.label_encoder.classes_)

        # 5. Datasets
        self.train_ds = PrecomputedBirdDataset(
            self.train_df, self.label_encoder, data_root=abs_root,
            resize_shape=self.resize_shape, augmentation=self.train_transform 
        )
        self.val_ds = PrecomputedBirdDataset(
            self.val_df, self.label_encoder, data_root=abs_root, resize_shape=self.resize_shape
        )
        self.test_ds = PrecomputedBirdDataset(
            self.test_df, self.label_encoder, data_root=abs_root, resize_shape=self.resize_shape
        )

    def train_dataloader(self):
        # Логика: если передан конфиг сэмплера (factory), используем его.
        # Если нет, но задан m_per_class > 0, создаем руками (старый способ).
        
        sampler = None
        
        if self.sampler_factory is not None:
            print("Using Configured Sampler...")
            # 1. Получаем метки
            try:
                labels = self.label_encoder.transform(self.train_ds.df['species'].values)
            except ValueError as e:
                print("🔴 Label mismatch! You need to delete checkpoints/label_encoder.pkl")
                raise e
            
            # 2. Досоздаем сэмплер, передавая недостающий аргумент labels
            sampler = self.sampler_factory(labels=labels)

        elif self.m_per_class > 0:
            # Fallback (если конфига сэмплера нет, но параметр числа есть)
            print(f"Using Manual MPerClassSampler (m={self.m_per_class})")
            labels = self.label_encoder.transform(self.train_ds.df['species'].values)
            sampler = MPerClassSampler(labels, self.m_per_class, self.batch_size)

        # Возвращаем DataLoader
        if sampler:
            # Если есть сэмплер, shuffle должен быть False (или отсутствовать), batch_size управляется сэмплером
            return DataLoader(
                self.train_ds, 
                batch_sampler=sampler, 
                num_workers=self.num_workers, 
                pin_memory=True,
                persistent_workers=True # Для скорости (trainer=fast)
            )
        else:
            return DataLoader(
                self.train_ds, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                shuffle=True, 
                pin_memory=True,
                persistent_workers=True
            )

    def val_dataloader(self):
        if len(self.val_ds) == 0: return None
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        if len(self.test_ds) == 0: return None
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True, persistent_workers=True)