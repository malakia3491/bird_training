import os
import glob
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from hydra.utils import to_absolute_path

from src.data_loaders.bird_dataset import PrecomputedBirdDataset
from src.utils.serialization import load_pickle, save_pickle

class BirdDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__()
        # Принимаем аргументы напрямую
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.label_encoder = None

    def setup(self, stage=None):
        # Hydra иногда оставляет пути относительными, фиксим это
        abs_root = to_absolute_path(self.root_dir) if not os.path.isabs(self.root_dir) else self.root_dir
        
        # 1. Загрузка LabelEncoder
        le_path = os.path.join(abs_root, 'checkpoints', 'label_encoder.pkl')
        if os.path.exists(le_path):
            print(f"Loading LabelEncoder from {le_path}")
            self.label_encoder = load_pickle(le_path)
        else:
            print("LabelEncoder not found! Creating from manifest...")
            manifest_path = os.path.join(abs_root, 'manifest.csv')
            if not os.path.exists(manifest_path):
                 raise FileNotFoundError(f"Manifest not found at {manifest_path}")
            
            df = pd.read_csv(manifest_path)
            species_list = df['species'].dropna().unique()
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(species_list)
            save_pickle(self.label_encoder, le_path)
            print(f"LabelEncoder saved. Classes: {len(self.label_encoder.classes_)}")

        self.num_classes = len(self.label_encoder.classes_)

        # 2. Сборка Train DataFrame
        train_split_dir = os.path.join(abs_root, 'splits', 'train')
        train_dfs = []
        train_files = glob.glob(os.path.join(train_split_dir, "*.csv"))
        
        if not train_files:
             fallback = os.path.join(abs_root, 'splits', 'train.csv')
             if os.path.exists(fallback):
                 train_files = [fallback]
             else:
                 raise FileNotFoundError(f"No train csv files found in {train_split_dir}")

        for f in train_files:
            train_dfs.append(pd.read_csv(f))
        
        self.train_df = pd.concat(train_dfs, ignore_index=True)

        # 3. Сборка Val DataFrame
        val_split_dir = os.path.join(abs_root, 'splits', 'val')
        val_dfs = []
        val_files = glob.glob(os.path.join(val_split_dir, "*.csv"))
        
        if not val_files:
             fallback = os.path.join(abs_root, 'splits', 'val.csv')
             if os.path.exists(fallback):
                 val_files = [fallback]
        
        if val_files:
            for f in val_files:
                val_dfs.append(pd.read_csv(f))
            self.val_df = pd.concat(val_dfs, ignore_index=True)
        else:
            self.val_df = pd.DataFrame()

        print(f"Dataset Setup Complete. Train: {len(self.train_df)}, Val: {len(self.val_df)}")

        self.train_ds = PrecomputedBirdDataset(
            self.train_df, 
            self.label_encoder, 
            data_root=abs_root
        )
        self.val_ds = PrecomputedBirdDataset(
            self.val_df, 
            self.label_encoder, 
            data_root=abs_root
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False,
            pin_memory=True
        )