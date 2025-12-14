import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config import (
    DATA_ROOT, MANIFEST_PATH, SPLITS_DIR,
    TOP_N_KNOWN, MIN_SEGMENTS_KNOWN,
    UNSEEN_COUNT, MIN_SEGMENTS_UNSEEN, MAX_SEGMENTS_UNSEEN,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
)

MIN_FILES_PER_CLASS = 5


def fix_mel_path(old_path):
    filename = os.path.basename(old_path)
    parent = os.path.basename(os.path.dirname(old_path))
    return os.path.join(DATA_ROOT, 'mel', parent, filename)


def load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(f"Манифест не найден: {MANIFEST_PATH}")
    df = pd.read_csv(MANIFEST_PATH)
    df['mel_path'] = df['mel_path'].apply(fix_mel_path)

    before = len(df)
    df = df[df['mel_path'].apply(os.path.exists)]
    after = len(df)
    if before != after:
        print(f"Отфильтровано несуществующих: {before - after}")

    print(f"Загружено: {len(df)} сегментов, {df['species'].nunique()} видов")
    return df


def select_known_classes(df):
    species_counts = df['species'].value_counts()
    valid_species = species_counts[species_counts >= MIN_SEGMENTS_KNOWN]

    files_per_species = df.groupby('species')['original_file'].nunique()
    valid_species = valid_species[valid_species.index.isin(
        files_per_species[files_per_species >= MIN_FILES_PER_CLASS].index
    )]

    actual_n = min(TOP_N_KNOWN, len(valid_species))
    known_species = valid_species.head(actual_n).index.tolist()
    print(f"Known: {actual_n} видов (>= {MIN_SEGMENTS_KNOWN} сегментов, >= {MIN_FILES_PER_CLASS} файлов)")
    return known_species


def select_unseen_classes(df, known_species):
    species_counts = df['species'].value_counts()
    species_counts = species_counts[~species_counts.index.isin(known_species)]

    files_per_species = df.groupby('species')['original_file'].nunique()

    valid_species = species_counts[
        (species_counts >= MIN_SEGMENTS_UNSEEN) &
        (species_counts <= MAX_SEGMENTS_UNSEEN)
    ]
    valid_species = valid_species[valid_species.index.isin(
        files_per_species[files_per_species >= 2].index
    )]

    if len(valid_species) < UNSEEN_COUNT:
        unseen_species = valid_species.index.tolist()
    else:
        np.random.seed(RANDOM_SEED)
        unseen_species = np.random.choice(
            valid_species.index, size=UNSEEN_COUNT, replace=False
        ).tolist()

    print(f"Unseen: {len(unseen_species)} видов")
    return unseen_species


def stratified_split(df):
    files_df = df.groupby('original_file')['species'].first().reset_index()
    print(f"Файлов до фильтрации: {len(files_df)}")

    species_file_counts = files_df['species'].value_counts()
    valid_species = species_file_counts[species_file_counts >= MIN_FILES_PER_CLASS].index

    files_df = files_df[files_df['species'].isin(valid_species)]
    df = df[df['species'].isin(valid_species)]
    print(f"Файлов после фильтрации: {len(files_df)}")
    print(f"Классов: {files_df['species'].nunique()}")

    train_files, temp_files = train_test_split(
        files_df, train_size=TRAIN_RATIO,
        stratify=files_df['species'],
        random_state=RANDOM_SEED
    )

    temp_species_counts = temp_files['species'].value_counts()
    valid_temp_species = temp_species_counts[temp_species_counts >= 2].index
    temp_files = temp_files[temp_files['species'].isin(valid_temp_species)]

    val_ratio_adj = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_files, test_files = train_test_split(
        temp_files, train_size=val_ratio_adj,
        stratify=temp_files['species'],
        random_state=RANDOM_SEED
    )

    train_df = df[df['original_file'].isin(train_files['original_file'])].copy()
    val_df = df[df['original_file'].isin(val_files['original_file'])].copy()
    test_df = df[df['original_file'].isin(test_files['original_file'])].copy()

    train_files_set = set(train_df['original_file'].unique())
    val_files_set = set(val_df['original_file'].unique())
    test_files_set = set(test_df['original_file'].unique())

    assert len(train_files_set & val_files_set) == 0, "Leakage: train/val"
    assert len(train_files_set & test_files_set) == 0, "Leakage: train/test"
    assert len(val_files_set & test_files_set) == 0, "Leakage: val/test"

    print(f"Train: {len(train_files_set)} файлов, {len(train_df)} сегментов")
    print(f"Val: {len(val_files_set)} файлов, {len(val_df)} сегментов")
    print(f"Test: {len(test_files_set)} файлов, {len(test_df)} сегментов")

    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df, unseen_df):
    os.makedirs(SPLITS_DIR, exist_ok=True)

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    unseen_df = unseen_df.copy()

    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    unseen_df['split'] = 'unseen_test'

    train_df.to_csv(os.path.join(SPLITS_DIR, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(SPLITS_DIR, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(SPLITS_DIR, 'test.csv'), index=False)
    unseen_df.to_csv(os.path.join(SPLITS_DIR, 'unseen_test.csv'), index=False)

    print(f"Сохранено: {SPLITS_DIR}")


def main():
    print("=" * 60)
    print("CREATE SPLITS")
    print("=" * 60)

    df = load_manifest()

    known_species = select_known_classes(df)
    unseen_species = select_unseen_classes(df, known_species)

    known_df = df[df['species'].isin(known_species)].copy()
    unseen_df = df[df['species'].isin(unseen_species)].copy()

    print(f"\nKnown: {len(known_df)} сегментов")
    print(f"Unseen: {len(unseen_df)} сегментов")

    if len(known_df) > 0:
        train_df, val_df, test_df = stratified_split(known_df)
        save_splits(train_df, val_df, test_df, unseen_df)
    else:
        print("Недостаточно данных!")


if __name__ == '__main__':
    main()
