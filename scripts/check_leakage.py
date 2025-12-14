import pandas as pd
import os
import glob

def check_leakage(root_dir):
    print(f"Checking {root_dir}...")
    
    # Загружаем Train
    train_files = glob.glob(os.path.join(root_dir, 'splits/train/*.csv'))
    if not train_files: train_files = [os.path.join(root_dir, 'splits/train.csv')]
    train_df = pd.concat([pd.read_csv(f) for f in train_files])
    
    # Загружаем Val
    val_files = glob.glob(os.path.join(root_dir, 'splits/val/*.csv'))
    if not val_files: val_files = [os.path.join(root_dir, 'splits/val.csv')]
    val_df = pd.concat([pd.read_csv(f) for f in val_files])

    # Получаем множества оригинальных файлов
    # Убедись, что колонка называется 'original_file' или 'filename' (без _seg...)
    # В твоем примере манифеста колонка: original_file
    train_originals = set(train_df['original_file'].unique())
    val_originals = set(val_df['original_file'].unique())

    # Ищем пересечения
    intersection = train_originals.intersection(val_originals)
    
    print(f"Train unique files: {len(train_originals)}")
    print(f"Val unique files: {len(val_originals)}")
    print(f"INTERSECTION (LEAKAGE): {len(intersection)}")
    
    if len(intersection) > 0:
        print("🔴 CRITICAL ERROR: Data Leakage detected!")
        print("Модель запоминает файлы, а не учит птиц.")
    else:
        print("🟢 Split is correct. No leakage.")

# Укажи путь к папке data_russian
check_leakage("D:/coding/data/birds_common/data_russian")