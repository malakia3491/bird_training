import os
import argparse
import pandas as pd
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import shutil

# --- КОНФИГУРАЦИЯ ---
# Параметры должны совпадать с твоим конфигом обучения!
SAMPLE_RATE = 32000
N_FFT = 1024
WIN_LENGTH = 1024
HOP_LENGTH = 320
N_MELS = 128
F_MIN = 50
F_MAX = 14000
TOP_DB = 80.0

class AudioProcessor:
    def __init__(self):
        # Инициализируем трансформации (будут созданы в каждом процессе)
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
        self.resamplers = {} # Кэш ресемплеров

    def get_resampler(self, orig_sr):
        if orig_sr not in self.resamplers:
            self.resamplers[orig_sr] = T.Resample(orig_sr, SAMPLE_RATE)
        return self.resamplers[orig_sr]

    def process_file(self, file_path):
        try:
            # 1. Загрузка
            waveform, sr = torchaudio.load(file_path)
            
            # 2. В моно
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 3. Ресемплинг
            if sr != SAMPLE_RATE:
                resampler = self.get_resampler(sr)
                waveform = resampler(waveform)

            # 4. Мел-спектрограмма
            mel = self.mel_transform(waveform)
            log_mel = self.db_transform(mel)
            
            # (1, F, T) -> (F, T) для экономии места на диске (numpy array)
            return log_mel.squeeze(0).numpy().astype(np.float32), waveform.shape[1] / SAMPLE_RATE
            
        except Exception as e:
            return None, str(e)

# Глобальная функция для multiprocessing
def worker(args):
    source_path, dest_mel_path, processor = args
    
    if os.path.exists(dest_mel_path):
        # Если уже есть - пропускаем (удобно для дозагрузки)
        # Но проверим, не битый ли файл
        try:
            np.load(dest_mel_path)
            return True, None, 0 # 0 duration (лениво считать)
        except:
            pass # Пересчитываем

    mel_data, info = processor.process_file(source_path)
    
    if mel_data is None:
        return False, info, 0 # info содержит ошибку

    # Сохраняем
    np.save(dest_mel_path, mel_data)
    return True, None, info

def main():
    parser = argparse.ArgumentParser(description="Подготовка SSL датасета (OGG -> NPY)")
    parser.add_argument("--manifest_txt", type=str, required=True, help="Путь к ssl_manifest_12.0gb.txt")
    parser.add_argument("--output_dir", type=str, required=True, help="Куда сохранять новый датасет")
    parser.add_argument("--num_workers", type=int, default=8, help="Количество процессов")
    parser.add_argument("--copy_audio", action="store_true", help="Копировать ли исходные ogg файлы в папку records (долго!)")
    
    args = parser.parse_args()

    # Пути
    root_out = Path(args.output_dir)
    mel_out = root_out / "mel"
    records_out = root_out / "records"
    
    mel_out.mkdir(parents=True, exist_ok=True)
    if args.copy_audio:
        records_out.mkdir(parents=True, exist_ok=True)

    # Чтение манифеста
    print(f"Reading manifest: {args.manifest_txt}")
    with open(args.manifest_txt, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Found {len(lines)} files. Preparing tasks...")

    tasks = []
    metadata = []
    
    # Инициализируем процессор один раз (для проверки путей), но копировать его в процессы будем аккуратно
    # PyTorch трансформации иногда плохо пиклятся, поэтому лучше создавать их внутри воркера.
    # Но для простоты передадим объект, если Windows не будет ругаться.
    # Если будет - перенесем создание внутрь.
    processor = AudioProcessor()

    for line in tqdm(lines, desc="Preparing paths"):
        # line: ssl_dataset/acafly/XC130140.ogg
        src_path = Path(line)
        
        # Парсим структуру
        # Предполагаем структуру: <dataset_root>/<class>/<filename>
        # Нам нужно вытащить класс (species)
        parts = src_path.parts
        if len(parts) >= 2:
            species = parts[-2]
            filename = parts[-1]
            stem = src_path.stem # XC130140
        else:
            print(f"Skipping weird path: {line}")
            continue

        # Путь назначения для mel
        # mel/acafly/XC130140.npy
        dest_mel_folder = mel_out / species
        dest_mel_folder.mkdir(exist_ok=True)
        dest_mel_path = dest_mel_folder / (stem + ".npy")
        
        # Копирование аудио (если нужно)
        if args.copy_audio:
            dest_audio_folder = records_out / species
            dest_audio_folder.mkdir(exist_ok=True)
            dest_audio_path = dest_audio_folder / filename
            if not dest_audio_path.exists():
                try:
                    shutil.copy2(src_path, dest_audio_path)
                except Exception as e:
                    print(f"Copy error {src_path}: {e}")

        # Добавляем в список задач
        # (Мы передаем абсолютный путь к source, если скрипт запущен не из корня, это важно)
        tasks.append((str(src_path.resolve()), str(dest_mel_path), processor))
        
        # Метаданные для CSV (пока без длительности, её вернет воркер)
        metadata.append({
            "original_path": str(src_path.resolve()),
            "mel_path": str(dest_mel_path.resolve()), # Абсолютный путь для CSV
            "species": species,
            "filename": stem
        })

    print(f"Processing {len(tasks)} files with {args.num_workers} workers...")

    success_count = 0
    results_meta = []
    
    # Запуск процесса
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # map возвращает результаты в том же порядке, что и tasks
        results = list(tqdm(executor.map(worker, tasks), total=len(tasks), desc="Converting"))

    # Сборка результатов
    for i, (success, error, duration) in enumerate(results):
        if success:
            meta = metadata[i]
            meta['duration'] = duration
            # Для SSL часто ставят класс 'unknown' или реальный, если он есть.
            # Мы оставим реальный (species), но в обучении будем его игнорировать.
            results_meta.append(meta)
            success_count += 1
        else:
            print(f"Failed {metadata[i]['original_path']}: {error}")

    print(f"Finished! Success: {success_count}/{len(tasks)}")

    # Сохранение манифеста
    df = pd.DataFrame(results_meta)
    manifest_path = root_out / "manifest.csv"
    df.to_csv(manifest_path, index=False)
    print(f"Manifest saved to: {manifest_path}")

    # Создание splits папки (для совместимости с DataModule)
    splits_dir = root_out / "splits" / "train"
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Для SSL мы можем использовать все данные как train
    df.to_csv(splits_dir / "ssl_all.csv", index=False)
    print(f"Split created at: {splits_dir / 'ssl_all.csv'}")

if __name__ == "__main__":
    # Фикс для Windows multiprocessing
    torch.set_num_threads(1) 
    main()