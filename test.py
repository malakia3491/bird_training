import torch
import pytorch_lightning as pl
import pandas as pd
from hydra.utils import instantiate
from omegaconf import OmegaConf
import os
import shutil

CKPT_PATH = "D:/coding/source/dissertation/bird_training/outputs/2025-12-17/20-36-11/checkpoints/epoch=15-val_loss=0.3166.ckpt"
DATA_ROOT = "D:/coding/data/birds_common/data_russian" 

def main():
    print(f"🔍 Loading Checkpoint: {CKPT_PATH}")
    
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
        
    checkpoint = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    cfg = checkpoint['hyper_parameters']['cfg']
    cfg.data.root_dir = DATA_ROOT
    
    # --- ХАК ДЛЯ ИСПРАВЛЕНИЯ МЕТОК ---
    # Мы временно переименуем глобальный label_encoder.pkl, чтобы DataModule не нашел его
    # и создал новый, правильный энкодер на основе CSV файлов эксперимента.
    
    encoder_path = os.path.join(DATA_ROOT, 'checkpoints', 'label_encoder.pkl')
    temp_encoder_path = os.path.join(DATA_ROOT, 'checkpoints', 'label_encoder.pkl.bak')
    renamed = False
    
    if os.path.exists(encoder_path):
        print("⚠️ Found existing LabelEncoder. Hiding it to force clean regeneration...")
        try:
            os.rename(encoder_path, temp_encoder_path)
            renamed = True
        except OSError:
            print("❌ Cannot rename LabelEncoder (file used?). Logic might fail.")

    try:
        # 2. Инициализируем данные
        print("📦 Initializing DataModule (Rebuilding Encoder)...")
        from src.data_loaders.bird_datamodule import BirdDataModule
        
        dm = BirdDataModule(
            root_dir=cfg.data.root_dir,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            resize_shape=cfg.data.get('resize_shape', None)
        )
        dm.setup()
        
        # 3. Инициализируем модель
        print(f"🧠 Initializing Model (Num Classes in Config: {cfg.model.num_classes})...")
        print(f"   Num Classes in Data: {dm.num_classes}")
        
        if dm.num_classes != cfg.model.num_classes:
            print("🚨 WARNING: Mismatch in class count! Model might fail.")
        
        from src.system import BirdClassifier
        # Эвристика для выбора класса
        if 'head' in cfg and 'ArcFace' in cfg.head.get('_target_', ''):
            from src.system_metric import MetricLearningSystem
            model_cls = MetricLearningSystem
        else:
            model_cls = BirdClassifier

        model = model_cls(cfg)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        # 4. Трейнер
        trainer = pl.Trainer(accelerator="gpu", devices=1, logger=False)

        # 5. Функция оценки
        results = {}

        def evaluate_split(split_name, dataloader):
            if dataloader is None or len(dataloader) == 0:
                print(f"⚠️  Skipping {split_name}: DataLoader is empty.")
                return None
            print(f"\n🚀 Testing on {split_name} Set...")
            metrics_list = trainer.test(model, dataloaders=dataloader, verbose=False)
            metrics = metrics_list[0]
            clean_metrics = {k.replace('test_', ''): v for k, v in metrics.items()}
            return clean_metrics

        # --- ЗАПУСК ---
        results['Test'] = evaluate_split('Test', dm.test_dataloader())
        results['Val'] = evaluate_split('Validation', dm.val_dataloader())
        # Train можно пропустить для скорости, если уверен
        results['Train'] = evaluate_split('Train', dm.train_dataloader())

        # --- ВЫВОД ---
        print("\n" + "="*60)
        print(f"📊 FINAL REPORT: {cfg.project_name}")
        print("="*60)
        
        df_data = []
        metric_names = ['loss', 'f1', 'acc', 'precision', 'recall']
        
        for split_name, res in results.items():
            if res is None: continue
            row = {'Split': split_name}
            for m in metric_names:
                val = "N/A"
                for k, v in res.items():
                    if m in k:
                        val = f"{v:.4f}"
                        break
                row[m.capitalize()] = val
            df_data.append(row)

        df = pd.DataFrame(df_data)
        print(df.to_string(index=False))
        print("="*60)
        
        output_csv = "final_metrics_verification.csv"
        df.to_csv(output_csv, index=False)
        print(f"Saved to {output_csv}")

    finally:
        # ВОЗВРАЩАЕМ ФАЙЛ НА МЕСТО
        if renamed and os.path.exists(temp_encoder_path):
            print("Restoring original LabelEncoder file...")
            if os.path.exists(encoder_path):
                os.remove(encoder_path) # Удаляем тот, что создали сейчас
            os.rename(temp_encoder_path, encoder_path)

if __name__ == "__main__":
    main()