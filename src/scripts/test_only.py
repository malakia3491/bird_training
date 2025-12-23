import torch
import pytorch_lightning as pl
from system import BirdClassifier
from data_loaders.bird_datamodule import BirdDataModule
from utils.reporter import ExperimentReporter
from omegaconf import OmegaConf

# УКАЖИ ПУТЬ К ТВОЕМУ ЧЕКПОИНТУ ЗДЕСЬ
CKPT_PATH = "D:/coding/source/dissertation/bird_training/outputs/2025-12-21/13-36-03/checkpoints/epoch=39-val_loss=2.1018.ckpt"
DATA_ROOT = "D:/coding/data/birds_common/data_russian" # Или processed_ssl_dataset, смотря на чем учил

def main():
    print(f"Loading checkpoint: {CKPT_PATH}")
    
    # 1. Грузим всё из чекпоинта (безопасно)
    checkpoint = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    
    # Восстанавливаем конфиг
    cfg = checkpoint['hyper_parameters']['cfg']
    
    # 2. Данные
    dm = BirdDataModule(
        root_dir=DATA_ROOT,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers
    )
    dm.setup()
    
    # 3. Модель
    model = BirdClassifier(cfg)
    model.load_state_dict(checkpoint['state_dict'])
    
    # 4. Трейнер (только для теста)
    trainer = pl.Trainer(accelerator="gpu", devices=1)
    
    # 5. Запуск
    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    main()