import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import os
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger # <--- Новый импорт

# Импорт instantiate для создания классов из конфига
from hydra.utils import instantiate

from src.system import BirdClassifier
from src.utils.reporter import ExperimentReporter

@hydra.main(config_path="configs", config_name="train_config", version_base="1.3")
def main(cfg: DictConfig):
    pl.seed_everything(42)

    # 0. Получаем путь к папке outputs (которую создала Hydra)
    # Это папка вида: outputs/2025-12-14/19-45-00
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"📂 Рабочая папка эксперимента: {output_dir}")

    # 1. Данные
    dm = instantiate(cfg.data)
    dm.setup()
    
    # Обновляем конфиг модели количеством классов
    cfg.model.num_classes = dm.num_classes
    
    # 2. Модель
    model = BirdClassifier(cfg)

    # 3. Логгер (чтобы не создавалась папка lightning_logs в корне)
    # save_dir=output_dir -> логи упадут внутрь папки Hydra
    # name="logs" -> подпапка logs
    # version="" -> не создавать version_0, version_1...
    logger = CSVLogger(save_dir=output_dir, name="logs", version="")

    # 4. Callbacks
    # Чекпоинты сохраняем явно в output_dir/checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"), # <--- Явный путь
        monitor='val_loss',
        filename='{epoch}-{val_loss:.4f}',
        save_top_k=1,
        mode='min',
    )
    
    # Наш репортер
    reporter = ExperimentReporter(cfg, output_dir)

    # 5. Трейнер
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        callbacks=[checkpoint_callback, reporter],
        logger=logger, # <--- Подключаем наш логгер
        log_every_n_steps=cfg.trainer.log_every_n_steps
    )

    # 6. Обучение
    trainer.fit(model, dm)

if __name__ == "__main__":
    main()