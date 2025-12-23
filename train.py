import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import os
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from hydra.utils import instantiate

from src.system import BirdClassifier
from src.utils.reporter import ExperimentReporter

@hydra.main(config_path="configs", config_name="train_config", version_base="1.3")
def main(cfg: DictConfig):
    pl.seed_everything(42)

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"📂 Рабочая папка эксперимента: {output_dir}")

    # 1. Данные
    dm = instantiate(cfg.data)
    dm.setup()
    
    # Обновляем конфиг модели количеством классов
    cfg.model.num_classes = dm.num_classes
    
    # 2. Модель
    model = BirdClassifier(cfg)

    # 3. Логгер
    logger = CSVLogger(save_dir=output_dir, name="logs", version="")

    # 4. Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        monitor='val_loss',
        filename='{epoch}-{val_loss:.4f}',
        save_top_k=1,
        mode='min',
    )
    
    reporter = ExperimentReporter(cfg, output_dir)

    # 5. Трейнер
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        callbacks=[checkpoint_callback, reporter],
        logger=logger,
        log_every_n_steps=cfg.trainer.log_every_n_steps
    )

    # 6. Обучение
    trainer.fit(model, dm)
    
    # 7. Тестирование (С ИСПРАВЛЕНИЕМ)
    print("\nStarting Testing on 'splits/test'...")
    
    # Получаем путь к лучшему чекпоинту
    best_path = checkpoint_callback.best_model_path
    
    if best_path:
        print(f"Loading best checkpoint: {best_path}")
        # --- FIX: Грузим вручную с weights_only=False ---
        # Это обходит защиту PyTorch 2.6 для Hydra-конфигов
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        
        # Запускаем тест БЕЗ ckpt_path (так как веса уже в модели)
        trainer.test(model, datamodule=dm)
    else:
        print("⚠️ No checkpoint found! Testing with final weights.")
        trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    main()