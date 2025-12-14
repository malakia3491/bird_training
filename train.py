import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from hydra.utils import instantiate

from src.system import BirdClassifier
from src.utils.reporter import ExperimentReporter
    
@hydra.main(config_path="configs", config_name="train_config", version_base="1.3")
def main(cfg: DictConfig):
    pl.seed_everything(42)

    # 1. Данные

    dm = instantiate(cfg.data)
    dm.setup() # Запускаем setup явно, чтобы узнать num_classes
    
    # Обновляем конфиг модели количеством классов из данных
    cfg.model.num_classes = dm.num_classes
    
    # 2. Модель
    model = BirdClassifier(cfg)

    # 3. Callbacks
    # Чекпоинты (сохраняем лучшую модель по val_loss)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )
    
    # Наш репортер
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    reporter = ExperimentReporter(cfg, output_dir)

    # 4. Трейнер
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        callbacks=[checkpoint_callback, reporter],
        log_every_n_steps=10
    )

    # 5. Обучение
    trainer.fit(model, dm)

if __name__ == "__main__":
    main()