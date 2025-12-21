"""
Обучение SimCLR с Multi-Crop аугментациями.

Multi-Crop создает несколько views разного размера:
- 2 больших view (стандартный SimCLR)
- N маленьких crop (дополнительные positives)

Запуск:
    python train_ssl_multicrop.py

С кастомным конфигом:
    python train_ssl_multicrop.py --config-name=train_ssl_multicrop
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from src.system_ssl_multicrop import SSLMultiCropSystem
from src.data_loaders.ssl_datamodule import SSLDataModule


@hydra.main(config_path="configs", config_name="train_ssl_multicrop", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Seed
    pl.seed_everything(42)

    # DataModule
    datamodule = SSLDataModule(cfg)

    # Model
    model = SSLMultiCropSystem(cfg)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="val_ssl_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            filename="multicrop-{epoch:02d}-{val_ssl_loss:.4f}"
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Logger
    logger = WandbLogger(
        project=cfg.project_name,
        name=f"multicrop_bs{cfg.data.batch_size}_ep{cfg.trainer.max_epochs}",
        save_dir="outputs"
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=cfg.trainer.get("accumulate_grad_batches", 1),
        precision=cfg.trainer.get("precision", "16-mixed"),
        gradient_clip_val=cfg.trainer.get("gradient_clip_val", 1.0),
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(model, datamodule)

    print(f"\nОбучение завершено!")
    print(f"Лучший checkpoint: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
