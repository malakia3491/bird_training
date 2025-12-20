"""
Supervised Contrastive Learning Training Script.

Usage:
    python train_supcon.py

Отличия от SimCLR (train_ssl.py):
- Использует метки классов
- Сближает ВСЕ записи одного класса, а не только аугментации одного файла
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from hydra.utils import instantiate
import os

from src.system_supcon import SupConSystem
from src.data_loaders.ssl_transforms import create_ssl_transform
from src.data_loaders.supcon_datamodule import SupConDataModule
from src.utils.ssl_reporter import SSLExperimentReporter


@hydra.main(config_path="configs", config_name="train_supcon_config", version_base="1.3")
def main(cfg: DictConfig):
    pl.seed_everything(42)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    OmegaConf.resolve(cfg)

    # 1. Аугментации
    if "augmentation" in cfg:
        ssl_transform = instantiate(cfg.augmentation)
        print(f"[SupCon] Аугментации: {cfg.augmentation._target_}")
    else:
        ssl_transform = create_ssl_transform(
            mode='contrastive',
            strength='strong',
            input_size=(128, 313)
        )

    # 2. Данные с метками
    dm = SupConDataModule(
        root_dir=cfg.data.root_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        train_transform=ssl_transform,
        val_transform=ssl_transform,
        max_duration=cfg.data.get('max_duration', 10.0),
        val_split=cfg.data.get('val_split', 0.1),
    )

    dm.setup()

    print(f"[SupCon] Классов: {dm.num_classes}")

    # 3. Модель
    model = SupConSystem(cfg)

    # 4. Логгер и callbacks
    logger = CSVLogger(save_dir=output_dir, name="logs", version="")

    # Используем reporter от SSL (с небольшими изменениями имён метрик)
    reporter = SSLExperimentReporter(cfg, output_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename='supcon-{epoch:02d}-{train_supcon_loss:.4f}',
        monitor='val_supcon_loss',
        save_top_k=1,
        mode='min',
        save_last=True
    )

    # 5. Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.get('precision', '16-mixed'),
        accumulate_grad_batches=cfg.trainer.get('accumulate_grad_batches', 1),
        callbacks=[checkpoint_callback, reporter],
        logger=logger,
        log_every_n_steps=10
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
