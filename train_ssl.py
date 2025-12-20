import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from hydra.utils import instantiate
import os

from src.system_ssl import SSLSystem
from src.data_loaders.ssl_transforms import ContrastiveTransform, create_ssl_transform
from src.data_loaders.ssl_datamodule import SSLDataModule
from src.utils.ssl_reporter import SSLExperimentReporter


@hydra.main(config_path="configs", config_name="train_ssl_config", version_base="1.3")
def main(cfg: DictConfig):
    pl.seed_everything(42)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # 1. Resolve конфига (превращаем переменные в числа)
    OmegaConf.resolve(cfg)

    # 2. Аугментации (через Hydra или фабрику)
    if "augmentation" in cfg:
        ssl_transform = instantiate(cfg.augmentation)
        print(f"[SSL] Используем аугментации из конфига: {cfg.augmentation._target_}")
    else:
        # Fallback: сильные аугментации по умолчанию
        ssl_transform = create_ssl_transform(
            mode='contrastive',
            strength='strong',
            input_size=(128, 313)
        )
        print("[SSL] Используем дефолтные сильные аугментации")

    # 3. Данные - используем SSLDataModule для on-the-fly конвертации
    dm = SSLDataModule(
        root_dir=cfg.data.get('root_dir'),
        manifest_txt=cfg.data.get('manifest_txt'),
        manifest_csv=cfg.data.get('manifest_csv'),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        train_transform=ssl_transform,
        val_transform=ssl_transform,
        max_duration=cfg.data.get('max_duration', 10.0),
        val_split=cfg.data.get('val_split', 0.1),
    )

    dm.setup()
    
    # 4. Модель
    model = SSLSystem(cfg)

    # 5. Логгер
    logger = CSVLogger(save_dir=output_dir, name="logs", version="")
    
    reporter = SSLExperimentReporter(cfg, output_dir)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename='ssl_model-{epoch:02d}-{train_ssl_loss:.4f}',
        monitor='val_ssl_loss',
        save_top_k=1,
        mode='min',
        save_last=True
    )

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[checkpoint_callback, reporter],
        logger=logger,
        log_every_n_steps=10
    )

    trainer.fit(model, dm)

if __name__ == "__main__":
    main()