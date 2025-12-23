import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import os
from hydra.utils import instantiate # <--- ВАЖНО

from src.system_ssl import SSLSystem
from src.data_loaders.bird_datamodule import BirdDataModule
from src.utils.ssl_reporter import SSLExperimentReporter

@hydra.main(config_path="configs", config_name="train_ssl_config", version_base="1.3")
def main(cfg: DictConfig):
    pl.seed_everything(42)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    OmegaConf.resolve(cfg)

    # 1. Аугментации
    # Теперь мы просто инстанцируем класс, указанный в конфиге (через _target_)
    # Все параметры (noise_dir, mask_ratio) подтянутся автоматически
    print(f"🔧 SSL Augmentations: {cfg.augmentation._target_}")
    ssl_transform = instantiate(cfg.augmentation)
    ssl_val_transform = instantiate(cfg.val_augmentation)
    # 2. Данные
    dm = BirdDataModule(
        root_dir=cfg.data.root_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        train_transform=ssl_transform,
        val_transform=ssl_val_transform
    )
    
    dm.setup()
    
    # 3. Модель
    model = SSLSystem(cfg)

    # 4. Логгер
    logger = CSVLogger(save_dir=output_dir, name="logs", version="")
    reporter = SSLExperimentReporter(cfg, output_dir)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        monitor=f"val_{model.ssl_mode}_loss", 
        filename='ssl-{epoch:02d}-{train_ssl_loss:.4f}',
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
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches
    )

    trainer.fit(model, dm)

if __name__ == "__main__":
    main()