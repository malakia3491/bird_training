import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import os
from hydra.utils import instantiate

# Импортируем новую систему
from src.system_triplet import TripletSystem
from src.utils.reporter import ExperimentReporter

@hydra.main(config_path="configs", config_name="train_metric_config", version_base="1.3")
def main(cfg: DictConfig):
    pl.seed_everything(42)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    # 1. Данные
    # Используем стандартный BirdDataModule (тот же, что для классификации)
    dm = instantiate(cfg.data)
    dm.setup()
    
    # Обновляем конфиг модели числом классов (Критично для ArcFace!)
    cfg.model.num_classes = dm.num_classes
    
    print(f"Dataset classes: {dm.num_classes}")

    # 2. Модель (Metric Learning)
    model = TripletSystem(cfg)

    # 3. Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        monitor='val_f1', # Для метрик леарнинг лучше следить за F1, а не Loss
        filename='arcface-{epoch:02d}-{val_f1:.4f}',
        save_top_k=1,
        mode='max', # Максимизируем F1
        save_last=True
    )
    
    # Используем стандартный репортер (он умеет строить матрицы ошибок)
    reporter = ExperimentReporter(cfg, output_dir)
    logger = CSVLogger(save_dir=output_dir, name="logs", version="")

    # 4. Трейнер
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        callbacks=[checkpoint_callback, reporter],
        logger=logger,
        log_every_n_steps=10
    )

    # 5. Обучение
    trainer.fit(model, dm)

    print("\nStarting Testing on 'splits/test'...")
    # ckpt_path="best" загрузит лучшие веса (по val_loss или val_f1), сохраненные чекпоинтом
    trainer.test(model, datamodule=dm, ckpt_path="best")
    

if __name__ == "__main__":
    main()