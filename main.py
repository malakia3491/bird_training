import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from hydra.utils import instantiate, get_class
import os
import torch

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # 1. Resolve & Seed
    OmegaConf.resolve(cfg)
    pl.seed_everything(cfg.seed)
    
    # 2. Output Dir
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"🚀 Experiment: {cfg.experiment_name} / {cfg.run_name}")
    print(f"📂 Output Dir: {output_dir}")

    # 3. Данные (DataModule)
    # --- ЛОГИКА ОБРАБОТКИ АУГМЕНТАЦИЙ ---
    # Проверяем, что пришло в augmentation: реальный конфиг или заглушка (target=None)
    aug_config = cfg.get("augmentation")
    
    if aug_config is None or aug_config.get("_target_") is None:
        # Случай "none.yaml" -> Выключаем аугментацию в данных
        print("⚠️  Augmentation: Disabled (None)")
        
        # Если задача требует аугментации, падаем с понятной ошибкой
        if cfg.task.get("name") in ["ssl_contrastive", "supcon_learning"]:
             raise ValueError(
                 f"❌ CRITICAL: Task '{cfg.task.name}' requires augmentation!\n"
                 "Please add `- override /augmentation: contrastive_strong` to your experiment file."
             )
        
        # Зануляем трансформ в конфиге данных перед инстанцированием
        if "train_transform" in cfg.data:
            cfg.data.train_transform = None
            
    else:
        print(f"🔧 Augmentation: {aug_config._target_}")
        # Здесь cfg.data.train_transform уже ссылается на cfg.augmentation
        # Hydra корректно инстанцирует это рекурсивно.

    dm = instantiate(cfg.data)
    dm.setup()
    
    # Обновляем num_classes
    if hasattr(dm, "num_classes") and dm.num_classes:
        print(f"ℹ️  Detected {dm.num_classes} classes.")
        cfg.model.num_classes = dm.num_classes
    else:
        cfg.model.num_classes = 0

    # 4. Система (Модель)
    print(f"🧠 System Class: {cfg.task.system_class._target_}")
    SystemClass = get_class(cfg.task.system_class._target_)
    model = SystemClass(cfg) 

    # 5. Logger & Callbacks
    logger = CSVLogger(save_dir=output_dir, name="logs", version="")
    ckpt_cb = instantiate(cfg.callbacks.model_checkpoint, dirpath=os.path.join(output_dir, "checkpoints"))
    reporter = instantiate(cfg.callbacks.reporter, cfg=cfg, output_dir=output_dir)

    # 6. Trainer
    trainer = instantiate(
        cfg.trainer,
        callbacks=[ckpt_cb, reporter],
        logger=logger
    )

    # 7. Train
    trainer.fit(model, dm)
    
    # 8. Test (Only for supervised tasks)
    if cfg.task.get("name") in ["classification", "metric_learning"]:
        print("\n🧪 Starting Test Cycle...")
        if ckpt_cb.best_model_path and os.path.exists(ckpt_cb.best_model_path):
            print(f"Loading best: {ckpt_cb.best_model_path}")
            checkpoint = torch.load(ckpt_cb.best_model_path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
            trainer.test(model, datamodule=dm)
        else:
            print("⚠️ Testing with final weights (no checkpoint found).")
            trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    main()