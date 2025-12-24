import pytorch_lightning as pl
import torch
import hydra
from omegaconf import DictConfig
import os

class BaseAudioSystem(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # 1. Инициализация (стандарт для всех задач)
        self.frontend = hydra.utils.instantiate(cfg.frontend)
        self.backbone = hydra.utils.instantiate(cfg.model)
        
        # Если в конфиге есть флаг compile=True
        if cfg.get("compile", False):
            print("🚀 EXTREME SPEEDUP: Compiling Backbone with torch.compile()...")
            # Windows имеет ограниченную поддержку, но 'default' или 'inductor' часто работают
            # На Linux идеально mode="reduce-overhead"
            try:
                self.backbone = torch.compile(self.backbone)
                self.frontend = torch.compile(self.frontend)
                print("✅ Compilation initialized.")
            except Exception as e:
                print(f"⚠️ Compilation failed (Windows?): {e}")  
        
        # Вычисляем размерность эмбеддинга
        self.embed_dim = self.backbone.embed_dim if hasattr(self.backbone, 'embed_dim') else 2048

        # 2. Загрузка весов от другого эксперимента (Chaining)
        if cfg.get("pretrained_from"):
            self._load_pretrained_weights(cfg.pretrained_from)

    def _load_pretrained_weights(self, ckpt_path):
        print(f"🔄 Loading backbone weights from: {ckpt_path}")
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")
            
        # Грузим безопасно (обходя защиту PyTorch 2.6 для Hydra)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]
        
        # Фильтруем веса: оставляем только backbone
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("backbone."):
                # Убираем префикс "backbone." чтобы загрузить прямо в self.backbone
                new_key = k.replace("backbone.", "")
                new_state_dict[new_key] = v
        
        # Загружаем (strict=False, так как мы грузим ТОЛЬКО бэкбон, без головы)
        missing, unexpected = self.backbone.load_state_dict(new_state_dict, strict=False)
        print(f"✅ Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    def configure_optimizers(self):
        # Универсальный конфигуратор из Hydra
        optimizer = hydra.utils.instantiate(self.cfg.optimizer, params=self.parameters())
        if "scheduler" in self.cfg:
            scheduler = hydra.utils.instantiate(self.cfg.scheduler, optimizer=optimizer)
            return [optimizer], [scheduler]
        return optimizer

    def forward(self, x):
        # Базовый проход: Аудио -> Эмбеддинг
        return self.backbone(self.frontend(x))