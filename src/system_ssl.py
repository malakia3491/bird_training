import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig

class SSLSystem(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.ssl_mode = cfg.get("ssl_mode", "contrastive") # contrastive, denoising, mae

        # 1. Frontend & Backbone (Всегда нужны)
        self.frontend = hydra.utils.instantiate(cfg.frontend)
        self.backbone = hydra.utils.instantiate(cfg.model)
        
        embed_dim = self.backbone.embed_dim if hasattr(self.backbone, 'embed_dim') else 2048

        # 2. Head (Projector) OR Decoder
        if self.ssl_mode == "contrastive":
            # Для SimCLR нужен Проектор (вектор -> вектор)
            self.head = hydra.utils.instantiate(cfg.head, in_features=embed_dim)
        else:
            # Для DAE/MAE нужен Декодер (вектор -> картинка)
            # cfg.decoder должен быть определен в конфиге
            self.head = hydra.utils.instantiate(cfg.decoder, in_features=embed_dim)

        # 3. Loss
        self.loss_fn = hydra.utils.instantiate(cfg.loss)

    def forward(self, x):
        # Возвращаем эмбеддинги для валидации
        return self.backbone(self.frontend(x))

    def _shared_step(self, batch):
        # Batch приходит от SSLAugmentations
        data, _ = batch
        
        # --- MODE 1: CONTRASTIVE (SimCLR) ---
        if self.ssl_mode == "contrastive":
            # data это список [x1, x2]
            x1, x2 = data[0], data[1]
            
            h1 = self.backbone(self.frontend(x1))
            h2 = self.backbone(self.frontend(x2))
            
            z1 = self.head(h1) # Projector
            z2 = self.head(h2) # Projector
            
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
            
            loss = self.loss_fn(z1, z2)
            return loss

        # --- MODE 2 & 3: GENERATIVE (Denoising / MAE) ---
        else:
            # data это кортеж (input, target)
            # input = зашумленный/маскированный
            # target = чистый оригинал
            inputs, targets = data[0], data[1]
            
            # Энкодер видит только испорченное
            features = self.backbone(self.frontend(inputs))
            
            # Декодер пытается восстановить оригинал
            reconstruction = self.head(features)
            
            # Лосс (обычно MSE)
            loss = self.loss_fn(reconstruction, targets)
            return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log(f"train_{self.ssl_mode}_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # На валидации аугментации могут отличаться, но логика та же
        loss = self._shared_step(batch)
        self.log(f"val_{self.ssl_mode}_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.cfg.optimizer, params=self.parameters())
        if "scheduler" in self.cfg:
            scheduler = hydra.utils.instantiate(self.cfg.scheduler, optimizer=optimizer)
            return [optimizer], [scheduler]
        return optimizer