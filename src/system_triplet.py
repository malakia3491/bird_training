import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, ListConfig

torch.serialization.add_safe_globals([DictConfig, ListConfig])

class TripletSystem(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.frontend = hydra.utils.instantiate(cfg.frontend)
        self.backbone = hydra.utils.instantiate(cfg.model)
        
        embed_dim = self.backbone.embed_dim if hasattr(self.backbone, 'embed_dim') else 2048
        
        # Для Triplet Loss обычно используют проекцию в небольшое пространство (128-512)
        # Можно использовать ProjectionHead из SSL
        self.head = hydra.utils.instantiate(cfg.head, in_features=embed_dim)
        
        self.loss_fn = hydra.utils.instantiate(cfg.loss)

    def forward(self, x):
        # Forward возвращает эмбеддинги
        x = self.frontend(x)
        x = self.backbone(x)
        x = self.head(x) # Projection to 128/256 dims
        # Важно: L2 нормализация обязательна для Triplet Loss
        x = F.normalize(x, p=2, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        
        embeddings = self.forward(inputs)
        
        # Лосс сам найдет тройки
        loss = self.loss_fn(embeddings, targets)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Валидация для Triplet сложнее (нужен KNN).
        # Пока просто логируем лосс (насколько хорошо разделяет)
        inputs, targets = batch
        embeddings = self.forward(inputs)
        loss = self.loss_fn(embeddings, targets)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.cfg.optimizer, params=self.parameters())
        if "scheduler" in self.cfg:
            scheduler = hydra.utils.instantiate(self.cfg.scheduler, optimizer=optimizer)
            return [optimizer], [scheduler]
        return optimizer