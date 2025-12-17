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

        # 1. Frontend
        self.frontend = hydra.utils.instantiate(cfg.frontend)

        # 2. Backbone
        self.backbone = hydra.utils.instantiate(cfg.model)

        # 3. Projector
        # Определяем размер эмбеддинга динамически
        embed_dim = self.backbone.embed_dim if hasattr(self.backbone, 'embed_dim') else 2048
        
        self.projector = hydra.utils.instantiate(
            cfg.head, 
            in_features=embed_dim
        )

        # 4. Loss
        self.loss_fn = hydra.utils.instantiate(cfg.loss)

    def forward(self, x):
        # Используется для получения эмбеддингов
        return self.backbone(self.frontend(x))

    def _shared_step(self, batch):
        data, _ = batch
        
        # Если работает аугментация, data - это список [x1, x2]
        if isinstance(data, list):
            x1, x2 = data[0], data[1]
        else:
            # Fallback (на случай, если аугментация не сработала или на валидации)
            # Добавляем шум, чтобы создать "вторую версию"
            x1 = data
            x2 = data + 0.01 * torch.randn_like(data)
            
        h1 = self.backbone(self.frontend(x1))
        h2 = self.backbone(self.frontend(x2))

        z1 = self.projector(h1)
        z2 = self.projector(h2)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        loss = self.loss_fn(z1, z2)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train_ssl_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("val_ssl_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Создаем оптимизатор из конфига
        optimizer = hydra.utils.instantiate(self.cfg.optimizer, params=self.parameters())
        
        # Создаем планировщик, если он есть
        if "scheduler" in self.cfg:
            scheduler = hydra.utils.instantiate(
                self.cfg.scheduler, 
                optimizer=optimizer
            )
            return [optimizer], [scheduler]
            
        return optimizer