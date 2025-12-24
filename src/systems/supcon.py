import torch
import torch.nn.functional as F
import hydra
from src.systems.base import BaseAudioSystem

class SupConSystem(BaseAudioSystem):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.head = hydra.utils.instantiate(
            cfg.head, 
            in_features=self.embed_dim
        )
        self.loss_fn = hydra.utils.instantiate(cfg.loss)

    def _unpack_views(self, views):
        """
        Универсальная распаковка views.
        Может прийти:
        1. List[Tensor(B, ...), Tensor(B, ...)] - Стандартный collate
        2. Tensor(B, 2, ...) - Если collate склеил их
        """
        if isinstance(views, torch.Tensor):
            # views shape: (Batch, 2, 1, F, T) -> (Batch, 2, F, T) sometimes
            # Нам нужно разделить по 2-му измерению (dim=1)
            # unbind вернет кортеж (x1, x2), где каждый (Batch, ...)
            x1, x2 = torch.unbind(views, dim=1)
        elif isinstance(views, (list, tuple)):
            x1, x2 = views[0], views[1]
        else:
            raise TypeError(f"Unknown views type: {type(views)}")
            
        return x1, x2

    def _forward_step(self, images, labels, prefix):
        # 1. Forward
        features = self.backbone(self.frontend(images))
        projections = self.head(features)
        projections = F.normalize(projections, dim=1)
        
        # 2. Loss
        loss = self.loss_fn(projections, labels)
        
        # Логируем специфичное имя (чтобы видеть в прогресс-баре)
        self.log(f"{prefix}_supcon_loss", loss, prog_bar=True)
        
        # --- ИСПРАВЛЕНИЕ: Логируем стандартное имя для Callbacks/Reporter ---
        # Теперь ModelCheckpoint найдет 'val_loss' или 'train_loss'
        self.log(f"{prefix}_loss", loss, prog_bar=False) 
        # --------------------------------------------------------------------
        
        return loss

    def training_step(self, batch, batch_idx):
        (views, labels) = batch
        
        # --- ИСПРАВЛЕНИЕ: Безопасная распаковка ---
        x1, x2 = self._unpack_views(views)
        # ------------------------------------------
        
        # Теперь x1 и x2 точно имеют размер (Batch, ...)
        # Конкатенируем батч -> (2N, ...)
        images = torch.cat([x1, x2], dim=0)
        targets = torch.cat([labels, labels], dim=0)
        
        return self._forward_step(images, targets, "train")

    def validation_step(self, batch, batch_idx):
        (data, labels) = batch
        
        # На валидации аугментаций нет, поэтому data - это просто тензор (Batch, ...)
        # Но если вдруг пришел список (вдруг ауги включены):
        if isinstance(data, (list, tuple)) or (isinstance(data, torch.Tensor) and data.ndim == 5):
             # Если пришла пара, берем только первый взгляд
             if isinstance(data, torch.Tensor):
                 images = data[:, 0]
             else:
                 images = data[0]
        else:
             images = data
             
        return self._forward_step(images, labels, "val")

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.cfg.optimizer, params=self.parameters())
        if "scheduler" in self.cfg:
            scheduler = hydra.utils.instantiate(self.cfg.scheduler, optimizer=optimizer)
            return [optimizer], [scheduler]
        return optimizer