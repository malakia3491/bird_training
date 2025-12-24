import torch
import torch.nn.functional as F
import hydra
from src.systems.base import BaseAudioSystem

class SSLSystem(BaseAudioSystem):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ssl_mode = cfg.get("ssl_mode", "contrastive") # contrastive / denoising / reconstruction

        # Голова (Projector или Decoder)
        # В конфиге task мы мапим нужное (head или decoder) в поле `head`
        self.head = hydra.utils.instantiate(
            cfg.head, 
            in_features=self.embed_dim
        )
        
        self.loss_fn = hydra.utils.instantiate(cfg.loss)

    def _shared_step(self, batch, prefix="train"):
        data, _ = batch # Метки не нужны
        
        # --- SimCLR ---
        if self.ssl_mode == "contrastive":
            x1, x2 = data[0], data[1]
            h1 = self.backbone(self.frontend(x1))
            h2 = self.backbone(self.frontend(x2))
            
            z1 = F.normalize(self.head(h1), dim=1)
            z2 = F.normalize(self.head(h2), dim=1)
            
            loss = self.loss_fn(z1, z2)
            
        # --- DAE / MAE ---
        else:
            inputs, targets = data[0], data[1]
            if len(data) == 3: mask = data[2] # Для MAE может быть маска
            
            features = self.backbone(self.frontend(inputs))
            reconstruction = self.head(features)
            loss = self.loss_fn(reconstruction, targets)

        self.log(f"{prefix}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")