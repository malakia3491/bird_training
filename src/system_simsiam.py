"""
SimSiam (Simple Siamese) System.

Проще чем BYOL: нет EMA target network.
Работает через stop-gradient и симметричный loss.

Paper: "Exploring Simple Siamese Representation Learning" (Chen & He, 2020)

Архитектура:
    x1 -> encoder -> projector -> z1 -> predictor -> p1
    x2 -> encoder -> projector -> z2 -> predictor -> p2

    loss = -0.5 * (cos_sim(p1, stop_grad(z2)) + cos_sim(p2, stop_grad(z1)))
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig


class SimSiamSystem(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # 1. Frontend
        self.frontend = hydra.utils.instantiate(cfg.frontend)

        # 2. Backbone (encoder)
        self.backbone = hydra.utils.instantiate(cfg.model)

        # 3. Projector
        embed_dim = self.backbone.embed_dim if hasattr(self.backbone, 'embed_dim') else 2048
        proj_hidden = cfg.head.get('hidden_dim', 2048)
        proj_out = cfg.head.get('out_dim', 2048)  # SimSiam использует больший out_dim

        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, proj_hidden),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_hidden),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_out),
            nn.BatchNorm1d(proj_out, affine=False)  # Без affine параметров
        )

        # 4. Predictor (асимметрия - ключ к работе SimSiam)
        pred_hidden = cfg.get('predictor', {}).get('hidden_dim', 512)

        self.predictor = nn.Sequential(
            nn.Linear(proj_out, pred_hidden),
            nn.BatchNorm1d(pred_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden, proj_out)
        )

    def forward(self, x):
        """Возвращает эмбеддинг для downstream tasks."""
        h = self.backbone(self.frontend(x))
        if h.ndim == 4:
            h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return h

    def _cosine_similarity_loss(self, p, z):
        """Negative cosine similarity."""
        z = z.detach()  # stop gradient
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()

    def _shared_step(self, batch):
        data, _ = batch

        if isinstance(data, list):
            x1, x2 = data[0], data[1]
        else:
            x1 = data
            x2 = data + 0.01 * torch.randn_like(data)

        # Encode
        h1 = self.backbone(self.frontend(x1))
        h2 = self.backbone(self.frontend(x2))

        # Handle different output shapes
        if h1.ndim == 4:
            h1 = F.adaptive_avg_pool2d(h1, 1).flatten(1)
            h2 = F.adaptive_avg_pool2d(h2, 1).flatten(1)

        # Project
        z1 = self.projector(h1.unsqueeze(-1).unsqueeze(-1) if h1.ndim == 2 else h1)
        z2 = self.projector(h2.unsqueeze(-1).unsqueeze(-1) if h2.ndim == 2 else h2)

        # Predict
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # Symmetric loss
        loss = 0.5 * (self._cosine_similarity_loss(p1, z2) +
                      self._cosine_similarity_loss(p2, z1))

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train_simsiam_loss", loss, prog_bar=True)

        # Логируем std проекций для мониторинга коллапса
        # Если std -> 0, модель коллапсирует
        with torch.no_grad():
            data, _ = batch
            if isinstance(data, list):
                x = data[0]
            else:
                x = data
            h = self.backbone(self.frontend(x))
            if h.ndim == 4:
                h = F.adaptive_avg_pool2d(h, 1).flatten(1)
            z = self.projector(h.unsqueeze(-1).unsqueeze(-1) if h.ndim == 2 else h)
            std = z.std(dim=0).mean()
            self.log("z_std", std, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("val_simsiam_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # SimSiam работает лучше с SGD и высоким LR
        if self.cfg.get('use_sgd', False):
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.cfg.optimizer.get('lr', 0.05),
                momentum=0.9,
                weight_decay=self.cfg.optimizer.get('weight_decay', 0.0001)
            )
        else:
            optimizer = hydra.utils.instantiate(self.cfg.optimizer, params=self.parameters())

        if "scheduler" in self.cfg:
            scheduler = hydra.utils.instantiate(
                self.cfg.scheduler,
                optimizer=optimizer
            )
            return [optimizer], [scheduler]

        return optimizer
