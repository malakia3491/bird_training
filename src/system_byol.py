"""
BYOL (Bootstrap Your Own Latent) System.

Отличие от SimSiam: использует EMA (Exponential Moving Average) target network.

Paper: "Bootstrap Your Own Latent" (Grill et al., 2020)

Архитектура:
    Online:  x1 -> encoder -> projector -> predictor -> p1
    Target:  x2 -> encoder_ema -> projector_ema ────────> z2 (stop gradient)

    loss = MSE(normalize(p1), normalize(z2))

Target network обновляется через EMA:
    target_params = tau * target_params + (1 - tau) * online_params
"""

import copy
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig


class BYOLSystem(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # EMA параметры
        self.tau_base = cfg.get('tau', 0.996)  # Начальное значение tau
        self.tau = self.tau_base

        # 1. Frontend (общий для online и target)
        self.frontend = hydra.utils.instantiate(cfg.frontend)

        # 2. Online encoder (backbone)
        self.online_encoder = hydra.utils.instantiate(cfg.model)

        # 3. Online projector
        embed_dim = self.online_encoder.embed_dim if hasattr(self.online_encoder, 'embed_dim') else 2048
        proj_hidden = cfg.head.get('hidden_dim', 4096)
        proj_out = cfg.head.get('out_dim', 256)

        self.online_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, proj_hidden),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_out)
        )

        # 4. Predictor (только online)
        pred_hidden = cfg.get('predictor', {}).get('hidden_dim', 4096)

        self.predictor = nn.Sequential(
            nn.Linear(proj_out, pred_hidden),
            nn.BatchNorm1d(pred_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden, proj_out)
        )

        # 5. Target networks (EMA копии)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        # Отключаем градиенты для target
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    def forward(self, x):
        """Возвращает эмбеддинг для downstream tasks."""
        h = self.online_encoder(self.frontend(x))
        if h.ndim == 4:
            h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return h

    @torch.no_grad()
    def _update_target_network(self):
        """EMA update target networks."""
        for online_params, target_params in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            target_params.data = (
                self.tau * target_params.data + (1 - self.tau) * online_params.data
            )

        for online_params, target_params in zip(
            self.online_projector.parameters(),
            self.target_projector.parameters()
        ):
            target_params.data = (
                self.tau * target_params.data + (1 - self.tau) * online_params.data
            )

    def _byol_loss(self, p, z):
        """
        BYOL loss: normalized MSE.
        Эквивалентно 2 - 2 * cosine_similarity.
        """
        p = F.normalize(p, dim=-1, p=2)
        z = F.normalize(z, dim=-1, p=2)
        return 2 - 2 * (p * z).sum(dim=-1).mean()

    def _encode_online(self, x):
        """Online branch: encoder -> projector."""
        h = self.online_encoder(self.frontend(x))
        if h.ndim == 4:
            h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        # Projector ожидает 4D, но мы уже flatten
        z = self.online_projector[2:](h)  # Skip pool and flatten
        return z

    def _encode_target(self, x):
        """Target branch: encoder_ema -> projector_ema (no grad)."""
        with torch.no_grad():
            h = self.target_encoder(self.frontend(x))
            if h.ndim == 4:
                h = F.adaptive_avg_pool2d(h, 1).flatten(1)
            z = self.target_projector[2:](h)
        return z

    def _shared_step(self, batch):
        data, _ = batch

        if isinstance(data, list):
            x1, x2 = data[0], data[1]
        else:
            x1 = data
            x2 = data + 0.01 * torch.randn_like(data)

        # Online: encode -> project -> predict
        z1_online = self._encode_online(x1)
        z2_online = self._encode_online(x2)
        p1 = self.predictor(z1_online)
        p2 = self.predictor(z2_online)

        # Target: encode -> project (no grad)
        z1_target = self._encode_target(x1)
        z2_target = self._encode_target(x2)

        # Symmetric loss
        loss = 0.5 * (self._byol_loss(p1, z2_target) + self._byol_loss(p2, z1_target))

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train_byol_loss", loss, prog_bar=True)
        self.log("tau", self.tau, prog_bar=False)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update target network after each training step."""
        self._update_target_network()

        # Cosine schedule for tau (увеличиваем tau к концу обучения)
        # tau = 1 - (1 - tau_base) * (cos(pi * k / K) + 1) / 2
        if self.trainer.max_epochs:
            progress = self.current_epoch / self.trainer.max_epochs
            self.tau = 1 - (1 - self.tau_base) * (1 + torch.cos(torch.tensor(progress * 3.14159))) / 2
            self.tau = self.tau.item()

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("val_byol_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # BYOL использует LARS optimizer в оригинале, но AdamW тоже работает
        optimizer = hydra.utils.instantiate(self.cfg.optimizer, params=self.parameters())

        if "scheduler" in self.cfg:
            scheduler = hydra.utils.instantiate(
                self.cfg.scheduler,
                optimizer=optimizer
            )
            return [optimizer], [scheduler]

        return optimizer
