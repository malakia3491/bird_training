import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig


class SSLMultiCropSystem(pl.LightningModule):
    """
    SSL система с поддержкой Multi-Crop.

    Multi-Crop создает несколько views разного размера:
    - Большие views (полный размер) - основные positive pairs
    - Маленькие views (crop) - дополнительные positives

    Все views от одного сэмпла считаются positives друг для друга.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # Frontend
        self.frontend = hydra.utils.instantiate(cfg.frontend)

        # Backbone
        self.backbone = hydra.utils.instantiate(cfg.model)

        # Projector
        embed_dim = self.backbone.embed_dim if hasattr(self.backbone, 'embed_dim') else 2048

        self.projector = hydra.utils.instantiate(
            cfg.head,
            in_features=embed_dim
        )

        # Loss
        self.loss_fn = hydra.utils.instantiate(cfg.loss)

        # Temperature для multi-crop loss
        self.temperature = getattr(cfg.loss, 'temperature', 0.5)

    def forward(self, x):
        return self.backbone(self.frontend(x))

    def _compute_embeddings(self, views):
        """Вычисляет нормализованные эмбеддинги для всех views."""
        embeddings = []
        for v in views:
            h = self.backbone(self.frontend(v))
            z = self.projector(h)
            z = F.normalize(z, dim=1)
            embeddings.append(z)
        return embeddings

    def _multicrop_loss(self, embeddings):
        """
        Multi-Crop NT-Xent Loss.

        Все views от одного сэмпла - positives.
        Views от разных сэмплов - negatives.
        """
        n_views = len(embeddings)
        batch_size = embeddings[0].shape[0]

        # Конкатенируем все embeddings
        # Shape: [n_views * batch_size, dim]
        z = torch.cat(embeddings, dim=0)

        # Similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature

        # Маска для удаления диагонали
        n_total = n_views * batch_size
        mask = torch.eye(n_total, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float('-inf'))

        # Создаем маску positives
        # Сэмпл i в view v является positive для сэмпла i во всех других views
        positive_mask = torch.zeros(n_total, n_total, dtype=torch.bool, device=z.device)

        for i in range(n_views):
            for j in range(n_views):
                if i != j:
                    # View i, sample k -> View j, sample k (positives)
                    for k in range(batch_size):
                        idx_i = i * batch_size + k
                        idx_j = j * batch_size + k
                        positive_mask[idx_i, idx_j] = True

        # NT-Xent loss
        # Для каждой строки: log_softmax и затем выбираем positives
        log_prob = F.log_softmax(sim, dim=1)

        # Среднее по всем positive pairs
        loss = -log_prob[positive_mask].mean()

        return loss

    def _shared_step(self, batch):
        data, _ = batch

        # Multi-crop возвращает список views
        if isinstance(data, list):
            if len(data) == 2:
                # Стандартный случай: 2 views
                z1, z2 = self._compute_embeddings(data)
                loss = self.loss_fn(z1, z2)
            else:
                # Multi-crop: много views
                embeddings = self._compute_embeddings(data)
                loss = self._multicrop_loss(embeddings)
        else:
            # Fallback: создаем второй view с шумом
            x1 = data
            x2 = data + 0.01 * torch.randn_like(data)
            z1, z2 = self._compute_embeddings([x1, x2])
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
        optimizer = hydra.utils.instantiate(self.cfg.optimizer, params=self.parameters())

        if "scheduler" in self.cfg:
            scheduler = hydra.utils.instantiate(
                self.cfg.scheduler,
                optimizer=optimizer
            )
            return [optimizer], [scheduler]

        return optimizer
