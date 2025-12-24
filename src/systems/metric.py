import torch
import hydra
import torchmetrics
from src.systems.base import BaseAudioSystem

class MetricSystem(BaseAudioSystem):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Голова (ArcFace или Projection для ProtoNet)
        self.head = hydra.utils.instantiate(
            cfg.head, 
            in_features=self.embed_dim, 
            num_classes=cfg.model.num_classes
        )
        self.loss_fn = hydra.utils.instantiate(cfg.loss)
        
        # Метрики
        nc = cfg.model.num_classes
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=nc, average='macro')

    def forward(self, x, labels=None):
        feats = super().forward(x)
        # Если голова ArcFace, она принимает labels
        # Если голова Linear/Projection, она игнорирует labels (или падает, если передать)
        
        # Хак для универсальности: проверяем сигнатуру или пробуем
        try:
            return self.head(feats, labels)
        except TypeError:
            return self.head(feats)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        embeddings_or_logits = self.forward(inputs, targets)
        
        loss = self.loss_fn(embeddings_or_logits, targets)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        # На валидации ArcFace возвращает cosine similarity (logits)
        out = self.forward(inputs, labels=None)
        
        # Если это ProtoNet/Triplet, out - это эмбеддинги. Лосс их съест.
        # Если это ArcFace, out - это логиты. Лосс (CE) их съест.
        loss = self.loss_fn(out, targets)
        
        # Метрики считаем, только если это ArcFace (есть логиты)
        # Для ProtoNet/Triplet валидация сложнее (KNN), пока просто лосс
        if out.shape[1] == self.cfg.model.num_classes:
            preds = torch.argmax(out, dim=1)
            self.val_f1(preds, targets)
            self.log("val_f1", self.val_f1, prog_bar=True)
            
        self.log("val_loss", loss, prog_bar=True)