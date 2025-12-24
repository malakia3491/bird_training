import torch
import hydra
import torchmetrics
from src.systems.base import BaseAudioSystem

class ClassificationSystem(BaseAudioSystem):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Голова (Linear)
        self.head = hydra.utils.instantiate(
            cfg.head, 
            in_features=self.embed_dim, 
            num_classes=cfg.model.num_classes
        )
        
        # Лосс
        self.loss_fn = hydra.utils.instantiate(cfg.loss)
        
        # Метрики
        nc = cfg.model.num_classes
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=nc)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=nc)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=nc, average='macro')
        self.val_cm = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=nc)

    def forward(self, x):
        # Переопределяем forward, чтобы он возвращал логиты
        feats = super().forward(x) # Backbone output
        return self.head(feats)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        loss = self.loss_fn(logits, targets)
        
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, targets)
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        loss = self.loss_fn(logits, targets)
        
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        self.val_cm(preds, targets)
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)