import pytorch_lightning as pl
import torch
import hydra
from omegaconf import DictConfig
import torchmetrics

class BirdClassifier(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # 1. Модули
        self.frontend = hydra.utils.instantiate(cfg.frontend)
        self.backbone = hydra.utils.instantiate(cfg.model)
        
        if cfg.model.get("freeze_backbone", False):
            print("❄️ FREEZING BACKBONE (Linear Probing Mode)")
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval() 
        
        # Определяем размерность эмбеддинга (если бэкбон уже инициализирован)
        # Если нет - hydra сама создаст объект
        embed_dim = self.backbone.embed_dim if hasattr(self.backbone, 'embed_dim') else 1280
        
        self.head = hydra.utils.instantiate(
            cfg.head, 
            in_features=embed_dim, 
            num_classes=cfg.model.num_classes
        )
        self.loss_fn = hydra.utils.instantiate(cfg.loss)

        # 2. Метрики
        num_classes = cfg.model.num_classes
        # Важно: task="multiclass"
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')
        
        # Матрица ошибок (не логируем каждый шаг, считаем в конце)
        self.val_cm = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        spec = self.frontend(x)
        features = self.backbone(spec)
        logits = self.head(features)
        return logits

    def on_train_start(self):
        if self.cfg.model.get("freeze_backbone", False):
            self.backbone.eval()

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.loss_fn(outputs, targets)
        
        preds = torch.argmax(outputs, dim=1)
        self.train_acc(preds, targets)
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.loss_fn(outputs, targets)
        
        preds = torch.argmax(outputs, dim=1)
        
        # Обновляем метрики
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        self.val_precision(preds, targets)
        self.val_recall(preds, targets)
        
        # Накапливаем данные для матрицы ошибок
        self.val_cm.update(preds, targets)
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        # Очищаем матрицу в конце эпохи (хотя torchmetrics делает это сам, для надежности)
        pass

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.cfg.optimizer, params=self.parameters())
        if "scheduler" in self.cfg and self.cfg.scheduler:
            scheduler = hydra.utils.instantiate(self.cfg.scheduler, optimizer=optimizer)
            return [optimizer], [scheduler]
        return optimizer