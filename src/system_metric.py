import pytorch_lightning as pl
import torch
import hydra
from omegaconf import DictConfig, ListConfig
import torchmetrics

torch.serialization.add_safe_globals([DictConfig, ListConfig])

class MetricLearningSystem(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # 1. Frontend & Backbone
        self.frontend = hydra.utils.instantiate(cfg.frontend)
        self.backbone = hydra.utils.instantiate(cfg.model)
        
        # Определяем размерность эмбеддинга
        embed_dim = self.backbone.embed_dim if hasattr(self.backbone, 'embed_dim') else 2048
        
        # 2. Metric Head (ArcFace/CosFace)
        # Важно: ArcFace требует знать количество классов при инициализации
        self.head = hydra.utils.instantiate(
            cfg.head, 
            in_features=embed_dim, 
            num_classes=cfg.model.num_classes
        )
        
        # 3. Loss (Обычно CrossEntropy, применяемая к логитам из ArcFace)
        self.loss_fn = hydra.utils.instantiate(cfg.loss)

        # 4. Метрики
        # Мы используем те же метрики классификации, так как ArcFace решает задачу классификации,
        # но делает это через угловые отступы.
        nc = cfg.model.num_classes
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=nc)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=nc)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=nc, average='macro')
        
        # Можно добавить метрику "ConfusionMatrix" для репортера, если нужно
        self.val_cm = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=nc)

    def forward(self, x, labels=None):
        """
        Args:
            x: Входное аудио/спектрограмма
            labels: Истинные метки (нужны для ArcFace во время обучения)
        """
        spec = self.frontend(x)
        features = self.backbone(spec)
        
        # Передаем эмбеддинги и метки в голову (ArcFace)
        # Если labels=None (валидация/тест), голова вернет просто косинусное сходство
        logits = self.head(features, labels)
        
        return logits

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        
        # Важно: передаем targets в forward!
        logits = self.forward(inputs, targets)
        
        loss = self.loss_fn(logits, targets)
        
        # Считаем точность на лету (ArcFace логиты можно использовать как обычные)
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, targets)
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        
        # На валидации мы не используем Margin (штраф), поэтому labels=None.
        # Модель просто предсказывает класс на основе косинусной близости к центроидам.
        logits = self.forward(inputs, labels=None)
        
        loss = self.loss_fn(logits, targets)
        
        preds = torch.argmax(logits, dim=1)
        
        # Метрики
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        self.val_cm(preds, targets)
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def configure_optimizers(self):
        # Оптимизатор учит и Backbone, и ArcFace Head (центроиды классов)
        optimizer = hydra.utils.instantiate(self.cfg.optimizer, params=self.parameters())
        
        if "scheduler" in self.cfg:
            scheduler = hydra.utils.instantiate(self.cfg.scheduler, optimizer=optimizer)
            return [optimizer], [scheduler]
            
        return optimizer