import os
import pytorch_lightning as pl
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
import torch

class ExperimentReporter(pl.Callback):
    def __init__(self, cfg, output_dir):
        self.cfg = cfg
        self.output_dir = output_dir
        self.report_path = os.path.join(output_dir, "REPORT.md")
        
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_f1": []
        }

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        
        def get_val(key):
            val = metrics.get(key, None)
            return val.item() if val is not None else None

        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(get_val("train_loss"))
        self.history["val_loss"].append(get_val("val_loss"))
        self.history["train_acc"].append(get_val("train_acc"))
        self.history["val_acc"].append(get_val("val_acc"))
        self.history["val_f1"].append(get_val("val_f1"))

    def _plot_curves(self):
        epochs = self.history["epoch"]
        if not epochs: return None, None

        # 1. Loss
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.history["train_loss"], label="Train Loss", marker='o')
        plt.plot(epochs, self.history["val_loss"], label="Val Loss", marker='o')
        plt.title("Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        loss_path = os.path.join(self.output_dir, "loss_curve.png")
        plt.savefig(loss_path)
        plt.close()
        
        # 2. Metrics
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.history["train_acc"], label="Train Accuracy", linestyle='--')
        plt.plot(epochs, self.history["val_acc"], label="Val Accuracy", marker='s')
        plt.plot(epochs, self.history["val_f1"], label="Val F1 (Macro)", marker='^')
        plt.title("Metrics Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        metrics_path = os.path.join(self.output_dir, "metrics_curve.png")
        plt.savefig(metrics_path)
        plt.close()
        
        return "loss_curve.png", "metrics_curve.png"

    def _plot_confusion_matrix(self, trainer, pl_module):
        # 1. Вычисляем матрицу
        # pl_module.val_cm хранит состояние с валидации
        cm_tensor = pl_module.val_cm.compute()
        cm = cm_tensor.cpu().numpy()
        
        # 2. Достаем имена классов
        # Пытаемся добраться до LabelEncoder через DataModule
        class_names = None
        if hasattr(trainer.datamodule, 'label_encoder'):
            class_names = trainer.datamodule.label_encoder.classes_
        
        # Если классов слишком много, имена не влезут
        if class_names is not None and len(class_names) > 50:
            print("Слишком много классов для подписей, используем индексы")
            class_names = None

        # 3. Рисуем
        plt.figure(figsize=(12, 10))
        # Нормализуем по строкам (True Label), чтобы видеть проценты
        # Добавляем epsilon чтобы не делить на 0
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
        
        sns.heatmap(
            cm_normalized, 
            annot=True if (class_names is None or len(class_names) < 20) else False, 
            fmt=".2f", 
            cmap="Blues",
            xticklabels=class_names if class_names is not None else "auto",
            yticklabels=class_names if class_names is not None else "auto"
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Normalized Confusion Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        cm_path = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        
        return "confusion_matrix.png"

    def on_train_end(self, trainer, pl_module):
        loss_img, metrics_img = self._plot_curves()
        
        # Рисуем матрицу ошибок
        try:
            cm_img = self._plot_confusion_matrix(trainer, pl_module)
        except Exception as e:
            print(f"Не удалось построить Confusion Matrix: {e}")
            cm_img = None

        final_val_loss = self.history["val_loss"][-1] if self.history["val_loss"] else "N/A"
        final_val_acc = self.history["val_acc"][-1] if self.history["val_acc"] else "N/A"
        final_val_f1 = self.history["val_f1"][-1] if self.history["val_f1"] else "N/A"
        
        # Достаем метрики Precision/Recall
        metrics = trainer.callback_metrics
        final_prec = metrics.get("val_precision", torch.tensor(0.0)).item()
        final_rec = metrics.get("val_recall", torch.tensor(0.0)).item()

        config_yaml = OmegaConf.to_yaml(self.cfg)
        
        # Обработка имени фронтенда (чтобы не падало, если его нет)
        frontend_name = self.cfg.frontend.get('name', 'unknown')

        md_content = f"""# 📊 Отчет эксперимента: {self.cfg.project_name}

**ID:** `{os.path.basename(self.output_dir)}`  
**Дата:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Frontend:** `{frontend_name}`
**Model:** `{self.cfg.model.name}`

## 1. Основные результаты (Summary)

| Метрика | Значение (Final) |
| :--- | :--- |
| **Validation Loss** | **{final_val_loss:.4f}** |
| **Validation F1 (Macro)** | **{final_val_f1:.4f}** |
| **Validation Accuracy** | {final_val_acc:.4f} |
| **Precision (Macro)** | {final_prec:.4f} |
| **Recall (Macro)** | {final_rec:.4f} |

## 2. Визуализация

### Матрица ошибок (Confusion Matrix)
Показывает, какие классы путает модель.
![Confusion Matrix]({cm_img})

### Графики обучения
| Loss | Metrics |
| :---: | :---: |
| ![Loss Curve]({loss_img}) | ![Metrics Curve]({metrics_img}) |

## 3. Конфигурация запуска

<details>
<summary>🔽 Нажмите, чтобы развернуть полный конфиг</summary>

```yaml
{config_yaml}
</details>
Generated by ExperimentReporter
"""
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(md_content)
            print(f"\n📝 Report saved to: {self.report_path}")