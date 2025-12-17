import os
import pytorch_lightning as pl
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from omegaconf import OmegaConf
import torch

class ExperimentReporter(pl.Callback):
    def __init__(self, cfg, output_dir):
        self.cfg = cfg
        self.output_dir = output_dir
        self.report_path = os.path.join(output_dir, "REPORT.md")
        
        # Храним историю, чтобы брать финальные значения отсюда
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
        
        # Функция для безопасного извлечения (если метрики нет, вернет None)
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
        # Фильтруем None значения (например, если val_loss еще не посчитан)
        valid_train = [(e, v) for e, v in zip(epochs, self.history["train_loss"]) if v is not None]
        valid_val = [(e, v) for e, v in zip(epochs, self.history["val_loss"]) if v is not None]
        
        if valid_train:
            plt.plot(*zip(*valid_train), label="Train Loss", marker='o')
        if valid_val:
            plt.plot(*zip(*valid_val), label="Val Loss", marker='o')
            
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
        valid_t_acc = [(e, v) for e, v in zip(epochs, self.history["train_acc"]) if v is not None]
        valid_v_acc = [(e, v) for e, v in zip(epochs, self.history["val_acc"]) if v is not None]
        valid_v_f1 = [(e, v) for e, v in zip(epochs, self.history["val_f1"]) if v is not None]

        if valid_t_acc: plt.plot(*zip(*valid_t_acc), label="Train Accuracy", linestyle='--')
        if valid_v_acc: plt.plot(*zip(*valid_v_acc), label="Val Accuracy", marker='s')
        if valid_v_f1: plt.plot(*zip(*valid_v_f1), label="Val F1 (Macro)", marker='^')
        
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
        try:
            # Тут мы вынуждены считать, так как матрицу не сохраняем в историю
            cm_tensor = pl_module.val_cm.compute()
            cm = cm_tensor.cpu().numpy()
        except Exception:
            return None

        class_names = None
        if hasattr(trainer.datamodule, 'label_encoder'):
            class_names = trainer.datamodule.label_encoder.classes_
        
        if class_names is not None and len(class_names) > 50:
            class_names = None

        plt.figure(figsize=(12, 10))
        # Нормализация
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
        cm_img = self._plot_confusion_matrix(trainer, pl_module)

        # 1. Берем основные метрики ИЗ ИСТОРИИ (это надежнее, чем compute() в конце)
        def get_last(key):
            lst = self.history.get(key, [])
            # Берем последнее не-None значение
            valid = [x for x in lst if x is not None]
            return valid[-1] if valid else 0.0

        final_val_loss = get_last("val_loss")
        final_acc = get_last("val_acc")
        final_f1 = get_last("val_f1")

        # 2. Precision и Recall мы не хранили в истории (в system.py не было self.log для них),
        # поэтому пробуем вычислить сейчас. Если модуль сброшен - будет 0.
        try:
            final_prec = pl_module.val_precision.compute().item()
            final_rec = pl_module.val_recall.compute().item()
        except:
            final_prec = 0.0
            final_rec = 0.0

        # Форматирование
        def fmt(val): return f"{val:.4f}" if isinstance(val, (int, float)) else str(val)

        config_yaml = OmegaConf.to_yaml(self.cfg)
        frontend_name = self.cfg.frontend.get('name', 'unknown')

        md_content = f"""# 📊 Отчет эксперимента: {self.cfg.project_name}

**ID:** `{os.path.basename(self.output_dir)}`  
**Дата:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Frontend:** `{frontend_name}`
**Model:** `{self.cfg.model.name}`

## 1. Основные результаты (Summary)

| Метрика | Значение (Final) |
| :--- | :--- |
| **Validation Loss** | **{fmt(final_val_loss)}** |
| **Validation F1 (Macro)** | **{fmt(final_f1)}** |
| **Validation Accuracy** | {fmt(final_acc)} |
| **Precision (Macro)** | {fmt(final_prec)} |
| **Recall (Macro)** | {fmt(final_rec)} |

## 2. Визуализация

### Матрица ошибок
![Confusion Matrix]({cm_img})

### Графики
| Loss | Metrics |
| :---: | :---: |
| ![Loss Curve]({loss_img}) | ![Metrics Curve]({metrics_img}) |

## 3. Конфигурация
<details>
<summary>Развернуть</summary>

```yaml
{config_yaml}
</details>
"""
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(md_content)
            print(f"\n📝 Report saved to: {self.report_path}")