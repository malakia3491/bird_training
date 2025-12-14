import os
import pytorch_lightning as pl
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
import torch
from sklearn.metrics import classification_report, precision_recall_fscore_support

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
        # 1. Генерируем графики
        loss_img, metrics_img = self._plot_curves()
        
        # 2. Считаем Confusion Matrix и Метрики через sklearn (самый надежный способ)
        try:
            # Получаем все предсказания с валидации
            cm_tensor = pl_module.val_cm.compute()
            # Нам нужны не сама матрица, а предсказания. 
            # Но torchmetrics хранит их внутри val_cm, но не отдает напрямую списком.
            # Поэтому проще посчитать метрики на основе накопленной матрицы (TP, FP, FN...)
            # НО! Scikit-learn требует списки y_true, y_pred.
            
            # --- ВАРИАНТ B: Берем метрики, которые уже посчитал pl_module ---
            # pl_module.val_f1 и др. уже посчитаны в конце эпохи
            final_acc = pl_module.val_acc.compute().item()
            final_f1 = pl_module.val_f1.compute().item()
            final_prec = pl_module.val_precision.compute().item()
            final_rec = pl_module.val_recall.compute().item()
            
            # Строим матрицу
            cm_img = self._plot_confusion_matrix(trainer, pl_module)
            
        except Exception as e:
            print(f"Ошибка расчета метрик: {e}")
            final_acc, final_f1, final_prec, final_rec = 0, 0, 0, 0
            cm_img = None

        final_val_loss = self.history["val_loss"][-1] if self.history["val_loss"] else "N/A"
        
        # Формируем конфиг
        config_yaml = OmegaConf.to_yaml(self.cfg)
        frontend_name = self.cfg.frontend.get('name', 'unknown')

        # MD Отчет
        md_content = f"""# 📊 Отчет эксперимента: {self.cfg.project_name}

**ID:** `{os.path.basename(self.output_dir)}`  
**Дата:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Frontend:** `{frontend_name}`
**Model:** `{self.cfg.model.name}`

## 1. Основные результаты (Summary)

| Метрика | Значение (Final) |
| :--- | :--- |
| **Validation Loss** | **{final_val_loss:.4f}** |
| **Validation F1 (Macro)** | **{final_f1:.4f}** |
| **Validation Accuracy** | {final_acc:.4f} |
| **Precision (Macro)** | {final_prec:.4f} |
| **Recall (Macro)** | {final_rec:.4f} |

## 2. Визуализация

### Матрица ошибок
![Confusion Matrix]({cm_img})

### Графики
| Loss | Metrics |
| :---: | :---: |
| ![Loss Curve]({loss_img}) | ![Metrics Curve]({metrics_img}) |

## 3. Конфигурация
<details><summary>Развернуть</summary>

```yaml
{config_yaml}
/details>
"""
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(md_content)
            print(f"\n📝 Report saved to: {self.report_path}")