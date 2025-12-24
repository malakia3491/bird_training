import os
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import OmegaConf

class UniversalReporter(pl.Callback):
    def __init__(self, cfg, output_dir):
        self.cfg = cfg
        self.output_dir = output_dir
        self.report_path = os.path.join(output_dir, "REPORT.md")

    def on_train_end(self, trainer, pl_module):
        # 1. Рисуем график лоссов (универсально)
        self._plot_metrics(trainer)
        
        # 2. Генерируем отчет
        task_name = self.cfg.task.get("name", "unknown")
        
        md = f"# 📊 Experiment Report: {self.cfg.experiment_name}\n\n"
        md += f"**Run:** `{self.cfg.run_name}`\n"
        md += f"**Task:** `{task_name}`\n"
        md += f"**Model:** `{self.cfg.model.name}`\n\n"
        
        # Достаем метрики из callback_metrics
        metrics = trainer.callback_metrics
        final_loss = metrics.get("val_loss", metrics.get("train_loss", 0.0))
        final_f1 = metrics.get("val_f1", "N/A")
        
        md += "## Results\n"
        md += f"| Metric | Value |\n|---|---|\n"
        md += f"| Final Loss | {final_loss:.4f} |\n"
        md += f"| Final F1 | {final_f1} |\n\n"
        
        md += "## Training Dynamics\n![Loss Curve](loss_curve.png)\n\n"
        
        md += "## Config\n<details><summary>YAML</summary>\n\n```yaml\n"
        md += OmegaConf.to_yaml(self.cfg)
        md += "\n```\n</details>"
        
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"📝 Report saved: {self.report_path}")

    def _plot_metrics(self, trainer):
        # Простая реализация: берем из CSVLogger, если он есть
        log_dir = trainer.logger.log_dir
        metrics_path = os.path.join(log_dir, "metrics.csv")
        
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            # Группируем по эпохам
            epoch_data = df.groupby("epoch").mean(numeric_only=True)
            
            plt.figure(figsize=(10, 6))
            for col in epoch_data.columns:
                if "loss" in col and "step" not in col:
                    plt.plot(epoch_data.index, epoch_data[col], label=col, marker='o')
            
            plt.title("Training Loss")
            plt.xlabel("Epoch")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, "loss_curve.png"))
            plt.close()