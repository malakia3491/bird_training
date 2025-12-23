import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_metrics(csv_path, output_path=None):
    if not os.path.exists(csv_path):
        print(f"❌ Файл не найден: {csv_path}")
        return

    print(f"📂 Чтение: {csv_path}")
    df = pd.read_csv(csv_path)

    # 1. Агрегация по эпохам
    # PyTorch Lightning пишет логи по шагам. 
    # Чтобы получить график "по эпохам", нужно сгруппировать и усреднить.
    # Это схлопнет все шаги внутри одной эпохи в одно число.
    epoch_data = df.groupby("epoch").mean(numeric_only=True)

    # 2. Поиск колонок с лоссом
    loss_cols = [c for c in epoch_data.columns if 'loss' in c and c != 'step']
    
    if not loss_cols:
        print("⚠️ В файле не найдены колонки с 'loss'.")
        return

    print(f"🔎 Найдены метрики: {loss_cols}")

    # 3. Рисуем (Стиль 1-в-1 как в ExperimentReporter)
    plt.figure(figsize=(10, 6))

    for col in loss_cols:
        # Берем данные по эпохам
        series = epoch_data[col].dropna()
        
        if len(series) == 0: continue

        # Рисуем линию с маркерами
        plt.plot(series.index, series.values, label=col, marker='o')

    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    if output_path:
        plt.savefig(output_path)
        print(f"✅ График сохранен: {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=str, help="Путь к metrics.csv")
    parser.add_argument("--output", type=str, default=None, help="Куда сохранить (png)")
    
    args = parser.parse_args()
    
    plot_metrics(args.csv_file, args.output)