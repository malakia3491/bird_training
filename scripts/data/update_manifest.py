import pandas as pd
from config import MANIFEST_PATH, MIN_SNR_DB

SILENCE_CLASS = 'silence'


def main():
    print("=" * 60)
    print("UPDATE MANIFEST")
    print("=" * 60)

    df = pd.read_csv(MANIFEST_PATH)
    print(f"Загружено: {len(df)} сегментов")

    if 'original_species' not in df.columns:
        df['original_species'] = df['species']

    df['is_silence'] = df['snr'] < MIN_SNR_DB
    df.loc[df['is_silence'], 'species'] = SILENCE_CLASS

    silence_count = df['is_silence'].sum()
    bird_count = len(df) - silence_count

    print(f"\nПтицы: {bird_count}")
    print(f"Тишина: {silence_count}")
    print(f"Классов: {df['species'].nunique()}")

    df.to_csv(MANIFEST_PATH, index=False)
    print(f"\nСохранено: {MANIFEST_PATH}")


if __name__ == '__main__':
    main()
