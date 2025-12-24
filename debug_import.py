import sys
import os

print(f"Python executable: {sys.executable}")
print(f"Current working dir: {os.getcwd()}")

try:
    print("Попытка импорта AudioBackbone...")
    from src.modules.models.audio_backbone import AudioBackbone
    print("✅ УСПЕХ! Класс найден и импортирован.")
except ImportError as e:
    print("\n❌ ОШИБКА ИМПОРТА:")
    print(e)
    print("\nСкорее всего, не установлена библиотека, которая используется внутри audio_backbone.py")
except Exception as e:
    print(f"\n❌ ДРУГАЯ ОШИБКА: {e}")