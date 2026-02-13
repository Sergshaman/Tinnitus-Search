from huggingface_hub import snapshot_download
from pathlib import Path
import os

# Путь относительно корня проекта
MODEL_DIR = Path(__file__).parent / "models" / "all-MiniLM-L6-v2"
REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"

print(f"Загрузка модели {REPO_ID} в {MODEL_DIR}...")

try:
    # Создаём родительскую директорию, если её нет
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    
    # Загружаем модель
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=MODEL_DIR,
        # Убраны устаревшие параметры: resume_download, local_dir_use_symlinks
    )
    
    # Проверяем наличие ключевых файлов
    if (MODEL_DIR / "config.json").exists() and (MODEL_DIR / "pytorch_model.bin").exists():
        print("✅ Модель успешно загружена!")
        print(f"📁 Путь к модели: {MODEL_DIR}")
        print(f"📦 Размер: {sum(f.stat().st_size for f in MODEL_DIR.rglob('*') if f.is_file()) / 1024 / 1024:.2f} MB")
    else:
        print("⚠️  Модель загружена, но отсутствуют ключевые файлы. Проверьте содержимое папки.")
        
except Exception as e:
    print(f"❌ Ошибка загрузки: {type(e).__name__}: {e}")
    print(f"\n💡 Возможные причины:")
    print(f"   • Нет интернет-соединения")
    print(f"   • Брандмауэр/антивирус блокирует доступ к huggingface.co")
    print(f"   • Нет прав на запись в D:\\tinnitus_db\\")
    print(f"\n🔧 Рекомендации:")
    print(f"   1. Проверьте подключение к интернету")
    print(f"   2. Запустите PowerShell от имени администратора")
    print(f"   3. Попробуйте альтернативный путь: {Path.cwd() / 'models' / 'all-MiniLM-L6-v2'}")