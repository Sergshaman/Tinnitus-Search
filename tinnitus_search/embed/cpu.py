"""
Модуль для генерации эмбеддингов с использованием CPU.
Использует sentence-transformers с моделью all-MiniLM-L6-v2
для создания эмбеддингов кода на CPU.
"""
import logging
from pathlib import Path
from typing import List, Union
from sentence_transformers import SentenceTransformer
from threading import Lock

logger = logging.getLogger(__name__)

class CPUEmbedder:
    """
    Класс для генерации эмбеддингов с использованием CPU.
    Использует модель all-MiniLM-L6-v2 через sentence-transformers.
    Реализует паттерн Singleton для повторного использования модели.
    Поддерживает загрузку локальной модели из ./models/all-MiniLM-L6-v2
    """
    _instance = None
    _model_lock = Lock()

    def __new__(cls):
        """
        Реализация паттерна Singleton для экономии памяти.
        """
        if cls._instance is None:
            with cls._model_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Инициализирует модель эмбеддингов.
        """
        # Предотвращаем повторную инициализацию экземпляра
        if not hasattr(self, '_initialized'):
            self.model_name = "all-MiniLM-L6-v2"
            self.model = None
            self._load_model()
            self._initialized = True

    def _load_model(self):
        """
        Загружает модель эмбеддингов из локальной директории или с Hugging Face Hub.
        Сначала проверяет наличие модели в ./models/all-MiniLM-L6-v2 относительно корня проекта.
        """
        # Определяем путь к локальной модели (относительно корня проекта)
        # Ищем в: <корень_проекта>/models/all-MiniLM-L6-v2
        local_model_path = Path(__file__).parent.parent.parent / "models" / "all-MiniLM-L6-v2"
        
        # Проверяем наличие ключевых файлов модели
        if local_model_path.exists() and (local_model_path / "config.json").exists():
            logger.info(f"✅ Загрузка локальной модели из: {local_model_path.resolve()}")
            model_source = str(local_model_path.resolve())
        else:
            logger.info(f"⚠️  Локальная модель не найдена в {local_model_path}. Используется загрузка с Hugging Face Hub.")
            model_source = self.model_name
        
        try:
            logger.info(f"Загрузка модели эмбеддингов из: {model_source}")
            self.model = SentenceTransformer(model_source)
            logger.info(f"✅ Модель успешно загружена. Размерность эмбеддингов: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"❌ Ошибка при загрузке модели из {model_source}: {type(e).__name__}: {e}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """
        Генерирует эмбеддинг для одного текста.
        Args:
            text: Входной текст (чаще всего чанк кода)
        Returns:
            Эмбеддинг в виде списка float значений
        """
        if self.model is None:
            raise RuntimeError("Модель не была загружена")
        embedding = self.model.encode([text])[0].tolist()
        return embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Генерирует эмбеддинги для списка текстов.
        Args:
            texts: Список входных текстов (чанков кода)
        Returns:
            Список эмбеддингов, каждый из которых является списком float значений
        """
        if self.model is None:
            raise RuntimeError("Модель не была загружена")
        embeddings = self.model.encode(texts).tolist()
        return embeddings

    def get_embedding_dimension(self) -> int:
        """
        Возвращает размерность эмбеддингов.
        Returns:
            Размерность эмбеддингов (для all-MiniLM-L6-v2 это 384)
        """
        if self.model is None:
            raise RuntimeError("Модель не была загружена")
        return self.model.get_sentence_embedding_dimension()

# Пример использования
if __name__ == "__main__":
    # Создаем эмбеддер
    embedder = CPUEmbedder()
    # Тестовые данные
    test_texts = [
        "def hello_world(): print('Hello, World!')",
        "class Database: def connect(self): pass",
        "async def fetch_data(url): return await http.get(url)"
    ]
    # Генерируем эмбеддинги
    embeddings = embedder.embed_texts(test_texts)
    print(f"Сгенерировано {len(embeddings)} эмбеддингов")
    print(f"Размерность каждого эмбеддинга: {len(embeddings[0])}")
    print(f"Пример эмбеддинга для '{test_texts[0]}':")
    print(embeddings[0][:10], "...")  # Показываем первые 10 элементов