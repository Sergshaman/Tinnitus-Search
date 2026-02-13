"""
Модуль интеграции с Continue.dev.

Предоставляет инструмент для семантического поиска по кодовой базе
с механизмами отказоустойчивости, автоматическим fallback на встроенный
поиск и lazy-актуализацией индекса перед каждым запросом.
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# === Логирование ===
LOG_FILE = Path.home() / ".continue" / "tinnitus_search.log"
LOG_FILE.parent.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Попытка импорта базового класса из Continue
try:
    from continuedev.core.config import CustomTool
    CONTINUE_AVAILABLE = True
except ImportError:
    class CustomTool:
        def __init__(self, sdk: Any):
            self.sdk = sdk
    CONTINUE_AVAILABLE = False
    logger.warning("Continue SDK not available — running in standalone mode")


class HybridSearchTool(CustomTool):
    """
    Семантический поиск по коду через ChromaDB
    с автоматическим fallback на стандартный поиск Continue.
    
    Перед каждым запросом вызывается ensure_fresh_index(),
    чтобы результаты всегда соответствовали текущему состоянию файлов.
    """

    name = "search_repo"
    description = (
        "Семантический поиск по коду Tinnitus. "
        "Использует ChromaDB с откатом на стандартный поиск."
    )

    def __init__(self, sdk: Any):
        super().__init__(sdk)
        self.sdk = sdk
        self.repo_root = Path.cwd()

        # TTL-кеширование состояния индекса (5 секунд)
        self._index_healthy: Optional[bool] = None
        self._last_health_check: float = 0
        self._health_check_ttl: float = 5.0

        self._store = None
        self._store_imported = False
        self._indexer = None
        self._indexer_imported = False

    # ── Lazy-импорт ChromaStore ──

    def _import_chroma_store(self) -> bool:
        if self._store_imported:
            return self._store is not None

        try:
            from tinnitus_search.index.chroma_store import ChromaStore
            self._store = ChromaStore()
            self._store_imported = True
            return True
        except Exception as e:
            logger.warning(f"Не удалось импортировать ChromaStore: {e}")
            self._store_imported = True
            return False

    # ── Lazy-импорт Indexer ──

    def _import_indexer(self) -> bool:
        if self._indexer_imported:
            return self._indexer is not None

        try:
            from tinnitus_search.index.indexer import Indexer
            self._indexer = Indexer(str(self.repo_root))
            self._indexer_imported = True
            return True
        except Exception as e:
            logger.warning(f"Не удалось импортировать Indexer: {e}")
            self._indexer_imported = True
            return False

    # ── Проверка здоровья индекса ──

    def _is_index_healthy(self) -> bool:
        now = time.time()
        if (now - self._last_health_check < self._health_check_ttl
                and self._index_healthy is not None):
            return self._index_healthy

        if not self._import_chroma_store():
            self._index_healthy = False
        else:
            try:
                self._index_healthy = self._store.count_chunks() > 0
            except Exception:
                self._index_healthy = False

        self._last_health_check = now
        return self._index_healthy

    # ── Lazy Reindex (Стратегия 1) ──

    def _ensure_fresh(self) -> int:
        """
        Вызывает ensure_fresh_index() из Indexer.
        Возвращает количество обновлённых файлов или 0 при ошибке.
        """
        if not self._import_indexer():
            return 0

        try:
            updated = self._indexer.ensure_fresh_index()
            if updated > 0:
                logger.info(f"Tool auto-reindex: обновлено {updated} файлов")
                # Сбрасываем кеш здоровья, чтобы пересчитать после обновления
                self._index_healthy = None
                self._last_health_check = 0
            return updated
        except Exception as e:
            logger.warning(f"Tool auto-reindex failed: {e}")
            return 0

    # ── Форматирование результатов ──

    def _format_results(self, results: Dict[str, Any]) -> str:
        if not results.get('documents') or not results['documents'][0]:
            return ""

        documents = results['documents'][0]
        metadatas = (results['metadatas'][0]
                     if results.get('metadatas')
                     else [{}] * len(documents))

        parts = []
        for doc, meta in zip(documents, metadatas):
            path = str(meta.get('file_path', 'unknown')).replace('\\', '/')
            start = meta.get('start_line', 1)
            parts.append(f"### {path}:{start}\n```python\n{doc.strip()}\n```")

        return "\n\n".join(parts)

    # ── Основной метод поиска ──

    async def run(self, query: str, max_results: int = 6, **kwargs) -> str:
        """
        Выполняет поиск:
          1. Актуализирует индекс (ensure_fresh_index)
          2. Ищет через ChromaDB
          3. При ошибке — fallback на sdk.search_code()
        """
        try:
            # ── Шаг 1: актуализация индекса ──
            self._ensure_fresh()

            # ── Шаг 2: поиск через ChromaDB ──
            if self._is_index_healthy():
                try:
                    results = self._store.search(query, n_results=max_results)
                    formatted = self._format_results(results)
                    if formatted:
                        logger.info("Успешный поиск через ChromaDB")
                        return formatted
                except Exception as e:
                    logger.warning(f"Сбой ChromaDB: {e}")

            # ── Шаг 3: fallback на Continue SDK ──
            logger.info("Используется fallback поиск (sdk.search_code)")
            return await self.sdk.search_code(query)

        except Exception as e:
            logger.error(f"Полный отказ поиска: {e}")
            return f"Ошибка поиска. Проверьте лог: {LOG_FILE}"


def create_hybrid_search_tool(sdk: Any) -> HybridSearchTool:
    """Фабричная функция для создания инструмента поиска."""
    return HybridSearchTool(sdk)