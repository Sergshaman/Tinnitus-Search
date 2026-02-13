"""
Модуль для работы с векторным хранилищем ChromaDB.
Обеспечивает семантический поиск по индексированным чанкам кода.
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

import chromadb
from chromadb.config import Settings
from chromadb.api.types import QueryResult

# Импорт класса чанка (убедись, что путь соответствует структуре проекта)
try:
    from ..chunkers.python_ast import Chunk
except ImportError:
    # Заглушка для предотвращения падения при тестах
    from dataclasses import dataclass
    @dataclass
    class Chunk:
        text: str
        start_line: int
        end_line: int
        content_hash: str
        metadata: Dict[str, Any]

logger = logging.getLogger(__name__)

class ChromaStore:
    """
    Класс для взаимодействия с ChromaDB.
    Использует локальное хранилище в директории пользователя .continue/chroma/
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        """
        Инициализирует хранилище. По умолчанию использует ~/.continue/chroma/tinnitus-search/
        """
        if persist_directory is None:
            # Абсолютный путь для надежности в Windows
            self.persist_directory = str(Path.home() / ".continue" / "chroma" / "tinnitus-search")
        else:
            self.persist_directory = persist_directory
        
        # Убедимся, что путь - абсолютный
        self.persist_directory = str(Path(self.persist_directory).resolve())
        
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Настройка клиента
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        
        # Создаем или получаем коллекцию
        self.collection = self.client.get_or_create_collection(
            name="tinnitus_repo_smart_v1",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"ChromaStore инициализирован в: {self.persist_directory}")

    # --- Методы для интеграции с Continue Tool ---

    def is_available(self) -> bool:
        """Проверка доступности клиента."""
        return self.client is not None

    def is_compatible(self) -> bool:
        """Проверка совместимости схемы (заглушка)."""
        return True

    def count_chunks(self) -> int:
        """Возвращает общее количество чанков в базе."""
        return self.collection.count()

    # --- Основная логика работы с данными ---

    def _generate_chunk_id(self, file_path: str, start_line: int, end_line: int, content_preview: str = "") -> str:
        """Генерирует стабильный ID для чанка.
        
        Формат ID: hash(file_path + start_line + content_preview)[:16]_{start_line}_{end_line}
        Это гарантирует уникальность даже для одинаковых структур кода в разных файлах.
        """
        # Нормализуем путь для кросс-платформенной совместимости
        normalized_path = Path(file_path).as_posix()
        # Включаем путь к файлу, номер начальной строки и предварительный просмотр содержимого
        content = f"{normalized_path}:{start_line}:{content_preview[:100]}".encode('utf-8')
        file_hash = hashlib.sha256(content).hexdigest()[:16]
        return f"{file_hash}_{start_line}_{end_line}"

    def add_chunks(self, chunks: List[Chunk], file_path: str) -> None:
        """Добавляет чанки в коллекцию с предварительной фильтрацией дубликатов."""
        if not chunks:
            return
        
        # Получаем существующие ID из коллекции для проверки
        existing_ids = set()
        try:
            # Получаем все ID из коллекции (ограничиваем размер выборки для производительности)
            all_docs = self.collection.get(limit=100000)
            if all_docs and "ids" in all_docs:
                existing_ids = set(all_docs["ids"])
        except Exception as e:
            logger.debug(f"Не удалось получить существующие ID: {e}")
        
        ids, documents, metadatas = [], [], []
        seen_ids = set()  # Для отслеживания дубликатов внутри текущего batch
        duplicates_count = 0
        
        for chunk in chunks:
            # Создаем краткий предварительный просмотр содержимого для уникальности ID
            content_preview = chunk.text.strip().split('\n')[0]  # Берем первую строку
            chunk_id = self._generate_chunk_id(file_path, chunk.start_line, chunk.end_line, content_preview)
            
            # Проверяем на дубликаты внутри текущего batch
            if chunk_id in seen_ids:
                logger.debug(f"Пропущен дубликат ID внутри batch: {chunk_id}")
                duplicates_count += 1
                continue
            
            # Проверяем на дубликаты с существующими ID в базе
            if chunk_id in existing_ids:
                logger.debug(f"Пропущен существующий ID в базе: {chunk_id}")
                duplicates_count += 1
                continue
            
            seen_ids.add(chunk_id)
            ids.append(chunk_id)
            documents.append(chunk.text)
            
            meta = chunk.metadata.copy()
            meta["file_path"] = Path(file_path).as_posix() # Windows-safe
            meta["start_line"] = chunk.start_line
            meta["end_line"] = chunk.end_line
            meta["content_hash"] = chunk.content_hash
            metadatas.append(meta)
        
        if duplicates_count > 0:
            logger.info(f"Пропущено {duplicates_count} дубликатов при добавлении чанков из {file_path}")
        
        if not ids:
            logger.debug(f"Нет новых чанков для добавления из {file_path}")
            return
        
        # Добавляем чанки в коллекцию
        try:
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
        except ValueError as e:
            if "Expected IDs to be unique" in str(e):
                # Запасной вариант: добавляем по одному при непредвиденных дубликатах
                logger.warning(f"Обнаружены непредвиденные дубликаты ID при добавлении чанков из {file_path}: {e}")
                for i, chunk_id in enumerate(ids):
                    try:
                        self.collection.add(ids=[chunk_id], documents=[documents[i]], metadatas=[metadatas[i]])
                    except ValueError as sub_e:
                        if "Expected IDs to be unique" in str(sub_e):
                            logger.warning(f"Пропущен дублирующийся ID: {chunk_id}")
                        else:
                            raise sub_e
            else:
                raise e

    def update_chunks(self, chunks: List[Chunk], file_path: str) -> None:
        """Обновляет чанки (upsert)."""
        if not chunks:
            return
            
        ids, documents, metadatas = [], [], []
        for chunk in chunks:
            # Создаем краткий предварительный просмотр содержимого для уникальности ID
            content_preview = chunk.text.strip().split('\n')[0]  # Берем первую строку
            chunk_id = self._generate_chunk_id(file_path, chunk.start_line, chunk.end_line, content_preview)
            ids.append(chunk_id)
            documents.append(chunk.text)
            
            meta = chunk.metadata.copy()
            meta["file_path"] = Path(file_path).as_posix()
            meta["start_line"] = chunk.start_line
            meta["end_line"] = chunk.end_line
            meta["content_hash"] = chunk.content_hash
            metadatas.append(meta)
        
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def delete_chunks_by_file(self, file_path: str) -> None:
        """Удаляет все записи, относящиеся к конкретному файлу."""
        path_str = Path(file_path).as_posix()
        results = self.collection.get(where={"file_path": path_str})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])

    def search(self, query: str, n_results: int = 10) -> QueryResult:
        """Семантический поиск."""
        return self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

    def get_all_files(self) -> List[str]:
        """Список всех проиндексированных файлов."""
        all_docs = self.collection.get(include=["metadatas"])
        return list({m["file_path"] for m in all_docs["metadatas"] if "file_path" in m})