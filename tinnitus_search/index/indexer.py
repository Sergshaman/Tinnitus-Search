""" tinnitus_search/index/indexer.py
Модуль для индексации репозитория.
Сканирует Python-файлы, применяет чанкер и эмбеддинги,
добавляет результаты в векторное хранилище.
Поддерживает инкрементальное обновление на основе хешей файлов.
Поддерживает быструю актуализацию индекса перед поиском (ensure_fresh_index).
"""
import os
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set
from tqdm import tqdm
from filelock import FileLock
import pathspec  # Для обработки .gitignore-подобных паттернов
from ..chunkers.python_ast import ASTPythonChunker, Chunk
from ..embed.cpu import CPUEmbedder
from .chroma_store import ChromaStore

logger = logging.getLogger(__name__)


class Indexer:
    """
    Класс для индексации репозитория Python-кода.
    Поддерживает полную и инкрементальную индексацию,
    использует хеши файлов для определения изменений.
    Метод ensure_fresh_index() позволяет быстро актуализировать
    индекс перед поиском без полной переиндексации.
    """

    def __init__(self, repo_path: str, persist_directory: Optional[str] = None):
        """
        Инициализирует индексатор.
        Args:
            repo_path: Путь к репозиторию для индексации
            persist_directory: Директория для хранения индекса
        """
        self.repo_path = Path(repo_path).resolve()
        self.persist_directory = persist_directory
        self.chunker = ASTPythonChunker()
        self.embedder = CPUEmbedder()
        self.store = ChromaStore(persist_directory)
        # Путь к файлу состояния (метаданные индекса)
        self.state_file = self.repo_path / ".tinnitus_index_state.json"
        self.lock_file = self.repo_path / ".tinnitus_index.lock"

        # 1. Базовые паттерны, которые игнорируются всегда
        self.ignore_patterns = {
            ".git", "__pycache__", ".venv", "venv", "env",
            ".continue", ".vscode", ".idea", "node_modules",
            ".pytest_cache", ".tox", "dist", "build", "*.egg-info"
        }

        # 2. Попытка загрузить пользовательские правила из .continueignore
        ignore_file = self.repo_path / ".continueignore"
        all_patterns = list(self.ignore_patterns)

        if ignore_file.exists():
            try:
                with open(ignore_file, "r", encoding="utf-8") as f:
                    custom_patterns = [
                        line.strip() for line in f
                        if line.strip() and not line.startswith("#")
                    ]
                    all_patterns.extend(custom_patterns)
                logger.info(f"Загружены правила игнорирования из {ignore_file}")
            except Exception as e:
                logger.error(f"Ошибка при чтении .continueignore: {e}")

        # 3. Создаем объект PathSpec для эффективной проверки путей
        self.spec = pathspec.PathSpec.from_lines(
            pathspec.patterns.GitWildMatchPattern,
            all_patterns
        )

    def _should_ignore_path(self, path: Path) -> bool:
        """
        Проверяет, следует ли игнорировать указанный путь, используя pathspec.
        Args:
            path: Путь для проверки
        Returns:
            True, если путь следует игнорировать
        """
        try:
            # Вычисляем относительный путь от корня репозитория
            rel_path = str(path.relative_to(self.repo_path))
            # pathspec требует использования '/' даже в Windows для соответствия паттернам
            rel_path_posix = rel_path.replace('\\', '/')
            return self.spec.match_file(rel_path_posix)
        except ValueError:
            # Если путь вне репозитория, на всякий случай игнорируем
            return True

    def _get_file_hash(self, file_path: Path) -> str:
        """
        Вычисляет SHA256 хеш содержимого файла.
        Args:
            file_path: Путь к файлу
        Returns:
            Хеш файла в hex формате
        """
        with open(file_path, 'rb') as f:
            content = f.read()
        return hashlib.sha256(content).hexdigest()

    def _load_index_state(self) -> Dict[str, str]:
        """
        Загружает состояние индекса из файла.
        Returns:
            Словарь с хешами файлов {file_path: hash}
        """
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                logger.warning(f"Не удалось загрузить состояние индекса из {self.state_file}")
        return {}

    def _save_index_state(self, state: Dict[str, str]) -> None:
        """
        Сохраняет состояние индекса в файл.
        Args:
            state: Словарь с хешами файлов {file_path: hash}
        """
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def _get_python_files(self) -> List[Path]:
        """
        Находит все Python-файлы в репозитории с учётом правил игнорирования.
        Returns:
            Список путей к Python-файлам
        """
        python_files = []
        for py_file in self.repo_path.rglob("*.py"):
            if not self._should_ignore_path(py_file):
                python_files.append(py_file)
        return python_files

    def _process_file(self, file_path: Path) -> List[Chunk]:
        """
        Обрабатывает один файл: чанкер -> эмбеддинги.
        Args:
            file_path: Путь к файлу для обработки
        Returns:
            Список чанков с эмбеддингами
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            chunks = self.chunker.chunk(content)
            file_hash = self._get_file_hash(file_path)
            for chunk in chunks:
                chunk.content_hash = file_hash
            return chunks
        except Exception as e:
            logger.error(f"Ошибка при обработке файла {file_path}: {e}")
            return []

    def _index_single_file(self, file_path: Path) -> bool:
        """
        Индексирует один файл.
        Args:
            file_path: Путь к файлу для индексации
        Returns:
            True, если файл был успешно проиндексирован
        """
        try:
            chunks = self._process_file(file_path)
            if chunks:
                self.store.delete_chunks_by_file(str(file_path))
                self.store.add_chunks(chunks, str(file_path))
                return True
            return False
        except Exception as e:
            logger.error(f"Ошибка при индексации файла {file_path}: {e}")
            return False

    # ────────────────────────────────────────────────────────────────
    #  Стратегия 1: Lazy Reindex — быстрая актуализация перед поиском
    # ────────────────────────────────────────────────────────────────
    def ensure_fresh_index(self) -> int:
        """
        Быстрая проверка и переиндексация только изменённых файлов.
        Предназначена для вызова **перед поиском**. Сравнивает хеши файлов
        с сохранённым состоянием индекса и переиндексирует только те,
        которые реально изменились, были добавлены или удалены.
        В отличие от incremental_index() **не** использует FileLock
        с timeout=0, поэтому не падает при конкурентных вызовах —
        просто пропускает, если блокировка занята.
        Returns:
            Количество файлов, которые были обновлены в индексе.
        """
        lock = FileLock(str(self.lock_file), timeout=1)
        try:
            lock.acquire()
        except Exception:
            # Блокировка занята — кто-то уже индексирует, не блокируем поиск
            logger.debug("ensure_fresh_index: блокировка занята, пропускаем")
            return 0

        try:
            python_files = self._get_python_files()
            current_state = self._load_index_state()
            new_state = dict(current_state)  # копия, чтобы не потерять данные
            updated = 0

            # --- Проверяем изменённые и новые файлы ---
            for file_path in python_files:
                rel_path = str(file_path.relative_to(self.repo_path))
                try:
                    file_hash = self._get_file_hash(file_path)
                except OSError as e:
                    logger.warning(f"Не удалось прочитать {file_path}: {e}")
                    continue

                # Файл не изменился — пропускаем
                if rel_path in current_state and current_state[rel_path] == file_hash:
                    new_state[rel_path] = file_hash
                    continue

                # Файл изменился или новый — переиндексируем
                if self._index_single_file(file_path):
                    logger.info(f"ensure_fresh_index: переиндексирован {rel_path}")
                    updated += 1
                    new_state[rel_path] = file_hash

            # --- Проверяем удалённые файлы ---
            existing_rel_paths = {
                str(pf.relative_to(self.repo_path)) for pf in python_files
            }
            deleted_files = set(current_state.keys()) - existing_rel_paths
            for deleted_rel in deleted_files:
                full_path = self.repo_path / deleted_rel
                self.store.delete_chunks_by_file(str(full_path))
                logger.info(f"ensure_fresh_index: удалён из индекса {deleted_rel}")
                if deleted_rel in new_state:
                    del new_state[deleted_rel]
                updated += 1

            # --- Сохраняем обновлённое состояние ---
            if updated > 0:
                self._save_index_state(new_state)
                logger.info(
                    f"ensure_fresh_index: обновлено {updated} файлов, "
                    f"всего чанков в базе: {self.store.count_chunks()}"
                )
            return updated
        finally:
            lock.release()

    # ────────────────────────────────────────────────────────────────
    #  Полная индексация
    # ────────────────────────────────────────────────────────────────
    def full_index(self, force: bool = False) -> None:
        """
        Выполняет полную индексацию репозитория.
        Args:
            force: Принудительно запустить индексацию даже при наличии блокировки
        """
        with FileLock(str(self.lock_file), timeout=0) if not force else open(os.devnull, 'w'):
            logger.info(f"Начинаем полную индексацию репозитория: {self.repo_path}")
            python_files = self._get_python_files()
            logger.info(f"Найдено {len(python_files)} Python-файлов для индексации")
            current_state = self._load_index_state()
            new_state = {}
            processed_count = 0

            for file_path in tqdm(python_files, desc="Индексация файлов", unit="файл"):
                rel_path = str(file_path.relative_to(self.repo_path))
                file_hash = self._get_file_hash(file_path)
                new_state[rel_path] = file_hash

                if rel_path in current_state and current_state[rel_path] == file_hash:
                    continue

                if self._index_single_file(file_path):
                    processed_count += 1

            self._save_index_state(new_state)
            logger.info(f"Полная индексация завершена. Обработано {processed_count} файлов.")
            logger.info(f"Всего чанков в индексе: {self.store.count_chunks()}")

    # ────────────────────────────────────────────────────────────────
    #  Инкрементальная индексация
    # ────────────────────────────────────────────────────────────────
    def incremental_index(self) -> None:
        """
        Выполняет инкрементальную индексацию репозитория.
        Индексирует только измененные, добавленные или удаленные файлы.
        """
        with FileLock(str(self.lock_file), timeout=0):
            logger.info(f"Начинаем инкрементальную индексацию репозитория: {self.repo_path}")
            python_files = self._get_python_files()
            current_state = self._load_index_state()
            new_state = {}
            existing_rel_paths = {str(pf.relative_to(self.repo_path)) for pf in python_files}
            deleted_files = set(current_state.keys()) - existing_rel_paths

            for deleted_file in deleted_files:
                full_path = self.repo_path / deleted_file
                self.store.delete_chunks_by_file(str(full_path))
                logger.debug(f"Удалены чанки для файла {deleted_file} (файл удалён)")

            processed_count = 0
            for file_path in tqdm(python_files, desc="Проверка изменений", unit="файл"):
                rel_path = str(file_path.relative_to(self.repo_path))
                file_hash = self._get_file_hash(file_path)
                new_state[rel_path] = file_hash

                if rel_path in current_state:
                    if current_state[rel_path] == file_hash:
                        continue
                    else:
                        logger.debug(f"Файл {rel_path} изменился, будет переиндексирован")
                else:
                    logger.debug(f"Новый файл {rel_path}, будет индексирован")

                if self._index_single_file(file_path):
                    processed_count += 1

            self._save_index_state(new_state)
            logger.info(f"Инкрементальная индексация завершена. Обработано {processed_count} файлов.")
            logger.info(f"Всего чанков в индексе: {self.store.count_chunks()}")

    # ────────────────────────────────────────────────────────────────
    #  Стратегия 2: Индексация / удаление одного файла
    # ────────────────────────────────────────────────────────────────
    def index_file(self, file_path: str) -> None:
        """
        Индексирует один указанный файл.
        Удобно вызывать из CLI или хука агента сразу после изменения файла:
        tinnitus-search reindex-file ./path/to/changed.py
        Args:
            file_path: Путь к файлу для индексации
        """
        file_path_obj = Path(file_path).resolve()
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        if not file_path_obj.suffix.lower() == '.py':
            raise ValueError(f"Файл должен быть Python-файлом: {file_path}")

        logger.info(f"Индексация файла: {file_path_obj}")
        lock = FileLock(str(self.lock_file), timeout=5)
        with lock:
            if self._index_single_file(file_path_obj):
                # Обновляем состояние
                try:
                    rel_path = str(file_path_obj.relative_to(self.repo_path))
                except ValueError:
                    # Файл вне repo_path — используем абсолютный путь
                    rel_path = str(file_path_obj)

                current_state = self._load_index_state()
                current_state[rel_path] = self._get_file_hash(file_path_obj)
                self._save_index_state(current_state)
                logger.info(f"Файл {file_path_obj} успешно проиндексирован")
            else:
                logger.warning(f"Не удалось проиндексировать файл {file_path_obj} (нет чанков)")

    def remove_file(self, file_path: str) -> None:
        """
        Удаляет файл из индекса.
        Args:
            file_path: Путь к файлу для удаления из индекса
        """
        logger.info(f"Удаление файла из индекса: {file_path}")
        lock = FileLock(str(self.lock_file), timeout=5)
        with lock:
            self.store.delete_chunks_by_file(file_path)
            try:
                current_state = self._load_index_state()
                rel_path = str(Path(file_path).relative_to(self.repo_path))
                if rel_path in current_state:
                    del current_state[rel_path]
                self._save_index_state(current_state)
                logger.info(f"Файл {file_path} успешно удалён из индекса")
            except Exception as e:
                logger.error(f"Ошибка при удалении файла из состояния индекса: {e}")