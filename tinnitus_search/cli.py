"""
CLI интерфейс для tinnitus-search.
Предоставляет команды для индексации, поиска, переиндексации одного файла и диагностики.
"""

import typer
import sys
from pathlib import Path
from typing import Optional

# Configure UTF-8 encoding for Windows to fix Cyrillic display issues
if sys.platform == "win32":
    import io
    import os
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from .index.indexer import Indexer
from .index.chroma_store import ChromaStore

app = typer.Typer(help="CLI для поиска по коду Tinnitus Search")


# ────────────────────────────────────────────────────────────────
#  index — полная / инкрементальная индексация репозитория
# ────────────────────────────────────────────────────────────────

@app.command()
def index(
    repo_path: str = typer.Argument(..., help="Путь к репозиторию для индексации"),
    full: bool = typer.Option(False, "--full", "-f", help="Выполнить полную индексацию"),
    incremental: bool = typer.Option(
        True, "--incremental/--no-incremental",
        help="Выполнить инкрементальную индексацию"
    ),
    force: bool = typer.Option(
        False, "--force", "-F",
        help="Принудительно запустить индексацию (игнорировать блокировку)"
    ),
    db_path: str = typer.Option(
        None, "--db-path", "-d",
        help="Путь к директории для хранения базы индексов"
    ),
):
    """Индексирует Python-файлы в репозитории."""
    try:
        indexer = Indexer(repo_path, persist_directory=db_path)

        if full:
            typer.echo(f"Выполняется полная индексация репозитория: {repo_path}")
            indexer.full_index(force=force)
        else:
            typer.echo(f"Выполняется инкрементальная индексация репозитория: {repo_path}")
            indexer.incremental_index()

        typer.echo("Индексация завершена успешно!")

    except KeyboardInterrupt:
        typer.echo("\nИндексация прервана пользователем.", err=True)
        sys.exit(1)
    except Exception as e:
        typer.echo(f"Ошибка при индексации: {e}", err=True)
        sys.exit(1)


# ────────────────────────────────────────────────────────────────
#  reindex-file — переиндексация одного конкретного файла
#  (Стратегия 2: Targeted Reindex)
# ────────────────────────────────────────────────────────────────

@app.command("reindex-file")
def reindex_file(
    file_path: str = typer.Argument(..., help="Путь к изменённому Python-файлу"),
    repo_path: str = typer.Option(
        ".", "--repo", "-r",
        help="Путь к репозиторию (нужен для вычисления относительных путей)"
    ),
    db_path: str = typer.Option(
        None, "--db-path", "-d",
        help="Путь к директории для хранения базы индексов"
    ),
):
    """
    Переиндексирует один конкретный файл.

    Идеально для вызова из хука агента или скрипта сразу после
    редактирования файла:

        tinnitus-search reindex-file ./tinnitus_search/cli.py
        tinnitus-search reindex-file src/app.py --repo /projects/myapp --db-path /data/idx
    """
    try:
        resolved = Path(file_path).resolve()
        if not resolved.exists():
            typer.echo(f"[ERROR] Файл не найден: {resolved}", err=True)
            sys.exit(1)

        if resolved.suffix.lower() != ".py":
            typer.echo(f"[ERROR] Поддерживаются только .py файлы: {resolved}", err=True)
            sys.exit(1)

        indexer = Indexer(repo_path, persist_directory=db_path)
        indexer.index_file(str(resolved))
        typer.echo(f"[OK] Файл переиндексирован: {resolved}")

    except KeyboardInterrupt:
        typer.echo("\nПрервано пользователем.", err=True)
        sys.exit(1)
    except Exception as e:
        typer.echo(f"[ERROR] {e}", err=True)
        sys.exit(1)


# ────────────────────────────────────────────────────────────────
#  remove-file — удаление файла из индекса
# ────────────────────────────────────────────────────────────────

@app.command("remove-file")
def remove_file(
    file_path: str = typer.Argument(..., help="Путь к файлу для удаления из индекса"),
    repo_path: str = typer.Option(".", "--repo", "-r", help="Путь к репозиторию"),
    db_path: str = typer.Option(None, "--db-path", "-d", help="Путь к БД"),
):
    """Удаляет указанный файл из индекса (без удаления самого файла)."""
    try:
        indexer = Indexer(repo_path, persist_directory=db_path)
        indexer.remove_file(file_path)
        typer.echo(f"[OK] Файл удалён из индекса: {file_path}")
    except Exception as e:
        typer.echo(f"[ERROR] {e}", err=True)
        sys.exit(1)


# ────────────────────────────────────────────────────────────────
#  search — семантический поиск с автоматической актуализацией
#  (Стратегия 1: Lazy Reindex)
# ────────────────────────────────────────────────────────────────

@app.command()
def search(
    query: str = typer.Argument(..., help="Поисковый запрос"),
    limit: int = typer.Option(10, "--limit", "-l", help="Количество результатов"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Подробный вывод"),
    db_path: str = typer.Option(
        None, "--db-path", "-d",
        help="Путь к директории для хранения базы индексов"
    ),
    repo_path: str = typer.Option(
        ".", "--repo", "-r",
        help="Путь к репозиторию (для auto-reindex)"
    ),
    auto_reindex: bool = typer.Option(
        True, "--auto-reindex/--no-auto-reindex",
        help="Автоматически обновить индекс перед поиском"
    ),
):
    """
    Выполняет семантический поиск по индексу.

    По умолчанию перед поиском запускается быстрая проверка хешей файлов.
    Если какие-то файлы изменились — они будут переиндексированы автоматически.
    Отключить: --no-auto-reindex
    """
    try:
        # ── Стратегия 1: Lazy Reindex перед поиском ──
        if auto_reindex:
            try:
                indexer = Indexer(repo_path, persist_directory=db_path)
                updated = indexer.ensure_fresh_index()
                if updated > 0:
                    typer.echo(f"[REINDEX] Обновлено {updated} файл(ов) перед поиском")
            except Exception as e:
                # Не блокируем поиск из-за ошибки переиндексации
                typer.echo(f"[WARN] Авто-актуализация не удалась: {e}", err=True)

        store = ChromaStore(persist_directory=db_path)
        chunk_count = store.count_chunks()

        if chunk_count == 0:
            typer.echo("Индекс пуст. Сначала выполните индексацию:", err=True)
            typer.echo("  tinnitus-search index .", err=True)
            return

        typer.echo(f"Поиск: '{query}' ({chunk_count} чанков в базе)")
        results = store.search(query, n_results=min(limit, chunk_count))

        if not results.get('documents') or not results['documents'][0]:
            typer.echo("Ничего не найдено.")
            return

        documents = results['documents'][0]
        metadatas = results['metadatas'][0] if results.get('metadatas') else []
        distances = results['distances'][0] if results.get('distances') else []

        typer.echo(f"\nРезультаты (топ-{len(documents)}):")
        typer.echo("=" * 60)

        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            typer.echo(f"\n{i+1}. Файл: {meta.get('file_path', 'N/A')}")
            typer.echo(f"   Строки: {meta.get('start_line', 'N/A')}-{meta.get('end_line', 'N/A')}")
            if verbose:
                typer.echo(f"   Контекст: {meta.get('parent_context', 'N/A')}")
                typer.echo(f"   Тип: {meta.get('symbol_type', 'N/A')}")
                typer.echo(f"   Сходство: {1 - dist:.3f}")
            typer.echo(f"   Код:\n{doc}")
            typer.echo("-" * 60)

    except KeyboardInterrupt:
        typer.echo("\nПоиск прерван пользователем.", err=True)
        sys.exit(1)
    except Exception as e:
        typer.echo(f"Ошибка при поиске: {e}", err=True)
        sys.exit(1)


# ────────────────────────────────────────────────────────────────
#  doctor — диагностика
# ────────────────────────────────────────────────────────────────

@app.command()
def doctor(
    repo_path: str = typer.Option(".", "--repo", "-r", help="Путь к репозиторию"),
    db_path: str = typer.Option(None, "--db-path", "-d", help="Путь к БД"),
):
    """Диагностика состояния системы."""
    try:
        typer.echo("=== Диагностика tinnitus-search ===\n")

        repo_path_obj = Path(repo_path).resolve()
        if not repo_path_obj.exists():
            typer.echo(f"❌ Путь не существует: {repo_path}", err=True)
            return

        typer.echo(f"[PATH] Репозиторий: {repo_path_obj}")

        # Подсчёт файлов через Indexer (с учётом ignore-паттернов)
        indexer = Indexer(str(repo_path_obj), persist_directory=db_path)
        python_files = indexer._get_python_files()
        typer.echo(f"[FILE] Найдено Python-файлов (с учётом игнорирования): {len(python_files)}")

        # Проверка состояния индекса
        state = indexer._load_index_state()
        typer.echo(f"[STATE] Файлов в state-файле: {len(state)}")

        # Быстрая проверка: сколько файлов устарело
        stale_count = 0
        for pf in python_files:
            rel = str(pf.relative_to(indexer.repo_path))
            try:
                fh = indexer._get_file_hash(pf)
            except OSError:
                stale_count += 1
                continue
            if rel not in state or state[rel] != fh:
                stale_count += 1
        if stale_count > 0:
            typer.echo(f"[WARN] Устаревших / новых файлов: {stale_count}")
        else:
            typer.echo("[OK] Все файлы актуальны")

        # Проверка БД
        typer.echo("\n[STORE] Проверка векторного хранилища...")
        count = indexer.store.count_chunks()
        typer.echo(f"[CHUNK] Чанков в индексе: {count}")

        if count == 0:
            typer.echo("[WARN] Индекс пуст. Нужно запустить: tinnitus-search index .")

        # Проверка эмбеддингов
        typer.echo("\n[EMBED] Проверка эмбеддингов...")
        try:
            dim = indexer.embedder.get_embedding_dimension()
            typer.echo(f"[OK] Модель загружена, размерность: {dim}")
        except Exception as e:
            typer.echo(f"[ERROR] Ошибка модели: {e}")

        # Проверка чанкера
        typer.echo("\n[CHUNKER] Проверка чанкера...")
        if indexer.chunker:
            typer.echo("[OK] Чанкер инициализирован")

        typer.echo("\n=== Диагностика завершена ===")

    except Exception as e:
        typer.echo(f"Ошибка при диагностике: {e}", err=True)


# ────────────────────────────────────────────────────────────────
#  info
# ────────────────────────────────────────────────────────────────

@app.command()
def info():
    """Информация о системе."""
    try:
        import tinnitus_search
        version = getattr(tinnitus_search, "__version__", "0.1.0")
        typer.echo(f"[INFO] tinnitus-search версия: {version}")
        typer.echo("[INFO] Семантический поиск для Python-кодовых баз")
    except Exception:
        typer.echo("[INFO] tinnitus-search версия: dev")


if __name__ == "__main__":
    app()