"""
Тесты для чанкеров tinnitus-search.

Этот тест проверяет, что AST-чанкер корректно обрабатывает
вложенные классы и сохраняет информацию о родительском классе
в метаданных чанка.
"""

import pytest
from tinnitus_search.chunkers.python_ast import ASTPythonChunker, Chunk


def test_ast_chunker_nested_classes():
    """Проверяет, что чанкер правильно обрабатывает вложенные классы и сохраняет контекст."""
    code_with_nested_classes = '''
class UserManager:
    """Класс для управления пользователями."""
    
    def __init__(self):
        self.users = []
    
    def add_user(self, name):
        """Добавляет пользователя."""
        user = {"name": name, "id": len(self.users)}
        self.users.append(user)
        return user
    
    async def get_user_async(self, user_id):
        """Асинхронно получает пользователя по ID."""
        for user in self.users:
            if user["id"] == user_id:
                return user
        return None

def standalone_function():
    """Функция вне класса."""
    return "standalone"
'''

    chunker = ASTPythonChunker()
    chunks = chunker.chunk(code_with_nested_classes)
    
    # Проверяем, что чанки были созданы
    assert len(chunks) > 0, "Должны быть созданы чанки из кода"
    
    # Находим чанки, связанные с методами класса
    method_chunks = [chunk for chunk in chunks if chunk.metadata.get("symbol_type") == "method"]
    function_chunks = [chunk for chunk in chunks if chunk.metadata.get("symbol_type") == "function"]
    
    # Проверяем, что найдены методы класса
    assert len(method_chunks) > 0, "Должны быть найдены методы класса"
    
    # Проверяем, что у методов есть информация о родительском классе
    for chunk in method_chunks:
        parent_context = chunk.metadata.get("parent_context")
        assert parent_context is not None, f"У чанка метода должен быть parent_context: {chunk.metadata}"
        assert "UserManager" in parent_context, f"Parent context должен содержать имя класса: {parent_context}"
    
    # Проверяем, что у обычных функций нет родительского контекста
    for chunk in function_chunks:
        parent_context = chunk.metadata.get("parent_context")
        assert parent_context == "", f"У обычной функции не должно быть parent_context: {parent_context}"
    
    # Проверяем, что у методов есть правильные типы
    add_user_chunk = None
    for chunk in chunks:
        if chunk.metadata.get("symbol_name") == "add_user":
            add_user_chunk = chunk
            break
    
    assert add_user_chunk is not None, "Должен быть найден метод add_user"
    assert add_user_chunk.metadata["symbol_type"] == "method", "add_user должен быть помечен как метод"
    assert add_user_chunk.metadata["parent_context"] == "UserManager", "Контекст должен быть UserManager"
    

def test_ast_chunker_preserves_decorators():
    """Проверяет, что чанкер сохраняет декораторы."""
    code_with_decorator = '''
class APIService:
    @staticmethod
    def connect(url):
        return f"Connected to {url}"
        
    @property
    def status(self):
        return "active"
        
    async def fetch_data(self):
        return {}
'''
    
    chunker = ASTPythonChunker()
    chunks = chunker.chunk(code_with_decorator)
    
    # Проверяем, что в чанках есть декораторы
    method_chunks = [chunk for chunk in chunks if chunk.metadata.get("symbol_type") == "method"]
    func_chunks = [chunk for chunk in chunks if chunk.metadata.get("symbol_type") == "function"]
    
    # Хотя бы один из чанков должен содержать декоратор @staticmethod
    has_staticmethod = any("@staticmethod" in chunk.text for chunk in (method_chunks + func_chunks))
    assert has_staticmethod, "Декораторы должны сохраняться в чанках"
    
    # Проверяем, что свойства также обрабатываются
    property_chunks = [chunk for chunk in chunks if "@property" in chunk.text]
    assert len(property_chunks) > 0, "Свойства должны быть обнаружены"


def test_ast_chunker_line_numbers():
    """Проверяет, что чанкер сохраняет правильные номера строк."""
    code = '''
class TestClass:
    def test_method(self):
        return 42
'''
    
    chunker = ASTPythonChunker()
    chunks = chunker.chunk(code)
    
    # У чанков должны быть корректные номера строк
    for chunk in chunks:
        assert isinstance(chunk.start_line, int) and chunk.start_line >= 0, "Номер начальной строки должен быть целым числом >= 0"
        assert isinstance(chunk.end_line, int) and chunk.end_line >= chunk.start_line, "Номер конечной строки должен быть >= начальной"


def test_context_breadcrumbs():
    """Проверяет формирование 'хлебных крошек' контекста."""
    code = '''
class Level1:
    def method1(self):
        pass
        
    class Level2:
        def method2(self):
            pass
            
        def method2_alt(self):
            pass
'''
    
    chunker = ASTPythonChunker()
    chunks = chunker.chunk(code)
    
    # Ищем методы во вложенном классе
    level2_methods = [c for c in chunks 
                      if c.metadata.get("parent_context") == "Level1 > Level2" 
                      and c.metadata.get("symbol_type") == "method"]
    
    assert len(level2_methods) == 2, "Должно быть 2 метода во втором уровне вложенности"
    
    # Проверяем, что контекст правильно формируется
    for method in level2_methods:
        assert "Level1 > Level2" in method.metadata["parent_context"], \
            f"Контекст должен содержать полный путь: {method.metadata['parent_context']}"


if __name__ == "__main__":
    # Запуск тестов
    test_ast_chunker_nested_classes()
    test_ast_chunker_preserves_decorators()
    test_ast_chunker_line_numbers()
    test_context_breadcrumbs()
    print("Все тесты пройдены успешно!")