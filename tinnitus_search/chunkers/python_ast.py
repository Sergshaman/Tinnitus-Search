import ast
import logging
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any

# Настройка логгера
logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    start_line: int = 0
    end_line: int = 0
    content_hash: str = ""

class ContextAwareVisitor(ast.NodeVisitor):
    def __init__(self, source_code: str, min_tokens: int = 50):
        self.source_code = source_code
        self.source_lines = source_code.splitlines(keepends=True)
        self.chunks: List[Chunk] = []
        self.context_stack: List[str] = [] 
        self.min_tokens = min_tokens

    def _get_node_source(self, node: ast.AST) -> tuple[str, int, int]:
        """Извлекает исходный код ноды, включая декораторы."""
        start_lineno = getattr(node, 'lineno', 1)
        
        # Учитываем декораторы (они начинаются раньше самой функции/класса)
        if hasattr(node, 'decorator_list') and node.decorator_list:
            start_lineno = min(d.lineno for d in node.decorator_list)
        
        end_lineno = getattr(node, 'end_lineno', start_lineno)
        
        # Извлекаем строки кода
        source = "".join(self.source_lines[start_lineno - 1 : end_lineno])
        return source, start_lineno, end_lineno

    def _make_breadcrumb(self, node_name: str) -> str:
        """Создает путь контекста, например: 'UserManager > verify_token'"""
        if not self.context_stack:
            return node_name
        return f"{' > '.join(self.context_stack)} > {node_name}"

    def _process_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], is_async: bool = False):
        """Единая логика для синхронных и асинхронных функций."""
        function_source, start, end = self._get_node_source(node)
        symbol_name = self._make_breadcrumb(node.name)
        
        # Важно для LLM: добавляем контекст прямо в текст
        context_header = f"# Context: {symbol_name}\n"
        full_text = context_header + function_source

        chunk = Chunk(
            text=full_text,
            start_line=start,
            end_line=end,
            metadata={
                "symbol_name": node.name,
                "symbol_type": "method" if self.context_stack else "function",
                "parent_context": " > ".join(self.context_stack),
                "is_async": is_async,
                "lines": f"{start}-{end}"
            }
        )
        self.chunks.append(chunk)

    def visit_ClassDef(self, node: ast.ClassDef):
        """Обработка классов с защитой от пустых тел."""
        start_lineno = node.lineno
        if node.decorator_list:
            start_lineno = min(d.lineno for d in node.decorator_list)
        
        # Определяем конец заголовка (до первого метода или до конца класса)
        if node.body:
            body_start = getattr(node.body[0], 'lineno', node.end_lineno)
        else:
            body_start = node.end_lineno

        header_text = "".join(self.source_lines[start_lineno - 1 : body_start - 1])

        if len(header_text.strip()) > 20: 
            self.chunks.append(Chunk(
                text=f"# Context: {node.name} (Class Definition)\n{header_text}",
                start_line=start_lineno,
                end_line=body_start - 1,
                metadata={
                    "symbol_name": node.name,
                    "symbol_type": "class",
                    "parent_context": " > ".join(self.context_stack)
                }
            ))

        self.context_stack.append(node.name)
        self.generic_visit(node)
        self.context_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._process_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._process_function(node, is_async=True)


class ASTPythonChunker:
    """Интерфейс для нарезки Python кода с учетом AST."""
    def __init__(self, target_tokens: int = 400, overlap: int = 50):
        self.target_tokens = target_tokens
        self.overlap = overlap

    def chunk(self, text: str) -> List[Chunk]:
        if not text.strip():
            return []
            
        try:
            visitor = ContextAwareVisitor(text)
            tree = ast.parse(text)
            visitor.visit(tree)
            
            final_chunks = []
            for chunk in visitor.chunks:
                # Если функция слишком большая, режем её на части
                if self._estimate_tokens(chunk.text) > self.target_tokens:
                    final_chunks.extend(self._split_large_chunk(chunk))
                else:
                    final_chunks.append(chunk)
            
            # Генерация хешей контента для каждого чанка
            for c in final_chunks:
                c.content_hash = hashlib.sha256(c.text.encode('utf-8')).hexdigest()
                
            return final_chunks

        except SyntaxError as e:
            logger.warning(f"Syntax error in code, skipping AST: {e}")
            return []
        except Exception as e:
            logger.error(f"Error in AST chunker: {e}")
            return []

    def _estimate_tokens(self, text: str) -> int:
        # Для кода 1 токен в среднем это 3-4 символа
        return len(text) // 3

    def _split_large_chunk(self, chunk: Chunk) -> List[Chunk]:
        """Разрезает длинные блоки кода, сохраняя # Context заголовок."""
        lines = chunk.text.splitlines(keepends=True)
        header = lines[0] if lines and lines[0].startswith("# Context:") else ""
        body = lines[1:] if header else lines
        
        sub_chunks = []
        max_chars = self.target_tokens * 3
        current_batch = []
        current_chars = 0
        part_idx = 0
        current_start_line = chunk.start_line

        for line in body:
            current_batch.append(line)
            current_chars += len(line)
            
            if current_chars >= max_chars:
                text_part = header + "".join(current_batch)
                # Вычисляем конечную строку для этого суб-чанка
                current_end_line = current_start_line + len(current_batch) - 1
                new_chunk = Chunk(
                    text=text_part,
                    start_line=current_start_line,
                    end_line=current_end_line,
                    metadata={**chunk.metadata, "part": part_idx, "is_split": True}
                )
                sub_chunks.append(new_chunk)
                
                # Реализация Overlap (нахлеста)
                overlap_size = 0
                overlap_lines = []
                for l in reversed(current_batch):
                    overlap_size += len(l)
                    overlap_lines.insert(0, l)
                    if overlap_size >= (self.overlap * 3):
                        break
                
                # Обновляем начальную строку для следующего чанка
                # Учитываем, что overlap_lines содержит строки из конца current_batch
                lines_in_overlap = len(overlap_lines)
                current_start_line = current_end_line - lines_in_overlap + 1
                
                current_batch = overlap_lines
                current_chars = overlap_size
                part_idx += 1

        if current_batch:
            current_end_line = current_start_line + len(current_batch) - 1
            sub_chunks.append(Chunk(
                text=header + "".join(current_batch),
                start_line=current_start_line,
                end_line=current_end_line,
                metadata={**chunk.metadata, "part": part_idx, "is_split": True}
            ))

        return sub_chunks