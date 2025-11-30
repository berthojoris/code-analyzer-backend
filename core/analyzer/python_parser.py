import ast
from pathlib import Path
from typing import List

from .base_parser import BaseParser, CodeChunk


class PythonParser(BaseParser):
    """Parser for Python source files using AST."""

    def get_supported_extensions(self) -> List[str]:
        return [".py"]

    def parse_file(self, file_path: Path) -> List[CodeChunk]:
        """Parse a Python file and extract functions, classes, and imports."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []

        chunks = []
        try:
            tree = ast.parse(content)
            chunks.extend(self._extract_chunks(tree, content, str(file_path)))
        except SyntaxError:
            # Fall back to simple chunking if AST parsing fails
            return self.chunk_text(content, str(file_path), "python")

        # If no meaningful chunks found, chunk the whole file
        if not chunks:
            return self.chunk_text(content, str(file_path), "python")

        return chunks

    def _extract_chunks(
        self, tree: ast.AST, source: str, file_path: str
    ) -> List[CodeChunk]:
        """Extract meaningful chunks from AST."""
        chunks = []
        lines = source.split("\n")

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                chunks.append(self._create_function_chunk(node, lines, file_path))
            elif isinstance(node, ast.AsyncFunctionDef):
                chunks.append(self._create_function_chunk(node, lines, file_path))
            elif isinstance(node, ast.ClassDef):
                chunks.append(self._create_class_chunk(node, lines, file_path))

        return [c for c in chunks if c is not None]

    def _create_function_chunk(
        self, node: ast.FunctionDef, lines: List[str], file_path: str
    ) -> CodeChunk:
        """Create a chunk from a function definition."""
        start = node.lineno - 1
        end = node.end_lineno or node.lineno
        content = "\n".join(lines[start:end])

        # Determine importance based on docstring and decorators
        importance = 1.0
        if ast.get_docstring(node):
            importance += 0.2
        if node.decorator_list:
            importance += 0.1

        return CodeChunk(
            file_path=file_path,
            content=content,
            language="python",
            chunk_type="function",
            name=node.name,
            start_line=start,
            end_line=end,
            importance=importance,
        )

    def _create_class_chunk(
        self, node: ast.ClassDef, lines: List[str], file_path: str
    ) -> CodeChunk:
        """Create a chunk from a class definition."""
        start = node.lineno - 1
        end = node.end_lineno or node.lineno
        content = "\n".join(lines[start:end])

        # Classes are generally more important
        importance = 1.3
        if ast.get_docstring(node):
            importance += 0.2

        return CodeChunk(
            file_path=file_path,
            content=content,
            language="python",
            chunk_type="class",
            name=node.name,
            start_line=start,
            end_line=end,
            importance=importance,
        )


class GenericParser(BaseParser):
    """Generic parser for languages without AST support."""

    def __init__(self, language: str, extensions: List[str], max_chunk_size: int = 2000):
        super().__init__(max_chunk_size)
        self.language = language
        self.extensions = extensions

    def get_supported_extensions(self) -> List[str]:
        return self.extensions

    def parse_file(self, file_path: Path) -> List[CodeChunk]:
        """Parse file using simple text chunking."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []

        return self.chunk_text(content, str(file_path), self.language)
