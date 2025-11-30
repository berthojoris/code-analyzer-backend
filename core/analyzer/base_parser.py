from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class CodeChunk:
    file_path: str
    content: str
    language: str
    chunk_type: str  # "function", "class", "module", "block"
    name: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    importance: float = 1.0


class BaseParser(ABC):
    """Abstract base class for language-specific code parsers."""

    def __init__(self, max_chunk_size: int = 2000):
        self.max_chunk_size = max_chunk_size

    @abstractmethod
    def parse_file(self, file_path: Path) -> List[CodeChunk]:
        """Parse a file and return code chunks."""
        pass

    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        pass

    def chunk_text(self, text: str, file_path: str, language: str) -> List[CodeChunk]:
        """Split text into chunks if it exceeds max size."""
        if len(text) <= self.max_chunk_size:
            return [
                CodeChunk(
                    file_path=file_path,
                    content=text,
                    language=language,
                    chunk_type="block",
                )
            ]

        chunks = []
        lines = text.split("\n")
        current_chunk = []
        current_size = 0
        start_line = 0

        for i, line in enumerate(lines):
            line_size = len(line) + 1
            if current_size + line_size > self.max_chunk_size and current_chunk:
                chunks.append(
                    CodeChunk(
                        file_path=file_path,
                        content="\n".join(current_chunk),
                        language=language,
                        chunk_type="block",
                        start_line=start_line,
                        end_line=i - 1,
                    )
                )
                current_chunk = []
                current_size = 0
                start_line = i

            current_chunk.append(line)
            current_size += line_size

        if current_chunk:
            chunks.append(
                CodeChunk(
                    file_path=file_path,
                    content="\n".join(current_chunk),
                    language=language,
                    chunk_type="block",
                    start_line=start_line,
                    end_line=len(lines) - 1,
                )
            )

        return chunks
