from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter

LANGUAGE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".vue": "vue",
    ".svelte": "svelte",
}

IGNORED_DIRS = {
    "node_modules",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".next",
    "target",
    "vendor",
    ".idea",
    ".vscode",
}


def detect_language(repo_path: str) -> str:
    """Detect the dominant programming language in a repository."""
    path = Path(repo_path)
    language_counts: Counter = Counter()

    for file_path in path.rglob("*"):
        if file_path.is_file() and not _should_ignore(file_path):
            ext = file_path.suffix.lower()
            if ext in LANGUAGE_EXTENSIONS:
                language_counts[LANGUAGE_EXTENSIONS[ext]] += 1

    if not language_counts:
        return "unknown"

    return language_counts.most_common(1)[0][0]


def get_language_stats(repo_path: str) -> Dict[str, int]:
    """Get file counts per language in the repository."""
    path = Path(repo_path)
    language_counts: Counter = Counter()

    for file_path in path.rglob("*"):
        if file_path.is_file() and not _should_ignore(file_path):
            ext = file_path.suffix.lower()
            if ext in LANGUAGE_EXTENSIONS:
                language_counts[LANGUAGE_EXTENSIONS[ext]] += 1

    return dict(language_counts)


def get_code_files(repo_path: str, language: Optional[str] = None) -> List[Path]:
    """Get all code files, optionally filtered by language."""
    path = Path(repo_path)
    code_files = []

    for file_path in path.rglob("*"):
        if file_path.is_file() and not _should_ignore(file_path):
            ext = file_path.suffix.lower()
            if ext in LANGUAGE_EXTENSIONS:
                if language is None or LANGUAGE_EXTENSIONS[ext] == language:
                    code_files.append(file_path)

    return code_files


def _should_ignore(file_path: Path) -> bool:
    """Check if a file path should be ignored."""
    parts = file_path.parts
    return any(ignored in parts for ignored in IGNORED_DIRS)
