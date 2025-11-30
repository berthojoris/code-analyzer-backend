import pytest
from unittest.mock import patch, MagicMock

from api.routes.index import extract_repo_name, get_parser_for_language
from core.analyzer.python_parser import PythonParser, GenericParser


def test_extract_repo_name_github_url():
    """Test extracting repo name from GitHub URL."""
    url = "https://github.com/owner/repo-name"
    assert extract_repo_name(url) == "owner/repo-name"


def test_extract_repo_name_with_git_suffix():
    """Test extracting repo name from URL with .git suffix."""
    url = "https://github.com/owner/repo-name.git"
    assert extract_repo_name(url) == "owner/repo-name"


def test_extract_repo_name_trailing_slash():
    """Test extracting repo name from URL with trailing slash."""
    url = "https://github.com/owner/repo-name/"
    assert extract_repo_name(url) == "owner/repo-name"


def test_get_parser_for_python():
    """Test that Python parser is returned for Python language."""
    parser = get_parser_for_language("python")
    assert isinstance(parser, PythonParser)


def test_get_parser_for_javascript():
    """Test that GenericParser is returned for JavaScript."""
    parser = get_parser_for_language("javascript")
    assert isinstance(parser, GenericParser)
    assert ".js" in parser.get_supported_extensions()


def test_get_parser_for_unknown():
    """Test that GenericParser is returned for unknown language."""
    parser = get_parser_for_language("unknown-lang")
    assert isinstance(parser, GenericParser)


class TestPythonParser:
    """Tests for the Python parser."""

    def test_supported_extensions(self):
        """Test that Python parser supports .py extension."""
        parser = PythonParser()
        assert ".py" in parser.get_supported_extensions()

    def test_parse_empty_content(self, tmp_path):
        """Test parsing an empty Python file."""
        test_file = tmp_path / "empty.py"
        test_file.write_text("")
        
        parser = PythonParser()
        chunks = parser.parse_file(test_file)
        assert len(chunks) == 0 or all(c.content.strip() == "" for c in chunks)

    def test_parse_simple_function(self, tmp_path):
        """Test parsing a simple Python function."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def hello_world():
    print("Hello, World!")
""")
        
        parser = PythonParser()
        chunks = parser.parse_file(test_file)
        assert len(chunks) >= 1
        assert any("hello_world" in c.content for c in chunks)


class TestLanguageDetector:
    """Tests for language detection."""

    def test_detect_python_files(self, tmp_path):
        """Test detection of Python as dominant language."""
        from core.analyzer.language_detector import detect_language
        
        # Create Python files
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "utils.py").write_text("def util(): pass")
        
        result = detect_language(str(tmp_path))
        assert result == "python"

    def test_detect_no_code_files(self, tmp_path):
        """Test detection with no code files."""
        from core.analyzer.language_detector import detect_language
        
        (tmp_path / "readme.md").write_text("# Readme")
        
        result = detect_language(str(tmp_path))
        assert result == "unknown"
