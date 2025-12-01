"""
Base analyzer for all static analysis tools.
"""
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

from ..database.models import LintingIssue, QualityMetric, AnalysisCache


@dataclass
class AnalysisResult:
    """Result of a static analysis operation."""
    tool: str
    success: bool
    duration_ms: int
    issues: List[Dict[str, Any]]
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BaseAnalyzer(ABC):
    """Abstract base class for all static analysis tools."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tool_name = self.__class__.__name__.replace("Analyzer", "").lower()

    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        pass

    @abstractmethod
    def analyze_file(self, file_path: Path, repository_id: int) -> AnalysisResult:
        """Analyze a single file and return results."""
        pass

    def get_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file content."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except (IOError, OSError) as e:
            raise ValueError(f"Cannot read file {file_path}: {e}")

    def should_analyze_file(self, file_path: Path, cache: Optional[Dict[str, Any]] = None) -> bool:
        """Check if file should be analyzed based on cache and file hash."""
        if cache is None:
            return True

        try:
            file_hash = self.get_file_hash(file_path)
            cache_key = f"{self.tool_name}:{file_path}"

            if cache_key in cache:
                cached_hash = cache[cache_key].get('file_hash')
                if cached_hash == file_hash:
                    # File hasn't changed, return cached result
                    return False
        except (ValueError, KeyError):
            pass

        return True

    def validate_file_size(self, file_path: Path, max_size_mb: int = 10) -> bool:
        """Validate that file size is within acceptable limits."""
        try:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            return file_size_mb <= max_size_mb
        except (OSError, AttributeError):
            return False

    def get_cache_key(self, file_path: Path) -> str:
        """Generate cache key for file analysis."""
        return f"{self.tool_name}:{file_path}"

    def prepare_analysis_command(self, file_path: Path) -> List[str]:
        """Prepare command line for running analysis tool."""
        raise NotImplementedError("Subclasses must implement this method")

    def parse_tool_output(self, output: str, file_path: Path) -> AnalysisResult:
        """Parse the output from analysis tool."""
        raise NotImplementedError("Subclasses must implement this method")

    async def analyze_with_cache(
        self,
        file_path: Path,
        repository_id: int,
        cache: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """Analyze file with caching support."""
        start_time = datetime.now()

        # Check if we should analyze this file
        if not self.should_analyze_file(file_path, cache):
            return AnalysisResult(
                tool=self.tool_name,
                success=True,
                duration_ms=0,
                issues=[],
                metrics={}
            )

        # Validate file size
        if not self.validate_file_size(file_path):
            return AnalysisResult(
                tool=self.tool_name,
                success=False,
                duration_ms=0,
                error=f"File size exceeds maximum allowed size"
            )

        try:
            # Prepare and execute analysis command
            command = self.prepare_analysis_command(file_path)

            # Execute analysis (this would be implemented in subclasses)
            result = await self._execute_analysis_command(command, file_path)

            # Calculate duration
            end_time = datetime.now()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)

            result.duration_ms = duration_ms

            return result

        except Exception as e:
            return AnalysisResult(
                tool=self.tool_name,
                success=False,
                duration_ms=0,
                error=str(e)
            )

    async def _execute_analysis_command(self, command: List[str], file_path: Path) -> AnalysisResult:
        """Execute analysis command and return parsed result."""
        import subprocess
        import asyncio

        try:
            # Run the analysis command
            process = await asyncio.create_subprocess_exec(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=file_path.parent
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                return AnalysisResult(
                    tool=self.tool_name,
                    success=False,
                    error=f"Analysis failed with return code {process.returncode}: {stderr.decode('utf-8', errors='ignore')}"
                )

            # Parse the output
            return self.parse_tool_output(stdout.decode('utf-8', errors='ignore'), file_path)

        except Exception as e:
            return AnalysisResult(
                tool=self.tool_name,
                success=False,
                error=f"Failed to execute analysis: {e}"
            )

    def create_linting_issues(
        self,
        result: AnalysisResult,
        file_path: Path,
        repository_id: int
    ) -> List[LintingIssue]:
        """Convert analysis result to LintingIssue objects."""
        issues = []

        for issue_data in result.issues:
            issue = LintingIssue(
                repository_id=repository_id,
                file_path=str(file_path.relative_to(file_path.parent.parent)),
                line_number=issue_data.get('line', 0),
                column_number=issue_data.get('column', 0),
                rule_id=issue_data.get('rule', 'unknown'),
                severity=issue_data.get('severity', 'info'),
                message=issue_data.get('message', ''),
                tool=self.tool_name,
                category=issue_data.get('category', 'style')
            )
            issues.append(issue)

        return issues

    def create_quality_metric(
        self,
        result: AnalysisResult,
        file_path: Path,
        repository_id: int
    ) -> Optional[QualityMetric]:
        """Convert analysis result to QualityMetric object."""
        if not result.metrics:
            return None

        metric = QualityMetric(
            repository_id=repository_id,
            file_path=str(file_path.relative_to(file_path.parent.parent)),
            function_name=result.metrics.get('function_name'),
            cyclomatic_complexity=result.metrics.get('complexity', 1),
            cognitive_complexity=result.metrics.get('cognitive_complexity', 1),
            maintainability_index=result.metrics.get('maintainability_index', 100.0),
            technical_debt_ratio=result.metrics.get('technical_debt_ratio', 0.0),
            language=self.detect_language(file_path),
            analysis_date=datetime.now()
        )

        return metric

    def detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        extension = file_path.suffix.lower()

        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.go': 'go',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.rs': 'rust',
            '.scala': 'scala',
            '.sh': 'shell',
            '.sql': 'sql'
        }

        return language_map.get(extension, 'unknown')