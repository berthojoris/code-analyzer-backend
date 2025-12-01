"""
Linting and style checking analyzer.
"""
import json
import re
from pathlib import Path
from typing import Dict, Any, List

from .base_analyzer import BaseAnalyzer, AnalysisResult
from ..database.models import LintingIssue


class LintingAnalyzer(BaseAnalyzer):
    """Main linting analyzer that orchestrates multiple linting tools."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.python_linter = PythonLinter(config)
        self.javascript_linter = JavaScriptLinter(config)
        self.general_linter = GeneralLinter(config)

    def get_supported_extensions(self) -> List[str]:
        """Return all supported file extensions."""
        return ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h']

    async def analyze_file(self, file_path: Path, repository_id: int) -> AnalysisResult:
        """Analyze file with appropriate linter."""
        file_extension = file_path.suffix.lower()

        if file_extension == '.py':
            return await self.python_linter.analyze_file(file_path, repository_id)
        elif file_extension in ['.js', '.ts', '.jsx', '.tsx']:
            return await self.javascript_linter.analyze_file(file_path, repository_id)
        else:
            return await self.general_linter.analyze_file(file_path, repository_id)


class PythonLinter(BaseAnalyzer):
    """Python code linter using ruff."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.tool_name = "ruff"

    def get_supported_extensions(self) -> List[str]:
        return ['.py']

    def prepare_analysis_command(self, file_path: Path) -> List[str]:
        """Prepare ruff command for analysis."""
        config_path = self.config.get('ruff_config_path')
        command = ['ruff', 'check', '--output-format=json', '--no-fix']

        if config_path and Path(config_path).exists():
            command.extend(['--config', config_path])

        command.append(str(file_path))
        return command

    def parse_tool_output(self, output: str, file_path: Path) -> AnalysisResult:
        """Parse ruff JSON output."""
        try:
            data = json.loads(output)
            issues = []

            for issue in data:
                parsed_issue = {
                    'rule_id': issue.get('code', 'unknown'),
                    'severity': self._map_severity(issue.get('level', 'info')),
                    'message': issue.get('message', ''),
                    'line': issue.get('location', {}).get('row', 0),
                    'column': issue.get('location', {}).get('column', 0),
                    'category': self._categorize_issue(issue.get('code', 'unknown')),
                    'suggestion': self._get_suggestion(issue)
                }
                issues.append(parsed_issue)

            return AnalysisResult(
                tool=self.tool_name,
                success=True,
                duration_ms=0,  # Will be set by base analyzer
                issues=issues
            )

        except json.JSONDecodeError as e:
            return AnalysisResult(
                tool=self.tool_name,
                success=False,
                error=f"Failed to parse ruff output: {e}"
            )

    def _map_severity(self, level: str) -> str:
        """Map ruff severity levels."""
        severity_map = {
            'error': 'error',
            'warning': 'warning',
            'info': 'info',
            'hint': 'info'
        }
        return severity_map.get(level, 'info')

    def _categorize_issue(self, code: str) -> str:
        """Categorize ruff issues."""
        if code.startswith('E'):  # Error codes
            return 'error'
        elif code.startswith('W'):  # Warning codes
            return 'warning'
        elif code.startswith('F'):  # Pyflakes codes
            return 'undefined'
        else:
            return 'style'

    def _get_suggestion(self, issue: Dict[str, Any]) -> str:
        """Get suggestion for fixing the issue."""
        fix = issue.get('fix')
        if fix and 'applicable' in fix:
            available_fixes = fix.get('available', [])
            if available_fixes:
                return f"Auto-fix available: {available_fixes[0].get('message', 'Unknown fix')}"
        return ""


class JavaScriptLinter(BaseAnalyzer):
    """JavaScript/TypeScript linter using ESLint."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.tool_name = "eslint"

    def get_supported_extensions(self) -> List[str]:
        return ['.js', '.ts', '.jsx', '.tsx']

    def prepare_analysis_command(self, file_path: Path) -> List[str]:
        """Prepare ESLint command for analysis."""
        config_path = self.config.get('eslint_config_path')
        command = ['npx', 'eslint', '--format=json']

        if config_path and Path(config_path).exists():
            command.extend(['--config', config_path])

        command.append(str(file_path))
        return command

    def parse_tool_output(self, output: str, file_path: Path) -> AnalysisResult:
        """Parse ESLint JSON output."""
        try:
            data = json.loads(output)
            issues = []

            for file_result in data:
                for issue in file_result.get('messages', []):
                    parsed_issue = {
                        'rule_id': issue.get('ruleId', 'unknown'),
                        'severity': self._map_severity(issue.get('severity', 'info')),
                        'message': issue.get('message', ''),
                        'line': issue.get('line', 0),
                        'column': issue.get('column', 0),
                        'category': self._categorize_issue(issue.get('ruleId', 'unknown')),
                        'suggestion': self._get_suggestion(issue)
                    }
                    issues.append(parsed_issue)

            return AnalysisResult(
                tool=self.tool_name,
                success=True,
                duration_ms=0,
                issues=issues
            )

        except json.JSONDecodeError as e:
            return AnalysisResult(
                tool=self.tool_name,
                success=False,
                error=f"Failed to parse ESLint output: {e}"
            )

    def _map_severity(self, severity: str) -> str:
        """Map ESLint severity levels."""
        severity_map = {
            'error': 'error',
            'warn': 'warning',
            'warning': 'warning',
            'info': 'info'
        }
        return severity_map.get(severity, 'info')

    def _categorize_issue(self, rule_id: str) -> str:
        """Categorize ESLint issues."""
        if rule_id:
            if 'no-unused-vars' in rule_id:
                return 'unused'
            elif 'no-console' in rule_id:
                return 'best-practices'
            elif 'semi' in rule_id:
                return 'style'
            elif 'quotes' in rule_id:
                return 'style'
            elif 'indent' in rule_id:
                return 'style'
            else:
                return 'error'
        return 'unknown'

    def _get_suggestion(self, issue: Dict[str, Any]) -> str:
        """Get suggestion for fixing the issue."""
        return issue.get('suggestion', '')


class GeneralLinter(BaseAnalyzer):
    """General linter using flake8 for multiple languages."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.tool_name = "flake8"

    def get_supported_extensions(self) -> List[str]:
        return ['.py', '.js', '.java', '.cpp', '.c', '.h']

    def prepare_analysis_command(self, file_path: Path) -> List[str]:
        """Prepare flake8 command for analysis."""
        config_path = self.config.get('flake8_config_path')
        command = ['flake8', '--format=json']

        if config_path and Path(config_path).exists():
            command.extend(['--config', config_path])

        command.append(str(file_path))
        return command

    def parse_tool_output(self, output: str, file_path: Path) -> AnalysisResult:
        """Parse flake8 JSON output."""
        try:
            data = json.loads(output)
            issues = []

            for issue in data:
                parsed_issue = {
                    'rule_id': f"{issue.get('code', 'unknown')}",
                    'severity': self._map_severity(issue.get('text', '')),
                    'message': issue.get('text', ''),
                    'line': issue.get('line_number', 0),
                    'column': issue.get('column_number', 0),
                    'category': self._categorize_issue(issue.get('code', 'unknown')),
                    'suggestion': ''
                }
                issues.append(parsed_issue)

            return AnalysisResult(
                tool=self.tool_name,
                success=True,
                duration_ms=0,
                issues=issues
            )

        except json.JSONDecodeError as e:
            return AnalysisResult(
                tool=self.tool_name,
                success=False,
                error=f"Failed to parse flake8 output: {e}"
            )

    def _map_severity(self, text: str) -> str:
        """Map flake8 severity levels."""
        text_lower = text.lower() if text else ''
        if 'error' in text_lower:
            return 'error'
        elif 'warning' in text_lower:
            return 'warning'
        else:
            return 'info'

    def _categorize_issue(self, code: str) -> str:
        """Categorize flake8 issues."""
        if not code:
            return 'unknown'

        # Flake8 codes: E=error, W=warning, F=Pyflakes, etc.
        if code.startswith('E'):
            return 'error'
        elif code.startswith('W'):
            return 'warning'
        elif code.startswith('F'):
            return 'undefined'
        else:
            return 'style'