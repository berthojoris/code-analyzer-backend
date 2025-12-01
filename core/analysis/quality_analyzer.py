"""
Code quality metrics analyzer.
"""
import ast
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import radon.complexity as radon_cc
import radon.metrics as radon_metrics
import lizard

from .base_analyzer import BaseAnalyzer, AnalysisResult
from ..database.models import QualityMetric


class QualityAnalyzer(BaseAnalyzer):
    """Code quality metrics analyzer."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.complexity_threshold = config.get('complexity_threshold', 10)

    def get_supported_extensions(self) -> List[str]:
        """Return supported file extensions for quality analysis."""
        return ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h']

    async def analyze_file(self, file_path: Path, repository_id: int) -> AnalysisResult:
        """Analyze file and return quality metrics."""
        file_extension = file_path.suffix.lower()

        if file_extension == '.py':
            return await self._analyze_python_file(file_path, repository_id)
        elif file_extension in ['.js', '.ts']:
            return await self._analyze_javascript_file(file_path, repository_id)
        else:
            return await self._analyze_general_file(file_path, repository_id)

    async def _analyze_python_file(self, file_path: Path, repository_id: int) -> AnalysisResult:
        """Analyze Python file for quality metrics."""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Calculate complexity metrics
            complexity_metrics = self._calculate_python_complexity(content)

            # Calculate maintainability index
            maintainability_index = self._calculate_maintainability_index(content, complexity_metrics)

            # Calculate Halstead metrics
            halstead_metrics = self._calculate_halstead_metrics(content)

            # Calculate additional metrics
            additional_metrics = self._calculate_additional_python_metrics(content, file_path)

            # Combine all metrics
            metrics = {
                'cyclomatic_complexity': complexity_metrics['cyclomatic'],
                'cognitive_complexity': complexity_metrics['cognitive'],
                'halstead_volume': halstead_metrics['volume'],
                'halstead_difficulty': halstead_metrics['difficulty'],
                'maintainability_index': maintainability_index,
                'technical_debt_ratio': additional_metrics['technical_debt_ratio'],
                'code_smells_count': additional_metrics['code_smells_count'],
                'duplicated_lines': 0,  # Will be calculated separately
                'lines_of_code': additional_metrics['lines_of_code'],
                'comment_lines': additional_metrics['comment_lines'],
                'blank_lines': additional_metrics['blank_lines'],
                'language': 'python'
            }

            return AnalysisResult(
                tool=self.tool_name,
                success=True,
                duration_ms=0,  # Will be set by base analyzer
                issues=[],  # Quality analyzer focuses on metrics
                metrics=metrics
            )

        except Exception as e:
            return AnalysisResult(
                tool=self.tool_name,
                success=False,
                error=f"Failed to analyze Python file: {e}"
            )

    async def _analyze_javascript_file(self, file_path: Path, repository_id: int) -> AnalysisResult:
        """Analyze JavaScript/TypeScript file for quality metrics."""
        try:
            # Use Lizard for multi-language complexity analysis
            file_analysis = lizard.analyze_file(str(file_path))

            if not file_analysis:
                return AnalysisResult(
                    tool=self.tool_name,
                    success=False,
                    error="Could not analyze JavaScript file"
                )

            # Get function-level metrics
            functions = file_analysis.function_list
            total_complexity = sum(f.cyclomatic_complexity for f in functions if f.cyclomatic_complexity)
            max_complexity = max(f.cyclomatic_complexity for f in functions if f.cyclomatic_complexity) if functions else 0
            avg_complexity = total_complexity / len(functions) if functions else 0

            # Calculate maintainability index (simplified)
            maintainability_index = self._calculate_maintainability_index_from_lizard(file_analysis)

            # Calculate additional metrics
            lines_info = self._get_file_lines_info(file_path)

            metrics = {
                'cyclomatic_complexity': int(total_complexity),
                'cognitive_complexity': int(max_complexity),  # Lizard doesn't provide cognitive complexity
                'maintainability_index': maintainability_index,
                'technical_debt_ratio': 0.0,  # Would need more complex analysis
                'code_smells_count': len([f for f in functions if f.cyclomatic_complexity > self.complexity_threshold]),
                'duplicated_lines': 0,  # Will be calculated separately
                'lines_of_code': lines_info['code_lines'],
                'comment_lines': lines_info['comment_lines'],
                'blank_lines': lines_info['blank_lines'],
                'language': 'javascript' if file_path.suffix == '.js' else 'typescript'
            }

            return AnalysisResult(
                tool=self.tool_name,
                success=True,
                duration_ms=0,
                issues=[],
                metrics=metrics
            )

        except Exception as e:
            return AnalysisResult(
                tool=self.tool_name,
                success=False,
                error=f"Failed to analyze JavaScript/TypeScript file: {e}"
            )

    async def _analyze_general_file(self, file_path: Path, repository_id: int) -> AnalysisResult:
        """Analyze general source file for basic quality metrics."""
        try:
            # Use Lizard for general analysis
            file_analysis = lizard.analyze_file(str(file_path))

            if not file_analysis:
                return AnalysisResult(
                    tool=self.tool_name,
                    success=False,
                    error="Could not analyze file"
                )

            # Get basic metrics
            functions = file_analysis.function_list
            total_complexity = sum(f.cyclomatic_complexity for f in functions if f.cyclomatic_complexity)
            avg_complexity = total_complexity / len(functions) if functions else 0

            lines_info = self._get_file_lines_info(file_path)
            language = self.detect_language(file_path)

            metrics = {
                'cyclomatic_complexity': int(total_complexity),
                'cognitive_complexity': int(total_complexity),  # Simplified
                'maintainability_index': self._calculate_maintainability_index_from_lizard(file_analysis),
                'technical_debt_ratio': 0.0,
                'code_smells_count': len([f for f in functions if f.cyclomatic_complexity > self.complexity_threshold]),
                'duplicated_lines': 0,
                'lines_of_code': lines_info['code_lines'],
                'comment_lines': lines_info['comment_lines'],
                'blank_lines': lines_info['blank_lines'],
                'language': language
            }

            return AnalysisResult(
                tool=self.tool_name,
                success=True,
                duration_ms=0,
                issues=[],
                metrics=metrics
            )

        except Exception as e:
            return AnalysisResult(
                tool=self.tool_name,
                success=False,
                error=f"Failed to analyze file: {e}"
            )

    def _calculate_python_complexity(self, content: str) -> Dict[str, int]:
        """Calculate Python complexity metrics using Radon."""
        try:
            # Calculate cyclomatic complexity
            cc = radon_cc.cc_visit(content)
            cyclomatic = cc.total_complexity

            # Calculate cognitive complexity (simplified)
            cognitive = self._calculate_cognitive_complexity(content)

            return {
                'cyclomatic': cyclomatic,
                'cognitive': cognitive
            }
        except Exception:
            return {'cyclomatic': 1, 'cognitive': 1}

    def _calculate_cognitive_complexity(self, content: str) -> int:
        """Calculate cognitive complexity for Python code."""
        # Simplified cognitive complexity calculation
        complexity = 0

        # Count cognitive complexity keywords
        complexity_patterns = [
            r'\bif\b', r'\belif\b', r'\belse\b', r'\bfor\b', r'\bwhile\b',
            r'\bcase\b', r'\bexcept\b', r'\bfinally\b',
            r'\band\b', r'\bor\b', r'\bnot\b'
        ]

        for pattern in complexity_patterns:
            matches = len(re.findall(pattern, content))
            complexity += matches

        # Add nesting complexity
        nesting_level = 0
        for char in content:
            if char == '{':
                nesting_level += 1
            elif char == '}':
                nesting_level -= 1
            elif char == '\n':
                complexity += max(0, nesting_level - 1)

        return max(1, complexity)

    def _calculate_halstead_metrics(self, content: str) -> Dict[str, float]:
        """Calculate Halstead metrics."""
        try:
            # Extract operators and operands (simplified)
            operators = len(re.findall(r'[+\-*/%=<>!&|^~|&]|\b(and|or|not)\b', content))
            operands = len(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', content))

            # Calculate basic Halstead metrics
            n1 = operators  # Number of distinct operators
            n2 = operands  # Number of distinct operands
            N1 = operators  # Total number of operators
            N2 = operands  # Total number of operands

            # Calculate vocabulary and length
            vocabulary = n1 + n2
            length = N1 + N2

            # Calculate volume and difficulty
            if vocabulary > 0:
                volume = length * (log2(vocabulary) if vocabulary > 1 else 0)
                difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
            else:
                volume = 0
                difficulty = 0

            return {
                'volume': volume,
                'difficulty': difficulty
            }
        except Exception:
            return {'volume': 0.0, 'difficulty': 0.0}

    def _calculate_maintainability_index(self, content: str, complexity_metrics: Dict[str, int]) -> float:
        """Calculate maintainability index (simplified)."""
        try:
            # Get lines of code
            lines = len(content.split('\n'))

            # Simplified maintainability index calculation
            cyclomatic = complexity_metrics.get('cyclomatic', 1)
            cognitive = complexity_metrics.get('cognitive', 1)
            avg_complexity = (cyclomatic + cognitive) / 2

            # Maintainability index (0-100 scale, higher is better)
            if lines > 0:
                mi = max(0, 171 - 5.2 * log2(avg_complexity) - 0.23 * cyclomatic - 16.2 * log2(lines))
                return round(mi, 2)
            return 100.0
        except Exception:
            return 100.0

    def _calculate_maintainability_index_from_lizard(self, file_analysis) -> float:
        """Calculate maintainability index from Lizard analysis."""
        try:
            nloc = file_analysis.nloc
            complexity = file_analysis.CCN  # Total cyclomatic complexity

            if nloc > 0:
                # Simplified maintainability calculation
                cc_avg = complexity / len(file_analysis.function_list) if file_analysis.function_list else 1
                mi = max(0, 171 - 5.2 * (cc_avg ** 0.5) - 0.23 * complexity - 16.2 * (nloc ** 0.5))
                return round(mi, 2)
            return 100.0
        except Exception:
            return 100.0

    def _calculate_additional_python_metrics(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Calculate additional Python-specific metrics."""
        lines_info = self._get_file_lines_info(file_path)

        # Calculate technical debt ratio (simplified)
        complexity_issues = self._count_complexity_issues(content)
        technical_debt_ratio = complexity_issues / lines_info['code_lines'] if lines_info['code_lines'] > 0 else 0

        # Count code smells
        code_smells_count = self._count_code_smells(content)

        return {
            'lines_of_code': lines_info['code_lines'],
            'comment_lines': lines_info['comment_lines'],
            'blank_lines': lines_info['blank_lines'],
            'technical_debt_ratio': technical_debt_ratio,
            'code_smells_count': code_smells_count
        }

    def _count_complexity_issues(self, content: str) -> int:
        """Count potential complexity issues."""
        # Count long functions, deep nesting, etc.
        issues = 0

        # Check for functions with high complexity (simplified)
        functions = re.findall(r'def\s+\w+\([^)]*\):', content)
        for func in functions:
            func_lines = len(func.split('\n'))
            if func_lines > 20:  # Long function
                issues += 1

        return issues

    def _count_code_smells(self, content: str) -> int:
        """Count common code smells."""
        smells = 0

        # Common Python code smells (simplified detection)
        if 'eval(' in content:
            smells += 1
        if 'exec(' in content:
            smells += 1
        if re.search(r'except\s*:\s*pass', content):
            smells += 1
        if len(re.findall(r'if.*==.*True', content)) > 3:  # Too many True checks
            smells += 1

        return smells

    def _get_file_lines_info(self, file_path: Path) -> Dict[str, int]:
        """Get line count information for a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            code_lines = 0
            comment_lines = 0
            blank_lines = 0

            for line in lines:
                line = line.strip()
                if not line:
                    blank_lines += 1
                elif line.startswith('#') or line.startswith('//') or line.startswith('/*'):
                    comment_lines += 1
                else:
                    code_lines += 1

            return {
                'code_lines': code_lines,
                'comment_lines': comment_lines,
                'blank_lines': blank_lines
            }
        except Exception:
            return {'code_lines': 0, 'comment_lines': 0, 'blank_lines': 0}