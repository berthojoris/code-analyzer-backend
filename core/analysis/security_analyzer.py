"""
Security scanning analyzer using open-source tools.
"""
import asyncio
import json
import os
import re
import subprocess
from typing import Dict, Any, List, Optional
from pathlib import Path

from .base_analyzer import BaseAnalyzer, AnalysisResult
from ..database.models import get_db_session
from ...database.models import SecurityFinding
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SecurityFinding:
    """Model for security vulnerabilities."""
    def __init__(
        self,
        cve_id: Optional[str] = None,
        severity: str = "medium",
        title: str,
        description: str,
        affected_file: str,
        line_number: int,
        column_number: int,
        recommendation: str,
        tool: str = "unknown",
        references: List[str] = None
    ):
        self.cve_id = cve_id
        self.severity = severity
        self.title = title
        self.description = description
        self.affected_file = affected_file
        self.line_number = line_number
        self.column_number = column_number
        self.recommendation = recommendation
        self.tool = tool
        self.references = references or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        return {
            "cve_id": self.cve_id,
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "affected_file": self.affected_file,
            "line_number": self.line_number,
            "column_number": self.column_number,
            "recommendation": self.recommendation,
            "tool": self.tool,
            "references": self.references
        }


class SecurityAnalyzer(BaseAnalyzer):
    """Main security analyzer using open-source tools."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.tool_name = "security_analyzer"
        self.bandit_enabled = config.get('security_enabled', True)
        self.safety_enabled = config.get('safety_enabled', True)
        self.semgrep_enabled = config.get('semgrep_enabled', True)
        self.max_file_size_mb = config.get('max_file_size_mb', 50)

    def get_supported_extensions(self) -> List[str]:
        """Return supported file extensions for security scanning."""
        return ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.php', '.rb', '.go']

    def should_analyze_file(self, file_path: Path) -> bool:
        """Check if file should be scanned for security."""
        try:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            return file_size_mb <= self.max_file_size_mb
        except (OSError, AttributeError) as e:
            logger.warning(f"Cannot analyze file {file_path}: {e}")
            return False

    async def analyze_file(self, file_path: Path, repository_id: int) -> AnalysisResult:
        """Analyze file for security vulnerabilities."""
        start_time = datetime.now()
        file_extension = file_path.suffix.lower()

        # Run appropriate security tools based on file type
        try:
            if file_extension == '.py' and self.bandit_enabled:
                bandit_result = await self._run_bandit(file_path)
                if not bandit_result.success:
                    return bandit_result
            elif self.safety_enabled:
                safety_result = await self._run_safety(file_path)
                if not safety_result.success:
                    return safety_result
            elif self.semgrep_enabled:
                semgrep_result = await self._run_semgrep(file_path)
                if not semgrep_result.success:
                    return semgrep_result
            else:
                # General security scan using multiple tools
                return await self._run_general_security_scan(file_path)
        except Exception as e:
            return AnalysisResult(
                tool=self.tool_name,
                success=False,
                duration_ms=0,
                error=str(e)
            )

    async def _run_bandit(self, file_path: Path) -> AnalysisResult:
        """Run bandit security scan on Python file."""
        try:
            command = ['bandit', '-r', file, '--format=json']
            result = await self._execute_security_command(command, file_path)

            if result.success and result.output:
                issues = []
                try:
                    data = json.loads(result.output)
                    for issue in data.get('results', []):
                        security_finding = SecurityFinding(
                            cve_id=f"CVE-{issue.get('test_name', '').replace('_', '-')}",
                            severity=self._map_bandit_severity(issue.get('issue_severity', 'medium')),
                            title=issue.get('test_name', 'Security Issue'),
                            description=issue.get('issue_text', ''),
                            affected_file=str(file_path),
                            line_number=issue.get('line_number', 0),
                            column_number=issue.get('col', 0),
                            recommendation=issue.get('more_info', ''),
                            tool="bandit"
                        )
                        issues.append(security_finding)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse bandit output: {result.output}")
                    return AnalysisResult(
                        tool="bandit",
                        success=False,
                        error="Failed to parse bandit output"
                    )

                return AnalysisResult(
                    tool="bandit",
                    success=True,
                    duration_ms=0,
                    issues=issues
                )

        except Exception as e:
            return AnalysisResult(
                tool="bandit",
                success=False,
                error=f"Bandit scan failed: {e}"
            )

    async def _run_safety(self, file_path: Path) -> AnalysisResult:
        """Run safety dependency check on Python file."""
        try:
            command = ['safety', 'check', '--json', '--output', '/tmp/safety_output.json', str(file_path)]
            result = await self._execute_security_command(command, file_path)

            if result.success and result.output:
                issues = []
                try:
                    data = json.loads(result.output)
                    for vulnerability in data.get('vulnerabilities', []):
                        security_finding = SecurityFinding(
                            cve_id=vulnerability.get('id', ''),
                            severity="high",
                            title=f"Dependency vulnerability: {vulnerability.get('advisory', '')}",
                            description=vulnerability.get('advisory', ''),
                            affected_file=str(file_path),
                            line_number=0,
                            column_number=0,
                            recommendation="Update dependency using pip",
                            tool="safety"
                        )
                        issues.append(security_finding)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse safety output: {result.output}")
                    return AnalysisResult(
                        tool="safety",
                        success=False,
                        error="Failed to parse safety output"
                    )

                return AnalysisResult(
                    tool="safety",
                    success=True,
                    duration_ms=0,
                    issues=issues
                )

        except Exception as e:
            return AnalysisResult(
                tool="safety",
                success=False,
                error=f"Safety check failed: {e}"
            )

    async def _run_semgrep(self, file_path: Path) -> AnalysisResult:
        """Run semgrep security scan on file."""
        try:
            command = ['semgrep', '--config=auto', '--json', '--output', '/tmp/semgrep_output.json', str(file_path)]
            result = await self._execute_security_command(command, file_path)

            if result.success and result.output:
                issues = []
                try:
                    data = json.loads(result.output)
                    for issue in data.get('results', []):
                        security_finding = SecurityFinding(
                            cve_id=issue.get('check_id', '').replace('.', '-'),
                            severity=self._map_semgrep_severity(issue.get('metadata', {}).get('issue.severity', 'info')),
                            title=issue.get('message', 'Security Issue'),
                            description=issue.get('message', ''),
                            affected_file=str(file_path),
                            line_number=issue.get('start', {}).get('line', 0),
                            column_number=issue.get('col', 0),
                            recommendation=issue.get('fix', ''),
                            tool="semgrep"
                        )
                        issues.append(security_finding)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse semgrep output: {result.output}")
                    return AnalysisResult(
                        tool="semgrep",
                        success=False,
                        error="Failed to parse semgrep output"
                    )

                return AnalysisResult(
                    tool="semgrep",
                    success=True,
                    duration_ms=0,
                    issues=issues
                )

        except Exception as e:
            return AnalysisResult(
                tool="semgrep",
                success=False,
                error=f"Semgrep scan failed: {e}"
            )

    async def _run_general_security_scan(self, file_path: Path) -> AnalysisResult:
        """Run general security scan using available tools."""
        findings = []

        # Check for hardcoded secrets (basic pattern)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

                # Common secret patterns
                secret_patterns = [
                    r'(password\s*=\s*[\'"]\s*\w*(\w\s*[\'"]\s*)',
                    r'(api[_-]?key\s*=\s*[\'"]\s*\w*(\w\s*[\'"]\s*)',
                    r'(token\s*=\s*[\'"]\s*\w*(\w\s*[\'"]\s*)',
                    r'(secret\s*=\s*[\'"]\s*\w*(\w\s*[\'"]\s*)'
                ]

                line_number = 0
                for pattern in secret_patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        for line_match in content[match.start():match.end()].split('\n'):
                            if re.search(pattern, line_match):
                                security_finding = SecurityFinding(
                                    cve_id="SEC-001",
                                    severity="high",
                                    title="Hardcoded secret detected",
                                    description=f"Potential hardcoded secret found at line {line_num}",
                                    affected_file=str(file_path),
                                    line_number=line_num,
                                    column_number=match.start() - content[:match.start()].count('\n') + 1,
                                    recommendation="Remove hardcoded secrets and use environment variables",
                                    tool="security_analyzer"
                                )
                                findings.append(security_finding)

                return AnalysisResult(
                    tool=self.tool_name,
                    success=True,
                    duration_ms=0,
                    issues=findings
                )

        except Exception as e:
            return AnalysisResult(
                tool=self.tool_name,
                success=False,
                error=f"Security scan failed: {e}"
            )

    def _map_bandit_severity(self, severity: str) -> str:
        """Map bandit severity levels."""
        severity_map = {
            'low': 'info',
            'medium': 'warning',
            'high': 'error'
        }
        return severity_map.get(severity, 'info')

    def _map_semgrep_severity(self, metadata: Dict[str, Any], severity: str) -> str:
        """Map semgrep severity levels."""
        severity_map = {
            'INFO': 'info',
            'WARNING': 'warning',
            'ERROR': 'error'
        }
        # Handle severity overrides in metadata
        if metadata and 'issue' in metadata and 'severity' in metadata['issue']:
            return metadata['issue']['severity']
        return severity_map.get(severity, 'info')

    async def _execute_security_command(self, command: List[str], file_path: Path) -> AnalysisResult:
        """Execute security analysis command."""
        try:
            # Create temporary directory for output
            os.makedirs('/tmp', exist_ok=True)

            # Execute command
            process = await asyncio.create_subprocess_exec(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=file_path.parent
            )

            stdout, stderr = await process.communicate()
            return_code = process.returncode

            # Clean up
            if os.path.exists('/tmp/safety_output.json'):
                os.remove('/tmp/safety_output.json')
            if os.path.exists('/tmp/semgrep_output.json'):
                os.remove('/tmp/semgrep_output.json')

            return AnalysisResult(
                tool=command[0],
                success=return_code == 0,
                duration_ms=0,
                output=stdout.decode('utf-8') if stdout else '',
                issues=[]
            )

        except Exception as e:
            return AnalysisResult(
                tool=self.tool_name,
                success=False,
                error=f"Failed to execute {command[0]}: {e}"
            )


class SecurityScanResult:
    """Model for security scan results."""
    def __init__(
        self,
        repository_id: int,
        repository_name: str,
        scan_date: str,
        tools_used: List[str],
        total_findings: int,
        findings_by_severity: Dict[str, int],
        scan_duration_ms: int,
        error_message: Optional[str] = None
    ):
        self.repository_id = repository_id
        self.repository_name = repository_name
        self.scan_date = scan_date
        self.tools_used = tools_used
        self.total_findings = total_findings
        self.findings_by_severity = findings_by_severity
        self.scan_duration_ms = scan_duration_ms
        self.error_message = error_message

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        return {
            "repository_id": self.repository_id,
            "repository_name": self.repository_name,
            "scan_date": self.scan_date,
            "tools_used": self.tools_used,
            "total_findings": self.total_findings,
            "findings_by_severity": self.findings_by_severity,
            "scan_duration_ms": self.scan_duration_ms,
            "error_message": self.error_message
        }