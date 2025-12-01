"""
Main security scanner orchestrator
Coordinates multiple security scanning tools
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import tempfile
import shutil
from pathlib import Path

from .bandit_scanner import BanditScanner
from .safety_scanner import SafetyScanner
from .semgrep_scanner import SemgrepScanner
from utils.logger import get_logger

logger = get_logger(__name__)


class Severity(Enum):
    """Security vulnerability severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityIssue:
    """Represents a security vulnerability"""
    tool_name: str
    file_path: str
    line_number: int
    column_number: Optional[int]
    severity: Severity
    confidence: str
    issue_type: str
    issue_id: str
    title: str
    message: str
    cwe_id: Optional[str] = None
    owasp_category: Optional[str] = None
    references: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.references is None:
            self.references = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert security issue to dictionary"""
        return {
            "tool_name": self.tool_name,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "column_number": self.column_number,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "issue_type": self.issue_type,
            "issue_id": self.issue_id,
            "title": self.title,
            "message": self.message,
            "cwe_id": self.cwe_id,
            "owasp_category": self.owasp_category,
            "references": self.references,
            "metadata": self.metadata
        }


@dataclass
class SecurityReport:
    """Complete security scan report"""
    repo_id: str
    scan_id: str
    timestamp: str
    total_issues: int
    issues_by_severity: Dict[str, int]
    issues: List[SecurityIssue]
    scan_summary: Dict[str, Any]
    tool_reports: Dict[str, Any]
    scan_duration: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert security report to dictionary"""
        return {
            "repo_id": self.repo_id,
            "scan_id": self.scan_id,
            "timestamp": self.timestamp,
            "total_issues": self.total_issues,
            "issues_by_severity": self.issues_by_severity,
            "issues": [issue.to_dict() for issue in self.issues],
            "scan_summary": self.scan_summary,
            "tool_reports": self.tool_reports,
            "scan_duration": self.scan_duration
        }


class SecurityScanner:
    """Main security scanner that orchestrates multiple tools"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize security scanner with configuration"""
        self.config = config or {}
        self.bandit_scanner = BanditScanner(config.get('bandit', {}))
        self.safety_scanner = SafetyScanner(config.get('safety', {}))
        self.semgrep_scanner = SemgrepScanner(config.get('semgrep', {}))

        # Configure enabled scanners
        self.enabled_scanners = {
            'bandit': self.bandit_scanner,
            'safety': self.safety_scanner,
            'semgrep': self.semgrep_scanner
        }

        if 'enabled_tools' in self.config:
            self.enabled_scanners = {
                name: scanner for name, scanner in self.enabled_scanners.items()
                if name in self.config['enabled_tools']
            }

    async def scan_repository(self, repo_path: str, repo_id: str) -> SecurityReport:
        """
        Perform comprehensive security scan on a repository

        Args:
            repo_path: Path to the repository to scan
            repo_id: Repository ID for tracking

        Returns:
            SecurityReport with all findings
        """
        import time
        import uuid
        from datetime import datetime

        scan_id = str(uuid.uuid4())
        start_time = time.time()

        logger.info(f"Starting security scan for repo {repo_id} with scan_id {scan_id}")

        all_issues = []
        tool_reports = {}
        issues_by_severity = {level.value: 0 for level in Severity}

        # Run all enabled scanners
        scanner_tasks = []
        for scanner_name, scanner in self.enabled_scanners.items():
            if scanner_name == 'safety':
                # Safety scans dependencies, not source files
                continue
            task = self._run_scanner(scanner_name, scanner, repo_path)
            scanner_tasks.append(task)

        # Run scanners concurrently
        scanner_results = await asyncio.gather(*scanner_tasks, return_exceptions=True)

        # Process scanner results
        for i, result in enumerate(scanner_results):
            scanner_name = list(self.enabled_scanners.keys())[i]
            if isinstance(result, Exception):
                logger.error(f"Scanner {scanner_name} failed: {result}")
                tool_reports[scanner_name] = {
                    "status": "failed",
                    "error": str(result)
                }
                continue

            issues = result
            all_issues.extend(issues)

            # Update severity counts
            for issue in issues:
                issues_by_severity[issue.severity.value] += 1

            tool_reports[scanner_name] = {
                "status": "completed",
                "issues_found": len(issues),
                "severity_breakdown": {
                    severity.value: sum(1 for issue in issues
                                     if issue.severity == severity)
                    for severity in Severity
                }
            }

        # Run dependency scanning with Safety
        if 'safety' in self.enabled_scanners:
            try:
                dependency_issues = await self.safety_scanner.scan_dependencies(repo_path)
                all_issues.extend(dependency_issues)

                for issue in dependency_issues:
                    issues_by_severity[issue.severity.value] += 1

                tool_reports['safety'] = {
                    "status": "completed",
                    "issues_found": len(dependency_issues),
                    "severity_breakdown": {
                        severity.value: sum(1 for issue in dependency_issues
                                         if issue.severity == severity)
                        for severity in Severity
                    }
                }
            except Exception as e:
                logger.error(f"Safety scanner failed: {e}")
                tool_reports['safety'] = {
                    "status": "failed",
                    "error": str(e)
                }

        end_time = time.time()
        scan_duration = end_time - start_time

        # Create scan summary
        scan_summary = {
            "scan_id": scan_id,
            "repo_path": repo_path,
            "tools_used": list(self.enabled_scanners.keys()),
            "high_risk_files": self._get_high_risk_files(all_issues),
            "top_vulnerability_types": self._get_top_vulnerability_types(all_issues),
            "recommendations": self._generate_recommendations(all_issues)
        }

        # Create and return security report
        report = SecurityReport(
            repo_id=repo_id,
            scan_id=scan_id,
            timestamp=datetime.utcnow().isoformat(),
            total_issues=len(all_issues),
            issues_by_severity=issues_by_severity,
            issues=all_issues,
            scan_summary=scan_summary,
            tool_reports=tool_reports,
            scan_duration=scan_duration
        )

        logger.info(f"Security scan completed for repo {repo_id}. "
                   f"Found {len(all_issues)} issues in {scan_duration:.2f}s")

        return report

    async def _run_scanner(self, scanner_name: str, scanner, repo_path: str) -> List[SecurityIssue]:
        """Run a specific scanner and return issues"""
        logger.info(f"Running {scanner_name} scanner...")

        try:
            if scanner_name == 'bandit':
                return await scanner.scan_python_files(repo_path)
            elif scanner_name == 'semgrep':
                return await scanner.scan_repository(repo_path)
            else:
                logger.warning(f"Unknown scanner: {scanner_name}")
                return []
        except Exception as e:
            logger.error(f"Error running {scanner_name} scanner: {e}")
            raise

    def _get_high_risk_files(self, issues: List[SecurityIssue]) -> List[Dict[str, Any]]:
        """Identify files with the most security issues"""
        file_issue_counts = {}
        for issue in issues:
            file_path = issue.file_path
            if file_path not in file_issue_counts:
                file_issue_counts[file_path] = []
            file_issue_counts[file_path].append(issue)

        high_risk_files = []
        for file_path, file_issues in file_issue_counts.items():
            if len(file_issues) >= 3:  # Files with 3+ issues
                high_risk_files.append({
                    "file_path": file_path,
                    "issue_count": len(file_issues),
                    "high_severity_count": sum(1 for issue in file_issues
                                              if issue.severity in [Severity.HIGH, Severity.CRITICAL]),
                    "vulnerability_types": list(set(issue.issue_type for issue in file_issues))
                })

        return sorted(high_risk_files,
                      key=lambda x: x["issue_count"],
                      reverse=True)[:10]  # Top 10 high-risk files

    def _get_top_vulnerability_types(self, issues: List[SecurityIssue]) -> List[Dict[str, Any]]:
        """Get most common vulnerability types"""
        type_counts = {}
        for issue in issues:
            issue_type = issue.issue_type
            if issue_type not in type_counts:
                type_counts[issue_type] = []
            type_counts[issue_type].append(issue)

        top_types = []
        for issue_type, type_issues in type_counts.items():
            top_types.append({
                "issue_type": issue_type,
                "count": len(type_issues),
                "high_severity_count": sum(1 for issue in type_issues
                                         if issue.severity in [Severity.HIGH, Severity.CRITICAL])
            })

        return sorted(top_types,
                      key=lambda x: x["count"],
                      reverse=True)[:10]  # Top 10 vulnerability types

    def _generate_recommendations(self, issues: List[SecurityIssue]) -> List[str]:
        """Generate security improvement recommendations"""
        recommendations = []

        # Analyze issue patterns
        severity_counts = {level.value: 0 for level in Severity}
        for issue in issues:
            severity_counts[issue.severity.value] += 1

        # High severity recommendations
        if severity_counts[Severity.CRITICAL.value] > 0:
            recommendations.append(
                "CRITICAL vulnerabilities found. Address immediately before deployment."
            )

        if severity_counts[Severity.HIGH.value] > 5:
            recommendations.append(
                "Multiple HIGH severity vulnerabilities detected. "
                "Prioritize security fixes and consider security audit."
            )

        # Tool-specific recommendations
        issue_types = set(issue.issue_type for issue in issues)

        if any("injection" in issue_type.lower() for issue_type in issue_types):
            recommendations.append(
                "Injection vulnerabilities detected. Use parameterized queries and input validation."
            )

        if any("hardcoded" in issue_type.lower() for issue_type in issue_types):
            recommendations.append(
                "Hardcoded credentials detected. Use environment variables or secure vaults."
            )

        if any("crypto" in issue_type.lower() for issue_type in issue_types):
            recommendations.append(
                "Cryptographic issues found. Use well-vetted cryptographic libraries."
            )

        # General recommendations
        if len(issues) > 20:
            recommendations.append(
                "High number of security issues. Consider implementing secure coding practices and regular security reviews."
            )

        return recommendations[:10]  # Limit to top 10 recommendations

    def get_enabled_tools(self) -> List[str]:
        """Get list of enabled security scanning tools"""
        return list(self.enabled_scanners.keys())