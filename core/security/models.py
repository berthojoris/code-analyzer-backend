"""
Security scanner models and shared classes
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


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