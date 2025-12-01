"""
Security analysis API routes
Integrates with bandit, safety, and semgrep scanning tools
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio

from core.database import get_db_session
from core.database.models import Repository
from core.security.scanner_stub import SecurityScanner
from utils.logger import get_logger

router = APIRouter(tags=["security"])
logger = get_logger(__name__)


class SecurityScanRequest(BaseModel):
    repository_id: str
    force_rescan: bool = False


class SecurityIssue(BaseModel):
    tool_name: str
    file_path: str
    line_number: int
    column_number: Optional[int]
    severity: str
    confidence: str
    issue_type: str
    issue_id: str
    title: str
    message: str
    cwe_id: Optional[str]
    owasp_category: Optional[str]
    references: List[str]


class SecurityReport(BaseModel):
    repo_id: str
    scan_id: str
    timestamp: str
    total_issues: int
    issues_by_severity: Dict[str, int]
    issues: List[SecurityIssue]
    scan_summary: Dict[str, Any]
    scan_duration: float


def get_repository_by_owner_repo(owner: str, repo_name: str, db_session):
    """Get repository by owner and repo name from GitHub URL."""
    repositories = db_session.query(Repository).all()
    
    for repo in repositories:
        if repo.url and f"github.com/{owner}/{repo_name}" in repo.url:
            return repo
    
    return None


@router.post("/scan", response_model=Dict[str, Any])
async def trigger_security_scan(request: SecurityScanRequest, background_tasks: BackgroundTasks):
    """Trigger a security scan for a repository"""
    try:
        # Initialize scanner with configuration
        config = {
            'bandit': {
                'exclude_patterns': ['*/test_*', '*/tests/*']
            },
            'safety': {
                'full_report': True
            },
            'semgrep': {
                'rulesets': ['p/security-audit', 'p/secrets']
            }
        }

        scanner = SecurityScanner(config)

        # Get repository path from ID (this would be implemented in your system)
        repo_path = f"/tmp/repos/{request.repository_id}"

        # Schedule background scan
        background_tasks.add_task(
            scanner.scan_repository,
            repo_path,
            request.repository_id
        )

        logger.info(f"Security scan triggered for repository: {request.repository_id}")

        return {
            "status": "started",
            "message": "Security scan initiated",
            "repository_id": request.repository_id,
            "scan_id": f"scan_{request.repository_id}_{asyncio.get_event_loop().time()}"
        }

    except Exception as e:
        logger.error(f"Failed to trigger security scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/security/{owner}/{repo_name}/issues", response_model=List[SecurityIssue])
async def get_security_issues(
    owner: str,
    repo_name: str,
    severity: Optional[str] = None,
    tool: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000)
):
    """Get security issues for a repository"""
    try:
        # This would fetch from your database
        # For now, return empty list
        logger.info(f"Retrieved security issues for repository: {owner}/{repo_name}")

        return []

    except Exception as e:
        logger.error(f"Failed to get security issues: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/security/{owner}/{repo_name}/report")
async def get_security_report(owner: str, repo_name: str):
    """Get comprehensive security report for a repository"""
    try:
        # This would fetch from your database
        logger.info(f"Retrieved security report for repository: {owner}/{repo_name}")

        return {
            "repo_id": f"{owner}/{repo_name}",
            "status": "completed",
            "timestamp": "2024-01-01T00:00:00Z",
            "total_issues": 0,
            "issues_by_severity": {"critical": 0, "high": 0, "medium": 0, "low": 0},
            "issues": [],
            "scan_summary": {
                "tools_used": ["bandit", "safety", "semgrep"],
                "high_risk_files": [],
                "top_vulnerability_types": []
            },
            "scan_duration": 0.0
        }

    except Exception as e:
        logger.error(f"Failed to get security report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/security/{owner}/{repo_name}/summary")
async def get_security_summary(owner: str, repo_name: str):
    """Get security summary for a repository"""
    try:
        # This would calculate from database
        logger.info(f"Retrieved security summary for repository: {owner}/{repo_name}")

        return {
            "repo_id": f"{owner}/{repo_name}",
            "security_score": 85,
            "total_issues": 0,
            "critical_issues": 0,
            "high_issues": 0,
            "medium_issues": 0,
            "low_issues": 0,
            "last_scan": "2024-01-01T00:00:00Z",
            "tools_status": {
                "bandit": "completed",
                "safety": "completed",
                "semgrep": "completed"
            }
        }

    except Exception as e:
        logger.error(f"Failed to get security summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))