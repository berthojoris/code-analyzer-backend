"""
Linting analysis endpoints.
"""
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.database import get_db_session
from core.database.models import LintingIssue, Repository
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["linting"])


class LintingIssueResponse(BaseModel):
    """Response model for linting issues."""
    id: int
    repository_id: int
    file_path: str
    line_number: int
    column_number: int
    rule_id: str
    severity: str
    message: str
    tool: str
    category: str
    suggestion: Optional[str] = None


class LintingIssuesResponse(BaseModel):
    """Response model for collection of linting issues."""
    repository_id: int
    repository_name: str
    total_issues: int
    error_count: int
    warning_count: int
    info_count: int
    issues: List[LintingIssueResponse]


def get_repository_by_owner_repo(owner: str, repo_name: str, db_session):
    """Get repository by owner and repo name from GitHub URL."""
    repositories = db_session.query(Repository).all()
    
    for repo in repositories:
        if repo.url and f"github.com/{owner}/{repo_name}" in repo.url:
            return repo
    
    return None


@router.get("/linting/{owner}/{repo_name}", response_model=LintingIssuesResponse)
async def get_linting_issues(
    owner: str,
    repo_name: str,
    severity: Optional[str] = Query(None, description="Filter by severity"),
    tool: Optional[str] = Query(None, description="Filter by tool"),
    file_path: Optional[str] = Query(None, description="Filter by file path"),
    db_session = Depends(get_db_session)
):
    """Get all linting issues for a repository with optional filters."""
    repository = get_repository_by_owner_repo(owner, repo_name, db_session)
    if not repository:
        raise HTTPException(status_code=404, detail="Repository not found")

    query = db_session.query(LintingIssue).filter_by(repository_id=repository.id)

    # Apply filters
    if severity:
        query = query.filter_by(severity=severity)
    if tool:
        query = query.filter_by(tool=tool)
    if file_path:
        query = query.filter(LintingIssue.file_path.like(f"%{file_path}%"))

    # Get all issues
    issues = query.all()

    # Count severity levels
    error_count = len([issue for issue in issues if issue.severity == 'error'])
    warning_count = len([issue for issue in issues if issue.severity == 'warning'])
    info_count = len([issue for issue in issues if issue.severity == 'info'])

    # Convert to response format
    issue_responses = [
        LintingIssueResponse(
            id=issue.id,
            repository_id=issue.repository_id,
            file_path=issue.file_path,
            line_number=issue.line_number,
            column_number=issue.column_number,
            rule_id=issue.rule_id,
            severity=issue.severity,
            message=issue.message,
            tool=issue.tool,
            category=issue.category,
            suggestion=issue.suggestion
        )
        for issue in issues
    ]

    return LintingIssuesResponse(
        repository_id=repository.id,
        repository_name=repository.name,
        total_issues=len(issues),
        error_count=error_count,
        warning_count=warning_count,
        info_count=info_count,
        issues=issue_responses
    )


@router.get("/linting/{owner}/{repo_name}/file/{file_path:path}", response_model=List[LintingIssueResponse])
async def get_file_linting_issues(
    owner: str,
    repo_name: str,
    file_path: str,
    db_session = Depends(get_db_session)
):
    """Get linting issues for a specific file."""
    repository = get_repository_by_owner_repo(owner, repo_name, db_session)
    if not repository:
        raise HTTPException(status_code=404, detail="Repository not found")

    issues = db_session.query(LintingIssue).filter_by(
        repository_id=repository.id,
        file_path=file_path
    ).all()

    return [
        LintingIssueResponse(
            id=issue.id,
            repository_id=issue.repository_id,
            file_path=issue.file_path,
            line_number=issue.line_number,
            column_number=issue.column_number,
            rule_id=issue.rule_id,
            severity=issue.severity,
            message=issue.message,
            tool=issue.tool,
            category=issue.category,
            suggestion=issue.suggestion
        )
        for issue in issues
    ]


@router.get("/linting/{owner}/{repo_name}/severity/{severity}", response_model=List[LintingIssueResponse])
async def get_issues_by_severity(
    owner: str,
    repo_name: str,
    severity: str,
    db_session = Depends(get_db_session)
):
    """Get linting issues filtered by severity."""
    repository = get_repository_by_owner_repo(owner, repo_name, db_session)
    if not repository:
        raise HTTPException(status_code=404, detail="Repository not found")

    issues = db_session.query(LintingIssue).filter_by(
        repository_id=repository.id,
        severity=severity
    ).all()

    return [
        LintingIssueResponse(
            id=issue.id,
            repository_id=issue.repository_id,
            file_path=issue.file_path,
            line_number=issue.line_number,
            column_number=issue.column_number,
            rule_id=issue.rule_id,
            severity=issue.severity,
            message=issue.message,
            tool=issue.tool,
            category=issue.category,
            suggestion=issue.suggestion
        )
        for issue in issues
    ]