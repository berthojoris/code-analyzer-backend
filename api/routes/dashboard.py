"""
Dashboard API routes for repository analytics and metrics visualization.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.database import get_db_session
from core.database.models import Repository, LintingIssue, QualityMetric
from core.dashboard import MetricsAggregator, TrendAnalyzer
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["dashboard"])


class RepositoryDashboardResponse(BaseModel):
    """Response model for repository dashboard data."""
    repository_id: int
    repository_name: str
    owner: str
    repo_name: str
    analysis_status: str
    last_analyzed: Optional[datetime]
    total_files: int
    analyzed_files: int
    dominant_language: str
    health_score: float
    issue_summary: Dict[str, Any]
    quality_summary: Dict[str, Any]
    security_summary: Dict[str, Any]
    trend_data: Dict[str, Any]


class DashboardOverviewResponse(BaseModel):
    """Response model for dashboard overview."""
    total_repositories: int
    active_analyses: int
    total_issues: int
    avg_health_score: float
    language_distribution: Dict[str, int]
    recent_activity: List[Dict[str, Any]]


def parse_github_repo(repo_url: str) -> tuple[str, str]:
    """Parse GitHub repository URL to extract owner and repo name."""
    # Handle different GitHub URL formats
    if "github.com" in repo_url:
        parts = repo_url.split("github.com/")[-1].split("/")
        if len(parts) >= 2:
            owner = parts[0]
            repo_name = parts[1].replace(".git", "")
            return owner, repo_name
    return "", ""


def get_repository_by_owner_repo(owner: str, repo_name: str, db_session) -> Optional[Repository]:
    """Get repository by owner and repo name from GitHub URL."""
    # Try to find repository by matching URL pattern
    repositories = db_session.query(Repository).all()
    
    for repo in repositories:
        if repo.url and f"github.com/{owner}/{repo_name}" in repo.url:
            return repo
    
    return None


@router.get("/dashboard/{owner}/{repo_name}", response_model=RepositoryDashboardResponse)
async def get_repository_dashboard(
    owner: str,
    repo_name: str,
    db_session = Depends(get_db_session)
):
    """Get comprehensive dashboard data for a specific repository."""
    repository = get_repository_by_owner_repo(owner, repo_name, db_session)
    
    if not repository:
        raise HTTPException(status_code=404, detail="Repository not found")
    
    # Get linting issues summary
    linting_issues = db_session.query(LintingIssue).filter_by(repository_id=repository.id).all()
    
    issue_summary = {
        "total_issues": len(linting_issues),
        "error_count": len([issue for issue in linting_issues if issue.severity == 'error']),
        "warning_count": len([issue for issue in linting_issues if issue.severity == 'warning']),
        "info_count": len([issue for issue in linting_issues if issue.severity == 'info']),
        "files_with_issues": len(set(issue.file_path for issue in linting_issues))
    }
    
    # Get quality metrics summary
    quality_metrics = db_session.query(QualityMetric).filter_by(repository_id=repository.id).all()
    
    if quality_metrics:
        avg_complexity = sum(m.cyclomatic_complexity for m in quality_metrics if m.cyclomatic_complexity) / len(quality_metrics)
        avg_maintainability = sum(m.maintainability_index for m in quality_metrics if m.maintainability_index) / len(quality_metrics)
        max_complexity = max((m.cyclomatic_complexity for m in quality_metrics if m.cyclomatic_complexity), default=0)
    else:
        avg_complexity = 0.0
        avg_maintainability = 100.0
        max_complexity = 0
    
    quality_summary = {
        "avg_cyclomatic_complexity": round(avg_complexity, 2),
        "avg_maintainability_index": round(avg_maintainability, 2),
        "max_complexity": max_complexity,
        "total_code_lines": sum(m.lines_of_code for m in quality_metrics if m.lines_of_code),
        "total_files_analyzed": len(set(m.file_path for m in quality_metrics))
    }
    
    # Calculate health score (0-100)
    health_score = 100.0
    
    # Deduct points for issues
    if issue_summary['total_issues'] > 0:
        health_score -= min(30, issue_summary['error_count'] * 5 + issue_summary['warning_count'] * 2)
    
    # Deduct points for complexity
    if avg_complexity > 10:
        health_score -= 10
    elif avg_complexity > 5:
        health_score -= 5
    
    # Deduct points for maintainability
    if avg_maintainability < 50:
        health_score -= 15
    elif avg_maintainability < 70:
        health_score -= 10
    
    health_score = max(0.0, min(100.0, health_score))
    
    # Security summary (placeholder - would integrate with security scanner)
    security_summary = {
        "total_vulnerabilities": 0,
        "critical_vulnerabilities": 0,
        "high_vulnerabilities": 0,
        "medium_vulnerabilities": 0,
        "low_vulnerabilities": 0,
        "security_score": 100.0,
        "last_scan": repository.last_analyzed.isoformat() if repository.last_analyzed else None
    }
    
    # Trend data (placeholder - would use TrendAnalyzer)
    trend_data = {
        "complexity_trend": [],
        "maintainability_trend": [],
        "issue_trend": [],
        "dates": []
    }
    
    return RepositoryDashboardResponse(
        repository_id=repository.id,
        repository_name=repository.name,
        owner=owner,
        repo_name=repo_name,
        analysis_status=repository.analysis_status,
        last_analyzed=repository.last_analyzed,
        total_files=repository.total_files,
        analyzed_files=repository.analyzed_files,
        dominant_language=repository.dominant_language or "Unknown",
        health_score=round(health_score, 1),
        issue_summary=issue_summary,
        quality_summary=quality_summary,
        security_summary=security_summary,
        trend_data=trend_data
    )


@router.get("/dashboard/overview", response_model=DashboardOverviewResponse)
async def get_dashboard_overview(db_session = Depends(get_db_session)):
    """Get overall dashboard statistics and overview."""
    repositories = db_session.query(Repository).all()
    
    total_repositories = len(repositories)
    active_analyses = len([r for r in repositories if r.analysis_status == 'running'])
    
    # Count total issues across all repositories
    total_issues = db_session.query(LintingIssue).count()
    
    # Calculate average health score
    health_scores = []
    for repo in repositories:
        if repo.analysis_status == 'completed':
            # Simple health calculation
            issues = db_session.query(LintingIssue).filter_by(repository_id=repo.id).count()
            score = max(0, 100 - issues * 2)  # Deduct 2 points per issue
            health_scores.append(score)
    
    avg_health_score = sum(health_scores) / len(health_scores) if health_scores else 100.0
    
    # Language distribution
    language_dist = {}
    for repo in repositories:
        lang = repo.dominant_language or "Unknown"
        language_dist[lang] = language_dist.get(lang, 0) + 1
    
    # Recent activity (last 5 analyzed repositories)
    recent_repos = sorted(
        [r for r in repositories if r.last_analyzed],
        key=lambda x: x.last_analyzed,
        reverse=True
    )[:5]
    
    recent_activity = []
    for repo in recent_repos:
        recent_activity.append({
            "repository_name": repo.name,
            "analysis_date": repo.last_analyzed.isoformat(),
            "status": repo.analysis_status,
            "total_files": repo.total_files
        })
    
    return DashboardOverviewResponse(
        total_repositories=total_repositories,
        active_analyses=active_analyses,
        total_issues=total_issues,
        avg_health_score=round(avg_health_score, 1),
        language_distribution=language_dist,
        recent_activity=recent_activity
    )


@router.get("/dashboard/{owner}/{repo_name}/metrics")
async def get_repository_metrics(
    owner: str,
    repo_name: str,
    metric_type: str = Query("all", description="Type of metrics: all, quality, linting, security"),
    db_session = Depends(get_db_session)
):
    """Get specific metrics for a repository."""
    repository = get_repository_by_owner_repo(owner, repo_name, db_session)
    
    if not repository:
        raise HTTPException(status_code=404, detail="Repository not found")
    
    metrics = {}
    
    if metric_type in ["all", "quality"]:
        quality_metrics = db_session.query(QualityMetric).filter_by(repository_id=repository.id).all()
        metrics["quality"] = {
            "total_metrics": len(quality_metrics),
            "avg_complexity": sum(m.cyclomatic_complexity for m in quality_metrics if m.cyclomatic_complexity) / len(quality_metrics) if quality_metrics else 0,
            "avg_maintainability": sum(m.maintainability_index for m in quality_metrics if m.maintainability_index) / len(quality_metrics) if quality_metrics else 100,
            "files_analyzed": len(set(m.file_path for m in quality_metrics))
        }
    
    if metric_type in ["all", "linting"]:
        linting_issues = db_session.query(LintingIssue).filter_by(repository_id=repository.id).all()
        metrics["linting"] = {
            "total_issues": len(linting_issues),
            "by_severity": {
                "error": len([i for i in linting_issues if i.severity == "error"]),
                "warning": len([i for i in linting_issues if i.severity == "warning"]),
                "info": len([i for i in linting_issues if i.severity == "info"])
            },
            "files_with_issues": len(set(i.file_path for i in linting_issues))
        }
    
    if metric_type in ["all", "security"]:
        # Placeholder for security metrics
        metrics["security"] = {
            "total_vulnerabilities": 0,
            "by_severity": {"critical": 0, "high": 0, "medium": 0, "low": 0},
            "last_scan": repository.last_analyzed.isoformat() if repository.last_analyzed else None
        }
    
    return {
        "repository_id": repository.id,
        "repository_name": repository.name,
        "metric_type": metric_type,
        "metrics": metrics
    }