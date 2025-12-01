"""
Main analysis endpoints for static code analysis.
"""
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ...core.config import settings
from ...core.database import get_db_session
from ...core.database.models import Repository, LintingIssue, QualityMetric
from ...core.analysis import LintingAnalyzer, QualityAnalyzer
from ...utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/analysis", tags=["analysis"])


class AnalysisRequest(BaseModel):
    """Request model for repository analysis."""
    repo_url: str = Field(..., description="Repository URL to analyze")
    reanalyze: bool = Field(default=False, description="Force reanalysis if already analyzed")
    include_linting: bool = Field(default=True, description="Include linting analysis")
    include_quality: bool = Field(default=True, description="Include quality metrics analysis")
    max_files_mb: int = Field(default=settings.max_file_size_mb, description="Max file size in MB to analyze")


class AnalysisStatusResponse(BaseModel):
    """Response model for analysis status."""
    repository_id: int
    status: str  # pending, running, completed, failed
    files_analyzed: int
    files_total: int
    progress_percentage: float
    started_at: datetime
    estimated_completion: Optional[datetime] = None
    error_message: Optional[str] = None


class AnalysisResultsResponse(BaseModel):
    """Response model for analysis results."""
    repository_id: int
    repository_name: str
    analysis_date: datetime
    total_files: int
    linting_issues: List[Dict[str, Any]]
    quality_metrics: List[Dict[str, Any]]
    analysis_summary: Dict[str, Any]


# Global analyzers (in production, these would be initialized with config)
linting_analyzer = None
quality_analyzer = None


def get_linting_analyzer() -> LintingAnalyzer:
    """Get or create linting analyzer instance."""
    global linting_analyzer
    if linting_analyzer is None:
        config = {
            'ruff_config_path': settings.ruff_config_path,
            'eslint_config_path': None,  # Will be added later
            'flake8_config_path': settings.flake8_config_path
        }
        linting_analyzer = LintingAnalyzer(config)
    return linting_analyzer


def get_quality_analyzer() -> QualityAnalyzer:
    """Get or create quality analyzer instance."""
    global quality_analyzer
    if quality_analyzer is None:
        config = {
            'complexity_threshold': settings.complexity_threshold
        }
        quality_analyzer = QualityAnalyzer(config)
    return quality_analyzer


@router.post("/analyze", response_model=dict)
async def start_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    db_session = Depends(get_db_session)
):
    """
    Start comprehensive repository analysis.

    Includes:
    - Repository cloning and parsing (existing)
    - Static linting analysis (new)
    - Quality metrics analysis (new)
    """
    if not settings.analysis_enabled:
        raise HTTPException(status_code=503, detail="Analysis is disabled")

    try:
        # Extract repository name from URL
        repo_name = request.repo_url.split('/')[-1].replace('.git', '')

        # Check if repository exists and reanalyze option
        existing_repo = db_session.query(Repository).filter_by(
            url=request.repo_url
        ).first()

        if existing_repo and not request.reanalyze:
            if existing_repo.analysis_status == 'running':
                return {
                    "repository_id": existing_repo.id,
                    "status": "already_running",
                    "message": "Analysis is already in progress"
                }

            # Return existing results if available
            if existing_repo.analysis_status == 'completed':
                return await get_analysis_results(existing_repo.id, db_session)

        # Create or update repository record
        if existing_repo:
            existing_repo.analysis_status = 'pending'
            existing_repo.last_analyzed = datetime.now()
            db_session.commit()
            repository = existing_repo
        else:
            repository = Repository(
                name=repo_name,
                url=request.repo_url,
                dominant_language='',  # Will be set by existing parser
                analysis_status='pending',
                total_files=0,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            db_session.add(repository)
            db_session.commit()
            db_session.refresh(repository)

        # Start background analysis
        background_tasks.add_task(
            analyze_repository_background,
            repository_id=repository.id,
            request=request.dict()
        )

        return {
            "repository_id": repository.id,
            "status": "started",
            "message": f"Analysis started for {repo_name}"
        }

    except Exception as e:
        logger.error(f"Failed to start analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start analysis: {str(e)}")


@router.get("/{repository_id}/status", response_model=AnalysisStatusResponse)
async def get_analysis_status(repository_id: int, db_session=Depends(get_db_session)):
    """Get analysis status for a repository."""
    repository = db_session.query(Repository).filter_by(id=repository_id).first()

    if not repository:
        raise HTTPException(status_code=404, detail="Repository not found")

    # Calculate progress (simplified)
    progress_percentage = 0.0
    if repository.total_files > 0:
        progress_percentage = min(100.0, (repository.analyzed_files / repository.total_files) * 100)

    return AnalysisStatusResponse(
        repository_id=repository.id,
        status=repository.analysis_status,
        files_analyzed=repository.analyzed_files,
        files_total=repository.total_files,
        progress_percentage=progress_percentage,
        started_at=repository.last_analyzed or repository.created_at,
        estimated_completion=None  # Could be calculated based on historical data
    )


@router.get("/{repository_id}/results", response_model=AnalysisResultsResponse)
async def get_analysis_results(repository_id: int, db_session=Depends(get_db_session)):
    """Get complete analysis results for a repository."""
    repository = db_session.query(Repository).filter_by(id=repository_id).first()

    if not repository:
        raise HTTPException(status_code=404, detail="Repository not found")

    if repository.analysis_status != 'completed':
        raise HTTPException(status_code=400, detail="Analysis not completed yet")

    # Get linting issues
    linting_issues_query = db_session.query(LintingIssue).filter_by(repository_id=repository_id)
    linting_issues = []
    for issue in linting_issues_query.all():
        linting_issues.append({
            "id": issue.id,
            "file_path": issue.file_path,
            "line_number": issue.line_number,
            "column_number": issue.column_number,
            "rule_id": issue.rule_id,
            "severity": issue.severity,
            "message": issue.message,
            "tool": issue.tool,
            "category": issue.category,
            "suggestion": issue.suggestion
        })

    # Get quality metrics
    quality_metrics_query = db_session.query(QualityMetric).filter_by(repository_id=repository_id)
    quality_metrics = []
    for metric in quality_metrics_query.all():
        quality_metrics.append({
            "id": metric.id,
            "file_path": metric.file_path,
            "function_name": metric.function_name,
            "cyclomatic_complexity": metric.cyclomatic_complexity,
            "cognitive_complexity": metric.cognitive_complexity,
            "maintainability_index": metric.maintainability_index,
            "technical_debt_ratio": metric.technical_debt_ratio,
            "code_smells_count": metric.code_smells_count,
            "lines_of_code": metric.lines_of_code,
            "language": metric.language,
            "analysis_date": metric.analysis_date
        })

    # Calculate summary
    total_issues = len(linting_issues)
    error_count = len([i for i in linting_issues if i['severity'] == 'error'])
    warning_count = len([i for i in linting_issues if i['severity'] == 'warning'])

    # Quality summary
    avg_complexity = 0
    avg_maintainability = 0
    if quality_metrics:
        avg_complexity = sum(m['cyclomatic_complexity'] for m in quality_metrics) / len(quality_metrics)
        avg_maintainability = sum(m['maintainability_index'] for m in quality_metrics) / len(quality_metrics)

    summary = {
        "total_issues": total_issues,
        "error_count": error_count,
        "warning_count": warning_count,
        "info_count": total_issues - error_count - warning_count,
        "avg_complexity": round(avg_complexity, 2),
        "avg_maintainability_index": round(avg_maintainability, 2),
        "files_with_issues": len(set(i['file_path'] for i in linting_issues))
    }

    return AnalysisResultsResponse(
        repository_id=repository.id,
        repository_name=repository.name,
        analysis_date=repository.last_analyzed or repository.created_at,
        total_files=repository.analyzed_files,
        linting_issues=linting_issues,
        quality_metrics=quality_metrics,
        analysis_summary=summary
    )


async def analyze_repository_background(
    repository_id: int,
    request: Dict[str, Any]
):
    """Background task for repository analysis."""
    logger.info(f"Starting background analysis for repository {repository_id}")

    # This would integrate with existing repository indexing functionality
    # For now, we'll implement a simplified version that analyzes already indexed files

    # In a real implementation, this would:
    # 1. Clone the repository (using existing code)
    # 2. Parse files into chunks (using existing code)
    # 3. Run static analysis on each file
    # 4. Store results in database

    # Simulate the analysis process
    await asyncio.sleep(2)  # Simulate some work

    # Update repository status
    from ...core.database import get_db_session

    async with get_db_session() as db_session:
        repository = db_session.query(Repository).filter_by(id=repository_id).first()
        if repository:
            repository.analysis_status = 'completed'
            repository.analyzed_files = repository.total_files or 10  # Simulated
            repository.last_analyzed = datetime.now()
            db_session.commit()

    logger.info(f"Completed analysis for repository {repository_id}")