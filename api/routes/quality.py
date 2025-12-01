"""
Quality metrics endpoints.
"""
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.database import get_db_session
from core.database.models import QualityMetric, Repository
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/quality", tags=["quality"])


class QualityMetricResponse(BaseModel):
    """Response model for quality metrics."""
    id: int
    repository_id: int
    file_path: str
    function_name: Optional[str]
    cyclomatic_complexity: int
    cognitive_complexity: int
    maintainability_index: float
    technical_debt_ratio: float
    code_smells_count: int
    lines_of_code: int
    comment_lines: int
    blank_lines: int
    language: str
    analysis_date: str


class QualityOverviewResponse(BaseModel):
    """Response model for quality overview."""
    repository_id: int
    repository_name: str
    total_files: int
    analyzed_files: int
    avg_cyclomatic_complexity: float
    avg_maintainability_index: float
    max_complexity: int
    max_maintainability: float
    total_code_lines: int
    total_comment_lines: int
    total_blank_lines: int
    quality_distribution: Dict[str, Any]


@router.get("/{repository_id}", response_model=QualityOverviewResponse)
async def get_quality_metrics(repository_id: int, db_session = Depends(get_db_session)):
    """Get quality metrics overview for a repository."""
    repository = db_session.query(Repository).filter_by(id=repository_id).first()
    if not repository:
        raise HTTPException(status_code=404, detail="Repository not found")

    metrics = db_session.query(QualityMetric).filter_by(repository_id=repository_id).all()

    if not metrics:
        return QualityOverviewResponse(
            repository_id=repository_id,
            repository_name=repository.name,
            total_files=0,
            analyzed_files=0,
            avg_cyclomatic_complexity=0.0,
            avg_maintainability_index=100.0,
            max_complexity=0,
            max_maintainability=100.0,
            total_code_lines=0,
            total_comment_lines=0,
            total_blank_lines=0,
            quality_distribution={}
        )

    # Calculate aggregate metrics
    total_files = len(set(m.file_path for m in metrics))
    analyzed_files = len(metrics)

    # Average complexity and maintainability
    complexities = [m.cyclomatic_complexity for m in metrics if m.cyclomatic_complexity]
    maintainability_scores = [m.maintainability_index for m in metrics if m.maintainability_index]

    avg_complexity = sum(complexities) / len(complexities) if complexities else 0.0
    avg_maintainability = sum(maintainability_scores) / len(maintainability_scores) if maintainability_scores else 100.0

    # Maximum values
    max_complexity = max(complexities) if complexities else 0
    max_maintainability = max(maintainability_scores) if maintainability_scores else 100.0

    # Totals
    total_code_lines = sum(m.lines_of_code for m in metrics if m.lines_of_code)
    total_comment_lines = sum(m.comment_lines for m in metrics if m.comment_lines)
    total_blank_lines = sum(m.blank_lines for m in metrics if m.blank_lines)

    # Quality distribution
    quality_distribution = {
        "low_complexity_files": len([m for m in metrics if m.cyclomatic_complexity <= 5]),
        "medium_complexity_files": len([m for m in metrics if 5 < m.cyclomatic_complexity <= 10]),
        "high_complexity_files": len([m for m in metrics if m.cyclomatic_complexity > 10]),
        "excellent_maintainability": len([m for m in metrics if m.maintainability_index >= 85]),
        "good_maintainability": len([m for m in metrics if 70 <= m.maintainability_index < 85]),
        "moderate_maintainability": len([m for m in metrics if 50 <= m.maintainability_index < 70]),
        "poor_maintainability": len([m for m in metrics if m.maintainability_index < 50]),
        "total_code_smells": sum(m.code_smells_count for m in metrics if m.code_smells_count),
        "avg_technical_debt": sum(m.technical_debt_ratio for m in metrics if m.technical_debt_ratio) / len(metrics)
    }

    return QualityOverviewResponse(
        repository_id=repository_id,
        repository_name=repository.name,
        total_files=total_files,
        analyzed_files=analyzed_files,
        avg_cyclomatic_complexity=round(avg_complexity, 2),
        avg_maintainability_index=round(avg_maintainability, 2),
        max_complexity=max_complexity,
        max_maintainability=max_maintainability,
        total_code_lines=total_code_lines,
        total_comment_lines=total_comment_lines,
        total_blank_lines=total_blank_lines,
        quality_distribution=quality_distribution
    )


@router.get("/{repository_id}/complexity", response_model=List[QualityMetricResponse])
async def get_complexity_metrics(repository_id: int, db_session = Depends(get_db_session)):
    """Get complexity metrics for a repository."""
    repository = db_session.query(Repository).filter_by(id=repository_id).first()
    if not repository:
        raise HTTPException(status_code=404, detail="Repository not found")

    metrics = db_session.query(QualityMetric).filter_by(repository_id=repository_id).all()

    return [
        QualityMetricResponse(
            id=m.id,
            repository_id=m.repository_id,
            file_path=m.file_path,
            function_name=m.function_name,
            cyclomatic_complexity=m.cyclomatic_complexity,
            cognitive_complexity=m.cognitive_complexity,
            maintainability_index=m.maintainability_index,
            technical_debt_ratio=m.technical_debt_ratio or 0.0,
            code_smells_count=m.code_smells_count,
            lines_of_code=m.lines_of_code,
            comment_lines=m.comment_lines,
            blank_lines=m.blank_lines,
            language=m.language,
            analysis_date=m.analysis_date.isoformat()
        )
        for m in metrics if m.cyclomatic_complexity and m.cyclomatic_complexity > 0
    ]


@router.get("/{repository_id}/maintainability", response_model=List[QualityMetricResponse])
async def get_maintainability_metrics(repository_id: int, db_session = Depends(get_db_session)):
    """Get maintainability metrics for a repository."""
    repository = db_session.query(Repository).filter_by(id=repository_id).first()
    if not repository:
        raise HTTPException(status_code=404, detail="Repository not found")

    metrics = db_session.query(QualityMetric).filter_by(repository_id=repository_id).all()

    return [
        QualityMetricResponse(
            id=m.id,
            repository_id=m.repository_id,
            file_path=m.file_path,
            function_name=m.function_name,
            cyclomatic_complexity=m.cyclomatic_complexity,
            cognitive_complexity=m.cognitive_complexity,
            maintainability_index=m.maintainability_index,
            technical_debt_ratio=m.technical_debt_ratio or 0.0,
            code_smells_count=m.code_smells_count,
            lines_of_code=m.lines_of_code,
            comment_lines=m.comment_lines,
            blank_lines=m.blank_lines,
            language=m.language,
            analysis_date=m.analysis_date.isoformat()
        )
        for m in metrics if m.maintainability_index is not None
    ]


@router.get("/{repository_id}/trends", response_model=Dict[str, Any])
async def get_quality_trends(
    repository_id: int,
    db_session = Depends(get_db_session)
):
    """Get quality trends over time for a repository."""
    repository = db_session.query(Repository).filter_by(id=repository_id).first()
    if not repository:
        raise HTTPException(status_code=404, detail="Repository not found")

    # This would normally query historical data
    # For now, return current metrics with trend indicators
    metrics = db_session.query(QualityMetric).filter_by(repository_id=repository_id).all()

    if not metrics:
        return {"message": "No quality metrics available for trends analysis"}

    # Calculate trends (simplified)
    latest_metrics = sorted(metrics, key=lambda m: m.analysis_date, reverse=True)[:10]

    trends = {
        "total_analyses": len(metrics),
        "recent_complexity_trend": [m.cyclomatic_complexity for m in latest_metrics if m.cyclomatic_complexity],
        "recent_maintainability_trend": [m.maintainability_index for m in latest_metrics if m.maintainability_index],
        "analysis_dates": [m.analysis_date.isoformat() for m in latest_metrics]
    }

    return trends