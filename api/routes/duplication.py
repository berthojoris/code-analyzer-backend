"""
Duplicate code detection endpoints for analyzing code duplication and redundancy.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel

from core.database import get_db_session
from core.database.models import Repository
from core.duplication.detector import DuplicateCodeDetector, DuplicationType
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["duplication"])


class DuplicateBlockResponse(BaseModel):
    """Response model for duplicate code blocks."""
    file_path: str
    start_line: int
    end_line: int
    block_id: str
    content_hash: str
    similarity_score: float
    duplication_type: str
    complexity_score: Optional[float]
    line_count: int


class DuplicationGroupResponse(BaseModel):
    """Response model for duplicate code groups."""
    group_id: str
    duplication_type: str
    similarity_threshold: float
    blocks: List[DuplicateBlockResponse]
    total_lines: int
    total_files: int
    max_similarity: float
    min_similarity: float
    avg_similarity: float
    complexity_score: Optional[float]
    cluster_id: Optional[str]


class DuplicationOverviewResponse(BaseModel):
    """Response model for duplication overview."""
    repository_id: int
    repository_name: str
    total_files_scanned: int
    total_lines_scanned: int
    total_duplicate_lines: int
    duplication_percentage: float
    duplicate_groups: List[DuplicationGroupResponse]
    scan_id: str
    timestamp: str
    recommendations: List[str]


class DuplicationStatisticsResponse(BaseModel):
    """Response model for duplication statistics."""
    repository_id: int
    repository_name: str
    total_groups: int
    total_blocks: int
    exact_duplicates: int
    structural_duplicates: int
    logical_duplicates: int
    partial_duplicates: int
    average_similarity: float
    files_with_duplication: int
    languages_affected: List[str]


class FileStatisticsResponse(BaseModel):
    """Response model for file-level duplication statistics."""
    file_path: str
    total_lines: int
    duplicate_lines: int
    duplicate_blocks: int
    groups_involved: int
    duplication_percentage: float
    max_similarity: float


class LanguageStatisticsResponse(BaseModel):
    """Response model for language-level duplication statistics."""
    language: str
    total_groups: int
    total_blocks: int
    total_lines: int
    avg_similarity: float
    duplication_types: Dict[str, int]


def get_repository_by_owner_repo(owner: str, repo_name: str, db_session):
    """Get repository by owner and repo name from GitHub URL."""
    repositories = db_session.query(Repository).all()
    
    for repo in repositories:
        if repo.url and f"github.com/{owner}/{repo_name}" in repo.url:
            return repo
    
    return None


@router.get("/duplication/{owner}/{repo_name}", response_model=DuplicationOverviewResponse)
async def get_duplication_overview(
    owner: str,
    repo_name: str,
    min_similarity: float = Query(0.8, description="Minimum similarity threshold"),
    min_block_size: int = Query(10, description="Minimum block size in lines"),
    db_session = Depends(get_db_session)
):
    """Get duplicate code overview for a repository."""
    repository = get_repository_by_owner_repo(owner, repo_name, db_session)
    if not repository:
        raise HTTPException(status_code=404, detail="Repository not found")

    # Check if repository has been cloned
    if not repository.dominant_language:
        return DuplicationOverviewResponse(
            repository_id=repository.id,
            repository_name=repository.name,
            total_files_scanned=0,
            total_lines_scanned=0,
            total_duplicate_lines=0,
            duplication_percentage=0.0,
            duplicate_groups=[],
            scan_id="",
            timestamp=datetime.utcnow().isoformat(),
            recommendations=["Repository needs to be indexed first before duplication analysis."]
        )

    # In a real implementation, this would:
    # 1. Check cache for recent scan
    # 2. Load or trigger duplicate code detection
    # 3. Return the analysis results
    
    # Mock response with sample data based on repository metrics
    mock_groups = []
    
    if repository.total_files > 0:
        # Generate mock duplication data
        mock_groups = [
            DuplicationGroupResponse(
                group_id=f"group_{i}",
                duplication_type="exact" if i % 2 == 0 else "structural",
                similarity_threshold=min_similarity,
                blocks=[
                    DuplicateBlockResponse(
                        file_path=f"src/module{i}/file_{j}.py",
                        start_line=10,
                        end_line=25,
                        block_id=f"block_{i}_{j}",
                        content_hash=f"hash_{i}_{j}",
                        similarity_score=0.9 + (j * 0.05),
                        duplication_type="exact" if i % 2 == 0 else "structural",
                        complexity_score=5.0,
                        line_count=15
                    )
                    for j in range(3)
                ],
                total_lines=45,
                total_files=3,
                max_similarity=0.95,
                min_similarity=0.85,
                avg_similarity=0.9,
                complexity_score=5.0,
                cluster_id=f"cluster_{i}"
            )
            for i in range(min(5, repository.total_files // 10))
        ]
    
    total_duplicate_lines = sum(group.total_lines for group in mock_groups)
    total_lines_scanned = repository.total_lines or 1000
    duplication_percentage = (total_duplicate_lines / max(1, total_lines_scanned)) * 100
    
    return DuplicationOverviewResponse(
        repository_id=repository.id,
        repository_name=repository.name,
        total_files_scanned=repository.total_files or 0,
        total_lines_scanned=total_lines_scanned,
        total_duplicate_lines=total_duplicate_lines,
        duplication_percentage=duplication_percentage,
        duplicate_groups=mock_groups,
        scan_id=f"scan_{repository.id}_{datetime.utcnow().timestamp()}",
        timestamp=datetime.utcnow().isoformat(),
        recommendations=[
            "Consider extracting common utility functions to reduce duplication",
            "Review structural patterns that could be abstracted",
            "Implement code review guidelines to prevent new duplication"
        ]
    )


@router.get("/duplication/{owner}/{repo_name}/groups", response_model=List[DuplicationGroupResponse])
async def get_duplication_groups(
    owner: str,
    repo_name: str,
    min_similarity: float = Query(0.8, description="Minimum similarity threshold"),
    db_session = Depends(get_db_session)
):
    """Get all duplicate groups for a repository."""
    repository = get_repository_by_owner_repo(owner, repo_name, db_session)
    if not repository:
        raise HTTPException(status_code=404, detail="Repository not found")

    # Mock duplicate groups
    mock_groups = []
    
    if repository.total_files > 0:
        for i in range(min(10, repository.total_files // 5)):
            dup_type = "exact" if i % 3 == 0 else "structural" if i % 3 == 1 else "partial"
            mock_groups.append(
                DuplicationGroupResponse(
                    group_id=f"group_{i}",
                    duplication_type=dup_type,
                    similarity_threshold=min_similarity,
                    blocks=[
                        DuplicateBlockResponse(
                            file_path=f"src/components/feature_{j}.py" if j % 2 == 0 else f"lib/utils/module_{j}.py",
                            start_line=20 + (j * 10),
                            end_line=35 + (j * 10),
                            block_id=f"block_{i}_{j}",
                            content_hash=f"hash_{i}_{j}",
                            similarity_score=0.85 + (i * 0.03),
                            duplication_type=dup_type,
                            complexity_score=4.0 + (i * 0.5),
                            line_count=15
                        )
                        for j in range(2, 5)
                    ],
                    total_lines=45 * 3,
                    total_files=3,
                    max_similarity=0.95,
                    min_similarity=0.85,
                    avg_similarity=0.9,
                    complexity_score=5.0,
                    cluster_id=f"cluster_{i // 2}"
                )
            )
    
    return mock_groups


@router.get("/duplication/{owner}/{repo_name}/type/{duplication_type}", response_model=List[DuplicationGroupResponse])
async def get_duplication_by_type(
    owner: str,
    repo_name: str,
    duplication_type: str,
    db_session = Depends(get_db_session)
):
    """Get duplicate groups filtered by duplication type."""
    repository = get_repository_by_owner_repo(owner, repo_name, db_session)
    if not repository:
        raise HTTPException(status_code=404, detail="Repository not found")

    if duplication_type not in ["exact", "structural", "logical", "partial"]:
        raise HTTPException(status_code=400, detail="Invalid duplication type")

    # Mock filtered groups
    mock_groups = []
    
    if repository.total_files > 0:
        for i in range(min(5, repository.total_files // 10)):
            mock_groups.append(
                DuplicationGroupResponse(
                    group_id=f"group_{duplication_type}_{i}",
                    duplication_type=duplication_type,
                    similarity_threshold=0.8,
                    blocks=[
                        DuplicateBlockResponse(
                            file_path=f"src/{duplication_type}_{i}/file.py",
                            start_line=15,
                            end_line=30,
                            block_id=f"block_{duplication_type}_{i}_{j}",
                            content_hash=f"hash_{duplication_type}_{i}_{j}",
                            similarity_score=0.9,
                            duplication_type=duplication_type,
                            complexity_score=5.0,
                            line_count=15
                        )
                        for j in range(3)
                    ],
                    total_lines=45,
                    total_files=3,
                    max_similarity=0.9,
                    min_similarity=0.9,
                    avg_similarity=0.9,
                    complexity_score=5.0,
                    cluster_id=f"cluster_{duplication_type}"
                )
            )
    
    return mock_groups


@router.get("/duplication/{owner}/{repo_name}/statistics", response_model=DuplicationStatisticsResponse)
async def get_duplication_statistics(
    owner: str,
    repo_name: str,
    db_session = Depends(get_db_session)
):
    """Get duplication metrics and statistics."""
    repository = get_repository_by_owner_repo(owner, repo_name, db_session)
    if not repository:
        raise HTTPException(status_code=404, detail="Repository not found")

    # Mock statistics based on repository size
    total_groups = min(25, repository.total_files // 4) if repository.total_files > 0 else 0
    exact_duplicates = total_groups // 3
    structural_duplicates = total_groups // 3
    logical_duplicates = total_groups // 6
    partial_duplicates = total_groups - (exact_duplicates + structural_duplicates + logical_duplicates)
    
    return DuplicationStatisticsResponse(
        repository_id=repository.id,
        repository_name=repository.name,
        total_groups=total_groups,
        total_blocks=total_groups * 3,
        exact_duplicates=exact_duplicates,
        structural_duplicates=structural_duplicates,
        logical_duplicates=logical_duplicates,
        partial_duplicates=partial_duplicates,
        average_similarity=0.87,
        files_with_duplication=min(repository.total_files // 2, 50) if repository.total_files > 0 else 0,
        languages_affected=[repository.dominant_language] if repository.dominant_language else ["Python"]
    )


@router.get("/duplication/{owner}/{repo_name}/files", response_model=List[FileStatisticsResponse])
async def get_file_duplication_statistics(
    owner: str,
    repo_name: str,
    db_session = Depends(get_db_session)
):
    """Get file-level duplication statistics."""
    repository = get_repository_by_owner_repo(owner, repo_name, db_session)
    if not repository:
        raise HTTPException(status_code=404, detail="Repository not found")

    # Mock file statistics
    mock_files = []
    
    if repository.total_files > 0:
        for i in range(min(20, repository.total_files)):
            total_lines = 50 + (i * 10)
            duplicate_lines = total_lines // 3
            duplication_percentage = (duplicate_lines / max(1, total_lines)) * 100
            
            mock_files.append(
                FileStatisticsResponse(
                    file_path=f"src/module_{i}/file.py",
                    total_lines=total_lines,
                    duplicate_lines=duplicate_lines,
                    duplicate_blocks=2,
                    groups_involved=1,
                    duplication_percentage=duplication_percentage,
                    max_similarity=0.92
                )
            )
    
    return mock_files


@router.get("/duplication/{owner}/{repo_name}/languages", response_model=List[LanguageStatisticsResponse])
async def get_language_duplication_statistics(
    owner: str,
    repo_name: str,
    db_session = Depends(get_db_session)
):
    """Get language-level duplication statistics."""
    repository = get_repository_by_owner_repo(owner, repo_name, db_session)
    if not repository:
        raise HTTPException(status_code=404, detail="Repository not found")

    # Mock language statistics
    language = repository.dominant_language or "Python"
    
    return [
        LanguageStatisticsResponse(
            language=language,
            total_groups=15,
            total_blocks=45,
            total_lines=675,
            avg_similarity=0.88,
            duplication_types={
                "exact": 5,
                "structural": 6,
                "logical": 2,
                "partial": 2
            }
        )
    ]


@router.post("/duplication/{owner}/{repo_name}/scan")
async def trigger_duplicate_scan(
    owner: str,
    repo_name: str,
    background_tasks: BackgroundTasks,
    min_similarity: float = Query(0.8, description="Minimum similarity threshold"),
    min_block_size: int = Query(10, description="Minimum block size in lines"),
    db_session = Depends(get_db_session)
):
    """Trigger a duplicate code scan for a repository."""
    repository = get_repository_by_owner_repo(owner, repo_name, db_session)
    if not repository:
        raise HTTPException(status_code=404, detail="Repository not found")

    # In a real implementation, this would:
    # 1. Check if repository is cloned
    # 2. Trigger the duplicate code detector
    # 3. Store results in cache/database
    # 4. Return scan job ID
    
    scan_id = f"scan_{repository.id}_{datetime.utcnow().timestamp()}"
    
    return {
        "repository_id": repository.id,
        "repository_name": repository.name,
        "scan_id": scan_id,
        "status": "started",
        "message": "Duplicate code scan started",
        "estimated_completion": "3-5 minutes",
        "parameters": {
            "min_similarity": min_similarity,
            "min_block_size": min_block_size
        }
    }
