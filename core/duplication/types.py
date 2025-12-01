"""
Shared types and models for duplicate code detection.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class DuplicationType(Enum):
    """Types of code duplication"""
    EXACT = "exact"  # Identical code blocks
    STRUCTURAL = "structural"  # Similar structure, different content
    LOGICAL = "logical"  # Same logic, different implementation
    PARTIAL = "partial"  # Partial code duplication


@dataclass
class DuplicateBlock:
    """Represents a duplicated code block"""
    file_path: str
    start_line: int
    end_line: int
    block_id: str
    content_hash: str
    similarity_score: float
    duplication_type: DuplicationType
    complexity_score: Optional[float] = None
    line_count: Optional[int] = None

    def __post_init__(self):
        if self.line_count is None:
            self.line_count = self.end_line - self.start_line + 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert duplicate block to dictionary"""
        return {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "block_id": self.block_id,
            "content_hash": self.content_hash,
            "similarity_score": self.similarity_score,
            "duplication_type": self.duplication_type.value,
            "complexity_score": self.complexity_score,
            "line_count": self.line_count
        }


@dataclass
class DuplicationGroup:
    """Group of similar code blocks"""
    group_id: str
    duplication_type: DuplicationType
    similarity_threshold: float
    blocks: List[DuplicateBlock]
    total_lines: int
    total_files: int
    max_similarity: float
    min_similarity: float
    avg_similarity: float
    complexity_score: Optional[float] = None
    cluster_id: Optional[str] = None

    def __post_init__(self):
        if not self.blocks:
            return

        self.total_lines = sum(block.line_count for block in self.blocks)
        self.total_files = len(set(block.file_path for block in self.blocks))

        similarities = [block.similarity_score for block in self.blocks]
        self.max_similarity = max(similarities)
        self.min_similarity = min(similarities)
        self.avg_similarity = sum(similarities) / len(similarities)

    def to_dict(self) -> Dict[str, Any]:
        """Convert duplication group to dictionary"""
        return {
            "group_id": self.group_id,
            "duplication_type": self.duplication_type.value,
            "similarity_threshold": self.similarity_threshold,
            "blocks": [block.to_dict() for block in self.blocks],
            "total_lines": self.total_lines,
            "total_files": self.total_files,
            "max_similarity": self.max_similarity,
            "min_similarity": self.min_similarity,
            "avg_similarity": self.avg_similarity,
            "complexity_score": self.complexity_score,
            "cluster_id": self.cluster_id
        }
