"""
Main duplicate code detector orchestrator
Coordinates multiple detection methods for comprehensive duplicate code analysis
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import hashlib
from pathlib import Path
import difflib
import re

from .jscpd_detector import JscpdDetector
from .similarity_analyzer import SimilarityAnalyzer
from .clustering_engine import ClusteringEngine
from utils.logger import get_logger

logger = get_logger(__name__)


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


@dataclass
class DuplicationReport:
    """Complete duplicate code analysis report"""
    repo_id: str
    scan_id: str
    timestamp: str
    total_files_scanned: int
    total_lines_scanned: int
    total_duplicate_lines: int
    duplication_percentage: float
    duplicate_groups: List[DuplicationGroup]
    file_statistics: Dict[str, Dict[str, Any]]
    language_statistics: Dict[str, Dict[str, Any]]
    complexity_statistics: Dict[str, Any]
    recommendations: List[str]
    scan_duration: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert duplication report to dictionary"""
        return {
            "repo_id": self.repo_id,
            "scan_id": self.scan_id,
            "timestamp": self.timestamp,
            "total_files_scanned": self.total_files_scanned,
            "total_lines_scanned": self.total_lines_scanned,
            "total_duplicate_lines": self.total_duplicate_lines,
            "duplication_percentage": self.duplication_percentage,
            "duplicate_groups": [group.to_dict() for group in self.duplicate_groups],
            "file_statistics": self.file_statistics,
            "language_statistics": self.language_statistics,
            "complexity_statistics": self.complexity_statistics,
            "recommendations": self.recommendations,
            "scan_duration": self.scan_duration
        }


class DuplicateCodeDetector:
    """Main duplicate code detector orchestrator"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize duplicate code detector with configuration"""
        self.config = config or {}
        self.jscpd_detector = JscpdDetector(config.get('jscpd', {}))
        self.similarity_analyzer = SimilarityAnalyzer(config.get('similarity', {}))
        self.clustering_engine = ClusteringEngine(config.get('clustering', {}))

        # Configuration
        self.min_similarity_threshold = config.get('min_similarity_threshold', 0.8)
        self.min_block_size = config.get('min_block_size', 10)  # Minimum lines
        self.max_block_size = config.get('max_block_size', 1000)  # Maximum lines
        self.exclude_patterns = config.get('exclude_patterns', [
            '*/test_*',
            '*/tests/*',
            '*/__pycache__/*',
            '*/node_modules/*',
            '*/vendor/*',
            '*/dist/*',
            '*/build/*',
            '*/target/*'
        ])

    async def scan_repository(self, repo_path: str, repo_id: str) -> DuplicationReport:
        """
        Perform comprehensive duplicate code analysis on a repository

        Args:
            repo_path: Path to the repository to scan
            repo_id: Repository ID for tracking

        Returns:
            DuplicationReport with all findings
        """
        import time
        import uuid
        from datetime import datetime

        scan_id = str(uuid.uuid4())
        start_time = time.time()

        logger.info(f"Starting duplicate code scan for repo {repo_id} with scan_id {scan_id}")

        try:
            # Find source files
            source_files = self._find_source_files(repo_path)
            if not source_files:
                logger.info("No source files found for duplicate code detection")
                return self._create_empty_report(repo_id, scan_id, start_time)

            # Scan with different methods
            exact_duplicates = await self._scan_exact_duplicates(source_files)
            similar_duplicates = await self._scan_similar_duplicates(source_files)
            structural_duplicates = await self._scan_structural_duplicates(source_files)

            # Combine and merge results
            all_groups = self._merge_duplicate_results(
                exact_duplicates, similar_duplicates, structural_duplicates
            )

            # Apply clustering
            clustered_groups = await self._apply_clustering(all_groups)

            # Generate statistics and recommendations
            file_stats = self._calculate_file_statistics(clustered_groups, source_files)
            language_stats = self._calculate_language_statistics(clustered_groups)
            complexity_stats = self._calculate_complexity_statistics(clustered_groups)
            recommendations = self._generate_recommendations(clustered_groups)

            # Calculate overall metrics
            total_lines = sum(len(self._get_file_lines(file_path)) for file_path in source_files)
            duplicate_lines = sum(group.total_lines for group in clustered_groups)
            duplication_percentage = (duplicate_lines / max(1, total_lines)) * 100

            end_time = time.time()
            scan_duration = end_time - start_time

            # Create report
            report = DuplicationReport(
                repo_id=repo_id,
                scan_id=scan_id,
                timestamp=datetime.utcnow().isoformat(),
                total_files_scanned=len(source_files),
                total_lines_scanned=total_lines,
                total_duplicate_lines=duplicate_lines,
                duplication_percentage=duplication_percentage,
                duplicate_groups=clustered_groups,
                file_statistics=file_stats,
                language_statistics=language_stats,
                complexity_statistics=complexity_stats,
                recommendations=recommendations,
                scan_duration=scan_duration
            )

            logger.info(f"Duplicate code scan completed for repo {repo_id}. "
                       f"Found {len(clustered_groups)} duplicate groups, "
                       f"{duplication_percentage:.1f}% duplication")

            return report

        except Exception as e:
            logger.error(f"Duplicate code scan failed for repo {repo_id}: {e}")
            return self._create_error_report(repo_id, scan_id, start_time, str(e))

    def _find_source_files(self, repo_path: str) -> List[str]:
        """Find source files to analyze"""
        source_extensions = {
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.go', '.rb', '.php',
            '.c', '.cpp', '.cxx', '.cc', '.h', '.hpp', '.hxx', '.cs', '.scala',
            '.kt', '.swift', '.rs', '.dart', '.lua', '.pl', '.pm', '.r', '.R',
            '.m', '.mm', '.sh', '.bash', '.zsh', '.fish'
        }

        source_files = []

        for root, dirs, files in os.walk(repo_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs
                      if not any(d.startswith(pattern.split('/')[-1].replace('*', ''))
                                for pattern in self.exclude_patterns
                                if '/' in pattern)]

            for file in files:
                file_path = os.path.join(root, file)
                if Path(file_path).suffix.lower() in source_extensions:
                    if not self._is_excluded(file_path):
                        source_files.append(file_path)

        return source_files

    def _is_excluded(self, file_path: str) -> bool:
        """Check if file should be excluded from analysis"""
        file_path = file_path.replace('\\', '/')
        for pattern in self.exclude_patterns:
            if pattern.startswith('*/'):
                if pattern[2:] in file_path:
                    return True
            elif pattern.endswith('/*'):
                if file_path.startswith(pattern[:-2]):
                    return True
            elif pattern in file_path:
                return True
        return False

    async def _scan_exact_duplicates(self, source_files: List[str]) -> List[DuplicationGroup]:
        """Scan for exact duplicate code blocks"""
        try:
            logger.info("Scanning for exact duplicate code blocks")
            return await self.jscpd_detector.scan_for_exact_duplicates(source_files)
        except Exception as e:
            logger.error(f"Exact duplicate scan failed: {e}")
            return []

    async def _scan_similar_duplicates(self, source_files: List[str]) -> List[DuplicationGroup]:
        """Scan for similar code blocks using token-based similarity"""
        try:
            logger.info("Scanning for similar duplicate code blocks")
            return await self.similarity_analyzer.find_similar_blocks(
                source_files, threshold=self.min_similarity_threshold
            )
        except Exception as e:
            logger.error(f"Similar duplicate scan failed: {e}")
            return []

    async def _scan_structural_duplicates(self, source_files: List[str]) -> List[DuplicationGroup]:
        """Scan for structural duplicates using AST analysis"""
        try:
            logger.info("Scanning for structural duplicate code blocks")
            return await self.similarity_analyzer.find_structural_duplicates(source_files)
        except Exception as e:
            logger.error(f"Structural duplicate scan failed: {e}")
            return []

    def _merge_duplicate_results(self, exact: List[DuplicationGroup],
                                similar: List[DuplicationGroup],
                                structural: List[DuplicationGroup]) -> List[DuplicationGroup]:
        """Merge duplicate detection results and remove overlaps"""
        all_groups = exact + similar + structural

        # Remove overlapping groups
        merged_groups = []
        for group in all_groups:
            if not self._overlaps_with_existing(group, merged_groups):
                merged_groups.append(group)

        # Sort by importance (more files, higher similarity, larger blocks)
        merged_groups.sort(key=lambda g: (
            g.total_files * -1,  # More files = higher priority
            g.avg_similarity * -1,  # Higher similarity = higher priority
            g.total_lines * -1,  # Larger blocks = higher priority
        ))

        return merged_groups

    def _overlaps_with_existing(self, group: DuplicationGroup,
                                 existing_groups: List[DuplicationGroup]) -> bool:
        """Check if group overlaps with existing groups"""
        for existing_group in existing_groups:
            for block1 in group.blocks:
                for block2 in existing_group.blocks:
                    if (block1.file_path == block2.file_path and
                        self._line_ranges_overlap(block1, block2)):
                        return True
        return False

    def _line_ranges_overlap(self, block1: DuplicateBlock, block2: DuplicateBlock) -> bool:
        """Check if two line ranges overlap"""
        return not (block1.end_line < block2.start_line or block2.end_line < block1.start_line)

    async def _apply_clustering(self, groups: List[DuplicationGroup]) -> List[DuplicationGroup]:
        """Apply clustering to group similar duplicate patterns"""
        try:
            logger.info("Applying clustering to duplicate groups")
            return await self.clustering_engine.cluster_duplicate_groups(groups)
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return groups

    def _get_file_lines(self, file_path: str) -> List[str]:
        """Get lines from file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.readlines()
        except Exception:
            return []

    def _calculate_file_statistics(self, groups: List[DuplicationGroup],
                                   source_files: List[str]) -> Dict[str, Dict[str, Any]]:
        """Calculate file-level duplication statistics"""
        file_stats = {}

        # Initialize file stats
        for file_path in source_files:
            file_stats[file_path] = {
                "total_lines": len(self._get_file_lines(file_path)),
                "duplicate_lines": 0,
                "duplicate_blocks": 0,
                "groups_involved": 0,
                "duplication_percentage": 0,
                "max_similarity": 0
            }

        # Calculate duplication metrics
        for group in groups:
            for block in group.blocks:
                if block.file_path in file_stats:
                    file_stats[block.file_path]["duplicate_lines"] += block.line_count
                    file_stats[block.file_path]["duplicate_blocks"] += 1
                    file_stats[block.file_path]["groups_involved"] += 1
                    file_stats[block.file_path]["max_similarity"] = max(
                        file_stats[block.file_path]["max_similarity"],
                        block.similarity_score
                    )

        # Calculate percentages
        for file_path, stats in file_stats.items():
            total_lines = stats["total_lines"]
            if total_lines > 0:
                stats["duplication_percentage"] = (
                    (stats["duplicate_lines"] / total_lines) * 100
                )

        return file_stats

    def _calculate_language_statistics(self, groups: List[DuplicationGroup]) -> Dict[str, Dict[str, Any]]:
        """Calculate language-level duplication statistics"""
        language_stats = {}

        for group in groups:
            for block in group.blocks:
                language = self._get_language_from_file(block.file_path)
                if language not in language_stats:
                    language_stats[language] = {
                        "total_groups": 0,
                        "total_blocks": 0,
                        "total_lines": 0,
                        "avg_similarity": 0,
                        "duplication_types": {}
                    }

                stats = language_stats[language]
                stats["total_groups"] += 1
                stats["total_blocks"] += 1
                stats["total_lines"] += block.line_count

                dup_type = group.duplication_type.value
                if dup_type not in stats["duplication_types"]:
                    stats["duplication_types"][dup_type] = 0
                stats["duplication_types"][dup_type] += 1

        # Calculate averages
        for language, stats in language_stats.items():
            total_blocks = stats["total_blocks"]
            if total_blocks > 0:
                stats["avg_similarity"] = sum(
                    block.similarity_score for group in groups
                    for block in group.blocks
                    if self._get_language_from_file(block.file_path) == language
                ) / total_blocks

        return language_stats

    def _calculate_complexity_statistics(self, groups: List[DuplicationGroup]) -> Dict[str, Any]:
        """Calculate complexity-related statistics"""
        complexity_stats = {
            "total_groups": len(groups),
            "avg_group_size": 0,
            "avg_lines_per_group": 0,
            "max_group_lines": 0,
            "complex_duplication_threshold": 100,  # Lines
            "complex_groups": 0,
            "duplication_type_distribution": {},
            "similarity_distribution": {
                "high": 0,  # > 0.9
                "medium": 0,  # 0.7 - 0.9
                "low": 0  # < 0.7
            }
        }

        if not groups:
            return complexity_stats

        total_group_sizes = sum(len(group.blocks) for group in groups)
        total_group_lines = sum(group.total_lines for group in groups)
        max_group_lines = max(group.total_lines for group in groups)

        complexity_stats["avg_group_size"] = total_group_sizes / len(groups)
        complexity_stats["avg_lines_per_group"] = total_group_lines / len(groups)
        complexity_stats["max_group_lines"] = max_group_lines

        # Count complex groups
        complexity_stats["complex_groups"] = sum(
            1 for group in groups if group.total_lines > complexity_stats["complex_duplication_threshold"]
        )

        # Duplication type distribution
        for group in groups:
            dup_type = group.duplication_type.value
            if dup_type not in complexity_stats["duplication_type_distribution"]:
                complexity_stats["duplication_type_distribution"][dup_type] = 0
            complexity_stats["duplication_type_distribution"][dup_type] += 1

        # Similarity distribution
        for group in groups:
            if group.avg_similarity > 0.9:
                complexity_stats["similarity_distribution"]["high"] += 1
            elif group.avg_similarity > 0.7:
                complexity_stats["similarity_distribution"]["medium"] += 1
            else:
                complexity_stats["similarity_distribution"]["low"] += 1

        return complexity_stats

    def _get_language_from_file(self, file_path: str) -> str:
        """Get programming language from file extension"""
        extension = Path(file_path).suffix.lower()
        language_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.jsx': 'JavaScript',
            '.ts': 'TypeScript',
            '.tsx': 'TypeScript',
            '.java': 'Java',
            '.go': 'Go',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.c': 'C',
            '.cpp': 'C++',
            '.cxx': 'C++',
            '.cc': 'C++',
            '.h': 'C/C++',
            '.hpp': 'C++',
            '.hxx': 'C++',
            '.cs': 'C#',
            '.scala': 'Scala',
            '.kt': 'Kotlin',
            '.swift': 'Swift',
            '.rs': 'Rust',
            '.dart': 'Dart',
            '.lua': 'Lua',
            '.pl': 'Perl',
            '.pm': 'Perl',
            '.r': 'R',
            '.R': 'R',
            '.m': 'Objective-C',
            '.mm': 'Objective-C++',
            '.sh': 'Shell',
            '.bash': 'Shell',
            '.zsh': 'Shell',
            '.fish': 'Shell'
        }
        return language_map.get(extension, 'Unknown')

    def _generate_recommendations(self, groups: List[DuplicationGroup]) -> List[str]:
        """Generate recommendations for reducing code duplication"""
        recommendations = []

        if not groups:
            return ["No code duplication detected. Continue following DRY principles."]

        total_groups = len(groups)
        total_duplicate_lines = sum(group.total_lines for group in groups)
        max_duplication_percentage = max(
            (group.total_lines / max(1, sum(
                block.line_count for file_block in group.blocks
                for file_block in [block]
            ))) * 100 for group in groups
        )

        # High-level recommendations
        if total_groups > 20:
            recommendations.append(
                "High number of duplicate code blocks detected. "
                "Consider a comprehensive refactoring effort."
            )

        if total_duplicate_lines > 1000:
            recommendations.append(
                f"Over {total_duplicate_lines} lines of duplicate code found. "
                "Extract common functionality into shared libraries or utilities."
            )

        if max_duplication_percentage > 50:
            recommendations.append(
                "Some files have high duplication rates. "
                "Consider refactoring to eliminate repeated patterns."
            )

        # Pattern-specific recommendations
        exact_groups = [g for g in groups if g.duplication_type == DuplicationType.EXACT]
        if len(exact_groups) > total_groups * 0.5:
            recommendations.append(
                "Many exact duplicates found. Use refactoring tools to extract common functions."
            )

        structural_groups = [g for g in groups if g.duplication_type == DuplicationType.STRUCTURAL]
        if structural_groups:
            recommendations.append(
                "Structural duplication detected. "
                "Consider using design patterns and templates to standardize similar structures."
            )

        # File-specific recommendations
        files_with_high_duplication = []
        for group in groups:
            file_counts = {}
            for block in group.blocks:
                if block.file_path not in file_counts:
                    file_counts[block.file_path] = 0
                file_counts[block.file_path] += 1

            for file_path, count in file_counts.items():
                if count > 3:  # File appears in multiple duplicate groups
                    files_with_high_duplication.append(file_path)

        if files_with_high_duplication:
            recommendations.append(
                f"Files with high duplication: {', '.join(Path(f).name for f in files_with_high_duplication[:5])}. "
                "Prioritize these files for refactoring."
            )

        # Tool recommendations
        recommendations.append("Consider using automated refactoring tools to extract duplicate code.")
        recommendations.append("Implement code review guidelines to catch duplication early.")
        recommendations.append("Set up static analysis alerts for new duplicate code.")

        return recommendations[:10]  # Limit to top 10 recommendations

    def _create_empty_report(self, repo_id: str, scan_id: str, start_time: float) -> DuplicationReport:
        """Create empty report when no source files are found"""
        import time
        from datetime import datetime

        return DuplicationReport(
            repo_id=repo_id,
            scan_id=scan_id,
            timestamp=datetime.utcnow().isoformat(),
            total_files_scanned=0,
            total_lines_scanned=0,
            total_duplicate_lines=0,
            duplication_percentage=0.0,
            duplicate_groups=[],
            file_statistics={},
            language_statistics={},
            complexity_statistics={},
            recommendations=["No source files found for analysis."],
            scan_duration=time.time() - start_time
        )

    def _create_error_report(self, repo_id: str, scan_id: str, start_time: float, error: str) -> DuplicationReport:
        """Create error report when scan fails"""
        import time
        from datetime import datetime

        return DuplicationReport(
            repo_id=repo_id,
            scan_id=scan_id,
            timestamp=datetime.utcnow().isoformat(),
            total_files_scanned=0,
            total_lines_scanned=0,
            total_duplicate_lines=0,
            duplication_percentage=0.0,
            duplicate_groups=[],
            file_statistics={},
            language_statistics={},
            complexity_statistics={},
            recommendations=[f"Scan failed: {error}"],
            scan_duration=time.time() - start_time
        )