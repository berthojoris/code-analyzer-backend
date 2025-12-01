"""
Duplicate code detection module
Supports jscpd integration, similarity analysis, and clustering for similar code blocks
"""

from .detector import DuplicateCodeDetector
from .jscpd_detector import JscpdDetector
from .similarity_analyzer import SimilarityAnalyzer
from .clustering_engine import ClusteringEngine

__all__ = [
    'DuplicateCodeDetector',
    'JscpdDetector',
    'SimilarityAnalyzer',
    'ClusteringEngine'
]