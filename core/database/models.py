"""
Database models for code analysis results and metrics.
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, Integer, String, DateTime, Float, Text, Boolean,
    ForeignKey, Index, JSON, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Repository(Base):
    """Repository model for analysis tracking."""
    __tablename__ = "repositories"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    url = Column(String(500), nullable=False, unique=True)
    dominant_language = Column(String(50), nullable=False)
    last_analyzed = Column(DateTime, nullable=True)
    analysis_status = Column(String(20), nullable=False, default="pending")  # pending, running, completed, failed
    total_files = Column(Integer, default=0)
    analyzed_files = Column(Integer, default=0)
    total_lines = Column(Integer, default=0)
    total_functions = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    linting_issues = relationship("LintingIssue", back_populates="repository", cascade="all, delete-orphan")
    quality_metrics = relationship("QualityMetric", back_populates="repository", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Repository(id={self.id}, name='{self.name}', status='{self.analysis_status}')>"


class LintingIssue(Base):
    """Model for storing linting and style check issues."""
    __tablename__ = "linting_issues"

    id = Column(Integer, primary_key=True, index=True)
    repository_id = Column(Integer, ForeignKey("repositories.id"), nullable=False, index=True)
    file_path = Column(String(500), nullable=False, index=True)
    line_number = Column(Integer, nullable=False)
    column_number = Column(Integer, nullable=False)
    rule_id = Column(String(100), nullable=False)
    severity = Column(String(20), nullable=False, index=True)  # error, warning, info
    message = Column(Text, nullable=False)
    tool = Column(String(50), nullable=False, index=True)  # ruff, flake8, eslint
    category = Column(String(100), nullable=True)  # style, error, complexity, etc.
    suggestion = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    repository = relationship("Repository", back_populates="linting_issues")

    # Indexes for performance
    __table_args__ = (
        Index('idx_linting_repo_file', 'repository_id', 'file_path'),
        Index('idx_linting_severity_tool', 'severity', 'tool'),
    )

    def __repr__(self):
        return f"<LintingIssue(id={self.id}, file='{self.file_path}', line={self.line_number}, severity='{self.severity}')>"


class QualityMetric(Base):
    """Model for storing code quality metrics."""
    __tablename__ = "quality_metrics"

    id = Column(Integer, primary_key=True, index=True)
    repository_id = Column(Integer, ForeignKey("repositories.id"), nullable=False, index=True)
    file_path = Column(String(500), nullable=False, index=True)
    function_name = Column(String(255), nullable=True)

    # Complexity metrics
    cyclomatic_complexity = Column(Integer, nullable=False, default=1)
    cognitive_complexity = Column(Integer, nullable=False, default=1)
    halstead_volume = Column(Float, nullable=True)
    halstead_difficulty = Column(Float, nullable=True)

    # Quality metrics
    maintainability_index = Column(Float, nullable=True)  # 0-100 scale
    technical_debt_ratio = Column(Float, nullable=True)  # in hours
    code_smells_count = Column(Integer, default=0)
    duplicated_lines = Column(Integer, default=0)

    # Size metrics
    lines_of_code = Column(Integer, nullable=False)
    comment_lines = Column(Integer, default=0)
    blank_lines = Column(Integer, default=0)

    # Additional metadata
    language = Column(String(50), nullable=False)
    analysis_date = Column(DateTime, default=datetime.utcnow)

    # JSON field for additional metrics
    additional_metrics = Column(JSON, nullable=True)  # Tool-specific metrics

    # Relationships
    repository = relationship("Repository", back_populates="quality_metrics")

    # Indexes for performance
    __table_args__ = (
        Index('idx_quality_repo_file', 'repository_id', 'file_path'),
        Index('idx_quality_complexity', 'cyclomatic_complexity'),
        Index('idx_quality_maintainability', 'maintainability_index'),
    )

    def __repr__(self):
        return f"<QualityMetric(id={self.id}, file='{self.file_path}', complexity={self.cyclomatic_complexity})>"


class AnalysisCache(Base):
    """Model for caching analysis results to avoid reprocessing."""
    __tablename__ = "analysis_cache"

    id = Column(Integer, primary_key=True, index=True)
    repository_id = Column(Integer, ForeignKey("repositories.id"), nullable=False, index=True)
    file_path = Column(String(500), nullable=False)
    file_hash = Column(String(64), nullable=False, index=True)  # SHA-256 hash
    analysis_type = Column(String(50), nullable=False)  # linting, quality, security
    analysis_result = Column(JSON, nullable=False)  # Cached analysis result
    tool_version = Column(String(50), nullable=True)  # Tool version used
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)  # Cache expiration

    # Indexes for performance
    __table_args__ = (
        Index('idx_cache_hash', 'file_hash'),
        Index('idx_cache_repo_file', 'repository_id', 'file_path'),
    )

    def __repr__(self):
        return f"<AnalysisCache(id={self.id}, file='{self.file_path}', type='{self.analysis_type}')>"