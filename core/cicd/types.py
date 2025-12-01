"""
Shared types and models for CI/CD integration.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class CIPlatform(Enum):
    """Supported CI/CD platforms"""
    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    JENKINS = "jenkins"
    AZURE_PIPELINES = "azure_pipelines"
    CIRCLECI = "circleci"
    TRAVIS_CI = "travis_ci"
    APPVEYOR = "appveyor"


@dataclass
class CIJob:
    """Represents a CI/CD job"""
    platform: CIPlatform
    job_id: str
    job_name: str
    status: str
    started_at: str
    completed_at: Optional[str]
    duration_seconds: Optional[int]
    artifacts: List[str]
    test_results: Optional[Dict[str, Any]]
    build_logs: Optional[str]
    environment: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary"""
        return {
            "platform": self.platform.value,
            "job_id": self.job_id,
            "job_name": self.job_name,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "artifacts": self.artifacts,
            "test_results": self.test_results,
            "build_logs": self.build_logs,
            "environment": self.environment,
            "metadata": self.metadata
        }


@dataclass
class CIPipeline:
    """Represents a complete CI/CD pipeline"""
    pipeline_id: str
    platform: CIPlatform
    repository: str
    branch: str
    commit_sha: str
    status: str
    jobs: List[CIJob]
    started_at: str
    completed_at: Optional[str]
    total_duration_seconds: Optional[int]
    analysis_results: Dict[str, Any]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline to dictionary"""
        return {
            "pipeline_id": self.pipeline_id,
            "platform": self.platform.value,
            "repository": self.repository,
            "branch": self.branch,
            "commit_sha": self.commit_sha,
            "status": self.status,
            "jobs": [job.to_dict() for job in self.jobs],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_duration_seconds": self.total_duration_seconds,
            "analysis_results": self.analysis_results,
            "recommendations": self.recommendations
        }
