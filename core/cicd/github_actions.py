"""
GitHub Actions integration for CI/CD analysis
Supports workflow analysis and job monitoring
"""

import os
import asyncio
import yaml
from typing import List, Dict, Any, Optional
from pathlib import Path

from .types import CIPlatform, CIPipeline, CIJob
from utils.logger import get_logger

logger = get_logger(__name__)


class GitHubActionsIntegrator:
    """GitHub Actions CI/CD integration"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    async def analyze_pipeline(self, repo_path: str, repo_id: str) -> Dict[str, Any]:
        """Analyze GitHub Actions workflows"""
        workflows_dir = Path(repo_path) / '.github' / 'workflows'

        if not workflows_dir.exists():
            return {"workflows_count": 0, "has_workflows": False}

        workflows = list(workflows_dir.glob('*.yml')) + list(workflows_dir.glob('*.yaml'))

        analysis = {
            "workflows_count": len(workflows),
            "has_workflows": len(workflows) > 0,
            "workflow_files": [w.name for w in workflows],
            "has_testing": False,
            "has_security": False,
            "has_deployment": False,
            "test_jobs_count": 0,
            "security_jobs_count": 0,
            "deployment_jobs_count": 0
        }

        for workflow_file in workflows:
            try:
                with open(workflow_file, 'r') as f:
                    workflow = yaml.safe_load(f)

                jobs = workflow.get('jobs', {})
                for job_name, job_config in jobs.items():
                    steps = job_config.get('steps', [])

                    # Check for testing
                    if any('test' in str(step).lower() for step in steps):
                        analysis["has_testing"] = True
                        analysis["test_jobs_count"] += 1

                    # Check for security
                    if any('security' in str(step).lower() or 'bandit' in str(step).lower() for step in steps):
                        analysis["has_security"] = True
                        analysis["security_jobs_count"] += 1

                    # Check for deployment
                    if any('deploy' in str(step).lower() for step in steps):
                        analysis["has_deployment"] = True
                        analysis["deployment_jobs_count"] += 1

            except Exception as e:
                logger.warning(f"Failed to analyze workflow {workflow_file}: {e}")

        return analysis