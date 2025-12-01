"""
CI/CD integrator for multi-platform pipeline integration
Coordinates analysis across different CI/CD systems
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import yaml

from .github_actions import GitHubActionsIntegrator
from .gitlab_ci import GitLabCIIntegrator
from .jenkins import JenkinsIntegrator
from .artifact_scanner import ArtifactScanner
from .build_analyzer import BuildAnalyzer
from .types import CIPlatform, CIJob, CIPipeline
from utils.logger import get_logger

logger = get_logger(__name__)


class CIIntegrator:
    """Main CI/CD integration orchestrator"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize CI integrator with configuration"""
        self.config = config
        self.supported_platforms = config.get('supported_platforms', [
            CIPlatform.GITHUB_ACTIONS,
            CIPlatform.GITLAB_CI,
            CIPlatform.JENKINS
        ])

        # Initialize platform integrators
        self.platform_integrators = {}
        if CIPlatform.GITHUB_ACTIONS in self.supported_platforms:
            self.platform_integrators[CIPlatform.GITHUB_ACTIONS] = GitHubActionsIntegrator(
                config.get('github_actions', {})
            )
        if CIPlatform.GITLAB_CI in self.supported_platforms:
            self.platform_integrators[CIPlatform.GITLAB_CI] = GitLabCIIntegrator(
                config.get('gitlab_ci', {})
            )
        if CIPlatform.JENKINS in self.supported_platforms:
            self.platform_integrators[CIPlatform.JENKINS] = JenkinsIntegrator(
                config.get('jenkins', {})
            )

        # Initialize analyzers
        self.artifact_scanner = ArtifactScanner(config.get('artifact_scanner', {}))
        self.build_analyzer = BuildAnalyzer(config.get('build_analyzer', {}))

    async def detect_ci_platform(self, repo_path: str) -> List[CIPlatform]:
        """
        Detect which CI/CD platforms are configured in a repository

        Args:
            repo_path: Path to repository to analyze

        Returns:
            List of detected CI platforms
        """
        detected_platforms = []

        for platform in CIPlatform:
            if await self._is_platform_configured(repo_path, platform):
                detected_platforms.append(platform)

        logger.info(f"Detected CI platforms: {[p.value for p in detected_platforms]}")
        return detected_platforms

    async def _is_platform_configured(self, repo_path: str, platform: CIPlatform) -> bool:
        """Check if a specific CI platform is configured"""
        platform_config_paths = {
            CIPlatform.GITHUB_ACTIONS: [
                '.github/workflows',
                '.github/workflows/*.yml',
                '.github/workflows/*.yaml'
            ],
            CIPlatform.GITLAB_CI: [
                '.gitlab-ci.yml',
                '.gitlab-ci.yaml'
            ],
            CIPlatform.JENKINS: [
                'Jenkinsfile',
                'Jenkinsfile.*'
            ],
            CIPlatform.AZURE_PIPELINES: [
                'azure-pipelines.yml',
                'azure-pipelines.yaml',
                '.azure/pipelines/*.yml',
                '.azure/pipelines/*.yaml'
            ],
            CIPlatform.CIRCLECI: [
                '.circleci/config.yml',
                '.circleci/config.yaml'
            ],
            CIPlatform.TRAVIS_CI: [
                '.travis.yml'
            ],
            CIPlatform.APPVEYOR: [
                'appveyor.yml',
                '.appveyor.yml'
            ]
        }

        config_paths = platform_config_paths.get(platform, [])
        repo_path = Path(repo_path)

        for config_path in config_paths:
            if '*' in config_path:
                # Handle glob patterns
                pattern = config_path.replace('*', '*')
                if list(repo_path.glob(pattern)):
                    return True
            else:
                full_path = repo_path / config_path
                if full_path.exists():
                    return True

        return False

    async def integrate_ci_analysis(self, repo_path: str, repo_id: str) -> Dict[str, Any]:
        """
        Integrate CI/CD analysis with existing analysis pipeline

        Args:
            repo_path: Path to repository to analyze
            repo_id: Repository ID for tracking

        Returns:
            CI/CD integration results
        """
        logger.info(f"Starting CI/CD integration analysis for repo {repo_id}")

        try:
            # Detect CI platforms
            detected_platforms = await self.detect_ci_platform(repo_path)

            if not detected_platforms:
                return self._create_empty_integration_result(repo_id, "No CI/CD configuration detected")

            # Analyze each detected platform
            platform_results = {}
            for platform in detected_platforms:
                try:
                    if platform in self.platform_integrators:
                        result = await self.platform_integrators[platform].analyze_pipeline(repo_path, repo_id)
                        platform_results[platform.value] = result
                    else:
                        logger.warning(f"Platform {platform.value} detected but not supported")
                except Exception as e:
                    logger.error(f"Failed to analyze platform {platform.value}: {e}")
                    platform_results[platform.value] = {"error": str(e)}

            # Scan artifacts
            artifact_results = await self.artifact_scanner.scan_artifacts(repo_path)

            # Analyze build configuration
            build_results = await self.build_analyzer.analyze_build_config(repo_path)

            # Generate integration recommendations
            recommendations = self._generate_ci_recommendations(
                detected_platforms, platform_results, artifact_results, build_results
            )

            # Create comprehensive result
            integration_result = {
                "repo_id": repo_id,
                "detected_platforms": [p.value for p in detected_platforms],
                "platform_analysis": platform_results,
                "artifact_analysis": artifact_results,
                "build_analysis": build_results,
                "recommendations": recommendations,
                "integration_score": self._calculate_integration_score(
                    detected_platforms, platform_results, artifact_results, build_results
                ),
                "timestamp": self._get_timestamp()
            }

            logger.info(f"CI/CD integration analysis completed for repo {repo_id}")
            return integration_result

        except Exception as e:
            logger.error(f"CI/CD integration analysis failed for repo {repo_id}: {e}")
            return self._create_error_integration_result(repo_id, str(e))

    async def monitor_ci_pipeline(self, repo_path: str, pipeline_id: str) -> Optional[CIPipeline]:
        """
        Monitor a specific CI/CD pipeline for analysis opportunities

        Args:
            repo_path: Path to repository
            pipeline_id: Pipeline identifier

        Returns:
            CIPipeline object or None if not found
        """
        try:
            # Detect platform from pipeline_id or repository config
            detected_platforms = await self.detect_ci_platform(repo_path)
            if not detected_platforms:
                return None

            # Use first detected platform for monitoring
            platform = detected_platforms[0]
            if platform not in self.platform_integrators:
                return None

            # Get pipeline status from platform
            pipeline = await self.platform_integrators[platform].get_pipeline_status(pipeline_id)

            # Run analysis on pipeline artifacts and jobs
            if pipeline:
                for job in pipeline.jobs:
                    if job.artifacts:
                        artifact_analysis = await self.artifact_scanner.analyze_job_artifacts(
                            job.artifacts, repo_path
                        )
                        job.metadata['artifact_analysis'] = artifact_analysis

                    if job.test_results:
                        test_analysis = self._analyze_test_results(job.test_results)
                        job.metadata['test_analysis'] = test_analysis

                # Generate pipeline-level recommendations
                pipeline.recommendations = self._generate_pipeline_recommendations(pipeline)

            return pipeline

        except Exception as e:
            logger.error(f"Failed to monitor pipeline {pipeline_id}: {e}")
            return None

    async def generate_ci_report(self, repo_path: str, repo_id: str, format: str = "json") -> str:
        """
        Generate comprehensive CI/CD integration report

        Args:
            repo_path: Path to repository
            repo_id: Repository ID
            format: Output format (json, yaml, html)

        Returns:
            Report as string
        """
        try:
            # Perform integration analysis
            analysis_result = await self.integrate_ci_analysis(repo_path, repo_id)

            # Format report based on requested format
            if format.lower() == "json":
                return json.dumps(analysis_result, indent=2)
            elif format.lower() == "yaml":
                return yaml.dump(analysis_result, default_flow_style=False)
            elif format.lower() == "html":
                return self._generate_html_report(analysis_result)
            else:
                raise ValueError(f"Unsupported report format: {format}")

        except Exception as e:
            logger.error(f"Failed to generate CI/CD report: {e}")
            return json.dumps({"error": str(e)})

    def _generate_ci_recommendations(self, platforms: List[CIPlatform],
                                     platform_results: Dict[str, Any],
                                     artifact_results: Dict[str, Any],
                                     build_results: Dict[str, Any]) -> List[str]:
        """Generate CI/CD improvement recommendations"""
        recommendations = []

        # Platform-specific recommendations
        for platform in platforms:
            platform_key = platform.value
            if platform_key in platform_results:
                result = platform_results[platform_key]

                if platform == CIPlatform.GITHUB_ACTIONS:
                    if result.get('workflows_count', 0) == 0:
                        recommendations.append("Consider adding GitHub Actions workflows for automated testing and deployment")
                    elif result.get('test_jobs_count', 0) == 0:
                        recommendations.append("Add automated testing jobs to GitHub Actions workflows")
                    elif result.get('security_jobs_count', 0) == 0:
                        recommendations.append("Add security scanning jobs to GitHub Actions workflows")

                elif platform == CIPlatform.GITLAB_CI:
                    if not result.get('has_gitlab_ci', False):
                        recommendations.append("Consider adding .gitlab-ci.yml for CI/CD automation")
                    elif not result.get('has_security_scanning', False):
                        recommendations.append("Add security scanning stages to GitLab CI pipeline")

                elif platform == CIPlatform.JENKINS:
                    if not result.get('has_jenkinsfile', False):
                        recommendations.append("Consider adding Jenkinsfile for pipeline-as-code")
                    elif not result.get('has_parallel_stages', False):
                        recommendations.append("Consider parallel stages in Jenkins pipeline for faster builds")

        # Artifact recommendations
        if artifact_results.get('security_issues', 0) > 0:
            recommendations.append("Implement artifact security scanning before deployment")

        if artifact_results.get('large_artifacts', 0) > 0:
            recommendations.append("Optimize artifact sizes to improve build performance")

        if not artifact_results.get('has_checksums', False):
            recommendations.append("Add artifact checksums for integrity verification")

        # Build recommendations
        if build_results.get('build_time_seconds', 0) > 600:  # 10 minutes
            recommendations.append("Optimize build configuration to reduce build time")

        if not build_results.get('has_caching', False):
            recommendations.append("Implement build caching to improve performance")

        if not build_results.get('has_parallel_builds', False):
            recommendations.append("Consider parallel builds to improve speed")

        # General recommendations
        if len(platforms) > 1:
            recommendations.append("Consider consolidating CI/CD platforms to reduce complexity")

        if not any(p in platforms for p in [CIPlatform.GITHUB_ACTIONS, CIPlatform.GITLAB_CI]):
            recommendations.append("Consider modern CI/CD platforms with better integration capabilities")

        recommendations.extend([
            "Implement regular security scanning in CI/CD pipeline",
            "Add automated testing with code coverage reporting",
            "Set up failure notifications and rollback mechanisms",
            "Monitor pipeline performance and optimize bottlenecks"
        ])

        return list(set(recommendations))[:15]  # Limit to top 15 recommendations

    def _analyze_test_results(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze test results for insights"""
        analysis = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "test_coverage": 0,
            "failure_rate": 0,
            "recommendations": []
        }

        try:
            # Extract test metrics
            if 'tests' in test_results:
                analysis["total_tests"] = len(test_results['tests'])
                for test in test_results['tests']:
                    status = test.get('status', '').lower()
                    if status == 'passed':
                        analysis["passed_tests"] += 1
                    elif status == 'failed':
                        analysis["failed_tests"] += 1
                    elif status in ['skipped', 'pending']:
                        analysis["skipped_tests"] += 1

            # Calculate rates
            if analysis["total_tests"] > 0:
                analysis["failure_rate"] = analysis["failed_tests"] / analysis["total_tests"]

            # Extract coverage if available
            if 'coverage' in test_results:
                analysis["test_coverage"] = test_results['coverage'].get('percentage', 0)

            # Generate recommendations
            if analysis["failure_rate"] > 0.1:  # > 10% failure rate
                analysis["recommendations"].append("High test failure rate detected - review failing tests")

            if analysis["test_coverage"] < 80:
                analysis["recommendations"].append("Increase test coverage to at least 80%")

            if analysis["skipped_tests"] / max(1, analysis["total_tests"]) > 0.2:  # > 20% skipped
                analysis["recommendations"].append("Review and fix skipped tests")

        except Exception as e:
            logger.warning(f"Failed to analyze test results: {e}")

        return analysis

    def _generate_pipeline_recommendations(self, pipeline: CIPipeline) -> List[str]:
        """Generate pipeline-specific recommendations"""
        recommendations = []

        # Performance recommendations
        if pipeline.total_duration_seconds and pipeline.total_duration_seconds > 1800:  # 30 minutes
            recommendations.append("Pipeline duration exceeds 30 minutes - consider optimization")

        # Failure recommendations
        failed_jobs = [job for job in pipeline.jobs if job.status.lower() == 'failed']
        if failed_jobs:
            recommendations.append(f"Address {len(failed_jobs)} failed jobs in pipeline")

        # Flaky test recommendations
        flaky_jobs = [
            job for job in pipeline.jobs
            if job.test_results and job.test_results.get('flaky_tests', 0) > 0
        ]
        if flaky_jobs:
            recommendations.append("Fix flaky tests causing pipeline instability")

        return recommendations

    def _calculate_integration_score(self, platforms: List[CIPlatform],
                                   platform_results: Dict[str, Any],
                                   artifact_results: Dict[str, Any],
                                   build_results: Dict[str, Any]) -> float:
        """Calculate overall CI/CD integration score (0-100)"""
        score = 0

        # Platform presence (30 points)
        if platforms:
            score += 20
            if any(p in [CIPlatform.GITHUB_ACTIONS, CIPlatform.GITLAB_CI] for p in platforms):
                score += 10

        # Platform configuration quality (30 points)
        for platform in platforms:
            platform_key = platform.value
            if platform_key in platform_results:
                result = platform_results[platform_key]
                if result.get('has_workflows', False):
                    score += 5
                if result.get('has_testing', False):
                    score += 3
                if result.get('has_security', False):
                    score += 2

        # Build optimization (20 points)
        if build_results.get('has_caching', False):
            score += 8
        if build_results.get('has_parallel_builds', False):
            score += 7
        if build_results.get('build_time_seconds', 600) < 600:  # Less than 10 minutes
            score += 5

        # Artifact management (20 points)
        if artifact_results.get('has_security_scanning', False):
            score += 10
        if artifact_results.get('has_artifact_size_limits', False):
            score += 5
        if artifact_results.get('has_checksums', False):
            score += 5

        return min(100, score)

    def _create_empty_integration_result(self, repo_id: str, message: str) -> Dict[str, Any]:
        """Create empty integration result"""
        return {
            "repo_id": repo_id,
            "detected_platforms": [],
            "platform_analysis": {},
            "artifact_analysis": {},
            "build_analysis": {},
            "recommendations": [message],
            "integration_score": 0,
            "timestamp": self._get_timestamp()
        }

    def _create_error_integration_result(self, repo_id: str, error: str) -> Dict[str, Any]:
        """Create error integration result"""
        return {
            "repo_id": repo_id,
            "detected_platforms": [],
            "platform_analysis": {"error": error},
            "artifact_analysis": {"error": error},
            "build_analysis": {"error": error},
            "recommendations": [f"CI/CD analysis failed: {error}"],
            "integration_score": 0,
            "timestamp": self._get_timestamp()
        }

    def _generate_html_report(self, analysis_result: Dict[str, Any]) -> str:
        """Generate HTML report from analysis results"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>CI/CD Integration Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background-color: #f4f4f4; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .score { font-size: 24px; font-weight: bold; color: #2ecc71; }
                .recommendations { background-color: #fff3cd; padding: 15px; border-radius: 5px; }
                .platform { background-color: #e7f3ff; padding: 10px; margin: 10px 0; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>CI/CD Integration Report</h1>
                <p>Repository: {repo_id}</p>
                <p>Generated: {timestamp}</p>
                <div class="score">Integration Score: {integration_score}/100</div>
            </div>

            <div class="section">
                <h2>Detected Platforms</h2>
                {platforms_html}
            </div>

            <div class="section">
                <h2>Recommendations</h2>
                <div class="recommendations">
                    <ul>
                        {recommendations_html}
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """

        # Format platforms
        platforms_html = ""
        for platform in analysis_result.get('detected_platforms', []):
            platforms_html += f'<div class="platform">{platform}</div>'

        # Format recommendations
        recommendations_html = ""
        for rec in analysis_result.get('recommendations', []):
            recommendations_html += f"<li>{rec}</li>"

        return html_template.format(
            repo_id=analysis_result.get('repo_id', 'Unknown'),
            timestamp=analysis_result.get('timestamp', 'Unknown'),
            integration_score=analysis_result.get('integration_score', 0),
            platforms_html=platforms_html,
            recommendations_html=recommendations_html
        )

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()