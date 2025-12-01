"""
Metadata enricher for GitHub repositories
Enriches repository data with additional information and insights
"""

import asyncio
import re
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from collections import Counter, defaultdict
import json

from .client import GitHubClient, Repository, Commit
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EnrichedRepository:
    """Repository with enriched metadata"""
    base_repository: Repository
    health_score: float
    activity_metrics: Dict[str, Any]
    code_quality_indicators: Dict[str, Any]
    security_indicators: Dict[str, Any]
    community_metrics: Dict[str, Any]
    technical_metrics: Dict[str, Any]
    recommendations: List[str]
    tags: List[str]
    risk_assessment: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "base_repository": self.base_repository.to_dict(),
            "health_score": self.health_score,
            "activity_metrics": self.activity_metrics,
            "code_quality_indicators": self.code_quality_indicators,
            "security_indicators": self.security_indicators,
            "community_metrics": self.community_metrics,
            "technical_metrics": self.technical_metrics,
            "recommendations": self.recommendations,
            "tags": self.tags,
            "risk_assessment": self.risk_assessment
        }


class MetadataEnricher:
    """Enriches repository metadata with analysis and insights"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize metadata enricher with configuration"""
        self.config = config
        self.github_client = GitHubClient(config.get('github', {}))
        self.analysis_period_days = config.get('analysis_period_days', 90)
        self.health_score_weights = config.get('health_score_weights', {
            'activity': 0.3,
            'code_quality': 0.25,
            'security': 0.2,
            'community': 0.15,
            'maintenance': 0.1
        })

        # File patterns for different analyses
        self.config_files = {
            'ci': ['.github/workflows/*.yml', '.github/workflows/*.yaml',
                   '.travis.yml', '.gitlab-ci.yml', 'Jenkinsfile'],
            'testing': ['pytest.ini', 'tox.ini', '.coveragerc', 'jest.config.js',
                       'karma.conf.js', 'test/', 'tests/', '__tests__/'],
            'documentation': ['README.md', 'CHANGELOG.md', 'CONTRIBUTING.md',
                              'LICENSE', 'docs/', 'doc/', 'docs/'],
            'dependencies': ['requirements.txt', 'package.json', 'Pipfile',
                            'pyproject.toml', 'composer.json', 'Gemfile'],
            'security': ['.bandit', '.safety-policy.json', 'security.md',
                        '.github/SECURITY.md']
        }

        # Risk indicators
        self.high_risk_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'system\s*\(',
            r'shell_exec\s*\(',
            r'os\.system\s*\(',
            r'subprocess\.call\s*\(',
            r'subprocess\.run\s*\(',
            r'pickle\.loads?\s*\(',
            r'marshal\.loads?\s*\('
        ]

    async def enrich_repository(self, owner: str, repo: str) -> Optional[EnrichedRepository]:
        """
        Enrich repository with comprehensive metadata analysis

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            EnrichedRepository object or None if enrichment fails
        """
        try:
            async with self.github_client:
                logger.info(f"Enriching repository metadata for {owner}/{repo}")

                # Get base repository data
                base_repo = await self.github_client.get_repository(owner, repo)
                if not base_repo:
                    return None

                # Enrich with additional data
                enriched_data = await self._gather_enrichment_data(base_repo)

                # Calculate health score and indicators
                health_score, indicators = self._calculate_health_score(base_repo, enriched_data)

                # Generate recommendations and tags
                recommendations = self._generate_recommendations(base_repo, enriched_data, indicators)
                tags = self._generate_tags(base_repo, enriched_data, indicators)

                # Risk assessment
                risk_assessment = self._assess_risk(base_repo, enriched_data)

                return EnrichedRepository(
                    base_repository=base_repo,
                    health_score=health_score,
                    activity_metrics=indicators.get('activity', {}),
                    code_quality_indicators=indicators.get('code_quality', {}),
                    security_indicators=indicators.get('security', {}),
                    community_metrics=indicators.get('community', {}),
                    technical_metrics=indicators.get('technical', {}),
                    recommendations=recommendations,
                    tags=tags,
                    risk_assessment=risk_assessment
                )

        except Exception as e:
            logger.error(f"Failed to enrich repository {owner}/{repo}: {e}")
            return None

    async def _gather_enrichment_data(self, repo: Repository) -> Dict[str, Any]:
        """Gather data for repository enrichment"""
        owner = repo.owner['login']
        repo_name = repo.name

        enrichment_data = {
            "commits": [],
            "files": [],
            "contributors": [],
            "issues": [],
            "pull_requests": [],
            "releases": []
        }

        try:
            # Get recent commits
            enrichment_data["commits"] = await self.github_client.get_repository_commits(
                owner, repo_name, per_page=100
            )

            # Get repository files (root level)
            enrichment_data["files"] = await self.github_client.get_repository_contents(
                owner, repo_name
            )

            # Get contributors
            enrichment_data["contributors"] = await self._get_contributors(owner, repo_name)

            # Get issues (open and recent closed)
            enrichment_data["issues"] = await self._get_issues(owner, repo_name)

            # Get pull requests
            enrichment_data["pull_requests"] = await self._get_pull_requests(owner, repo_name)

            # Get releases
            enrichment_data["releases"] = await self._get_releases(owner, repo_name)

        except Exception as e:
            logger.warning(f"Failed to gather some enrichment data: {e}")

        return enrichment_data

    async def _get_contributors(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """Get repository contributors"""
        try:
            endpoint = f"repos/{owner}/{repo}/contributors"
            return await self.github_client._make_request('GET', endpoint)
        except Exception as e:
            logger.warning(f"Failed to get contributors for {owner}/{repo}: {e}")
            return []

    async def _get_issues(self, owner: str, repo: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get repository issues (open and recent closed)"""
        issues_data = {"open": [], "recent_closed": []}

        try:
            # Get open issues
            endpoint = f"repos/{owner}/{repo}/issues"
            params = {"state": "open", "per_page": 100}
            issues_data["open"] = await self.github_client._make_request('GET', endpoint, params=params)

            # Get recently closed issues (last 30 days)
            since_date = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            params = {"state": "closed", "since": since_date, "per_page": 100}
            issues_data["recent_closed"] = await self.github_client._make_request('GET', endpoint, params=params)

        except Exception as e:
            logger.warning(f"Failed to get issues for {owner}/{repo}: {e}")

        return issues_data

    async def _get_pull_requests(self, owner: str, repo: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get repository pull requests"""
        pr_data = {"open": [], "recent_closed": []}

        try:
            # Get open PRs
            endpoint = f"repos/{owner}/{repo}/pulls"
            params = {"state": "open", "per_page": 100}
            pr_data["open"] = await self.github_client._make_request('GET', endpoint, params=params)

            # Get recently closed PRs
            since_date = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            params = {"state": "closed", "since": since_date, "per_page": 100}
            pr_data["recent_closed"] = await self.github_client._make_request('GET', endpoint, params=params)

        except Exception as e:
            logger.warning(f"Failed to get pull requests for {owner}/{repo}: {e}")

        return pr_data

    async def _get_releases(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """Get repository releases"""
        try:
            endpoint = f"repos/{owner}/{repo}/releases"
            params = {"per_page": 10}
            return await self.github_client._make_request('GET', endpoint, params=params)
        except Exception as e:
            logger.warning(f"Failed to get releases for {owner}/{repo}: {e}")
            return []

    def _calculate_health_score(self, repo: Repository, enrichment_data: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
        """Calculate repository health score and various indicators"""
        indicators = {
            'activity': self._calculate_activity_metrics(repo, enrichment_data),
            'code_quality': self._calculate_code_quality_indicators(repo, enrichment_data),
            'security': self._calculate_security_indicators(repo, enrichment_data),
            'community': self._calculate_community_metrics(repo, enrichment_data),
            'technical': self._calculate_technical_metrics(repo, enrichment_data),
            'maintenance': self._calculate_maintenance_metrics(repo, enrichment_data)
        }

        # Calculate weighted health score
        activity_score = indicators['activity'].get('activity_score', 0)
        quality_score = indicators['code_quality'].get('quality_score', 0)
        security_score = indicators['security'].get('security_score', 0)
        community_score = indicators['community'].get('community_score', 0)
        maintenance_score = indicators['maintenance'].get('maintenance_score', 0)

        health_score = (
            activity_score * self.health_score_weights['activity'] +
            quality_score * self.health_score_weights['code_quality'] +
            security_score * self.health_score_weights['security'] +
            community_score * self.health_score_weights['community'] +
            maintenance_score * self.health_score_weights['maintenance']
        )

        # Normalize to 0-100 scale
        health_score = min(100, max(0, health_score))

        return health_score, indicators

    def _calculate_activity_metrics(self, repo: Repository, enrichment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate repository activity metrics"""
        commits = enrichment_data.get('commits', [])
        issues = enrichment_data.get('issues', {})
        prs = enrichment_data.get('pull_requests', {})

        # Commit activity (last 90 days)
        now = datetime.now(timezone.utc)
        recent_commits = [
            c for c in commits
            if datetime.fromisoformat(c.commit['committer']['date'].replace('Z', '+00:00')) >
               now - timedelta(days=self.analysis_period_days)
        ]

        commit_frequency = len(recent_commits) / max(1, self.analysis_period_days / 7)  # commits per week

        # Issue activity
        open_issues = len(issues.get('open', []))
        recently_closed = len(issues.get('recent_closed', []))
        issue_closure_rate = recently_closed / max(1, open_issues + recently_closed)

        # PR activity
        open_prs = len(prs.get('open', []))
        recently_closed_prs = len(prs.get('recent_closed', []))
        pr_merge_rate = recently_closed_prs / max(1, open_prs + recently_closed_prs)

        # Calculate activity score
        activity_score = 0
        if commit_frequency > 1:
            activity_score += 30
        elif commit_frequency > 0.1:
            activity_score += 15

        if issue_closure_rate > 0.7:
            activity_score += 25
        elif issue_closure_rate > 0.3:
            activity_score += 10

        if pr_merge_rate > 0.7:
            activity_score += 25
        elif pr_merge_rate > 0.3:
            activity_score += 10

        if repo.pushed_at:
            last_push = datetime.fromisoformat(repo.pushed_at.replace('Z', '+00:00'))
            days_since_push = (now - last_push).days
            if days_since_push < 7:
                activity_score += 20
            elif days_since_push < 30:
                activity_score += 10

        return {
            'activity_score': activity_score,
            'commit_frequency': commit_frequency,
            'recent_commits': len(recent_commits),
            'open_issues': open_issues,
            'issue_closure_rate': issue_closure_rate,
            'open_pull_requests': open_prs,
            'pr_merge_rate': pr_merge_rate,
            'last_push_days': (now - datetime.fromisoformat(repo.pushed_at.replace('Z', '+00:00'))).days if repo.pushed_at else None
        }

    def _calculate_code_quality_indicators(self, repo: Repository, enrichment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate code quality indicators"""
        files = enrichment_data.get('files', [])
        languages = repo.languages or {}

        # Check for quality-related files
        quality_indicators = {
            'has_ci': False,
            'has_tests': False,
            'has_documentation': False,
            'has_license': False,
            'has_code_style_config': False,
            'has_readme': False
        }

        for file_item in files:
            if isinstance(file_item, dict):
                file_name = file_item.get('name', '')
                file_path = file_item.get('path', '')

                # Check for CI configuration
                if any(pattern.replace('*', '') in file_path for pattern in self.config_files['ci']):
                    quality_indicators['has_ci'] = True

                # Check for testing configuration
                if any(pattern.replace('*', '') in file_path for pattern in self.config_files['testing']):
                    quality_indicators['has_tests'] = True

                # Check for documentation
                if file_name in ['README.md', 'CONTRIBUTING.md', 'CHANGELOG.md'] or \
                   any(pattern.replace('*', '') in file_path for pattern in self.config_files['documentation']):
                    quality_indicators['has_documentation'] = True

                # Check for license
                if file_name.lower().startswith('license') or file_name.lower().startswith('copying'):
                    quality_indicators['has_license'] = True

                # Check for code style configuration
                if file_name in ['.editorconfig', '.prettierrc', '.eslintrc.js', 'pyproject.toml']:
                    quality_indicators['has_code_style_config'] = True

                if file_name.lower() == 'readme.md':
                    quality_indicators['has_readme'] = True

        # Calculate quality score
        quality_score = 0
        if quality_indicators['has_ci']:
            quality_score += 20
        if quality_indicators['has_tests']:
            quality_score += 20
        if quality_indicators['has_documentation']:
            quality_score += 20
        if quality_indicators['has_license']:
            quality_score += 15
        if quality_indicators['has_code_style_config']:
            quality_score += 15
        if quality_indicators['has_readme']:
            quality_score += 10

        # Language diversity consideration
        language_count = len(languages)
        if language_count > 1:
            quality_score += 5

        return {
            'quality_score': quality_score,
            'indicators': quality_indicators,
            'language_diversity': language_count,
            'dominant_language': max(languages.items(), key=lambda x: x[1])[0] if languages else None
        }

    def _calculate_security_indicators(self, repo: Repository, enrichment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate security indicators"""
        files = enrichment_data.get('files', [])
        commits = enrichment_data.get('commits', [])

        security_indicators = {
            'has_security_policy': False,
            'has_security_config': False,
            'recent_security_commits': 0,
            'dependency_files': 0,
            'has_security_issues': False
        }

        for file_item in files:
            if isinstance(file_item, dict):
                file_name = file_item.get('name', '')
                file_path = file_item.get('path', '')

                # Check for security policy
                if 'security' in file_path.lower() or file_name.lower().startswith('security'):
                    security_indicators['has_security_policy'] = True

                # Check for security configuration
                if any(pattern.replace('*', '') in file_path for pattern in self.config_files['security']):
                    security_indicators['has_security_config'] = True

                # Count dependency files
                if any(pattern.replace('*', '') in file_path for pattern in self.config_files['dependencies']):
                    security_indicators['dependency_files'] += 1

        # Check for security-related commits
        security_keywords = ['security', 'vulnerability', 'cve', 'patch', 'fix', 'secure']
        for commit in commits[:50]:  # Check recent commits
            commit_message = commit.commit.get('message', '').lower()
            if any(keyword in commit_message for keyword in security_keywords):
                security_indicators['recent_security_commits'] += 1

        # Calculate security score
        security_score = 0
        if security_indicators['has_security_policy']:
            security_score += 25
        if security_indicators['has_security_config']:
            security_score += 25
        if security_indicators['recent_security_commits'] > 0:
            security_score += 20
        if security_indicators['dependency_files'] > 0:
            security_score += 15
        if not repo.is_private:
            security_score += 15  # Bonus for public repos (more scrutiny)

        return {
            'security_score': security_score,
            'indicators': security_indicators
        }

    def _calculate_community_metrics(self, repo: Repository, enrichment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate community metrics"""
        contributors = enrichment_data.get('contributors', [])

        community_score = 0

        # Contributor diversity
        contributor_count = len(contributors)
        if contributor_count > 10:
            community_score += 25
        elif contributor_count > 3:
            community_score += 15
        elif contributor_count > 1:
            community_score += 5

        # Star and fork metrics
        if repo.stargazers_count > 1000:
            community_score += 20
        elif repo.stargazers_count > 100:
            community_score += 10
        elif repo.stargazers_count > 10:
            community_score += 5

        if repo.forks_count > 100:
            community_score += 15
        elif repo.forks_count > 10:
            community_score += 8
        elif repo.forks_count > 1:
            community_score += 3

        # Watchers
        if repo.watchers_count > 50:
            community_score += 10
        elif repo.watchers_count > 10:
            community_score += 5

        # Issues and PR engagement
        if repo.open_issues_count > 20:
            community_score += 10
        elif repo.open_issues_count > 5:
            community_score += 5

        return {
            'community_score': community_score,
            'contributor_count': contributor_count,
            'stargazers_count': repo.stargazers_count,
            'forks_count': repo.forks_count,
            'watchers_count': repo.watchers_count,
            'open_issues_count': repo.open_issues_count
        }

    def _calculate_technical_metrics(self, repo: Repository, enrichment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate technical metrics"""
        commits = enrichment_data.get('commits', [])

        # Repository size and complexity indicators
        repo_size_mb = repo.size / 1024  # GitHub size is in KB

        # Commit patterns
        commit_messages = [commit.commit.get('message', '') for commit in commits[:100]]
        avg_commit_message_length = sum(len(msg) for msg in commit_messages) / max(1, len(commit_messages))

        # Branch diversity (default branch vs others)
        has_multiple_branches = repo.default_branch != 'main' and repo.default_branch != 'master'

        return {
            'repository_size_mb': repo_size_mb,
            'avg_commit_message_length': avg_commit_message_length,
            'has_multiple_branches': has_multiple_branches,
            'language': repo.language,
            'total_commits': len(commits),
            'topics_count': len(repo.topics),
            'has_wiki': repo.has_wiki,
            'has_pages': repo.has_pages,
            'has_projects': repo.has_projects
        }

    def _calculate_maintenance_metrics(self, repo: Repository, enrichment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate maintenance metrics"""
        releases = enrichment_data.get('releases', [])
        commits = enrichment_data.get('commits', [])

        now = datetime.now(timezone.utc)

        # Release frequency
        recent_releases = [
            r for r in releases
            if datetime.fromisoformat(r['published_at'].replace('Z', '+00:00')) >
               now - timedelta(days=365)
        ]

        # Last update metrics
        if repo.pushed_at:
            last_push = datetime.fromisoformat(repo.pushed_at.replace('Z', '+00:00'))
            days_since_last_push = (now - last_push).days
        else:
            days_since_last_push = float('inf')

        # Recent activity
        recent_commits = [
            c for c in commits
            if datetime.fromisoformat(c.commit['committer']['date'].replace('Z', '+00:00')) >
               now - timedelta(days=30)
        ]

        maintenance_score = 0
        if len(recent_releases) > 4:  # Quarterly releases
            maintenance_score += 25
        elif len(recent_releases) > 1:
            maintenance_score += 15

        if days_since_last_push < 7:
            maintenance_score += 25
        elif days_since_last_push < 30:
            maintenance_score += 15
        elif days_since_last_push < 90:
            maintenance_score += 5

        if len(recent_commits) > 10:
            maintenance_score += 25
        elif len(recent_commits) > 3:
            maintenance_score += 15
        elif len(recent_commits) > 0:
            maintenance_score += 5

        if repo.has_issues and repo.open_issues_count < 20:
            maintenance_score += 25
        elif repo.open_issues_count < 50:
            maintenance_score += 10

        return {
            'maintenance_score': maintenance_score,
            'recent_releases_count': len(recent_releases),
            'days_since_last_push': days_since_last_push,
            'recent_commits_count': len(recent_commits),
            'total_releases': len(releases)
        }

    def _generate_recommendations(self, repo: Repository, enrichment_data: Dict[str, Any],
                                 indicators: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []

        activity_metrics = indicators.get('activity', {})
        quality_metrics = indicators.get('code_quality', {})
        security_metrics = indicators.get('security', {})
        community_metrics = indicators.get('community', {})

        # Activity recommendations
        if activity_metrics.get('commit_frequency', 0) < 0.1:
            recommendations.append("Consider increasing commit frequency to show active development")

        if activity_metrics.get('issue_closure_rate', 0) < 0.3:
            recommendations.append("Improve issue management and closure rate")

        if activity_metrics.get('pr_merge_rate', 0) < 0.3:
            recommendations.append("Streamline pull request review and merge process")

        # Quality recommendations
        quality_indicators = quality_metrics.get('indicators', {})
        if not quality_indicators.get('has_ci', False):
            recommendations.append("Add continuous integration (CI) configuration")

        if not quality_indicators.get('has_tests', False):
            recommendations.append("Add automated testing configuration")

        if not quality_indicators.get('has_documentation', False):
            recommendations.append("Improve project documentation (README, contributing guide)")

        if not quality_indicators.get('has_license', False):
            recommendations.append("Add a license file to specify usage terms")

        # Security recommendations
        security_indicators = security_metrics.get('indicators', {})
        if not security_indicators.get('has_security_policy', False):
            recommendations.append("Create a security policy and vulnerability disclosure process")

        if security_indicators.get('dependency_files', 0) == 0:
            recommendations.append("Implement dependency management and regular updates")

        # Community recommendations
        if community_metrics.get('contributor_count', 0) == 1:
            recommendations.append("Encourage community contributions with clear guidelines")

        if community_metrics.get('stargazers_count', 0) < 10:
            recommendations.append("Improve project visibility and marketing")

        return recommendations[:10]  # Limit to top 10 recommendations

    def _generate_tags(self, repo: Repository, enrichment_data: Dict[str, Any],
                       indicators: Dict[str, Any]) -> List[str]:
        """Generate descriptive tags for the repository"""
        tags = []

        # Size tags
        if repo.size > 10000:  # > 10MB
            tags.append("large-project")
        elif repo.size < 1000:  # < 1MB
            tags.append("small-project")

        # Activity tags
        activity_metrics = indicators.get('activity', {})
        if activity_metrics.get('commit_frequency', 0) > 1:
            tags.append("active-development")
        elif activity_metrics.get('days_since_last_push', 0) > 180:
            tags.append("inactive")

        # Quality tags
        quality_indicators = indicators.get('code_quality', {}).get('indicators', {})
        if all(quality_indicators.get(k, False) for k in ['has_ci', 'has_tests', 'has_documentation']):
            tags.append("well-maintained")
        if quality_indicators.get('has_ci', False):
            tags.append("ci-enabled")
        if quality_indicators.get('has_tests', False):
            tags.append("tested")

        # Community tags
        community_metrics = indicators.get('community', {})
        if community_metrics.get('stargazers_count', 0) > 1000:
            tags.append("popular")
        if community_metrics.get('contributor_count', 0) > 10:
            tags.append("community-driven")

        # Technology tags
        if repo.language:
            tags.append(repo.language.lower())

        # Type tags
        if repo.is_private:
            tags.append("private")
        else:
            tags.append("open-source")

        if repo.is_fork:
            tags.append("fork")
        if repo.has_wiki:
            tags.append("documentation-rich")
        if repo.has_pages:
            tags.append("website")

        return list(set(tags))  # Remove duplicates

    def _assess_risk(self, repo: Repository, enrichment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess repository risk factors"""
        risk_factors = []
        risk_score = 0

        # Inactivity risk
        if repo.pushed_at:
            last_push = datetime.fromisoformat(repo.pushed_at.replace('Z', '+00:00'))
            days_inactive = (datetime.now(timezone.utc) - last_push).days
            if days_inactive > 365:
                risk_factors.append("High inactivity (over 1 year)")
                risk_score += 30
            elif days_inactive > 90:
                risk_factors.append("Recent inactivity (over 3 months)")
                risk_score += 15

        # Size and complexity risk
        if repo.size > 50000:  # > 50MB
            risk_factors.append("Very large repository")
            risk_score += 20

        # Dependency risk
        commits = enrichment_data.get('commits', [])
        security_commits = sum(1 for commit in commits[:100]
                             if any(keyword in commit.commit.get('message', '').lower()
                                   for keyword in ['vulnerability', 'security', 'cve']))
        if security_commits > 5:
            risk_factors.append("High number of security-related commits")
            risk_score += 25

        # Contributor concentration risk
        contributors = enrichment_data.get('contributors', [])
        if len(contributors) <= 1:
            risk_factors.append("Single contributor (key person risk)")
            risk_score += 20
        elif len(contributors) <= 2:
            risk_factors.append("Very few contributors")
            risk_score += 10

        # Issue management risk
        if repo.open_issues_count > 100:
            risk_factors.append("High number of open issues")
            risk_score += 15

        # License risk
        if not repo.license:
            risk_factors.append("No license file")
            risk_score += 10

        # Determine risk level
        if risk_score >= 60:
            risk_level = "high"
        elif risk_score >= 30:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "mitigation_suggestions": self._generate_mitigation_suggestions(risk_factors)
        }

    def _generate_mitigation_suggestions(self, risk_factors: List[str]) -> List[str]:
        """Generate risk mitigation suggestions"""
        suggestions = []

        for factor in risk_factors:
            if "inactivity" in factor.lower():
                suggestions.append("Establish regular maintenance schedule and community engagement")
            elif "large repository" in factor.lower():
                suggestions.append("Consider repository splitting or modularization")
            elif "security" in factor.lower():
                suggestions.append("Implement regular security audits and dependency updates")
            elif "contributor" in factor.lower():
                suggestions.append("Create contributor guidelines and mentorship programs")
            elif "issues" in factor.lower():
                suggestions.append("Implement issue triage and management processes")
            elif "license" in factor.lower():
                suggestions.append("Add appropriate license file")

        return list(set(suggestions))  # Remove duplicates