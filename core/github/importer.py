"""
GitHub repository importer for importing and analyzing public repositories
Supports cloning, initial analysis, and database storage
"""

import os
import asyncio
import subprocess
import tempfile
import shutil
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import uuid
from datetime import datetime, timezone

from .client import GitHubClient, Repository
from utils.logger import get_logger

logger = get_logger(__name__)


class RepositoryImporter:
    """GitHub repository importer with analysis capabilities"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize repository importer with configuration"""
        self.config = config
        self.github_client = GitHubClient(config.get('github', {}))
        self.clone_timeout = config.get('clone_timeout', 300)  # 5 minutes
        self.max_repo_size = config.get('max_repo_size_mb', 1000)  # 1GB
        self.temp_dir = config.get('temp_dir', tempfile.gettempdir())
        self.preserve_cloned_repos = config.get('preserve_cloned_repos', False)

        # Analysis modules (will be injected)
        self.security_scanner = None
        self.linting_scanner = None
        self.quality_scanner = None

    def set_analysis_modules(self, security_scanner=None, linting_scanner=None, quality_scanner=None):
        """Set analysis modules for repository scanning"""
        self.security_scanner = security_scanner
        self.linting_scanner = linting_scanner
        self.quality_scanner = quality_scanner

    async def import_repository(self, owner: str, repo: str,
                               branch: Optional[str] = None,
                               analyze: bool = True) -> Dict[str, Any]:
        """
        Import and optionally analyze a GitHub repository

        Args:
            owner: Repository owner
            repo: Repository name
            branch: Specific branch to import (defaults to default branch)
            analyze: Whether to run analysis after import

        Returns:
            Import result with metadata and analysis results
        """
        import_id = str(uuid.uuid4())
        logger.info(f"Starting repository import: {owner}/{repo} (ID: {import_id})")

        try:
            async with self.github_client:
                # Get repository information
                repo_data = await self.github_client.get_repository(owner, repo)
                if not repo_data:
                    raise ValueError(f"Repository {owner}/{repo} not found")

                # Check repository size
                if repo_data.size > self.max_repo_size * 1024:  # GitHub API returns size in KB
                    raise ValueError(f"Repository too large: {repo_data.size}KB > {self.max_repo_size}MB")

                # Clone repository
                clone_path = await self._clone_repository(repo_data, branch)
                if not clone_path:
                    raise RuntimeError(f"Failed to clone repository {owner}/{repo}")

                try:
                    # Get additional metadata
                    repo_data.languages = await self.github_client.get_repository_languages(owner, repo)
                    recent_commits = await self.github_client.get_repository_commits(owner, repo, per_page=10)

                    # Build import result
                    result = {
                        "import_id": import_id,
                        "status": "success",
                        "repository": repo_data.to_dict(),
                        "clone_path": clone_path,
                        "languages": repo_data.languages,
                        "recent_commits": [commit.to_dict() for commit in recent_commits],
                        "import_timestamp": datetime.now(timezone.utc).isoformat(),
                        "analysis_results": None
                    }

                    # Run analysis if requested
                    if analyze:
                        result["analysis_results"] = await self._analyze_repository(
                            clone_path, import_id, repo_data.to_dict()
                        )

                    logger.info(f"Successfully imported repository {owner}/{repo}")
                    return result

                finally:
                    # Clean up cloned repository
                    if not self.preserve_cloned_repos and os.path.exists(clone_path):
                        shutil.rmtree(clone_path, ignore_errors=True)
                        logger.info(f"Cleaned up cloned repository at {clone_path}")

        except Exception as e:
            logger.error(f"Failed to import repository {owner}/{repo}: {e}")
            return {
                "import_id": import_id,
                "status": "error",
                "error": str(e),
                "repository": None,
                "import_timestamp": datetime.now(timezone.utc).isoformat()
            }

    async def import_multiple_repositories(self, repositories: List[Dict[str, Any]],
                                           max_concurrent: int = 3) -> List[Dict[str, Any]]:
        """
        Import multiple repositories concurrently

        Args:
            repositories: List of repository specifications
            max_concurrent: Maximum concurrent imports

        Returns:
            List of import results
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def import_with_semaphore(repo_spec: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                return await self.import_repository(
                    repo_spec['owner'],
                    repo_spec['repo'],
                    repo_spec.get('branch'),
                    repo_spec.get('analyze', True)
                )

        tasks = [import_with_semaphore(repo_spec) for repo_spec in repositories]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                results[i] = {
                    "import_id": str(uuid.uuid4()),
                    "status": "error",
                    "error": str(result),
                    "repository": repositories[i],
                    "import_timestamp": datetime.now(timezone.utc).isoformat()
                }

        return results

    async def search_and_import(self, query: str, max_repos: int = 10,
                                sort: str = "stars", analyze: bool = True) -> List[Dict[str, Any]]:
        """
        Search repositories and import results

        Args:
            query: Search query
            max_repos: Maximum repositories to import
            sort: Sort field (stars, forks, updated)
            analyze: Whether to analyze imported repositories

        Returns:
            List of import results
        """
        logger.info(f"Searching repositories with query: {query}")

        try:
            async with self.github_client:
                # Search repositories
                search_results = await self.github_client.search_repositories(
                    query=query,
                    sort=sort,
                    per_page=max_repos
                )

                if not search_results:
                    logger.info(f"No repositories found for query: {query}")
                    return []

                # Prepare repository specifications for import
                repo_specs = []
                for repo_data in search_results[:max_repos]:
                    repo_specs.append({
                        "owner": repo_data['owner']['login'],
                        "repo": repo_data['name'],
                        "analyze": analyze
                    })

                logger.info(f"Found {len(repo_specs)} repositories, starting import")

                # Import repositories
                return await self.import_multiple_repositories(repo_specs)

        except Exception as e:
            logger.error(f"Failed to search and import repositories: {e}")
            return []

    async def _clone_repository(self, repo_data: Repository, branch: Optional[str] = None) -> Optional[str]:
        """Clone repository to local directory"""
        try:
            # Create temporary directory for clone
            clone_dir = tempfile.mkdtemp(prefix=f"github_clone_{repo_data.name}_", dir=self.temp_dir)
            clone_url = repo_data.clone_url

            # Build git clone command
            cmd = ['git', 'clone', '--depth', '1']  # Shallow clone
            if branch:
                cmd.extend(['--branch', branch])
            cmd.extend([clone_url, clone_dir])

            logger.info(f"Cloning repository {repo_data.full_name} to {clone_dir}")

            # Execute clone command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.clone_timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise RuntimeError(f"Repository clone timed out after {self.clone_timeout} seconds")

            if process.returncode != 0:
                error_msg = stderr.decode('utf-8') if stderr else "Unknown git error"
                raise RuntimeError(f"Git clone failed: {error_msg}")

            logger.info(f"Successfully cloned repository {repo_data.full_name}")
            return clone_dir

        except Exception as e:
            logger.error(f"Failed to clone repository {repo_data.full_name}: {e}")
            # Clean up on failure
            if 'clone_dir' in locals() and os.path.exists(clone_dir):
                shutil.rmtree(clone_dir, ignore_errors=True)
            return None

    async def _analyze_repository(self, repo_path: str, repo_id: str,
                                 repo_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Run analysis on cloned repository"""
        analysis_results = {
            "repo_id": repo_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "security_analysis": None,
            "linting_analysis": None,
            "quality_analysis": None
        }

        # Run security analysis
        if self.security_scanner:
            try:
                logger.info(f"Running security analysis on repository {repo_id}")
                security_report = await self.security_scanner.scan_repository(repo_path, repo_id)
                analysis_results["security_analysis"] = security_report.to_dict()
            except Exception as e:
                logger.error(f"Security analysis failed: {e}")
                analysis_results["security_analysis"] = {
                    "error": str(e),
                    "status": "failed"
                }

        # Run linting analysis
        if self.linting_scanner:
            try:
                logger.info(f"Running linting analysis on repository {repo_id}")
                # linting_scanner should have appropriate method
                linting_report = await self.linting_scanner.scan_repository(repo_path, repo_id)
                analysis_results["linting_analysis"] = linting_report
            except Exception as e:
                logger.error(f"Linting analysis failed: {e}")
                analysis_results["linting_analysis"] = {
                    "error": str(e),
                    "status": "failed"
                }

        # Run quality analysis
        if self.quality_scanner:
            try:
                logger.info(f"Running quality analysis on repository {repo_id}")
                # quality_scanner should have appropriate method
                quality_report = await self.quality_scanner.scan_repository(repo_path, repo_id)
                analysis_results["quality_analysis"] = quality_report
            except Exception as e:
                logger.error(f"Quality analysis failed: {e}")
                analysis_results["quality_analysis"] = {
                    "error": str(e),
                    "status": "failed"
                }

        return analysis_results

    async def get_repository_statistics(self, owner: str, repo: str) -> Dict[str, Any]:
        """Get repository statistics without full import"""
        try:
            async with self.github_client:
                # Get basic repository info
                repo_data = await self.github_client.get_repository(owner, repo)
                if not repo_data:
                    return {"error": "Repository not found"}

                # Get additional statistics
                languages = await self.github_client.get_repository_languages(owner, repo)
                recent_commits = await self.github_client.get_repository_commits(owner, repo, per_page=100)
                rate_limit_info = await self.github_client.get_rate_limit_info()

                # Calculate statistics
                total_commits = len(recent_commits)
                commit_activity = self._calculate_commit_activity(recent_commits)
                language_diversity = len(languages) if languages else 0
                primary_language = max(languages.items(), key=lambda x: x[1])[0] if languages else None

                return {
                    "repository": repo_data.to_dict(),
                    "languages": languages,
                    "statistics": {
                        "total_commits": total_commits,
                        "language_diversity": language_diversity,
                        "primary_language": primary_language,
                        "commit_activity": commit_activity
                    },
                    "rate_limit": rate_limit_info,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

        except Exception as e:
            logger.error(f"Failed to get repository statistics for {owner}/{repo}: {e}")
            return {"error": str(e)}

    def _calculate_commit_activity(self, commits: List) -> Dict[str, int]:
        """Calculate commit activity statistics"""
        activity = {"daily": 0, "weekly": 0, "monthly": 0}
        now = datetime.now(timezone.utc)

        for commit in commits:
            commit_date = datetime.fromisoformat(
                commit.commit['committer']['date'].replace('Z', '+00:00')
            )
            days_ago = (now - commit_date).days

            if days_ago <= 1:
                activity["daily"] += 1
            if days_ago <= 7:
                activity["weekly"] += 1
            if days_ago <= 30:
                activity["monthly"] += 1

        return activity

    async def cleanup_temporary_files(self):
        """Clean up temporary files and directories"""
        try:
            temp_pattern = os.path.join(self.temp_dir, "github_clone_*")
            temp_dirs = [d for d in Path(self.temp_dir).glob("github_clone_*") if d.is_dir()]

            for temp_dir in temp_dirs:
                # Check if directory is older than 1 hour
                dir_age = datetime.now().timestamp() - temp_dir.stat().st_mtime
                if dir_age > 3600:  # 1 hour
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    logger.info(f"Cleaned up old temporary directory: {temp_dir}")

        except Exception as e:
            logger.warning(f"Failed to cleanup temporary files: {e}")