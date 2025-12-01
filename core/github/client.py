"""
GitHub API client for repository management and integration
"""

import os
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import base64
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Repository:
    """GitHub repository information"""
    id: int
    name: str
    full_name: str
    description: Optional[str]
    html_url: str
    clone_url: str
    ssh_url: str
    default_branch: str
    language: Optional[str]
    languages: Dict[str, int]
    stargazers_count: int
    forks_count: int
    open_issues_count: int
    size: int
    created_at: str
    updated_at: str
    pushed_at: str
    owner: Dict[str, Any]
    license: Optional[Dict[str, Any]]
    topics: List[str]
    is_private: bool
    is_fork: bool
    has_issues: bool
    has_projects: bool
    has_wiki: bool
    has_pages: bool
    has_downloads: bool
    watchers_count: int
    network_count: int
    subscribers_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert repository to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "full_name": self.full_name,
            "description": self.description,
            "html_url": self.html_url,
            "clone_url": self.clone_url,
            "ssh_url": self.ssh_url,
            "default_branch": self.default_branch,
            "language": self.language,
            "languages": self.languages,
            "stargazers_count": self.stargazers_count,
            "forks_count": self.forks_count,
            "open_issues_count": self.open_issues_count,
            "size": self.size,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "pushed_at": self.pushed_at,
            "owner": self.owner,
            "license": self.license,
            "topics": self.topics,
            "is_private": self.is_private,
            "is_fork": self.is_fork,
            "has_issues": self.has_issues,
            "has_projects": self.has_projects,
            "has_wiki": self.has_wiki,
            "has_pages": self.has_pages,
            "has_downloads": self.has_downloads,
            "watchers_count": self.watchers_count,
            "network_count": self.network_count,
            "subscribers_count": self.subscribers_count
        }


@dataclass
class Commit:
    """GitHub commit information"""
    sha: str
    message: str
    author: Dict[str, Any]
    committer: Dict[str, Any]
    tree: Dict[str, Any]
    url: str
    html_url: str
    parents: List[Dict[str, Any]]
    stats: Optional[Dict[str, int]]
    files: Optional[List[Dict[str, Any]]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert commit to dictionary"""
        return {
            "sha": self.sha,
            "message": self.message,
            "author": self.author,
            "committer": self.committer,
            "tree": self.tree,
            "url": self.url,
            "html_url": self.html_url,
            "parents": self.parents,
            "stats": self.stats,
            "files": self.files
        }


class GitHubClient:
    """GitHub API client for repository operations"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize GitHub client with configuration"""
        self.config = config
        self.token = config.get('github_token', os.getenv('GITHUB_TOKEN'))
        self.api_base_url = config.get('api_base_url', 'https://api.github.com')
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        self.rate_limit_delay = config.get('rate_limit_delay', 1)

        if not self.token:
            logger.warning("GitHub token not provided. API rate limits will be very restrictive.")

        self._session = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def _ensure_session(self):
        """Ensure aiohttp session is created"""
        if self._session is None or self._session.closed:
            headers = {
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'CodeAnalyzer-Backend/1.0'
            }

            if self.token:
                headers['Authorization'] = f'token {self.token}'

            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout
            )

    async def close(self):
        """Close aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated request to GitHub API"""
        await self._ensure_session()

        url = f"{self.api_base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        # Add pagination support
        all_items = []
        page = 1
        per_page = 100

        while True:
            params = kwargs.get('params', {})
            params['page'] = page
            params['per_page'] = per_page
            kwargs['params'] = params

            async with self._session.request(method, url, **kwargs) as response:
                # Handle rate limiting
                if response.status == 403:
                    rate_limit_remaining = response.headers.get('X-RateLimit-Remaining', '0')
                    if rate_limit_remaining == '0':
                        reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                        current_time = int(datetime.now().timestamp())
                        sleep_time = max(reset_time - current_time, 60)
                        logger.info(f"Rate limit reached. Waiting {sleep_time} seconds...")
                        await asyncio.sleep(sleep_time)
                        continue

                response.raise_for_status()

                if response.content_type == 'application/json':
                    data = await response.json()

                    # Handle paginated responses
                    if isinstance(data, list):
                        all_items.extend(data)
                        if len(data) < per_page:
                            break
                        page += 1
                    else:
                        return data
                else:
                    return await response.text()

        return all_items

    async def get_repository(self, owner: str, repo: str) -> Optional[Repository]:
        """
        Get repository information

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            Repository object or None if not found
        """
        try:
            endpoint = f"repos/{owner}/{repo}"
            data = await self._make_request('GET', endpoint)

            return Repository(
                id=data['id'],
                name=data['name'],
                full_name=data['full_name'],
                description=data.get('description'),
                html_url=data['html_url'],
                clone_url=data['clone_url'],
                ssh_url=data['ssh_url'],
                default_branch=data['default_branch'],
                language=data.get('language'),
                languages={},  # Will be populated separately
                stargazers_count=data['stargazers_count'],
                forks_count=data['forks_count'],
                open_issues_count=data['open_issues_count'],
                size=data['size'],
                created_at=data['created_at'],
                updated_at=data['updated_at'],
                pushed_at=data['pushed_at'],
                owner=data['owner'],
                license=data.get('license'),
                topics=data.get('topics', []),
                is_private=data['private'],
                is_fork=data['fork'],
                has_issues=data['has_issues'],
                has_projects=data['has_projects'],
                has_wiki=data['has_wiki'],
                has_pages=data['has_pages'],
                has_downloads=data['has_downloads'],
                watchers_count=data['watchers_count'],
                network_count=data['network_count'],
                subscribers_count=data['subscribers_count']
            )
        except Exception as e:
            logger.error(f"Failed to get repository {owner}/{repo}: {e}")
            return None

    async def get_repository_languages(self, owner: str, repo: str) -> Dict[str, int]:
        """Get repository language breakdown"""
        try:
            endpoint = f"repos/{owner}/{repo}/languages"
            return await self._make_request('GET', endpoint)
        except Exception as e:
            logger.error(f"Failed to get languages for {owner}/{repo}: {e}")
            return {}

    async def get_repository_commits(self, owner: str, repo: str,
                                    sha: Optional[str] = None,
                                    per_page: int = 100,
                                    max_commits: int = 1000) -> List[Commit]:
        """
        Get repository commits

        Args:
            owner: Repository owner
            repo: Repository name
            sha: Branch or commit SHA (defaults to default branch)
            per_page: Number of commits per page
            max_commits: Maximum number of commits to retrieve

        Returns:
            List of Commit objects
        """
        try:
            endpoint = f"repos/{owner}/{repo}/commits"
            params = {}
            if sha:
                params['sha'] = sha
            params['per_page'] = min(per_page, 100)

            data = await self._make_request('GET', endpoint, params=params)

            # Limit results
            commits = []
            for i, commit_data in enumerate(data[:max_commits]):
                commit = Commit(
                    sha=commit_data['sha'],
                    message=commit_data['commit']['message'],
                    author=commit_data['commit']['author'],
                    committer=commit_data['commit']['committer'],
                    tree=commit_data['commit']['tree'],
                    url=commit_data['url'],
                    html_url=commit_data['html_url'],
                    parents=commit_data['parents'],
                    stats=commit_data.get('stats'),
                    files=commit_data.get('files')
                )
                commits.append(commit)

            return commits
        except Exception as e:
            logger.error(f"Failed to get commits for {owner}/{repo}: {e}")
            return []

    async def get_commit_details(self, owner: str, repo: str, sha: str) -> Optional[Commit]:
        """Get detailed commit information including file changes"""
        try:
            endpoint = f"repos/{owner}/{repo}/commits/{sha}"
            data = await self._make_request('GET', endpoint)

            return Commit(
                sha=data['sha'],
                message=data['commit']['message'],
                author=data['commit']['author'],
                committer=data['commit']['committer'],
                tree=data['commit']['tree'],
                url=data['url'],
                html_url=data['html_url'],
                parents=data['parents'],
                stats=data.get('stats'),
                files=data.get('files')
            )
        except Exception as e:
            logger.error(f"Failed to get commit {sha} for {owner}/{repo}: {e}")
            return None

    async def get_repository_contents(self, owner: str, repo: str,
                                     path: str = "",
                                     ref: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get repository directory contents"""
        try:
            endpoint = f"repos/{owner}/{repo}/contents/{path}"
            params = {}
            if ref:
                params['ref'] = ref

            data = await self._make_request('GET', endpoint, params=params)

            if isinstance(data, dict) and data.get('type') == 'file':
                return [data]
            elif isinstance(data, list):
                return data
            else:
                return []
        except Exception as e:
            logger.error(f"Failed to get contents for {owner}/{repo}/{path}: {e}")
            return []

    async def get_file_content(self, owner: str, repo: str, path: str,
                             ref: Optional[str] = None) -> Optional[str]:
        """Get file content as string"""
        try:
            endpoint = f"repos/{owner}/{repo}/contents/{path}"
            params = {}
            if ref:
                params['ref'] = ref

            data = await self._make_request('GET', endpoint, params=params)

            if data.get('type') == 'file' and data.get('content'):
                content = base64.b64decode(data['content']).decode('utf-8')
                return content
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to get file content for {owner}/{repo}/{path}: {e}")
            return None

    async def search_repositories(self, query: str, sort: str = "updated",
                                 order: str = "desc", per_page: int = 100) -> List[Dict[str, Any]]:
        """
        Search repositories

        Args:
            query: Search query
            sort: Sort field (stars, forks, updated)
            order: Sort order (asc, desc)
            per_page: Results per page

        Returns:
            List of repository data
        """
        try:
            endpoint = "search/repositories"
            params = {
                'q': query,
                'sort': sort,
                'order': order,
                'per_page': min(per_page, 100)
            }

            data = await self._make_request('GET', endpoint, params=params)
            return data.get('items', [])
        except Exception as e:
            logger.error(f"Failed to search repositories with query '{query}': {e}")
            return []

    async def get_user_repositories(self, username: str,
                                  type: str = "all",
                                  sort: str = "updated",
                                  direction: str = "desc") -> List[Dict[str, Any]]:
        """
        Get user repositories

        Args:
            username: GitHub username
            type: Repository type (all, owner, member)
            sort: Sort field (created, updated, pushed, full_name)
            direction: Sort direction (asc, desc)

        Returns:
            List of repository data
        """
        try:
            endpoint = f"users/{username}/repos"
            params = {
                'type': type,
                'sort': sort,
                'direction': direction
            }

            return await self._make_request('GET', endpoint, params=params)
        except Exception as e:
            logger.error(f"Failed to get repositories for user {username}: {e}")
            return []

    async def create_webhook(self, owner: str, repo: str,
                           webhook_url: str,
                           events: List[str],
                           secret: Optional[str] = None) -> Dict[str, Any]:
        """
        Create repository webhook

        Args:
            owner: Repository owner
            repo: Repository name
            webhook_url: URL to receive webhook events
            events: List of events to subscribe to
            secret: Optional secret for webhook signature verification

        Returns:
            Webhook data
        """
        try:
            endpoint = f"repos/{owner}/{repo}/hooks"

            payload = {
                "name": "web",
                "active": True,
                "events": events,
                "config": {
                    "url": webhook_url,
                    "content_type": "json"
                }
            }

            if secret:
                payload["config"]["secret"] = secret

            return await self._make_request('POST', endpoint, json=payload)
        except Exception as e:
            logger.error(f"Failed to create webhook for {owner}/{repo}: {e}")
            raise

    async def list_webhooks(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """List repository webhooks"""
        try:
            endpoint = f"repos/{owner}/{repo}/hooks"
            return await self._make_request('GET', endpoint)
        except Exception as e:
            logger.error(f"Failed to list webhooks for {owner}/{repo}: {e}")
            return []

    async def delete_webhook(self, owner: str, repo: str, hook_id: int) -> bool:
        """Delete repository webhook"""
        try:
            endpoint = f"repos/{owner}/{repo}/hooks/{hook_id}"
            await self._make_request('DELETE', endpoint)
            return True
        except Exception as e:
            logger.error(f"Failed to delete webhook {hook_id} for {owner}/{repo}: {e}")
            return False

    async def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get current rate limit information"""
        try:
            endpoint = "rate_limit"
            return await self._make_request('GET', endpoint)
        except Exception as e:
            logger.error(f"Failed to get rate limit info: {e}")
            return {}