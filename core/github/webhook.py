"""
GitHub webhook handler for real-time repository analysis
Handles push, pull request, and other GitHub events for automatic scanning
"""

import os
import hmac
import hashlib
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import json

from .client import GitHubClient, Commit
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WebhookEvent:
    """GitHub webhook event"""
    event_type: str
    delivery_id: str
    payload: Dict[str, Any]
    repository: Dict[str, Any]
    action: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class PushEvent:
    """GitHub push event details"""
    ref: str
    before_sha: str
    after_sha: str
    forced: bool
    commits: List[Dict[str, Any]]
    head_commit: Optional[Dict[str, Any]]
    repository: Dict[str, Any]
    pusher: Dict[str, Any]
    sender: Dict[str, Any]


@dataclass
class PullRequestEvent:
    """GitHub pull request event details"""
    action: str
    number: int
    pull_request: Dict[str, Any]
    repository: Dict[str, Any]
    sender: Dict[str, Any]


class WebhookHandler:
    """GitHub webhook event handler"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize webhook handler with configuration"""
        self.config = config
        self.secret = config.get('webhook_secret', os.getenv('GITHUB_WEBHOOK_SECRET'))
        self.github_client = None
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.analysis_queue = asyncio.Queue()
        self._processing_task = None

        # Event priorities (lower number = higher priority)
        self.event_priorities = {
            'push': 1,
            'pull_request': 2,
            'issues': 3,
            'release': 4,
            'create': 5,
            'delete': 6,
            'fork': 7,
            'watch': 8
        }

    async def start(self):
        """Start webhook handler and background processing"""
        self.github_client = GitHubClient(self.config.get('github', {}))
        await self.github_client._ensure_session()

        # Start background processing task
        if not self._processing_task:
            self._processing_task = asyncio.create_task(self._process_events())

        logger.info("Webhook handler started")

    async def stop(self):
        """Stop webhook handler and cleanup"""
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
            self._processing_task = None

        if self.github_client:
            await self.github_client.close()

        logger.info("Webhook handler stopped")

    def register_handler(self, event_type: str, handler: Callable):
        """Register event handler for specific event type"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type} events")

    def unregister_handler(self, event_type: str, handler: Callable):
        """Unregister event handler"""
        if event_type in self.event_handlers:
            try:
                self.event_handlers[event_type].remove(handler)
                logger.info(f"Unregistered handler for {event_type} events")
            except ValueError:
                logger.warning(f"Handler not found for {event_type} events")

    def verify_signature(self, payload_body: bytes, signature_header: str) -> bool:
        """
        Verify GitHub webhook signature

        Args:
            payload_body: Raw webhook payload bytes
            signature_header: X-Hub-Signature-256 header value

        Returns:
            True if signature is valid
        """
        if not self.secret:
            logger.warning("No webhook secret configured, skipping signature verification")
            return True

        if not signature_header:
            logger.error("No signature header provided")
            return False

        # Extract hash algorithm and signature
        try:
            algorithm, signature = signature_header.split('=', 1)
        except ValueError:
            logger.error(f"Invalid signature header format: {signature_header}")
            return False

        if algorithm != 'sha256':
            logger.error(f"Unsupported signature algorithm: {algorithm}")
            return False

        # Calculate expected signature
        expected_signature = hmac.new(
            self.secret.encode('utf-8'),
            payload_body,
            hashlib.sha256
        ).hexdigest()

        # Compare signatures securely
        return hmac.compare_digest(signature, expected_signature)

    def parse_webhook_event(self, event_type: str, delivery_id: str,
                          payload_body: str) -> Optional[WebhookEvent]:
        """
        Parse webhook event payload

        Args:
            event_type: GitHub event type
            delivery_id: X-GitHub-Delivery header
            payload_body: Raw JSON payload

        Returns:
            WebhookEvent object or None if parsing fails
        """
        try:
            payload = json.loads(payload_body)
            repository = payload.get('repository', {})
            action = payload.get('action')

            return WebhookEvent(
                event_type=event_type,
                delivery_id=delivery_id,
                payload=payload,
                repository=repository,
                action=action
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse webhook payload: {e}")
            return None

    async def handle_webhook(self, event_type: str, delivery_id: str,
                           payload_body: str) -> Dict[str, Any]:
        """
        Handle incoming webhook event

        Args:
            event_type: GitHub event type
            delivery_id: X-GitHub-Delivery header
            payload_body: Raw JSON payload

        Returns:
            Processing result
        """
        try:
            # Parse event
            event = self.parse_webhook_event(event_type, delivery_id, payload_body)
            if not event:
                return {"success": False, "error": "Failed to parse webhook payload"}

            # Queue event for processing
            priority = self.event_priorities.get(event_type, 10)
            await self.analysis_queue.put((priority, event))

            logger.info(f"Queued {event_type} event {delivery_id} with priority {priority}")

            return {"success": True, "queued": True, "event_id": delivery_id}

        except Exception as e:
            logger.error(f"Failed to handle webhook {delivery_id}: {e}")
            return {"success": False, "error": str(e)}

    async def _process_events(self):
        """Background task to process queued events"""
        logger.info("Started webhook event processing task")

        pending_events = []

        while True:
            try:
                # Wait for event or process pending events
                try:
                    timeout = 1.0  # Process every second if events are pending
                    priority, event = await asyncio.wait_for(
                        self.analysis_queue.get(),
                        timeout=timeout
                    )
                    pending_events.append((priority, event))
                except asyncio.TimeoutError:
                    pass

                # Sort events by priority
                pending_events.sort(key=lambda x: x[0])

                # Process highest priority events
                processed_count = 0
                max_events_per_batch = 5

                while pending_events and processed_count < max_events_per_batch:
                    priority, event = pending_events.pop(0)
                    await self._process_single_event(event)
                    processed_count += 1

                # Small delay to prevent CPU spinning
                if not pending_events:
                    await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event processing task: {e}")
                await asyncio.sleep(1)

        logger.info("Stopped webhook event processing task")

    async def _process_single_event(self, event: WebhookEvent):
        """Process a single webhook event"""
        try:
            logger.info(f"Processing {event.event_type} event {event.delivery_id}")

            # Get handlers for this event type
            handlers = self.event_handlers.get(event.event_type, [])
            if not handlers:
                logger.debug(f"No handlers registered for {event.event_type}")
                return

            # Call all handlers
            tasks = []
            for handler in handlers:
                try:
                    task = asyncio.create_task(handler(event))
                    tasks.append(task)
                except Exception as e:
                    logger.error(f"Failed to create task for handler: {e}")

            # Wait for all handlers to complete
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Handler {i} failed: {result}")

            logger.info(f"Completed processing {event.event_type} event {event.delivery_id}")

        except Exception as e:
            logger.error(f"Failed to process event {event.delivery_id}: {e}")

    async def handle_push_event(self, event: WebhookEvent) -> Optional[PushEvent]:
        """Handle push event and extract commit information"""
        try:
            payload = event.payload
            ref = payload.get('ref', '')
            before_sha = payload.get('before', '')
            after_sha = payload.get('after', '')
            forced = payload.get('forced', False)
            commits = payload.get('commits', [])
            head_commit = payload.get('head_commit')
            repository = payload.get('repository', {})
            pusher = payload.get('pusher', {})
            sender = payload.get('sender', {})

            push_event = PushEvent(
                ref=ref,
                before_sha=before_sha,
                after_sha=after_sha,
                forced=forced,
                commits=commits,
                head_commit=head_commit,
                repository=repository,
                pusher=pusher,
                sender=sender
            )

            # Extract changed files from commits
            changed_files = set()
            for commit in commits:
                if 'added' in commit:
                    changed_files.update(commit['added'])
                if 'modified' in commit:
                    changed_files.update(commit['modified'])
                if 'removed' in commit:
                    changed_files.update(commit['removed'])

            logger.info(f"Push event: {len(commits)} commits, {len(changed_files)} files changed")
            return push_event

        except Exception as e:
            logger.error(f"Failed to handle push event: {e}")
            return None

    async def handle_pull_request_event(self, event: WebhookEvent) -> Optional[PullRequestEvent]:
        """Handle pull request event"""
        try:
            payload = event.payload
            action = payload.get('action', '')
            number = payload.get('number', 0)
            pull_request = payload.get('pull_request', {})
            repository = payload.get('repository', {})
            sender = payload.get('sender', {})

            pr_event = PullRequestEvent(
                action=action,
                number=number,
                pull_request=pull_request,
                repository=repository,
                sender=sender
            )

            logger.info(f"Pull request event: {action} #{number}")
            return pr_event

        except Exception as e:
            logger.error(f"Failed to handle pull request event: {e}")
            return None

    async def get_commit_files(self, owner: str, repo: str, sha: str) -> List[Dict[str, Any]]:
        """Get files changed in a commit"""
        try:
            if not self.github_client:
                logger.error("GitHub client not initialized")
                return []

            commit = await self.github_client.get_commit_details(owner, repo, sha)
            if commit and commit.files:
                return commit.files
            return []

        except Exception as e:
            logger.error(f"Failed to get files for commit {sha}: {e}")
            return []

    async def get_pull_request_files(self, owner: str, repo: str, pr_number: int) -> List[Dict[str, Any]]:
        """Get files changed in a pull request"""
        try:
            if not self.github_client:
                logger.error("GitHub client not initialized")
                return []

            # Use GitHub REST API to get PR files
            endpoint = f"repos/{owner}/{repo}/pulls/{pr_number}/files"
            data = await self.github_client._make_request('GET', endpoint)
            return data if isinstance(data, list) else []

        except Exception as e:
            logger.error(f"Failed to get files for PR #{pr_number}: {e}")
            return []

    async def create_default_handlers(self, analysis_callback: Callable):
        """Create default event handlers for repository analysis"""
        async def push_handler(event: WebhookEvent):
            """Handler for push events"""
            push_event = await self.handle_push_event(event)
            if push_event and push_event.repository:
                repo_data = push_event.repository
                owner = repo_data['owner']['login']
                repo = repo_data['name']

                # Trigger analysis for changed files
                if push_event.head_commit:
                    files = await self.get_commit_files(owner, repo, push_event.head_commit['id'])
                    await analysis_callback(
                        owner=owner,
                        repo=repo,
                        event_type='push',
                        files=files,
                        commit_sha=push_event.head_commit['id']
                    )

        async def pull_request_handler(event: WebhookEvent):
            """Handler for pull request events"""
            pr_event = await self.handle_pull_request_event(event)
            if pr_event and pr_event.repository:
                repo_data = pr_event.repository
                owner = repo_data['owner']['login']
                repo = repo_data['name']

                # Trigger analysis for PR if opened, synchronized, or reopened
                if pr_event.action in ['opened', 'synchronize', 'reopened']:
                    files = await self.get_pull_request_files(owner, repo, pr_event.number)
                    await analysis_callback(
                        owner=owner,
                        repo=repo,
                        event_type='pull_request',
                        files=files,
                        pr_number=pr_event.number,
                        action=pr_event.action
                    )

        # Register default handlers
        self.register_handler('push', push_handler)
        self.register_handler('pull_request', pull_request_handler)

        logger.info("Created default webhook handlers")