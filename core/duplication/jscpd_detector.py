"""
JSCPD detector for exact duplicate code detection
Integrates with jscpd tool for JavaScript/TypeScript and other languages
"""

import os
import asyncio
import json
import subprocess
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import hashlib

from .types import DuplicationGroup, DuplicateBlock, DuplicationType
from utils.logger import get_logger

logger = get_logger(__name__)


class JscpdDetector:
    """JSCPD duplicate code detector integration"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize JSCPD detector with configuration"""
        self.config = config or {}
        self.min_lines = config.get('min_lines', 5)
        self.max_lines = config.get('max_lines', 1000)
        self.min_tokens = config.get('min_tokens', 50)
        self.ignore_patterns = config.get('ignore_patterns', [
            'coverage/**',
            'node_modules/**',
            'vendor/**',
            'dist/**',
            'build/**',
            'test/**',
            'tests/**',
            '**/*.min.js',
            '**/*.min.css',
            '**/*.d.ts'
        ])
        self.formats = config.get('formats', [
            'javascript', 'typescript', 'jsx', 'tsx', 'python', 'java',
            'csharp', 'go', 'ruby', 'php', 'scala', 'swift', 'kotlin'
        ])
        self.timeout = config.get('timeout', 120)  # 2 minutes

    async def scan_for_exact_duplicates(self, file_paths: List[str]) -> List[DuplicationGroup]:
        """
        Scan files for exact duplicate code using JSCPD

        Args:
            file_paths: List of file paths to scan

        Returns:
            List of DuplicationGroup objects with exact duplicates
        """
        logger.info(f"Starting JSCPD exact duplicate scan for {len(file_paths)} files")

        try:
            # Check if jscpd is available
            if not await self._is_jscpd_available():
                logger.warning("JSCPD not available, skipping exact duplicate detection")
                return []

            # Filter files for supported formats
            supported_files = self._filter_supported_files(file_paths)
            if not supported_files:
                logger.info("No supported files for JSCPD scanning")
                return []

            # Run JSCPD scan
            jscpd_output = await self._run_jscpd_scan(supported_files)

            # Parse results
            duplicate_groups = self._parse_jscpd_output(jscpd_output)

            logger.info(f"JSCPD scan completed. Found {len(duplicate_groups)} duplicate groups")
            return duplicate_groups

        except Exception as e:
            logger.error(f"JSCPD exact duplicate scan failed: {e}")
            return []

    async def _is_jscpd_available(self) -> bool:
        """Check if JSCPD CLI is available"""
        try:
            process = await asyncio.create_subprocess_exec(
                'jscpd', '--version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            return process.returncode == 0
        except Exception:
            return False

    def _filter_supported_files(self, file_paths: List[str]) -> List[str]:
        """Filter files for supported formats and ignore patterns"""
        supported_extensions = {
            '.js', '.jsx', '.ts', '.tsx', '.py', '.java', '.cs', '.go',
            '.rb', '.php', '.scala', '.swift', '.kt', '.m', '.rs',
            '.dart', '.lua', '.pl', '.pm', '.r', '.R', '.cpp', '.c',
            '.h', '.hpp', '.cxx', '.cc'
        }

        supported_files = []
        for file_path in file_paths:
            if self._is_ignored_file(file_path):
                continue

            if Path(file_path).suffix.lower() in supported_extensions:
                supported_files.append(file_path)

        return supported_files

    def _is_ignored_file(self, file_path: str) -> bool:
        """Check if file should be ignored"""
        file_path = file_path.replace('\\', '/')
        for pattern in self.ignore_patterns:
            if '*/' in pattern:
                if pattern[2:] in file_path:
                    return True
            elif pattern.endswith('/**'):
                if file_path.startswith(pattern[:-3]):
                    return True
            elif '**/' in pattern:
                if pattern[3:] in file_path:
                    return True
            elif pattern in file_path:
                return True
        return False

    async def _run_jscpd_scan(self, file_paths: List[str]) -> Dict[str, Any]:
        """Execute JSCPD scan and return JSON output"""
        try:
            # Create temporary config file
            config = {
                "threshold": 0,  # Find all duplicates
                "reporters": ["json"],
                "ignore": self.ignore_patterns,
                "format": self.formats,
                "minLines": self.min_lines,
                "maxLines": self.max_lines,
                "minTokens": self.min_tokens,
                "gitignore": True,
                "path": [str(Path(fp).parent) for fp in file_paths[:5]]  # Limit to prevent command line length
            }

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as config_file:
                json.dump(config, config_file)
                config_file_path = config_file.name

            try:
                # Build JSCPD command
                cmd = [
                    'jscpd',
                    '--config', config_file_path,
                    '--output', tempfile.gettempdir(),
                    '--json'
                ]

                # Add individual files (limit to prevent command line issues)
                for file_path in file_paths[:50]:  # Limit to prevent command line length
                    cmd.append(file_path)

                # Execute JSCPD
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )

                if process.returncode != 0 and process.returncode != 1:
                    # JSCPD returns 1 when duplicates are found
                    logger.warning(f"JSCPD execution failed: {stderr.decode()}")
                    return {"duplication": {"files": []}}

                # Read JSON output
                output_file = os.path.join(tempfile.gettempdir(), 'jscpd-report.json')
                if os.path.exists(output_file):
                    with open(output_file, 'r') as f:
                        output = json.load(f)
                    os.remove(output_file)  # Clean up
                    return output
                else:
                    return {"duplication": {"files": []}}

            finally:
                os.unlink(config_file_path)

        except asyncio.TimeoutError:
            logger.error(f"JSCPD scan timed out after {self.timeout} seconds")
            return {"duplication": {"files": []}}
        except Exception as e:
            logger.error(f"Failed to run JSCPD scan: {e}")
            return {"duplication": {"files": []}}

    def _parse_jscpd_output(self, jscpd_output: Dict[str, Any]) -> List[DuplicationGroup]:
        """Parse JSCPD JSON output into DuplicationGroup objects"""
        duplicate_groups = []

        try:
            duplication_data = jscpd_output.get('duplication', {})
            files_data = duplication_data.get('files', [])

            # Group duplicates by similarity
            duplication_map = {}

            for file_data in files_data:
                try:
                    for dup in file_data.get('duplication', []):
                        # Create hash for this duplicate block
                        content = dup.get('fragment', '')
                        content_hash = hashlib.md5(content.encode()).hexdigest()
                        similarity = dup.get('similarity', 1.0)  # JSCPD usually returns 1.0 for exact

                        if content_hash not in duplication_map:
                            duplication_map[content_hash] = {
                                'content_hash': content_hash,
                                'content': content,
                                'similarity': similarity,
                                'blocks': []
                            }

                        # Create duplicate block
                        block = DuplicateBlock(
                            file_path=file_data.get('name', ''),
                            start_line=dup.get('start', {}).get('line', 0),
                            end_line=dup.get('end', {}).get('line', 0),
                            block_id=f"{file_data.get('name', '')}_{dup.get('start', {}).get('line', 0)}",
                            content_hash=content_hash,
                            similarity_score=similarity,
                            duplication_type=DuplicationType.EXACT,
                            line_count=dup.get('size', {}).get('lines', 0)
                        )

                        duplication_map[content_hash]['blocks'].append(block)

                except Exception as e:
                    logger.warning(f"Failed to parse duplicate from JSCPD output: {e}")
                    continue

            # Convert map to DuplicationGroup objects
            for content_hash, dup_data in duplication_map.items():
                if len(dup_data['blocks']) >= 2:  # Need at least 2 blocks for duplication
                    group = DuplicationGroup(
                        group_id=f"jscpd_exact_{content_hash[:8]}",
                        duplication_type=DuplicationType.EXACT,
                        similarity_threshold=1.0,  # Exact match
                        blocks=dup_data['blocks']
                    )
                    duplicate_groups.append(group)

        except Exception as e:
            logger.error(f"Failed to parse JSCPD output: {e}")

        return duplicate_groups

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return self.formats

    def update_config(self, new_config: Dict[str, Any]):
        """Update detector configuration"""
        self.config.update(new_config)
        if 'min_lines' in new_config:
            self.min_lines = new_config['min_lines']
        if 'max_lines' in new_config:
            self.max_lines = new_config['max_lines']
        if 'min_tokens' in new_config:
            self.min_tokens = new_config['min_tokens']
        if 'ignore_patterns' in new_config:
            self.ignore_patterns = new_config['ignore_patterns']
        if 'formats' in new_config:
            self.formats = new_config['formats']
        if 'timeout' in new_config:
            self.timeout = new_config['timeout']