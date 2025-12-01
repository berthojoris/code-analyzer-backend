"""
Semgrep scanner for multi-language static analysis security
Integrates with Semgrep tool for comprehensive security analysis across languages
"""

import os
import asyncio
import json
import subprocess
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import tempfile

from .models import SecurityIssue, Severity
from utils.logger import get_logger

logger = get_logger(__name__)


class SemgrepScanner:
    """Semgrep multi-language security scanner"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Semgrep scanner with configuration"""
        self.config = config or {}
        self.exclude_patterns = self.config.get('exclude_patterns', [
            '*/test_*',
            '*/tests/*',
            '*/__pycache__/*',
            '*/node_modules/*',
            '*/target/*',
            '*/build/*',
            '*/dist/*'
        ])
        self.rulesets = self.config.get('rulesets', [
            'p/security-audit',
            'p/secrets',
            'p/owasp-top-ten',
            'p/cwe-top-25',
            'p/semgrep-python',
            'p/semgrep-javascript',
            'p/semgrep-java',
            'p/semgrep-go'
        ])
        self.languages = self.config.get('languages', [
            'python', 'javascript', 'java', 'go', 'ruby', 'php', 'c', 'cpp'
        ])
        self.max_target_bytes = self.config.get('max_target_bytes', 1000000)  # 1MB
        self.timeout = self.config.get('timeout', 300)  # 5 minutes

    async def scan_repository(self, repo_path: str) -> List[SecurityIssue]:
        """
        Scan repository using Semgrep for multi-language security analysis

        Args:
            repo_path: Path to repository to scan

        Returns:
            List of SecurityIssue objects
        """
        logger.info(f"Starting Semgrep multi-language scan in {repo_path}")

        try:
            # Check if Semgrep is available
            if not await self._is_semgrep_available():
                logger.warning("Semgrep not available, skipping multi-language security scan")
                return []

            # Detect languages in repository
            detected_languages = self._detect_languages(repo_path)
            if not detected_languages:
                logger.info("No supported source files found for Semgrep scanning")
                return []

            # Run Semgrep scan
            semgrep_output = await self._run_semgrep_scan(repo_path, detected_languages)

            # Parse results
            issues = self._parse_semgrep_output(semgrep_output, repo_path)

            logger.info(f"Semgrep scan completed. Found {len(issues)} security issues")
            return issues

        except Exception as e:
            logger.error(f"Semgrep scan failed: {e}")
            # Don't raise exception for scanning failures
            return []

    async def _is_semgrep_available(self) -> bool:
        """Check if Semgrep CLI is available"""
        try:
            process = await asyncio.create_subprocess_exec(
                'semgrep', '--version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            return process.returncode == 0
        except Exception:
            return False

    def _detect_languages(self, repo_path: str) -> Set[str]:
        """Detect programming languages in the repository"""
        detected = set()
        language_extensions = {
            'python': ['.py'],
            'javascript': ['.js', '.jsx', '.mjs', '.ts', '.tsx'],
            'java': ['.java'],
            'go': ['.go'],
            'ruby': ['.rb'],
            'php': ['.php'],
            'c': ['.c', '.h'],
            'cpp': ['.cpp', '.cxx', '.cc', '.hpp', '.hxx'],
            'typescript': ['.ts', '.tsx'],
            'kotlin': ['.kt', '.kts'],
            'scala': ['.scala', '.sc'],
            'csharp': ['.cs'],
            'dart': ['.dart'],
            'lua': ['.lua'],
            'perl': ['.pl', '.pm'],
            'r': ['.r', '.R'],
            'rust': ['.rs'],
            'swift': ['.swift']
        }

        for root, dirs, files in os.walk(repo_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs
                      if not any(d.startswith(pattern.split('/')[-1].replace('*', ''))
                                for pattern in self.exclude_patterns
                                if '/' in pattern)]

            for file in files:
                file_path = os.path.join(root, file)
                if not self._is_excluded(file_path):
                    file_ext = Path(file).suffix.lower()
                    for lang, extensions in language_extensions.items():
                        if file_ext in extensions:
                            detected.add(lang)
                            break

        # Filter to only supported languages
        return detected.intersection(self.languages)

    async def _run_semgrep_scan(self, repo_path: str, languages: Set[str]) -> Dict[str, Any]:
        """Execute Semgrep scan with appropriate configuration"""
        cmd = [
            'semgrep',
            'scan',
            '--config=auto',  # Auto-configure based on languages
            '--json',
            '--quiet',
            '--no-rewrite-rule-ids',
            '--metrics=off',
            f'--max-target-bytes={self.max_target_bytes}'
        ]

        # Add custom rulesets
        if self.rulesets:
            for ruleset in self.rulesets:
                cmd.extend(['--config', ruleset])

        # Add exclude patterns
        exclude_list = []
        for pattern in self.exclude_patterns:
            if '*/' in pattern:
                exclude_list.append(pattern[2:])
            elif pattern.endswith('/*'):
                exclude_list.append(pattern[:-2])
            else:
                exclude_list.append(pattern)

        if exclude_list:
            cmd.extend(['--exclude', ','.join(exclude_list)])

        # Add target directory
        cmd.append(repo_path)

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=repo_path
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout
            )

            if process.returncode != 0 and process.returncode != 1:
                # Semgrep returns 1 when findings are present
                logger.warning(f"Semgrep execution failed: {stderr.decode()}")
                return {"results": []}

            if stdout:
                return json.loads(stdout.decode('utf-8'))
            else:
                return {"results": []}

        except asyncio.TimeoutError:
            logger.error(f"Semgrep scan timed out after {self.timeout} seconds")
            return {"results": []}
        except subprocess.CalledProcessError as e:
            logger.error(f"Semgrep execution failed: {e}")
            return {"results": []}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Semgrep JSON output: {e}")
            return {"results": []}

    def _parse_semgrep_output(self, semgrep_output: Dict[str, Any], repo_path: str) -> List[SecurityIssue]:
        """Parse Semgrep JSON output into SecurityIssue objects"""
        issues = []

        for result in semgrep_output.get('results', []):
            try:
                # Extract metadata
                metadata = result.get('metadata', {})
                extra = result.get('extra', {})

                # Map Semgrep severity to our Severity enum
                semgrep_severity = metadata.get('severity', '').lower()
                if semgrep_severity == 'error':
                    severity = Severity.CRITICAL
                elif semgrep_severity == 'warning':
                    severity = Severity.HIGH
                elif semgrep_severity == 'info':
                    severity = Severity.MEDIUM
                else:
                    severity = Severity.LOW

                # Map Semgrep confidence
                confidence = metadata.get('confidence', '').lower()
                if confidence not in ['high', 'medium', 'low']:
                    confidence = 'medium'

                # Extract file and location information
                file_path = result.get('path', '')
                start_line = result.get('start', {}).get('line', 0)
                start_col = result.get('start', {}).get('col', 0)

                # Extract CWE and OWASP information
                cwe_id = None
                owasp_category = None

                if 'cwe' in metadata:
                    cwe_numbers = metadata['cwe']
                    if isinstance(cwe_numbers, list) and cwe_numbers:
                        cwe_id = f"CWE-{cwe_numbers[0]}"
                    elif isinstance(cwe_numbers, (int, str)):
                        cwe_id = f"CWE-{cwe_numbers}"

                if 'owasp' in metadata:
                    owasp_list = metadata['owasp']
                    if isinstance(owasp_list, list) and owasp_list:
                        owasp_category = f"A{owasp_list[0]:02d}:2021-{self._get_owasp_name(owasp_list[0])}"

                # Build message
                message = metadata.get('message', result.get('message', ''))
                if extra.get('fix', {}).get('regex'):
                    fix_regex = extra['fix']['regex']
                    if fix_regex.get('replacement'):
                        message += f"\nSuggested fix: {fix_regex['replacement']}"

                # Collect references
                references = []
                if 'references' in metadata:
                    references.extend(metadata['references'])
                if 'source' in metadata:
                    source_url = metadata['source']
                    if isinstance(source_url, str):
                        references.append(source_url)

                issue = SecurityIssue(
                    tool_name="semgrep",
                    file_path=file_path,
                    line_number=start_line,
                    column_number=start_col,
                    severity=severity,
                    confidence=confidence,
                    issue_type=metadata.get('category', 'security'),
                    issue_id=result.get('check_id', ''),
                    title=metadata.get('name', 'Security Issue'),
                    message=message,
                    cwe_id=cwe_id,
                    owasp_category=owasp_category,
                    references=references,
                    metadata={
                        'rule_id': result.get('check_id', ''),
                        'rule_name': metadata.get('name', ''),
                        'category': metadata.get('category', ''),
                        'technology': metadata.get('technology', []),
                        'language': result.get('extra', {}).get('language', ''),
                        'semgrep_severity': metadata.get('severity', ''),
                        'semgrep_confidence': metadata.get('confidence', ''),
                        'end_line': result.get('end', {}).get('line', start_line),
                        'end_col': result.get('end', {}).get('col', start_col),
                        'code_snippet': extra.get('lines', ''),
                        'metavars': extra.get('metavars', {}),
                        'fix': extra.get('fix', {}),
                        'source_file_url': f"https://semgrep.dev/r/{result.get('check_id', '')}"
                    }
                )
                issues.append(issue)

            except Exception as e:
                logger.warning(f"Failed to parse Semgrep result: {e}")
                continue

        return issues

    def _get_owasp_name(self, owasp_number: int) -> str:
        """Get OWASP Top 10 category name from number"""
        owasp_mapping = {
            1: "Broken Access Control",
            2: "Cryptographic Failures",
            3: "Injection",
            4: "Insecure Design",
            5: "Security Misconfiguration",
            6: "Vulnerable and Outdated Components",
            7: "Identification and Authentication Failures",
            8: "Software and Data Integrity Failures",
            9: "Security Logging and Monitoring Failures",
            10: "Server-Side Request Forgery"
        }
        return owasp_mapping.get(owasp_number, "Unknown")

    def _is_excluded(self, file_path: str) -> bool:
        """Check if file should be excluded from scanning"""
        file_path = file_path.replace('\\', '/')
        for pattern in self.exclude_patterns:
            if pattern.startswith('*/'):
                if pattern[2:] in file_path:
                    return True
            elif pattern.endswith('/*'):
                if file_path.startswith(pattern[:-2]):
                    return True
            elif pattern in file_path:
                return True
        return False