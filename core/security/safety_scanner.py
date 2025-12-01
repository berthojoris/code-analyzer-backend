"""
Safety scanner for Python dependency vulnerabilities
Integrates with Safety tool to check known vulnerabilities in dependencies
"""

import os
import asyncio
import json
import subprocess
from typing import List, Dict, Any, Optional
from pathlib import Path
import re

from .scanner import SecurityIssue, Severity
from utils.logger import get_logger

logger = get_logger(__name__)


class SafetyScanner:
    """Safety dependency vulnerability scanner"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Safety scanner with configuration"""
        self.config = config or {}
        self.ignore_vulnerabilities = self.config.get('ignore_vulnerabilities', [])
        self.output_format = self.config.get('output_format', 'json')
        self.check_full_report = self.config.get('check_full_report', True)
        self.db_path = self.config.get('db_path', None)  # Custom vulnerability database

    async def scan_dependencies(self, repo_path: str) -> List[SecurityIssue]:
        """
        Scan Python dependencies for known vulnerabilities using Safety

        Args:
            repo_path: Path to repository to scan

        Returns:
            List of SecurityIssue objects
        """
        logger.info(f"Starting Safety dependency scan in {repo_path}")

        try:
            # Find dependency files
            dependency_files = self._find_dependency_files(repo_path)
            if not dependency_files:
                logger.info("No Python dependency files found for Safety scanning")
                return []

            # Run Safety scan for each dependency file
            all_issues = []
            for dep_file in dependency_files:
                issues = await self._scan_dependency_file(repo_path, dep_file)
                all_issues.extend(issues)

            logger.info(f"Safety scan completed. Found {len(all_issues)} vulnerable dependencies")
            return all_issues

        except Exception as e:
            logger.error(f"Safety scan failed: {e}")
            # Don't raise exception for dependency scanning failures
            # Just return empty list
            return []

    def _find_dependency_files(self, repo_path: str) -> List[str]:
        """Find Python dependency files to scan"""
        dependency_files = []

        # Common Python dependency files
        dep_file_patterns = [
            'requirements.txt',
            'requirements/*.txt',
            'requirements-dev.txt',
            'requirements-test.txt',
            'Pipfile',
            'Pipfile.lock',
            'pyproject.toml',
            'setup.py',
            'setup.cfg'
        ]

        for pattern in dep_file_patterns:
            for file_path in Path(repo_path).glob(pattern):
                if file_path.is_file():
                    dependency_files.append(str(file_path))

        return dependency_files

    async def _scan_dependency_file(self, repo_path: str, dep_file: str) -> List[SecurityIssue]:
        """Scan a specific dependency file for vulnerabilities"""
        file_name = Path(dep_file).name
        logger.debug(f"Scanning dependency file: {file_name}")

        try:
            if file_name == 'Pipfile.lock':
                return await self._scan_pipfile_lock(dep_file)
            elif file_name == 'pyproject.toml':
                return await self._scan_pyproject_toml(dep_file)
            elif file_name.endswith('.txt'):
                return await self._scan_requirements_file(dep_file)
            else:
                # For other files, try to extract dependencies and scan with Safety
                return await self._scan_with_safety(dep_file)

        except Exception as e:
            logger.warning(f"Failed to scan {file_name}: {e}")
            return []

    async def _scan_pipfile_lock(self, pipfile_lock_path: str) -> List[SecurityIssue]:
        """Scan Pipfile.lock for vulnerabilities"""
        try:
            with open(pipfile_lock_path, 'r', encoding='utf-8') as f:
                pipfile_data = json.load(f)

            # Extract default package dependencies
            dependencies = []
            default_packages = pipfile_data.get('default', {})
            for package_name, package_info in default_packages.items():
                version = package_info.get('version', '').replace('==', '')
                if version:
                    dependencies.append(f"{package_name}=={version}")

            return await self._scan_dependency_list(dependencies, pipfile_lock_path)

        except Exception as e:
            logger.error(f"Failed to parse Pipfile.lock: {e}")
            return []

    async def _scan_pyproject_toml(self, pyproject_path: str) -> List[SecurityIssue]:
        """Scan pyproject.toml for vulnerabilities"""
        try:
            import toml

            with open(pyproject_path, 'r', encoding='utf-8') as f:
                pyproject_data = toml.load(f)

            dependencies = []

            # Extract dependencies from different sections
            # Poetry format
            if 'tool' in pyproject_data and 'poetry' in pyproject_data['tool']:
                poetry_deps = pyproject_data['tool']['poetry'].get('dependencies', {})
                for package_name, version_spec in poetry_deps.items():
                    if isinstance(version_spec, str):
                        # Convert version spec to simple version for scanning
                        version = re.sub(r'[^0-9.]', '', version_spec.split(',')[0])
                        if version:
                            dependencies.append(f"{package_name}=={version}")

            # PEP 621 format
            elif 'project' in pyproject_data:
                project_deps = pyproject_data['project'].get('dependencies', [])
                for dep_spec in project_deps:
                    # Extract package name and version
                    match = re.match(r'([a-zA-Z0-9\-_.]+)[>=<!=]+([0-9.]+)', dep_spec)
                    if match:
                        package_name, version = match.groups()
                        dependencies.append(f"{package_name}=={version}")

            return await self._scan_dependency_list(dependencies, pyproject_path)

        except ImportError:
            logger.warning("toml library not available for parsing pyproject.toml")
            return []
        except Exception as e:
            logger.error(f"Failed to parse pyproject.toml: {e}")
            return []

    async def _scan_requirements_file(self, requirements_path: str) -> List[SecurityIssue]:
        """Scan requirements.txt file for vulnerabilities"""
        try:
            dependencies = []
            with open(requirements_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith('#') or line.startswith('-'):
                        continue

                    # Parse package specification
                    # Examples: requests==2.25.1, django>=3.2,<4.0
                    match = re.match(r'([a-zA-Z0-9\-_.]+)(?:[>=<!=]+([0-9.]+))?', line)
                    if match:
                        package_name, version = match.groups()
                        if version:
                            dependencies.append(f"{package_name}=={version}")
                        else:
                            # If no version specified, we can't scan effectively
                            logger.debug(f"Skipping {package_name} (no version specified)")

            return await self._scan_dependency_list(dependencies, requirements_path)

        except Exception as e:
            logger.error(f"Failed to parse requirements file: {e}")
            return []

    async def _scan_with_safety(self, file_path: str) -> List[SecurityIssue]:
        """Use Safety CLI to scan dependency file"""
        try:
            cmd = ['safety', 'check', '--json', '--short-report']

            if self.db_path:
                cmd.extend(['--db', self.db_path])

            cmd.extend(['-r', file_path])

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0 and process.returncode != 1:
                # Safety returns 1 when vulnerabilities are found
                logger.warning(f"Safety CLI failed with code {process.returncode}: {stderr.decode()}")
                return []

            if stdout:
                safety_output = json.loads(stdout.decode('utf-8'))
                return self._parse_safety_output(safety_output, file_path)
            else:
                return []

        except subprocess.CalledProcessError as e:
            logger.warning(f"Safety CLI execution failed: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Safety JSON output: {e}")
            return []
        except Exception as e:
            logger.warning(f"Error running Safety: {e}")
            return []

    async def _scan_dependency_list(self, dependencies: List[str], source_file: str) -> List[SecurityIssue]:
        """Scan a list of dependencies for vulnerabilities"""
        if not dependencies:
            return []

        try:
            # Create temporary requirements file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                for dep in dependencies:
                    temp_file.write(f"{dep}\n")
                temp_file_path = temp_file.name

            try:
                # Use Safety CLI to scan the temporary file
                cmd = ['safety', 'check', '--json', '--short-report']

                if self.db_path:
                    cmd.extend(['--db', self.db_path])

                cmd.extend(['-r', temp_file_path])

                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                stdout, stderr = await process.communicate()

                if process.returncode != 0 and process.returncode != 1:
                    logger.warning(f"Safety failed for {source_file}: {stderr.decode()}")
                    return []

                if stdout:
                    safety_output = json.loads(stdout.decode('utf-8'))
                    issues = self._parse_safety_output(safety_output, source_file)

                    # Update file_path for issues to point to original source file
                    for issue in issues:
                        issue.file_path = source_file

                    return issues
                else:
                    return []

            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)

        except Exception as e:
            logger.warning(f"Failed to scan dependencies from {source_file}: {e}")
            return []

    def _parse_safety_output(self, safety_output: Dict[str, Any], source_file: str) -> List[SecurityIssue]:
        """Parse Safety JSON output into SecurityIssue objects"""
        issues = []

        for vuln in safety_output.get('vulnerabilities', []):
            try:
                # Determine severity based on CVSS score or Safety analysis
                safety_severity = vuln.get('v', '').lower()
                if 'critical' in safety_severity or vuln.get('v') == 'CRITICAL':
                    severity = Severity.CRITICAL
                elif 'high' in safety_severity or vuln.get('v') == 'HIGH':
                    severity = Severity.HIGH
                elif 'medium' in safety_severity or vuln.get('v') == 'MEDIUM':
                    severity = Severity.MEDIUM
                else:
                    severity = Severity.LOW

                # Map to OWASP category
                owasp_category = "A06:2021-Vulnerable and Outdated Components"

                # Extract CVE if available
                cve_id = vuln.get('cve')
                if cve_id and not cve_id.startswith('CVE-'):
                    cve_id = f"CVE-{cve_id}"

                # Construct advisory and references
                advisory = vuln.get('advisory', '')
                references = []

                # Add common reference URLs
                package_name = vuln.get('package_name', '')
                vuln_id = vuln.get('id', '')
                if vuln_id:
                    references.append(f"https://pyup.io/packages/{package_name}/vuln/{vuln_id}/")
                if cve_id:
                    references.append(f"https://cve.mitre.org/cgi-bin/cvename.cgi?name={cve_id}")

                issue = SecurityIssue(
                    tool_name="safety",
                    file_path=source_file,
                    line_number=1,  # Dependencies are usually listed at file level
                    column_number=None,
                    severity=severity,
                    confidence='high',  # Safety has high confidence in known vulnerabilities
                    issue_type="dependency_vulnerability",
                    issue_id=vuln.get('id', ''),
                    title=f"Vulnerable dependency: {package_name}",
                    message=advisory or f"Known security vulnerability in {package_name} {vuln.get('analyzed_version', '')}",
                    cwe_id=cve_id,
                    owasp_category=owasp_category,
                    references=references,
                    metadata={
                        'package_name': package_name,
                        'vulnerable_version': vuln.get('vulnerable_spec', ''),
                        'analyzed_version': vuln.get('analyzed_version', ''),
                        'vulnerability_id': vuln.get('id', ''),
                        'cvss_score': vuln.get('cvss_score'),
                        'cvss_vector': vuln.get('cvss_vector'),
                        'safety_severity': vuln.get('v', ''),
                        'ignore_reason': vuln.get('ignore_reason'),
                        'source_file': source_file
                    }
                )

                # Skip if vulnerability is in ignore list
                if issue.issue_id not in self.ignore_vulnerabilities:
                    issues.append(issue)

            except Exception as e:
                logger.warning(f"Failed to parse Safety vulnerability: {e}")
                continue

        return issues