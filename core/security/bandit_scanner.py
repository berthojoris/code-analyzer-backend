"""
Bandit security scanner for Python code
Integrates with Bandit tool for Python-specific security analysis
"""

import os
import asyncio
import json
import subprocess
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile

from .scanner import SecurityIssue, Severity
from utils.logger import get_logger

logger = get_logger(__name__)


class BanditScanner:
    """Bandit Python security scanner integration"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Bandit scanner with configuration"""
        self.config = config or {}
        self.exclude_patterns = self.config.get('exclude_patterns', [
            '*/test_*',
            '*/tests/*',
            '*/__pycache__/*',
            '*/node_modules/*'
        ])
        self.confidence_level = self.config.get('confidence_level', 'low')
        self.severity_level = self.config.get('severity_level', 'low')
        self.output_format = self.config.get('output_format', 'json')

    async def scan_python_files(self, repo_path: str) -> List[SecurityIssue]:
        """
        Scan Python files using Bandit

        Args:
            repo_path: Path to repository to scan

        Returns:
            List of SecurityIssue objects
        """
        logger.info(f"Starting Bandit scan for Python files in {repo_path}")

        try:
            # Find Python files
            python_files = self._find_python_files(repo_path)
            if not python_files:
                logger.info("No Python files found for Bandit scanning")
                return []

            # Run Bandit scan
            bandit_output = await self._run_bandit_scan(repo_path)

            # Parse results
            issues = self._parse_bandit_output(bandit_output)

            logger.info(f"Bandit scan completed. Found {len(issues)} security issues")
            return issues

        except Exception as e:
            logger.error(f"Bandit scan failed: {e}")
            raise

    def _find_python_files(self, repo_path: str) -> List[str]:
        """Find Python files to scan"""
        python_files = []

        for root, dirs, files in os.walk(repo_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs
                      if not any(d.startswith(pattern.split('/')[-1].replace('*', ''))
                                for pattern in self.exclude_patterns
                                if '/' in pattern)]

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    if not self._is_excluded(file_path):
                        python_files.append(file_path)

        return python_files

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

    async def _run_bandit_scan(self, repo_path: str) -> Dict[str, Any]:
        """Execute Bandit scanner and get results"""
        cmd = [
            'bandit',
            '-r', repo_path,
            '-f', 'json',
            '--quiet'
        ]

        # Add exclusions
        if self.exclude_patterns:
            exclude_args = []
            for pattern in self.exclude_patterns:
                if '*/' in pattern:
                    exclude_args.append(pattern[2:])
                elif pattern.endswith('/*'):
                    exclude_args.append(pattern[:-2])
                else:
                    exclude_args.append(pattern)
            if exclude_args:
                cmd.extend(['--exclude', ','.join(exclude_args)])

        # Set confidence level
        if self.confidence_level:
            cmd.extend(['-i', self.confidence_level])

        # Set severity level
        if self.severity_level:
            cmd.extend(['-i', self.severity_level])

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=repo_path
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0 and process.returncode != 1:
                # Bandit returns 1 when issues are found, 0 for no issues
                error_msg = stderr.decode('utf-8') if stderr else "Unknown error"
                raise subprocess.CalledProcessError(process.returncode, cmd, error_msg)

            # Parse JSON output
            if stdout:
                return json.loads(stdout.decode('utf-8'))
            else:
                return {"results": [], "metrics": {}}

        except subprocess.CalledProcessError as e:
            logger.error(f"Bandit execution failed: {e}")
            # Return empty results on execution failure
            return {"results": [], "metrics": {}}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Bandit JSON output: {e}")
            return {"results": [], "metrics": {}}

    def _parse_bandit_output(self, bandit_output: Dict[str, Any]) -> List[SecurityIssue]:
        """Parse Bandit JSON output into SecurityIssue objects"""
        issues = []

        for result in bandit_output.get('results', []):
            try:
                # Map Bandit severity to our Severity enum
                bandit_severity = result.get('issue_severity', '').lower()
                if bandit_severity == 'high':
                    severity = Severity.HIGH
                elif bandit_severity == 'medium':
                    severity = Severity.MEDIUM
                elif bandit_severity == 'low':
                    severity = Severity.LOW
                else:
                    severity = Severity.MEDIUM  # Default

                # Map Bandit confidence
                bandit_confidence = result.get('issue_confidence', '').lower()
                if bandit_confidence in ['high', 'medium', 'low']:
                    confidence = bandit_confidence
                else:
                    confidence = 'medium'

                # Extract CWE ID if available
                cwe_id = None
                if 'cwe_id' in result:
                    cwe_id = f"CWE-{result['cwe_id']}"
                elif 'test_id' in result:
                    # Map Bandit test IDs to CWE where possible
                    cwe_mapping = {
                        'B101': 'CWE-89',  # assert_used
                        'B102': 'CWE-22',  # exec_used
                        'B103': 'CWE-78',  # set_bad_file_permissions
                        'B104': 'CWE-20',  # hardcoded_bind_all_interfaces
                        'B105': 'CWE-327',  # hardcoded_password_string
                        'B106': 'CWE-256',  # hardcoded_password_funcarg
                        'B107': 'CWE-327',  # hardcoded_password_default
                        'B108': 'CWE-20',  # hardcoded_tmp_directory
                        'B110': 'CWE-362',  # try_except_pass
                        'B112': 'CWE-20',  # try_except_continue
                        'B201': 'CWE-394',  # flask_debug_true
                        'B301': 'CWE-94',  # pickle
                        'B302': 'CWE-502',  # marshal
                        'B303': 'CWE-200',  # md5
                        'B304': 'CWE-327',  # ciphers
                        'B305': 'CWE-326',  # cipher_modes
                        'B306': 'CWE-327',  # mktemp_q
                        'B307': 'CWE-327',  # mktemp_r
                        'B308': 'CWE-327',  # mktemp_s
                        'B309': 'CWE-327',  # mktemp_u
                        'B310': 'CWE-327',  # mktemp_umask
                        'B311': 'CWE-327',  # mktemp_umask_r
                        'B312': 'CWE-326',  # telnetlib
                        'B313': 'CWE-327',  # xmlrpc_client
                        'B314': 'CWE-327',  # xmlrpc_server
                        'B315': 'CWE-94',  # xml_bad_cElementTree
                        'B316': 'CWE-326',  # xml_bad_expatbuilder
                        'B317': 'CWE-327',  # xml_bad_expatreader
                        'B318': 'CWE-327',  # xml_bad_sax
                        'B319': 'CWE-327',  # xml_bad_elementtree
                        'B320': 'CWE-327',  # xml_bad_minidom
                        'B321': 'CWE-327',  # xml_bad_pulldom
                        'B322': 'CWE-327',  # xml_bad_xmlrpc
                        'B323': 'CWE-326',  # xml_bad_etree
                        'B324': 'CWE-94',  # hashlib
                        'B325': 'CWE-327',  # tempnam
                        'B401': 'CWE-79',  # import_telnetlib
                        'B402': 'CWE-94',  # import_ftplib
                        'B403': 'CWE-502',  # import_pickle
                        'B404': 'CWE-502',  # import_subprocess
                        'B405': 'CWE-502',  # import_xml_pickle
                        'B406': 'CWE-94',  # import_xml_etree
                        'B407': 'CWE-502',  # import_xml_expatbuilder
                        'B408': 'CWE-326',  # import_xml_expatreader
                        'B409': 'CWE-327',  # import_xml_sax
                        'B410': 'CWE-327',  # import_xml_elementtree
                        'B411': 'CWE-327',  # import_xml_minidom
                        'B412': 'CWE-327',  # import_xml_pulldom
                        'B413': 'CWE-327',  # import_xmlrpc
                        'B501': 'CWE-327',  # request_with_no_cert_validation
                        'B502': 'CWE-310',  # ssl_with_bad_version
                        'B503': 'CWE-310',  # ssl_with_bad_defaults
                        'B504': 'CWE-310',  # ssl_with_no_version
                        'B505': 'CWE-327',  # weak_cryptographic_key
                        'B506': 'CWE-326',  # djangos bad secret
                        'B507': 'CWE-22',  # djangos sql_injection
                        'B601': 'CWE-89',  # paramiko_calls
                        'B602': 'CWE-78',  # subprocess_popen_with_shell_equals_true
                        'B603': 'CWE-78',  # subprocess_without_shell_equals_true
                        'B604': 'CWE-78',  # any_other_function_with_shell_equals_true
                        'B605': 'CWE-78',  # start_process_with_a_shell
                        'B606': 'CWE-89',  # start_process_with_no_shell
                        'B607': 'CWE-94',  # start_process_with_partial_path
                        'B608': 'CWE-89',  # hardcoded_sql_expressions
                        'B609': 'CWE-89',  # linux_commandsWildcardInjection
                        'B610': 'CWE-22',  # django_extra_used
                        'B611': 'CWE-89',  # django_raw_used
                        'B701': 'CWE-94',  # jinja2_autoescape_false
                        'B702': 'CWE-94',  # mako_templates
                        'B703': 'CWE-94',  # django_mark_safe
                    }
                    cwe_id = cwe_mapping.get(result.get('test_id', ''))

                # Map to OWASP Top 10 categories
                owasp_mapping = {
                    'hardcoded_password': 'A02:2021-Cryptographic Failures',
                    'injection': 'A03:2021-Injection',
                    'xss': 'A03:2021-Injection',
                    'csrf': 'A01:2021-Broken Access Control',
                    'ssrf': 'A10:2021-Server-Side Request Forgery',
                    'rce': 'A01:2021-Broken Access Control',
                    'path_traversal': 'A01:2021-Broken Access Control',
                    'weak_crypto': 'A02:2021-Cryptographic Failures',
                    'bad_random': 'A02:2021-Cryptographic Failures'
                }

                test_name = result.get('test_name', '').lower()
                owasp_category = None
                for owasp_key, owasp_value in owasp_mapping.items():
                    if owasp_key in test_name:
                        owasp_category = owasp_value
                        break

                issue = SecurityIssue(
                    tool_name="bandit",
                    file_path=result.get('filename', ''),
                    line_number=result.get('line_number', 0),
                    column_number=result.get('col_offset', None),
                    severity=severity,
                    confidence=confidence,
                    issue_type=result.get('test_name', 'unknown'),
                    issue_id=result.get('test_id', ''),
                    title=result.get('test_name', 'Security Issue'),
                    message=result.get('issue_text', ''),
                    cwe_id=cwe_id,
                    owasp_category=owasp_category,
                    references=result.get('more_info', []),
                    metadata={
                        'code_snippet': result.get('code', ''),
                        'test_name': result.get('test_name', ''),
                        'plugin_name': result.get('plugin_name', ''),
                        'plugin_version': result.get('plugin_version', ''),
                        'bandit_severity': result.get('issue_severity', ''),
                        'bandit_confidence': result.get('issue_confidence', '')
                    }
                )
                issues.append(issue)

            except Exception as e:
                logger.warning(f"Failed to parse Bandit result: {e}")
                continue

        return issues