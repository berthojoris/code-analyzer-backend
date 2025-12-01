"""
Similarity analyzer for token-based and structural code duplication detection
Supports multiple similarity metrics and AST-based analysis
"""

import os
import asyncio
import re
import tokenize
import ast
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import hashlib
from pathlib import Path
import difflib
from collections import Counter
import json

from .detector import DuplicationGroup, DuplicateBlock, DuplicationType
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TokenBlock:
    """Tokenized code block"""
    file_path: str
    start_line: int
    end_line: int
    tokens: List[str]
    content_hash: str
    line_count: int

    def __post_init__(self):
        self.content_hash = hashlib.md5(' '.join(self.tokens).encode()).hexdigest()


@dataclass
class StructuralPattern:
    """AST-based structural pattern"""
    pattern_id: str
    pattern_hash: str
    node_type: str
    structure_signature: str
    complexity: int
    parameters: Dict[str, Any]


class SimilarityAnalyzer:
    """Token-based and structural similarity analyzer"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize similarity analyzer with configuration"""
        self.config = config or {}
        self.min_similarity = config.get('min_similarity', 0.8)
        self.min_block_size = config.get('min_block_size', 10)  # Minimum lines
        self.max_block_size = config.get('max_block_size', 500)  # Maximum lines
        self.token_types = config.get('token_types', ['NAME', 'OP', 'STRING', 'NUMBER'])
        self.ignore_keywords = config.get('ignore_keywords', [
            'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'try', 'except',
            'finally', 'with', 'import', 'from', 'as', 'return', 'yield',
            'function', 'const', 'let', 'var', 'if', 'else', 'for', 'while'
        ])
        self.structural_similarity_threshold = config.get('structural_similarity_threshold', 0.75)

    async def find_similar_blocks(self, file_paths: List[str], threshold: float = None) -> List[DuplicationGroup]:
        """
        Find similar code blocks using token-based analysis

        Args:
            file_paths: List of file paths to analyze
            threshold: Minimum similarity threshold (uses config default if None)

        Returns:
            List of DuplicationGroup objects with similar blocks
        """
        if threshold is None:
            threshold = self.min_similarity

        logger.info(f"Starting token-based similarity analysis with threshold {threshold}")

        try:
            # Tokenize all files
            token_blocks = await self._tokenize_files(file_paths)
            if not token_blocks:
                logger.info("No tokenizable content found")
                return []

            # Find similar blocks
            similar_groups = await self._find_token_similarity(token_blocks, threshold)

            logger.info(f"Token-based similarity analysis completed. Found {len(similar_groups)} groups")
            return similar_groups

        except Exception as e:
            logger.error(f"Token-based similarity analysis failed: {e}")
            return []

    async def find_structural_duplicates(self, file_paths: List[str]) -> List[DuplicationGroup]:
        """
        Find structural duplicates using AST analysis

        Args:
            file_paths: List of file paths to analyze

        Returns:
            List of DuplicationGroup objects with structural duplicates
        """
        logger.info("Starting structural duplicate analysis")

        try:
            # Parse AST for all files
            structural_patterns = await self._extract_structural_patterns(file_paths)
            if not structural_patterns:
                logger.info("No structural patterns found")
                return []

            # Find similar structures
            structural_groups = await self._find_structural_similarity(structural_patterns)

            logger.info(f"Structural duplicate analysis completed. Found {len(structural_groups)} groups")
            return structural_groups

        except Exception as e:
            logger.error(f"Structural duplicate analysis failed: {e}")
            return []

    async def _tokenize_files(self, file_paths: List[str]) -> List[TokenBlock]:
        """Tokenize all files into blocks"""
        token_blocks = []

        for file_path in file_paths:
            try:
                blocks = await self._tokenize_file(file_path)
                token_blocks.extend(blocks)
            except Exception as e:
                logger.warning(f"Failed to tokenize file {file_path}: {e}")
                continue

        return token_blocks

    async def _tokenize_file(self, file_path: str) -> List[TokenBlock]:
        """Tokenize a single file into blocks"""
        language = self._detect_language(file_path)
        token_blocks = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            if language == 'python':
                blocks = await self._tokenize_python(content, file_path)
            elif language in ['javascript', 'typescript', 'jsx', 'tsx']:
                blocks = await self._tokenize_javascript(content, file_path)
            elif language in ['java', 'csharp', 'go', 'rust']:
                blocks = await self._tokenize_c_style(content, file_path)
            else:
                blocks = await self._tokenize_generic(content, file_path)

            token_blocks.extend(blocks)

        except Exception as e:
            logger.warning(f"Tokenization failed for {file_path}: {e}")

        return token_blocks

    async def _tokenize_python(self, content: str, file_path: str) -> List[TokenBlock]:
        """Tokenize Python code"""
        blocks = []
        lines = content.split('\n')

        try:
            # Use Python tokenize module
            tokens = list(tokenize.generate_tokens(StringIO(content).readline))

            # Group tokens into blocks by lines
            current_block_tokens = []
            current_block_start = 1

            for i, line in enumerate(lines, 1):
                line_tokens = []

                # Get tokens for this line
                for token in tokens:
                    if token.start[0] == i and token.type in [
                        tokenize.NAME, tokenize.OP, tokenize.STRING,
                        tokenize.NUMBER, tokenize.ERRORTOKEN
                    ]:
                        token_str = token.string
                        if token_str not in self.ignore_keywords:
                            line_tokens.append(token_str)

                if line_tokens:
                    current_block_tokens.extend(line_tokens)

                # Create blocks based on size thresholds
                if i - current_block_start >= self.max_block_size or \
                   (current_block_tokens and i - current_block_start >= self.min_block_size and \
                    self._is_block_boundary(lines[i-2] if i > 1 else '', line)):

                    if len(current_block_tokens) >= 5:  # Minimum tokens
                        block = TokenBlock(
                            file_path=file_path,
                            start_line=current_block_start,
                            end_line=i-1,
                            tokens=current_block_tokens,
                            content_hash='',
                            line_count=i - current_block_start
                        )
                        blocks.append(block)

                    current_block_tokens = []
                    current_block_start = i

            # Add remaining tokens as final block
            if current_block_tokens and len(current_block_tokens) >= 5:
                block = TokenBlock(
                    file_path=file_path,
                    start_line=current_block_start,
                    end_line=len(lines),
                    tokens=current_block_tokens,
                    content_hash='',
                    line_count=len(lines) - current_block_start + 1
                )
                blocks.append(block)

        except Exception as e:
            logger.warning(f"Python tokenization failed for {file_path}: {e}")
            # Fallback to generic tokenization
            return await self._tokenize_generic(content, file_path)

        return blocks

    async def _tokenize_javascript(self, content: str, file_path: str) -> List[TokenBlock]:
        """Tokenize JavaScript/TypeScript code"""
        # Simple regex-based tokenization for JavaScript
        tokens = []
        lines = content.split('\n')

        # JavaScript token patterns
        patterns = [
            r'\bfunction\b', r'\bconst\b', r'\blet\b', r'\bvar\b',
            r'\bif\b', r'\belse\b', r'\bfor\b', r'\bwhile\b',
            r'\breturn\b', r'\bclass\b', r'\bextends\b',
            r'[a-zA-Z_]\w*',  # Identifiers
            r'\d+(?:\.\d+)?',  # Numbers
            r'"[^"]*"', r"'[^']*'",  # Strings
            r'[+\-*/%=<>!&|^~?:]',  # Operators
        ]

        combined_pattern = '|'.join(f'({pattern})' for pattern in patterns)

        blocks = []
        current_block_tokens = []
        current_block_start = 1

        for i, line in enumerate(lines, 1):
            line_tokens = []

            for match in re.finditer(combined_pattern, line):
                for group_idx, token in enumerate(match.groups()):
                    if token and token not in self.ignore_keywords:
                        line_tokens.append(token)

            if line_tokens:
                current_block_tokens.extend(line_tokens)

            # Create blocks
            if i - current_block_start >= self.max_block_size or \
               (current_block_tokens and i - current_block_start >= self.min_block_size and \
                self._is_block_boundary(lines[i-2] if i > 1 else '', line)):

                if len(current_block_tokens) >= 5:
                    block = TokenBlock(
                        file_path=file_path,
                        start_line=current_block_start,
                        end_line=i-1,
                        tokens=current_block_tokens,
                        content_hash='',
                        line_count=i - current_block_start
                    )
                    blocks.append(block)

                current_block_tokens = []
                current_block_start = i

        # Add final block
        if current_block_tokens and len(current_block_tokens) >= 5:
            block = TokenBlock(
                file_path=file_path,
                start_line=current_block_start,
                end_line=len(lines),
                tokens=current_block_tokens,
                content_hash='',
                line_count=len(lines) - current_block_start + 1
            )
            blocks.append(block)

        return blocks

    async def _tokenize_c_style(self, content: str, file_path: str) -> List[TokenBlock]:
        """Tokenize C-style languages (Java, C#, Go, Rust)"""
        # Similar to JavaScript tokenization with language-specific adjustments
        return await self._tokenize_javascript(content, file_path)

    async def _tokenize_generic(self, content: str, file_path: str) -> List[TokenBlock]:
        """Generic tokenization fallback"""
        lines = content.split('\n')
        blocks = []
        current_block_start = 1

        for i in range(0, len(lines), self.min_block_size):
            block_end = min(i + self.min_block_size, len(lines))
            block_lines = lines[i:block_end]

            if not block_lines:
                continue

            # Simple word-based tokenization
            tokens = []
            for line in block_lines:
                words = re.findall(r'\b\w+\b', line)
                for word in words:
                    if word not in self.ignore_keywords and len(word) > 1:
                        tokens.append(word.lower())

            if len(tokens) >= 5:
                block = TokenBlock(
                    file_path=file_path,
                    start_line=i + 1,
                    end_line=block_end,
                    tokens=tokens,
                    content_hash='',
                    line_count=block_end - i
                )
                blocks.append(block)

        return blocks

    async def _extract_structural_patterns(self, file_paths: List[str]) -> List[StructuralPattern]:
        """Extract AST-based structural patterns from files"""
        patterns = []

        for file_path in file_paths:
            try:
                file_patterns = await self._extract_file_patterns(file_path)
                patterns.extend(file_patterns)
            except Exception as e:
                logger.warning(f"Failed to extract patterns from {file_path}: {e}")
                continue

        return patterns

    async def _extract_file_patterns(self, file_path: str) -> List[StructuralPattern]:
        """Extract structural patterns from a single file"""
        language = self._detect_language(file_path)
        patterns = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            if language == 'python':
                patterns = await self._extract_python_patterns(content, file_path)
            elif language in ['javascript', 'typescript']:
                patterns = await self._extract_javascript_patterns(content, file_path)
            else:
                patterns = await self._extract_generic_patterns(content, file_path)

        except Exception as e:
            logger.warning(f"Pattern extraction failed for {file_path}: {e}")

        return patterns

    async def _extract_python_patterns(self, content: str, file_path: str) -> List[StructuralPattern]:
        """Extract patterns from Python AST"""
        patterns = []

        try:
            tree = ast.parse(content)

            class PatternExtractor(ast.NodeVisitor):
                def visit_FunctionDef(self, node):
                    pattern_id = f"function_{node.lineno}_{node.name}"
                    structure_signature = self._get_function_signature(node)
                    complexity = self._calculate_ast_complexity(node)

                    pattern = StructuralPattern(
                        pattern_id=pattern_id,
                        pattern_hash=hashlib.md5(structure_signature.encode()).hexdigest(),
                        node_type='FunctionDef',
                        structure_signature=structure_signature,
                        complexity=complexity,
                        parameters={
                            'name': node.name,
                            'args_count': len(node.args.args),
                            'has_defaults': bool(node.args.defaults),
                            'has_varargs': bool(node.args.vararg),
                            'has_kwargs': bool(node.args.kwarg)
                        }
                    )
                    patterns.append(pattern)
                    self.generic_visit(node)

                def visit_ClassDef(self, node):
                    pattern_id = f"class_{node.lineno}_{node.name}"
                    structure_signature = self._get_class_signature(node)
                    complexity = self._calculate_ast_complexity(node)

                    pattern = StructuralPattern(
                        pattern_id=pattern_id,
                        pattern_hash=hashlib.md5(structure_signature.encode()).hexdigest(),
                        node_type='ClassDef',
                        structure_signature=structure_signature,
                        complexity=complexity,
                        parameters={
                            'name': node.name,
                            'bases_count': len(node.bases),
                            'methods_count': len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                        }
                    )
                    patterns.append(pattern)
                    self.generic_visit(node)

                def _get_function_signature(self, node):
                    # Extract structural signature (node types, control flow, etc.)
                    signature_parts = []
                    signature_parts.append('FunctionDef')

                    # Add argument structure
                    if node.args.args:
                        signature_parts.append(f'args:{len(node.args.args)}')

                    # Add control flow
                    for child in ast.walk(node):
                        if isinstance(child, ast.If):
                            signature_parts.append('if')
                        elif isinstance(child, ast.For):
                            signature_parts.append('for')
                        elif isinstance(child, ast.While):
                            signature_parts.append('while')
                        elif isinstance(child, ast.Try):
                            signature_parts.append('try')

                    return '_'.join(signature_parts)

                def _get_class_signature(self, node):
                    signature_parts = ['ClassDef']
                    signature_parts.append(f'bases:{len(node.bases)}')

                    methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                    signature_parts.append(f'methods:{len(methods)}')

                    return '_'.join(signature_parts)

                def _calculate_ast_complexity(self, node):
                    complexity = 1  # Base complexity
                    for child in ast.walk(node):
                        if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                            complexity += 1
                        elif isinstance(child, ast.BoolOp):
                            complexity += len(child.values) - 1
                    return complexity

            extractor = PatternExtractor()
            extractor.visit(tree)

        except Exception as e:
            logger.warning(f"Python AST parsing failed for {file_path}: {e}")

        return patterns

    async def _extract_javascript_patterns(self, content: str, file_path: str) -> List[StructuralPattern]:
        """Extract patterns from JavaScript/TypeScript (simplified)"""
        # This is a simplified implementation - in practice, you'd use a proper JS parser
        patterns = []

        # Use regex to find function patterns
        function_patterns = re.finditer(
            r'(?:function\s+(\w+)|(\w+)\s*=\s*(?:function|\([^)]*\)\s*=>))\s*\([^)]*\)\s*{',
            content, re.MULTILINE
        )

        for match in function_patterns:
            func_name = match.group(1) or match.group(2) or 'anonymous'
            line_num = content[:match.start()].count('\n') + 1

            pattern = StructuralPattern(
                pattern_id=f"js_function_{line_num}_{func_name}",
                pattern_hash=hashlib.md5(match.group().encode()).hexdigest(),
                node_type='Function',
                structure_signature='js_function',
                complexity=1,  # Simplified
                parameters={'name': func_name}
            )
            patterns.append(pattern)

        return patterns

    async def _extract_generic_patterns(self, content: str, file_path: str) -> List[StructuralPattern]:
        """Extract generic patterns from unknown languages"""
        patterns = []

        # Look for common structural elements
        lines = content.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()

            # Function-like patterns
            if re.match(r'(?:def|function|func)\s+\w+', line):
                pattern = StructuralPattern(
                    pattern_id=f"generic_function_{i+1}",
                    pattern_hash=hashlib.md5(line.encode()).hexdigest(),
                    node_type='Function',
                    structure_signature='generic_function',
                    complexity=1,
                    parameters={'line': i+1}
                )
                patterns.append(pattern)

            # Class-like patterns
            elif re.match(r'(?:class|interface)\s+\w+', line):
                pattern = StructuralPattern(
                    pattern_id=f"generic_class_{i+1}",
                    pattern_hash=hashlib.md5(line.encode()).hexdigest(),
                    node_type='Class',
                    structure_signature='generic_class',
                    complexity=1,
                    parameters={'line': i+1}
                )
                patterns.append(pattern)

        return patterns

    async def _find_token_similarity(self, token_blocks: List[TokenBlock], threshold: float) -> List[DuplicationGroup]:
        """Find similar token blocks using similarity metrics"""
        similar_groups = []

        # Group blocks by content hash for exact matches
        hash_groups = {}
        for block in token_blocks:
            if block.content_hash not in hash_groups:
                hash_groups[block.content_hash] = []
            hash_groups[block.content_hash].append(block)

        # Find similar blocks
        processed_hashes = set()

        for hash1, blocks1 in hash_groups.items():
            if hash1 in processed_hashes:
                continue

            similar_blocks = blocks1.copy()
            processed_hashes.add(hash1)

            # Find blocks with high similarity
            for hash2, blocks2 in hash_groups.items():
                if hash2 in processed_hashes:
                    continue

                # Check similarity between blocks
                for block1 in blocks1:
                    for block2 in blocks2:
                        similarity = await self._calculate_token_similarity(block1, block2)
                        if similarity >= threshold:
                            similar_blocks.append(block2)
                            processed_hashes.add(hash2)
                            break

            # Create duplicate group if we have enough similar blocks
            if len(similar_blocks) >= 2:
                duplicate_blocks = []
                for block in similar_blocks:
                    # Calculate similarity to the first block as reference
                    ref_block = similar_blocks[0]
                    similarity = await self._calculate_token_similarity(block, ref_block)

                    dup_block = DuplicateBlock(
                        file_path=block.file_path,
                        start_line=block.start_line,
                        end_line=block.end_line,
                        block_id=block.content_hash,
                        content_hash=block.content_hash,
                        similarity_score=similarity,
                        duplication_type=DuplicationType.LOGICAL if similarity < 1.0 else DuplicationType.EXACT,
                        line_count=block.line_count
                    )
                    duplicate_blocks.append(dup_block)

                group = DuplicationGroup(
                    group_id=f"similar_{hash1[:8]}",
                    duplication_type=DuplicationType.LOGICAL,
                    similarity_threshold=threshold,
                    blocks=duplicate_blocks
                )
                similar_groups.append(group)

        return similar_groups

    async def _find_structural_similarity(self, patterns: List[StructuralPattern]) -> List[DuplicationGroup]:
        """Find similar structural patterns"""
        similar_groups = []

        # Group patterns by structure hash
        structure_groups = {}
        for pattern in patterns:
            if pattern.structure_signature not in structure_groups:
                structure_groups[pattern.structure_signature] = []
            structure_groups[pattern.structure_signature].append(pattern)

        # Find patterns with similar structure
        processed_signatures = set()

        for sig1, patterns1 in structure_groups.items():
            if sig1 in processed_signatures:
                continue

            similar_patterns = patterns1.copy()
            processed_signatures.add(sig1)

            # Find patterns with similar structure
            for sig2, patterns2 in structure_groups.items():
                if sig2 in processed_signatures:
                    continue

                # Check structural similarity
                for pat1 in patterns1:
                    for pat2 in patterns2:
                        similarity = await self._calculate_structural_similarity(pat1, pat2)
                        if similarity >= self.structural_similarity_threshold:
                            similar_patterns.append(pat2)
                            processed_signatures.add(sig2)
                            break

            # Create duplicate group
            if len(similar_patterns) >= 2:
                duplicate_blocks = []
                for pattern in similar_patterns:
                    # Extract file info from pattern_id
                    parts = pattern.pattern_id.split('_')
                    if len(parts) >= 2:
                        line_num = int(parts[-2])
                    else:
                        line_num = 1

                    dup_block = DuplicateBlock(
                        file_path="",  # Pattern doesn't track full path in this simplified version
                        start_line=line_num,
                        end_line=line_num,
                        block_id=pattern.pattern_id,
                        content_hash=pattern.pattern_hash,
                        similarity_score=1.0,  # Structural matches are considered similar
                        duplication_type=DuplicationType.STRUCTURAL,
                        line_count=1
                    )
                    duplicate_blocks.append(dup_block)

                group = DuplicationGroup(
                    group_id=f"structural_{pattern.structure_signature[:8]}",
                    duplication_type=DuplicationType.STRUCTURAL,
                    similarity_threshold=self.structural_similarity_threshold,
                    blocks=duplicate_blocks
                )
                similar_groups.append(group)

        return similar_groups

    async def _calculate_token_similarity(self, block1: TokenBlock, block2: TokenBlock) -> float:
        """Calculate similarity between two token blocks"""
        # Use Jaccard similarity for token sets
        tokens1 = set(block1.tokens)
        tokens2 = set(block2.tokens)

        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))

        return intersection / union if union > 0 else 0.0

    async def _calculate_structural_similarity(self, pattern1: StructuralPattern, pattern2: StructuralPattern) -> float:
        """Calculate similarity between two structural patterns"""
        if pattern1.node_type != pattern2.node_type:
            return 0.0

        # Compare structure signatures
        if pattern1.structure_signature == pattern2.structure_signature:
            return 1.0

        # Compare parameters
        param_similarity = 0.0
        common_params = set(pattern1.parameters.keys()).intersection(pattern2.parameters.keys())

        if common_params:
            matches = 0
            for param in common_params:
                if pattern1.parameters[param] == pattern2.parameters[param]:
                    matches += 1
            param_similarity = matches / len(common_params)

        # Consider complexity difference
        complexity_diff = abs(pattern1.complexity - pattern2.complexity)
        max_complexity = max(pattern1.complexity, pattern2.complexity, 1)
        complexity_similarity = 1.0 - (complexity_diff / max_complexity)

        # Combine similarities
        return (param_similarity * 0.7 + complexity_similarity * 0.3)

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        extension = Path(file_path).suffix.lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.cpp': 'c',
            '.cxx': 'c',
            '.cc': 'c',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'c'
        }
        return language_map.get(extension, 'unknown')

    def _is_block_boundary(self, prev_line: str, current_line: str) -> bool:
        """Check if current line marks a natural code block boundary"""
        current_stripped = current_line.strip()

        # Check for common block boundaries
        if current_stripped.startswith(('def ', 'function ', 'class ', 'if ', 'for ', 'while ', 'try:')):
            return True

        if current_stripped in ['}', ']', ')', 'else:', 'elif ', 'except ', 'finally:']:
            return True

        if prev_line.strip().endswith(':'):
            return True

        return False