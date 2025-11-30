import os
import shutil
import tempfile
from typing import Optional
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from git import Repo

from core.analyzer.language_detector import (
    detect_language,
    get_code_files,
    get_language_stats,
)
from core.analyzer.python_parser import PythonParser, GenericParser
from core.analyzer.base_parser import CodeChunk
from core.embeddings.embeddings import get_embedding_service
from core.vector_store.pinecone_store import get_pinecone_store
from core.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


def build_embedding_text(chunk: CodeChunk) -> str:
    """Build semantically enriched text for embedding.
    
    Combines metadata with code content to improve vector search relevance.
    """
    parts = []
    
    # Add language context
    parts.append(f"Language: {chunk.language}")
    
    # Add file path for context
    parts.append(f"File: {chunk.file_path}")
    
    # Add chunk type and name if available
    if chunk.name:
        parts.append(f"{chunk.chunk_type.capitalize()} name: {chunk.name}")
    else:
        parts.append(f"Type: {chunk.chunk_type}")
    
    # Add the actual code
    parts.append(f"Code:\n{chunk.content}")
    
    return "\n".join(parts)


class IndexRequest(BaseModel):
    repo_url: str = Field(..., description="GitHub repository URL to index")
    reindex: bool = Field(default=False, description="Force reindex if already exists")


class IndexResponse(BaseModel):
    status: str
    repo_name: str
    files_processed: int
    chunks_indexed: int
    dominant_language: str
    language_stats: dict
    message: str


def extract_repo_name(repo_url: str) -> str:
    """Extract repository name from GitHub URL."""
    parsed = urlparse(repo_url)
    path = parsed.path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    parts = path.split("/")
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return parts[-1] if parts else "unknown"


def clone_repository(repo_url: str, target_dir: str) -> str:
    """Clone a GitHub repository to the target directory."""
    logger.info(f"Cloning repository: {repo_url}")
    Repo.clone_from(repo_url, target_dir, depth=1)
    return target_dir


def get_parser_for_language(language: str):
    """Get appropriate parser for the language."""
    if language == "python":
        return PythonParser()
    else:
        # Map language to extensions
        extension_map = {
            "javascript": [".js", ".jsx"],
            "typescript": [".ts", ".tsx"],
            "java": [".java"],
            "go": [".go"],
            "rust": [".rs"],
            "ruby": [".rb"],
        }
        extensions = extension_map.get(language, [f".{language}"])
        return GenericParser(language, extensions)


@router.post("/index", response_model=IndexResponse)
async def index_repository(request: IndexRequest):
    """Index a GitHub repository for semantic search."""
    repo_url = request.repo_url.strip()
    
    # Validate URL
    if not repo_url.startswith(("https://github.com/", "http://github.com/")):
        raise HTTPException(
            status_code=400,
            detail="Only public GitHub repositories are supported",
        )

    repo_name = extract_repo_name(repo_url)
    
    # Check if repo already indexed (skip if reindex is requested)
    if not request.reindex:
        store = get_pinecone_store()
        if store.repo_exists(repo_name):
            logger.info(f"Repository {repo_name} already indexed, skipping")
            return IndexResponse(
                status="already_indexed",
                repo_name=repo_name,
                files_processed=0,
                chunks_indexed=0,
                dominant_language="",
                language_stats={},
                message=f"Repository '{repo_name}' is already indexed. Use reindex=true to re-index.",
            )
    
    temp_dir = None

    try:
        # Create temporary directory for cloning
        temp_dir = tempfile.mkdtemp(prefix="repo_", dir=settings.temp_repo_dir)
        os.makedirs(settings.temp_repo_dir, exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix="repo_", dir=settings.temp_repo_dir)

        # Clone the repository
        repo_path = clone_repository(repo_url, temp_dir)

        # Detect dominant language
        dominant_language = detect_language(repo_path)
        language_stats = get_language_stats(repo_path)
        logger.info(f"Detected language: {dominant_language}, stats: {language_stats}")

        # Get code files
        code_files = get_code_files(repo_path)
        logger.info(f"Found {len(code_files)} code files")

        # Parse files into chunks
        all_chunks: list[CodeChunk] = []
        parser = get_parser_for_language(dominant_language)
        generic_parser = GenericParser("generic", [])

        for file_path in code_files:
            try:
                ext = file_path.suffix.lower()
                if ext in parser.get_supported_extensions():
                    chunks = parser.parse_file(file_path)
                else:
                    # Use generic parser for other files
                    generic_parser.language = "unknown"
                    chunks = generic_parser.parse_file(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"Failed to parse {file_path}: {e}")

        logger.info(f"Generated {len(all_chunks)} code chunks")

        if not all_chunks:
            return IndexResponse(
                status="success",
                repo_name=repo_name,
                files_processed=len(code_files),
                chunks_indexed=0,
                dominant_language=dominant_language,
                language_stats=language_stats,
                message="No code chunks to index",
            )

        # Generate embeddings with enriched semantic context
        embedding_service = get_embedding_service()
        chunk_texts = [build_embedding_text(chunk) for chunk in all_chunks]
        
        # Process in batches to avoid rate limits
        batch_size = 50
        all_embeddings = []
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i : i + batch_size]
            embeddings = embedding_service.create_embeddings_batch(batch)
            all_embeddings.extend(embeddings)
            logger.info(f"Generated embeddings for {len(all_embeddings)}/{len(chunk_texts)} chunks")

        # Store in Pinecone
        store = get_pinecone_store()
        
        # Delete existing vectors if reindexing
        if request.reindex:
            store.delete_repo(repo_name)

        chunks_indexed = store.upsert_chunks(all_chunks, all_embeddings, repo_name)

        return IndexResponse(
            status="success",
            repo_name=repo_name,
            files_processed=len(code_files),
            chunks_indexed=chunks_indexed,
            dominant_language=dominant_language,
            language_stats=language_stats,
            message=f"Successfully indexed {repo_name}",
        )

    except Exception as e:
        logger.error(f"Failed to index repository: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to index repository: {str(e)}",
        )
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
