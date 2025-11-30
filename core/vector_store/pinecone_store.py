from typing import List, Dict, Any, Optional
import hashlib

from pinecone import Pinecone, ServerlessSpec

from core.config import settings
from core.analyzer.base_parser import CodeChunk
from core.embeddings.embeddings import EMBEDDING_DIMENSIONS
from utils.logger import get_logger

logger = get_logger(__name__)


class PineconeStore:
    """Vector store using Pinecone for code chunk storage and retrieval."""

    def __init__(self):
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index_name = settings.pinecone_index_name
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)

    def _ensure_index_exists(self):
        """Create the index if it doesn't exist."""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=EMBEDDING_DIMENSIONS,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=settings.pinecone_environment,
                ),
            )

    def upsert_chunks(
        self,
        chunks: List[CodeChunk],
        embeddings: List[List[float]],
        repo_name: str,
    ) -> int:
        """Store code chunks with their embeddings."""
        if not chunks or not embeddings:
            return 0

        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = self._generate_chunk_id(repo_name, chunk.file_path, chunk.start_line)
            vectors.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": {
                    "repo": repo_name,
                    "file": chunk.file_path,
                    "language": chunk.language,
                    "snippet": chunk.content[:1000],  # Limit snippet size
                    "chunk_type": chunk.chunk_type,
                    "name": chunk.name or "",
                    "importance": chunk.importance,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                },
            })

        # Upsert in batches of 100
        batch_size = 100
        upserted = 0
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self.index.upsert(vectors=batch)
            upserted += len(batch)
            logger.info(f"Upserted {upserted}/{len(vectors)} vectors")

        return upserted

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        repo_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar code chunks."""
        filter_dict = None
        if repo_filter:
            filter_dict = {"repo": {"$eq": repo_filter}}
            logger.info(f"Searching with repo filter: {repo_filter}")

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict,
        )
        
        logger.info(f"Found {len(results.matches)} matches")

        return [
            {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata,
            }
            for match in results.matches
        ]

    def delete_repo(self, repo_name: str) -> None:
        """Delete all vectors for a repository."""
        self.index.delete(filter={"repo": {"$eq": repo_name}})
        logger.info(f"Deleted vectors for repo: {repo_name}")

    def repo_exists(self, repo_name: str) -> bool:
        """Check if a repository has been indexed in Pinecone."""
        try:
            # Query with a zero vector and repo filter to check if any vectors exist
            zero_vector = [0.0] * EMBEDDING_DIMENSIONS
            results = self.index.query(
                vector=zero_vector,
                top_k=1,
                filter={"repo": {"$eq": repo_name}},
                include_metadata=False,
            )
            exists = len(results.matches) > 0
            logger.info(f"Repo '{repo_name}' exists in Pinecone: {exists}")
            return exists
        except Exception as e:
            logger.error(f"Error checking repo existence: {e}")
            return False

    def _generate_chunk_id(self, repo: str, file_path: str, start_line: int) -> str:
        """Generate a unique ID for a code chunk."""
        content = f"{repo}:{file_path}:{start_line}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]


def get_pinecone_store() -> PineconeStore:
    """Factory function to get Pinecone store instance."""
    return PineconeStore()
