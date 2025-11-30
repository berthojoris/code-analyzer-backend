from typing import List

from openai import OpenAI

from core.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


class EmbeddingService:
    """Service for generating embeddings using OpenAI API."""

    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = EMBEDDING_MODEL

    def create_embedding(self, text: str) -> List[float]:
        """Create an embedding for a single text."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error("Failed to create embedding", error=str(e))
            raise

    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts in batch."""
        if not texts:
            return []

        try:
            # OpenAI API supports batch embedding
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error("Failed to create batch embeddings", error=str(e))
            raise


def get_embedding_service() -> EmbeddingService:
    """Factory function to get embedding service instance."""
    return EmbeddingService()
