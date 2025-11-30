import os
import pytest
from unittest.mock import patch

# Set test environment variables before importing app modules
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["PINECONE_API_KEY"] = "test-key"
os.environ["PINECONE_INDEX_NAME"] = "test-index"
os.environ["ALLOWED_ORIGINS"] = "http://localhost:3000"


@pytest.fixture(autouse=True)
def mock_settings():
    """Mock settings for all tests."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "PINECONE_API_KEY": "test-key",
        "PINECONE_INDEX_NAME": "test-index",
        "ALLOWED_ORIGINS": "http://localhost:3000",
    }):
        yield
