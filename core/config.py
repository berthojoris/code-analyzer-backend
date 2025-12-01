from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Existing settings
    openai_api_key: str = Field(default="")
    pinecone_api_key: str = Field(default="")
    pinecone_index_name: str = Field(default="code-analyzer-index")
    pinecone_environment: str = Field(default="us-east-1")
    allowed_origins: str = Field(default="http://localhost:3000")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    temp_repo_dir: str = Field(default="./temp_repos")

    # Database Configuration
    database_url: str = Field(default="sqlite:///./analysis.db")
    database_pool_size: int = Field(default=5)
    database_max_overflow: int = Field(default=10)

    # Analysis Settings
    analysis_enabled: bool = Field(default=True)
    linting_enabled: bool = Field(default=True)
    quality_analysis_enabled: bool = Field(default=True)
    complexity_threshold: int = Field(default=10)
    max_file_size_mb: int = Field(default=10)

    # Tool Configurations
    ruff_config_path: str = Field(default="ruff.toml")
    flake8_config_path: str = Field(default=".flake8")
    black_config_path: str = Field(default="pyproject.toml")

    # Performance Settings
    analysis_timeout_seconds: int = Field(default=300)
    max_concurrent_analyses: int = Field(default=4)
    cache_analysis_results: bool = Field(default=True)

    # Analysis Settings
    analysis_enabled: bool = Field(default=True)
    linting_enabled: bool = Field(default=True)
    quality_analysis_enabled: bool = Field(default=True)
    complexity_threshold: int = Field(default=10)
    max_file_size_mb: int = Field(default=10)

    # Tool Configurations
    ruff_config_path: str = Field(default="ruff.toml")
    flake8_config_path: str = Field(default=".flake8")
    black_config_path: str = Field(default="pyproject.toml")

    # Performance Settings
    analysis_timeout_seconds: int = Field(default=300)
    max_concurrent_analyses: int = Field(default=4)
    cache_analysis_results: bool = Field(default=True)

    @property
    def allowed_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    def validate(self) -> None:
        errors = []
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required")
        if not self.pinecone_api_key:
            errors.append("PINECONE_API_KEY is required")
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")


settings = Settings()
