from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str = Field(default="")
    pinecone_api_key: str = Field(default="")
    pinecone_index_name: str = Field(default="code-analyzer-index")
    pinecone_environment: str = Field(default="us-east-1")
    allowed_origins: str = Field(default="http://localhost:3000")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    temp_repo_dir: str = Field(default="./temp_repos")

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
