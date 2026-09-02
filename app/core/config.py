from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_timeout_seconds: float = Field(default=120.0, gt=0)
    max_upload_size_mb: int = Field(default=20, gt=0)
    max_pdf_pages: int = Field(default=300, gt=0)
    llm_chunk_max_chars: int = Field(default=30_000, gt=0)
    chunk_overlap_pages: int = Field(default=1, ge=0)
