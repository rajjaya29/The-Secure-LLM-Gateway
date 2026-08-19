"""Configuration settings for The Secure LLM Gateway."""

from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    APP_NAME: str = "The Secure LLM Gateway"

    # API-Key Authentication (X-API-Key)
    REQUIRE_API_KEY: bool = True
    VALID_API_KEYS: List[str] = [
        "sk-test-key-123",
        "demo-key",
        "sk-admin-master-key",
    ]

    # In-Memory Sliding-Window Rate Limiter
    ENABLE_RATE_LIMITING: bool = True
    RATE_LIMIT_REQUESTS: int = 60  # Max requests per sliding window
    RATE_LIMIT_WINDOW_SECONDS: int = 60  # Window size in seconds

    # Semantic Cache Settings (Sentence Transformers + ChromaDB)
    ENABLE_SEMANTIC_CACHE: bool = True
    SEMANTIC_SIMILARITY_THRESHOLD: float = Field(
        default=0.90,
        description="Cosine similarity threshold (>= 0.90) to consider a query a cache hit"
    )
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHROMA_PERSIST_DIR: Optional[str] = None  # None for in-memory, or path string
    CACHE_ISOLATE_BY_API_KEY: bool = True

    # Prompt Validation (Security Filtering)
    ENABLE_PROMPT_VALIDATION: bool = True
    ENABLE_PII_SCRUBBING: bool = True

    # Structured SQLite Request Logging
    SQLITE_DB_PATH: str = "gateway_logs.db"

    # Upstream Provider Configuration
    DEFAULT_PROVIDER: str = "mock"  # "mock", "openai", "anthropic", "ollama"
    MOCK_LLM_LATENCY_MS: float = 980.0  # Simulated realistic upstream LLM latency (~980ms)
    PROVIDER_TIMEOUT_SECONDS: float = 30.0

    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"

    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_BASE_URL: str = "https://api.anthropic.com/v1"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"


settings = Settings()
