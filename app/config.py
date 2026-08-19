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

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    APP_NAME: str = "Secure LLM Gateway"
    API_KEY: Optional[str] = None  # If set, clients must send Bearer token

    # Semantic Cache Settings
    ENABLE_SEMANTIC_CACHE: bool = True
    CACHE_SIMILARITY_THRESHOLD: float = Field(
        default=0.92,
        description="Cosine similarity threshold (>= 0.92) to consider a query a cache hit"
    )
    CACHE_MAX_ENTRIES: int = 10000
    CACHE_TTL_SECONDS: int = 86400  # 24 hours
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    # Security & Guardrails Settings
    ENABLE_INJECTION_GUARDRAIL: bool = True
    GUARDRAIL_BLOCK_INJECTIONS: bool = True  # Block (HTTP 400) or Flag/Sanitize
    INJECTION_CONFIDENCE_THRESHOLD: float = 0.70

    ENABLE_PII_SCRUBBING: bool = True
    PII_MASK_STYLE: str = "tokenized"  # "tokenized" e.g. <EMAIL_1> or "redacted" [REDACTED_EMAIL]
    
    ENABLE_OUTPUT_GUARDRAIL: bool = True
    OUTPUT_LEAK_PREVENTION: bool = True

    # Resilience & Rate Limiting
    ENABLE_RATE_LIMITING: bool = True
    RATE_LIMIT_RPM: int = 120  # Requests per minute
    RATE_LIMIT_TPM: int = 100000  # Tokens per minute
    
    # Circuit Breaker & Retry
    CIRCUIT_BREAKER_FAIL_THRESHOLD: int = 3
    CIRCUIT_BREAKER_RECOVERY_SECONDS: int = 30
    MAX_PROVIDER_RETRIES: int = 2
    PROVIDER_TIMEOUT_SECONDS: float = 30.0

    # Provider & Router Configuration
    DEFAULT_PROVIDER: str = "mock"  # "mock", "openai", "anthropic", "ollama"
    PROVIDER_PRIORITY: List[str] = ["mock", "openai", "ollama", "anthropic"]
    
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_BASE_URL: str = "https://api.anthropic.com/v1"
    
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"


settings = Settings()
