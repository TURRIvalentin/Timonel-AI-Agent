"""Application configuration via pydantic-settings.

All settings are read from environment variables (case-insensitive) and,
when present, from a ``.env`` file at the project root.  No value is
hard-coded to a machine-specific path.

Usage::

    from src.config import settings

    print(settings.pdf_directory)      # Path object, absolute
    print(settings.embeddings_provider)  # "huggingface" | "openai"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Resolves to the project root regardless of the working directory at runtime.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central, validated configuration for Timonel RAG.

    All attributes map 1-to-1 to an environment variable of the same name
    (upper-cased).  Defaults are safe for a freshly cloned repo — no
    absolute paths, no secrets required for the HuggingFace + Ollama path.

    Attributes:
        pdf_directory: Directory that contains source PDF documents.
        chroma_db_dir: Persistence directory for the ChromaDB vector store.
        db_path: SQLite file path for query audit logs.
        embeddings_provider: Which embedding backend to use.
        llm_provider: Which LLM backend to use.
        hf_embedding_model: HuggingFace sentence-transformer model name.
        openai_model_name: OpenAI chat model identifier.
        local_llm_model: Ollama model name.
        retriever_k: Number of chunks returned per retrieval query.
        chunk_size: Target character count per document chunk.
        chunk_overlap: Character overlap between consecutive chunks.
        openai_api_key: OpenAI API key (required for OpenAI providers).
        log_level: Python logging level name.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Paths (defaults are absolute so the app is CWD-independent) ---
    pdf_directory: Path = Field(default=_PROJECT_ROOT / "data")
    chroma_db_dir: Path = Field(default=_PROJECT_ROOT / "chroma_db")
    db_path: Path = Field(default=_PROJECT_ROOT / "data" / "timonel_logs.db")

    # --- Providers ---
    embeddings_provider: Literal["huggingface", "openai"] = "huggingface"
    llm_provider: Literal["openai", "ollama"] = "openai"

    # --- Models ---
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_model_name: str = "gpt-3.5-turbo"
    local_llm_model: str = "llama3"

    # --- Retrieval tuning ---
    retriever_k: int = Field(default=4, ge=1)
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=200, ge=0)

    # --- Secrets ---
    openai_api_key: str = ""

    # --- Observability ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @field_validator("pdf_directory", "chroma_db_dir", "db_path", mode="before")
    @classmethod
    def _expand_path(cls, v: object) -> Path:
        """Expand ``~`` in user-provided paths."""
        return Path(str(v)).expanduser()

    @model_validator(mode="after")
    def _warn_missing_api_key(self) -> Settings:
        """Emit a warning when an OpenAI provider is selected without a key."""
        needs_key = self.embeddings_provider == "openai" or self.llm_provider == "openai"
        if needs_key and not self.openai_api_key:
            logger.warning(
                "OPENAI_API_KEY is not set but an OpenAI provider is configured. "
                "Set OPENAI_API_KEY in your .env file or environment."
            )
        return self


settings = Settings()
