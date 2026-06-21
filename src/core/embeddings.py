"""Shared embedding model factory.

Both the ingest and retrieval pipelines use this module so the same
model is never instantiated with inconsistent configurations.
"""

from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from src.config import Settings

logger = logging.getLogger(__name__)


def get_embeddings(settings: Settings) -> Embeddings:
    """Return the embedding model specified by *settings*.

    Args:
        settings: Application settings instance.

    Returns:
        A LangChain ``Embeddings`` object ready for use with ChromaDB.

    Raises:
        ValueError: When ``settings.embeddings_provider`` is not recognised.
    """
    provider = settings.embeddings_provider
    if provider == "openai":
        logger.info("Embedding backend: OpenAI")
        return OpenAIEmbeddings(api_key=settings.openai_api_key)  # type: ignore[arg-type]
    if provider == "huggingface":
        logger.info("Embedding backend: HuggingFace — %s", settings.hf_embedding_model)
        return HuggingFaceEmbeddings(model_name=settings.hf_embedding_model)
    raise ValueError(f"Unknown embeddings_provider: {provider!r}")
