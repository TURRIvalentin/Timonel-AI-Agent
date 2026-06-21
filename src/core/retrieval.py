"""Retrieval-Augmented Generation chain.

Provides a factory for the LangChain retrieval chain and a thin wrapper
that returns structured output.  No I/O side effects — no prints, no
Streamlit state — so the same functions can be consumed by the Streamlit
UI, the CLI, and the future REST API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from src.config import Settings
from src.core.embeddings import get_embeddings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly based on the "
    "provided context fragments.\n"
    "If the answer cannot be found in the context, say so honestly — "
    "do not fabricate information.\n\n"
    "Context:\n{context}"
)


@dataclass
class QueryResult:
    """Structured result returned by :func:`run_query`.

    Attributes:
        question: The original user question.
        answer: The generated answer text.
        sources: Retrieved document chunks used to form the answer.
    """

    question: str
    answer: str
    sources: list[Document] = field(default_factory=list)


def get_llm(settings: Settings) -> BaseChatModel:
    """Return the LLM configured by *settings*.

    Args:
        settings: Application settings.

    Returns:
        A LangChain ``BaseChatModel`` implementation.

    Raises:
        ValueError: When ``settings.llm_provider`` is not recognised.
        ImportError: When the required provider library is not installed.
    """
    provider = settings.llm_provider
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        logger.info("LLM backend: OpenAI — %s", settings.openai_model_name)
        return ChatOpenAI(
            model=settings.openai_model_name,
            temperature=0,
            api_key=settings.openai_api_key,  # type: ignore[arg-type]
        )
    if provider == "ollama":
        from langchain_community.chat_models import ChatOllama

        logger.info("LLM backend: Ollama — %s", settings.local_llm_model)
        return ChatOllama(model=settings.local_llm_model, temperature=0)
    raise ValueError(f"Unknown llm_provider: {provider!r}")


def setup_qa_chain(settings: Settings) -> Runnable[dict[str, Any], dict[str, Any]]:
    """Build and return the retrieval-augmented generation chain.

    The returned chain accepts ``{"input": "<question>"}`` and returns a
    dict with ``"answer"`` (str) and ``"context"`` (list[Document]) keys.

    Args:
        settings: Application settings.

    Returns:
        A LangChain ``Runnable`` chain.

    Raises:
        RuntimeError: When the ChromaDB directory is missing or empty.
    """
    if not settings.chroma_db_dir.exists():
        raise RuntimeError(
            f"ChromaDB directory not found: {settings.chroma_db_dir}. "
            "Run `python cli.py ingest` first."
        )

    embeddings = get_embeddings(settings)
    logger.info("Loading ChromaDB from %s", settings.chroma_db_dir)
    vector_store = Chroma(
        persist_directory=str(settings.chroma_db_dir),
        embedding_function=embeddings,
    )

    # Fail fast if the collection is empty rather than returning empty answers.
    try:
        count = vector_store._collection.count()
        if count == 0:
            raise RuntimeError(
                "The ChromaDB collection is empty. Run `python cli.py ingest` to populate it."
            )
        logger.info("ChromaDB ready — %d chunks available.", count)
    except RuntimeError:
        raise
    except Exception:
        logger.debug("Could not read collection count — proceeding without check.")

    retriever = vector_store.as_retriever(search_kwargs={"k": settings.retriever_k})
    llm = get_llm(settings)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )
    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)


def run_query(question: str, settings: Settings) -> QueryResult:
    """Execute a retrieval query and return a structured result.

    Args:
        question: Natural-language question to answer.
        settings: Application settings.

    Returns:
        A :class:`QueryResult` containing the answer and cited source chunks.
    """
    chain = setup_qa_chain(settings)
    logger.info("Running query: %r", question)
    raw = chain.invoke({"input": question})
    return QueryResult(
        question=question,
        answer=raw.get("answer", ""),
        sources=raw.get("context", []),
    )
