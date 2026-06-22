"""Document ingest pipeline.

Loads PDF files from disk, splits them into overlapping chunks, computes
embeddings, and persists everything to ChromaDB.  This module is pure
business logic — no CLI output, no Streamlit state, no side effects
beyond disk I/O.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Settings
from src.core.embeddings import get_embeddings

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Summary returned by :func:`ingest` after a successful pipeline run.

    Attributes:
        pages_loaded: Total PDF pages parsed (one page = one raw Document).
        chunks_indexed: Number of text chunks embedded and stored in ChromaDB.
    """

    pages_loaded: int
    chunks_indexed: int


def load_documents(directory: Path) -> list[Document]:
    """Load all PDF files from *directory* recursively.

    Corrupted or unreadable PDFs are logged and skipped so that one bad
    file does not abort the entire pipeline.

    Args:
        directory: Root path to scan for ``.pdf`` files.

    Returns:
        A flat list of ``Document`` objects (one per PDF page) with
        ``source`` and ``filename`` metadata fields populated.

    Raises:
        FileNotFoundError: When *directory* does not exist.
    """
    if not directory.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {directory}")

    pdf_files = list(directory.rglob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", directory)
        return []

    logger.info("Found %d PDF file(s) in %s", len(pdf_files), directory)
    docs: list[Document] = []
    for pdf_path in pdf_files:
        logger.debug("Loading: %s", pdf_path)
        try:
            pages = PyPDFLoader(str(pdf_path)).load()
            for page in pages:
                page.metadata["source"] = str(pdf_path)
                page.metadata["filename"] = pdf_path.name
            docs.extend(pages)
            logger.info("Loaded %d page(s) from %s", len(pages), pdf_path.name)
        except Exception:
            logger.exception("Failed to load %s — skipping", pdf_path)

    logger.info("Total pages loaded: %d", len(docs))
    return docs


def chunk_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split *documents* into overlapping text chunks.

    Args:
        documents: Raw ``Document`` objects to split.
        chunk_size: Target character count per chunk.
        chunk_overlap: Character overlap between consecutive chunks.

    Returns:
        List of smaller ``Document`` chunks, each preserving the original
        ``source`` and ``page`` metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(
        "Split %d document(s) into %d chunk(s) (chunk_size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


def build_vector_store(chunks: list[Document], settings: Settings) -> Chroma:
    """Embed *chunks* and persist them to ChromaDB.

    Args:
        chunks: Text chunks to embed and index.
        settings: Application settings (provides paths and embedding config).

    Returns:
        The populated ``Chroma`` vector store instance.

    Raises:
        ValueError: When *chunks* is empty.
    """
    if not chunks:
        raise ValueError("Cannot build a vector store from an empty chunk list.")

    embeddings = get_embeddings(settings)
    persist_dir = str(settings.chroma_db_dir)
    logger.info("Persisting %d chunk(s) to ChromaDB at %s", len(chunks), persist_dir)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    logger.info("Vector store built successfully.")
    return vector_store


def ingest(settings: Settings) -> IngestResult:
    """Run the full ingest pipeline end-to-end.

    Loads PDFs → chunks text → embeds → persists to ChromaDB.

    Args:
        settings: Application settings.

    Returns:
        An :class:`IngestResult` with counts of pages loaded and chunks indexed.

    Raises:
        FileNotFoundError: When the configured PDF directory does not exist.
        RuntimeError: When no valid documents were found to index.
    """
    documents = load_documents(settings.pdf_directory)
    if not documents:
        raise RuntimeError(
            f"No PDF documents found in {settings.pdf_directory}. "
            "Add PDF files to the data directory before running ingest."
        )

    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
    settings.chroma_db_dir.mkdir(parents=True, exist_ok=True)
    build_vector_store(chunks, settings)
    logger.info("Ingest complete — %d chunks indexed.", len(chunks))
    return IngestResult(pages_loaded=len(documents), chunks_indexed=len(chunks))
