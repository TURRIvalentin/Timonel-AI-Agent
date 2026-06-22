"""Tests for src/core/ingest.py — chunking, document loading, vector store build.

All external I/O (PyPDFLoader, ChromaDB, HuggingFace) is mocked so that:
- No network calls are made.
- No model is downloaded.
- No real ChromaDB files are written (tmp_path is used when needed).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.config import Settings
from src.core.ingest import build_vector_store, chunk_documents, ingest, load_documents

# ---------------------------------------------------------------------------
# chunk_documents — pure function, no mocking required
# ---------------------------------------------------------------------------


class TestChunkDocuments:
    def test_large_document_is_split(self) -> None:
        content = "word " * 500  # 2500 chars > chunk_size=200
        doc = Document(page_content=content, metadata={"source": "test.pdf"})
        chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1

    def test_small_document_stays_as_one_chunk(self) -> None:
        doc = Document(page_content="short text", metadata={"source": "x.pdf"})
        chunks = chunk_documents([doc], chunk_size=1000, chunk_overlap=0)
        assert len(chunks) == 1

    def test_metadata_preserved_on_chunks(self) -> None:
        doc = Document(
            page_content="hello world test",
            metadata={"source": "test.pdf", "page": 3},
        )
        chunks = chunk_documents([doc], chunk_size=1000, chunk_overlap=0)
        assert all(c.metadata["source"] == "test.pdf" for c in chunks)
        assert all(c.metadata["page"] == 3 for c in chunks)

    def test_empty_input_returns_empty_list(self) -> None:
        assert chunk_documents([], chunk_size=200, chunk_overlap=20) == []

    def test_overlap_creates_content_redundancy(self) -> None:
        content = "a" * 400
        doc = Document(page_content=content, metadata={})
        chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=50)
        total_chars = sum(len(c.page_content) for c in chunks)
        # Total across overlapping chunks must exceed the original length
        assert total_chars > len(content)

    def test_multiple_documents_all_chunked(self) -> None:
        docs = [
            Document(page_content="word " * 100, metadata={"source": f"doc{i}.pdf"})
            for i in range(3)
        ]
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)
        assert len(chunks) >= 3  # at minimum one chunk per doc


# ---------------------------------------------------------------------------
# load_documents — filesystem interaction, PyPDFLoader mocked
# ---------------------------------------------------------------------------


class TestLoadDocuments:
    def test_missing_directory_raises_file_not_found(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            load_documents(missing)

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        assert load_documents(tmp_path) == []

    def test_non_pdf_files_are_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "readme.txt").write_text("not a pdf")
        (tmp_path / "data.csv").write_text("a,b,c")
        assert load_documents(tmp_path) == []

    def test_corrupted_pdf_is_skipped_not_raised(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"not a real pdf content")
        # Must return a list (possibly empty), not raise
        result = load_documents(tmp_path)
        assert isinstance(result, list)

    def test_valid_pdf_is_loaded_with_metadata(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()
        fake_page = Document(page_content="page content", metadata={"page": 0})

        with patch("src.core.ingest.PyPDFLoader") as mock_cls:
            mock_loader = MagicMock()
            mock_loader.load.return_value = [fake_page]
            mock_cls.return_value = mock_loader

            result = load_documents(tmp_path)

        assert len(result) == 1
        assert result[0].metadata["source"] == str(pdf_path)
        assert result[0].metadata["filename"] == "test.pdf"

    def test_multiple_pdfs_all_loaded(self, tmp_path: Path) -> None:
        for name in ("a.pdf", "b.pdf", "c.pdf"):
            (tmp_path / name).touch()

        fake_page = Document(page_content="content", metadata={"page": 0})
        with patch("src.core.ingest.PyPDFLoader") as mock_cls:
            mock_loader = MagicMock()
            mock_loader.load.return_value = [fake_page]
            mock_cls.return_value = mock_loader

            result = load_documents(tmp_path)

        assert len(result) == 3


# ---------------------------------------------------------------------------
# build_vector_store — Chroma and get_embeddings mocked
# ---------------------------------------------------------------------------


class TestBuildVectorStore:
    def test_empty_chunks_raises_value_error(self, test_settings: Settings) -> None:
        with pytest.raises(ValueError, match="empty"):
            build_vector_store([], test_settings)

    def test_calls_chroma_from_documents(self, test_settings: Settings) -> None:
        chunks = [Document(page_content="hello", metadata={"source": "x.pdf"})]
        fake_store = MagicMock()

        with (
            patch("src.core.ingest.get_embeddings", return_value=MagicMock()),
            patch("src.core.ingest.Chroma.from_documents", return_value=fake_store),
        ):
            result = build_vector_store(chunks, test_settings)

        assert result is fake_store

    def test_persist_directory_passed_to_chroma(self, test_settings: Settings) -> None:
        chunks = [Document(page_content="hello", metadata={"source": "x.pdf"})]

        with (
            patch("src.core.ingest.get_embeddings", return_value=MagicMock()),
            patch("src.core.ingest.Chroma.from_documents") as mock_chroma,
        ):
            mock_chroma.return_value = MagicMock()
            build_vector_store(chunks, test_settings)

        call_kwargs = mock_chroma.call_args.kwargs
        assert call_kwargs["persist_directory"] == str(test_settings.chroma_db_dir)


# ---------------------------------------------------------------------------
# ingest — end-to-end orchestrator, all external calls mocked
# ---------------------------------------------------------------------------


class TestIngest:
    def test_raises_when_pdf_dir_missing(self, test_settings: Settings) -> None:
        # pdf_directory doesn't exist in tmp_path
        with pytest.raises(FileNotFoundError):
            ingest(test_settings)

    def test_raises_when_no_pdfs_found(self, test_settings: Settings, tmp_path: Path) -> None:
        test_settings.pdf_directory.mkdir(parents=True, exist_ok=True)
        with pytest.raises(RuntimeError, match="No PDF documents found"):
            ingest(test_settings)

    def test_full_pipeline_with_mocked_io(self, test_settings: Settings, tmp_path: Path) -> None:
        pdf_path = test_settings.pdf_directory
        pdf_path.mkdir(parents=True, exist_ok=True)
        (pdf_path / "test.pdf").touch()

        fake_page = Document(page_content="word " * 100, metadata={"page": 0})

        with patch("src.core.ingest.PyPDFLoader") as mock_loader_cls:
            mock_loader = MagicMock()
            mock_loader.load.return_value = [fake_page]
            mock_loader_cls.return_value = mock_loader

            with (
                patch("src.core.ingest.get_embeddings", return_value=MagicMock()),
                patch("src.core.ingest.Chroma.from_documents", return_value=MagicMock()),
            ):
                ingest(test_settings)  # must not raise
