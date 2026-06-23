"""Tests for src/startup.py — auto-ingest chain initialisation.

No Streamlit, no network, no model downloads: all external calls are mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import Settings
from src.startup import init_rag_chain


class TestInitRagChain:
    def test_returns_chain_when_vector_store_ready(self, test_settings: Settings) -> None:
        mock_chain = MagicMock()
        with patch("src.startup.setup_qa_chain", return_value=mock_chain) as mock_setup:
            result = init_rag_chain(test_settings)
        assert result is mock_chain
        mock_setup.assert_called_once_with(test_settings)

    def test_reraises_unexpected_runtime_error(self, test_settings: Settings) -> None:
        with patch(
            "src.startup.setup_qa_chain",
            side_effect=RuntimeError("unexpected internal failure"),
        ):
            with pytest.raises(RuntimeError, match="unexpected internal failure"):
                init_rag_chain(test_settings)

    def test_auto_ingest_triggered_on_missing_vector_store(
        self, test_settings: Settings, tmp_path: Path
    ) -> None:
        """When ChromaDB is missing, ingest() is called and chain is returned."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "manual.pdf").touch()
        cfg = test_settings.model_copy(update={"pdf_directory": pdf_dir})

        mock_chain = MagicMock()

        with (
            patch(
                "src.startup.setup_qa_chain",
                side_effect=[
                    RuntimeError(
                        "ChromaDB directory not found: chroma_db. "
                        "Run `python cli.py ingest` first."
                    ),
                    mock_chain,
                ],
            ) as mock_setup,
            patch("src.startup.ingest") as mock_ingest,
        ):
            result = init_rag_chain(cfg)

        assert result is mock_chain
        mock_ingest.assert_called_once_with(cfg)
        assert mock_setup.call_count == 2

    def test_auto_ingest_triggered_on_empty_collection(
        self, test_settings: Settings, tmp_path: Path
    ) -> None:
        """When ChromaDB collection is empty, ingest() is called."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "guide.pdf").touch()
        cfg = test_settings.model_copy(update={"pdf_directory": pdf_dir})

        mock_chain = MagicMock()

        with (
            patch(
                "src.startup.setup_qa_chain",
                side_effect=[
                    RuntimeError("The ChromaDB collection is empty. Run `python cli.py ingest`."),
                    mock_chain,
                ],
            ),
            patch("src.startup.ingest"),
        ):
            result = init_rag_chain(cfg)

        assert result is mock_chain

    def test_raises_friendly_error_when_no_pdfs_and_store_missing(
        self, test_settings: Settings, tmp_path: Path
    ) -> None:
        """Empty pdf_directory + missing store → descriptive RuntimeError (no stack trace)."""
        pdf_dir = tmp_path / "empty_pdfs"
        pdf_dir.mkdir()
        cfg = test_settings.model_copy(update={"pdf_directory": pdf_dir})

        with (
            patch(
                "src.startup.setup_qa_chain",
                side_effect=RuntimeError("ChromaDB directory not found"),
            ),
            pytest.raises(RuntimeError, match="no PDF files found"),
        ):
            init_rag_chain(cfg)

    def test_raises_friendly_error_when_pdf_dir_absent(
        self, test_settings: Settings, tmp_path: Path
    ) -> None:
        """Non-existent pdf_directory + missing store → descriptive RuntimeError."""
        cfg = test_settings.model_copy(update={"pdf_directory": tmp_path / "nonexistent"})

        with (
            patch(
                "src.startup.setup_qa_chain",
                side_effect=RuntimeError("ChromaDB directory not found"),
            ),
            pytest.raises(RuntimeError, match="no PDF files found"),
        ):
            init_rag_chain(cfg)

    def test_ingest_not_called_when_store_is_ready(self, test_settings: Settings) -> None:
        """Happy path: ingest() must never be invoked when setup_qa_chain succeeds."""
        with (
            patch("src.startup.setup_qa_chain", return_value=MagicMock()),
            patch("src.startup.ingest") as mock_ingest,
        ):
            init_rag_chain(test_settings)

        mock_ingest.assert_not_called()
