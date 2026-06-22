"""Tests for src/core/retrieval.py — QA chain, fast-fail, run_query.

ChromaDB, embedding models, and the LLM are all mocked so no external
calls or model downloads happen during these tests.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.config import Settings
from src.core.retrieval import QueryResult, get_llm, run_query, setup_qa_chain

# ---------------------------------------------------------------------------
# QueryResult dataclass
# ---------------------------------------------------------------------------


class TestQueryResult:
    def test_sources_defaults_to_empty_list(self) -> None:
        r = QueryResult(question="Q?", answer="A.")
        assert r.sources == []

    def test_stores_all_fields(self) -> None:
        doc = Document(page_content="ctx", metadata={})
        r = QueryResult(question="Q?", answer="A.", sources=[doc])
        assert r.question == "Q?"
        assert r.answer == "A."
        assert len(r.sources) == 1

    def test_two_results_with_same_content_are_equal(self) -> None:
        r1 = QueryResult(question="Q", answer="A")
        r2 = QueryResult(question="Q", answer="A")
        assert r1 == r2


# ---------------------------------------------------------------------------
# get_llm — provider dispatch
# ---------------------------------------------------------------------------


class TestGetLlm:
    def test_unknown_provider_raises_value_error(self, test_settings: Settings) -> None:
        # Force an unsupported provider by going around Literal validation
        test_settings_copy = test_settings.model_copy(update={"llm_provider": "unknown"})  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown llm_provider"):
            get_llm(test_settings_copy)

    def test_openai_provider_returns_chat_model(self, test_settings: Settings) -> None:
        # ChatOpenAI is a lazy import inside get_llm(); patch at the source module.
        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            llm = get_llm(test_settings)
        assert llm is mock_cls.return_value

    def test_ollama_provider_returns_chat_model(self, test_settings: Settings) -> None:
        s = test_settings.model_copy(update={"llm_provider": "ollama"})
        # ChatOllama is a lazy import inside get_llm(); patch at the source module.
        with patch("langchain_community.chat_models.ChatOllama") as mock_cls:
            mock_cls.return_value = MagicMock()
            llm = get_llm(s)
        assert llm is mock_cls.return_value


# ---------------------------------------------------------------------------
# setup_qa_chain — fast-fail guards
# ---------------------------------------------------------------------------


class TestSetupQaChain:
    def test_raises_when_chroma_dir_missing(self, test_settings: Settings, tmp_path: Path) -> None:
        # chroma_db_dir points to a non-existent path
        s = test_settings.model_copy(update={"chroma_db_dir": tmp_path / "nonexistent"})
        with pytest.raises(RuntimeError, match="not found"):
            setup_qa_chain(s)

    def test_raises_when_collection_is_empty(
        self, test_settings: Settings, tmp_path: Path
    ) -> None:
        chroma_dir = tmp_path / "empty_chroma"
        chroma_dir.mkdir()
        s = test_settings.model_copy(update={"chroma_db_dir": chroma_dir})

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_store = MagicMock()
        mock_store._collection = mock_collection

        with (
            patch("src.core.retrieval.get_embeddings", return_value=MagicMock()),
            patch("src.core.retrieval.Chroma", return_value=mock_store),
            pytest.raises(RuntimeError, match="empty"),
        ):
            setup_qa_chain(s)

    def test_builds_chain_when_collection_nonempty(
        self, test_settings: Settings, tmp_path: Path
    ) -> None:
        chroma_dir = tmp_path / "chroma"
        chroma_dir.mkdir()
        s = test_settings.model_copy(update={"chroma_db_dir": chroma_dir})

        mock_collection = MagicMock()
        mock_collection.count.return_value = 10
        mock_store = MagicMock()
        mock_store._collection = mock_collection
        mock_store.as_retriever.return_value = MagicMock()

        with (
            patch("src.core.retrieval.get_embeddings", return_value=MagicMock()),
            patch("src.core.retrieval.Chroma", return_value=mock_store),
            patch("src.core.retrieval.get_llm", return_value=MagicMock()),
            patch(
                "src.core.retrieval.create_retrieval_chain",
                return_value=MagicMock(),
            ) as mock_chain_factory,
        ):
            result = setup_qa_chain(s)

        assert result is mock_chain_factory.return_value


# ---------------------------------------------------------------------------
# run_query — structured result
# ---------------------------------------------------------------------------


class TestRunQuery:
    def test_returns_query_result_instance(self, test_settings: Settings) -> None:
        doc = Document(
            page_content="context passage",
            metadata={"source": "doc.pdf", "page": 1},
        )
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"answer": "The answer.", "context": [doc]}

        with patch("src.core.retrieval.setup_qa_chain", return_value=mock_chain):
            result = run_query("What is this?", test_settings)

        assert isinstance(result, QueryResult)
        assert result.question == "What is this?"
        assert result.answer == "The answer."
        assert len(result.sources) == 1
        assert result.sources[0].metadata["source"] == "doc.pdf"

    def test_missing_answer_key_returns_empty_string(self, test_settings: Settings) -> None:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {}

        with patch("src.core.retrieval.setup_qa_chain", return_value=mock_chain):
            result = run_query("Q?", test_settings)

        assert result.answer == ""
        assert result.sources == []

    def test_missing_context_key_returns_empty_sources(self, test_settings: Settings) -> None:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"answer": "Yes."}

        with patch("src.core.retrieval.setup_qa_chain", return_value=mock_chain):
            result = run_query("Q?", test_settings)

        assert result.answer == "Yes."
        assert result.sources == []

    def test_chain_is_invoked_with_correct_key(self, test_settings: Settings) -> None:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"answer": "ok", "context": []}

        with patch("src.core.retrieval.setup_qa_chain", return_value=mock_chain):
            run_query("my question", test_settings)

        mock_chain.invoke.assert_called_once_with({"input": "my question"})
