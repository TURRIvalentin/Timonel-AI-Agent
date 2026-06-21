"""Tests for the FastAPI REST endpoints.

All tests use FastAPI's TestClient (backed by httpx).  The core layer
(setup_qa_chain, execute_query, ingest) and the DB layer (init_db,
save_query_log) are fully mocked so no network calls, model downloads,
or real disk I/O happen.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_qa_chain
from src.api.main import app
from src.core.ingest import IngestResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_chain() -> MagicMock:
    """A mock QA chain whose invoke() returns a canned answer."""
    chain = MagicMock()
    chain.invoke.return_value = {
        "answer": "Starboard is the right side of a vessel.",
        "context": [],
    }
    return chain


@pytest.fixture()
def api_client(mock_chain: MagicMock) -> TestClient:
    """TestClient with lifespan skipped and a mock chain injected.

    * ``setup_qa_chain`` and ``init_db`` in main.py are patched so the
      lifespan never touches real services.
    * ``get_qa_chain`` dependency is overridden to return ``mock_chain``
      directly, bypassing ``app.state`` entirely.
    * ``save_query_log`` is patched to avoid real DB writes in /query tests.
    """
    with (
        patch("src.api.main.setup_qa_chain", return_value=mock_chain),
        patch("src.api.main.init_db"),
        patch("src.api.routes.save_query_log"),
    ):
        app.dependency_overrides[get_qa_chain] = lambda: mock_chain
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client  # type: ignore[misc]
        app.dependency_overrides.clear()


@pytest.fixture()
def api_client_no_chain() -> TestClient:
    """TestClient where the vector store is not initialised (chain = None).

    The lifespan raises RuntimeError from setup_qa_chain, setting
    app.state.qa_chain to None.  The get_qa_chain dependency is NOT
    overridden, so /query correctly returns 503.
    """
    with (
        patch(
            "src.api.main.setup_qa_chain",
            side_effect=RuntimeError("ChromaDB not found."),
        ),
        patch("src.api.main.init_db"),
        TestClient(app, raise_server_exceptions=False) as client,
    ):
        yield client  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_returns_200_when_chain_ready(self, api_client: TestClient) -> None:
        response = api_client.get("/health")
        assert response.status_code == 200

    def test_status_ok_when_chain_ready(self, api_client: TestClient) -> None:
        data = api_client.get("/health").json()
        assert data["status"] == "ok"
        assert data["vector_store"] == "ready"

    def test_vector_store_not_initialized_when_no_chain(
        self, api_client_no_chain: TestClient
    ) -> None:
        data = api_client_no_chain.get("/health").json()
        assert data["status"] == "ok"
        assert data["vector_store"] == "not_initialized"


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------


class TestQueryEndpoint:
    def test_returns_200_with_valid_question(self, api_client: TestClient) -> None:
        response = api_client.post("/query", json={"question": "What is starboard?"})
        assert response.status_code == 200

    def test_response_contains_answer_and_sources(self, api_client: TestClient) -> None:
        data = api_client.post("/query", json={"question": "What is starboard?"}).json()
        assert data["answer"] == "Starboard is the right side of a vessel."
        assert isinstance(data["sources"], list)
        assert data["question"] == "What is starboard?"

    def test_chain_is_invoked_with_correct_input(
        self, api_client: TestClient, mock_chain: MagicMock
    ) -> None:
        api_client.post("/query", json={"question": "Define port."})
        mock_chain.invoke.assert_called_once_with({"input": "Define port."})

    def test_sources_are_mapped_from_document_metadata(
        self, api_client: TestClient, mock_chain: MagicMock
    ) -> None:
        from langchain_core.documents import Document

        doc = Document(
            page_content="The hull is the body of the ship.",
            metadata={"source": "/data/manual.pdf", "page": 7},
        )
        mock_chain.invoke.return_value = {"answer": "The hull.", "context": [doc]}

        data = api_client.post("/query", json={"question": "What is the hull?"}).json()
        assert len(data["sources"]) == 1
        assert data["sources"][0]["source"] == "/data/manual.pdf"
        assert data["sources"][0]["page"] == 7
        assert "hull" in data["sources"][0]["content"]

    def test_empty_question_returns_422(self, api_client: TestClient) -> None:
        response = api_client.post("/query", json={"question": ""})
        assert response.status_code == 422

    def test_missing_question_field_returns_422(self, api_client: TestClient) -> None:
        response = api_client.post("/query", json={})
        assert response.status_code == 422

    def test_returns_503_when_chain_not_ready(self, api_client_no_chain: TestClient) -> None:
        response = api_client_no_chain.post("/query", json={"question": "anything?"})
        assert response.status_code == 503

    def test_503_detail_mentions_ingest(self, api_client_no_chain: TestClient) -> None:
        detail = api_client_no_chain.post("/query", json={"question": "anything?"}).json()[
            "detail"
        ]
        assert "ingest" in detail.lower()

    def test_chain_exception_returns_503(
        self, api_client: TestClient, mock_chain: MagicMock
    ) -> None:
        mock_chain.invoke.side_effect = RuntimeError("ChromaDB unreachable.")
        response = api_client.post("/query", json={"question": "Will this fail?"})
        assert response.status_code == 503


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------


class TestIngestEndpoint:
    def test_returns_200_on_success(self, api_client: TestClient) -> None:
        with (
            patch(
                "src.api.routes.run_ingest",
                return_value=IngestResult(pages_loaded=12, chunks_indexed=60),
            ),
            patch("src.api.routes.setup_qa_chain"),
        ):
            response = api_client.post("/ingest")
        assert response.status_code == 200

    def test_response_contains_counts(self, api_client: TestClient) -> None:
        with (
            patch(
                "src.api.routes.run_ingest",
                return_value=IngestResult(pages_loaded=12, chunks_indexed=60),
            ),
            patch("src.api.routes.setup_qa_chain"),
        ):
            data = api_client.post("/ingest").json()
        assert data["status"] == "ok"
        assert data["pages_loaded"] == 12
        assert data["chunks_indexed"] == 60

    def test_missing_pdf_dir_returns_503(self, api_client: TestClient) -> None:
        with patch(
            "src.api.routes.run_ingest",
            side_effect=FileNotFoundError("PDF directory does not exist: /app/data"),
        ):
            response = api_client.post("/ingest")
        assert response.status_code == 503

    def test_no_pdfs_found_returns_503(self, api_client: TestClient) -> None:
        with patch(
            "src.api.routes.run_ingest",
            side_effect=RuntimeError("No PDF documents found"),
        ):
            response = api_client.post("/ingest")
        assert response.status_code == 503

    def test_reloads_chain_after_success(
        self, api_client: TestClient, mock_chain: MagicMock
    ) -> None:
        """After a successful ingest, app.state.qa_chain should be refreshed."""
        fresh_chain = MagicMock()
        with (
            patch(
                "src.api.routes.run_ingest",
                return_value=IngestResult(pages_loaded=5, chunks_indexed=25),
            ),
            patch("src.api.routes.setup_qa_chain", return_value=fresh_chain),
        ):
            api_client.post("/ingest")

        # The dependency override still returns mock_chain for /query,
        # but we can verify the chain was set on app.state.
        assert app.state.qa_chain is fresh_chain
