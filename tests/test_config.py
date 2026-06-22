"""Tests for src/config.py — Settings parsing, validation, and coercion."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import _PROJECT_ROOT, Settings


class TestDefaultValues:
    """Verify field defaults without instantiating (avoids .env interference)."""

    def test_embeddings_provider_default(self) -> None:
        assert Settings.model_fields["embeddings_provider"].default == "huggingface"

    def test_llm_provider_default(self) -> None:
        assert Settings.model_fields["llm_provider"].default == "openai"

    def test_retriever_k_default(self) -> None:
        assert Settings.model_fields["retriever_k"].default == 4

    def test_chunk_size_default(self) -> None:
        assert Settings.model_fields["chunk_size"].default == 1000

    def test_chunk_overlap_default(self) -> None:
        assert Settings.model_fields["chunk_overlap"].default == 200

    def test_log_level_default(self) -> None:
        assert Settings.model_fields["log_level"].default == "INFO"

    def test_pdf_directory_default_is_absolute(self) -> None:
        default = Settings.model_fields["pdf_directory"].default
        assert isinstance(default, Path)
        assert default.is_absolute()

    def test_pdf_directory_default_under_project_root(self) -> None:
        default = Settings.model_fields["pdf_directory"].default
        assert default == _PROJECT_ROOT / "data"

    def test_chroma_db_dir_default_under_project_root(self) -> None:
        default = Settings.model_fields["chroma_db_dir"].default
        assert default == _PROJECT_ROOT / "chroma_db"


class TestValidation:
    """Pydantic constraint violations must raise ValidationError."""

    def test_chunk_size_below_minimum_raises(self) -> None:
        with pytest.raises(ValidationError, match="chunk_size"):
            Settings(chunk_size=50)  # ge=100

    def test_retriever_k_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="retriever_k"):
            Settings(retriever_k=0)  # ge=1

    def test_invalid_embeddings_provider_raises(self) -> None:
        with pytest.raises(ValidationError):
            Settings(embeddings_provider="cohere")  # type: ignore[arg-type]

    def test_invalid_llm_provider_raises(self) -> None:
        with pytest.raises(ValidationError):
            Settings(llm_provider="anthropic")  # type: ignore[arg-type]

    def test_invalid_log_level_raises(self) -> None:
        with pytest.raises(ValidationError):
            Settings(log_level="VERBOSE")  # type: ignore[arg-type]

    def test_negative_chunk_overlap_raises(self) -> None:
        with pytest.raises(ValidationError, match="chunk_overlap"):
            Settings(chunk_overlap=-1)  # ge=0


class TestInitKwargPriority:
    """Init kwargs must override whatever is in env / .env."""

    def test_explicit_chunk_size_is_used(self) -> None:
        s = Settings(chunk_size=500)
        assert s.chunk_size == 500

    def test_explicit_embeddings_provider_is_used(self) -> None:
        s = Settings(embeddings_provider="openai", openai_api_key="sk-test")
        assert s.embeddings_provider == "openai"

    def test_explicit_retriever_k_is_used(self) -> None:
        s = Settings(retriever_k=10)
        assert s.retriever_k == 10

    def test_explicit_path_is_used(self, tmp_path: Path) -> None:
        s = Settings(pdf_directory=tmp_path)  # type: ignore[arg-type]
        assert s.pdf_directory == tmp_path


class TestEnvVarOverride:
    """Environment variables must override defaults (but not init kwargs)."""

    def test_chunk_size_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHUNK_SIZE", "300")
        s = Settings()
        assert s.chunk_size == 300

    def test_string_env_var_coerced_to_int(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RETRIEVER_K", "7")
        s = Settings()
        assert s.retriever_k == 7
        assert isinstance(s.retriever_k, int)

    def test_embeddings_provider_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMBEDDINGS_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-test")
        s = Settings()
        assert s.embeddings_provider == "openai"

    def test_log_level_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        s = Settings()
        assert s.log_level == "DEBUG"


class TestPathExpansion:
    """The _expand_path validator must handle tilde and return a Path."""

    def test_tilde_is_expanded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PDF_DIRECTORY", "~/my_docs")
        s = Settings()
        assert "~" not in str(s.pdf_directory)

    def test_pdf_directory_is_path_object(self) -> None:
        s = Settings()
        assert isinstance(s.pdf_directory, Path)

    def test_chroma_db_dir_is_path_object(self) -> None:
        s = Settings()
        assert isinstance(s.chroma_db_dir, Path)
