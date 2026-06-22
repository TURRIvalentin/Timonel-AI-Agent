"""Tests for src/core/embeddings.py — embedding model factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config import Settings
from src.core.embeddings import get_embeddings


class TestGetEmbeddings:
    def test_huggingface_branch_called_with_correct_model(self, test_settings: Settings) -> None:
        with patch("src.core.embeddings.HuggingFaceEmbeddings") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = get_embeddings(test_settings)

        mock_cls.assert_called_once_with(model_name=test_settings.hf_embedding_model)
        assert result is mock_cls.return_value

    def test_openai_branch_called_with_api_key(self, test_settings: Settings) -> None:
        s = test_settings.model_copy(
            update={"embeddings_provider": "openai", "openai_api_key": "sk-test"}
        )
        with patch("src.core.embeddings.OpenAIEmbeddings") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = get_embeddings(s)

        mock_cls.assert_called_once_with(api_key="sk-test")
        assert result is mock_cls.return_value

    def test_unknown_provider_raises_value_error(self, test_settings: Settings) -> None:
        # model_copy bypasses Literal validation — lets us exercise the else-branch.
        s = test_settings.model_copy(update={"embeddings_provider": "unknown"})  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown embeddings_provider"):
            get_embeddings(s)
