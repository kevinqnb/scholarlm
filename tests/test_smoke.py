"""Smoke tests for the four main classes.

Verifies that import chains work and classes can be instantiated correctly.
Heavy dependencies (LLM loading, GPU) are mocked out so these run anywhere.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel


# ── imports ───────────────────────────────────────────────────────────────────


def test_import_documentlm():
    from scholarlm import DocumentLM
    assert DocumentLM is not None


def test_import_measurementlm():
    from scholarlm import MeasurementLM
    assert MeasurementLM is not None


def test_import_judgementlm():
    from scholarlm import JudgementLM
    assert JudgementLM is not None


def test_import_tablecleaner():
    from scholarlm import TableCleaner
    assert TableCleaner is not None


# ── instantiation ─────────────────────────────────────────────────────────────


def test_documentlm_instantiation():
    """DocumentLM sets up defaults without loading any model."""
    from scholarlm import DocumentLM

    with patch("scholarlm.documentlm.LLM") as mock_llm:
        mock_llm.return_value = MagicMock()
        doc_lm = DocumentLM(model="mock-model")

    assert doc_lm.model == "mock-model"
    assert isinstance(doc_lm.ocr_prompt, str)
    assert doc_lm.max_tokens > 0


def test_documentlm_custom_prompt():
    from scholarlm import DocumentLM

    with patch("scholarlm.documentlm.LLM") as mock_llm:
        mock_llm.return_value = MagicMock()
        doc_lm = DocumentLM(model="mock-model", ocr_prompt="Custom prompt.")

    assert doc_lm.ocr_prompt == "Custom prompt."


def test_tablecleaner_instantiation_openai():
    """TableCleaner with openai backend instantiates without a GPU."""
    from scholarlm import TableCleaner

    with patch("openai.OpenAI") as mock_client:
        mock_client.return_value = MagicMock()
        cleaner = TableCleaner(backend="openai", openai_api_key="test-key")

    assert cleaner.backend == "openai"


def test_tablecleaner_vllm_requires_model_name():
    """TableCleaner with vllm backend raises if model_name is omitted."""
    from scholarlm import TableCleaner

    with pytest.raises(ValueError, match="model_name"):
        TableCleaner(backend="vllm")


def test_measurementlm_instantiation():
    """MeasurementLM stores config correctly; LLM loading is mocked."""

    class DummySchema(BaseModel):
        items: list[str]

    with patch("scholarlm.measurementlm.LLM") as mock_llm:
        mock_llm.return_value = MagicMock()
        from scholarlm import MeasurementLM

        mlm = MeasurementLM(
            model_name="mock-model",
            entity_identification_prompt="Find entities.",
            entity_identification_schema=DummySchema,
            attribute_info_dict={"ph": {"description": "pH level"}},
        )

    assert mlm.model_name == "mock-model"
    assert mlm.attribute_info_dict == {"ph": {"description": "pH level"}}
    mock_llm.assert_called_once_with(model="mock-model")


def test_judgementlm_instantiation():
    """JudgementLM stores config correctly; model loading is mocked."""
    mock_model = MagicMock()
    mock_model.tokenizer.pad_token = "[PAD]"
    mock_model.model.layers = [MagicMock()] * 4
    mock_model.config.num_attention_heads = 8
    mock_model.config.num_key_value_heads = 8
    mock_model.config.hidden_size = 512

    with patch("scholarlm.judgementlm.StandardizedTransformer", return_value=mock_model):
        from scholarlm import JudgementLM

        jlm = JudgementLM(model_name="mock-model", verbose=False)

    assert jlm.model_name == "mock-model"
    assert jlm.n_layers == 4
    assert jlm.n_heads == 8
    assert jlm.head_dim == 512 // 8
