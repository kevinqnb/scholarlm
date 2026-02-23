"""
TableCleaner: VLM-powered OCR table cleaning / normalization.

Normalize and restructure the tables that are already present in the OCR text.
Operates only on pages that contain ``<table>`` tags; 
missing-table reconstruction is intentionally out of scope here.

Backends
--------
Both methods support:
  - ``"vllm"``   – local inference (recommended for large-scale / cost-sensitive runs)
  - ``"openai"`` – API inference via the OpenAI client
"""

import re
import json
from copy import deepcopy
from itertools import count as itercount
from typing import Optional

from .instruction_prompts import (
    CLEAN_TABLE_INSTRUCTIONS_V3,
)
from .utils import process_pdf


# ---------------------------------------------------------------------------
# TableCleaner
# ---------------------------------------------------------------------------

class TableCleaner:
    """
    VLM-powered table cleaning for OCR-extracted research paper text.

    Args:
        backend: ``"vllm"`` (local) or ``"openai"`` (API).
        model_name: Required for the vllm backend.  Must be a multimodal VLM
            that supports image inputs, e.g. ``"Qwen/Qwen2.5-VL-7B-Instruct"``.
        sampling_params: Override default sampling parameters.  For the vllm
            backend these map to ``vllm.SamplingParams`` kwargs; for openai
            they map to ``client.chat.completions.create`` kwargs.
        openai_api_key: Required for the openai backend (falls back to the
            ``OPENAI_API_KEY`` environment variable).
        openai_model: OpenAI model identifier.  Default: ``"gpt-4o"``.
        openai_rate_limit: Maximum requests per minute for the OpenAI backend.
            Default: 30.
        target_longest_dim: Maximum pixel size for the longest edge when
            rendering PDF pages.  Default: 1536.
    """

    VLLM_BACKEND = "vllm"
    OPENAI_BACKEND = "openai"

    def __init__(
        self,
        backend: str = "vllm",
        model_name: str = None,
        sampling_params: dict = {},
        openai_api_key: str = None,
        openai_model: str = "gpt-4o",
        openai_rate_limit: int = 30,
        target_longest_dim: int = 1536,
    ):
        self.backend = backend
        self.target_longest_dim = target_longest_dim

        if backend == self.VLLM_BACKEND:
            from vllm import LLM, SamplingParams as VLLMSamplingParams
            from vllm.sampling_params import GuidedDecodingParams

            if model_name is None:
                raise ValueError("model_name is required for the vllm backend.")

            self.model_name = model_name
            self._SamplingParams = VLLMSamplingParams
            self._GuidedDecodingParams = GuidedDecodingParams
            self.sampling_params = {
                "temperature": 0.1,
                "max_tokens": 16384,
            } | sampling_params
            self.llm = LLM(model=model_name)

        elif backend == self.OPENAI_BACKEND:
            from openai import OpenAI
            import os

            if openai_api_key is None:
                openai_api_key = os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=openai_api_key)
            self.openai_model = openai_model
            self.openai_rate_limit = openai_rate_limit
            self.sampling_params = {
                "max_completion_tokens": 8192,
            } | sampling_params

        else:
            raise ValueError(
                f"Unknown backend: {backend!r}. Choose 'vllm' or 'openai'."
            )

    # -----------------------------------------------------------------------
    # Text helpers
    # -----------------------------------------------------------------------

    def _get_page_numbers(self, text: str) -> list[int]:
        return [int(p) for p in re.findall(r'<page number="(\d+)">', text)]

    def _get_page_text(self, text: str, page_number: int) -> str:
        tag = f'<page number="{page_number}">'
        start = text.find(tag)
        if start == -1:
            return ""
        start += len(tag)
        end = text.find("</page>", start)
        if end == -1:
            return ""
        return text[start:end].strip()

    def _has_tables(self, page_text: str) -> bool:
        return bool(re.search(r'<table number="\d+">', page_text))

    def _replace_page_content(
        self, full_text: str, page_number: int, new_content: str
    ) -> str:
        """Splice ``new_content`` into the ``<page number="N">…</page>`` block."""
        open_tag = f'<page number="{page_number}">'
        close_tag = "</page>"
        start = full_text.find(open_tag)
        if start == -1:
            return full_text
        content_start = start + len(open_tag)
        content_end = full_text.find(close_tag, content_start)
        if content_end == -1:
            return full_text
        return (
            full_text[:content_start]
            + "\n"
            + new_content
            + "\n"
            + full_text[content_end:]
        )

    def _load_images(
        self,
        pdf_paths: Optional[list[str]],
        images: Optional[list[list[str]]],
    ) -> list[list[str]]:
        if images is not None:
            return images
        if pdf_paths is None:
            raise ValueError("Either pdf_paths or images must be provided.")
        print("Loading PDF images...")
        return [
            process_pdf(p, target_longest_dim=self.target_longest_dim)
            for p in pdf_paths
        ]

    # -----------------------------------------------------------------------
    # Message builders
    # -----------------------------------------------------------------------

    def _image_block(self, image_b64: str) -> dict:
        block: dict = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
        }
        if self.backend == self.OPENAI_BACKEND:
            block["image_url"]["detail"] = "high"
        return block

    def _build_cleaning_message(
        self, image_b64: str, page_text: str
    ) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    self._image_block(image_b64),
                    {
                        "type": "text",
                        "text": (
                            f"## Instructions:\n{CLEAN_TABLE_INSTRUCTIONS_V3}\n\n"
                            f"## OCR Text:\n{page_text}"
                        ),
                    },
                ],
            }
        ]

    # -----------------------------------------------------------------------
    # Backend dispatch
    # -----------------------------------------------------------------------

    def _run_vllm_cleaning(self, messages: list[list[dict]]) -> list[str]:
        params = self._SamplingParams(**self.sampling_params)
        responses = self.llm.chat(messages=messages, sampling_params=params)
        return [r.outputs[0].text for r in responses]

    def _run_openai_batch(
        self,
        messages: list[list[dict]],
        extra_kwargs: dict = {},
    ) -> list[Optional[str]]:
        import time
        import backoff
        from openai import RateLimitError, APIError

        delay = 60.0 / max(self.openai_rate_limit, 1)
        api_kwargs = {
            "model": self.openai_model,
            **{k: v for k, v in self.sampling_params.items()},
            **extra_kwargs,
        }

        @backoff.on_exception(
            backoff.expo, (RateLimitError, APIError), max_time=120, max_tries=6
        )
        def _call(msgs):
            time.sleep(delay)
            return self.client.chat.completions.create(messages=msgs, **api_kwargs)

        results: list[Optional[str]] = []
        for i, msgs in enumerate(messages):
            print(f"  OpenAI call {i + 1}/{len(messages)}")
            try:
                resp = _call(msgs)
                results.append(resp.choices[0].message.content.strip())
            except Exception as exc:
                print(f"  Error on call {i + 1}: {exc}")
                results.append(None)
        return results

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def clean(
        self,
        texts: list[str],
        pdf_paths: list[str] = None,
        images: list[list[str]] = None,
    ) -> list[str]:
        """
        Normalize and restructure tables that are present in the OCR text.

        For each qualifying page the model receives the page image alongside
        the full page OCR text and returns the corrected page text with
        normalized tables in place.

        Args:
            texts: OCR text strings (one per document). 
            pdf_paths: Paths to the source PDF files.
            images: Pre-rendered page images.

        Returns:
            Cleaned OCR text strings in the same order as ``texts``.
        """
        all_images = self._load_images(pdf_paths, images)

        messages: list[list[dict]] = []
        message_ids: list[tuple[int, int]] = []

        for doc_idx, (text, doc_images) in enumerate(zip(texts, all_images)):
            for page_number in self._get_page_numbers(text):
                page_text = self._get_page_text(text, page_number)
                if not self._has_tables(page_text):
                    continue
                if page_number >= len(doc_images):
                    continue
                messages.append(
                    self._build_cleaning_message(
                        doc_images[page_number], page_text
                    )
                )
                message_ids.append((doc_idx, page_number))

        if not messages:
            print("No pages with tables found. Nothing to clean.")
            return deepcopy(texts)

        print(f"Cleaning tables on {len(messages)} pages...")
        if self.backend == self.VLLM_BACKEND:
            responses = self._run_vllm_cleaning(messages)
        else:
            responses = self._run_openai_batch(messages)

        cleaned_texts = deepcopy(texts)
        for (doc_idx, page_number), resp in zip(message_ids, responses):
            if not resp:
                continue
            cleaned_texts[doc_idx] = self._replace_page_content(
                cleaned_texts[doc_idx], page_number, resp.strip()
            )

        return cleaned_texts

    def save(self, texts: list[str], output_paths: list[str]):
        """
        Save cleaned OCR texts to disk.

        Args:
            texts: Text strings to save (from ``clean()``).
            output_paths: Destination file paths, one per document.
        """
        for text, path in zip(texts, output_paths):
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
