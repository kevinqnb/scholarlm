"""TableCleaner: standalone table-cleaning pass over OCR text.

Fully separate from MeasurementLM -- this is now an explicit pre-processing
step (see experiments/run_table_cleaning.py), not something MeasurementLM.fit()
can be asked to do implicitly. Clean first, then point extraction's
params.ocr_dir at the cleaned output.
"""
from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path

from .instruction_prompts import CLEAN_TABLE_INSTRUCTIONS
from .measurementlm import BatchLLMBase, ContextLengthExceededError


class TableCleaner(BatchLLMBase):
    """Cleans and normalizes tables in OCR text using a vLLM-served model.

    For each page containing ``<table>`` tags, loads the pre-rendered page
    image (produced by ``process_pdfs.py``) and asks the model to correct
    and normalize the table markup against it. Pages without tables are
    returned unchanged.

    Args:
        model_name: The name or path of the vLLM-served model.
        sampling_params: Sampling parameters for text generation.
        api_base: Base URL of the vLLM OpenAI-compatible server.
        api_key: API key for the vLLM server (any non-empty string works).
        max_concurrent: Maximum concurrent in-flight requests.
        use_extra_body: Whether to forward sampling-params-derived fields
            (top_k, repetition_penalty, chat_template_kwargs) via extra_body.
        output_dir: If set, cleaned texts are saved as ``{stem}.txt`` files
            (where *stem* is the last component of the corresponding
            ``processed_pdf_dir`` path) in this directory.
    """

    def __init__(
        self,
        model_name: str,
        sampling_params: dict[str, any] = {},
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        max_concurrent: int = 32,
        use_extra_body: bool = True,
        output_dir: str | None = None,
    ):
        super().__init__(
            model_name=model_name,
            sampling_params=sampling_params,
            api_base=api_base,
            api_key=api_key,
            max_concurrent=max_concurrent,
            use_extra_body=use_extra_body,
        )
        self.output_dir = output_dir

    def clean(
        self,
        documents: list[str],
        processed_pdf_dirs: list[str],
    ) -> list[str]:
        """
        Clean and normalize tables in OCR text using the loaded vLLM model.

        Pre-processed images must be produced first by ``process_pdfs.py``,
        which saves each page as a base64 string at
        ``{processed_pdf_dir}/{page_index}.b64``.

        If ``self.output_dir`` is set, the cleaned texts are saved as
        ``{stem}.txt`` files (where *stem* is the last component of the
        ``processed_pdf_dir`` path) in that directory.

        Args:
            documents: OCR text strings, one per document.
            processed_pdf_dirs: Paths to the pre-processed image directories,
                one per document.  Each directory must contain ``{i}.b64``
                files (from ``process_pdfs.py``).

        Returns:
            Cleaned OCR text strings in the same order as ``documents``.
        """
        print("Loading pre-processed PDF images...")
        all_images: list[list[str]] = []
        for doc_dir in processed_pdf_dirs:
            doc_path = Path(doc_dir)
            if not doc_path.exists():
                raise FileNotFoundError(
                    f"Processed PDF directory not found: {doc_dir}\n"
                    f"Run 'python experiments/process_pdfs.py' first."
                )
            page_files = sorted(doc_path.glob("*.b64"), key=lambda p: int(p.stem))
            all_images.append([p.read_text().strip() for p in page_files])

        messages: list[list[dict]] = []
        message_ids: list[tuple[int, int]] = []  # (doc_idx, page_number)

        for doc_idx, (text, doc_images) in enumerate(zip(documents, all_images)):
            for page_number in self._get_page_numbers(text):
                page_text = self._get_page_text(text, page_number)
                if not re.search(r'<table number="\d+">', page_text):
                    continue
                if page_number >= len(doc_images):
                    continue
                image_b64 = doc_images[page_number]
                messages.append([{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                        {
                            "type": "text",
                            "text": (
                                f"## INSTRUCTIONS:\n{CLEAN_TABLE_INSTRUCTIONS}\n\n"
                                f"## OCR TEXT:\n{page_text}\n\n"
                                f"## QUERY:\nClean and normalize the tables in the OCR text, "
                                f"using the page image for reference. Return ONLY the cleaned "
                                f"OCR text for this page, with tables normalized and restructured "
                                f"as needed. Do NOT include any additional explanation, return "
                                f"ONLY the cleaned text.\n"
                            ),
                        },
                    ],
                }])
                message_ids.append((doc_idx, page_number))

        if not messages:
            print("No pages with tables found. Nothing to clean.")
            return deepcopy(documents)

        print(f"Cleaning tables on {len(messages)} pages...")
        response_texts = self._call_batch(
            messages,
            response_format=None,
            temperature=self.sampling_params.get('temperature'),
            max_tokens=16384,
            max_retries=4,
            max_concurrent=2,
            timeout=1200,
        )

        cleaned_documents = deepcopy(documents)
        for (doc_idx, page_number), cleaned_page_text in zip(message_ids, response_texts):
            if isinstance(cleaned_page_text, ContextLengthExceededError):
                self.context_length_exceeded_docs.add(doc_idx)
                continue
            cleaned_page_text = cleaned_page_text.strip()
            if not cleaned_page_text:
                continue
            open_tag = f'<page number="{page_number}">'
            close_tag = "</page>"
            full_text = cleaned_documents[doc_idx]
            start = full_text.find(open_tag)
            if start == -1:
                continue
            content_start = start + len(open_tag)
            content_end = full_text.find(close_tag, content_start)
            if content_end == -1:
                continue
            cleaned_documents[doc_idx] = (
                full_text[:content_start]
                + "\n"
                + cleaned_page_text
                + "\n"
                + full_text[content_end:]
            )

        if self.output_dir is not None:
            out_dir = Path(self.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            for proc_dir, cleaned_text in zip(processed_pdf_dirs, cleaned_documents):
                stem = Path(proc_dir).name
                out_file = out_dir / (stem + ".txt")
                with open(out_file, "w", encoding="utf-8") as fh:
                    fh.write(cleaned_text)
            print(f"Saved cleaned OCR to {out_dir}")

        return cleaned_documents
