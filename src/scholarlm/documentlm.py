import os
import re
import asyncio
import warnings
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import count
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
load_dotenv()
from scholarlm.utils import process_pdf, add_row_names

_QUALITY_PRESETS = {
    "accurate": {
        "target_longest_dim": 2048,
        "correct_orientation": True,
        "max_tokens": 16384,
        "max_retry_rounds": 5,
    },
    "fast": {
        "target_longest_dim": 1024,
        "correct_orientation": False,
        "max_tokens": 8192,
        "max_retry_rounds": 1,
    },
}


class DocumentLM:
    """
    A vision-language model (VLM) wrapper for converting PDF documents to markdown text.

    Args:
        model_name (str): Name or path of the VLM to use for OCR, served via a
            vLLM OpenAI-compatible endpoint.
        quality (str): Preset controlling the speed/accuracy trade-off. One of:
            - ``"accurate"`` (default): 2048px images, orientation correction,
              16384 max tokens, up to 5 retry rounds.
            - ``"fast"``: 1024px images, no orientation correction, 8192 max
              tokens, 1 retry round.
            Individual parameters below override the preset when provided.
        ocr_prompt (str): System prompt for the OCR task. Defaults to a standard
            instruction to convert PDF pages to markdown with HTML tables.
        sampling_params (dict[str, any]): Sampling parameters for text generation.
            ``max_tokens`` here overrides the quality preset.
        api_base (str): Base URL of the vLLM OpenAI-compatible endpoint.
        api_key (str): API key for the endpoint (use "EMPTY" for local vLLM).
        max_concurrent (int): Maximum number of concurrent async API calls and
            parallel PDF rendering threads.
        target_longest_dim (int | None): Render resolution override (pixels along
            the longest edge). Overrides the quality preset when set.
        correct_orientation (bool | None): Run Tesseract OSD to correct page
            rotation. Overrides the quality preset when set.
        max_retry_rounds (int | None): Maximum retry rounds for pages that exceed
            max_tokens or fail table extraction. Overrides the quality preset when
            set. Set to 0 to disable retries entirely.
    """
    def __init__(
        self,
        model_name: str,
        quality: str = "accurate",
        ocr_prompt: str = None,
        sampling_params: dict[str, any] = None,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        max_concurrent: int = 32,
        target_longest_dim: int | None = None,
        correct_orientation: bool | None = None,
        max_retry_rounds: int | None = None,
    ):
        if quality not in _QUALITY_PRESETS:
            raise ValueError(f"quality must be one of {list(_QUALITY_PRESETS)}, got {quality!r}")
        preset = _QUALITY_PRESETS[quality]

        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self.target_longest_dim = target_longest_dim if target_longest_dim is not None else preset["target_longest_dim"]
        self.correct_orientation = correct_orientation if correct_orientation is not None else preset["correct_orientation"]
        self.max_retry_rounds = max_retry_rounds if max_retry_rounds is not None else preset["max_retry_rounds"]

        self.sampling_params = {
            "temperature": 0.1,
            "max_tokens": preset["max_tokens"],
        } | (sampling_params or {})
        self.max_tokens = self.sampling_params.get("max_tokens", preset["max_tokens"])

        if ocr_prompt is None:
            self.ocr_prompt = (
                "Convert the pdf document to markdown text as accurately as possible. "
                "Display tables in html format. Rotate tables if they are presented sideways. "
                "Use hash symbols (e.g. #, ##) to indicate headings. "
                "Do not start a new line for italic items."
            )
        else:
            self.ocr_prompt = ocr_prompt

        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=api_base, timeout=600.0, max_retries=3)


    # -----------------------------------------------------------------------
    # Core API call helpers
    # -----------------------------------------------------------------------

    async def _acall_with_usage(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, int]:
        """Single async API call; returns (response_text, completion_tokens)."""
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.sampling_params.get("temperature", 0.1),
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }
        extra = {}
        if "top_k" in self.sampling_params:
            extra["top_k"] = self.sampling_params["top_k"]
        if "repetition_penalty" in self.sampling_params:
            extra["repetition_penalty"] = self.sampling_params["repetition_penalty"]
        if extra:
            kwargs["extra_body"] = extra
        try:
            response = await self.async_client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            return text, completion_tokens
        except Exception as e:
            print(f"API call failed: {e}")
            return "", 0

    def _call_batch_with_usage(
        self,
        message_sets: list[list[dict]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> list[tuple[str, int]]:
        """Dispatch all message sets concurrently; return (text, completion_tokens) pairs in order."""
        async def _run():
            sem = asyncio.Semaphore(self.max_concurrent)
            async def _limited(msgs):
                async with sem:
                    return await self._acall_with_usage(msgs, temperature, max_tokens)
            tasks = [_limited(msgs) for msgs in message_sets]
            return await asyncio.gather(*tasks)
        return asyncio.run(_run())


    # -----------------------------------------------------------------------
    # PDF OCR pipeline
    # -----------------------------------------------------------------------

    def _load_paper_images(
        self, i: int, filepath: str, processed_pdfs_dir: str | None
    ) -> tuple[int, list[str] | None]:
        """Render or load pre-rendered base64 images for one paper. Returns (i, images)."""
        if processed_pdfs_dir is not None:
            paper_code = Path(filepath).stem
            paper_dir = Path(processed_pdfs_dir) / paper_code
            if not paper_dir.exists():
                warnings.warn(
                    f"No pre-processed pages for '{paper_code}' in {processed_pdfs_dir}. Skipping."
                )
                return i, None
            b64_files = sorted(paper_dir.glob("*.b64"), key=lambda p: int(p.stem))
            if not b64_files:
                warnings.warn(f"Pre-processed directory {paper_dir} is empty. Skipping.")
                return i, None
            return i, [f.read_text() for f in b64_files]
        return i, process_pdf(
            filepath,
            target_longest_dim=self.target_longest_dim,
            correct_orientation=self.correct_orientation,
        )

    def fit(self, filepaths: list[str], processed_pdfs_dir: str | None = None):
        """
        OCR all pages of the provided PDF files and return the combined markdown text.

        Renders PDF pages in parallel (up to ``max_concurrent`` threads), then
        dispatches all pages to the VLM concurrently. Pages that exceed max_tokens
        or fail to produce expected table tags are retried with progressively higher
        temperature for up to ``max_retry_rounds`` rounds.

        Output text wraps each page in ``<page number="N">`` tags and numbers
        tables sequentially with ``<table number="N">`` tags.

        Args:
            filepaths (list[str]): Paths to the PDF files to process.
            processed_pdfs_dir (str | None): If provided, load pre-rendered base64
                page images from ``{processed_pdfs_dir}/{paper_code}/{page_index}.b64``
                instead of rendering PDFs at runtime.

        Returns:
            list[str]: Processed markdown text, one string per document.
        """
        self.filepaths = filepaths

        text_directory = os.getenv('POND_TEXT_PATH')
        if text_directory and not os.path.exists(text_directory):
            os.makedirs(text_directory, exist_ok=True)

        # Render all PDFs in parallel before any VLM work begins.
        paper_images: dict[int, list[str]] = {}
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            futures = {
                executor.submit(self._load_paper_images, i, fp, processed_pdfs_dir): i
                for i, fp in enumerate(filepaths)
            }
            for future in as_completed(futures):
                try:
                    i, b64_images = future.result()
                    if b64_images is not None:
                        paper_images[i] = b64_images
                except Exception as e:
                    i = futures[future]
                    warnings.warn(
                        f"Failed to process {filepaths[i]} with error: {e}. Skipping this file."
                    )

        messages = []
        message_paper_ids = []
        for i in range(len(filepaths)):
            if i not in paper_images:
                continue
            for img in paper_images[i]:
                image_data_uri = f'data:image/png;base64,{img}'
                messages.append([
                    {"role": "system", "content": self.ocr_prompt},
                    {
                        "role": "user",
                        "content": [{
                            "type": "image_url",
                            "image_url": {"url": image_data_uri},
                        }],
                    },
                ])
                message_paper_ids.append(i)

        results = self._call_batch_with_usage(messages)

        # Retry with higher temperature if max tokens exceeded or tables failed to be extracted.
        retry_round = 0
        temp = self.sampling_params.get("temperature", 0.1)
        while retry_round < self.max_retry_rounds and temp <= 1.0:
            retry_messages = []
            retry_message_ids = []
            for i, (text, completion_tokens) in enumerate(results):
                if completion_tokens >= self.max_tokens:
                    print(f"Max tokens exceeded for message {i}, retrying...")
                    retry_messages.append(messages[i])
                    retry_message_ids.append(i)
                    continue

                front_matter_match = re.match(r"^---\s*\n([\s\S]*?)\n---", text)
                if front_matter_match:
                    try:
                        metadata = yaml.safe_load(front_matter_match.group(1))
                    except yaml.YAMLError:
                        metadata = {}

                    is_table = metadata.get("is_table", False)
                    content_after_front_matter = text[front_matter_match.end():]
                    has_table_tags = "<table" in content_after_front_matter.lower()

                    if is_table and not has_table_tags:
                        print(f"Model reports is_table=True but no <table> tags found for message {i}, retrying...")
                        retry_messages.append([
                            {"role": "system", "content": self.ocr_prompt + " Focus on accurately extracting tables in html format."},
                            messages[i][1],
                        ])
                        retry_message_ids.append(i)

            if not retry_messages:
                break
            print(f"Retrying {len(retry_messages)} pages with temperature {temp + 0.2:.1f}...")
            temp += 0.2
            retry_round += 1
            retry_results = self._call_batch_with_usage(retry_messages, temperature=temp)
            for i, result in enumerate(retry_results):
                results[retry_message_ids[i]] = result

        response_texts = [text for text, _ in results]
        documents = [{} for _ in range(len(filepaths))]
        for i, text in enumerate(response_texts):
            paper_id = message_paper_ids[i]
            page_id = str(len(documents[paper_id]))
            cleaned_text = re.sub(r"^---[\s\S]*?---\s*", "", text)
            documents[paper_id][page_id] = cleaned_text

        self.text = []
        for document in documents:
            doc_chunks = document
            doc_text = ""
            pages = list(doc_chunks.keys())
            pages.sort(key=lambda x: int(x))
            for page_id in pages:
                chunk = doc_chunks[page_id]
                doc_text += f'<page number="{int(page_id)}">\n\n' + chunk + f"\n\n</page>\n\n"

            counter = count()
            doc_text = re.sub(
                r"<table>", lambda m: f'<table number="{next(counter) + 1}">', doc_text
            )

            self.text.append(doc_text)

        return self.text


    def save(self, filepaths: list[str]):
        """
        Save the processed OCR text to disk.

        Args:
            filepaths (list[str]): Output file paths, one per document. Must have
                the same length as the list passed to fit().
        """
        for i, text in enumerate(self.text):
            output_filepath = filepaths[i]
            with open(output_filepath, 'w', encoding='utf-8') as file:
                file.write(text)
