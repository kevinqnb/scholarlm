import os
import re
import asyncio
import warnings
import yaml
from itertools import count
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
load_dotenv()
from scholarlm.utils import process_pdf, add_row_names


class DocumentLM:
    """
    A vision-language model (VLM) wrapper for converting PDF documents to markdown text.

    Args:
        model_name (str): Name or path of the VLM to use for OCR, served via a
            vLLM OpenAI-compatible endpoint.
        ocr_prompt (str): System prompt for the OCR task. Defaults to a standard
            instruction to convert PDF pages to markdown with HTML tables.
        sampling_params (dict[str, any]): Sampling parameters for text generation.
            Defaults to temperature=0.1 and max_tokens=16384.
        api_base (str): Base URL of the vLLM OpenAI-compatible endpoint.
        api_key (str): API key for the endpoint (use "EMPTY" for local vLLM).
        max_concurrent (int): Maximum number of concurrent async API calls.
    """
    def __init__(
        self,
        model_name: str,
        ocr_prompt: str = None,
        sampling_params: dict[str, any] = None,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        max_concurrent: int = 32,
    ):
        self.model_name = model_name
        self.sampling_params = {
            "temperature": 0.1,
            "max_tokens": 16384,
        } | (sampling_params or {})
        self.max_tokens = self.sampling_params.get("max_tokens", 16384)
        self.max_concurrent = max_concurrent

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
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=api_base, timeout=300.0, max_retries=3)


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
            "max_tokens": max_tokens if max_tokens is not None else self.sampling_params.get("max_tokens", 16384),
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

    def fit(self, filepaths: list[str]):
        """
        OCR all pages of the provided PDF files and return the combined markdown text.

        Renders each PDF page as an image, sends it through the VLM with the OCR
        prompt, and reassembles page outputs into a single document per file. Pages
        that exceed max_tokens or fail to produce expected table tags are retried
        with progressively higher temperature.

        Output text wraps each page in ``<page number="N">`` tags and numbers
        tables sequentially with ``<table number="N">`` tags.

        Args:
            filepaths (list[str]): Paths to the PDF files to process.

        Returns:
            list[str]: Processed markdown text, one string per document.
        """
        self.filepaths = filepaths

        text_directory = os.getenv('POND_TEXT_PATH')
        if text_directory and not os.path.exists(text_directory):
            os.makedirs(text_directory, exist_ok=True)

        messages = []
        message_paper_ids = []
        for i, filepath in enumerate(filepaths):
            try:
                b64_images = process_pdf(filepath, target_longest_dim=2048)
            except Exception as e:
                warnings.warn(f"Failed to process {filepath} with error: {e}. Skipping this file.")
                continue

            for j, img in enumerate(b64_images):
                image_data_uri = f'data:image/png;base64,{img}'
                message = [
                    {"role": "system", "content": self.ocr_prompt},
                    {
                        "role": "user",
                        "content": [{
                            "type": "image_url",
                            "image_url": {"url": image_data_uri},
                        }],
                    },
                ]
                messages.append(message)
                message_paper_ids.append(i)

        results = self._call_batch_with_usage(messages)

        # Retry with higher temperature if max tokens exceeded or if tables failed to be extracted.
        temp = self.sampling_params.get("temperature", 0.1)
        while temp <= 1.0:
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
