import os
import re
import json
import warnings
import yaml
import pandas as pd
from itertools import count
from dotenv import load_dotenv
load_dotenv()
from scholarlm.utils import process_pdf, add_row_names

from vllm import LLM, SamplingParams



class DocumentLM:
    """
    A vision-language model (VLM) wrapper for converting PDF documents to markdown text.

    Args:
        model: Name or path of the VLM to use for OCR, passed directly to vLLM's LLM.
        ocr_prompt (str): System prompt for the OCR task. Defaults to a standard
            instruction to convert PDF pages to markdown with HTML tables.
        sampling_params (dict[str, any]): Sampling parameters passed to vLLM's
            SamplingParams. Defaults to temperature=0.1 and max_tokens=16384.
    """
    def __init__(
            self,
            model = None,
            ocr_prompt: str = None,
            sampling_params: dict[str, any] = None,
        ):
        
        self.model = model
        if ocr_prompt is None:
            self.ocr_prompt = (
                "Convert the pdf document to markdown text as accurately as possible. "
                "Display tables in html format. Rotate tables if they are presented sideways. "
                "Use hash symbols (e.g. #, ##) to indicate headings. "
                "Do not start a new line for italic items."
            )
        else:
            self.ocr_prompt = ocr_prompt

        self.max_tokens = 8192
        if sampling_params is None:
            self.sampling_params = SamplingParams(
                temperature = 0.1,
                max_tokens = 16384,
            )
        else:
            if sampling_params.get("max_tokens") is not None:
                self.max_tokens = sampling_params.get("max_tokens")
            params = {'temperature': 0.1} | sampling_params | {'max_tokens': 16384}
            self.sampling_params = SamplingParams(**params)

        #if self.ocr:
        self.vlm = LLM(self.model)


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
        if not os.path.exists(text_directory):
            os.makedirs(text_directory, exist_ok=True)


        messages = []
        message_paper_ids = []
        for i, filepath in enumerate(filepaths):
            try:
                b64_images = process_pdf(filepath, target_longest_dim = 2048)
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
                            "image_url": {
                            "url": image_data_uri
                            }
                        }],
                    },
                ]
                messages.append(message)
                message_paper_ids.append(i)


        responses = self.vlm.chat(messages = messages, sampling_params = self.sampling_params)

        # Retry with higher temperature if max tokens exceeded or if tables failed to be extracted.
        temp = self.sampling_params.temperature
        while temp <= 1.0:
            self.sampling_params.temperature = temp
            retry_messages = []
            retry_message_ids = []
            for i, r in enumerate(responses):
                if len(r.outputs[0].token_ids) >= self.max_tokens:
                    print(f"Max tokens exceeded for message {i}, retrying...")
                    retry_messages.append(messages[i])
                    retry_message_ids.append(i)
                    continue

                front_matter_match = re.match(r"^---\s*\n([\s\S]*?)\n---", r.outputs[0].text)
                if front_matter_match:
                    try:
                        metadata = yaml.safe_load(front_matter_match.group(1))
                    except yaml.YAMLError:
                        metadata = {}

                    is_table = metadata.get("is_table", False)

                    content_after_front_matter = r.outputs[0].text[front_matter_match.end():]
                    has_table_tags = "<table" in content_after_front_matter.lower()

                    if is_table and not has_table_tags:
                        print(f"Model reports is_table=True but no <table> tags found for message {i}, retrying...")
                        retry_messages.append([
                            {"role": "system", "content": self.ocr_prompt + " Focus on accurately extracting tables in html format."},
                            messages[i][1],  # reuse the same image message
                        ])
                        retry_message_ids.append(i)
                

            if len(retry_messages) == 0:
                break
            print(f"Retrying {len(retry_messages)} pages with temperature {temp:.1f}...")
            retry_responses = self.vlm.chat(messages = retry_messages, sampling_params = self.sampling_params)
            for i, r in enumerate(retry_responses):
                idx = retry_message_ids[i]
                responses[idx] = r

            temp += 0.2


        response_text = [r.outputs[0].text for r in responses]
        documents = [{} for _ in range(len(filepaths))]
        for i, text in enumerate(response_text):
            paper_id = message_paper_ids[i]
            page_id = str(len(documents[paper_id]))

            # Remove any front-matter from the text
            cleaned_text = re.sub(r"^---[\s\S]*?---\s*", "", text)
            documents[paper_id][page_id] = cleaned_text


        self.text = []
        for document in documents:
            doc_chunks = document
            doc_text = ""
            pages = list(doc_chunks.keys())
            pages.sort(key = lambda x: int(x))
            for page_id in pages:
                chunk = doc_chunks[page_id]
                # Add page number tags
                doc_text += f'<page number="{int(page_id)}">\n\n' + chunk + f"\n\n</page>\n\n"

            # Add unique IDs to each table
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