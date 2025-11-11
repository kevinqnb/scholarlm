import os
import re
import json
from pdf2image import convert_from_path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from .utils import encode_pil_image

# Docling imports
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    RapidOcrOptions,
)
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling_core.types.doc import DocItemLabel
from docling.chunking import HierarchicalChunker


class DocumentLM:
    """
    A class to manage and process markdown documents for use with a language model.

    Args:
        model: The vision language model to be used for processing the documents.
        ocr (bool): Whether to use OCR for image-based documents. Defaults to false, 
            in which case the expected input documents are text or markdown files instead of pdfs.
        ocr_prompt (str): The prompt to use for OCR processing when ocr is set to True.
        sampling_params (dict[str, any]): Sampling parameters for the LLM during OCR processing.
    Attributes:
        model: The vision language model to be used for processing the documents.
        ocr (bool): Whether to use OCR for image-based documents.
        ocr_prompt (str): The prompt to use for OCR processing.
        sampling_params (SamplingParams): Sampling parameters for the LLM during OCR processing.
        filepaths (list[str]): A list of file paths for the documents to be processed.
        docling_documents (list[Doc]): A list of loaded Docling document objects.
        docling_chunks (list[list[DocChunk]]): A 2D list of Docling chunks derived from the documents.
        chunks (list[str]): A 2d list of text chunks derived from the documents.
    """
    def __init__(
            self,
            model = None,
            ocr = False,
            ocr_prompt: str = None,
            sampling_params: dict[str, any] = None,
        ):
        if ocr and model is None:
            raise ValueError("An OCR-capable model must be provided when ocr is set to True.")
        
        self.model = model
        self.ocr = ocr
        if ocr_prompt is None:
            self.ocr_prompt = "Convert the pdf document to markdown as accurately as possible. Rotate tables if they are presented sideways. Use hash symbols (e.g. #, ##) to indicate headings. Do not start a new line for italic items."
        else:
            self.ocr_prompt = ocr_prompt

        if sampling_params is None:
            self.sampling_params = SamplingParams(
                temperature = 0.1,
                max_tokens = 4096,
            )
        else:
            self.sampling_params = SamplingParams(**sampling_params)

        self.filepaths = None
        self.docling_documents = None
        self.docling_chunks = None
        self.chunks = None

        #if self.ocr:
        self.vlm = LLM(self.model)


    def chunk(self):
        """
        Split loaded documents into smaller chunks, using a token-based approach.
        Returns:
            chunks (list): A 2d list of text chunks derived from the documents. Each sublist
                corresponds to a single document, and items within it correspond to small,
                paragraph sized pieces of text.
        """
        docling_documents = []
        docling_chunks = []
        for i, filepath in enumerate(self.filepaths):
            accelerator_options = AcceleratorOptions(
                num_threads=8, device=AcceleratorDevice.CPU
            )
            ocr_options = RapidOcrOptions(force_full_page_ocr=True)
            pipeline_options = PdfPipelineOptions(
                do_table_structure=True,
                do_formula_enrichment=True,
                generate_page_images=True,
                images_scale=5.0,
            )
            pipeline_options.table_structure_options.do_cell_matching = False
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
            pipeline_options.accelerator_options = accelerator_options
            pipeline_options.ocr_options = ocr_options

            # Standard converter:
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                        backend = DoclingParseV4DocumentBackend
                    )
                }
            )
            result = converter.convert(filepath)
            docling_documents.append(result.document)
            #chunker = HierarchicalChunker()
            #chunks = chunker.chunk(dl_doc = result.document)
            #docling_chunks.append([c for c in chunks])

            doc_chunks = []
            for (item,_) in result.document.iterate_items():
                if item.label == DocItemLabel.TEXT or item.label == DocItemLabel.TABLE:
                    doc_chunks.append(item)
                if item.label == DocItemLabel.TABLE:
                    print("TABLE:")
                    print(item)
                    print()
                    print()
            docling_chunks.append(doc_chunks)

        self.docling_documents = docling_documents
        self.docling_chunks = docling_chunks

    '''
    def chunk(self):
        """
        Split loaded documents into smaller chunks, using a token-based approach.
        Returns:
            chunks (list): A 2d list of text chunks derived from the documents. Each sublist
                corresponds to a single document, and items within it correspond to small,
                paragraph sized pieces of text.
        """
        base_message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text":  "Convert this page to docling."},
                    ],
                },
            ]

        # Initialize LLM
        llm = LLM(
            model="ibm-granite/granite-docling-258M",
            revision="untied",
            limit_mm_per_prompt={"image": 1}
        )
        processor = AutoProcessor.from_pretrained("ibm-granite/granite-docling-258M")

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            skip_special_tokens=False,
        )

        messages = []
        message_paper_ids = []
        for i, filepath in enumerate(self.filepaths):
            images = convert_from_path(filepath)
            for img in images:
                if img.mode == "RGBA":
                    img = img.convert("RGB")

                prompt = processor.apply_chat_template(base_message, add_generation_prompt=True)
                messages.append({"prompt": prompt, "multi_modal_data": {"image": img}})
                message_paper_ids.append(i)

        results = llm.generate(messages, sampling_params=sampling_params)
        doc_pairs = [[] for _ in range(len(self.filepaths))]
        for i, output in enumerate(results):
            doctags = output.outputs[0].text
            img = messages[i]["multi_modal_data"]["image"]
            doc_pairs[message_paper_ids[i]].append((doctags, img))

        docling_documents = []
        docling_chunks = []
        for pairlist in doc_pairs:
            doctags, imgs = zip(*pairlist)
            doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
                doctags, imgs
            )
            doc = DoclingDocument.load_from_doctags(doctags_doc, document_name="Document")
            docling_documents.append(doc)

            #chunker = HierarchicalChunker()
            #chunks = chunker.chunk(dl_doc = doc)
            #docling_chunks.append([c for c in chunks])
            doc_chunks = []
            for (item,_) in doc.iterate_items():
                if item.label == DocItemLabel.TEXT or item.label == DocItemLabel.TABLE:
                    doc_chunks.append(item)
            docling_chunks.append(doc_chunks)
                    

        self.docling_documents = docling_documents
        self.docling_chunks = docling_chunks
        '''
    

    def ocr_read(self) -> list[list[str]]:
        """
        Load pdf documents using OCR, based on provided metadata.

        Args:

        Returns:
            chunks (list[list[str]]): A 2d list of text chunks derived from the documents. Each sublist
                corresponds to a single document, and items within it correspond to small,
                paragraph sized pieces of text.
        """
        messages = []
        message_paper_ids = []
        supplementary_text = []
        for i, doc in enumerate(self.docling_documents):
            for j, chunk in enumerate(self.docling_chunks[i]):
                #item = chunk.meta.doc_items[0]
                item = chunk
                img = item.get_image(doc = doc)
                if img.mode == "RGBA":
                    img = img.convert("RGB")

                base64_image = encode_pil_image(img)
                image_data_uri = f'data:image/png;base64,{base64_image}'
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
                #heading = chunk.meta.headings[0] if chunk.meta.headings is not None else ""
                #caption = chunk.meta.captions[0] if chunk.meta.captions is not None else ""
                heading = ""
                caption = item.caption_text(doc = doc) if item.label == DocItemLabel.TABLE else ""
                supplementary_text.append({'heading': heading, 'caption': caption})
        
        responses = self.vlm.chat(messages = messages, sampling_params = self.sampling_params)
        response_markdown = [r.outputs[0].text for r in responses]
        chunks = [[] for _ in range(len(self.docling_documents))]
        for msg_idx, paper_idx in enumerate(message_paper_ids):
            # Remove any front-matter from the markdown
            cleaned_markdown = re.sub(r"^---[\s\S]*?---\s*", "", response_markdown[msg_idx])
            suplemented_markdown = (
                #supplementary_text[msg_idx]['heading'] + "\n\n" + 
                cleaned_markdown + "\n\n" + 
                supplementary_text[msg_idx]['caption']
            )
            chunks[paper_idx].append(suplemented_markdown)

        self.chunks = chunks
        return chunks


    def read(self) -> list[list[str]]:
        """
        Load text or markdown documents based on provided metadata.

        Args:

        Returns:
            chunks (list[list[str]]): A 2d list of text chunks derived from the documents. 
                Each sublist corresponds to a single document, and items within it correspond 
                to small, paragraph sized pieces of text.
        """
        chunks = []
        for filepath in self.filepaths:
            with open(filepath, 'r', encoding='utf-8') as file:
                doc_chunks = json.load(file)
                chunks.append(doc_chunks)
        self.chunks = chunks
        return chunks
    

    def fit(self, filepaths : list[str]):
        """
        Load documents based on provided metadata.

        Args:
            document_info (list[dict[str, dict]]): A list to store metadata about each document. 
                Each item in the list is a dictionary with a single key-value pair, where the key is
                the document's filename and the value is another dictionary containing key-value
                metadata, e.g. {'author': 'et al', 'title': 'xyz', 'date': '2028'}.
        """
        self.filepaths = filepaths
        if self.ocr:
            self.chunk()
            chunks = self.ocr_read()
        else:
            chunks = self.read()
        return chunks


    def save_chunks(self, folderpath: str):
        """
        Save text chunks to the specified folder.

        Args:
            folderpath (str): The path to the folder where the markdown documents will be saved.
        """
        for i, doc_chunks in enumerate(self.chunks):
            filename = self.filepaths[i]
            base_filename = os.path.basename(filename).replace('.pdf', '.json')
            save_path = os.path.join(folderpath, base_filename)
            with open(save_path, 'w', encoding='utf-8') as file:
                json.dump(doc_chunks, file)


    def save_images(self, folderpath: str):
        """
        Save document images to the specified folder.

        Args:
            folderpath (str): The path to the folder where the images will be saved.
        """
        for i, doc in enumerate(self.docling_documents):
            filename = self.filepaths[i]
            base_filename = os.path.basename(filename).replace('.pdf', '')
            doc_folderpath = os.path.join(folderpath, base_filename)
            os.makedirs(doc_folderpath, exist_ok=True)
            for j, chunk in enumerate(self.docling_chunks[i]):
                #item = chunk.meta.doc_items[0]
                item = chunk
                img = item.get_image(doc = doc)
                if img.mode == "RGBA":
                    img = img.convert("RGB")
                image_save_path = os.path.join(doc_folderpath, f'chunk_{j}.png')
                img.save(image_save_path)

            '''
            for j, table in enumerate(doc.tables):
                img = table.get_image(doc = doc)
                if img.mode == "RGBA":
                    img = img.convert("RGB")
                image_save_path = os.path.join(doc_folderpath, f'table_{j}.png')
                img.save(image_save_path)
            '''
