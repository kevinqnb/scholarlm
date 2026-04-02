"""
Dataset and model configuration dataclasses for the ScholarlM experiment framework.

These classes are the single source of truth for everything that varies between
datasets (entity schema, attributes, prompts, paths) and between extraction models
(HuggingFace ID, GPU count, generation parameters).  All downstream pipeline code
(run_extraction, run_judge, run_analysis) accepts these objects instead of
hard-coding dataset- or model-specific details.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel


@dataclass
class DatasetConfig:
    """
    All dataset-specific configuration needed to run the extraction and judge pipelines.

    Attributes:
        name: Short identifier used in output paths and CLI arguments (e.g. ``"pond"``).
        data_dir: Root directory for the dataset (e.g. ``"data/pond"``).  Raw OCR
            output lives at ``{data_dir}/ocr_output_raw/`` and PDFs at
            ``{data_dir}/pdfs/`` by convention.
        metadata_file: Path to the paper directory JSON that maps
            ``paper_code -> {title, author, year, ...}``.
        entity_schema: Pydantic ``BaseModel`` subclass whose fields define the
            entity representation for this dataset (e.g. name, location, date …).
        entity_identification_prompt: System prompt for the entity identification step
            passed to ``MeasurementLM``.
        entity_type_description: One-sentence description of what an entity *is*,
            used verbatim inside judge prompts (e.g. "A distinct aquatic ecosystem
            observation…").
        attribute_info_dict: Mapping ``attribute_name -> {"description": str, "units": list[str]}``
            passed to ``MeasurementLM`` and used to look up descriptions in judge prompts.
        paper_subset: Optional explicit list of paper codes (filename stems without
            ``.txt``) to process.  ``None`` processes all available papers that
            pass ``paper_filter``.
        paper_filter: Optional predicate ``(paper_metadata: dict) -> bool`` applied
            to each paper's metadata dict.  Only papers for which this returns
            ``True`` are included.  Applied *before* ``paper_subset`` intersection.
    """

    name: str
    data_dir: str
    metadata_file: str
    entity_schema: type[BaseModel]
    entity_identification_prompt: str
    entity_type_description: str
    attribute_info_dict: dict[str, dict]
    paper_subset: list[str] | None = None
    paper_filter: Callable[[dict], bool] | None = None


@dataclass
class ModelConfig:
    """
    Configuration for an extraction model instantiated via ``MeasurementLM``.

    Attributes:
        name: Short identifier used in output paths and CLI arguments
            (e.g. ``"qwen-2.5-72b"``).
        model_id: HuggingFace model ID or local path passed to vLLM.
        tensor_parallel_size: Number of GPUs to use for tensor parallelism.
            Passed directly to ``vllm.LLM(tensor_parallel_size=...)``.
        sampling_params: Generation parameters forwarded to ``vllm.SamplingParams``.
            Keys and defaults follow the vLLM ``SamplingParams`` signature.
    """

    name: str
    model_id: str
    tensor_parallel_size: int = 1
    sampling_params: dict = field(
        default_factory=lambda: {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 64,
            "max_tokens": 8192,
            "seed": 342,
        }
    )
