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
        measurement_event_schema: Optional Pydantic ``BaseModel`` subclass whose
            fields define a single measurement event (e.g. date, method, substrate,
            depth).  When set, the pipeline inserts an event-resolution step between
            attribute provenance and value extraction that enumerates the distinct
            measurement events present for each (entity, attribute, page)
            intersection.  ``None`` disables event resolution entirely and the
            pipeline behaves as it did before this feature was added.
        measurement_event_prompt: Dataset-specific instructions for the event
            resolution step, describing what constitutes a distinct measurement
            event and explaining each event field.  Required when
            ``measurement_event_schema`` is set; ignored otherwise.
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
    measurement_event_schema: type[BaseModel] | None = None
    measurement_event_prompt: str | None = None


@dataclass
class ModelConfig:
    """
    Configuration for an extraction model served via a vLLM OpenAI-compatible API.

    Attributes:
        name: Short identifier used in output paths and CLI arguments
            (e.g. ``"qwen-2.5-72b"``).
        model_id: HuggingFace model ID passed to the vLLM server at startup.
            Also used as the ``model`` field in API requests.
        sampling_params: Generation parameters forwarded to the API.
            Supported keys: ``temperature``, ``top_p``, ``top_k``,
            ``max_tokens``, ``repetition_penalty``.  ``seed`` is not forwarded
            (the OpenAI-compatible API does not support it).
    """

    name: str
    model_id: str
    sampling_params: dict = field(
        default_factory=lambda: {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 64,
            "max_tokens": 8192,
        }
    )
