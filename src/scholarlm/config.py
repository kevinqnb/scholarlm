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
        paper_exclude: Optional list of paper codes to unconditionally exclude from
            processing (e.g. papers whose data comes from figures or supplemental
            text).  Applied after ``paper_filter`` and before ``paper_subset``.
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
        direct_extraction_schema: Optional Pydantic ``BaseModel`` subclass used by
            Ablation 1 (direct triple extraction).  Must combine all entity fields,
            all measurement event fields, and the ``attribute``, ``value``, and
            ``units`` fields into a single flat schema.  ``None`` disables ablation 1.
        direct_extraction_prompt: Dataset-specific prompt for Ablation 1 that
            describes entities, measurement events, and attributes in a single
            combined block.  Required when ``direct_extraction_schema`` is set;
            ignored otherwise.
        ablation2_entity_schema: Optional Pydantic ``BaseModel`` subclass used by
            Ablation 2 (combined entity-attribute extraction).  Must include all
            normal entity fields plus two reserved fields: ``attribute (str)`` (exact
            attribute name from ``attribute_info_dict``) and ``attribute_terms
            (list[str])`` (terminology used in the document).  ``None`` disables
            ablation 2.
        ablation2_entity_identification_prompt: Dataset-specific prompt for Ablation
            2 that instructs the model to emit one item per (entity, attribute) pair
            rather than one item per entity.  Required when
            ``ablation2_entity_schema`` is set; ignored otherwise.
        judge_filter_fields: Optional list of field names to exclude from the
            judge prompt's entity and event descriptions.  Applied as a blocklist
            across both sections, so a single entry removes a field regardless of
            whether it belongs to the entity or event schema.  Use this to suppress
            noisy or irrelevant fields (e.g. ``location``) without altering the
            underlying schemas.  ``None`` applies no filtering.
        nuextract_examples: Optional list of small, hand-written few-shot examples
            for the NuExtract-2.0-8B baseline (``MeasurementLMNuExtract``), each a
            dict ``{"input": str, "output": str}`` where ``output`` is a JSON string
            matching ``direct_extraction_schema`` wrapped as ``{"items": [...]}``.
            NuExtract's calling convention has no field for freeform instructions
            (only a JSON template + optional examples), so these are the only way
            to give it dataset-specific attribute semantics; every value in
            ``output`` should be an exact substring of ``input`` (NuExtract's
            ``verbatim-string`` fields are trained to copy spans, not paraphrase).
            Synthetic text, not real paper excerpts — must never overlap with
            ``ground_truth_file`` papers.  Ignored by every other pipeline path.
        chatextract_property_names: Optional mapping from each ``attribute_info_dict``
            key to a short, human-readable *property phrase* for the ChatExtract
            baseline (``MeasurementLMChatExtract``).  ChatExtract is a single-property
            method whose prompts read "...a value of ``<PROPERTY>``..."; this supplies
            the ``<PROPERTY>`` string per attribute (e.g. ``"tn" -> "total nitrogen"``).
            Any attribute absent from this dict falls back to the key with underscores
            replaced by spaces.  Ignored by every other pipeline path.
        gliner_property_names: Optional mapping from each ``attribute_info_dict`` key
            to a short, human-readable *property phrase* for the GLiNER2 baseline
            (``MeasurementLMGliner``).  GLiNER runs one structured-extraction schema
            per attribute; this phrase names the measurement in that schema's fields
            (the richer per-attribute semantics come from
            ``attribute_info_dict[key]["description"]``, which GLiNER also consumes).
            ``None`` (or an absent key) falls back to ``chatextract_property_names``
            and then to the attribute key with underscores replaced by spaces, so
            populating this is optional.  Ignored by every other pipeline path.
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
    paper_exclude: list[str] | None = None
    measurement_event_schema: type[BaseModel] | None = None
    measurement_event_prompt: str | None = None
    direct_extraction_schema: type[BaseModel] | None = None
    direct_extraction_prompt: str | None = None
    ablation2_entity_schema: type[BaseModel] | None = None
    ablation2_entity_identification_prompt: str | None = None
    ground_truth_file: str | None = None
    unit_conversion_table: dict[str, dict[str, float]] = field(default_factory=dict)
    judge_filter_fields: list[str] | None = None
    judge_instructions: str | None = None
    nuextract_examples: list[dict] | None = None
    chatextract_property_names: dict[str, str] | None = None
    gliner_property_names: dict[str, str] | None = None


@dataclass
class ModelConfig:
    """
    Configuration for an extraction model.

    Attributes:
        name: Short identifier used in output paths and CLI arguments
            (e.g. ``"qwen-2.5-72b"``).
        model_id: HuggingFace model ID (vLLM) or API model name (frontier).
            Also used as the ``model`` field in API requests.
        hf_revision: HuggingFace commit SHA pinned for reproducibility.
            ``None`` means the default branch HEAD was used (less reproducible).
        sampling_params: Generation parameters forwarded to the API.
            Supported keys: ``temperature``, ``top_p``, ``top_k``,
            ``max_tokens``, ``repetition_penalty``, ``seed``.
        api_base: API base URL for frontier models (e.g.
            ``"https://api.openai.com/v1"``).  When ``None``, the model is
            assumed to be a vLLM instance and runners use their ``--api-base``
            CLI argument instead.
    """

    name: str
    model_id: str
    hf_revision: str | None = None
    sampling_params: dict = field(
        default_factory=lambda: {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 64,
            "max_tokens": 8192,
        }
    )
    api_base: str | None = None
