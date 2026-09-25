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
        collect_attribute_terms: Whether the attribute-detection step should request and
            forward per-attribute terminology ("terms"). Set to ``False`` for datasets
            whose ``attribute_info_dict`` collapses to a single abstract bucket (e.g.
            measeval's ``"measurement"``), where there is no real terminology to ask
            for and models tend to dump unrelated numeric values instead. See
            ``MeasurementLM.__init__``'s docstring for details. Defaults to ``True``.
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
        direct_extraction_schema_no_qualifiers: Optional variant of
            ``direct_extraction_schema`` with the qualifier/shape fields
            (``qualifiers``, ``point_value``, ``lower``, ``upper``,
            ``list_values``, ``tolerance``, ``standard_deviation``) removed.
            Used by Ablation 1 only when an experiment config explicitly sets
            ``params.include_qualifiers: false`` (see ``run_ablation.py``);
            ``None`` means this dataset has no such variant defined, which
            ``run_ablation.py`` fails loud on rather than falling back to the
            qualifier-bearing schema.
        direct_extraction_prompt_no_qualifiers: The ``direct_extraction_prompt``
            counterpart to ``direct_extraction_schema_no_qualifiers`` -- same
            rules.
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
        chatextract_entity_noun: Optional noun naming the kind of thing ChatExtract's
            "material"/"compound" slot refers to for this dataset (e.g. ``"water
            body"``, ``"site"``, ``"subject"``).  ChatExtract's reference prompts are
            materials-science-specific ("What is the *material* for which the
            {property} is given?", "Make sure it is a real *compound*."); this
            replaces every such occurrence with the supplied noun so the prompts
            read naturally for datasets whose entities aren't chemical compounds.
            Falls back to ``"material"`` (collapsing the original "material"/
            "compound" split onto one word) if unset.  Ignored by every other
            pipeline path.
        gliner_entity_description: Optional description of the measurement *subject*
            for the GLiNER2 baseline (``MeasurementLMGliner``), used in place of
            ``entity_type_description`` when building that baseline's subject field
            ("The name or identifier of ``<DESC>`` for which the {property} is
            reported").  Needed only when a dataset's entity is not the measurement
            subject: measeval enumerates *quantities* as entities (see
            ``experiments/dataset-configs/measeval.py``), which would otherwise ask GLiNER
            for "the name or identifier of a numerical quantity".  GLiNER is a flat
            span tagger with no pipeline structure to invert, so it stays
            subject-centric and reads this instead.  Falls back to
            ``entity_type_description`` when unset.  Ignored by every other
            pipeline path.
        gliner_field_descriptions: Optional mapping from an entity- or event-schema
            field name (e.g. ``"location"``, ``"date"``) to a description for that
            field's GLiNER2 structure field, used by the GLiNER2 baseline
            (``MeasurementLMGliner``) only — never fed into ``entity_schema``/
            ``measurement_event_schema`` themselves, so it can't change what the
            real pipeline (or any other baseline) sends for structured decoding.
            GLiNER2's structured extraction supports multiple linked sub-fields
            per structure (unlike ChatExtract's fixed Material/Value/Unit
            triple), so ``MeasurementLMGliner`` asks for every field with an
            entry here, in addition to the subject name/value/units — a field
            with no entry (e.g. ``identifiers``, an alias-resolution aid for the
            real pipeline's entity matching, not reported content) is never
            asked for. Values should be copied verbatim from this dataset's own
            prompts (typically ``direct_extraction_prompt``'s per-field bullets)
            rather than freshly authored, so GLiNER sees the same wording the
            real pipeline already uses. Ignored by every other pipeline path.
        baseline_filter_fields: Optional list of field names to omit from the
            NuExtract-2.0-8B and NuExtract3 baselines' decoding schema, prompt
            template, and few-shot examples (``MeasurementLMNuExtract`` /
            ``MeasurementLMNuExtract3``) — never fed into ``direct_extraction_schema``
            itself, so it can't change what Ablation 1 (which shares that same
            schema) asks for. Use this for a field that the real pipeline and its
            ablations should keep extracting but that a baseline method has no
            business reproducing (e.g. ``identifiers``, an alias-resolution aid
            for the real pipeline's entity matching — GLiNER already excludes it
            structurally via ``gliner_field_descriptions``, and ChatExtract's flat
            schema never included it, so this is currently only load-bearing for
            the two NuExtract baselines). ``None`` applies no filtering.
        strict_matching: Optional column mapping (ground-truth column name ->
            extraction column name) for exact-match comparison, passed to
            ``scholarlm.utils.data.match_datasets`` by
            ``analysis/match_cache.py`` and ``analysis/recovery_validity.py``
            (via ``analysis.match_cache.get_matching_config``) — the single
            centralized source of matching rules for that id-addressed
            evaluation path. Not read by the legacy, pre-id-addressing
            ``analysis/ablation.py``/``analysis/baselines.py`` (their own
            ``get_matching_rules`` predates this field and scores a different
            column shape — ``converted_value`` rather than ``point_value`` —
            against an earlier extraction/judge era; the two are allowed to
            diverge, see ``analysis/match_cache.py``'s module docstring).
            Required (raises) if unset when a dataset is used through
            ``match_cache.py``/``recovery_validity.py``.
        fuzzy_matching: Optional column mapping (ground-truth -> extraction)
            for fuzzy-score comparison, same consumer as ``strict_matching``.
            Required (raises) if unset when a dataset is used through
            ``match_cache.py``/``recovery_validity.py``; ``fuzzy_threshold``
            must also be set whenever this is.
        fuzzy_threshold: Optional selected/default operating threshold for
            ``fuzzy_matching`` scores, same consumer as ``strict_matching``.
            Applied on top of a match cache built at ``fuzzy_threshold=0.0``
            (every cache is built at 0.0 regardless of this value — see
            ``analysis/match_cache.py``'s module docstring) via
            ``analysis.match_cache.edges_above_threshold``/
            ``load_match_cache(..., fuzzy_threshold=...)``, never passed to
            the cache-building call itself. Required (raises) if unset
            whenever ``fuzzy_matching`` is.
        numeric_coerce: Optional subset of ``strict_matching``'s keys
            (ground-truth column names) that must be coerced to float on both
            sides before strict matching — see
            ``analysis/match_cache.py``'s ``_parse_numeric``/module docstring
            for why (a strict-match column can be numeric in the ground truth
            but a raw string in extraction output). ``None``/``[]`` applies no
            coercion.
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
    direct_extraction_schema_no_qualifiers: type[BaseModel] | None = None
    direct_extraction_prompt_no_qualifiers: str | None = None
    ablation2_entity_schema: type[BaseModel] | None = None
    ablation2_entity_identification_prompt: str | None = None
    ground_truth_file: str | None = None
    unit_conversion_table: dict[str, dict[str, float]] = field(default_factory=dict)
    collect_attribute_terms: bool = True
    judge_filter_fields: list[str] | None = None
    judge_instructions: str | None = None
    nuextract_examples: list[dict] | None = None
    chatextract_property_names: dict[str, str] | None = None
    chatextract_entity_noun: str | None = None
    gliner_property_names: dict[str, str] | None = None
    gliner_entity_description: str | None = None
    gliner_field_descriptions: dict[str, str] | None = None
    baseline_filter_fields: list[str] | None = None
    strict_matching: dict[str, str] | None = None
    fuzzy_matching: dict[str, str] | None = None
    fuzzy_threshold: float | None = None
    numeric_coerce: list[str] | None = None


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
            ``max_tokens``, ``repetition_penalty``, ``seed``, ``enable_thinking``
            (Qwen3-style chat-template kwarg), ``reasoning_effort`` (gpt-oss's
            harmony chat-template kwarg: low/medium/high).
        api_base: API base URL for frontier models (e.g.
            ``"https://api.openai.com/v1"``).  When ``None``, the model is
            assumed to be a vLLM instance and runners use their ``--api-base``
            CLI argument instead.
        device: Torch device to load weights onto directly (e.g. ``"cuda"``),
            for models a runner loads in-process rather than through a served
            API/vLLM endpoint (currently only the GLiNER baseline). ``None``
            for every other kind, which never reads this field. A runner that
            does load weights directly must fail loud if this is ``None``
            rather than falling back to CPU -- see measurementlm_gliner.py.
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
    device: str | None = None
