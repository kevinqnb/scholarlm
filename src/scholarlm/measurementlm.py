import json
from rapidfuzz import fuzz
from itertools import combinations
from pydantic import BaseModel
import numpy as np
import pandas as pd
import math
import re
from io import StringIO
import torch
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from .instruction_prompts import (
    ENTITY_TABLE_ENRICHMENT_INSTRUCTIONS,
    DETECT_ATTRIBUTES_BATCH_INSTRUCTIONS,
    DETECT_ATTRIBUTES_TABLE_BATCH_INSTRUCTIONS,
    FILTER_ENTITY_ATTRIBUTE_INSTRUCTIONS,
    EXTRACT_TEXT_VALUE_INSTRUCTIONS,
    EXTRACT_TABLE_VALUE_INSTRUCTIONS,
    STANDARDIZE_MEASUREMENTS_INSTRUCTIONS,
)


def response_validator(response_structure, response):
    pyd = response_structure.model_validate_json(response)
    out_dict = pyd.model_dump()
    return out_dict


class ListResponse(BaseModel):
    items: list[str]


class BooleanDecisionResponse(BaseModel):
    """Structured response that encourages reasoning before a boolean decision."""
    explanation: str
    answer: bool


class TextValueExtractionResponse(BaseModel):
    """Response for extracting a value from prose text."""
    explanation: str
    has_value: bool
    value: str | None = None
    units: str | None = None


class TableValueExtractionResponse(BaseModel):
    """Response for extracting a value from a table."""
    explanation: str
    has_value: bool
    row_index: str | None = None
    column_index: str | None = None
    units: str | None = None


class AttributeDetectionItem(BaseModel):
    """Detection result for a single attribute."""
    attribute_name: str
    explanation: str
    detected: bool
    terms: list[str]


class BatchAttributeDetectionResponse(BaseModel):
    """Batched detection results for all attributes."""
    items: list[AttributeDetectionItem]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            if math.isnan(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class MeasurementLM:
    """
    A language model class designed for organized collection of measurements from scientific text.

    Args:
        model_name (str): The name or path of the pre-trained language model from the huggingface
            collection.
        entity_identification_prompt (str): The prompt template for entity identification.
        entity_identification_schema (BaseModel): The pydantic schema for entity identification.
        attribute_info_dict (dict[str, any]): A dictionary containing information about the
            attributes to be measured. Each key is an attribute name, and each value is a dict
            with at least a 'description' key and optionally a 'units' key.
        sampling_params (dict[str, any]): A dictionary of sampling parameters for text generation.
    """
    def __init__(
        self,
        model_name: str,
        entity_identification_prompt: str,
        entity_identification_schema: BaseModel,
        attribute_info_dict: dict[str, any],
        sampling_params: dict[str, any] = {},
    ):
        self.model_name = model_name
        self.sampling_params = {
            "temperature" : 0.90,
            "top_p" : 0.95,
            "top_k" : 64,
            "repetition_penalty" : 1.0,
            "max_tokens" : 2048,
        } | sampling_params
        self.entity_identification_prompt = entity_identification_prompt
        self.entity_identification_schema = entity_identification_schema
        self.attribute_info_dict = attribute_info_dict
        self.llm = LLM(model=model_name)


    # -----------------------------------------------------------------------
    # Step 1: Entity extraction (full context + table enrichment)
    # -----------------------------------------------------------------------

    def _extract_entities(self):
        """
        Extracts entities from documents in two passes:
        1. Full-context extraction using the entity identification prompt and schema.
        2. Per-table enrichment that finds new entities or fills in missing fields
           on existing entities.

        Reads from self.data (one record per document) and returns one record per
        (document, entity) with entity schema fields merged in.
        """
        from pydantic import create_model

        IdentificationList = create_model(
            "IdentificationList",
            items=(list[self.entity_identification_schema], ...),
        )
        identification_list_json = IdentificationList.model_json_schema()
        entity_fields = list(self.entity_identification_schema.model_fields.keys())

        # --- Pass 1: Full-context entity identification ---
        messages = []
        for i, datapoint in enumerate(self.data):
            instructions = self.entity_identification_prompt
            context = datapoint['context']
            query = "Follow the instructions to identify the items mentioned in the context."
            prompt = (
                f"## Instructions:\n{instructions}\n\n## Context:\n{context}\n\n## Query:\n{query}"
            )
            messages.append([{"role": "user", "content": prompt}])

        guided_decoding_params = GuidedDecodingParams(
            json=identification_list_json
        )
        sampling_params = SamplingParams(
            **self.sampling_params,
            guided_decoding=guided_decoding_params
        )

        responses = self.llm.chat(messages=messages, sampling_params=sampling_params)
        response_texts = [r.outputs[0].text for r in responses]

        # Build per-document entity lists
        doc_entities: dict[int, list[dict]] = {}
        for i, r in enumerate(response_texts):
            try:
                resp_validated = response_validator(IdentificationList, r)
            except:
                print("Validation error in identification response.")
                resp_validated = {'items': []}

            doc_entities[i] = list(resp_validated['items'])

        # --- Pass 2: Table enrichment ---
        enrichment_messages = []
        enrichment_ids = []  # (doc_idx, table_number)
        for i, datapoint in enumerate(self.data):
            context = datapoint['context']
            tables = re.findall(r'<table number="(\d+)">', context)
            if not tables:
                continue

            existing_entities_summary = json.dumps(
                doc_entities.get(i, []), indent=2, ensure_ascii=False
            )

            for t in tables:
                t = int(t)
                table_start = context.find(f'<table number="{t}">') + len(f'<table number="{t}">')
                table_end = context.find('</table>', table_start)
                table_text = context[table_start:table_end].strip()
                if not table_text:
                    continue

                query = (
                    "Examine this table. Identify any new entities not in the list above, "
                    "and for existing entities extract any additional attribute values "
                    "(abbreviations, codes, etc.) that the table reveals."
                )
                prompt = (
                    f"## Instructions:\n{ENTITY_TABLE_ENRICHMENT_INSTRUCTIONS}\n\n"
                    f"## Already Identified Entities:\n{existing_entities_summary}\n\n"
                    f"## Table:\n{table_text}\n\n"
                    f"## Query:\n{query}"
                )
                enrichment_messages.append([{"role": "user", "content": prompt}])
                enrichment_ids.append((i, t))

        if enrichment_messages:
            enrichment_responses = self.llm.chat(
                messages=enrichment_messages, sampling_params=sampling_params
            )
            enrichment_texts = [r.outputs[0].text for r in enrichment_responses]

            for msg_idx, resp_text in enumerate(enrichment_texts):
                doc_idx, table_number = enrichment_ids[msg_idx]
                try:
                    resp_validated = response_validator(IdentificationList, resp_text)
                except:
                    print(f"Validation error in table enrichment response (doc {doc_idx}, table {table_number}).")
                    continue

                existing = doc_entities.get(doc_idx, [])
                for new_entity in resp_validated['items']:
                    # Try to match against existing entities by name
                    new_name = str(new_entity.get('name', '') or '').strip().lower()
                    best_match_score = 0
                    best_match_idx = -1
                    for eidx, existing_entity in enumerate(existing):
                        existing_name = str(existing_entity.get('name', '') or '').strip().lower()
                        if new_name and existing_name:
                            score = fuzz.token_set_ratio(new_name, existing_name)
                            if score > best_match_score:
                                best_match_score = score
                                best_match_idx = eidx

                    if best_match_score >= 80 and best_match_idx >= 0:
                        # Enrich: fill in None fields on the existing entity
                        for field in entity_fields:
                            if existing[best_match_idx].get(field) is None and new_entity.get(field) is not None:
                                existing[best_match_idx][field] = new_entity[field]
                    else:
                        # New entity
                        existing.append(new_entity)

                doc_entities[doc_idx] = existing

        # --- Build output: one record per (document, entity) ---
        entity_data = []
        for i, datapoint in enumerate(self.data):
            for entity in doc_entities.get(i, []):
                entity_data.append(datapoint | entity)

        return entity_data


    # -----------------------------------------------------------------------
    # Step 2: Document-level attribute detection (batched full context + per-table)
    # -----------------------------------------------------------------------

    def _format_attribute_list(self, attr_names=None):
        """Format attributes as a numbered list for inclusion in prompts.

        Args:
            attr_names: Subset of attribute names to include. If None, includes all.
        """
        if attr_names is None:
            attr_names = list(self.attribute_info_dict.keys())
        lines = []
        for idx, attr_name in enumerate(attr_names, 1):
            desc = self.attribute_info_dict[attr_name].get('description', '')
            lines.append(f"{idx}. {attr_name}: {desc}")
        return "\n".join(lines)

    def _detect_attributes(self):
        """
        Document-level attribute detection in two phases:
        A. Batched full-context detection — one prompt per document evaluating
           all attributes at once, with inline term identification.
        B. Batched per-table fallback — one prompt per (document, table) for
           attributes not yet detected in Phase A.

        Reads from self.data (one record per document) and returns a dict
        mapping document index to detected attributes with their terms:
            {doc_idx: {attr_name: [term, ...], ...}, ...}
        """
        attr_names = list(self.attribute_info_dict.keys())
        attribute_list_text = self._format_attribute_list()

        guided_decoding_params = GuidedDecodingParams(
            json=BatchAttributeDetectionResponse.model_json_schema()
        )
        sampling_params = SamplingParams(
            **self.sampling_params,
            guided_decoding=guided_decoding_params
        )

        # --- Phase A: Batched full-context detection (one prompt per document) ---
        messages = []
        message_ids = []  # doc_idx
        for i, datapoint in enumerate(self.data):
            context = datapoint['context']
            query = (
                f"Attributes to evaluate:\n{attribute_list_text}\n\n"
                f"For each attribute listed above, determine whether the document "
                f"contains any direct numerical measurements for that attribute. "
                f"Return one item per attribute using the exact attribute name.\n\n"
            )
            prompt = (
                f"## Instructions:\n{DETECT_ATTRIBUTES_BATCH_INSTRUCTIONS}\n\n"
                f"## Context:\n{context}\n\n## Query:\n{query}"
            )
            messages.append([{"role": "user", "content": prompt}])
            message_ids.append(i)

        responses = self.llm.chat(messages=messages, sampling_params=sampling_params)
        response_texts = [r.outputs[0].text for r in responses]

        # Build detection results and attribute terms per document
        detection_results: dict[int, dict[str, bool]] = {}
        attribute_terms: dict[int, dict[str, list[str]]] = {}

        for msg_idx, resp in enumerate(response_texts):
            doc_idx = message_ids[msg_idx]
            try:
                batch = response_validator(BatchAttributeDetectionResponse, resp)
            except:
                print("Validation error in batched attribute detection response.")
                detection_results[doc_idx] = {a: False for a in attr_names}
                continue

            responded_attrs = {}
            for item in batch['items']:
                responded_attrs[item['attribute_name']] = item

            detection_results[doc_idx] = {}
            attribute_terms[doc_idx] = {}
            for attr_name in attr_names:
                item = responded_attrs.get(attr_name)
                if item and item.get('detected', False):
                    detection_results[doc_idx][attr_name] = True
                    attribute_terms[doc_idx][attr_name] = item.get('terms', [])
                else:
                    detection_results[doc_idx][attr_name] = False

        # --- Phase B: Batched per-table fallback for undetected attributes ---
        table_messages = []
        table_message_ids = []  # (doc_idx, list_of_attr_names_queried)
        for i, datapoint in enumerate(self.data):
            undetected = [
                a for a in attr_names
                if not detection_results.get(i, {}).get(a, False)
            ]
            if not undetected:
                continue

            context = datapoint['context']
            tables = re.findall(r'<table number="(\d+)">', context)
            if not tables:
                continue

            undetected_list_text = self._format_attribute_list(undetected)

            for t in tables:
                t = int(t)
                table_start = context.find(f'<table number="{t}">') + len(f'<table number="{t}">')
                table_end = context.find('</table>', table_start)
                table_text = context[table_start:table_end].strip()
                if not table_text:
                    continue

                query = (
                    f"Attributes to evaluate:\n{undetected_list_text}\n\n"
                    f"For each attribute listed above, determine whether the table "
                    f"contains any direct numerical measurements for that attribute. "
                    f"Return one item per attribute using the exact attribute name.\n\n"
                )
                prompt = (
                    f"## Instructions:\n{DETECT_ATTRIBUTES_TABLE_BATCH_INSTRUCTIONS}\n\n"
                    f"## Context:\n{table_text}\n\n## Query:\n{query}"
                )
                table_messages.append([{"role": "user", "content": prompt}])
                table_message_ids.append((i, undetected))

        if table_messages:
            table_responses = self.llm.chat(
                messages=table_messages, sampling_params=sampling_params
            )
            table_response_texts = [r.outputs[0].text for r in table_responses]

            for msg_idx, resp in enumerate(table_response_texts):
                doc_idx, queried_attrs = table_message_ids[msg_idx]
                try:
                    batch = response_validator(BatchAttributeDetectionResponse, resp)
                except:
                    print("Validation error in batched table attribute detection response.")
                    continue

                responded_attrs = {}
                for item in batch['items']:
                    responded_attrs[item['attribute_name']] = item

                for attr_name in queried_attrs:
                    item = responded_attrs.get(attr_name)
                    if item and item.get('detected', False):
                        detection_results[doc_idx][attr_name] = True
                        if attr_name not in attribute_terms.get(doc_idx, {}):
                            attribute_terms.setdefault(doc_idx, {})[attr_name] = item.get('terms', [])

        # --- Build output: {doc_idx: {attr_name: terms}} for detected attrs ---
        doc_attributes: dict[int, dict[str, list[str]]] = {}
        for doc_idx in range(len(self.data)):
            detected = {}
            for attr_name in attr_names:
                if detection_results.get(doc_idx, {}).get(attr_name, False):
                    detected[attr_name] = attribute_terms.get(doc_idx, {}).get(attr_name, [])
            if detected:
                doc_attributes[doc_idx] = detected

        return doc_attributes


    # -----------------------------------------------------------------------
    # Step 2b: Filter (entity, attribute) pairs
    # -----------------------------------------------------------------------

    def _filter_entity_attribute_pairs(self):
        """
        Filters (entity, attribute) pairs by querying once per pair against the
        full document context with a boolean response.

        Reads from self.data (one record per (entity, attribute) pair) and
        returns the subset of records where the model confirms a measurement
        exists for that specific entity-attribute combination.
        """
        entity_fields = list(self.entity_identification_schema.model_fields.keys())

        messages = []
        message_ids = []  # record_idx
        for i, datapoint in enumerate(self.data):
            context = datapoint['context']
            attribute = datapoint.get('attribute')
            attr_description = self.attribute_info_dict[attribute]['description']
            entity_description = {
                k: v for k, v in datapoint.items() if k in entity_fields
            }
            query = (
                f"Attribute description: {attr_description}\n"
                f"Entity description: {entity_description}\n\n"
                f"Does the context provide a direct numerical measurement "
                f"for the described attribute in reference to the given entity?\n\n"
            )
            prompt = (
                f"## Instructions:\n{FILTER_ENTITY_ATTRIBUTE_INSTRUCTIONS}\n\n"
                f"## Context:\n{context}\n\n## Query:\n{query}"
            )
            messages.append([{"role": "user", "content": prompt}])
            message_ids.append(i)

        if not messages:
            return []

        guided_decoding_params = GuidedDecodingParams(
            json=BooleanDecisionResponse.model_json_schema()
        )
        sampling_params = SamplingParams(
            **self.sampling_params,
            guided_decoding=guided_decoding_params
        )
        responses = self.llm.chat(messages=messages, sampling_params=sampling_params)
        response_texts = [r.outputs[0].text for r in responses]

        filtered_data = []
        for msg_idx, resp in enumerate(response_texts):
            record_idx = message_ids[msg_idx]
            try:
                decision = response_validator(BooleanDecisionResponse, resp)
                answer = bool(decision.get('answer', False))
            except:
                print("Validation error in entity-attribute filter response.")
                answer = False

            if answer:
                filtered_data.append(self.data[record_idx])

        return filtered_data


    # -----------------------------------------------------------------------
    # Step 3: Extract values from text (per-page)
    # -----------------------------------------------------------------------

    def _extract_values_from_text(self):
        """
        Extracts measurement values from prose text on a per-page basis.

        For each (entity, attribute) pair in self.data, splits the document context
        into pages and asks whether a value is present. If yes, extracts the value
        and units.

        Returns records with 'value', 'units', 'page_number', and 'source'='text'.
        """
        entity_fields = list(self.entity_identification_schema.model_fields.keys())
        messages = []
        message_ids = []  # (record_idx, page_number)

        for i, datapoint in enumerate(self.data):
            context = datapoint['context']
            attribute = datapoint.get('attribute')
            attr_description = self.attribute_info_dict[attribute]['description']
            attr_terms = datapoint.get('attribute_terms', [])
            unit_options = self.attribute_info_dict[attribute].get('units', [])
            entity_description = {k: v for k, v in datapoint.items() if k in entity_fields}

            pages = re.findall(r'<page number="(\d+)">', context)
            for p in pages:
                p = int(p)
                page_start = context.find(f'<page number="{p}">') + len(f'<page number="{p}">')
                page_end = context.find('</page>', page_start)
                page_text = context[page_start:page_end].strip()
                if not page_text:
                    continue

                units_guidance = ""
                if unit_options:
                    units_guidance = (
                        f"Preferred unit options: {unit_options}. "
                        f"Strongly prioritize choosing the best option from this list. "
                        f"If none of the options fit, specify the unit exactly as it appears in the text.\n"
                    )

                query = (
                    f"Attribute description: {attr_description}\n"
                    f"Terminology used for the attribute: {attr_terms}\n"
                    f"Entity description: {entity_description}\n\n"
                    f"{units_guidance}"
                    f"Does this page contain a measured value for the given attribute and entity? "
                    f"If yes, extract the value and its units.\n\n"
                )
                prompt = (
                    f"## Instructions:\n{EXTRACT_TEXT_VALUE_INSTRUCTIONS}\n\n"
                    f"## Context:\n{page_text}\n\n## Query:\n{query}"
                )
                messages.append([{"role": "user", "content": prompt}])
                message_ids.append((i, p))

        if not messages:
            return []

        guided_decoding_params = GuidedDecodingParams(
            json=TextValueExtractionResponse.model_json_schema()
        )
        sampling_params = SamplingParams(
            **self.sampling_params,
            guided_decoding=guided_decoding_params
        )
        responses = self.llm.chat(messages=messages, sampling_params=sampling_params)
        response_texts = [r.outputs[0].text for r in responses]

        text_values = []
        for msg_idx, resp in enumerate(response_texts):
            record_idx, page_number = message_ids[msg_idx]
            try:
                result = response_validator(TextValueExtractionResponse, resp)
            except:
                print("Validation error in text value extraction response.")
                continue

            if result.get('has_value') and result.get('value') is not None:
                datapoint = self.data[record_idx]
                # Narrow context to the page text
                context = datapoint['context']
                page_start = context.find(f'<page number="{page_number}">') + len(f'<page number="{page_number}">')
                page_end = context.find('</page>', page_start)
                page_text = context[page_start:page_end].strip()

                text_values.append(
                    datapoint | {
                        'context': page_text,
                        'value': result['value'],
                        'units': result.get('units'),
                        'page_number': page_number,
                        'source': 'text',
                    }
                )

        return text_values


    # -----------------------------------------------------------------------
    # Step 4: Extract values from tables
    # -----------------------------------------------------------------------

    def _extract_values_from_tables(self):
        """
        Extracts measurement values from HTML tables.

        For each (entity, attribute) pair in self.data, iterates over tables in the
        document context. Provides the table HTML along with row names and column
        names as additional context. If a value is found, extracts the cell using
        the row and column indices.

        Returns records with 'value', 'units', 'table_number', 'row_index',
        'column_index', and 'source'='table'.
        """
        entity_fields = list(self.entity_identification_schema.model_fields.keys())
        messages = []
        message_ids = []  # (record_idx, table_number)
        table_cache = {}  # (record_idx, table_number) -> (table_text, row_names, column_names)

        for i, datapoint in enumerate(self.data):
            context = datapoint['context']
            attribute = datapoint.get('attribute')
            attr_description = self.attribute_info_dict[attribute]['description']
            attr_terms = datapoint.get('attribute_terms', [])
            unit_options = self.attribute_info_dict[attribute].get('units', [])
            entity_description = {k: v for k, v in datapoint.items() if k in entity_fields}

            tables = re.findall(r'<table number="(\d+)">', context)
            for t in tables:
                t = int(t)
                table_tag_start = context.find(f'<table number="{t}">')
                table_content_start = table_tag_start + len(f'<table number="{t}">')
                table_end = context.find('</table>', table_content_start)
                table_text = context[table_tag_start:table_end + len('</table>')].strip()
                if not table_text:
                    continue

                # Parse table to get row and column names
                try:
                    table_dfs = pd.read_html(StringIO(table_text))
                    table_df = table_dfs[0]
                    row_names = table_df.loc[:, "index"].to_list() if 'index' in table_df.columns else []
                    row_names = [str(name) for name in row_names]
                    column_names = [str(name) for name in table_df.columns.tolist()]
                    column_names = [name for name in column_names if name != 'index']
                except:
                    print(f"Error parsing table {t} in record {i}.")
                    continue

                table_cache[(i, t)] = (table_text, row_names, column_names)

                units_guidance = ""
                if unit_options:
                    units_guidance = (
                        f"Preferred unit options: {unit_options}. "
                        f"Strongly prioritize choosing the best option from this list. "
                        f"If none of the options fit, specify the unit exactly as it appears in the table.\n"
                    )

                query = (
                    f"Attribute description: {attr_description}\n"
                    f"Terminology used for the attribute: {attr_terms}\n"
                    f"Entity description: {entity_description}\n\n"
                    f"Row names in the table: {row_names}\n"
                    f"Column names in the table: {column_names}\n\n"
                    f"{units_guidance}"
                    f"Does this table contain a measured value for the given attribute and entity? "
                    f"If yes, provide the row_index and column_index names, and the units.\n\n"
                )
                prompt = (
                    f"## Instructions:\n{EXTRACT_TABLE_VALUE_INSTRUCTIONS}\n\n"
                    f"## Context:\n{table_text}\n\n## Query:\n{query}"
                )
                messages.append([{"role": "user", "content": prompt}])
                message_ids.append((i, t))

        if not messages:
            return []

        guided_decoding_params = GuidedDecodingParams(
            json=TableValueExtractionResponse.model_json_schema()
        )
        sampling_params = SamplingParams(
            **self.sampling_params,
            guided_decoding=guided_decoding_params
        )
        responses = self.llm.chat(messages=messages, sampling_params=sampling_params)
        response_texts = [r.outputs[0].text for r in responses]

        table_values = []
        for msg_idx, resp in enumerate(response_texts):
            record_idx, table_number = message_ids[msg_idx]
            try:
                result = response_validator(TableValueExtractionResponse, resp)
            except:
                print("Validation error in table value extraction response.")
                continue

            if not result.get('has_value'):
                continue
            row_index = result.get('row_index')
            column_index = result.get('column_index')
            if row_index is None or column_index is None:
                continue

            # Extract cell value from the table using pandas
            table_text, row_names, column_names = table_cache[(record_idx, table_number)]
            try:
                table_dfs = pd.read_html(StringIO(table_text))
                table_df = table_dfs[0]

                matched_rows = table_df.loc[table_df["index"] == row_index][column_index]
                if len(matched_rows) == 0:
                    print("No matching row found in table extraction.")
                    val = None
                elif len(matched_rows) == 1:
                    val = matched_rows.item()
                else:
                    print("Multiple matching rows found in table extraction, taking the first match.")
                    val = matched_rows.iloc[0]
            except:
                print(f"Error extracting value from table {table_number} in record {record_idx}.")
                val = None

            if val is not None:
                datapoint = self.data[record_idx]
                table_values.append(
                    datapoint | {
                        'context': table_text,
                        'value': val,
                        'units': result.get('units'),
                        'table_number': table_number,
                        'row_index': row_index,
                        'column_index': column_index,
                        'source': 'table',
                    }
                )

        return table_values


    # -----------------------------------------------------------------------
    # Step 5: Standardize and deduplicate
    # -----------------------------------------------------------------------

    def _standardize_and_deduplicate(self, similarity_threshold: float = 0.85):
        """
        Standardizes extracted measurement values and then de-duplicates records.

        Standardization: asks the LLM to clean up extracted values (remove uncertainty,
        normalize formatting, etc.).

        De-duplication: records sharing the same (document_id, attribute, value) are
        compared on entity identification attributes using fuzzy matching. Similar
        records are merged.

        Args:
            similarity_threshold (float): Minimum average attribute similarity
                to consider two records duplicates.

        Returns:
            list[dict]: Standardized, de-duplicated measurement records.
        """
        # --- Standardization ---
        entity_fields = list(self.entity_identification_schema.model_fields.keys())
        messages = []
        message_data_ids = []
        for i, datapoint in enumerate(self.data):
            context = datapoint['context']
            attribute = datapoint.get('attribute')
            attr_description = self.attribute_info_dict[attribute]['description']
            attr_terms = datapoint.get('attribute_terms', [])
            entity_description = {k: v for k, v in datapoint.items() if k in entity_fields}
            measurement_val = datapoint['value']

            query = (
                f"Attribute description: {attr_description}\n"
                f"Terminology used for the attribute: {attr_terms}\n"
                f"Entity description: {entity_description}\n"
                f"Extracted measurement: {measurement_val}\n\n"
                f"Standardize the measurement value for the given data point. "
            )
            prompt = (
                f"## Instructions:\n{STANDARDIZE_MEASUREMENTS_INSTRUCTIONS}\n\n"
                f"## Context:\n{context}\n\n## Query:\n{query}"
            )
            messages.append([{"role": "user", "content": prompt}])
            message_data_ids.append(i)

        sampling_params = SamplingParams(**self.sampling_params)
        responses = self.llm.chat(messages=messages, sampling_params=sampling_params)
        response_texts = [r.outputs[0].text for r in responses]

        standardized_data = [datapoint for datapoint in self.data]
        for i, resp in enumerate(response_texts):
            standardized_data[message_data_ids[i]]['value'] = resp.strip()

        # --- De-duplication ---
        self.data = standardized_data

        def _norm(v):
            if v is None:
                return None
            return str(v).strip().lower()

        def _flatten_field(v):
            if v is None:
                return []
            if isinstance(v, list):
                return [s for s in (_norm(x) for x in v) if s is not None]
            normed = _norm(v)
            return [normed] if normed is not None else []

        def _field_similarity(raw_a, raw_b):
            vals_a = _flatten_field(raw_a)
            vals_b = _flatten_field(raw_b)
            if not vals_a and not vals_b:
                return None
            if not vals_a or not vals_b:
                return 1.0
            best = max(
                fuzz.token_set_ratio(a, b) / 100.0
                for a in vals_a
                for b in vals_b
            )
            return best

        def _pair_similarity(a, b):
            scores = []
            for field in entity_fields:
                sim = _field_similarity(a.get(field), b.get(field))
                if sim is None:
                    continue
                scores.append(sim)
            return sum(scores) / len(scores) if scores else 1.0

        def _group_key(record):
            return (
                record.get('document_id'),
                record.get('attribute'),
                _norm(record.get('value')),
                _norm(record.get('units')),
            )

        groups: dict[tuple, list[int]] = {}
        for idx, record in enumerate(self.data):
            key = _group_key(record)
            groups.setdefault(key, []).append(idx)

        deduplicated: list[dict] = []

        for key, indices in groups.items():
            if len(indices) == 1:
                deduplicated.append(self.data[indices[0]])
                continue

            clusters: list[list[int]] = []
            assigned: set[int] = set()
            for i, j in combinations(range(len(indices)), 2):
                if _pair_similarity(self.data[indices[i]], self.data[indices[j]]) >= similarity_threshold:
                    ci = cj = None
                    for k, cl in enumerate(clusters):
                        if i in cl:
                            ci = k
                        if j in cl:
                            cj = k
                    if ci is not None and cj is not None:
                        if ci != cj:
                            clusters[ci].extend(clusters[cj])
                            clusters.pop(cj)
                    elif ci is not None:
                        clusters[ci].append(j)
                    elif cj is not None:
                        clusters[cj].append(i)
                    else:
                        clusters.append([i, j])
                    assigned.update([i, j])

            for i in range(len(indices)):
                if i not in assigned:
                    clusters.append([i])

            for cluster in clusters:
                cluster_records = [self.data[indices[i]] for i in cluster]
                merged = dict(cluster_records[0])

                for field in entity_fields:
                    unique_vals = []
                    seen: set = set()
                    for rec in cluster_records:
                        v = rec.get(field)
                        vals = v if isinstance(v, list) else [v]
                        for item in vals:
                            if item is None:
                                continue
                            normed = _norm(item)
                            if normed not in seen:
                                seen.add(normed)
                                unique_vals.append(item)
                    if len(unique_vals) == 0:
                        merged[field] = None
                    elif len(unique_vals) == 1:
                        merged[field] = unique_vals[0]
                    else:
                        merged[field] = sorted(
                            unique_vals,
                            key=lambda x: str(x) if x is not None else ''
                        )

                # Collect unique contexts and provenance metadata
                for prov_field in ('context', 'source', 'page_number', 'table_number', 'row_index', 'column_index'):
                    unique_vals = []
                    seen_prov: set = set()
                    for rec in cluster_records:
                        v = rec.get(prov_field)
                        if v is None:
                            continue
                        key_v = str(v)
                        if key_v not in seen_prov:
                            seen_prov.add(key_v)
                            unique_vals.append(v)
                    if len(unique_vals) == 0:
                        merged[prov_field] = None
                    elif len(unique_vals) == 1:
                        merged[prov_field] = unique_vals[0]
                    else:
                        merged[prov_field] = unique_vals

                deduplicated.append(merged)

        return deduplicated


    # -----------------------------------------------------------------------
    # Full pipeline
    # -----------------------------------------------------------------------

    def fit(
        self,
        documents: list[str],
    ):
        """
        Runs the full measurement extraction pipeline on the provided documents.

        Args:
            documents (list[str]): A list of text documents.
        Returns:
            measurements (list[dict]): A list of measurements extracted for identified items.
        """
        self.data = []
        for i, doc in enumerate(documents):
            self.data.append({'document_id': i, 'context': doc})
        doc_data = list(self.data)

        # Step 1: Entity extraction (full context + table enrichment)
        entity_data = self._extract_entities()

        # Step 2: Document-level attribute detection (full context + per-table)
        self.data = doc_data
        doc_attributes = self._detect_attributes()

        # Step 3: Cross-product entities × detected attributes per document
        entity_attribute_data = []
        for record in entity_data:
            doc_id = record['document_id']
            for attr_name, terms in doc_attributes.get(doc_id, {}).items():
                entity_attribute_data.append(record | {
                    'attribute': attr_name,
                    'attribute_terms': terms,
                })

        # Step 4: Filter (entity, attribute) pairs
        self.data = entity_attribute_data
        self.data = self._filter_entity_attribute_pairs()

        # Save filtered pairs for parallel extraction
        entity_attribute_data = list(self.data)

        # Step 5: Extract values from text (per-page)
        self.data = entity_attribute_data
        text_values = self._extract_values_from_text()

        # Step 6: Extract values from tables
        self.data = entity_attribute_data
        table_values = self._extract_values_from_tables()

        # Combine text and table extractions
        self.data = text_values + table_values

        # Step 7: Standardize and deduplicate
        self.data = self._standardize_and_deduplicate()

        return self.data


    def save(self, filepath: str):
        """
        Saves the measurement data to a JSON file.

        Args:
            filepath (str): The path to the file where the data will be saved.
        """
        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
