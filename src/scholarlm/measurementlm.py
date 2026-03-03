import json
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
    DETECT_ATTRIBUTES_BATCH_INSTRUCTIONS,
    ENTITY_PROVENANCE_INSTRUCTIONS,
    ATTRIBUTE_PROVENANCE_INSTRUCTIONS,
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


class AttributeDetectionItem(BaseModel):
    """Detection result for a single attribute."""
    attribute_name: str
    explanation: str
    detected: bool
    terms: list[str]


class BatchAttributeDetectionResponse(BaseModel):
    """Batched detection results for all attributes."""
    items: list[AttributeDetectionItem]


class ProvenanceResponse(BaseModel):
    """Structured response for provenance detection on a single page."""
    explanation: str
    has_data: bool
    in_table: bool


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
    # Helpers
    # -----------------------------------------------------------------------

    def _get_page_numbers(self, context: str) -> list[int]:
        """Return sorted list of page numbers found in *context*."""
        return [int(p) for p in re.findall(r'<page number="(\d+)">', context)]

    def _get_page_text(self, context: str, page_number: int) -> str:
        """Extract the text content of a single page from *context*."""
        tag = f'<page number="{page_number}">'
        start = context.find(tag)
        if start == -1:
            return ""
        start += len(tag)
        end = context.find('</page>', start)
        if end == -1:
            return ""
        return context[start:end].strip()

    def _get_table_numbers_on_page(self, page_text: str) -> list[int]:
        """Return sorted list of table numbers found in *page_text*."""
        return sorted(int(t) for t in re.findall(r'<table number="(\d+)">', page_text))

    # -----------------------------------------------------------------------
    # Step 1: Document Level entity extraction
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

        # --- Build output: one record per (document, entity) ---
        entity_data = []
        for i, datapoint in enumerate(self.data):
            for j, entity in enumerate(doc_entities.get(i, [])):
                entity_id = f"doc_{i}_entity_{j}"
                entity_data.append(datapoint | entity | {'entity_id': entity_id})

        return entity_data
    

    # -----------------------------------------------------------------------
    # Step 1b: Entity provenance
    # -----------------------------------------------------------------------

    def _entity_provenance(self, entity_data):
        """
        For each unique (document, entity), determine which pages contain
        data for that entity.

        Args:
            entity_data: list of entity records from _extract_entities()

        Returns:
            dict mapping (doc_id, entity_id) -> list[{"page": int, "table": int|None}]
        """
        entity_fields = list(self.entity_identification_schema.model_fields.keys())

        # Collect unique (doc_id, entity_id) pairs with their context and description
        unique_entities = {}
        for record in entity_data:
            key = (record['document_id'], record['entity_id'])
            if key not in unique_entities:
                unique_entities[key] = record

        messages = []
        message_ids = []  # (doc_id, entity_id, page_number)

        for (doc_id, entity_id), record in unique_entities.items():
            context = record['context']
            entity_description = {k: v for k, v in record.items() if k in entity_fields}
            pages = self._get_page_numbers(context)

            for p in pages:
                page_text = self._get_page_text(context, p)
                if not page_text:
                    continue

                query = (
                    f"Entity description: {entity_description}\n\n"
                    f"Does this page contain directly reported numerical measurements "
                    f"for the described entity? If yes, indicate whether the data "
                    f"appears in a table or in prose text.\n\n"
                )
                prompt = (
                    f"## Instructions:\n{ENTITY_PROVENANCE_INSTRUCTIONS}\n\n"
                    f"## Context:\n{page_text}\n\n## Query:\n{query}"
                )
                messages.append([{"role": "user", "content": prompt}])
                message_ids.append((doc_id, entity_id, p))

        if not messages:
            return {}

        guided_decoding_params = GuidedDecodingParams(
            json=ProvenanceResponse.model_json_schema()
        )
        sampling_params = SamplingParams(
            **self.sampling_params,
            guided_decoding=guided_decoding_params
        )
        responses = self.llm.chat(messages=messages, sampling_params=sampling_params)
        response_texts = [r.outputs[0].text for r in responses]

        provenance = {}
        for msg_idx, resp in enumerate(response_texts):
            doc_id, entity_id, page_number = message_ids[msg_idx]
            try:
                result = response_validator(ProvenanceResponse, resp)
            except:
                print("Validation error in entity provenance response.")
                continue

            if result.get('has_data'):
                key = (doc_id, entity_id)
                if result.get('in_table'):
                    page_text = self._get_page_text(
                        unique_entities[(doc_id, entity_id)]['context'],
                        page_number,
                    )
                    for t in self._get_table_numbers_on_page(page_text):
                        provenance.setdefault(key, []).append({
                            'page': page_number,
                            'table': t,
                        })
                else:
                    provenance.setdefault(key, []).append({
                        'page': page_number,
                        'table': None,
                    })

        return provenance


    # -----------------------------------------------------------------------
    # Step 2: Document-level attribute detection
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
    # Step 2b: Attribute provenance
    # -----------------------------------------------------------------------

    def _attribute_provenance(self, doc_attributes):
        """
        For each (document, detected attribute), determine which pages
        contain data for that attribute.

        Args:
            doc_attributes: dict from _detect_attributes()
                {doc_idx: {attr_name: [terms]}}

        Returns:
            dict mapping (doc_id, attr_name) -> list[{"page": int, "table": int|None}]
        """
        messages = []
        message_ids = []  # (doc_id, attr_name, page_number)

        for doc_idx, attrs in doc_attributes.items():
            # doc_idx may be int or str depending on caller
            doc_idx_int = int(doc_idx)
            context = self.data[doc_idx_int]['context']
            pages = self._get_page_numbers(context)

            for attr_name, terms in attrs.items():
                attr_description = self.attribute_info_dict[attr_name].get('description', '')

                for p in pages:
                    page_text = self._get_page_text(context, p)
                    if not page_text:
                        continue

                    query = (
                        f"Attribute: {attr_name}\n"
                        f"Attribute description: {attr_description}\n"
                        f"Terminology used for the attribute: {terms}\n\n"
                        f"Does this page contain directly reported numerical measurements "
                        f"for the described attribute? If yes, indicate whether the data "
                        f"appears in a table or in prose text.\n\n"
                    )
                    prompt = (
                        f"## Instructions:\n{ATTRIBUTE_PROVENANCE_INSTRUCTIONS}\n\n"
                        f"## Context:\n{page_text}\n\n## Query:\n{query}"
                    )
                    messages.append([{"role": "user", "content": prompt}])
                    message_ids.append((doc_idx_int, attr_name, p))

        if not messages:
            return {}

        guided_decoding_params = GuidedDecodingParams(
            json=ProvenanceResponse.model_json_schema()
        )
        sampling_params = SamplingParams(
            **self.sampling_params,
            guided_decoding=guided_decoding_params
        )
        responses = self.llm.chat(messages=messages, sampling_params=sampling_params)
        response_texts = [r.outputs[0].text for r in responses]

        provenance = {}
        for msg_idx, resp in enumerate(response_texts):
            doc_id, attr_name, page_number = message_ids[msg_idx]
            try:
                result = response_validator(ProvenanceResponse, resp)
            except:
                print("Validation error in attribute provenance response.")
                continue

            if result.get('has_data'):
                key = (doc_id, attr_name)
                if result.get('in_table'):
                    page_text = self._get_page_text(
                        self.data[doc_id]['context'],
                        page_number,
                    )
                    for t in self._get_table_numbers_on_page(page_text):
                        provenance.setdefault(key, []).append({
                            'page': page_number,
                            'table': t,
                        })
                else:
                    provenance.setdefault(key, []).append({
                        'page': page_number,
                        'table': None,
                    })

        return provenance


    # -----------------------------------------------------------------------
    # Step 3: Extract values from text (per-page)
    # -----------------------------------------------------------------------

    def _extract_values_from_text(self, entity_data, doc_attributes, entity_prov, attr_prov):
        """
        Extracts measurement values from prose text using provenance intersection.

        For each (entity, attribute) pair per document, finds pages where BOTH
        have provenance with table=None, and only prompts for those pages.

        Args:
            entity_data: list of entity records from _extract_entities()
            doc_attributes: dict from _detect_attributes()
            entity_prov: dict from _entity_provenance()
            attr_prov: dict from _attribute_provenance()

        Returns records with 'value', 'units', 'page_number', and 'source'='text'.
        """
        entity_fields = list(self.entity_identification_schema.model_fields.keys())
        messages = []
        message_ids = []  # (record_dict, page_number)

        for record in entity_data:
            doc_id = record['document_id']
            entity_id = record['entity_id']
            context = record['context']

            for attr_name, terms in doc_attributes.get(doc_id, {}).items():
                # Provenance intersection: find pages where both entity and attribute
                # have provenance with table=None (prose text)
                e_pages = {
                    entry['page'] for entry in entity_prov.get((doc_id, entity_id), [])
                    if entry['table'] is None
                }
                a_pages = {
                    entry['page'] for entry in attr_prov.get((doc_id, attr_name), [])
                    if entry['table'] is None
                }
                intersecting_pages = sorted(e_pages & a_pages)

                if not intersecting_pages:
                    continue

                attr_description = self.attribute_info_dict[attr_name]['description']
                unit_options = self.attribute_info_dict[attr_name].get('units', [])
                entity_description = {k: v for k, v in record.items() if k in entity_fields}

                pair_record = record | {
                    'attribute': attr_name,
                    'attribute_terms': terms,
                }

                for p in intersecting_pages:
                    page_text = self._get_page_text(context, p)
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
                        f"Terminology used for the attribute: {terms}\n"
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
                    message_ids.append((pair_record, p))

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
            pair_record, page_number = message_ids[msg_idx]
            try:
                result = response_validator(TextValueExtractionResponse, resp)
            except:
                print("Validation error in text value extraction response.")
                continue

            if result.get('has_value') and result.get('value') is not None:
                page_text = self._get_page_text(pair_record['context'], page_number)

                text_values.append(
                    pair_record | {
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

    def _extract_values_from_tables(self, entity_data, doc_attributes, entity_prov, attr_prov):
        """
        Extracts measurement values from HTML tables using provenance intersection.

        For each (entity, attribute) pair per document, finds table numbers where
        BOTH have provenance, and only prompts for those tables.

        Args:
            entity_data: list of entity records from _extract_entities()
            doc_attributes: dict from _detect_attributes()
            entity_prov: dict from _entity_provenance()
            attr_prov: dict from _attribute_provenance()

        Returns records with 'value', 'units', 'table_number', 'row_index',
        'column_index', and 'source'='table'.
        """
        entity_fields = list(self.entity_identification_schema.model_fields.keys())
        messages = []
        message_ids = []  # (record_dict, table_number)
        table_cache = {}  # (doc_id, table_number) -> (table_text, row_names, column_names)

        def _get_table(context, t, doc_id):
            """Parse and cache a table, returning (table_text, row_names, column_names) or None."""
            cache_key = (doc_id, t)
            if cache_key in table_cache:
                return table_cache[cache_key]
            tag = f'<table number="{t}">'
            table_tag_start = context.find(tag)
            if table_tag_start == -1:
                return None
            table_content_start = table_tag_start + len(tag)
            table_end = context.find('</table>', table_content_start)
            table_text = context[table_tag_start:table_end + len('</table>')].strip()
            if not table_text:
                return None
            try:
                table_dfs = pd.read_html(StringIO(table_text))
                table_df = table_dfs[0]
                row_names = table_df.loc[:, "index"].to_list() if 'index' in table_df.columns else []
                row_names = [str(name) for name in row_names]
                column_names = [str(name) for name in table_df.columns.tolist()]
                column_names = [name for name in column_names if name != 'index']
            except:
                print(f"Error parsing table {t} in doc {doc_id}.")
                return None
            table_cache[cache_key] = (table_text, row_names, column_names)
            return table_cache[cache_key]

        for record in entity_data:
            doc_id = record['document_id']
            entity_id = record['entity_id']
            context = record['context']

            for attr_name, terms in doc_attributes.get(doc_id, {}).items():
                # Provenance intersection: find table numbers where both entity
                # and attribute have provenance
                entity_prov_entries = entity_prov.get((doc_id, entity_id), [])
                e_tables = {
                    entry['table'] for entry in entity_prov_entries
                    if entry['table'] is not None
                }
                a_tables = {
                    entry['table'] for entry in attr_prov.get((doc_id, attr_name), [])
                    if entry['table'] is not None
                }
                intersecting_tables = sorted(e_tables & a_tables)

                # Map table number -> page number for provenance attribution.
                table_to_page = {
                    entry['table']: entry['page']
                    for entry in entity_prov_entries
                    if entry['table'] is not None
                }

                if not intersecting_tables:
                    continue

                attr_description = self.attribute_info_dict[attr_name]['description']
                unit_options = self.attribute_info_dict[attr_name].get('units', [])
                entity_description = {k: v for k, v in record.items() if k in entity_fields}

                pair_record = record | {
                    'attribute': attr_name,
                    'attribute_terms': terms,
                }

                for t in intersecting_tables:
                    parsed = _get_table(context, t, doc_id)
                    if parsed is None:
                        continue
                    table_text, row_names, column_names = parsed
                    table_page_number = table_to_page.get(t)

                    units_guidance = ""
                    if unit_options:
                        units_guidance = (
                            f"Preferred unit options: {unit_options}. "
                            f"Strongly prioritize choosing the best option from this list. "
                            f"If none of the options fit, specify the unit exactly as it appears in the table.\n"
                        )

                    query = (
                        f"Attribute description: {attr_description}\n"
                        f"Terminology used for the attribute: {terms}\n"
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
                    message_ids.append((pair_record, t, table_page_number))

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
            pair_record, table_number, page_number = message_ids[msg_idx]
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
            doc_id = pair_record['document_id']
            parsed = table_cache.get((doc_id, table_number))
            if parsed is None:
                continue
            table_text, row_names, column_names = parsed
            try:
                table_dfs = pd.read_html(StringIO(table_text))
                table_df = table_dfs[0]

                # Ensure row and column indices are strings for matching
                table_df.columns = [str(c) for c in table_df.columns]
                if 'index' in table_df.columns:
                    table_df['index'] = table_df['index'].astype(str)

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
                print(f"Error extracting value from table {table_number} in doc {doc_id}.")
                val = None

            if val is not None:
                table_values.append(
                    pair_record | {
                        'context': table_text,
                        'value': val,
                        'units': result.get('units'),
                        'page_number': page_number,
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

    def _standardize(self):
        """
        LLM-based value cleanup: standardizes extracted measurement values
        (removes uncertainty, normalizes formatting, etc.).

        Reads from self.data and returns the standardized list.
        """
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

        standardized_data = [dict(datapoint) for datapoint in self.data]
        for i, resp in enumerate(response_texts):
            standardized_data[message_data_ids[i]]['value'] = resp.strip()

        return standardized_data

    '''
    def _deduplicate(self, data):
        """
        Simple equality-based deduplication using entity_id.

        Groups records by (entity_id, attribute), then within each group
        compares values: np.isclose for numerics, case-insensitive string
        match otherwise. Keeps first occurrence, discards duplicates.

        Args:
            data: list of measurement records

        Returns:
            list[dict]: Deduplicated records.
        """
        def _norm(v):
            if v is None:
                return None
            return str(v).strip().lower()

        def _values_equal(a, b):
            """Compare two values: np.isclose for numerics, case-insensitive otherwise."""
            try:
                fa, fb = float(a), float(b)
                return np.isclose(fa, fb)
            except (ValueError, TypeError):
                return _norm(a) == _norm(b)

        groups: dict[tuple, list[int]] = {}
        for idx, record in enumerate(data):
            key = (record.get('entity_id'), record.get('attribute'))
            groups.setdefault(key, []).append(idx)

        deduplicated: list[dict] = []

        for key, indices in groups.items():
            kept_values = []  # (value, units) tuples already kept
            for idx in indices:
                record = data[idx]
                val = record.get('value')
                units = _norm(record.get('units'))
                is_dup = False
                for kept_val, kept_units in kept_values:
                    if _values_equal(val, kept_val) and units == kept_units:
                        is_dup = True
                        break
                if not is_dup:
                    deduplicated.append(record)
                    kept_values.append((val, units))

        return deduplicated
    '''

    def _deduplicate(self, data):
        """
        Equality-based deduplication using entity_id, with provenance aggregation.

        Groups records by (entity_id, attribute), then within each group
        compares values: np.isclose for numerics, case-insensitive string
        match otherwise. Keeps first occurrence and aggregates provenance
        fields (page_number, table_number, row_index, column_index, source,
        context) from duplicates into aligned lists.

        Args:
            data: list of measurement records

        Returns:
            list[dict]: Deduplicated records with aggregated provenance.
        """
        def _norm(v):
            if v is None:
                return None
            return str(v).strip().lower()

        def _values_equal(a, b):
            """Compare two values: np.isclose for numerics, case-insensitive otherwise."""
            try:
                fa, fb = float(a), float(b)
                return np.isclose(fa, fb)
            except (ValueError, TypeError):
                return _norm(a) == _norm(b)

        # Provenance fields to aggregate into aligned lists
        _PROV_FIELDS = ('page_number', 'table_number', 'row_index', 'column_index', 'source', 'context')

        def _extract_provenance(record):
            """Extract a provenance tuple from a record, using None for missing fields."""
            return {field: record.get(field) for field in _PROV_FIELDS}

        groups: dict[tuple, list[int]] = {}
        for idx, record in enumerate(data):
            key = (record.get('entity_id'), record.get('attribute'))
            groups.setdefault(key, []).append(idx)

        deduplicated: list[dict] = []

        for key, indices in groups.items():
            # Each entry: (value, units, index into deduplicated list)
            kept_values: list[tuple] = []
            for idx in indices:
                record = data[idx]
                val = record.get('value')
                units = _norm(record.get('units'))
                prov = _extract_provenance(record)

                is_dup = False
                for kept_val, kept_units, dedup_idx in kept_values:
                    if _values_equal(val, kept_val) and units == kept_units:
                        # Aggregate provenance: append all fields together to stay aligned
                        kept_record = deduplicated[dedup_idx]
                        for field in _PROV_FIELDS:
                            kept_record[field].append(prov[field])
                        is_dup = True
                        break

                if not is_dup:
                    # Create new record with provenance fields as single-element lists
                    new_record = {
                        k: v for k, v in record.items() if k not in _PROV_FIELDS
                    }
                    for field in _PROV_FIELDS:
                        new_record[field] = [prov[field]]
                    dedup_idx = len(deduplicated)
                    deduplicated.append(new_record)
                    kept_values.append((val, units, dedup_idx))

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

        # Step 1: Entity extraction
        entity_data = self._extract_entities()

        # Step 2: Entity provenance
        entity_prov = self._entity_provenance(entity_data)

        # Step 3: Document-level attribute detection
        self.data = doc_data
        doc_attributes = self._detect_attributes()

        # Step 4: Attribute provenance
        attr_prov = self._attribute_provenance(doc_attributes)

        # Step 5: Extract values from text (provenance intersection)
        text_values = self._extract_values_from_text(
            entity_data, doc_attributes, entity_prov, attr_prov
        )

        # Step 6: Extract values from tables (provenance intersection)
        table_values = self._extract_values_from_tables(
            entity_data, doc_attributes, entity_prov, attr_prov
        )

        # Combine text and table extractions
        self.data = text_values + table_values

        # Step 7: Standardize
        self.data = self._standardize()

        # Step 8: Deduplicate
        self.data = self._deduplicate(self.data)

        return self.data


    def save(self, filepath: str):
        """
        Saves the measurement data to a JSON file.

        Args:
            filepath (str): The path to the file where the data will be saved.
        """
        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
