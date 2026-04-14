"""MeasurementLM Ablation 5: No Explanations in Structured Responses

Ablation goal: understand what happens when the model is not asked to produce
chain-of-thought explanations in its structured JSON responses.

Changes from the baseline MeasurementLM:

1. New pydantic response models defined here that drop the `explanation` field:
     - AttributeDetectionItemNoExp       (no explanation)
     - BatchAttributeDetectionResponseNoExp
     - ProvenanceResponseNoExp           (no explanation)
     - TextValueExtractionResponseNoExp  (no explanation)
     - TableValueExtractionResponseNoExp (no explanation)

2. Five inference steps use no-explanation prompts and the new schemas:
     - _detect_attributes()          uses DETECT_ATTRIBUTES_BATCH_INSTRUCTIONS_NO_EXPLANATIONS
     - _entity_provenance()          uses ENTITY_PROVENANCE_INSTRUCTIONS_NO_EXPLANATIONS
     - _attribute_provenance()       uses ATTRIBUTE_PROVENANCE_INSTRUCTIONS_NO_EXPLANATIONS
     - _extract_values_from_text()   uses EXTRACT_TEXT_VALUE_INSTRUCTIONS_NO_EXPLANATIONS
     - _extract_values_from_tables() uses EXTRACT_TABLE_VALUE_INSTRUCTIONS_NO_EXPLANATIONS

3. _standardize() is unchanged: it returns free text without any explanation field.

Pipeline structure and all other logic are identical to the baseline.
"""

from io import StringIO
import pandas as pd
from pydantic import BaseModel

from .measurementlm import MeasurementLM, response_validator
from .instruction_prompts import (
    DETECT_ATTRIBUTES_BATCH_INSTRUCTIONS_NO_EXPLANATIONS,
    ENTITY_PROVENANCE_INSTRUCTIONS_NO_EXPLANATIONS,
    ATTRIBUTE_PROVENANCE_INSTRUCTIONS_NO_EXPLANATIONS,
    EXTRACT_TEXT_VALUE_INSTRUCTIONS_NO_EXPLANATIONS,
    EXTRACT_TABLE_VALUE_INSTRUCTIONS_NO_EXPLANATIONS,
)


# -----------------------------------------------------------------------
# Response models without explanation fields
# -----------------------------------------------------------------------

class AttributeDetectionItemNoExp(BaseModel):
    attribute_name: str
    detected: bool
    terms: list[str]


class BatchAttributeDetectionResponseNoExp(BaseModel):
    items: list[AttributeDetectionItemNoExp]


class ProvenanceResponseNoExp(BaseModel):
    has_data: bool
    in_table: bool


class TextValueExtractionResponseNoExp(BaseModel):
    has_value: bool
    value: str | None = None
    units: str | None = None


class TableValueExtractionResponseNoExp(BaseModel):
    has_value: bool
    row_index: str | None = None
    column_index: str | None = None
    units: str | None = None


class MeasurementLMAblation5(MeasurementLM):
    """
    Ablation 5: structured responses omit the explanation field.
    Uses no-explanation prompts and pydantic schemas without explanation.
    Pipeline structure is identical to the baseline.
    """

    # -----------------------------------------------------------------------
    # Step 2: Document-level attribute detection
    # -----------------------------------------------------------------------

    def _detect_attributes(self):
        """
        CHANGED: uses DETECT_ATTRIBUTES_BATCH_INSTRUCTIONS_NO_EXPLANATIONS
        and BatchAttributeDetectionResponseNoExp (no explanation field).
        """
        attr_names = list(self.attribute_info_dict.keys())
        attribute_list_text = self._format_attribute_list()

        # CHANGED: no-explanation schema
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "batch_attribute_detection",
                "schema": BatchAttributeDetectionResponseNoExp.model_json_schema(),
            },
        }

        messages = []
        message_ids = []
        for i, datapoint in enumerate(self.data):
            context = datapoint['context']
            query = (
                f"Attributes to evaluate:\n{attribute_list_text}\n\n"
                f"For each attribute listed above, determine whether the document "
                f"contains any direct numerical measurements for that attribute. "
                f"Return one item per attribute using the exact attribute name.\n\n"
            )
            # CHANGED: no-explanation prompt
            prompt = (
                f"## Instructions:\n{DETECT_ATTRIBUTES_BATCH_INSTRUCTIONS_NO_EXPLANATIONS}\n\n"
                f"## Context:\n{context}\n\n## Query:\n{query}"
            )
            messages.append([{"role": "user", "content": prompt}])
            message_ids.append(i)

        response_texts = self._call_batch(
            messages,
            response_format=response_format,
            max_retries=1,
            validator=lambda r: response_validator(BatchAttributeDetectionResponseNoExp, r),
        )

        detection_results: dict[int, dict[str, bool]] = {}
        attribute_terms: dict[int, dict[str, list[str]]] = {}

        for msg_idx, resp in enumerate(response_texts):
            doc_idx = message_ids[msg_idx]
            try:
                # CHANGED: no-explanation schema
                batch = response_validator(BatchAttributeDetectionResponseNoExp, resp)
            except Exception as e:
                print(f"Validation error in batched attribute detection response: {e}")
                print(f"Response text: {resp[:500]}")
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
    # Step 2a: Entity provenance
    # -----------------------------------------------------------------------

    def _entity_provenance(self, entity_data):
        """
        CHANGED: uses ENTITY_PROVENANCE_INSTRUCTIONS_NO_EXPLANATIONS
        and ProvenanceResponseNoExp (no explanation field).
        """
        entity_fields = list(self.entity_identification_schema.model_fields.keys())

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
                # CHANGED: no-explanation prompt
                prompt = (
                    f"## Instructions:\n{ENTITY_PROVENANCE_INSTRUCTIONS_NO_EXPLANATIONS}\n\n"
                    f"## Context:\n{page_text}\n\n## Query:\n{query}"
                )
                messages.append([{"role": "user", "content": prompt}])
                message_ids.append((doc_id, entity_id, p))

        if not messages:
            return {}

        # CHANGED: no-explanation schema
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "provenance_response",
                "schema": ProvenanceResponseNoExp.model_json_schema(),
            },
        }
        response_texts = self._call_batch(
            messages,
            response_format=response_format,
            max_retries=1,
            validator=lambda r: response_validator(ProvenanceResponseNoExp, r),
        )

        provenance = {}
        for msg_idx, resp in enumerate(response_texts):
            doc_id, entity_id, page_number = message_ids[msg_idx]
            try:
                # CHANGED: no-explanation schema
                result = response_validator(ProvenanceResponseNoExp, resp)
            except Exception as e:
                print(f"Validation error in entity provenance response: {e}")
                print(f"Response text: {resp[:500]}")
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
    # Step 2b: Attribute provenance
    # -----------------------------------------------------------------------

    def _attribute_provenance(self, doc_attributes):
        """
        CHANGED: uses ATTRIBUTE_PROVENANCE_INSTRUCTIONS_NO_EXPLANATIONS
        and ProvenanceResponseNoExp (no explanation field).
        """
        messages = []
        message_ids = []  # (doc_id, attr_name, page_number)

        for doc_idx, attrs in doc_attributes.items():
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
                    # CHANGED: no-explanation prompt
                    prompt = (
                        f"## INSTRUCTIONS:\n{ATTRIBUTE_PROVENANCE_INSTRUCTIONS_NO_EXPLANATIONS}\n\n"
                        f"## CONTEXT:\n{page_text}\n\n## QUERY:\n{query}"
                    )
                    messages.append([{"role": "user", "content": prompt}])
                    message_ids.append((doc_idx_int, attr_name, p))

        if not messages:
            return {}

        # CHANGED: no-explanation schema
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "provenance_response",
                "schema": ProvenanceResponseNoExp.model_json_schema(),
            },
        }
        response_texts = self._call_batch(
            messages,
            response_format=response_format,
            max_retries=1,
            validator=lambda r: response_validator(ProvenanceResponseNoExp, r),
        )

        provenance = {}
        for msg_idx, resp in enumerate(response_texts):
            doc_id, attr_name, page_number = message_ids[msg_idx]
            try:
                # CHANGED: no-explanation schema
                result = response_validator(ProvenanceResponseNoExp, resp)
            except Exception as e:
                print(f"Validation error in attribute provenance response: {e}")
                print(f"Response text: {resp[:500]}")
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
    # Step 5: Extract values from text
    # -----------------------------------------------------------------------

    def _extract_values_from_text(self, entity_data, doc_attributes, entity_prov, attr_prov):
        """
        CHANGED: uses EXTRACT_TEXT_VALUE_INSTRUCTIONS_NO_EXPLANATIONS
        and TextValueExtractionResponseNoExp (no explanation field).
        """
        entity_fields = list(self.entity_identification_schema.model_fields.keys())
        messages = []
        message_ids = []  # (record_dict, page_number)

        for record in entity_data:
            doc_id = record['document_id']
            entity_id = record['entity_id']
            context = record['context']

            for attr_name, terms in doc_attributes.get(doc_id, {}).items():
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
                    # CHANGED: no-explanation prompt
                    prompt = (
                        f"## INSTRUCTIONS:\n{EXTRACT_TEXT_VALUE_INSTRUCTIONS_NO_EXPLANATIONS}\n\n"
                        f"## CONTEXT:\n{page_text}\n\n## QUERY:\n{query}"
                    )
                    messages.append([{"role": "user", "content": prompt}])
                    message_ids.append((pair_record, p))

        if not messages:
            return []

        # CHANGED: no-explanation schema
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "text_value_extraction",
                "schema": TextValueExtractionResponseNoExp.model_json_schema(),
            },
        }
        response_texts = self._call_batch(
            messages,
            response_format=response_format,
            max_retries=1,
            validator=lambda r: response_validator(TextValueExtractionResponseNoExp, r),
        )

        text_values = []
        for msg_idx, resp in enumerate(response_texts):
            pair_record, page_number = message_ids[msg_idx]
            try:
                # CHANGED: no-explanation schema
                result = response_validator(TextValueExtractionResponseNoExp, resp)
            except Exception as e:
                print(f"Validation error in text value extraction response: {e}")
                print(f"Response text: {resp[:500]}")
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
    # Step 6: Extract values from tables
    # -----------------------------------------------------------------------

    def _extract_values_from_tables(self, entity_data, doc_attributes, entity_prov, attr_prov):
        """
        CHANGED: uses EXTRACT_TABLE_VALUE_INSTRUCTIONS_NO_EXPLANATIONS
        and TableValueExtractionResponseNoExp (no explanation field).
        """
        entity_fields = list(self.entity_identification_schema.model_fields.keys())
        messages = []
        message_ids = []  # (record_dict, table_number, page_number)
        table_cache = {}  # (doc_id, table_number) -> (table_text, row_names, column_names)

        def _get_table(context, t, doc_id):
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
            except Exception:
                print(f"Error parsing table {t} in doc {doc_id}.")
                return None
            table_cache[cache_key] = (table_text, row_names, column_names)
            return table_cache[cache_key]

        for record in entity_data:
            doc_id = record['document_id']
            entity_id = record['entity_id']
            context = record['context']

            for attr_name, terms in doc_attributes.get(doc_id, {}).items():
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
                    # CHANGED: no-explanation prompt
                    prompt = (
                        f"## Instructions:\n{EXTRACT_TABLE_VALUE_INSTRUCTIONS_NO_EXPLANATIONS}\n\n"
                        f"## Context:\n{table_text}\n\n## Query:\n{query}"
                    )
                    messages.append([{"role": "user", "content": prompt}])
                    message_ids.append((pair_record, t, table_page_number))

        if not messages:
            return []

        # CHANGED: no-explanation schema
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "table_value_extraction",
                "schema": TableValueExtractionResponseNoExp.model_json_schema(),
            },
        }
        response_texts = self._call_batch(
            messages,
            response_format=response_format,
            max_retries=1,
            validator=lambda r: response_validator(TableValueExtractionResponseNoExp, r),
        )

        table_values = []
        for msg_idx, resp in enumerate(response_texts):
            pair_record, table_number, page_number = message_ids[msg_idx]
            try:
                # CHANGED: no-explanation schema
                result = response_validator(TableValueExtractionResponseNoExp, resp)
            except Exception as e:
                print(f"Validation error in table value extraction response: {e}")
                print(f"Response text: {resp[:500]}")
                continue

            if not result.get('has_value'):
                continue
            row_index = result.get('row_index')
            column_index = result.get('column_index')
            if row_index is None or column_index is None:
                continue

            doc_id = pair_record['document_id']
            parsed = table_cache.get((doc_id, table_number))
            if parsed is None:
                continue
            table_text, row_names, column_names = parsed
            try:
                table_dfs = pd.read_html(StringIO(table_text))
                table_df = table_dfs[0]
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

            except Exception:
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
