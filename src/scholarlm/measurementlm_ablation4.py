"""MeasurementLM Ablation 4: Full-Document Context for Value Extraction

Ablation goal: understand what happens when value extraction (and event resolution)
are performed using the full document context rather than zooming in to a single
page or table.

Changes from the baseline MeasurementLM:

1. `_resolve_events()`: instead of slicing out a single page's text and asking
   about events there, the full document context is given to the model.
   Uses MEASUREMENT_EVENT_INSTRUCTIONS_FULL_CONTEXT.

2. `_extract_values_from_text()`: instead of slicing out the matched page's text
   and passing it as context, the full document context from the record's 'context'
   field is given to the model.

3. `_extract_values_from_tables()`: instead of slicing out the matched table's HTML
   and passing it as context, the full document context is given to the model.

4. Uses EXTRACT_TEXT_VALUE_INSTRUCTIONS_FULL_CONTEXT and
   EXTRACT_TABLE_VALUE_INSTRUCTIONS_FULL_CONTEXT for value extraction, and
   MEASUREMENT_EVENT_INSTRUCTIONS_FULL_CONTEXT for event resolution.

5. Output records store the full document context in their 'context' field (rather
   than the page text or table text), since the model operated on the full context.

Unchanged from baseline: all provenance steps, `_standardize()`, `_deduplicate()`.
"""

from .measurementlm import (
    MeasurementLM,
    TextValueExtractionResponse,
    TableValueExtractionResponse,
    response_validator,
)
from .instruction_prompts import (
    EXTRACT_TEXT_VALUE_INSTRUCTIONS_FULL_CONTEXT,
    EXTRACT_TABLE_VALUE_INSTRUCTIONS_FULL_CONTEXT,
    MEASUREMENT_EVENT_INSTRUCTIONS_FULL_CONTEXT,
)
from io import StringIO
from pydantic import create_model
import pandas as pd


class MeasurementLMAblation4(MeasurementLM):
    """
    Ablation 4: event resolution and value extraction use full document context
    instead of page- or table-level context.
    """

    # -----------------------------------------------------------------------
    # Step 4.5: Measurement event resolution  (full-document context)
    # -----------------------------------------------------------------------

    def _resolve_events(self, entity_data, doc_attributes, entity_prov, attr_prov):
        """Uses full document context with MEASUREMENT_EVENT_INSTRUCTIONS_FULL_CONTEXT."""
        if self.measurement_event_schema is None:
            return {}

        EventList = create_model(
            "EventList",
            items=(list[self.measurement_event_schema], ...),
        )
        event_list_json = EventList.model_json_schema()
        entity_fields = list(self.entity_identification_schema.model_fields.keys())

        unique_entities = {}
        for record in entity_data:
            key = (record["document_id"], record["entity_id"])
            if key not in unique_entities:
                unique_entities[key] = record

        messages = []
        message_ids = []  # (doc_id, entity_id, attr_name, page_number)

        for (doc_id, entity_id), record in unique_entities.items():
            context = record["context"]
            entity_description = {k: v for k, v in record.items() if k in entity_fields}

            for attr_name, _terms in doc_attributes.get(doc_id, {}).items():
                attr_description = self.attribute_info_dict[attr_name].get("description", "")

                e_pages = {entry["page"] for entry in entity_prov.get((doc_id, entity_id), [])}
                a_pages = {entry["page"] for entry in attr_prov.get((doc_id, attr_name), [])}
                intersecting_pages = sorted(e_pages & a_pages)

                for p in intersecting_pages:
                    query = (
                        f"Entity description: {entity_description}\n"
                        f"Attribute: {attr_name}\n"
                        f"Attribute description: {attr_description}\n\n"
                        f"Enumerate all distinct measurement events for the above entity "
                        f"and attribute found in the document.\n\n"
                    )
                    prompt = (
                        f"## INSTRUCTIONS:\n{MEASUREMENT_EVENT_INSTRUCTIONS_FULL_CONTEXT}\n\n"
                        f"## EVENT DETAILS:\n{self.measurement_event_prompt}\n\n"
                        f"## CONTEXT:\n{context}\n\n## QUERY:\n{query}"
                    )
                    messages.append([{"role": "user", "content": prompt}])
                    message_ids.append((doc_id, entity_id, attr_name, p))

        if not messages:
            return {}

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "event_list",
                "schema": event_list_json,
            },
        }
        response_texts = self._call_batch(
            messages,
            response_format=response_format,
            max_retries=1,
            max_tokens=16384,
            validator=lambda r: response_validator(EventList, r),
        )

        event_resolution = {}
        for msg_idx, resp in enumerate(response_texts):
            key = message_ids[msg_idx]
            try:
                result = response_validator(EventList, resp)
            except Exception as e:
                print(f"Validation error in event resolution response: {e}")
                print(f"Response text: {resp}")
                event_resolution[key] = []
                continue
            event_resolution[key] = result["items"]

        return event_resolution

    # -----------------------------------------------------------------------
    # Step 5: Extract values from text  (full-document context)
    # -----------------------------------------------------------------------

    def _extract_values_from_text(self, entity_data, doc_attributes, entity_prov, attr_prov, event_resolution=None):
        """
        Extract measurement values from prose text using full document context.

        The model receives the full document context instead of only the matched
        page's text. Uses EXTRACT_TEXT_VALUE_INSTRUCTIONS_FULL_CONTEXT.
        """
        entity_fields = list(self.entity_identification_schema.model_fields.keys())
        messages = []
        message_ids = []  # (record_dict, page_number)

        for record in entity_data:
            doc_id = record["document_id"]
            entity_id = record["entity_id"]
            context = record["context"]  # full document context

            for attr_name, terms in doc_attributes.get(doc_id, {}).items():
                e_pages = {
                    entry["page"] for entry in entity_prov.get((doc_id, entity_id), [])
                    if entry["table"] is None
                }
                a_pages = {
                    entry["page"] for entry in attr_prov.get((doc_id, attr_name), [])
                    if entry["table"] is None
                }
                intersecting_pages = sorted(e_pages & a_pages)

                if not intersecting_pages:
                    continue

                attr_description = self.attribute_info_dict[attr_name]["description"]
                unit_options = self.attribute_info_dict[attr_name].get("units", [])
                entity_description = {k: v for k, v in record.items() if k in entity_fields}

                pair_record = record | {
                    "attribute": attr_name,
                    "attribute_terms": terms,
                }

                for p in intersecting_pages:
                    units_guidance = ""
                    if unit_options:
                        units_guidance = (
                            f"Preferred unit options: {unit_options}. "
                            f"Strongly prioritize choosing the best option from this list. "
                            f"If none of the options fit, specify the unit exactly as it appears in the text.\n"
                        )

                    # Determine measurement events for this (entity, attribute, page)
                    if event_resolution is not None:
                        events = event_resolution.get((doc_id, entity_id, attr_name, p), [])
                        if not events:
                            events = [{f: None for f in self.measurement_event_schema.model_fields}]
                    else:
                        events = [None]

                    for event in events:
                        event_record = pair_record | (event if event is not None else {})
                        event_context = ""
                        if event and any(v is not None for v in event.values()):
                            event_context = f"Measurement event context: {event}\n"

                        query = (
                            f"Attribute description: {attr_description}\n"
                            f"Terminology used for the attribute: {terms}\n"
                            f"Entity description: {entity_description}\n"
                            f"{event_context}"
                            f"{units_guidance}"
                            f"Does the document contain a measured value for "
                            f"the given attribute and entity? "
                            f"If yes, extract the value and its units.\n\n"
                        )
                        prompt = (
                            f"## INSTRUCTIONS:\n{EXTRACT_TEXT_VALUE_INSTRUCTIONS_FULL_CONTEXT}\n\n"
                            f"## CONTEXT:\n{context}\n\n## QUERY:\n{query}"
                        )
                        messages.append([{"role": "user", "content": prompt}])
                        message_ids.append((event_record, p))

        if not messages:
            return []

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "text_value_extraction",
                "schema": TextValueExtractionResponse.model_json_schema(),
            },
        }
        response_texts = self._call_batch(
            messages,
            response_format=response_format,
            max_retries=1,
            validator=lambda r: response_validator(TextValueExtractionResponse, r),
        )

        text_values = []
        for msg_idx, resp in enumerate(response_texts):
            event_record, page_number = message_ids[msg_idx]
            try:
                result = response_validator(TextValueExtractionResponse, resp)
            except Exception as e:
                print(f"Validation error in text value extraction response: {e}")
                print(f"Response text: {resp[:500]}")
                continue

            if result.get("has_value") and result.get("value") is not None:
                text_values.append(
                    event_record | {
                        "context": event_record["context"],
                        "value": result["value"],
                        "units": result.get("units"),
                        "page_number": page_number,
                        "source": "text",
                    }
                )

        return text_values

    # -----------------------------------------------------------------------
    # Step 6: Extract values from tables  (full-document context)
    # -----------------------------------------------------------------------

    def _extract_values_from_tables(self, entity_data, doc_attributes, entity_prov, attr_prov, event_resolution=None):
        """
        Extract measurement values from HTML tables using full document context.

        The model receives the full document context instead of only the matched
        table's HTML. Uses EXTRACT_TABLE_VALUE_INSTRUCTIONS_FULL_CONTEXT.
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
            table_end = context.find("</table>", table_content_start)
            table_text = context[table_tag_start : table_end + len("</table>")].strip()
            if not table_text:
                return None
            try:
                table_dfs = pd.read_html(StringIO(table_text))
                table_df = table_dfs[0]
                row_names = (
                    table_df.loc[:, "index"].to_list()
                    if "index" in table_df.columns
                    else []
                )
                row_names = [str(name) for name in row_names]
                column_names = [str(name) for name in table_df.columns.tolist()]
                column_names = [name for name in column_names if name != "index"]
            except Exception:
                print(f"Error parsing table {t} in doc {doc_id}.")
                return None
            table_cache[cache_key] = (table_text, row_names, column_names)
            return table_cache[cache_key]

        for record in entity_data:
            doc_id = record["document_id"]
            entity_id = record["entity_id"]
            context = record["context"]  # full document context

            for attr_name, terms in doc_attributes.get(doc_id, {}).items():
                entity_prov_entries = entity_prov.get((doc_id, entity_id), [])
                e_tables = {
                    entry["table"] for entry in entity_prov_entries
                    if entry["table"] is not None
                }
                a_tables = {
                    entry["table"] for entry in attr_prov.get((doc_id, attr_name), [])
                    if entry["table"] is not None
                }
                intersecting_tables = sorted(e_tables & a_tables)

                table_to_page = {
                    entry["table"]: entry["page"]
                    for entry in entity_prov_entries
                    if entry["table"] is not None
                }

                if not intersecting_tables:
                    continue

                attr_description = self.attribute_info_dict[attr_name]["description"]
                unit_options = self.attribute_info_dict[attr_name].get("units", [])
                entity_description = {k: v for k, v in record.items() if k in entity_fields}

                pair_record = record | {
                    "attribute": attr_name,
                    "attribute_terms": terms,
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

                    # Determine measurement events for this (entity, attribute, page)
                    if event_resolution is not None:
                        events = event_resolution.get(
                            (doc_id, entity_id, attr_name, table_page_number), []
                        )
                        if not events:
                            events = [{f: None for f in self.measurement_event_schema.model_fields}]
                    else:
                        events = [None]

                    for event in events:
                        event_record = pair_record | (event if event is not None else {})
                        event_context = ""
                        if event and any(v is not None for v in event.values()):
                            event_context = f"Measurement event context: {event}\n"

                        query = (
                            f"Attribute description: {attr_description}\n"
                            f"Terminology used for the attribute: {terms}\n"
                            f"Entity description: {entity_description}\n"
                            f"{event_context}"
                            f"{units_guidance}"
                            f"Does the document contain a measured value for the given attribute "
                            f"and entity in a table? "
                            f"If yes, provide the row_index and column_index names, and the units.\n\n"
                        )
                        prompt = (
                            f"## INSTRUCTIONS:\n{EXTRACT_TABLE_VALUE_INSTRUCTIONS_FULL_CONTEXT}\n\n"
                            f"## CONTEXT:\n{context}\n\n## QUERY:\n{query}"
                        )
                        messages.append([{"role": "user", "content": prompt}])
                        message_ids.append((event_record, t, table_page_number))

        if not messages:
            return []

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "table_value_extraction",
                "schema": TableValueExtractionResponse.model_json_schema(),
            },
        }
        response_texts = self._call_batch(
            messages,
            response_format=response_format,
            max_retries=1,
            validator=lambda r: response_validator(TableValueExtractionResponse, r),
        )

        table_values = []
        for msg_idx, resp in enumerate(response_texts):
            event_record, table_number, page_number = message_ids[msg_idx]
            try:
                result = response_validator(TableValueExtractionResponse, resp)
            except Exception as e:
                print(f"Validation error in table value extraction response: {e}")
                print(f"Response text: {resp[:500]}")
                continue

            if not result.get("has_value"):
                continue
            row_index = result.get("row_index")
            column_index = result.get("column_index")
            if row_index is None or column_index is None:
                continue

            doc_id = event_record["document_id"]
            parsed = table_cache.get((doc_id, table_number))
            if parsed is None:
                continue
            table_text, row_names, column_names = parsed
            try:
                table_dfs = pd.read_html(StringIO(table_text))
                table_df = table_dfs[0]
                table_df.columns = [str(c) for c in table_df.columns]
                if "index" in table_df.columns:
                    table_df["index"] = table_df["index"].astype(str)
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
                    event_record | {
                        "context": event_record["context"],
                        "value": val,
                        "units": result.get("units"),
                        "page_number": page_number,
                        "table_number": table_number,
                        "row_index": row_index,
                        "column_index": column_index,
                        "source": "table",
                    }
                )

        return table_values
