"""MeasurementLM Ablation 3: Direct Table Value Extraction

Ablation goal: understand what happens when the model is asked to extract a table
value directly, rather than first identifying the row and column indices that
locate the value and then programmatically retrieving it.

Changes from the baseline MeasurementLM:

1. `_extract_values_from_tables()`: the query no longer lists row or column names
   and does not ask the model for row_index/column_index. Instead, the model
   is asked to find and return the value directly, using the same
   TextValueExtractionResponse structure as text-based extraction.

2. A new prompt EXTRACT_TABLE_VALUE_DIRECT_INSTRUCTIONS is used, which instructs
   the model to extract the value directly from the table rather than specifying
   indices.

3. After extraction, there is no programmatic pandas table indexing step; the
   value returned by the model is recorded directly into the dataset.

4. The output record for table extractions no longer contains 'row_index' or
   'column_index' fields, since the model does not produce these. It does still
   contain 'table_number', 'page_number', and 'source'='table'.

Unchanged from baseline: `_extract_values_from_text()`, all provenance steps,
`_standardize()`, `_deduplicate()`, and `fit()`.
"""

from .measurementlm import (
    MeasurementLM,
    TextValueExtractionResponse,
    response_validator,
)
from .instruction_prompts import (
    EXTRACT_TABLE_VALUE_DIRECT_INSTRUCTIONS,  # NEW: direct value extraction prompt
)
from io import StringIO
import pandas as pd


class MeasurementLMAblation3(MeasurementLM):
    """
    Ablation 3: table value extraction asks the model to return the value
    directly rather than row/column indices for programmatic lookup.
    """

    # -----------------------------------------------------------------------
    # Step 6: Extract values from tables  (direct value extraction)
    # -----------------------------------------------------------------------

    def _extract_values_from_tables(self, entity_data, doc_attributes, entity_prov, attr_prov):
        """
        Extract measurement values from HTML tables by asking the model to
        return the value directly.

        CHANGED:
        - Row and column name lists are NOT included in the query.
        - The model responds with TextValueExtractionResponse (value + units)
          instead of TableValueExtractionResponse (row_index + column_index + units).
        - No pandas indexing step after the model call; the value is used as-is.
        - EXTRACT_TABLE_VALUE_DIRECT_INSTRUCTIONS is used instead of
          EXTRACT_TABLE_VALUE_INSTRUCTIONS.
        """
        entity_fields = list(self.entity_identification_schema.model_fields.keys())
        messages = []
        message_ids = []  # (record_dict, table_number, page_number)

        def _get_table_text(context, t, doc_id):
            """Extract raw table HTML from context."""
            tag = f'<table number="{t}">'
            table_tag_start = context.find(tag)
            if table_tag_start == -1:
                return None
            table_content_start = table_tag_start + len(tag)
            table_end = context.find("</table>", table_content_start)
            table_text = context[table_tag_start : table_end + len("</table>")].strip()
            return table_text if table_text else None

        for record in entity_data:
            doc_id = record["document_id"]
            entity_id = record["entity_id"]
            context = record["context"]

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
                    table_text = _get_table_text(context, t, doc_id)
                    if table_text is None:
                        continue
                    table_page_number = table_to_page.get(t)

                    units_guidance = ""
                    if unit_options:
                        units_guidance = (
                            f"Preferred unit options: {unit_options}. "
                            f"Strongly prioritize choosing the best option from this list. "
                            f"If none of the options fit, specify the unit exactly as it appears in the table.\n"
                        )

                    # CHANGED: query no longer lists row/column names; asks for the value directly
                    query = (
                        f"Attribute description: {attr_description}\n"
                        f"Terminology used for the attribute: {terms}\n"
                        f"Entity description: {entity_description}\n\n"
                        f"{units_guidance}"
                        f"Does this table contain a measured value for the given attribute and entity? "
                        f"If yes, extract the value and its units directly from the table.\n\n"
                    )
                    # CHANGED: uses EXTRACT_TABLE_VALUE_DIRECT_INSTRUCTIONS
                    prompt = (
                        f"## INSTRUCTIONS:\n{EXTRACT_TABLE_VALUE_DIRECT_INSTRUCTIONS}\n\n"
                        f"## CONTEXT:\n{table_text}\n\n## QUERY:\n{query}"
                    )
                    messages.append([{"role": "user", "content": prompt}])
                    message_ids.append((pair_record, t, table_page_number, table_text))

        if not messages:
            return []

        # CHANGED: response schema is TextValueExtractionResponse, not TableValueExtractionResponse
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "text_value_extraction",
                "schema": TextValueExtractionResponse.model_json_schema(),
            },
        }
        response_texts = self._call_batch(messages, response_format=response_format)

        table_values = []
        for msg_idx, resp in enumerate(response_texts):
            pair_record, table_number, page_number, table_text = message_ids[msg_idx]
            try:
                result = response_validator(TextValueExtractionResponse, resp)
            except Exception as e:
                print(f"Validation error in direct table value extraction response: {e}")
                print(f"Response text: {resp[:500]}")
                continue

            # CHANGED: no pandas indexing; value comes directly from the model response
            if result.get("has_value") and result.get("value") is not None:
                table_values.append(
                    pair_record | {
                        "context": table_text,
                        "value": result["value"],
                        "units": result.get("units"),
                        "page_number": page_number,
                        "table_number": table_number,
                        # CHANGED: no row_index or column_index fields
                        "source": "table",
                    }
                )

        return table_values
