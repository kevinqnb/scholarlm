"""Prompt instruction strings used across `scholarlm`.
Notes:
- These are *instruction* blocks only. Query/context formatting remains in caller code.
"""

# --------------------------------------------
# Data Extraction Prompts
# --------------------------------------------

DETECT_ATTRIBUTES_BATCH_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to evaluate ALL of the listed attributes at once against context from a research paper, determining whether each attribute has any directly reported numerical measurements anywhere in the document.

Guidelines:
- You MUST return one item per attribute, using the EXACT attribute name provided. Do not rename, skip, or add attributes.
- Set detected to false if the given attribute does not appear in the context.
- Set detected to false if the context does not explicitly provide data for the given attribute.
- Set detected to false if the data reported is not a direct numerical measurement.
- Set detected to false if the data reported only contains values for parameter estimates or measures of fit for a statistical model.
- Set detected to false for cases where there is not a clear choice for a single, numerical data value.
- Set detected to true only if the context explicitly provides a direct numerical measurement for the given attribute.
- For each attribute, provide a brief explanation justifying your decision.
- When detected is true, populate the terms list with any terminology or abbreviations used in the context to refer to that attribute. Pay close attention to tables and figure captions, as these often contain abbreviations used in the main text. Do not infer, guess, or fabricate terms not explicitly present in the context.
- When detected is false, return an empty list for terms.
- Structure your response as a JSON object with an "items" list, where each item has "attribute_name", "explanation", "detected", and "terms" fields.
"""


ENTITY_PROVENANCE_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to determine if a single page of text from a research paper contains data for a described entity.

Guidelines:
- You will be provided with a single page of text from a research paper and a description of an entity.
- Set has_data to true only if the page contains directly reported numerical measurements associated with the described entity.
- Set has_data to false if the entity is not mentioned on the page, or if there are no numerical measurements for it.
- Set has_data to false if the data reported only contains values for parameter estimates or measures of fit for a statistical model.
- If has_data is true and the data appears within a table on the page, set in_table to true.
- If the data is in prose text (not in a table), set in_table to false.
- If has_data is false, set in_table to false.
- Provide a brief explanation justifying your decision.
- Structure your response as a JSON object with "explanation", "has_data", and "in_table" fields.
"""


ATTRIBUTE_PROVENANCE_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to determine if a single page of text from a research paper contains data for a described measurement attribute.

Guidelines:
- You will be provided with a single page of text from a research paper and a description of a measurement attribute.
- Set has_data to true only if the page contains directly reported numerical measurements for the described attribute.
- Set has_data to false if the attribute is not mentioned on the page, or if there are no numerical measurements for it.
- Set has_data to false if the data reported only contains values for parameter estimates or measures of fit for a statistical model.
- If has_data is true and the data appears within a table on the page, set in_table to true. If the data is in prose text (not in a table), set in_table to false.
- If has_data is false, set in_table to false.
- Provide a brief explanation justifying your decision.
- Structure your response as a JSON object with "explanation", "has_data", and "in_table" fields.
"""


MEASUREMENT_EVENT_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to identify all distinct measurement events for a given entity and attribute on a page of text from a research paper.

Guidelines:
- You will be provided with a single page of text from a research paper, a description of an entity, and a description of a measurement attribute.
- A measurement event is a specific instance of the attribute being measured for the entity — distinguished by contextual factors such as date, method, treatment condition, or other identifying information.
- For each distinct measurement event you identify, populate its fields as completely as the page text allows. Use ONLY information explicitly stated on the page. Do not infer, guess, or derive any field value. If a field value is not explicitly stated, set it to None.
- IMPORTANT: Do NOT produce multiple events that differ only by having a subset of the same information. Each event must capture as much identifying context as the text provides for that measurement. If date, method, and substrate are all stated for a particular measurement, output one event with all three fields populated — not three separate events for each possible subset.
- If the page contains no directly reported numerical measurements for the described entity and attribute, return an empty items list.
- Structure your response as a JSON object with an "items" list.
"""


EXTRACT_TEXT_VALUE_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to determine if a page of text from a research paper contains a measured value for a given (entity, attribute, event) item, and if so, to extract it.

Guidelines:
- If the page does not contain a relevant measurement, set has_value to false and leave value and units as null.
- If a measurement is found, set has_value to true, extract the value exactly as it appears in the context, and extract the units of measurement.
- Copy the value exactly as it appears — do not convert, round, or modify it.
- Do not include uncertainty measures, confidence intervals, or range bounds in the value field.
- If there are multiple types of values reported (e.g., mean, min, max), extract the mean or central value unless the attribute description directs otherwise.
- Give the value only in the value field, and do not include any units of measurement, descriptors, or explanation.
- Structure your response as a JSON object with "explanation", "has_value", "value", and "units" fields.
"""


EXTRACT_TABLE_VALUE_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to determine if an HTML table from a research paper contains a measured value for a given (entity, attribute, event) item, and if so, to identify the row and column needed to locate it.

You will be provided with:
- The full HTML table
- A list of row names in the table
- A list of column names in the table
- A description of the entity, attribute, and event to find

Guidelines:
- If the table does not contain a relevant measurement, set has_value to false and leave row_index, column_index, and units as null.
- If a measurement is found, set has_value to true, and provide the exact row_index name and column_index name needed to locate the cell.
- Your row_index and column_index must exactly match names from the provided lists.
- Also extract the units of measurement if identifiable from the table headers or context.
- If there are multiple types of values reported (e.g., mean, min, max), choose the row/column for the mean or central value unless the attribute description directs otherwise.
- Structure your response as a JSON object with "explanation", "has_value", "row_index", "column_index", and "units" fields.
"""


STANDARDIZE_MEASUREMENTS_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to assist in the data collection process by standardizing measurement values and units extracted from a research paper. You will be given the source text where the measurement was extracted from (either a page of prose text or an HTML table), a description of the specific entity and attribute, a list of available (preferred) units for the attribute, and an extracted measurement value with units. Use the provided source text to verify the original measurement context. Your task is to standardize both the extracted value and the units according to the following guidelines.

Value standardization guidelines:
- For numerical values associated with uncertainty measures (e.g., ± values, confidence intervals), report only the central value without any uncertainty information, unless the queried attribute specifically directs otherwise.
- For numerical values reported as ranges with a central value (e.g., 5 (3-7)) report only the central value, unless the queried attribute specifically directs otherwise.
- For numerical values reported as ranges without a central value (e.g., 3-7), choose the single value which best fits the queried attribute.
- For numerical values reported with inequalities (e.g., < 5), report the numerical value only without any additional formatting.
- For numerical values which are reported with a unit of measurement or other descriptor, convert the value to a standardized numerical format without any units or descriptors.
- If the value does not need any standardization (i.e. is a single numerical or descriptive value), return the value exactly as it is given.

Units standardization guidelines:
- If the extracted units are a notational variant of one of the available units (e.g., "mg/L" vs "mg L⁻¹", "μm" vs "um", "°C" vs "degrees C"), return the best matching entry from the available units list. You may infer notational variants based on common scientific usage.
- If the extracted units are not a notational variant of any available unit (i.e., they would require unit conversion to match, or there are no available units listed), return the extracted units unchanged.
- If the extracted units are null (not reported), return null.

- Provide a brief explanation of what standardization was applied to both value and units (or why none was needed).
- Structure your response as a JSON object with "explanation", "value", and "units" fields.
"""


# --------------------------------------------
# Ablation Prompts
# --------------------------------------------


# Ablation 1: Direct extraction (no pipeline structure)
DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to extract a complete list of measurement records from a research paper document in a single pass. Each record captures an entity, an attribute for measurement, the conditions of a specific measurement event, and its value.

Guidelines:
- You will be provided with dataset-specific extraction instructions describing the entities to identify, the target attributes, and the measurement event fields, along with the full document text.
- Identify all entities of the specified type present in the document, following the entity identification rules in the dataset-specific instructions.
- For each identified entity, identify all distinct measurement events and all attributes for which a direct numerical measurement is reported.
- Return one item per (entity, attribute, event) combination where a direct numerical measurement exists.
- Only include items where a direct numerical measurement is reported — omit absent data, model parameters, goodness-of-fit statistics, and qualitative descriptions.
- Extract the value exactly as it appears in the document — do not convert, round, or modify it.
- Do not include uncertainty measures, confidence intervals, or range bounds in the value field.
- If there are multiple types of values reported (e.g., mean, min, max), extract the mean or central value unless the attribute description directs otherwise.
- Give the value only in the value field; do not include any units, descriptors, or explanation there.
- For units, use the best fitting option from the attribute's listed preferred units if possible; otherwise specify the unit exactly as it appears in the text. Set units to null if no units are reported.
- Do NOT infer, guess, or derive any field value. If a field is not explicitly stated in the document, set it to null.
- Structure your response as a JSON object with an "items" list, where each item contains the entity fields, event fields, and "attribute", "value", and "units" fields as specified in the dataset-specific instructions.
"""



# Ablation 2: Combined (entity, attribute) pair provenance
ENTITY_ATTRIBUTE_PROVENANCE_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to determine if a single page of text from a research paper contains data for a described (entity, attribute) pair.

Guidelines:
- You will be provided with a single page of text from a research paper, a description of an entity, and a description of a measurement attribute.
- Set has_data to true only if the page contains directly reported numerical measurements associated with BOTH the described entity AND the described attribute simultaneously — i.e., the measurement of that attribute is reported for that specific entity.
- Set has_data to false if the entity is not mentioned on the page, if the attribute is not mentioned on the page, if there are no numerical measurements for the attribute, or if the measurements found are not clearly associated with the described entity.
- Set has_data to false if the data reported only contains values for parameter estimates or measures of fit for a statistical model.
- If has_data is true and the data appears within a table on the page, set in_table to true.
- If the data is in prose text (not in a table), set in_table to false.
- If has_data is false, set in_table to false.
- Provide a brief explanation justifying your decision.
- Structure your response as a JSON object with "explanation", "has_data", and "in_table" fields.
"""

# Ablation 3: Generated entity-attribute provenance
FULL_CONTEXT_PROVENANCE_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to identify all locations in a full research paper document where a directly reported numerical measurement exists for a described entity and attribute.

Guidelines:
- You will be given the full document text, a description of an entity, and a description of a measurement attribute.
- Identify every location in the document where a direct numerical measurement for BOTH the described entity AND the described attribute appears together.
- For each location found, report the page number and the table number (if applicable).
- Determine the page number using the closest preceding <page number="x"> tag in the document, and report x as the page_number.
- If the data appears within a table, determine the table number using the enclosing <table number="x"> tag and report x as the table_number. If the data is in prose text (not in a table), set table_number to null.
- Only report locations with directly reported numerical measurements. Do not include locations where only model parameters, goodness-of-fit statistics, or qualitative descriptions appear.
- If no qualifying locations are found, return an empty items list.
- Provide a brief explanation for each reported location.
- Structure your response as a JSON object with an "items" list, where each item has "explanation", "page_number", and "table_number" fields.
"""


# Ablation 4: Full-context measurement event resolution
MEASUREMENT_EVENT_INSTRUCTIONS_FULL_CONTEXT = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to identify all distinct measurement events for a given entity and attribute in a research paper.

Guidelines:
- You will be provided with the full document text from a research paper, a description of an entity, and a description of a measurement attribute.
- A measurement event is a specific instance of the attribute being measured for the entity — distinguished by contextual factors such as date, method, treatment condition, or other identifying information.
- For each distinct measurement event you identify, populate its fields as completely as the document text allows. Use ONLY information explicitly stated in the text. Do not infer, guess, or derive any field value. If a field value is not explicitly stated, set it to None.
- IMPORTANT: Do NOT produce multiple events that differ only by having a subset of the same information. Each event must capture as much identifying context as the text provides for that measurement. If date, method, and substrate are all stated for a particular measurement, output one event with all three fields populated — not three separate events for each possible subset.
- If the document contains no directly reported measuremente events for the described entity and attribute, return an empty items list.
- Structure your response as a JSON object with an "items" list.
"""


# Ablation 4: Full-context text value extraction
EXTRACT_TEXT_VALUE_INSTRUCTIONS_FULL_CONTEXT = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to determine if a research paper document contains a measured value for a given (entity, attribute, event) item, and if so, to extract it.

Guidelines:
- You will be given the full document text.
- If the document does not contain a relevant measurement, set has_value to false and leave value and units as null.
- If a measurement is found, set has_value to true, extract the value exactly as it appears in the context, and extract the units of measurement.
- Copy the value exactly as it appears — do not convert, round, or modify it.
- Do not include uncertainty measures, confidence intervals, or range bounds in the value field.
- If there are multiple types of values reported (e.g., mean, min, max), extract the mean or central value unless the attribute description directs otherwise.
- Give the value only in the value field, and do not include any units of measurement, descriptors, or explanation.
- Structure your response as a JSON object with "explanation", "has_value", "value", and "units" fields.
"""


# Ablation 4: Full-context table value extraction
EXTRACT_TABLE_VALUE_INSTRUCTIONS_FULL_CONTEXT = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to determine if a table within a research paper document contains a measured value for a given (entity, attribute, event) item, and if so, to identify the row and column needed to locate it.

Guidelines:
- You will be given the full document text.
- If the document does not contain a relevant measurement in a table, set has_value to false and leave row_index, column_index, and units as null.
- If a measurement is found, set has_value to true, and provide the exact row_index name and column_index name needed to locate the cell.
- Your row_index and column_index must exactly match names from the target table.
- Also extract the units of measurement if identifiable from the table headers or context.
- If there are multiple types of values reported (e.g., mean, min, max), choose the row/column for the mean or central value unless the attribute description directs otherwise.
- Structure your response as a JSON object with "explanation", "has_value", "row_index", "column_index", and "units" fields.
"""


# Ablation 5: Direct table value extraction (no row/column indexing)
EXTRACT_TABLE_VALUE_DIRECT_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to determine if an HTML table from a research paper contains a measured value for a given (entity, attribute, event) item, and if so, to extract it directly.

Guidelines:
- If the table does not contain a relevant measurement, set has_value to false and leave value and units as null.
- If a measurement is found, set has_value to true, extract the value exactly as it appears in the table, and extract the units of measurement.
- Copy the value exactly as it appears — do not convert, round, or modify it.
- Do not include uncertainty measures, confidence intervals, or range bounds in the value field.
- If there are multiple types of values reported (e.g., mean, min, max), extract the mean or central value unless the attribute description directs otherwise.
- Give the value only in the value field, and do not include any units of measurement, descriptors, or explanation.
- Structure your response as a JSON object with "explanation", "has_value", "value", and "units" fields.
"""


# Ablation 6: No explanation prompts (for all of the above)
DETECT_ATTRIBUTES_BATCH_INSTRUCTIONS_NO_EXPLANATIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to evaluate ALL of the listed attributes at once against context from a research paper, determining whether each attribute has any directly reported numerical measurements anywhere in the document.

Guidelines:
- You MUST return one item per attribute, using the EXACT attribute name provided. Do not rename, skip, or add attributes.
- Set detected to false if the given attribute does not appear in the context.
- Set detected to false if the context does not explicitly provide data for the given attribute.
- Set detected to false if the data reported is not a direct numerical measurement.
- Set detected to false if the data reported only contains values for parameter estimates or measures of fit for a statistical model.
- Set detected to false for cases where there is not a clear choice for a single, numerical data value.
- Set detected to true only if the context explicitly provides a direct numerical measurement for the given attribute.
- For each attribute, provide a brief explanation justifying your decision.
- When detected is true, populate the terms list with any terminology or abbreviations used in the context to refer to that attribute. Pay close attention to tables and figure captions, as these often contain abbreviations used in the main text. Do not infer, guess, or fabricate terms not explicitly present in the context.
- When detected is false, return an empty list for terms.
- Structure your response as a JSON object with an "items" list, where each item has "attribute_name", "detected", and "terms" fields.
"""

ENTITY_PROVENANCE_INSTRUCTIONS_NO_EXPLANATIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to determine if a single page of text from a research paper contains data for a described entity.

Guidelines:
- You will be provided with a single page of text from a research paper and a description of an entity.
- Set has_data to true only if the page contains directly reported numerical measurements associated with the described entity.
- Set has_data to false if the entity is not mentioned on the page, or if there are no numerical measurements for it.
- Set has_data to false if the data reported only contains values for parameter estimates or measures of fit for a statistical model.
- If has_data is true and the data appears within a table on the page, set in_table to true.
- If the data is in prose text (not in a table), set in_table to false.
- If has_data is false, set in_table to false.
- Structure your response as a JSON object with "has_data" and "in_table" fields.
"""


ATTRIBUTE_PROVENANCE_INSTRUCTIONS_NO_EXPLANATIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to determine if a single page of text from a research paper contains data for a described measurement attribute.

Guidelines:
- You will be provided with a single page of text from a research paper and a description of a measurement attribute.
- Set has_data to true only if the page contains directly reported numerical measurements for the described attribute.
- Set has_data to false if the attribute is not mentioned on the page, or if there are no numerical measurements for it.
- Set has_data to false if the data reported only contains values for parameter estimates or measures of fit for a statistical model.
- If has_data is true and the data appears within a table on the page, set in_table to true. If the data is in prose text (not in a table), set in_table to false.
- If has_data is false, set in_table to false.
- Structure your response as a JSON object with "has_data" and "in_table" fields.
"""


EXTRACT_TEXT_VALUE_INSTRUCTIONS_NO_EXPLANATIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to determine if a page of text from a research paper contains a measured value for a given (entity, attribute, event) item, and if so, to extract it.

Guidelines:
- If the page does not contain a relevant measurement, set has_value to false and leave value and units as null.
- If a measurement is found, set has_value to true, extract the value exactly as it appears in the context, and extract the units of measurement.
- Copy the value exactly as it appears — do not convert, round, or modify it.
- Do not include uncertainty measures, confidence intervals, or range bounds in the value field.
- If there are multiple types of values reported (e.g., mean, min, max), extract the mean or central value unless the attribute description directs otherwise.
- Give the value only in the value field, and do not include any units of measurement, descriptors, or explanation.
- Structure your response as a JSON object with "has_value", "value", and "units" fields.
"""

EXTRACT_TABLE_VALUE_INSTRUCTIONS_NO_EXPLANATIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to determine if an HTML table from a research paper contains a measured value for a given (entity, attribute, event) item, and if so, to identify the row and column needed to locate it.

You will be provided with:
- The full HTML table
- A list of row names in the table
- A list of column names in the table
- A description of the entity, attribute, and value to find

Guidelines:
- If the table does not contain a relevant measurement, set has_value to false and leave row_index, column_index, and units as null.
- If a measurement is found, set has_value to true, and provide the exact row_index name and column_index name needed to locate the cell.
- Your row_index and column_index must exactly match names from the provided lists.
- Also extract the units of measurement if identifiable from the table headers or context.
- If there are multiple types of values reported (e.g., mean, min, max), choose the row/column for the mean or central value unless the attribute description directs otherwise.
- Structure your response as a JSON object with "has_value", "row_index", "column_index", and "units" fields.
"""


# --------------------------------------------
# LLM as Judge Prompts
# --------------------------------------------

JUDGE_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews.

You will be given:
1) In ## CONTEXT: A text document representing a page from a research paper.
2) In ## QUERY: a description of an extracted entity, a target attribute for measurement, information about its measurement event, and the corresponding extracted value with its units.

Your task: decide whether this extraction is correct — that is, whether the extracted value (with its units) is actually reported in the document for the specified entity, attribute, and (if applicable) measurement event.

Respond 'true' ONLY if ALL of the following hold:
(A) The entity is real and distinct. It corresponds to an actual instance of the specified entity type in the document context — not something hypothetical, aggregated, or ambiguously described. An entity may be identified by an abbreviation or code; match it against the name or identifiers fields in the extracted entity description.
(B) The value is explicitly present within the context. Numerical identity is required: only trivial surface formatting differences are acceptable (e.g., 10 vs 10.0, 1,000 vs 1000, 1e-3 vs 0.001). Do not accept values that differ by rounding, averaging, unit conversion, or any other transformation.
(C) The value is assigned to the correct entity. The document makes clear the value belongs to the described entity, not to a different site, condition, subgroup, or an aggregate that includes other entities.
(D) The value is assigned to the correct attribute. The value corresponds to the specified attribute, not to a similarly named variable, proxy, or different operationalization of the same concept.
(E) The value is a direct measurement. It is a raw measurement or descriptive summary statistic of measurements (mean, median, SD, min, max, count, proportion, total) — not a model output (coefficient, odds ratio, p-value, CI bound, test statistic, goodness-of-fit metric, or correlation). It must appear as a standalone quantity: do not accept a value found only as an endpoint of a reported range (e.g., "ranged from 6.5 to 7.2") unless the target attribute specifically describes that endpoint.
(F) The units are correct. The units match those reported in the document for that value. Accept notational variants (e.g., "mg/L" vs "mg L⁻¹", "μm" vs "um", "°C" vs "degrees C"). Do not accept units that would require conversion to match (e.g., mg/L vs g/L, ha vs m²).
(G) If the context reports multiple values for the same entity and attribute (e.g. at different dates or times), the value extracted corresponds to the measurement event specified in the QUERY.

Respond 'false' if ANY criterion is not met, or if the evidence is ambiguous. Prefer 'false' when uncertain — the goal is high precision.

Respond with exactly one token: 'true' or 'false' (lowercase, no punctuation).
"""

# --------------------------------------------
# Table Cleaning
# --------------------------------------------

CLEAN_TABLE_INSTRUCTIONS = """
# Table Normalization Prompt

You are a document reconstruction engine. You will receive:

1. An image of a single PDF page from a research paper.
2. The OCR-parsed text of that page, with HTML tables inline at their original positions within `<table number="i">...</table>` tags.

Your task: reproduce the OCR text exactly as given, but replace each `<table>` block with a cleaned, normalized version. Do not modify any text outside of `<table>` tags.

## Table Normalization Rules

### Goal

Transform each table so that any cell can be uniquely addressed by a (row name, column name) pair and contains exactly one value. A downstream LLM will extract data by selecting a row name and a column name, so both axes must be meaningful and unambiguous.

This requires three properties:

1. **Every row has a meaningful, unique name.**
2. **Every column has a meaningful, unique name.**
3. **Every cell contains exactly one value.**

Only restructure a table when it violates one or more of these properties. If a table already satisfies all three, preserve its structure.

### Row Names (Index Column)

- You MUST create a create a new column named `index` containing a unique, meaningful identifier for each row.
- Populate the `index` column using one or more columns from the original table that uniquely identify each row. Prefer named entities (e.g., object names, study names, compound names, model names) over numerical IDs.
- If multiple columns are needed for uniqueness, combine them as a Python tuple: `('Category A', 'Sub-category 1')`.
- Every index value must be unique across all rows in the table.
- Columns used solely to construct the index may be removed if they carry no additional information beyond what the index captures. Columns that carry additional information should be retained.
- If the original table has no clear entity identifiers, use numerical row numbers as a last resort.

NOTE: This is a critical step for machine readability. Creating an 'index' column is what allows downstream code to refer to specific rows, so it must be populated with meaningful, unique identifiers.


### Column Names

- Every column must have a unique, descriptive name.
- Use lowercase with underscores (e.g., `dose_mg`, `response_rate`).
- Where feasible, incorporate units into the column name (e.g., `dose_mg` rather than `Dose (mg)`).
- If the original table has multi-level headers, concatenate the levels with underscores into a single name (e.g., `blood_pressure_systolic`).

### When to Restructure

Restructure a table only when its current layout prevents clean (row name, column name) addressing. The primary case:

- **Column headers encode both an entity and an attribute.** If a set of column headers can be factored as `{attribute} × {entity/condition}` (e.g., `dose_mg` repeated under `Drug A` and `Drug B`), the column names are not independent attributes — they bundle identity information that belongs in the row names. Unpivot the table so the entity/condition becomes part of the row index, and the columns become pure attribute names.
- **Heuristic:** If two or more columns would have the same attribute name once you strip out an entity or condition label, the table should be unpivoted along that entity/condition axis.
- Do NOT restructure tables where every column is already a distinct, independently meaningful attribute — even if the table looks "wide."

When restructuring involves unpivoting (melting) a wide table:

- Incorporate the new entity/condition label into the index as a tuple element, and also preserve it as its own column.
- If there are multiple entity/condition axes (e.g., `{attribute} × {drug} × {time_point}`), each should become its own column and tuple element in the index.
- Columns that are not part of the repeated group (e.g., metadata like name or category) should be carried through unchanged to every new row.

### Atomic Cell Values

- If a cell contains a main value bundled with a range, interval, or uncertainty (e.g., `3.5 (2.1–4.8)` or `12.3 ± 0.5`), split it into separate columns.
- Name the new columns descriptively based on context: e.g., `feature_mean` and `feature_ci`. If the statistic type is unclear, use `feature_val`, `feature_aux_1`, `feature_aux_2`, etc.

### Captions

- Table captions in the OCR text typically appear outside the `<table>` tags as free-standing text (e.g., "Table 1: Patient demographics..."). Move this caption text from its original position into `<caption>...</caption>` tags at the start of the corresponding `<table>` block. Remove the caption from its original location so it is not duplicated.
- After the original caption text, append a brief note describing any structural changes made (e.g., melting, transposing, column renaming) needed to interpret the new version of the table. If no changes were made, do not append anything.
- If a table gives units of measurement for any attribute, ensure that those units are clearly described in the caption of the output table -- even if they are also in the column names.

### Data Integrity

- Preserve all original data values. Your priority is to restructure and add indexing information, not to alter content.
- Only correct clear OCR errors or formatting artifacts (e.g., broken Unicode, misaligned cells) — use the page image as ground truth.
- Output tables must be valid HTML within `<table>...</table>` tags.
- If the table has a numbered tag, keep the same number in your output (e.g., `<table number="1">` should remain `<table number="1">`).
- If a table displays units for measurement, those units must be clearly presented in the caption AND indicated in the column names.

## Example

**Input table:**

```html
<table number="1">
<tr><th></th><th colspan="2">Drug A</th><th colspan="2">Drug B</th></tr>
<tr><th>Patient</th><th>Dose (mg)</th><th>Response</th><th>Dose (mg)</th><th>Response</th></tr>
<tr><td>P-001</td><td>50</td><td>0.82</td><td>75</td><td>0.91</td></tr>
<tr><td>P-002</td><td>50</td><td>0.67</td><td>75</td><td>0.73</td></tr>
</table>
```

**Why restructure:** The column headers factor as `{dose_mg, response} × {Drug A, Drug B}`. `Drug A` and `Drug B` are entities encoded in the column headers. A downstream LLM looking for P-001's dose under Drug A would more naturally address `row="('P-001', 'Drug A')", column="dose_mg"` than `row="P-001", column="dose_mg_drug_a"`.

**Output table:**

```html
<table number="1">
<caption>Patient drug response data measured across two drug treatments. Dose is measured in milligrams (mg). Response is a dimensionless score. Restructured from wide format: drug type (originally in column groups) moved to rows for clearer entity-attribute addressing.</caption>
<tr><th>index</th><th>drug</th><th>dose_mg</th><th>response</th></tr>
<tr><td>('P-001', 'Drug A')</td><td>Drug A</td><td>50</td><td>0.82</td></tr>
<tr><td>('P-001', 'Drug B')</td><td>Drug B</td><td>75</td><td>0.91</td></tr>
<tr><td>('P-002', 'Drug A')</td><td>Drug A</td><td>50</td><td>0.67</td></tr>
<tr><td>('P-002', 'Drug B')</td><td>Drug B</td><td>75</td><td>0.73</td></tr>
</table>
```

## Output Format

Return the full page text with normalized tables inline. Do not add any commentary, preamble, or explanation outside the reproduced text.
"""