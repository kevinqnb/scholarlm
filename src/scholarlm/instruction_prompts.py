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
- Set detected to true only if the context explicitly provides a direct numerical measurement for the given attribute.
- For each attribute, provide a brief explanation justifying your decision.
- When detected is true, populate a list called "terms" with any terminology or abbreviations used in the context to refer to that attribute. Pay close attention to tables and figure captions, which often contain abbreviations. Do not infer, guess, or fabricate terms not explicitly present in the context.
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
- You will be provided with a single page of text from a research paper, a description of an entity, and a description of an attribute to be measured.
- A measurement event is a specific instance of the attribute being measured for the entity — distinguished by contextual factors such as date, method, treatment condition, or other identifying information.
- There may be multiple distinct measurement events for the same entity and attribute.
- You will also be given a list of possible measurement event fields (e.g., date, method, treatment, substrate).
- For each distinct measurement event you identify, populate the given fields as completely as the page text allows. Use ONLY information explicitly stated on the page. Do not infer, guess, or derive any field value. If a field value is not explicitly stated, set it to None.
- IMPORTANT: Do NOT produce multiple events that differ only by having a subset of the same information. Each event must capture as much identifying context as the text provides for that measurement. If date, method, and substrate are all stated for a particular measurement, output one event with all three fields populated — not three separate events for each possible subset.
- If the page contains no directly reported numerical measurements for the described entity and attribute, return an empty items list.
- Structure your response as a JSON object with an "items" list.
"""


EXTRACT_TEXT_VALUE_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to determine if a page of text from a research paper contains a measurement value for a given attribute and entity, and if so, to extract it.

Guidelines:
- A measurement is relevant only if it is directly associated with the given entity, attribute, and event.
- If the page does not contain a relevant measurement, set the has_value field to false and leave the value and units fields as null.
- If a relevant measurement is found, set has_value to true and proceed to extract the value and units.
- Copy the value of the measurement directly to the value field of your response — do not convert, round, or modify it in any way.
- Do not include any extra uncertainty information, confidence intervals, range bounds, descriptors, or explanations in the value field.
- If there are multiple types of relevant measurements reported (e.g., mean, min, max), extract the mean or central value unless the attribute description unambiguously directs otherwise.
- Copy the units of measurement directly to the units field of your response — do not convert or modify them in any way. If no units are reported, set units to null.
- Structure your response as a JSON object with "explanation", "has_value", "value", and "units" fields.
"""


EXTRACT_TABLE_VALUE_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to determine if an HTML table from a research paper contains a measurement value for a given attribute and entity, and if so, to identify the row and column needed to locate it.

You will be provided with:
- The full HTML table
- A list of row names in the table
- A list of column names in the table
- A description of the entity, attribute, and event to find

Guidelines:
- A measurement is relevant only if it is directly associated with the given entity, attribute, and event.
- If the table does not contain a relevant measurement, set has_value to false and leave row_index, column_index, and units as null.
- If a relevant measurement is found, set has_value to true and proceed to extract the row index, column index, and units.
- A row or column index is relevant if it locates the cell containing the measurement for the given entity, attribute, and event.
- If there are multiple types of relevant measurements reported (e.g., mean, min, max) extract the indices which locate the mean or central value, unless the attribute description unambiguously directs otherwise.
- Copy the name for the relevant row index directly to the row_index field of your response — do not convert or modify it in any way.
- Copy the name for the relevant column index directly to the column_index field of your response — do not convert or modify it in any way.
- Copy the units of measurement directly to the units field of your response — do not convert or modify them in any way. If no units are reported, set units to null.
- Structure your response as a JSON object with "explanation", "has_value", "row_index", "column_index", and "units" fields.
"""


STANDARDIZE_MEASUREMENTS_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to assist in the data collection process by standardizing measurement values and units. You will be queried with a complete description of an extracted measurement, as well as a list of available (preferred) units to report in. Your task is to standardize both the extracted measurement's value and units according to the following guidelines.

Value standardization guidelines:
- For numerical values associated with uncertainty measures (e.g., ± values, confidence intervals), report only the central value without any uncertainty information, unless the queried attribute specifically directs otherwise.
- For numerical values reported as ranges with a central value (e.g., 5 (3-7)) report only the central value, unless the queried attribute specifically directs otherwise.
- For numerical values reported as ranges without a central value (e.g., 3-7), choose the single value which best fits the queried attribute.
- For numerical values reported with inequalities (e.g., < 5), report the numerical value only without any additional formatting.
- For numerical values which are reported with attached units of measurement or other descriptor, convert the value to a standardized numerical format without any attached units or descriptors.
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
DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to extract a complete list of measurement records from a research paper document in a single pass. Each record captures an entity, the conditions of a specific measurement event, the attribute measured, and its value.

Guidelines:
- You will be provided with dataset-specific extraction instructions describing the entities to identify, the measurement event fields, and the target attributes, along with the full document text.
- Identify all entities of the specified type present in the document, following the entity identification rules in the dataset-specific instructions.
- For each identified entity, identify all distinct measurement events and all attributes for which a direct numerical measurement is reported.
- Return one item per (entity, measurement event, attribute) combination where a direct numerical measurement exists.
- Only include items where a direct numerical measurement is reported — omit absent data, model parameters, goodness-of-fit statistics, and qualitative descriptions.
- Extract the value exactly as it appears in the document — do not convert, round, or modify it.
- Do not include uncertainty measures, confidence intervals, or range bounds in the value field.
- If there are multiple types of values reported (e.g., mean, min, max), extract the mean or central value unless the attribute description directs otherwise.
- Give the value only in the value field; do not include any units, descriptors, or explanation there.
- For units, use the best fitting option from the attribute's listed preferred units if possible; otherwise specify the unit exactly as it appears in the text. Set units to null if no units are reported.
- Do NOT infer, guess, or derive any field value. If a field is not explicitly stated in the document, set it to null.
- Structure your response as a JSON object with an "items" list, where each item contains the entity fields, event fields, and "attribute", "value", and "units" fields as specified in the dataset-specific instructions.
"""


# Ablation 2: Combined (entity, attribute) provenance
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


# Ablation 4: Full-context text value extraction
EXTRACT_TEXT_VALUE_INSTRUCTIONS_FULL_CONTEXT = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to determine if a research paper document contains a measured value for a given attribute and entity, and if so, to extract it.

Guidelines:
- You will be given the full document text.
- A measurement is relevant only if it is directly associated with the given entity, attribute, and event.
- If the document does not contain a relevant measurement, set the has_value field to false and leave the value and units fields as null.
- If a relevant measurement is found, set has_value to true and proceed to extract the value and units.
- Copy the value of the measurement directly to the value field of your response — do not convert, round, or modify it in any way.
- Do not include any extra uncertainty information, confidence intervals, range bounds, descriptors, or explanations in the value field.
- If there are multiple types of relevant measurements reported (e.g., mean, min, max), extract the mean or central value unless the attribute description unambiguously directs otherwise.
- Copy the units of measurement directly to the units field of your response — do not convert or modify them in any way. If no units are reported, set units to null.
- Structure your response as a JSON object with "explanation", "has_value", "value", and "units" fields.
"""


# Ablation 4: Full-context table value extraction
EXTRACT_TABLE_VALUE_INSTRUCTIONS_FULL_CONTEXT = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to determine if a research paper document contains a measured value for a given attribute and entity in a table, and if so, to identify the row and column needed to locate it.

Guidelines:
- You will be given the full document text.
- A measurement is relevant only if it is directly associated with the given entity, attribute, and event.
- If the document does not contain a relevant measurement in a table, set has_value to false and leave row_index, column_index, and units as null.
- If a relevant measurement is found, set has_value to true and proceed to extract the row index, column index, and units.
- A row or column index is relevant if it locates the cell containing the measurement for the given entity, attribute, and event.
- If there are multiple types of relevant measurements reported (e.g., mean, min, max), extract the indices which locate the mean or central value, unless the attribute description unambiguously directs otherwise.
- Copy the name for the relevant row index directly to the row_index field of your response — do not convert or modify it in any way.
- Copy the name for the relevant column index directly to the column_index field of your response — do not convert or modify it in any way.
- Copy the units of measurement directly to the units field of your response — do not convert or modify them in any way. If no units are reported, set units to null.
- Structure your response as a JSON object with "explanation", "has_value", "row_index", "column_index", and "units" fields.
"""


# Ablation 4: Full-context measurement event resolution
MEASUREMENT_EVENT_INSTRUCTIONS_FULL_CONTEXT = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to identify all distinct measurement events for a given entity and attribute in a research paper.

Guidelines:
- You will be provided with the full document text from a research paper, a description of an entity, and a description of a measurement attribute.
- A measurement event is a specific instance of the attribute being measured for the entity — distinguished by contextual factors such as date, method, treatment condition, or other identifying information.
- There may be multiple distinct measurement events for the same entity and attribute.
- You will also be given a list of possible measurement event fields (e.g., date, method, treatment, substrate).
- For each distinct measurement event you identify, populate the given fields as completely as the document text allows. Use ONLY information explicitly stated in the document. Do not infer, guess, or derive any field value. If a field value is not explicitly stated, set it to None.
- IMPORTANT: Do NOT produce multiple events that differ only by having a subset of the same information. Each event must capture as much identifying context as the text provides for that measurement. If date, method, and substrate are all stated for a particular measurement, output one event with all three fields populated — not three separate events for each possible subset.
- If the document contains no directly reported numerical measurements for the described entity and attribute, return an empty items list.
- Structure your response as a JSON object with an "items" list.
"""


# Ablation 5: Direct table value extraction (no row/column indexing)
EXTRACT_TABLE_VALUE_DIRECT_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to determine if an HTML table from a research paper contains a measured value for a given attribute and entity, and if so, to extract it directly.

Guidelines:
- A measurement is relevant only if it is directly associated with the given entity, attribute, and event.
- If the table does not contain a relevant measurement, set the has_value field to false and leave the value and units fields as null.
- If a relevant measurement is found, set has_value to true and proceed to extract the value and units.
- Copy the value of the measurement directly to the value field of your response — do not convert, round, or modify it in any way.
- Do not include any extra uncertainty information, confidence intervals, range bounds, descriptors, or explanations in the value field.
- If there are multiple types of relevant measurements reported (e.g., mean, min, max), extract the mean or central value unless the attribute description unambiguously directs otherwise.
- Copy the units of measurement directly to the units field of your response — do not convert or modify them in any way. If no units are reported, set units to null.
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
- When detected is true, populate a list called "terms" with any terminology or abbreviations used in the context to refer to that attribute. Pay close attention to tables and figure captions, which often contain abbreviations. Do not infer, guess, or fabricate terms not explicitly present in the context.
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


EXTRACT_TEXT_VALUE_INSTRUCTIONS_NO_EXPLANATIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to determine if a page of text from a research paper contains a measured value for a given attribute and entity, and if so, to extract it.

Guidelines:
- A measurement is relevant only if it is directly associated with the given entity, attribute, and event.
- If the page does not contain a relevant measurement, set the has_value field to false and leave the value and units fields as null.
- If a relevant measurement is found, set has_value to true and proceed to extract the value and units.
- Copy the value of the measurement directly to the value field of your response — do not convert, round, or modify it in any way.
- Do not include any extra uncertainty information, confidence intervals, range bounds, descriptors, or explanations in the value field.
- If there are multiple types of relevant measurements reported (e.g., mean, min, max), extract the mean or central value unless the attribute description unambiguously directs otherwise.
- Copy the units of measurement directly to the units field of your response — do not convert or modify them in any way. If no units are reported, set units to null.
- Structure your response as a JSON object with "has_value", "value", and "units" fields.
"""

EXTRACT_TABLE_VALUE_INSTRUCTIONS_NO_EXPLANATIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to determine if an HTML table from a research paper contains a measured value for a given attribute and entity, and if so, to identify the row and column needed to locate it.

You will be provided with:
- The full HTML table
- A list of row names in the table
- A list of column names in the table
- A description of the entity, attribute, and event to find

Guidelines:
- A measurement is relevant only if it is directly associated with the given entity, attribute, and event.
- If the table does not contain a relevant measurement, set has_value to false and leave row_index, column_index, and units as null.
- If a relevant measurement is found, set has_value to true and proceed to extract the row index, column index, and units.
- A row or column index is relevant if it locates the cell containing the measurement for the given entity, attribute, and event.
- If there are multiple types of relevant measurements reported (e.g., mean, min, max), extract the indices which locate the mean or central value, unless the attribute description unambiguously directs otherwise.
- Copy the name for the relevant row index directly to the row_index field of your response — do not convert or modify it in any way.
- Copy the name for the relevant column index directly to the column_index field of your response — do not convert or modify it in any way.
- Copy the units of measurement directly to the units field of your response — do not convert or modify them in any way. If no units are reported, set units to null.
- Structure your response as a JSON object with "has_value", "row_index", "column_index", and "units" fields.
"""


# --------------------------------------------
# LLM as Judge Prompts
# --------------------------------------------


# Unified judge prompt — works for both prose text and table sources.
# The page context is limited to the page(s) where the value was reported.
# 1) One or more pages from a research paper (in ## CONTEXT) — the specific page(s) where the value was reported.
JUDGE_INSTRUCTIONS_UNIFIED = """You are an expert in data extraction for systematic scientific literature reviews.

You will be given:
1) In ## CONTEXT: A full text document for a research paper.
2) In ## QUERY: a description of an extracted entity, a target attribute for measurement, and the corresponding extracted value with its units.

Your task: decide whether this extraction is correct — that is, whether the extracted value (with its units) is actually reported in the document for the specified attribute and entity.

Respond 'true' ONLY if ALL of the following hold:

(A) The entity is real and distinct. It corresponds to an actual, clearly identified instance of the specified entity type in the page context — not something hypothetical, aggregated, or ambiguously described. An entity may be identified by an abbreviation or code; match it against the name or identifiers fields in the extracted entity description.        

(B) The value is present within the context. It appears explicitly in the specified table, or in the prose text if no table is cited. Numerical identity is required: only trivial surface formatting differences are acceptable (e.g., 10 vs 10.0, 1,000 vs 1000, 1e-3 vs 0.001). Do not accept values that differ by rounding, averaging, unit conversion, or any other transformation.

(C) The value is assigned to the correct entity. The document makes clear the value belongs to the described entity, not to a different site, condition, subgroup, or an aggregate that includes other entities.

(D) The value is assigned to the correct attribute. The value corresponds to the specified attribute, not to a similarly named variable, proxy, or different operationalization of the same concept.

(E) The value is a directly reported quantity. It is a raw measurement or descriptive summary statistic (mean, median, SD, min, max, count, proportion, total) — not a model output (coefficient, odds ratio, p-value, CI bound, test statistic, goodness-of-fit metric, or correlation). It must appear as a standalone quantity: do not accept a value found only as an endpoint of a reported range (e.g., "ranged from 6.5 to 7.2") unless the target attribute specifically describes that endpoint.

(F) The units are correct. The units match those reported in the document for that value. Accept only trivial notational variants (e.g., "mg/L" vs "mg L⁻¹", "μm" vs "um", "°C" vs "degrees C"). Do not accept units that would require conversion to match (e.g., mg/L vs g/L, ha vs m²).

Respond 'false' if ANY criterion is not met, or if the evidence is ambiguous. Prefer 'false' when uncertain — the goal is high precision.

Respond with exactly one token: 'true' or 'false' (lowercase, no punctuation).
"""


# Variant of JUDGE_INSTRUCTIONS_UNIFIED that also receives the row name and
# column name used for table-sourced extractions, and checks their consistency.
JUDGE_INSTRUCTIONS_UNIFIED_TABLE = """You are an expert in data extraction for systematic scientific literature reviews.

You will be given:
1) One or more pages from a research paper (in ## CONTEXT) — the specific page(s) where the value was reported.
2) In ## QUERY: a description of the extracted entity instance and its type, the target attribute description and terminology, the source location where the value was reported (a specific table or prose text), and the extracted value with its units. For table-sourced extractions, the query also includes the row name and column name used to locate the value in the table.

Your task: decide whether this extraction is correct — that is, whether the extracted value (with its units) is actually reported in the document for the specified attribute and entity.

Respond 'true' ONLY if ALL of the following hold:

(A) The entity is real and distinct. It corresponds to an actual, clearly identified instance of the specified entity type in the page context — not something hypothetical, aggregated, or ambiguously described. An entity may be identified by an abbreviation or code; match it against the name or identifiers fields in the extracted entity description.

(B) The value is present within the context. It appears explicitly in the specified table, or in the prose text if no table is cited. Numerical identity is required: only trivial surface formatting differences are acceptable (e.g., 10 vs 10.0, 1,000 vs 1000, 1e-3 vs 0.001). Do not accept values that differ by rounding, averaging, unit conversion, or any other transformation.

(C) The value is assigned to the correct entity. The document makes clear the value belongs to the described entity, not to a different site, condition, subgroup, or an aggregate that includes other entities.

(D) The value is assigned to the correct attribute. The value corresponds to the specified attribute, not to a similarly named variable, proxy, or different operationalization of the same concept.

(E) The value is a directly reported quantity. It is a raw measurement or descriptive summary statistic (mean, median, SD, min, max, count, proportion, total) — not a model output (coefficient, odds ratio, p-value, CI bound, test statistic, goodness-of-fit metric, or correlation). It must appear as a standalone quantity: do not accept a value found only as an endpoint of a reported range (e.g., "ranged from 6.5 to 7.2") unless the target attribute specifically describes that endpoint.

(F) The units are correct. The units match those reported in the document for that value. Accept only trivial notational variants (e.g., "mg/L" vs "mg L⁻¹", "μm" vs "um", "°C" vs "degrees C"). Do not accept units that would require conversion to match (e.g., mg/L vs g/L, ha vs m²).

(G) For table-sourced extractions: the row name and column name are consistent with the extraction. Respond 'false' if the row or column name maps to a different entity or attribute than described.

Respond 'false' if ANY criterion is not met, or if the evidence is ambiguous. Prefer 'false' when uncertain — the goal is high precision.

Respond with exactly one token: 'true' or 'false' (lowercase, no punctuation).
"""


_JUDGE_INSTRUCTIONS_UNIFIED_EXAMPLES = """
---

Document context (for examples):
'''
<page number="1">
Ten-liter water samples were collected in October 2016, November 2016, and December 2016 from a temperate freshwater agricultural pond in central Maryland, United States (maximum depth of ca. 3.35 meters and a surface area of ca. 0.26 ha). A ProDSS digital sampling system was used to measure, in triplicate: the water temperature (°C), conductivity (SPC uS/cm), pH, dissolved oxygen (%), and turbidity (FNU).

<table number="1">
  <tr>
    <th>Water property</th>
    <th>October</th>
    <th>November</th>
    <th>December</th>
  </tr>
  <tr>
    <td>Ambient temp. (C)</td>
    <td>17.2</td>
    <td>12.2</td>
    <td>3.9</td>
  </tr>
  <tr>
    <td>Water temp. (C)</td>
    <td>19.8</td>
    <td>10.9</td>
    <td>7.4</td>
  </tr>
  <tr>
    <td>PH</td>
    <td>7.7</td>
    <td>7.56</td>
    <td>8.08</td>
  </tr>
  <tr>
    <td>Dissolved oxygen (%)</td>
    <td>116.4</td>
    <td>96.4</td>
    <td>117.7</td>
  </tr>
  <tr>
    <td>Nitrate (mg/L)</td>
    <td>0.63</td>
    <td>0.26</td>
    <td>0.19</td>
  </tr>
  <tr>
    <td>Chloride (mg/L)</td>
    <td>13.8</td>
    <td>13.3</td>
    <td>7.9</td>
  </tr>
  <tr>
    <td>Turbidity (FNU)</td>
    <td>30.2</td>
    <td>9.6</td>
    <td>3.4</td>
  </tr>
  <tr>
    <td>Precipitation<sup>†</sup> (in.)</td>
    <td>0</td>
    <td>0</td>
    <td>0.2</td>
  </tr>
  <tr>
    <td>Conductivity (SPC uS/cm)</td>
    <td>158.9</td>
    <td>160.8</td>
    <td>167.1</td>
  </tr>
  <tr>
    <td>Oxidation/reduction (mV)</td>
    <td>189.7</td>
    <td>159.8</td>
    <td>243.9</td>
  </tr>
</table>
</page>
'''

---

Example 1 — CORRECT prose extraction with event (all criteria satisfied):

Target entity type: A distinct aquatic ecosystem — a specific pond, lake, wetland, or similar water body.
Extracted entity: {'name': 'Agricultural Pond', 'location': 'central Maryland, United States', 'ecosystem': 'pond'}

Extracted event: {'date': 'October 2016', 'additional_details': None}

Target attribute: Surface area of the water body itself, representing the horizontal area of open water or the stated ecosystem boundary. This is NOT the same as watershed area, drainage basin area, catchment area, or littoral zone area.
Attribute terminology: ['surface area', 'area']

Source: prose text

Extracted value: 0.26
Extracted units: ha

Is the extracted (entity, event, attribute, value) tuple fully valid — meaning the entity is correctly identified, the event correctly describes the measurement context, and the extracted value correctly corresponds to the target attribute for that entity at the described event, as evidenced by the document?

VERDICT: true

Explanation: The page context describes "a temperate freshwater agricultural pond in central Maryland, United States" with "a surface area of ca. 0.26 ha." The entity is valid. Surface area is a fixed physical property of the pond, so it is consistent with the October 2016 event (the event does not contradict the value). The value 0.26 appears in the prose and is directly reported as the surface area in hectares. All criteria satisfied.

---

Example 2 — INCORRECT prose extraction (invalid entity):

Target entity type: A distinct aquatic ecosystem — a specific pond, lake, wetland, or similar water body.
Extracted entity: {'name': 'Lake Merhei', 'location': 'central Maryland, United States', 'ecosystem': 'lake'}

Extracted event: {'date': 'October 2016', 'additional_details': None}

Target attribute: Surface area of the water body itself, representing the horizontal area of open water or the stated ecosystem boundary.
Attribute terminology: ['surface area', 'area']

Source: prose text

Extracted value: 0.26
Extracted units: ha

Is the extracted (entity, event, attribute, value) tuple fully valid — meaning the entity is correctly identified, the event correctly describes the measurement context, and the extracted value correctly corresponds to the target attribute for that entity at the described event, as evidenced by the document?

VERDICT: false

Explanation: The page context describes an agricultural pond in central Maryland but makes no mention of any water body called "Lake Merhei". Criterion A is not satisfied.

---

Example 3 — CORRECT table extraction with event (all criteria satisfied):

Target entity type: A distinct aquatic ecosystem — a specific pond, lake, wetland, or similar water body.
Extracted entity: {'name': 'Agricultural Pond', 'location': 'central Maryland, United States', 'ecosystem': 'pond'}

Extracted event: {'date': 'November 2016', 'additional_details': None}

Target attribute: pH of the water, i.e., the negative logarithm of the hydrogen ion activity. This is a dimensionless quantity and should refer to a measured water pH value, not soil or sediment pH.
Attribute terminology: ['pH', 'ph']

Source: Table 1

Extracted value: 7.56
Extracted units: not reported

Is the extracted (entity, event, attribute, value) tuple fully valid — meaning the entity is correctly identified, the event correctly describes the measurement context, and the extracted value correctly corresponds to the target attribute for that entity at the described event, as evidenced by the document?

VERDICT: true

Explanation: The entity (agricultural pond in central Maryland) is valid. Table 1 contains water quality parameters for this pond across three sampling months. The November 2016 row gives ph = 7.56. The value 7.56 is directly reported in the table for the correct entity and attribute at the November 2016 event. pH is dimensionless, so "not reported" is correct. All criteria satisfied.

---

Example 4 — INCORRECT table extraction (value belongs to a different attribute):

Target entity type: A distinct aquatic ecosystem — a specific pond, lake, wetland, or similar water body.
Extracted entity: {'name': 'Agricultural Pond', 'location': 'central Maryland, United States', 'ecosystem': 'pond'}

Extracted event: {'date': 'October 2016', 'additional_details': None}

Target attribute: pH of the water, i.e., the negative logarithm of the hydrogen ion activity. This is a dimensionless quantity and should refer to a measured water pH value, not soil or sediment pH.
Attribute terminology: ['pH', 'ph']

Source: Table 1

Extracted value: 19.8
Extracted units: °C

Is the extracted (entity, event, attribute, value) tuple fully valid — meaning the entity is correctly identified, the event correctly describes the measurement context, and the extracted value correctly corresponds to the target attribute for that entity at the described event, as evidenced by the document?

VERDICT: false

Explanation: In Table 1, the value 19.8 °C for October 2016 corresponds to water temperature, not pH. The correct pH value for October 2016 is 7.8 (dimensionless). Criterion E is not satisfied.

"""  # _JUDGE_INSTRUCTIONS_UNIFIED_EXAMPLES (unused — kept for reference)


# Validate extracted measurement value (text source)
JUDGE_INSTRUCTIONS_TEXT = """You are an expert in data extraction for systematic scientific literature reviews.

You will be given:
1) A complete document from a research paper (in ## CONTEXT)
2) A predefined target attribute — a description and associated terminology specifying the type of measurement to extract. This is a fixed input; do not evaluate whether the attribute description is appropriate or well-formed.
3) A candidate (entity, value) extraction: the entity identified in the document, the page where the data was found, and the value extracted from the prose text for the target attribute.
Items 2 and 3 appear in ## QUERY.

Your task: decide whether the extracted (entity, attribute, value) triplet is fully valid — meaning the entity is correctly identified and the extracted value correctly corresponds to the target attribute for that entity, as evidenced by the document.

Note: The given entity may be over-specified (e.g., it may include a date or treatment that is not explicitly represented in the table), but it should not be under-specified (e.g., it should not be missing or disagreeing on key identifying information that is needed to extract the value). The extracted value must correctly correspond to the target attribute for the specified entity, even if the entity is more specific than what is strictly necessary to identify the correct value.

Decision rules:
- Respond 'true' ONLY if ALL of the following are satisfied:
  (A) Valid entity: The described entity is a real, distinct entity of the specified type as evidenced by the document. It must not be a hypothetical, aggregated, ambiguously described, or otherwise invalid instance of the entity type.
  (B) Value presence: The extracted value appears explicitly at the specified location in the document (same page and, if a table is cited, within that table). If the value appears elsewhere in the document but not at the cited location, respond 'false'. The value must be numerically identical — i.e., both represent exactly the same number on the number line, differing only in surface formatting (e.g., 10 vs 10.0, 1000 vs 1,000, 0.001 vs 1e-3, 1/2 vs 0.5, negative sign variants). Do not accept values that require unit conversion (0.05 vs 5%), rounding (3.14 vs 3.1), arithmetic, averaging, or any other numerical transformation.
  (C) Correct assignment to entity: The document clearly indicates the value refers to the specified entity (not a different study site, species, dataset split, subgroup, scenario, treatment, timepoint, or a set/aggregate where the entity is ambiguous).
  (D) Correct assignment to attribute: The value clearly corresponds to the specified attribute (not a related metric, proxy, similarly named variable, or a different operationalization).
  (E) Direct reported quantity: The value represents a directly reported measurement or descriptive summary statistic (e.g., mean, median, standard deviation, count, proportion, total, minimum, maximum) of the attribute — not a model output (regression coefficient, odds ratio, p-value, CI bound, test statistic, goodness-of-fit metric, correlation coefficient, or tuning parameter). The value must also be reported as a standalone quantity, not solely as an endpoint of a range, interval, or bound (e.g., reject "6.5" if it only appears in "ranged from 6.5 to 7.2" and is not independently stated as the attribute's value). Exception: if the attribute itself describes a bound or endpoint (e.g., "minimum pH", "lower bound of temperature range"), then extracting the corresponding endpoint value is acceptable.
  (F) Correct units: The units associated with the extracted value must match the units reported in the document for that quantity at the specified location. Accept only trivial notational variants (e.g., "μm" vs "um", "mg/L" vs "mg L⁻¹", "°C" vs "degrees C"). Do not accept values where the extracted units differ from the reported units in a way that would require conversion (e.g., "mg/L" vs "g/L", "ha" vs "m²", "%" vs "ppm").

- Respond 'false' if ANY of the following apply:
  - The described entity does not correspond to a valid entity of the specified type as described in the document.
  - The value does not appear at the specified location (same page and, if cited, same table) in the document exactly (aside from trivial formatting differences).
  - The value appears but is tied to a different entity or attribute than the one described.
  - The value appears only as an endpoint of a range, interval, or bound (e.g., "6.5–7.2", "ranged from X to Y", "between X and Y") and is not independently reported as a standalone quantity for the specified entity and attribute — unless the attribute itself describes that bound or endpoint.
  - The value is a model output, test statistic, or derived statistical quantity (e.g., regression coefficient, odds ratio, p-value, CI bound, goodness-of-fit metric, correlation coefficient) rather than a directly reported quantity or descriptive summary.
  - The value is only implied (requires calculation, unit conversion, or deduction from other reported numbers).
  - The units associated with the extracted value do not match the units reported in the document at the specified location (aside from trivial notational variants).

Handling conflicting information:
- If the document contains conflicting values for the same entity-attribute pair at different locations, evaluate only against the value at the specified location.

Default when ambiguous:
- When the evidence is ambiguous or you are less than confident that all criteria are satisfied, default to 'false'. The goal is high precision.

Below are examples of how to evaluate candidate triplets.

---
Document context (for example):
'''
<page number="0">
Bacterial and Viral Dynamics in a Temperate Agricultural Pond... 
<page number="1">
Therefore, we aimed to assess the bacterial and viral components of a temperate agricultural pond in the Mid-Atlantic, United States during the late growing season (October–December), a time when declining temperature and nutrient levels may impact the structure and function of the microbial assemblages. Specifically, we used 16S rRNA gene and shotgun metagenomic sequencing to: (i) survey the bacterial consortium utilizing different filter pore sizes (1 and 0.2 μm); (ii) characterize the diversity and abundance of the bacteriophage within the viral community; and (iii) compare the phylogeny of pond viromes across time using the phylogenetically relevant, and biologically meaningful, Pol I protein.\n\nMATERIALS AND METHODS\n\nStudy Site and Sample Collection\nTen-liter water samples were collected in October 2016, November 2016, and December 2016 from a temperate freshwater agricultural pond in central Maryland, United States (maximum depth of ca. 3.35 meters and a surface area of ca. 0.26 ha). A Honda WX10TA (32 GPM) water pump was used to collect water 15–30 cm below the surface into a sterile polypropylene carboy. Samples were kept in the dark at 4°C and processed within 24 h of collection. In addition, a ProDSS digital sampling system (YSI, Yellow Springs, OH, United States) was used to measure, in triplicate: the water temperature (°C), conductivity (SPC uS/cm), pH, dissolved oxygen (%), oxidation/reduction potential (mV), turbidity (FNU), nitrate (mg/L), and chloride (mg/L).\n\nSample Preparation\nViral and microbial fractions were separated through peristaltic filtration followed by an iron-based flocculation and resuspension of viral particles. Two 142 mm polycarbonate in-line filter holders (Geotech, CO, United States), one equipped with a 142-mm diameter Whatman 1 μm polycarbonate filter
</page>...
'''

Example 1 — CORRECT extraction (all criteria satisfied):

Target entity type: A distinct aquatic ecosystem — a specific pond, lake, wetland, or similar water body.
Extracted entity: {name: Agricultural Pond, location: Mid-Atlantic United States, ecosystem: pond}
Target attribute: Surface area of the water body itself (not the watershed or catchment area). This should represent the horizontal area of open water or the stated ecosystem boundary at the time of measurement or description.
Attribute terminology: surface area, area
Page number: 1
Extracted value: 0.26
Extracted units: ha

VERDICT: true

Explanation: The document describes "a temperate freshwater agricultural pond in central Maryland, United States" with "a surface area of ca. 0.26 ha." The entity is valid, the value 0.26 appears at the cited location as the surface area of this pond, and it is a directly reported quantity.

---

Example 2 — INCORRECT extraction (invalid entity):

Target entity type: A distinct aquatic ecosystem — a specific pond, lake, wetland, or similar water body.
Extracted entity: {name: Lake Merhei, location: Mid-Atlantic United States, ecosystem: pond}
Target attribute: Surface area of the water body itself (not the watershed or catchment area). This should represent the horizontal area of open water or the stated ecosystem boundary at the time of measurement or description.
Attribute terminology: surface area, area
Page number: 1
Extracted value: 0.26
Extracted units: ha

VERDICT: false

Explanation: The document describes an agricultural pond in central Maryland but makes no mention of any water body called "Lake Merhei." Criterion A is not satisfied.

---

Example 3 — INCORRECT extraction (value belongs to a different attribute):

Target entity type: A distinct aquatic ecosystem — a specific pond, lake, wetland, or similar water body.
Extracted entity: {name: Agricultural Pond, location: Mid-Atlantic United States, ecosystem: pond}
Target attribute: Surface area of the water body itself (not the watershed or catchment area). This should represent the horizontal area of open water or the stated ecosystem boundary at the time of measurement or description.
Attribute terminology: surface area, area
Page number: 1
Extracted value: 3.35
Extracted units: meters

VERDICT: false

Explanation: The value 3.35 appears on page 1 but corresponds to maximum depth ("ca. 3.35 meters"), not surface area. Criterion D is not satisfied.

---

Example 4 — INCORRECT extraction (value assigned to wrong attribute):

Target entity type: A distinct aquatic ecosystem — a specific pond, lake, wetland, or similar water body.
Extracted entity: {name: Agricultural Pond, location: Mid-Atlantic United States, ecosystem: pond}
Target attribute: Maximum water depth of the ecosystem, defined as the deepest point of the water body at the time of measurement or as reported in the source. This is not the mean or average depth.
Attribute terminology: maximum depth, depth
Page number: 1
Extracted value: 0.26
Extracted units: ha

VERDICT: false

Explanation: The value 0.26 appears on page 1 but corresponds to surface area ("ca. 0.26 ha"), not maximum depth. Criterion D is not satisfied.

---

Example 5 — INCORRECT extraction (unit mismatch):

Target entity type: A distinct aquatic ecosystem — a specific pond, lake, wetland, or similar water body.
Extracted entity: {name: Agricultural Pond, location: Mid-Atlantic United States, ecosystem: pond}
Target attribute: Maximum water depth of the ecosystem, defined as the deepest point of the water body at the time of measurement or as reported in the source. This is not the mean or average depth.
Attribute terminology: maximum depth, depth
Page number: 1
Extracted value: 3.35
Extracted units: cm

VERDICT: false

Explanation: The value 3.35 appears on page 1 and corresponds to maximum depth, but the document reports the depth in meters ("ca. 3.35 meters"), not centimeters. The extracted units (cm) do not match the reported units (meters). Criterion F is not satisfied.

---

Output format:
- Respond with a single token as either 'true' or 'false' (lowercase). Do not include any additional text or punctuation.
"""


# Validate extracted measurement value (table source)
JUDGE_INSTRUCTIONS_TABLE = """You are an expert in data extraction for systematic scientific literature reviews.

You will be given:
1) A complete document from a research paper (in ## CONTEXT)
2) A predefined target attribute — a description and associated terminology specifying the type of measurement to extract. This is a fixed input; do not evaluate whether the attribute description is appropriate or well-formed.
3) A candidate (entity, row index, column index) extraction: the entity identified in the document, the table where the data was found, and the row/column indices that locate the extracted value for the target attribute. The value is the cell at the intersection of the row and column indices.
Items 2 and 3 appear in ## QUERY.

Your task: decide whether the extracted (entity, attribute, row index, column index) tuple is fully valid — meaning the entity is correctly identified and together the row index and column index correctly locate the value for that (entity, target attribute) pair in the specified table.

Note: The given entity may be over-specified (e.g., it may include a date or treatment that is not explicitly represented in the table), but it should not be under-specified (e.g., it should not be missing or disagreeing on key identifying information that is needed to map to a specific row in the table). The row and column indices must together correctly locate the value for the specified attribute and entity, even if the entity is more specific than what is strictly necessary to identify the correct row.

Decision rules:
- Respond 'true' ONLY if ALL of the following are satisfied:
  (A) Valid entity: The described entity is a real, distinct entity of the specified type as evidenced by the document. It must not be a hypothetical, aggregated, ambiguously described, or otherwise invalid instance of the entity type.
  (B) Row index presence: The specified row index appears in the specified table in the document (same page and table number). Trivial formatting variants are acceptable (e.g., minor whitespace or punctuation differences).
  (C) Row-to-entity correspondence: The row identified by the row index clearly maps to the described entity — not a different study site, species, treatment, timepoint, or other entity of the same type.
  (D) Column index presence: The specified column index appears as a column header in the specified table. Trivial formatting variants are acceptable.
  (E) Column-to-attribute correspondence: The column identified by the column index clearly maps to the described attribute — not a related metric, proxy, similarly named variable, or different operationalization.
  (F) Direct reported quantity: The cell at the intersection of the row and column contains a directly reported measurement or descriptive summary statistic (e.g., mean, median, standard deviation, count, proportion, total, minimum, maximum) — not a model output (regression coefficient, odds ratio, p-value, CI bound, test statistic, goodness-of-fit metric, correlation coefficient, or tuning parameter).
  (G) Correct units: The extracted units match the units reported in the table for that column (e.g., in the column header or caption). Accept only trivial notational variants (e.g., "μm" vs "um", "mg/L" vs "mg L⁻¹", "°C" vs "degrees C"). Do not accept units that differ in a way that would require conversion.

- Respond 'false' if ANY of the following apply:
  - The described entity does not correspond to a valid entity of the specified type as described in the document.
  - The specified row index does not appear in the specified table.
  - The row index maps to a different entity than the one described (e.g., a different site, treatment, or timepoint).
  - The specified column index does not appear in the specified table.
  - The column index maps to a different attribute than the one described (e.g., a related metric or proxy).
  - The cell at the row/column intersection contains a model output, test statistic, or derived statistical quantity rather than a directly reported measurement.
  - The extracted units do not match the units reported for that column in the table (aside from trivial notational variants).

Default when ambiguous:
- When the evidence is ambiguous or you are less than confident that all criteria are satisfied, default to 'false'. The goal is high precision.

Below are examples of how to evaluate candidate tuples.

---
Document context (for example):
'''
<page number="0">
Bacterial and Viral Dynamics in a Temperate Agricultural Pond...
<page number="1">
...Ten-liter water samples were collected in October 2016, November 2016, and December 2016 from a temperate freshwater agricultural pond in central Maryland, United States (maximum depth of ca. 3.35 meters and a surface area of ca. 0.26 ha). A ProDSS digital sampling system (YSI) was used to measure, in triplicate: the water temperature (°C), conductivity (SPC uS/cm), pH, dissolved oxygen (%), oxidation/reduction potential (mV), turbidity (FNU), nitrate (mg/L), and chloride (mg/L)...
<page number="3">
<table number="1">
<caption>Water quality parameters measured at the agricultural pond during October, November, and December 2016.</caption>
<tr><th>index</th><th>temperature_c</th><th>ph</th><th>conductivity_spc_us_cm</th><th>dissolved_oxygen_pct</th></tr>
<tr><td>October 2016</td><td>14.2</td><td>7.8</td><td>312</td><td>88.4</td></tr>
<tr><td>November 2016</td><td>9.1</td><td>7.5</td><td>298</td><td>91.2</td></tr>
<tr><td>December 2016</td><td>4.3</td><td>7.3</td><td>285</td><td>94.7</td></tr>
</table>
</page>
...
'''

Example 1 — CORRECT extraction (all criteria satisfied):

Target entity type: A distinct aquatic ecosystem — a specific pond, lake, wetland, or similar water body.
Extracted entity: {name: Agricultural Pond, location: Mid-Atlantic United States, ecosystem: pond}
Target attribute: pH of the water, i.e., the negative logarithm of the hydrogen ion activity. This is a dimensionless quantity and should refer to a measured water pH value, not soil or sediment pH.
Attribute terminology: ph, pH
Page number: 3
Table number: 1
Extracted row index: October 2016
Extracted column index: ph
Extracted units: not reported

VERDICT: true

Explanation: The entity (agricultural pond in central Maryland) is valid. The row index "October 2016" appears in Table 1 on page 3 and maps to the October 2016 observation for this pond. The column index "ph" appears in the table and corresponds to water pH. The cell value is a directly reported measurement. pH is dimensionless, so "not reported" is correct. All criteria are satisfied.

---

Example 2 — INCORRECT extraction (column maps to wrong attribute):

Target entity type: A distinct aquatic ecosystem — a specific pond, lake, wetland, or similar water body.
Extracted entity: {name: Agricultural Pond, location: Mid-Atlantic United States, ecosystem: pond}
Target attribute: pH of the water, i.e., the negative logarithm of the hydrogen ion activity. This is a dimensionless quantity and should refer to a measured water pH value, not soil or sediment pH.
Attribute terminology: ph, pH
Page number: 3
Table number: 1
Extracted row index: October 2016
Extracted column index: temperature_c
Extracted units: °C

VERDICT: false

Explanation: The row index "October 2016" is correct for this entity. However, the column index "temperature_c" maps to water temperature, not pH. The target attribute is pH and the correct column is "ph". Criterion E is not satisfied.

---

Output format:
- Respond with a single token as either 'true' or 'false' (lowercase). Do not include any additional text or punctuation.
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

### Data Integrity

- Preserve all original data values. Your priority is to restructure and add indexing information, not to alter content.
- Only correct clear OCR errors or formatting artifacts (e.g., broken Unicode, misaligned cells) — use the page image as ground truth.
- Output tables must be valid HTML within `<table>...</table>` tags.
- If the table has a numbered tag, keep the same number in your output (e.g., `<table number="1">` should remain `<table number="1">`).

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
<caption>Patient drug response data. Restructured from wide format: drug type (originally in column groups) moved to rows.</caption>
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