"""Prompt instruction strings used across `scholarlm`.
Notes:
- These are *instruction* blocks only. Query/context formatting remains in caller code.
"""

# ---------------------------------------------------------------------------
# New pipeline prompts (attribute-based)
# ---------------------------------------------------------------------------

# Step 1: Entity table enrichment
ENTITY_TABLE_ENRICHMENT_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. You have already identified a set of entities from a research paper. You are now examining a specific table from that paper to find:

1. NEW entities that were not previously identified from the full text.
2. Additional information about EXISTING entities (such as abbreviations, codes, site identifiers, treatment states, or dates) that may be visible in this table but were not captured from the full text.

Guidelines:
- The context below contains an HTML table from the paper.
- You are also given the list of entities already identified.
- Extract any NEW entities visible in the table that are not already in the list.
- For entities already in the list, extract any additional attribute values (abbreviations, codes, etc.) that the table reveals.
- Do not fabricate or infer information not explicitly present in the table.
- Structure your response as a JSON list of items following the provided schema.
- If no new entities or enrichments are found, respond with an empty list.
"""

# Step 2: Attribute detection (full context)
DETECT_ATTRIBUTE_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to determine if context from a research paper contains data for a described attribute (measurement variable) in reference to a given entity.

Guidelines:
- Answer False if the given attribute or entity do not appear in the context.
- Answer False if the context does not explicitly provide data for the given attribute and entity.
- Answer False if the data reported is not a direct numerical measurement.
- Answer False if the data reported only contains values for parameter estimates or measures of fit for a statistical model.
- Answer False for cases where there is not a clear choice for a single, numerical data value.
- Answer True only if the context explicitly provides a direct numerical value measured for the given attribute, with respect to the entity in question.
- Along with your answer, provide a brief explanation justifying the reasons for your decision.
- Structure your response as a JSON object with "explanation" and "answer" fields.
"""

# Step 2: Attribute detection (single table)
DETECT_ATTRIBUTE_TABLE_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to determine if a specific HTML table from a research paper contains data for a described attribute (measurement variable) in reference to a given entity.

Guidelines:
- Answer False if the given attribute or entity do not appear in the table.
- Answer False if the table does not explicitly provide data for the given attribute and entity.
- Answer False if the data reported is not a direct numerical measurement.
- Answer False if the data reported only contains values for parameter estimates or measures of fit for a statistical model.
- Answer True only if the table explicitly provides a direct numerical value measured for the given attribute, with respect to the entity in question.
- Along with your answer, provide a brief explanation justifying the reasons for your decision.
- Structure your response as a JSON object with "explanation" and "answer" fields.
"""

# Step 2: Attribute term identification
IDENTIFY_ATTRIBUTE_TERMS_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to assist in locating data points, by first identifying terminology used within the context. Given context from a research paper and a specific attribute type, extract any terminology or abbreviations used to directly refer to the attribute in question.

Guidelines:
- Do not include any abbreviations that refer to similar concepts or measurements, but which are not direct in relation.
- Pay close attention to tables and figure captions, as these often contain the abbreviations used in the main text.
- Do not infer, guess, or fabricate any terms or abbreviations that are not explicitly present in the context.
- Structure your response as a list of strings, for example: ['term_1', 'term_2']
- If the context does not contain any terminology or abbreviations that directly refer to the given attribute, respond with an empty list.
"""

# Step 2 (batched): Document-level full-context attribute detection with inline term identification
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

# Step 2 (batched): Document-level per-table attribute detection with inline term identification
DETECT_ATTRIBUTES_TABLE_BATCH_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to evaluate ALL of the listed attributes at once against a specific HTML table from a research paper, determining whether each attribute has any directly reported numerical measurements within this table.

Guidelines:
- You MUST return one item per attribute, using the EXACT attribute name provided. Do not rename, skip, or add attributes.
- Set detected to false if the given attribute does not appear in the table.
- Set detected to false if the table does not explicitly provide data for the given attribute.
- Set detected to false if the data reported is not a direct numerical measurement.
- Set detected to false if the data reported only contains values for parameter estimates or measures of fit for a statistical model.
- Set detected to true only if the table explicitly provides a direct numerical measurement for the given attribute.
- For each attribute, provide a brief explanation justifying your decision.
- When detected is true, populate the terms list with any terminology or abbreviations used in the table to refer to that attribute. Do not infer, guess, or fabricate terms not explicitly present in the table.
- When detected is false, return an empty list for terms.
- Structure your response as a JSON object with an "items" list, where each item has "attribute_name", "explanation", "detected", and "terms" fields.
"""

# Step 2b: Filter (entity, attribute) pairs
FILTER_ENTITY_ATTRIBUTE_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to determine if context from a research paper contains a directly reported numerical measurement for a described attribute in reference to a given entity.

Guidelines:
- Answer False if the given attribute or entity do not appear in the context.
- Answer False if the context does not explicitly provide data for the given attribute and entity.
- Answer False if the data reported is not a direct numerical measurement.
- Answer False if the data reported only contains values for parameter estimates or measures of fit for a statistical model.
- Answer False for cases where there is not a clear choice for a single, numerical data value.
- Answer True only if the context explicitly provides a direct numerical value measured for the given attribute, with respect to the entity in question.
- Along with your answer, provide a brief explanation justifying the reasons for your decision.
- Structure your response as a JSON object with "explanation" and "answer" fields.
"""

# Step 3: Extract value from text (combined locate + extract)
EXTRACT_TEXT_VALUE_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to determine if a page of text from a research paper contains a measured value for a given attribute and entity, and if so, to extract it.

Guidelines:
- If the page does not contain a relevant measurement, set has_value to false and leave value and units as null.
- If a measurement is found, set has_value to true, extract the value exactly as it appears in the context, and extract the units of measurement.
- Copy the value exactly as it appears -- do not convert, round, or modify it.
- Do not include uncertainty measures, confidence intervals, or range bounds in the value field.
- If there are multiple types of values reported (e.g., mean, min, max), extract the mean or central value unless the attribute description directs otherwise.
- Give the value only in the value field, and do not include any units of measurement, descriptors, or explanation.
- Structure your response as a JSON object with "explanation", "has_value", "value", and "units" fields.
"""

# Step 4: Extract value from table (combined locate + extract)
EXTRACT_TABLE_VALUE_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to determine if an HTML table from a research paper contains a measured value for a given attribute and entity, and if so, to identify the row and column needed to locate it.

You will be provided with:
- The full HTML table
- A list of row names in the table
- A list of column names in the table
- A description of the entity and attribute to find

Guidelines:
- If the table does not contain a relevant measurement, set has_value to false and leave row_index, column_index, and units as null.
- If a measurement is found, set has_value to true, and provide the exact row_index name and column_index name needed to locate the cell.
- Your row_index and column_index must exactly match names from the provided lists.
- Also extract the units of measurement if identifiable from the table headers or context.
- If there are multiple types of values reported (e.g., mean, min, max), choose the row/column for the mean or central value unless the attribute description directs otherwise.
- Structure your response as a JSON object with "explanation", "has_value", "row_index", "column_index", and "units" fields.
"""


# ---------------------------------------------------------------------------
# Legacy prompts (kept for backward compatibility with older experiment scripts)
# ---------------------------------------------------------------------------

# Feature terms
IDENTIFY_FEATURE_TERMS_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to assist in locating data points, by first identifying terminology used within the context. Given context from a research paper and a specific feature type, extract any terminology or abbreviations used to directly refer to the feature in question.

Guidelines:
- Do not include any abbreviations that refer to similar concepts or measurements, but which are not direct in relation.
- Pay close attention to tables and figure captions, as these often contain the abbreviations used in the main text.
- Do not infer, guess, or fabricate any terms or abbreviations that are not explicitly present in the context.
- Structure your response as a list of strings, for example: ['term_1', 'term_2']
- If the context does not contain any terminology or abbreviations that directly refer to the given feature, respond with an empty list.
"""

IDENTIFY_FEATURE_UNITS_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to assist in the data collection process by specifying the units used for measurement of a given feature within context provided for a research paper.

Guidelines:
- To ensure units follow standard formatting conventions, you will be given a list of options and your response should be limited to options from among the given list.
- If, however, none of the options fit with what is seen in the context, respond with the unit exactly as it appears in the context.
- Your response should include the unit only, do not include any additional explanation or text.
- If the context does not explicitly provide data for the given feature, respond with 'None'.
"""


# Determine if features are present in context
IDENTIFY_ENTITY_FEATURE_PAIRS_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to determine if context from a research paper contains data for a described feature in reference to a given entity.

Guidelines:
- Answer False if the the given feature or entity do not appear in the context.
- Answer False if the context does not explicity provide data for the given feature and entity.
- Answer False if the data reported is not a direct numerical measurement.
- Answer False if the data reported only contains values for parameter estimates or measures of fit for a statistical model.
- Answer False for cases where there is not a clear choice for a single, numerical data value.
- Answer True only if the context explicity provides a direct numerical value measured for the given feature, with respect to the entity in question.
- Along with your answer, provide a brief explanation for justifying the reasons for your decision.
- Structure your response as a JSON object with "explanation" and "answer" fields.
"""


# Locating measurements on a single page
PAGE_LOCATE_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to assist in locating data points, by determining if they occur on a single, given page of context. You will be queried with a description of a specific entity and feature to be measured, and asked to determine if a relevant measurement occurs on the page.

Guidelines:
- Answer False if the the given feature or entity do not appear in the context.
- Answer False if the context does not explicity provide data for the given feature and entity.
- Answer False if the data reported is not a direct numerical measurement.
- Answer False if the data reported only contains values for parameter estimates or measures of fit for a statistical model.
- Answer False for cases where there is not a clear choice for a single, numerical data value.
- Along with your answer, provide a brief explanation for justifying the reasons for your decision.
- Structure your response as a JSON object with "explanation" and "answer" fields.
"""

# Locating measurements in a single table
TABLE_LOCATE_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to assist in locating data points, by determining if they occur within context from a single, given html table. You will be queried with a description of a specific entity and feature to be measured, and asked to determine if a relevant measurement occurs within the table.

Guidelines:
- Answer False if the the given feature or entity do not appear in the contextual table.
- Answer False if the contextual table does not explicity provide data for the given feature and entity.
- Answer False if the data reported is not a direct numerical measurement.
- Answer False if the data reported only contains values for parameter estimates or measures of fit for a statistical model.
- Answer False for cases where there is not a clear choice for a single, numerical data value.
- Answer True only if the contextual table explicity provides a direct numerical value measured for the given feature, with respect to the entity in question.
- Along with your answer, provide a brief explanation for justifying the reasons for your decision.
- Structure your response as a JSON object with "explanation" and "answer" fields.
"""

# Extract value from plain text (non-table)
MEASURE_VALUE_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to extract precise numerical data from context provided from a research paper. You will be queried with a description of a specific entity and feature to collect data for. Your task is to extract the corresponding value if it appears in the provided context.

Guidelines:
- Respond with the value None if the the given feature or entity do not appear in the context.
- Respond with the value None if the context does not explicity provide data for the given feature and entity.
- Respond with the value None if the data reported is not either a direct numerical measurement.
- Respond with the value None if the data reported only contains values for parameter estimates or measures of fit for a statistical model.
- Respond with the value None for cases where there is not a clear choice for a single numerical data value.
- Respond with the extracted value only if the context explicity provides a direct numerical value measured for the given feature, with respect to the entity in question.
- If a data point is extracted, copy the value exactly as it appears in the context.
- Give the value only, and do not include any units of measurement, descriptors, or explanation in your response.
- If the there are multiple types of values explicitly reported for a feature (e.g. mean, minimum, maximum, uncertainty) respond with the mean or central value only, unless the queried feature specifically directs otherwise.
"""

# Select row index in a table
MEASURE_TABLE_ROW_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to assist in locating data points within context from a single, given html table from a research paper. You will be queried with a description of a specific entity and feature to be measured, and asked to extract the corresponding row index name necessary to locate the measurement within the table.

Guidelines:
- If there are multiple row names that could apply, choose the one that is most specific to the entity or feature in question.
- If the there are multiple types of values explicitly reported for a feature (e.g. mean, minimum, maximum, uncertainty) respond with the row for the mean or central value only, unless the given measurement specifically directs otherwise.
- Your response must use the exact row index name as it appears in the table.
- If there is no row name corresponding to the relevant entity or feature, respond 'None'.
- Respond with the row name only, without any additional text or explanation.
"""

# Select column index in a table
MEASURE_TABLE_COLUMN_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to assist in locating data points within context from a single, given html table from a research paper. You will be queried with a description of a specific entity and feature to be measured, and asked to extract the corresponding column index name necessary to locate the measurement within the table.

Guidelines:
- If there are multiple column names that could apply, choose the one that is most specific to the entity or feature in question.
- If the there are multiple types of values explicitly reported for a feature (e.g. mean, minimum, maximum, uncertainty) respond with the column for the mean or central value only, unless the given measurement specifically directs otherwise.
- Your response must use the exact column index name as it appears in the table.
- If there is no column name corresponding to the relevant entity or feature, respond 'None'.
-  Respond with the column name only, without any additional text or explanation.
"""

# Standardize extracted measurement value
STANDARDIZE_MEASUREMENTS_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to assist in the data collection process by standardizing measurement values extracted from context provided for a research paper. You will be queried with a description of a specific entity and attribute to collect data for, along with an extracted measurement value. Your task is to standardize the extracted measurement value according to the following guidelines.

Guidelines:
- For numerical values associated with uncertainty measures (e.g., ± values, confidence intervals), report only the central value without any uncertainty information.
- For numerical values reported as ranges with a central value (e.g., 5 (3-7)), report only the central value unless the queried attribute specifically directs otherwise.
- For numerical values reported as ranges without a central value (e.g., 3-7), choose the single value which best fits the queried attribute.
- For numerical values reported with inequalities (e.g., < 5), report the numerical value only without any additional formatting.
- For numerical values which are reported with a unit of measurement or other descriptor, convert the value to a standardized numerical format without any units or descriptors.
- Your response should include the standardized value only, do not include any additional explanation or text.
- If the value does not need any standardization (i.e. is a single numerical or descriptive value), return the value exactly as it is given.
"""

# Choose / determine units
STANDARDIZE_UNITS_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to assist in the data collection process by providing units for measurement values extracted from context provided for a research paper. You will be queried with a description of a specific entity and feature to collect data for, along with an extracted measurement value. Your task is to determine the unit of measurement for that data point by referencing the context, and then choosing from a list of given unit options.

Guidelines:
- To ensure units follow standard formatting conventions, your response should be limited to options from among the given list.
- If, however, none of the options fit with what is seen in the context, respond with the unit exactly as it appears in the context.
- Your response should include the unit only, do not include any additional explanation or text.
"""

# Validate extracted measurement value
JUDGE_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews.

You will be given:
1) A passage of context from a research paper
2) A description of a candidate extracted data point (attribute description, entity description, and extracted value)

Your task: decide whether the extracted value is explicitly present in the context AND is correctly attributed to the specified entity and attribute.

Decision rules:
- Respond 'true' ONLY if all of the following are satisfied:
  (A) Value presence: The extracted value appears explicitly in the context (same number), or appears as an equivalent formatting of the same number (e.g., '10' vs '10.0'; thousands separators; leading/trailing zeros). Do not accept values that require arithmetic, unit conversion, averaging, or other inference.
  (B) Correct attribution to entity: The context clearly indicates the value refers to the specified entity (not a different study site, species, dataset split, subgroup, scenario, treatment, timepoint, or a set/aggregate where the entity is ambiguous).
  (C) Correct attribution to attribute: The value clearly corresponds to the specified attribute (not a related metric, proxy, similarly named variable, or a different operationalization).
  (D) Direct measurement: The value is a directly reported measurement (or directly reported descriptive statistic of the measurement such as mean/median explicitly tied to the attribute), not a model parameter, regression coefficient, odds ratio, p-value, CI bound, test statistic, goodness-of-fit, or tuning/optimization output.

- Respond 'false' if ANY of the following apply:
  - The value does not appear in the context text/table exactly (aside from trivial formatting differences).
  - The value appears but is tied to a different entity or attribute than the one described.
  - The only matching numbers are part of uncertainty/intervals (e.g., '±', CI ranges) and the extracted value is not shown as the central estimate in the context.
  - The value is only implied (requires calculation, conversion, or deduction).
  - There are multiple plausible candidate values within the context and the extracted value is not uniquely supported.

Output format:
- Respond with a single token as either 'true' or 'false' (lowercase). Do not include any additional text or punctuation.
"""


# Clean tables
CLEAN_TABLE_INSTRUCTIONS = """
You are an expert data engineer specializing in cleaning unstructured HTML tables for Python/Pandas processing. You will be provided with an HTML table from a research paper, along with an image of the page it was extracted from. Your job is to use the context to improve the structure and content of the table so that it is in accurate, clean, LLM readable, html format.

Formatting Instructions:
1. Standardize to 'long' format: If the table is 'wide' (e.g., it lists different categories side-by-side with repeating columns), you must 'melt' or unpivot the table by creating new rows for each category, while keeping unique, non-repeating column headers. You should not melt the table if columns are non-repeating or non-hierarchical, only do so if it is necessary for machine readability.
2. You must create a single index column so that rows are machine identifiable. The very first column of your output table must be named 'index'. This column must contain unique identifiers for each row. If the rows are hierarchical (e.g., Category -> Sub-category) or if you unpivoted the data, you must combine the identifying columns into a Python-tuple format. For example, if the indentifying columns are 'column A' and 'column B', each row should be identified by a tuple: '('column A' value, 'column B' value)'. Otherwise if the identifiers are simple and have no hierarchy, the index should be a single non-tupled value. Importantly, when choosing identifying attributes, you should give highest priority to columns which use descriptive names, even if they must be combined with other attribute columns to uniquely identify the row. If no such columns exist, you may use numerical columns to uniquely identify rows.
3. If there are multi-level column headers, flatten them in the same way by grouping the headers in a tuple format. For example, if the headers are 'Year' and 'Measurement', the combined header should be ('Year', 'Measurement').
4. Your job is mainly to modify structure without interfering on the data. However, if you notice any inaccuracies or inconsistencies in the given HTML table, you must correct them.
5. If a single cell contains a main value along with a separate range or interval of numbers, you must split these into separate columns. For example, if data is reported in a 'mean (minimum, maximum)' format, you should create three separate columns for the mean, minimum, and maximum. Similarly, if data is reported in a 'value ± uncertainty' format, you should create one column for the main value and another column for the uncertainty. Use the context to make sure that the new columns are clearly named according to the statistic they represent. For example, in a single feature broken into mean, minimum, and maximum features you may use names such as 'feature_1_mean', 'feature_1_min', 'feature_1_max'. If there is no clear indication of what the statistic is, use generic names like 'feature_1_val_1', 'feature_1_val_2', etc.
6. The response table must be in HTML format, and wrapped inside <table number="i"></table> tags. Make sure to keep the table number attribute exactly as it appears in the given HTML.
7. At the very beginning of the table HTML, include <caption></caption> tags and use these to briefly describe the table and the measurements included within it. Use available table captioning on the pdf page to help, but make sure to include any additional information which might be relevant to understand the new formatting. 
8. Provide only the raw HTML for the full table, do not stop early (even if it is repetitive) and do not include any additional text or explanations.
"""