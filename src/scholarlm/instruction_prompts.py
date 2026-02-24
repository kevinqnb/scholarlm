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


# --------------------------------------------
# LLM as Judge Prompts
# --------------------------------------------


# Validate extracted measurement value
JUDGE_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews.

You will be given:
1) A complete document from a research paper
2) A description of a candidate extracted (entity, attribute, value) triplet, including the page and table (if applicable) where the data was found

Your task: decide whether the extracted triplet is fully valid — meaning the entity is correctly identified, the attribute is correctly assigned, and the value is correctly extracted, all with respect to the full document.

Decision rules:
- Respond 'true' ONLY if ALL of the following are satisfied:
  (A) Valid entity: The described entity is a real, distinct entity of the specified type as evidenced by the document. It must not be a hypothetical, aggregated, ambiguously described, or otherwise invalid instance of the entity type.
  (B) Value presence: The extracted value appears explicitly at the specified location in the document (refer to the provided page and table number) as a numerically identical value — i.e., both represent exactly the same number on the number line, differing only in surface formatting (e.g., 10 vs 10.0, 1000 vs 1,000, 0.001 vs 1e-3, negative sign variants). Do not accept values that require unit conversion (0.05 vs 5%), rounding (3.14 vs 3.1), arithmetic, averaging, or any other numerical transformation.
  (C) Correct assignment to entity: The document clearly indicates the value refers to the specified entity (not a different study site, species, dataset split, subgroup, scenario, treatment, timepoint, or a set/aggregate where the entity is ambiguous).
  (D) Correct assignment to attribute: The value clearly corresponds to the specified attribute (not a related metric, proxy, similarly named variable, or a different operationalization).
  (E) Direct reported quantity: The value represents a directly reported measurement or descriptive statistic (mean, median, total) of the attribute — not a model output (regression coefficient, odds ratio, p-value, CI bound, test statistic, goodness-of-fit metric, or tuning parameter). The value must also be reported as a standalone quantity, not solely as an endpoint of a range, interval, or bound (e.g., reject "6.5" if it only appears in "ranged from 6.5 to 7.2" and is not independently stated as the attribute's value).

- Respond 'false' if ANY of the following apply:
  - The described entity does not correspond to a valid entity of the specified type as described in the document.
  - The value does not appear at the specified location in the document exactly (aside from trivial formatting differences).
  - The value appears but is tied to a different entity or attribute than the one described.
  - The value appears only as an endpoint of a range, interval, or bound (e.g., "6.5–7.2", "ranged from X to Y", "between X and Y") and is not independently reported as a standalone quantity for the specified entity and attribute.
  - The value is a model output, test statistic, or derived statistical quantity (e.g., regression coefficient, odds ratio, p-value, CI bound, goodness-of-fit metric) rather than a directly reported quantity or descriptive summary.
  - The value is only implied (requires calculation, unit conversion, or deduction from other reported numbers).

Output format:
- Respond with a single token as either 'true' or 'false' (lowercase). Do not include any additional text or punctuation.
"""


# --------------------------------------------
# Table Cleaning Prompts
# --------------------------------------------


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

CLEAN_TABLE_INSTRUCTIONS_V2 = """
You are a document reconstruction engine. You will receive:
1. An image of a single PDF page from a research paper.
2. The OCR-parsed text of that page, with HTML tables inline at their original positions within `<table number="i">...</table>` tags.

Your task: reproduce the OCR text exactly as given, but replace each `<table>` block with a cleaned, normalized version. Do not modify any text outside of `<table>` tags.

### Table Normalization Rules

**Goal:** Transform each table so that every data cell maps to exactly one (entity, attribute, value) triplet. An entity is what a row describes, an attribute is what a column measures, and a value is the cell content.

**Entity Index (first column):**
- You MUST create a create a new column named `index` as the first column of every output table. This is the entity identifier.
- To populate the 'index' column, use information from one or more columns from the original table that uniquely identify each row.
- When identifying rows always refer named entities (e.g., object names, study names, compound names, model names) over numerical IDs if possible.
- If multiple columns are needed to uniquely identify a row (e.g., a category and sub-category), combine them as a Python tuple: `('Category A', 'Sub-category 1')`.
- Every index value must be unique, and your choice of identifiers should be consistent across all rows in the table.

NOTE: This is a critical step for machine readability. The index column is what allows downstream code to refer to specific rows, so it must be populated with meaningful, unique identifiers. If the original table has no clear entity identifiers, you may use numerical row numbers, but this is a last resort.

**Melting wide tables:**
- If a table has repeating or hierarchical column groups (e.g., the same measurements repeated under different conditions), unpivot it into long format by creating new rows for each group.
- Do NOT melt tables where columns are genuinely distinct, non-repeating attributes.
- When melting, incorporate the group label into either the index (as a tuple element) or as a new column, whichever preserves clarity.

Melting wide tables:

- Each column should represent a distinct attribute (property or measurement) of the row entity. Each row should represent a distinct entity or observation.
- If column headers encode a second entity or condition rather than a distinct attribute — for example, the same measurement repeated across different drugs, time points, datasets, or experimental groups — unpivot the table into long format. Create new rows for each entity/condition, and add a column for the entity/condition label.
- Do NOT melt when columns represent genuinely different attributes of the same entity, even if they are related. For example, a table with columns accuracy, precision, recall, F1 for each model should remain wide, because these are distinct measurements. But a table with accuracy_dataset_A, accuracy_dataset_B should be melted, because dataset is an entity being encoded in the column headers.
- Heuristic: If you could factor a set of column headers into {attribute} × {entity/condition}, the table should be melted along the entity/condition axis.
- When melting, incorporate the new entity/condition label into the index as a tuple element, and also preserve it as its own column.

**Flattening multi-level headers:**
- If column headers span multiple rows, flatten them into a single header row using Python tuple format: `('Level 1', 'Level 2')`.



**Melting wide tables:**

Each column should represent a distinct attribute of the row entity. Each row should represent a distinct entity or observation.
If column headers encode a second entity or condition rather than a distinct attribute — for example, the same measurement repeated across different drugs, time points, datasets, or experimental groups — unpivot the table into long format. Create new rows for each entity/condition, and add a column for the entity/condition label.
Do NOT melt when columns represent genuinely different attributes of the same entity, even if the attributes are related. For example, a table with columns accuracy, precision, recall, f1 for each model should remain wide, because these are distinct measurements of each model. But a table with accuracy_dataset_a, accuracy_dataset_b should be melted, because dataset is an entity encoded in the column headers.
Heuristic: if you can factor a set of column headers into {attribute} × {entity/condition}, the table should be melted along the entity/condition axis. If there are multiple such axes (e.g., {attribute} × {drug} × {time_point}), melt along all of them, giving each its own column.
When melting, incorporate the new entity/condition label into the index as a tuple element, and also preserve it as its own column.
Columns that are not part of the repeated group (e.g., metadata like name or category) should be carried through unchanged to every new row.

**Column naming:**

Every column must have a unique, descriptive name.
Use lowercase with underscores (e.g., dose_mg, response_rate).
If the original table has multi-level headers, first apply the melting rule. For any multi-level headers that remain after melting, concatenate the levels with underscores into a single name (e.g., blood_pressure_systolic).
Where feasible, incorporate units into the column name (e.g., dose_mg rather than Dose (mg)).


**Splitting composite values:**
- If a cell contains a main value bundled with a range, interval, or uncertainty (e.g., `3.5 (2.1–4.8)` or `12.3 ± 0.5`), split it into separate columns.
- Name the new columns descriptively based on context: e.g., `feature_mean` and `feature_confidence_interval`. If the statistic type is unclear, use `feature_val`, `feature_aux_1`, `feature_aux_2`, etc.

**Captions:**
- Table captions in the OCR text typically appear outside the `<table>` tags as free-standing text (e.g., "Table 1: Patient demographics..."). You must move this caption text from its original position in the OCR output into `<caption>...</caption>` tags at the start of the corresponding `<table>` block. Remove the caption from its original location so it is not duplicated.
- After the original caption text, append a brief note describing any relevant information not already included, as well as any structural details needed to interpret the new version of the table. If no changes were needed, do not append anything.

**Data integrity:**
- Preserve all original data values. Your priority is to restructure and add additional indexing information. Only correct clear OCR errors or formatting artifacts (e.g., broken Unicode, misaligned cells) — use the page image as ground truth.
- Output tables must be valid HTML within `<table>...</table>` tags.
- If the table has a numbered tag, keep the same number in your output (e.g., `<table number="1">` should remain `<table number="1">`).

### Example

**Input table:**
```html
<table number="1">
<tr><th></th><th colspan="2">Drug A</th><th colspan="2">Drug B</th></tr>
<tr><th>Patient</th><th>Dose (mg)</th><th>Response</th><th>Dose (mg)</th><th>Response</th></tr>
<tr><td>P-001</td><td>50</td><td>0.82</td><td>75</td><td>0.91</td></tr>
<tr><td>P-002</td><td>50</td><td>0.67</td><td>75</td><td>0.73</td></tr>
</table>
```

**Output table:**
```html
<table number="1">
<caption>Patient drug response data. Melted from wide format; original columns grouped by drug type.</caption>
<tr><th>index</th><th>drug</th><th>dose_mg</th><th>response</th></tr>
<tr><td>('P-001', 'Drug A')</td><td>Drug A</td><td>50</td><td>0.82</td></tr>
<tr><td>('P-001', 'Drug B')</td><td>Drug B</td><td>75</td><td>0.91</td></tr>
<tr><td>('P-002', 'Drug A')</td><td>Drug A</td><td>50</td><td>0.67</td></tr>
<tr><td>('P-002', 'Drug B')</td><td>Drug B</td><td>75</td><td>0.73</td></tr>
</table>
```

Notice: the repeating column groups (Drug A, Drug B) were melted into rows, the entity index combines patient and drug as a tuple, and the drug label is also preserved as its own column for clarity.

### Output Format

Return the full page text with normalized tables inline. Do not add any commentary, preamble, or explanation outside the reproduced text.
"""


CLEAN_TABLE_INSTRUCTIONS_V3 = """
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