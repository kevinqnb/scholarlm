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
- Copy the value exactly as it appears, in full — including any range, list, inequality, mean/median/count label, or uncertainty measure (± value, confidence interval, standard deviation) reported alongside it. Do not convert, round, drop, or otherwise modify any part of it.
- If there are multiple, separate measurements reported (e.g., different sites, dates, or conditions), extract only the one relevant to the given (entity, attribute, event); do not merge separate measurements into one value.
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


STANDARDIZE_MEASUREMENTS_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to assist in the data collection process by standardizing the units of a measurement value extracted from a research paper. You will be given the source text where the measurement was extracted from (either a page of prose text or an HTML table), a description of the specific entity and attribute, a list of available (preferred) units for the attribute, and an extracted measurement value with units. Use the provided source text to verify the original measurement context. Your task is to standardize the units only, according to the following guidelines.

Units standardization guidelines:
- If the extracted units are a notational variant of one of the available units (e.g., "mg/L" vs "mg L⁻¹", "μm" vs "um", "°C" vs "degrees C"), return the best matching entry from the available units list. You may infer notational variants based on common scientific usage.
- If the extracted units are not a notational variant of any available unit (i.e., they would require unit conversion to match, or there are no available units listed), return the extracted units unchanged.
- If the extracted units are null (not reported), return null.
- Do NOT modify, standardize, round, or reformat the extracted measurement value in any way -- it is not part of this task and is handled separately.

- Provide a brief explanation of what unit standardization was applied (or why none was needed).
- Structure your response as a JSON object with "explanation" and "units" fields.
"""


# 2026-09-23-standardize-valueonly-01: value-only variant of
# STANDARDIZE_MEASUREMENTS_INSTRUCTIONS above -- no source-text grounding, no
# entity/attribute description, no attribute terminology. Tests the hypothesis
# that unit standardization doesn't need (and may be hurt by) the surrounding
# page/table context _standardize() otherwise supplies -- mirrors
# PARSE_QUANTITY_VALUE_ONLY_INSTRUCTIONS's rationale for the sibling
# _parse_quantities() step. The decision rule itself (best-matching notational
# variant, else unchanged, else null for null) is unchanged from the full-context
# version -- only the context shown to the model differs.
#
# The basis/component-annotation bullet below was added after rung-3-01's tiny
# end-to-end: without page context, gemma-3-27b was inconsistent on whether a
# substance-qualified compound unit (e.g. "mg N/L") counts as a notational
# variant of the plain unit on the list ("mg/L") -- it correctly dropped an
# analogous phosphorus annotation but kept the nitrogen one, on structurally
# identical inputs. The added bullet makes explicit, with dataset-independent
# examples, exactly the distinction the model needs and evidently wasn't
# getting reliably from the unit string alone: an annotation is safe to drop
# only when doing so requires no numeric conversion.
STANDARDIZE_MEASUREMENTS_VALUE_ONLY_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to assist in the data collection process by standardizing the units of a measurement value extracted from a research paper. You will be given a list of available (preferred) units for the attribute, and an extracted measurement value with units -- no other context is provided or needed. Your task is to standardize the units only, according to the following guidelines.

Units standardization guidelines:
- If the extracted units are a notational variant of one of the available units (e.g., "mg/L" vs "mg L⁻¹", "μm" vs "um", "°C" vs "degrees C"), return the best matching entry from the available units list. You may infer notational variants based on common scientific usage.
- If the extracted units carry an additional basis or component annotation that the best-matching available unit omits (e.g., "kg (dry matter)/ha" when the available unit is "kg/ha", or "counts (viable)/mL" when the available unit is "counts/mL"), it is fine to use the more generic available unit, as long as no numeric conversion is required -- only the annotation is being dropped, not the measured quantity. If the annotation instead names a genuinely different substance or basis that would require converting the value (not just relabeling it), leave the units unchanged.
- If the extracted units are not a notational variant of any available unit (i.e., they would require unit conversion to match, or there are no available units listed), return the extracted units unchanged.
- If the extracted units are null (not reported), return null.
- Do NOT modify, standardize, round, or reformat the extracted measurement value in any way -- it is not part of this task and is handled separately.

- Provide a brief explanation of what unit standardization was applied (or why none was needed).
- Structure your response as a JSON object with "explanation" and "units" fields.
"""


PARSE_QUANTITY_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to parse a single already-extracted measurement value into its structured components, using the source text it was extracted from as ground truth.

You will be given: the source text (a page or table) the value was extracted from, a description of the measurement attribute, and the extracted value and units as originally reported.

Guidelines:
- qualifiers: a list of zero or more tags describing the shape of the reported quantity. Use only tags from this set: "IsCount" (a count of discrete items, not a continuous measurement), "IsApproximate" (explicitly hedged, e.g. "~12", "about 50", "approximately"), "IsList" (an enumerated list of separate values, not a single number or range), "IsRange" (a reported interval or one-sided bound, e.g. "3-7", "< 5", "at least 10"), "IsMean" (an explicitly stated mean/average), "IsMedian" (an explicitly stated median), "HasTolerance" (an explicit +/- value or confidence interval is reported alongside the value), "HasSD" (an explicit standard deviation is reported alongside the value). These tags are independent and may combine freely when the text supports it (e.g. an approximate mean is ["IsApproximate", "IsMean"]; a mean reported together with a range, e.g. "5.2 (3.1-7.4)", is ["IsMean", "IsRange"]). Use an empty list for a plain, unhedged single value with no other qualifier.
- point_value: the single central value, when one is directly reported -- a plain point value, or the stated mean/median/count. Leave null if no single central value is reported (e.g. a bare range or list with no central value given).
- lower / upper: the bounds of a reported range or one-sided inequality. For a two-sided range, populate both. For a one-sided bound (e.g. "< 5", "at least 10"), populate only the reported side and leave the other null. Leave both null if no range or bound is reported.
- list_values: the parsed items of an enumerated list, in the order reported. Leave null unless "IsList" applies.
- tolerance: the confidence interval or +/- value exactly as reported (e.g. "± 0.5", "95% CI: 5-9"), as a freeform string. Leave null unless "HasTolerance" applies.
- standard_deviation: the standard deviation exactly as reported, as a freeform string. Leave null unless "HasSD" applies.
- Do NOT infer, guess, or derive any field. Use ONLY what is explicitly stated in the source text.
- Provide a brief explanation of your parsing decisions.
- Structure your response as a JSON object with "explanation", "qualifiers", "point_value", "lower", "upper", "list_values", "tolerance", and "standard_deviation" fields.
"""


# 2026-09-22-pond-parsequantities-valueonly-01: value-only variant of
# PARSE_QUANTITY_INSTRUCTIONS above -- no source-text grounding, no entity/
# attribute description, no units. Tests the hypothesis that this purely
# textual parsing task doesn't need (and may be hurt by) the surrounding
# page/table context _parse_quantities() otherwise supplies.
PARSE_QUANTITY_VALUE_ONLY_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to parse a single already-extracted measurement value into its structured components, using only the value string itself -- no other context is provided or needed.

You will be given: the extracted value, exactly as reported.

Guidelines:
- qualifiers: a list of zero or more tags describing the shape of the reported quantity. Use only tags from this set: "IsCount" (a count of discrete items, not a continuous measurement), "IsApproximate" (explicitly hedged, e.g. "~12", "about 50", "approximately"), "IsList" (an enumerated list of separate values, not a single number or range), "IsRange" (a reported interval or one-sided bound, e.g. "3-7", "< 5", "at least 10"), "IsMean" (an explicitly stated mean/average), "IsMedian" (an explicitly stated median), "HasTolerance" (an explicit +/- value or confidence interval is reported alongside the value), "HasSD" (an explicit standard deviation is reported alongside the value). These tags are independent and may combine freely when the text supports it (e.g. an approximate mean is ["IsApproximate", "IsMean"]; a mean reported together with a range, e.g. "5.2 (3.1-7.4)", is ["IsMean", "IsRange"]). Use an empty list for a plain, unhedged single value with no other qualifier.
- point_value: the single central value, when one is directly reported -- a plain point value, or the stated mean/median/count. Leave null if no single central value is reported (e.g. a bare range or list with no central value given). A plain number with no qualifying text (e.g. "2.4") IS its own point_value.
- lower / upper: the bounds of a reported range or one-sided inequality. For a two-sided range, populate both. For a one-sided bound (e.g. "< 5", "at least 10"), populate only the reported side and leave the other null. Leave both null if no range or bound is reported.
- list_values: the parsed items of an enumerated list, in the order reported. Leave null unless "IsList" applies.
- tolerance: the confidence interval or +/- value exactly as reported (e.g. "± 0.5", "95% CI: 5-9"), as a freeform string. Leave null unless "HasTolerance" applies.
- standard_deviation: the standard deviation exactly as reported, as a freeform string. Leave null unless "HasSD" applies.
- Do NOT infer, guess, or derive anything beyond what the value string itself states.
- Provide a brief explanation of your parsing decisions.
- Structure your response as a JSON object with "explanation", "qualifiers", "point_value", "lower", "upper", "list_values", "tolerance", and "standard_deviation" fields.
"""


# --------------------------------------------
# MeasurementLMv2 Prompts (quantity-first pipeline)
#
# v2 flips the extraction order: quantities are collected per (page, attribute)
# before any entity or event is known, deduplicated, grouped by document and
# source location (prose vs. a specific table), and only then attributed to an
# entity/event in one full-paper call per group. See
# src/scholarlm/measurementlmv2.py.
# --------------------------------------------

QUANTITY_COLLECTION_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to extract every directly reported quantity for a single measurement attribute from a single page of a research paper, pulling from both prose text and any tables on the page.

Guidelines:
- Scan the entire page — including prose text, tables, table captions, and footnotes — for every distinct numerical value reported for the given attribute.
- Return one item per distinct quantity found. If the same value is reported more than once on the page for what is clearly the same measurement, report it once. If the page reports multiple different measurements for the attribute (different entities, dates, conditions, etc.), report each one as a separate item — do not average or collapse them.
- Only include direct numerical measurements. Do NOT include model parameters, coefficients, p-values, or measures of statistical fit.
- If the page reports no data for the attribute, return an empty items list.

For each quantity, populate:
- value: the full raw value exactly as reported, in full — including any range, list, inequality, mean/median/count label, or uncertainty measure (± value, confidence interval, standard deviation) reported alongside it. Do not convert, round, drop, or otherwise modify any part of it.
- units: the unit of measurement as reported, or null if the attribute is dimensionless or no unit is given.
- qualifiers: a list of zero or more tags describing the shape of this quantity. Use only tags from this set, and combine them freely when the text supports it (e.g. an approximate mean is ["IsApproximate", "IsMean"]); use an empty list for a plain, unhedged single value:
  - "IsCount": a count of discrete items, not a continuous measurement.
  - "IsApproximate": explicitly hedged, e.g. "~12", "about 50", "approximately".
  - "IsList": an enumerated list of separate values, not a single number or range.
  - "IsRange": a reported interval or one-sided bound, e.g. "3-7", "< 5", "at least 10".
  - "IsMean": an explicitly stated mean/average.
  - "IsMedian": an explicitly stated median.
  - "HasTolerance": an explicit +/- value or confidence interval is reported alongside the value.
  - "HasSD": an explicit standard deviation is reported alongside the value.
- point_value: the single central value, when one is directly reported -- a plain point value, or the stated mean/median/count. Leave null if no single central value is reported (e.g. a bare range or list with no central value given).
- lower / upper: the bounds of a reported range or one-sided inequality. For a two-sided range, populate both. For a one-sided bound (e.g. "< 5", "at least 10"), populate only the reported side and leave the other null. Leave both null if no range or bound is reported.
- list_values: the parsed items of an enumerated list, in the order reported. Leave null unless "IsList" applies.
- tolerance: the confidence interval or +/- value exactly as reported (e.g. "± 0.5", "95% CI: 5-9"), as a freeform string. Leave null unless "HasTolerance" applies.
- standard_deviation: the standard deviation exactly as reported, as a freeform string. Leave null unless "HasSD" applies.
- table_number: if the quantity is reported within a table on this page, the table number from the enclosing `<table number="x">` tag. Otherwise null (the quantity is in prose text).

Strict rules:
- Do NOT infer, guess, or derive any value. Use only what is explicitly stated on the page.
- Structure your response as a JSON object with an "items" list, where each item has "value", "units", "qualifiers", "point_value", "lower", "upper", "list_values", "tolerance", "standard_deviation", and "table_number" fields.
"""


STANDARDIZE_QUANTITY_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to standardize the units of a single extracted quantity, using the page or table it was extracted from as the source of truth.

You will be given: the source text (a page or table), a description of the measurement attribute, a list of preferred units for the attribute, and the quantity as extracted (its value and units).

Guidelines:
- If the extracted units are a notational variant of one of the preferred units (e.g., "mg/L" vs "mg L⁻¹", "μm" vs "um", "°C" vs "degrees C"), return the matching preferred unit. You may infer notational variants based on common scientific usage.
- If the extracted units are not a notational variant of any preferred unit (i.e., they would require unit conversion to match, or there are no preferred units listed), return the extracted units unchanged.
- If the extracted units are null (not reported), return null.
- Do NOT modify, standardize, round, or reformat the extracted value or any of its qualifier/shape fields (point_value, lower, upper, list_values, tolerance, standard_deviation) in any way -- that is not part of this task and is handled separately.
- Provide a brief explanation of what unit standardization was applied (or why none was needed).
- Structure your response as a JSON object with "explanation" and "units" fields.
"""


CONTEXTUALIZE_QUANTITIES_INSTRUCTIONS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to identify everything each of a list of already-extracted quantities describes: which entity or entities it was measured for, and under what measurement event (date, method, condition, etc.).

You will be given: the full text of a research paper (with page and table boundaries marked), a list of quantities already extracted from this paper (each with its attribute, value, units, qualifiers, and shape fields, and the page number(s) where it was found), and reference descriptions of the entity and measurement-event fields to populate.

Guidelines:
- For every quantity in the list, first copy its attribute, value, units, qualifiers, point_value, lower, upper, list_values, tolerance, and standard_deviation fields back into your response item exactly as given, character-for-character — this is how your answer is matched back to the right quantity, so do not alter, round, standardize, or reformat them. A quantity's listing below only shows the fields it has; anything not listed for it is absent and must be copied back as JSON null (or an empty list for qualifiers/list_values), not as any word or placeholder text.
- Locate each quantity in the full paper using its given page number(s) (and table number, if the query says these quantities come from a table).
- Determine which entity (or entities) the quantity is reported for, and under what measurement event, using only information explicitly stated in the paper.
- In the ordinary case, a quantity describes exactly one (entity, event) combination — return a single item for it.
- If, and only if, the paper makes clear that this exact quantity is independently reported for more than one distinct entity or measurement event (e.g., two different sites happen to report the same rounded value), return one item per distinct (entity, event) combination — each copying back the same quantity fields.
- CRITICAL: when several entities appear together (e.g. in the same table or list), do NOT attach a quantity to all of them just because they share a category or context. Each entity you include must have this exact value reported for it individually — verify each candidate entity's own reported value before including it, and exclude any entity whose own value you cannot confirm matches, even if a similar or nearby entity's value does match.
- If you cannot confidently attribute a quantity to any entity, omit it from your response entirely rather than guessing — do not invent an item with blank entity/event fields just to have copied the quantity back.
- Populate every entity and event field as completely as the paper allows; use null for anything not explicitly stated. Do not infer, guess, or derive any field value.
- Structure your response as a JSON object with an "items" list, where each item has the quantity's fields (attribute, value, units, qualifiers, point_value, lower, upper, list_values, tolerance, standard_deviation) copied back, plus the entity fields and measurement-event fields described in the reference material provided in the query.
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
- Extract the value exactly as it appears in the document, in full — including any range, list, inequality, mean/median/count label, or uncertainty measure (± value, confidence interval, standard deviation) reported alongside it. Do not convert, round, drop, or otherwise modify any part of it.
- Give the value only in the value field; do not include any units, descriptors, or explanation there.
- For units, use the best fitting option from the attribute's listed preferred units if possible; otherwise specify the unit exactly as it appears in the text. Set units to null if no units are reported.
- qualifiers: a list of zero or more tags describing the shape of the reported value. Use only tags from this set, and combine them freely when the text supports it (e.g. an approximate mean is ["IsApproximate", "IsMean"]); use an empty list for a plain, unhedged single value:
  - "IsCount": a count of discrete items, not a continuous measurement.
  - "IsApproximate": explicitly hedged, e.g. "~12", "about 50", "approximately".
  - "IsList": an enumerated list of separate values, not a single number or range.
  - "IsRange": a reported interval or one-sided bound, e.g. "3-7", "< 5", "at least 10".
  - "IsMean": an explicitly stated mean/average.
  - "IsMedian": an explicitly stated median.
  - "HasTolerance": an explicit +/- value or confidence interval is reported alongside the value.
  - "HasSD": an explicit standard deviation is reported alongside the value.
- point_value: the single central value, when one is directly reported -- a plain point value, or the stated mean/median/count. Leave null if no single central value is reported (e.g. a bare range or list with no central value given).
- lower / upper: the bounds of a reported range or one-sided inequality. For a two-sided range, populate both. For a one-sided bound (e.g. "< 5", "at least 10"), populate only the reported side and leave the other null. Leave both null if no range or bound is reported.
- list_values: the parsed items of an enumerated list, in the order reported. Leave null unless "IsList" applies.
- tolerance: the confidence interval or +/- value exactly as reported (e.g. "± 0.5", "95% CI: 5-9"), as a freeform string. Leave null unless "HasTolerance" applies.
- standard_deviation: the standard deviation exactly as reported, as a freeform string. Leave null unless "HasSD" applies.
- Do NOT infer, guess, or derive any field value. If a field is not explicitly stated in the document, set it to null.
- Structure your response as a JSON object with an "items" list, where each item contains the entity fields, event fields, and "attribute", "value", "units", "qualifiers", "point_value", "lower", "upper", "list_values", "tolerance", and "standard_deviation" fields as specified in the dataset-specific instructions.
"""


# Ablation 1 variant: direct extraction with the qualifier/shape fields
# (qualifiers/point_value/lower/upper/list_values/tolerance/standard_deviation)
# removed -- identical to DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS otherwise,
# including the value-verbatim-extraction guideline below (still needed so
# analysis/postprocessing.py's parser has a full, unmodified value string to
# work from). See tests/test_ablation1_no_qualifiers.py for the exact
# line-level diff this must maintain against DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS.
DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS_NO_QUALIFIERS = """You are an expert in data extraction for systematic scientific literature reviews. Your task is to extract a complete list of measurement records from a research paper document in a single pass. Each record captures an entity, an attribute for measurement, the conditions of a specific measurement event, and its value.

Guidelines:
- You will be provided with dataset-specific extraction instructions describing the entities to identify, the target attributes, and the measurement event fields, along with the full document text.
- Identify all entities of the specified type present in the document, following the entity identification rules in the dataset-specific instructions.
- For each identified entity, identify all distinct measurement events and all attributes for which a direct numerical measurement is reported.
- Return one item per (entity, attribute, event) combination where a direct numerical measurement exists.
- Only include items where a direct numerical measurement is reported — omit absent data, model parameters, goodness-of-fit statistics, and qualitative descriptions.
- Extract the value exactly as it appears in the document, in full — including any range, list, inequality, mean/median/count label, or uncertainty measure (± value, confidence interval, standard deviation) reported alongside it. Do not convert, round, drop, or otherwise modify any part of it.
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
- Copy the value exactly as it appears, in full — including any range, list, inequality, mean/median/count label, or uncertainty measure (± value, confidence interval, standard deviation) reported alongside it. Do not convert, round, drop, or otherwise modify any part of it.
- If there are multiple, separate measurements reported (e.g., different sites, dates, or conditions), extract only the one relevant to the given (entity, attribute, event); do not merge separate measurements into one value.
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
- Copy the value exactly as it appears, in full — including any range, list, inequality, mean/median/count label, or uncertainty measure (± value, confidence interval, standard deviation) reported alongside it. Do not convert, round, drop, or otherwise modify any part of it.
- If there are multiple, separate measurements reported (e.g., different sites, dates, or conditions), extract only the one relevant to the given (entity, attribute, event); do not merge separate measurements into one value.
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
- Copy the value exactly as it appears, in full — including any range, list, inequality, mean/median/count label, or uncertainty measure (± value, confidence interval, standard deviation) reported alongside it. Do not convert, round, drop, or otherwise modify any part of it.
- If there are multiple, separate measurements reported (e.g., different sites, dates, or conditions), extract only the one relevant to the given (entity, attribute, event); do not merge separate measurements into one value.
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
1) In ## CONTEXT: The full text of a research paper.
2) In ## QUERY: a description of an extracted entity, a target attribute for measurement, information about its measurement event, and the corresponding extracted value with its units.

Your task: decide whether this extraction is correct — that is, whether the extracted value (with its units) is actually reported in the document for the specified entity, attribute, and (if applicable) measurement event.

Respond 'true' ONLY if ALL of the following hold:
(A) The entity is referenced within the context and is relevant to the specified entity type.
(B) The value is explicitly present within the context. Numerical identity is required: only trivial surface formatting differences are acceptable (e.g., 10 vs 10.0, 1,000 vs 1000, 1e-3 vs 0.001). Do not accept values that differ by rounding, averaging, unit conversion, or any other transformation.
(C) The value is assigned to the correct entity. The document makes clear the value belongs to the described entity, not to a different site, condition, subgroup, or an aggregate that includes other entities.
(D) The value is assigned to the correct attribute. The value corresponds to the specified attribute, not to a similarly named variable, proxy, or different operationalization of the same concept.
(E) The value is a direct measurement. It is a raw measurement or descriptive summary statistic of measurements (mean, median, SD, min, max, count, proportion, total) — not a model output (coefficient, odds ratio, p-value, CI bound, test statistic, goodness-of-fit metric, or correlation). It must appear as a standalone quantity: do not accept a value found only as an endpoint of a reported range (e.g., "ranged from 6.5 to 7.2") unless the target attribute specifically describes that endpoint.
(F) The units are correct. The units match those reported in the document for that value. Accept notational variants, including:
   - Standard formatting differences: "mg/L" vs "mg L⁻¹", "μm" vs "um", "°C" vs "degrees C"
   - Chemical species qualifiers that may be stated explicitly or implied by context: for measurements of a specific element or molecule, the species label (e.g., "N" for nitrogen, "P" for phosphorus, "C" for carbon, "C₂H₄" for ethylene) may appear in the units or be omitted when the attribute and context make the intended species unambiguous. For example, "nmol L⁻¹ h⁻¹" and "nmol N L⁻¹ h⁻¹" are equivalent for a nitrogen fixation rate measurement.
   Do not accept units that would require numerical conversion to match (e.g., mg/L vs g/L, ha vs m², nmol vs µmol).
(G) If the context reports multiple values for the same entity and attribute (e.g. at different dates or times), the value extracted corresponds to the measurement event specified in the QUERY.

Respond 'false' if ANY criterion is not met, or if the evidence is ambiguous.

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