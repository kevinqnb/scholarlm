"""MeasurementLM Ablation 7: No Standardization, Quantity Parsing, or Deduplication

Ablation goal: understand how much of recovery/validity comes from the
extraction pipeline itself (entity/attribute detection, provenance, event
resolution, value extraction) versus the post-extraction cleanup stage --
unit standardization, quantity-shape parsing, and duplicate merging.

Changes from the baseline MeasurementLM:

1. `_standardize()` is not called (as an LLM step) -- `units` is left exactly
   as extracted by the value-extraction step, never mapped onto the
   attribute's preferred unit list.
2. `_parse_quantities()` is not called (as an LLM step) -- rather than
   decomposing the raw `value` string into qualifier tags plus
   point_value/lower/upper/list_values/tolerance/standard_deviation via a
   dedicated LLM call, every one of those fields
   (`scholarlm.config.QUALIFIER_FIELD_NAMES`) is written as null on every
   record instead.
3. `_deduplicate()` is not called -- no merging of duplicate
   (entity_id, attribute, event) records; every surviving text/table
   extraction stays its own row, with its own provenance fields left as
   plain scalars rather than the aggregated-into-lists shape
   `_deduplicate()` produces. Each row is still individually identified --
   `run_ablation.py` assigns every surviving record a unique
   `measurement_id` after `fit()` returns, the same as it does for every
   other ablation.

Unchanged from baseline: `_extract_entities()`, `_entity_provenance()`,
`_detect_attributes()`, `_attribute_provenance()`, `_resolve_events()`,
`_extract_values_from_text()`, `_extract_values_from_tables()`, `fit()`
(inherited as-is -- it still calls `_standardize()` -> `_parse_quantities()`
-> `_deduplicate()` in sequence inside the "final" timed step; only what
those three methods *do* changes here).

A record with every qualifier field null and its raw `value`/`units`
untouched is exactly the "qualifiers unfilled" shape
`analysis/postprocessing.py`'s `_qualifiers_unfilled()` already recognizes
for ChatExtract/old-NuExtract/GLiNER-style output. Running
`analysis/postprocessing.py` (unmodified, same as for any other experiment
id) on this ablation's `final.json` fills point_value/qualifiers from the
raw `value` via `parsing.parse_quantity_shape_numeric()` and standardizes
units against the ground truth's own vocabulary, producing
`postprocessed.json` -- the file `analysis/match_cache.py` prefers whenever
it exists, and from there a clean set of records for `match_cache.py` to
match against ground truth.
"""

from .measurementlm import MeasurementLM
from .config import QUALIFIER_FIELD_NAMES


class MeasurementLMAblation7(MeasurementLM):
    """
    Ablation 7: the post-extraction cleanup stage (standardize, parse
    quantities, deduplicate) is skipped entirely. Qualifier/shape fields are
    left null on every record and no duplicate-merging happens; the
    extraction pipeline itself (steps 1-6) is unchanged.
    """

    def _standardize(self):
        """Skipped: `units` is left exactly as extracted -- no LLM
        unit-standardization call. Returns a plain copy of `self.data` so the
        "final" step's chain (`_standardize` -> `_parse_quantities` ->
        `_deduplicate`) still behaves as three independent list-in/list-out
        calls, matching the base class's own calling convention."""
        return [dict(datapoint) for datapoint in self.data]

    def _parse_quantities(self):
        """Skipped: rather than parsing the raw `value` string into shape
        fields via an LLM call, every qualifier/shape field
        (`QUALIFIER_FIELD_NAMES`) is written as null on every record."""
        return [
            dict(datapoint) | {field: None for field in QUALIFIER_FIELD_NAMES}
            for datapoint in self.data
        ]

    def _deduplicate(self, data):
        """Skipped: returns `data` unchanged -- no merging of duplicate
        (entity_id, attribute, event) records, no provenance aggregation
        into lists."""
        return data
