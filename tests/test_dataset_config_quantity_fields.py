"""DirectExtractionItemSchema's quantity-field types, across all 4 dataset
configs that define one.

2026-09-24-gptoss120b-parsequantity-schema-diag-{01,02} found gpt-oss-120b's
guided decoding stalls on a str-only point_value/lower/upper/tolerance/
standard_deviation schema and fixed it by widening those fields to
str | float | None. That shape is independently redefined in 3 places (see
the experiment note's "Net picture for a fix"): MeasurementLM's
ParseQuantityResponse and MeasurementLMv2's QuantityItem (both covered in
tests/test_measurementlm.py / tests/test_measurementlmv2.py), and each
dataset config's own DirectExtractionItemSchema, used by Ablation 1's
single-call direct extraction (and, incidentally, by the NuExtract/
NuExtract3/LangExtract baseline schema builders). This file verifies the
third place independently for every dataset that has one -- pond_ten/
nfix_ten's DirectExtractionItemSchema has no qualifier/shape fields at all
(a simpler, pre-qualifier schema) and is out of scope.
"""
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "experiments"))
sys.path.insert(0, str(_REPO / "src"))

from experiments.run_extraction import load_dataset_config  # noqa: E402

_SCALAR_FIELDS = ("point_value", "lower", "upper", "tolerance", "standard_deviation")


@pytest.mark.parametrize("dataset", ["pond", "nfix", "supermat", "measeval"])
def test_direct_extraction_schema_scalar_fields_accept_str_or_float(dataset):
    schema = load_dataset_config(dataset).direct_extraction_schema
    props = schema.model_json_schema()["properties"]
    for field in _SCALAR_FIELDS:
        any_of = props[field]["anyOf"]
        assert [t.get("type") for t in any_of] == ["string", "number", "null"], (
            f"{dataset}.DirectExtractionItemSchema.{field}: expected "
            f"str | float | None (in that order), got {any_of}"
        )
    assert props["list_values"]["anyOf"] == [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]
