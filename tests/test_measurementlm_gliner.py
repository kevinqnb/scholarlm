"""Unit tests for the GLiNER2 baseline adapter (measurementlm_gliner.py).

Rung 1 of the staged-gate ladder (see CLAUDE.md): hand-built fixtures, no real
GLiNER2 model loaded (``__init__`` calls ``GLiNER2.from_pretrained``, which
downloads/loads a real local checkpoint -- inappropriate for a fast unit
test), so instances are built with ``object.__new__`` and only the attributes
the methods under test actually read.

Covers the qualifier-field build in this session:
  * `_build_structure` requests the qualifier/shape fields (QUANTITY_FIELD_NAMES)
    as scalar `dtype="str"` GLiNER fields -- GLiNER2 has no array dtype, so
    (per its own docstring) `qualifiers`/`list_values` are comma-joined
    strings here, not JSON lists, unlike every other extraction method.
  * `_make_record` merges the `quantity` dict into the record alongside
    `attribute`/`value`/`units`.
"""
import sys
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scholarlm.measurementlm_gliner import QUANTITY_FIELD_NAMES, MeasurementLMGliner


class _EntitySchema(BaseModel):
    name: str | None
    location: str | None


class _EventSchema(BaseModel):
    date: str | None


_ATTRIBUTE_INFO = {
    "depth": {"description": "Maximum depth", "units": ["m"]},
}


class _FakeFieldBuilder:
    def __init__(self):
        self.fields: list[tuple[str, str, str]] = []  # (name, dtype, description)

    def field(self, name, dtype, description):
        self.fields.append((name, dtype, description))
        return self


class _FakeSchema:
    def __init__(self):
        self.builder = None
        self.built = False

    def structure(self, name):
        self.builder = _FakeFieldBuilder()
        return self.builder

    def build(self):
        self.built = True


class _FakeExtractor:
    def create_schema(self):
        return _FakeSchema()


def _make_gliner(**overrides):
    """Bare MeasurementLMGliner instance -- bypasses __init__ (and its real
    GLiNER2 model load) entirely, wiring only what _build_structure/_make_record
    read."""
    mlm = object.__new__(MeasurementLMGliner)
    mlm.extractor = _FakeExtractor()
    mlm.attribute_info_dict = _ATTRIBUTE_INFO
    mlm.entity_identification_schema = _EntitySchema
    mlm.measurement_event_schema = _EventSchema
    mlm.gliner_property_names = {}
    mlm.gliner_entity_description = None
    mlm.entity_type_description = "a distinct aquatic ecosystem"
    mlm.gliner_field_descriptions = {"location": "the ecosystem's location", "date": "the measurement date"}
    mlm.include_qualifiers = True
    mlm.chunk_size = 384
    mlm.chunk_overlap = 64
    mlm.threshold = 0.5
    mlm.batch_size = 8
    for k, v in overrides.items():
        setattr(mlm, k, v)
    return mlm


# ---------------------------------------------------------------------------
# _build_structure
# ---------------------------------------------------------------------------


def test_build_structure_requests_all_quantity_fields_as_scalar_str():
    mlm = _make_gliner()
    struct_name, schema = mlm._build_structure("depth")

    assert struct_name == "depth"
    assert schema.built
    field_names = [name for name, _, _ in schema.builder.fields]
    for f in QUANTITY_FIELD_NAMES:
        assert f in field_names
    # Every quantity field is scalar -- GLiNER2 has no array dtype.
    quantity_dtypes = {name: dtype for name, dtype, _ in schema.builder.fields if name in QUANTITY_FIELD_NAMES}
    assert set(quantity_dtypes.values()) == {"str"}


def test_build_structure_value_field_no_longer_asks_for_central_value_only():
    """The value field's description used to say 'give the mean or central
    value... do not include uncertainty measures, confidence intervals, or
    range bounds' -- that instruction must be gone now that value is meant
    to carry the full raw report."""
    mlm = _make_gliner()
    _, schema = mlm._build_structure("depth")
    value_desc = next(desc for name, _, desc in schema.builder.fields if name == "value")
    assert "central value" not in value_desc.lower()
    assert "do not include uncertainty" not in value_desc.lower()
    assert "in full" in value_desc.lower()


def test_build_structure_includes_dataset_configured_extra_fields():
    mlm = _make_gliner()
    _, schema = mlm._build_structure("depth")
    field_names = [name for name, _, _ in schema.builder.fields]
    assert "location" in field_names  # from gliner_field_descriptions
    assert "date" in field_names      # event field from gliner_field_descriptions


def test_build_structure_include_qualifiers_false_omits_quantity_fields():
    """params.include_qualifiers=false (run_baseline_gliner.py) -- the GLiNER
    counterpart of Ablation 1/NuExtract3/LangExtract's own flag."""
    mlm = _make_gliner(include_qualifiers=False)
    _, schema = mlm._build_structure("depth")
    field_names = [name for name, _, _ in schema.builder.fields]
    for f in QUANTITY_FIELD_NAMES:
        assert f not in field_names
    # name/value/units and dataset-configured extra fields are untouched.
    assert {"name", "value", "units", "location", "date"} <= set(field_names)


# ---------------------------------------------------------------------------
# _make_record
# ---------------------------------------------------------------------------


def _quantity(**overrides):
    base = {f: None for f in QUANTITY_FIELD_NAMES}
    base.update(overrides)
    return base


def test_make_record_merges_quantity_fields_into_record():
    mlm = _make_gliner()
    record = mlm._make_record(
        doc_idx=0, attribute="depth", name="Lake A", value="3-7", units="m",
        quantity=_quantity(qualifiers="IsRange", lower="3", upper="7"),
        page_num=1, extra={"location": "WI"},
    )

    assert record["attribute"] == "depth"
    assert record["value"] == "3-7"
    assert record["units"] == "m"
    assert record["qualifiers"] == "IsRange"  # comma-joined string, not a list
    assert record["lower"] == "3"
    assert record["upper"] == "7"
    assert record["point_value"] is None
    assert record["name"] == "Lake A"
    assert record["location"] == "WI"
    assert record["document_id"] == 0
    assert record["page_number"] == 1


def test_make_record_defaults_quantity_fields_to_none():
    mlm = _make_gliner()
    record = mlm._make_record(
        doc_idx=0, attribute="depth", name="Lake A", value="3.2", units="m",
        quantity=_quantity(), page_num=1, extra={},
    )
    for f in QUANTITY_FIELD_NAMES:
        assert record[f] is None


# ---------------------------------------------------------------------------
# _extract_records: include_qualifiers gates whether the 7 quantity keys are
# present at all on the final record, not just whether they're populated --
# matching Ablation 1/NuExtract3/LangExtract's no-qualifiers shape (missing
# keys, not None-valued) so every method's final.json has the same column
# set for a given include_qualifiers setting.
# ---------------------------------------------------------------------------


class _FakeBatchExtractor(_FakeExtractor):
    def __init__(self, canned_result):
        self.canned_result = canned_result

    def batch_extract(self, texts, schemas, **kwargs):
        return [self.canned_result for _ in texts]


_CANNED_ITEM = {
    "name": "Lake A", "value": "3-7", "units": "m",
    "qualifiers": "IsRange", "point_value": None, "lower": "3", "upper": "7",
    "list_values": None, "tolerance": None, "standard_deviation": None,
}


def test_extract_records_include_qualifiers_true_keeps_quantity_keys():
    mlm = _make_gliner(
        extractor=_FakeBatchExtractor({"depth": [dict(_CANNED_ITEM)]}),
        include_qualifiers=True,
    )
    records = mlm._extract_records(["Lake A has a depth of 3-7 m."])
    assert len(records) == 1
    for f in QUANTITY_FIELD_NAMES:
        assert f in records[0]
    assert records[0]["lower"] == "3"
    assert records[0]["upper"] == "7"


def test_extract_records_include_qualifiers_false_omits_quantity_keys():
    mlm = _make_gliner(
        extractor=_FakeBatchExtractor({"depth": [dict(_CANNED_ITEM)]}),
        include_qualifiers=False,
    )
    records = mlm._extract_records(["Lake A has a depth of 3-7 m."])
    assert len(records) == 1
    for f in QUANTITY_FIELD_NAMES:
        assert f not in records[0]
    assert records[0]["value"] == "3-7"
    assert records[0]["name"] == "Lake A"
