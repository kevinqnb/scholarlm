"""Pond extraction experiment — Ablation 1: Combined Entity-Attribute Extraction

Uses MeasurementLMAblation1, which collapses entity identification and attribute
detection into a single step that emits (entity, attribute) pairs directly.

Key differences from extract.py:
- POND_IDENTIFICATION_PROMPT instructs the model to pair each ecosystem with the
  attributes for which a direct numerical measurement is reported.
- ObservationAttributePairSchema extends the entity schema with 'attribute' and
  'attribute_terms' fields required by MeasurementLMAblation1.
- extract_entity_attribute_pairs() replaces both extract_entities() and
  detect_attributes() steps.
- entity_attribute_provenance() replaces both separate provenance steps.
- extract_values(), standardize_and_deduplicate() are unchanged in structure.
"""

import os
import json
import pandas as pd
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
from scholarlm.measurementlm_ablation1 import MeasurementLMAblation1
from scholarlm.measurementlm import NumpyEncoder
from scholarlm.utils import get_filenames_in_directory

from extract_prompts import POND_IDENTIFICATION_PROMPT_WITH_ATTRIBUTES

import random
import torch
random.seed(342)
torch.manual_seed(342)
torch.cuda.manual_seed(342)


main_directory = "data/pond"
ocr_directory = os.path.join(main_directory, "ocr_output_cleaned_gpt_5_mini")
with open(os.path.join(main_directory, "directory.json"), "r") as f:
    paper_info = json.load(f)

text_files = get_filenames_in_directory(ocr_directory, ignore=[".DS_Store", ".gitkeep"])
text_files.sort()

text_filepaths = []
text_info = []
for f in text_files:
    paper_code = f.replace(".txt", "")
    filepath = os.path.join(ocr_directory, f)
    metadata = paper_info.get(paper_code, {})
    metadata["paper_code"] = paper_code
    text_filepaths.append(filepath)
    text_info.append(metadata)

text = []
for filepath in text_filepaths:
    with open(filepath, "r", encoding="utf-8") as file:
        text.append(file.read())


# --- Ablation 1: Combined entity-attribute schema ---
#
# Extends the baseline entity schema with 'attribute' and 'attribute_terms'
# fields required by MeasurementLMAblation1.
class ObservationAttributePairSchema(BaseModel):
    # Entity identification fields (same as baseline ObservationSchema)
    name: str | None
    abbreviations: str | None
    location: str | None
    site: str | None
    state: str | None
    date: str | None
    ecosystem: str | None
    # Attribute fields (required by MeasurementLMAblation1)
    attribute: str
    attribute_terms: list[str]


attribute_info_dict = {
    "latitude": {
        "description": "Geographic latitude of the ecosystem location, expressed in a standard geographic coordinate system (e.g., WGS84). This should refer to the centroid or stated reference point of the ecosystem, not a bounding box or region.",
        "units": ["degrees", "radians"]
    },
    "longitude": {
        "description": "Geographic longitude of the ecosystem location, expressed in a standard geographic coordinate system (e.g., WGS84). This should refer to the centroid or stated reference point of the ecosystem, not a bounding box or region.",
        "units": ["degrees", "radians"]
    },
    "surface_area": {
        "description": "Surface area of the water body itself, representing the horizontal area of open water or the stated ecosystem boundary at the time of measurement or description. This is NOT the same as watershed area, drainage basin area, catchment area, or littoral zone area.",
        "units": ["km^2", "mi^2", "ha", "m^2", "acres"]
    },
    "max_depth": {
        "description": "Maximum physical water depth of the ecosystem, defined as the deepest point of the water body at the time of measurement or as reported in the source. This is NOT the same as mean depth, average depth, or Secchi depth (water transparency depth).",
        "units": ["m", "km", "ft"]
    },
    "vegetation_cover": {
        "description": "Fraction or percentage of the ecosystem surface area covered by aquatic macrophytes or other rooted/floating aquatic vegetation. This should refer to areal coverage, not biomass or volume. This is NOT the same as algal cover, periphyton cover, or phytoplankton density.",
        "units": ["percent", "fraction"]
    },
    "ph": {
        "description": "pH of the water, i.e., the negative logarithm of the hydrogen ion activity. This is a dimensionless quantity and should refer to a measured water pH value, not soil or sediment pH.",
        "units": []
    },
    "tn": {
        "description": "Total nitrogen (TN) concentration in the water column, representing the sum of all nitrogen forms — both dissolved and particulate, including nitrate (NO₃), nitrite (NO₂), ammonium (NH₃/NH₄⁺), and organic nitrogen. This must be the aggregate 'total nitrogen' value as explicitly reported in the source. This is NOT the same as individual nitrogen species (e.g., NO₃ alone, NO₂ alone, NH₃ alone, combined NO₃+NO₂, or particulate organic nitrogen [PON]) unless they are explicitly labeled as total nitrogen.",
        "units": ["µg/L", "mg/L", "μmol/L", "ppm", "ppb"]
    },
    "tp": {
        "description": "Total phosphorus (TP) concentration in the water column, representing the sum of all phosphorus forms — both dissolved and particulate. This must be the aggregate 'total phosphorus' value as explicitly reported in the source. This is NOT the same as individual phosphorus species (e.g., soluble reactive phosphorus [SRP], orthophosphate [PO₄³⁻], dissolved reactive phosphorus [DRP], or particulate phosphorus [PP]) unless they are explicitly labeled as total phosphorus.",
        "units": ["µg/L", "mg/L", "μmol/L", "ppm", "ppb"]
    },
    "chla": {
        "description": "Chlorophyll-a (Chl-a) concentration in the water column, used as a proxy for phytoplankton biomass. This should refer to extracted or in situ chlorophyll-a measurements only. This is NOT the same as total chlorophyll, chlorophyll-b, chlorophyll-c, pheophytin, or other pigment measurements unless they are explicitly labeled as chlorophyll-a.",
        "units": ["µg/L", "mg/L", "mg/m^3"]
    },
}


measurementlm = MeasurementLMAblation1(
    model_name="gaunernst/gemma-3-27b-it-qat-autoawq",
    entity_identification_prompt=POND_IDENTIFICATION_PROMPT_WITH_ATTRIBUTES,
    entity_identification_schema=ObservationAttributePairSchema,
    attribute_info_dict=attribute_info_dict,
    sampling_params={
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 8192,
        "seed": 342,
    },
)


def _serialize_prov(prov_dict):
    """Serialize tuple-keyed provenance dict for JSON (keys become 'doc_id|item_id' strings)."""
    return {f"{k[0]}|{k[1]}": v for k, v in prov_dict.items()}


def _deserialize_prov(json_dict):
    """Deserialize JSON provenance dict back to tuple keys."""
    out = {}
    for k, v in json_dict.items():
        parts = k.split("|", 1)
        try:
            doc_id = int(parts[0])
        except ValueError:
            doc_id = parts[0]
        out[(doc_id, parts[1])] = v
    return out


def extract_entity_attribute_pairs(text, outfile):
    """Step 1 (ablation 1): Extract (entity, attribute) pairs in a single pass."""
    print("Extracting (entity, attribute) pairs...")
    data = []
    for i, paper in enumerate(text):
        data.append({"document_id": i, "context": paper})

    measurementlm.data = data
    pair_data = measurementlm._extract_entity_attribute_pairs()

    # Strip context before saving
    save_data = [{k: v for k, v in r.items() if k != "context"} for r in pair_data]
    with open(outfile, "w") as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)


def entity_attribute_provenance(text, pairs_file, outfile):
    """Step 2 (ablation 1): Combined (entity, attribute) pair provenance."""
    print("Running entity-attribute pair provenance...")
    with open(pairs_file, "r") as f:
        pair_data = json.load(f)

    for record in pair_data:
        doc_id = record["document_id"]
        record["context"] = text[doc_id]

    prov = measurementlm._entity_attribute_provenance(pair_data)

    with open(outfile, "w") as f:
        json.dump(_serialize_prov(prov), f, indent=4, ensure_ascii=False)


def extract_values(text, pairs_file, pair_prov_file, outfile):
    """Steps 3+4 (ablation 1): Extract values from text and tables using pair provenance.

    Adapts pair_prov into the (entity_prov, attr_prov, doc_attributes) format
    expected by the unchanged extraction methods, mirroring the adaptation done
    in MeasurementLMAblation1.fit().
    """
    print("Extracting values...")
    with open(pairs_file, "r") as f:
        pair_data = json.load(f)

    with open(pair_prov_file, "r") as f:
        pair_prov = _deserialize_prov(json.load(f))

    for record in pair_data:
        doc_id = record["document_id"]
        record["context"] = text[doc_id]

    # Adapt pair_prov to the interface expected by the extraction methods.
    entity_prov = pair_prov
    attr_prov = {}
    doc_attributes = {}
    for record in pair_data:
        doc_id = record["document_id"]
        entity_id = record["entity_id"]
        attr_name = record["attribute"]
        terms = record.get("attribute_terms", [])

        doc_attributes.setdefault(doc_id, {})[attr_name] = terms

        attr_key = (doc_id, attr_name)
        for entry in pair_prov.get((doc_id, entity_id), []):
            attr_prov.setdefault(attr_key, []).append(entry)

    text_values = measurementlm._extract_values_from_text(
        pair_data, doc_attributes, entity_prov, attr_prov
    )
    table_values = measurementlm._extract_values_from_tables(
        pair_data, doc_attributes, entity_prov, attr_prov
    )
    data = text_values + table_values

    with open(outfile, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)


def standardize_and_deduplicate(infile, outfile):
    """Steps 5+6 (ablation 1): Standardize then deduplicate."""
    print("Standardizing and deduplicating...")
    with open(infile, "r") as f:
        data = json.load(f)

    measurementlm.data = data
    standardized = measurementlm._standardize()
    deduplicated = measurementlm._deduplicate(standardized)

    dataset = []
    for i, datapoint in enumerate(deduplicated):
        document_id = datapoint["document_id"]
        doc_metadata = text_info[document_id]
        dataset.append(doc_metadata | datapoint | {'measurement_id': i})

    with open(outfile, "w") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)


outfile1 = "data/experiments/2026_03_04/ablation1_pairs.json"
extract_entity_attribute_pairs(text, outfile1)

outfile2 = "data/experiments/2026_03_04/ablation1_pair_prov.json"
entity_attribute_provenance(text, outfile1, outfile2)

outfile3 = "data/experiments/2026_03_04/ablation1_values.json"
extract_values(text, outfile1, outfile2, outfile3)

outfile4 = "data/experiments/2026_03_04/ablation1_final.json"
standardize_and_deduplicate(outfile3, outfile4)
