import os
import json
import random
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

from scholarlm import JudgementLM

# (try to) set seeds for reproducibility
import random
import torch
random.seed(342)
torch.manual_seed(342)
torch.cuda.manual_seed(342)


####################################################################################################


class IdentificationSchema(BaseModel):
    name: str | None
    date: str | None
    location: str | None
    ecosystem: str | None
    model_config = {
        'title': 'Ecosystem Identifier',
        'entity_description': 'ponds, lakes, or wetlands',
        'primary_identifier': 'name'
    }

class MeasurementSchema(BaseModel):
    latitude: float | None = Field(
        description="latitude",
        json_schema_extra={'units': ["degrees", "radians"]}
    )
    longitude: float | None = Field(
        description="longitude",
        json_schema_extra={'units': ["degrees", "radians"]}
    )
    surface_area: float | None = Field(
        description="surface area",
        json_schema_extra={'units': ["km^2", "mi^2", "ha", "m^2", "acres"]}
    )
    max_depth: float | None = Field(
        description="maximum depth",
        json_schema_extra={'units': ["m", "km", "ft"]}
    )
    vegetation_cover: float | None = Field(
        description="aquatic macrophyte percent coverage",
        json_schema_extra={'units': ["percent", "fraction"]}
    )
    ph: float | None = Field(
        description="pH level",
        json_schema_extra={'units': None}
    )
    tn: float | None = Field(
        description="total nitrogen concentration",
        json_schema_extra={'units': ["µg/L", "mg/L", "μmol/L", "ppm", "ppb"]}
    )
    tp: float | None = Field(
        description="total phosphorus concentration",
        json_schema_extra={'units': ["µg/L", "mg/L", "μmol/L", "ppm", "ppb"]}
    )
    chla: float | None = Field(
        description="chlorophyll-a concentration",
        json_schema_extra={'units': ["µg/L", "mg/L", "mg/m^3"]}
    )


####################################################################################################


input_file = "data/12_03_25/pond_page_chunks2.json"

with open(input_file, "r") as f:
    result_dict = json.load(f)

judgelm = JudgementLM(
    #model_name = "gaunernst/gemma-3-27b-it-qat-autoawq",
    model_name = "cyankiwi/Olmo-3-32B-Think-AWQ-4bit",
    identification_schema=IdentificationSchema,
    measurement_schema=MeasurementSchema,
    sampling_params={
        "temperature": 0.0,
        "top_p" : 0.95,
        "top_k" : 64,
        "max_tokens" : 8192,
        "seed": 342,
    }
)

judged_data = judgelm.fit(data = result_dict)

output_file = "data/12_03_25/pond_page_chunks2_judged2_redo.json"
with open(output_file, "w") as f:
    json.dump(judged_data, f, indent=4, ensure_ascii=False)


####################################################################################################