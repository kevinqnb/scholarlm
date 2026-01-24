import json

validation_files_dict = {
    "gpt": "data/01_20_26/ten_validated_gpt.json",
    "gemini": "data/01_20_26/ten_validated_gemini.json",
    "claude": "data/01_20_26/ten_validated_claude.json",
}

data_combined_dict = {}

for v, vf in validation_files_dict.items():
    with open(vf, "r") as f:
        result_dict = json.load(f)

    for entry in result_dict:
        eid = entry["measurement_id"]
        entry_filter = {k: v for k, v in entry.items() if k not in ["validation", "validation_model", "validation_confidence"]}
        entry_filter[f"validation_{v}"] = entry["validation"]
        entry_filter[f"validation_confidence_{v}"] = entry.get("validation_confidence", "")
        if eid not in data_combined_dict:
            data_combined_dict[eid] = entry_filter
        else:
            data_combined_dict[eid] = data_combined_dict[eid] | entry_filter


data_combined = list(data_combined_dict.values())

for entry in data_combined:
    valid_vote = 0
    for v in validation_files_dict.keys():
        val_key = f"validation_{v}"
        val_result = entry[val_key].lower().strip()
        if "true" in val_result:
            valid_vote += 1
            entry[val_key] = True
        else:
            entry[val_key] = False

    entry["validation"] = valid_vote >= len(validation_files_dict) / 2


output_file = f"data/01_20_26/ten_validated_combined.json"
with open(output_file, "w") as f:
    json.dump(data_combined, f, indent=4, ensure_ascii=False)
    