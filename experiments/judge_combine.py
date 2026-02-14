import json

judgement_files_dict = {
    "gpt": "data/experiments/2026_02_11/new_ten_judged_gpt.json",
    "gemini": "data/experiments/2026_02_11/new_ten_judged_gemini.json",
    "claude": "data/experiments/2026_02_11/new_ten_judged_claude.json",
    "llama": "data/experiments/2026_02_11/new_ten_judged_llama.json",
}

data_combined_dict = {}

for j, jdf in judgement_files_dict.items():
    with open(jdf, "r") as f:
        result_dict = json.load(f)

    for entry in result_dict:
        eid = entry["measurement_id"]
        entry_filter = {k: v for k, v in entry.items() if k not in ["judgement", "judgement_model", "judgement_confidence"]}
        entry_filter[f"judgement_{j}"] = entry["judgement"]
        entry_filter[f"judgement_confidence_{j}"] = entry.get("judgement_confidence", "")
        if eid not in data_combined_dict:
            data_combined_dict[eid] = entry_filter
        else:
            data_combined_dict[eid] = data_combined_dict[eid] | entry_filter


data_combined = list(data_combined_dict.values())

for entry in data_combined:
    valid_vote = 0
    for j in judgement_files_dict.keys():
        jud_key = f"judgement_{j}"
        jud_result = entry[jud_key]
        if jud_result is True:
            valid_vote += 1
            
    entry["judgement_combined"] = valid_vote >= 2


output_file = f"data/experiments/2026_02_11/new_ten_judged_combined.json"
with open(output_file, "w") as f:
    json.dump(data_combined, f, indent=4, ensure_ascii=False)
    