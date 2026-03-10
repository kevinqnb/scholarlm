import json

judgement_files_dict = {
    "gpt": "data/experiments/2026_02_25/pond_openai_judged_gpt.json",
    "gemini": "data/experiments/2026_02_25/pond_openai_judged_gemini.json",
    "claude": "data/experiments/2026_02_25/pond_openai_judged_claude.json",
    "llama": "data/experiments/2026_02_25/pond_openai_judged_llama.json",
}

voting_models = ["gpt", "gemini", "claude"]
voting_threshold = 2

data_combined_dict = {}

for j, jdf in judgement_files_dict.items():
    with open(jdf, "r") as f:
        result_dict = json.load(f)

    for entry in result_dict:
        eid = entry["measurement_id"]
        filters  = [
            "judgement",
            "judgement_model",
            "judgement_prob",
            "judgement_p_true",
            "judgement_p_false",
            "judgement_logit_p_true",
            "judgement_logit_p_false",
            "judgement_raw_text",
        ]
        entry_filter = {k: v for k, v in entry.items() if k not in filters}
        for k in filters:
            if k in entry and k != "judgement_model":  # keep judgement_model as is for reference
                entry_filter[k + f"_{j}"] = entry[k]

        if eid not in data_combined_dict:
            data_combined_dict[eid] = entry_filter
        else:
            data_combined_dict[eid] = data_combined_dict[eid] | entry_filter


data_combined = list(data_combined_dict.values())

for entry in data_combined:
    valid_vote = 0
    for j in voting_models:
        jud_key = f"judgement_{j}"
        jud_result = entry[jud_key]
        if jud_result is True:
            valid_vote += 1
            
    entry["judgement_combined"] = valid_vote >= voting_threshold


output_file = f"data/experiments/2026_03_04/pond_judged_combined.json"
with open(output_file, "w") as f:
    json.dump(data_combined, f, indent=4, ensure_ascii=False)
    