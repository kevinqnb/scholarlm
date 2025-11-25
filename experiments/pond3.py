import json
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

with open("data/pond_adversarial_test_paged.json", "r") as f:
    result_dict = json.load(f)

instructions = (
    f"You are an expert in extracting precise numerical data from user provided, scientific text. "
    f"You will be queried with a description of an specific entity to be measured, along with the measurement type to report for. "
    f"Your task is to extract the corresponding value if it appears in the provided context. "
    f"Respond 'None' if the the given entity or measurement feature do not appear in the context. "
    f"Respond 'None' if the context does not explicity provide data for the given feature and entity. "
    f"Respond 'None' if the data reported is not either a direct numerical measurement or a mean of numerical measurements. "
    f"Respond 'None' for if the data reported only contains values for parameter estimates or other statistical measures of fit. "
    f"Respond 'None' for ranges of values, inequalties, or other cases where there is not a clear choice for a single numerical value. "
    f"Respond with the extracted value only if the context explicity provides a direct numerical value measured for the given feature, with respect to the entity in question. "
    f"Copy the value exactly as it appears in the context. "
    f"Give the value only, and do not include any units of measurement, descriptors, or explanation in your response. "
    f"If the value is associated with uncertainty measures (e.g., ± values, confidence intervals), report only the central value without any uncertainty information. "
)

messages = []
for i, item in enumerate(result_dict):
    context = item['context']
    entity = {k: v for k, v in item.items() if k in ['name', 'date', 'location', 'ecosystem']}
    measurement = item['measurement']
    query = "Extract the value of " + f"{measurement} for the entity {entity}."
    message = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": f"## Context:\n{context}\n\n## Query:\n{query}"},
    ]
    messages.append(message)


sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=20,
    top_p=0.95,
    top_k=64,
)

llm = LLM("meta-llama/Llama-3.1-8B-Instruct")

responses = llm.chat(messages = messages, sampling_params = sampling_params)
response_texts = [r.outputs[0].text for r in responses]

updated_result_dict = []
for i, resp in enumerate(response_texts):
    if resp.strip().lower() != 'none':
        item = result_dict[i]
        item['value'] = resp.strip()
        updated_result_dict.append(item)

outfile = f"data/pond_adversarial_test_paged_llama.json"
with open(outfile, 'w') as f:
    json.dump(updated_result_dict, f, indent=4)