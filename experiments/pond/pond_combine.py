import json
import glob
from scholarlm.measurementlm import NumpyEncoder

input_pattern = "data/experiments/2026_02_25/pond*_vllm.json"
outfile = "data/experiments/2026_02_25/pond_vllm.json"

combined = []
for filepath in sorted(glob.glob(input_pattern)):
    with open(filepath, 'r') as f:
        data = json.load(f)
    combined.extend(data)
    print(f"Loaded {len(data)} records from {filepath}")

# Sort by global document_id for consistency
combined.sort(key=lambda x: x.get('document_id', 0))

print(f"\nTotal records: {len(combined)}")
print(f"Unique document_ids: {len(set(r['document_id'] for r in combined))}")

# add measurement_id
for i, record in enumerate(combined):
    record['measurement_id'] = i

with open(outfile, 'w') as f:
    json.dump(combined, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

print(f"Combined output saved to {outfile}")