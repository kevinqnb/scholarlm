import sys
from pathlib import Path

REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'experiments'))
sys.path.insert(0, str(REPO_ROOT))

from analysis.loaders import load_trained_probe

DATASETS    = ['pond', 'nfix']
JUDGE_MODELS = ['llama-3.1-8b', 'mistral-7b', 'qwen-2.5-7b']
PROBE_TYPE  = 'head'

cache = {}
for ds in DATASETS:
    for jm in JUDGE_MODELS:
        pd_data = load_trained_probe(ds, jm, ptype=PROBE_TYPE)
        heads = pd_data['top_k_heads']
        cache[(ds, jm)] = set(map(tuple, heads))
        print(f"{ds:6s}  {jm:20s}  heads={heads}")

print()
for jm in JUDGE_MODELS:
    sets = [cache[(ds, jm)] for ds in DATASETS]
    shared = sets[0].intersection(*sets[1:])
    print(f"{jm:20s}  shared={sorted(shared)}  ({len(shared)} heads)")
