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

for ds in DATASETS:
    for jm in JUDGE_MODELS:
        pd_data = load_trained_probe(ds, jm, ptype=PROBE_TYPE)
        heads = pd_data['top_k_heads']
        print(f"{ds:6s}  {jm:20s}  heads={heads}")
