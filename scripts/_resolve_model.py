"""Print one model's serving facts as shell KEY=VALUE lines, for `eval` by
scripts/submit.sh and scripts/_run_experiment_job.sh.

Reuses experiments/model_registry.py (to decide frontier vs. local) and
experiments/config.yaml (for the local model's real SGE serve resources) --
it does not duplicate either.

Usage: python scripts/_resolve_model.py <model_key>
"""
from __future__ import annotations

import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))


def main() -> int:
    if len(sys.argv) != 2:
        print("echo 'usage: _resolve_model.py <model_key>' >&2; exit 1")
        return 1
    model = sys.argv[1]

    from model_registry import MODEL_REGISTRY

    if model not in MODEL_REGISTRY:
        print(f"echo 'error: unknown model {shlex.quote(model)}' >&2; exit 1")
        return 1

    model_config = MODEL_REGISTRY[model]
    is_frontier = model_config.api_base is not None

    print(f"MODEL_ID={shlex.quote(model_config.model_id)}")
    print(f"NEEDS_SERVER={'0' if is_frontier else '1'}")

    if is_frontier:
        return 0

    from utils import load_config

    cfg = load_config()
    serve = cfg.get("models", {}).get(model, {}).get("serve")
    if serve is None:
        print(
            f"echo 'error: model {shlex.quote(model)} has no serve section in "
            "experiments/config.yaml' >&2; exit 1"
        )
        return 1

    print(f"SERVE_PORT={int(serve['port'])}")
    print(f"GPU_MEMORY={shlex.quote(str(serve['gpu_memory']))}")
    print(f"GPU_C={shlex.quote(str(serve['gpu_capability']))}")
    print(f"SERVE_WALLTIME={shlex.quote(str(serve['walltime']))}")
    print(f"SERVE_OMP={int(serve['omp'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
