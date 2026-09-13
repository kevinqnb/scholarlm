"""experiments/_resolve_job.py <experiment-id>

Resolves everything experiments/submit.sh and experiments/_submit_job.sh need
to submit and run one experiment: which runner script, what SGE resource
request (if any), and -- for vLLM-served models -- the full server launch
parameters. Prints shell `KEY=VALUE` (and `KEY=(array)` for extra_vllm_args)
lines meant to be `eval`'d by the caller.

Every value is passed through shlex.quote(), so this is safe against
embedded quotes/spaces -- e.g. nuextract-2.0-8b's extra_vllm_args entry
--limit-mm-per-prompt '{"image": 2}' round-trips correctly through eval,
which a naive f-string KEY=VALUE emitter would not handle correctly.

Replaces scripts/_resolve_model.py (model -> SGE resources only): submit.sh
now dispatches by experiment-type, not just by model, so this resolver also
picks the runner script and validates the experiment config exists and is
well-formed before any qsub happens.

On error, prints a single `echo '...' >&2; exit 1` line instead of raising --
the caller `eval`s our stdout, so this is what makes that eval fail loud
rather than silently eval'ing nothing.
"""
from __future__ import annotations

import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import utils


def _fail(message: str) -> int:
    print(f"echo {shlex.quote('error: ' + message)} >&2; exit 1")
    return 0  # exit 0 ourselves -- the eval'd `exit 1` is what signals failure


def _emit(key: str, value: object) -> None:
    print(f"{key}={shlex.quote(str(value))}")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: _resolve_job.py <experiment-id>", file=sys.stderr)
        return 1
    experiment_id = args[0]

    try:
        job = utils.resolve_job(experiment_id)
    except (ValueError, FileNotFoundError) as e:
        return _fail(str(e))

    _emit("RUNNER", job["runner"])
    _emit("DATASET", job["dataset"])
    _emit("EXPERIMENT_TYPE", job["experiment_type"])
    _emit("CONFIG_PATH", job["config_path"])
    _emit("GPU_NEED", job["gpu_need"])

    gpu_need = job["gpu_need"]
    walltime = job["params"].get("walltime")

    if gpu_need in ("vllm_server", "direct_gpu"):
        resources = job["model_config"]["resources"]
        _emit("GPU_MEMORY", resources["gpu_memory"])
        _emit("GPU_C", resources["gpu_capability"])
        _emit("OMP", resources["omp"])
        _emit("WALLTIME", walltime or resources["walltime"])
        # Optional: pin a specific GPU type rather than just a capability
        # floor. Needed for some NNsight jobs on this cluster -- a gpu_c
        # floor alone let SGE schedule onto newer GPU types the installed
        # torch build (2.7.1+cu126, stops at sm_90) can't actually run on
        # (see qwen-2.5-7b's interp_judge model-config for the documented
        # case this was added for). Most model-configs don't need this.
        if "gpu_type" in resources:
            _emit("GPU_TYPE", resources["gpu_type"])
    else:
        # Frontier model, or no model at all: no serve/resources section to
        # default a walltime from -- params.walltime is required, per the
        # contract's "no magic numbers" rule (matches today's
        # scripts/submit.sh behavior for frontier-model configs).
        if not walltime:
            return _fail(
                f"{job['config_path']}: params.walltime is required for "
                f"experiment-type {job['experiment_type']!r} (no serve/resources "
                "section to default from)"
            )
        _emit("WALLTIME", walltime)

    if gpu_need == "vllm_server":
        mc = job["model_config"]
        _emit("MODEL_ID", mc["model_id"])
        _emit("SERVE_PORT", mc["serve"]["port"])
        _emit("MAX_MODEL_LEN", mc["serve"]["max_model_len"])
        _emit("GPU_MEMORY_UTILIZATION", mc["serve"]["gpu_memory_utilization"])
        _emit("QUANTIZATION", mc["serve"].get("quantization") or "")
        _emit("DTYPE", mc["serve"]["dtype"])
        _emit("SIF_IMAGE", mc["serve"]["sif_image"])
        extra_args = mc["serve"].get("extra_vllm_args", [])
        print("EXTRA_VLLM_ARGS=(" + " ".join(shlex.quote(a) for a in extra_args) + ")")
    elif gpu_need == "direct_gpu":
        _emit("MODEL_ID", job["model_config"]["model_id"])
    elif "model" in job:
        # "none" but still has a model (frontier API) -- runners resolve the
        # rest (api_base, sampling_params) from the model-config themselves.
        _emit("MODEL_ID", job["model_config"]["model_id"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
