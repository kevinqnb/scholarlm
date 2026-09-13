#!/bin/bash -l
# experiments/submit.sh <id> [--dry-run]
#
# General-purpose submission wrapper for every Tier-1 experiment type (see
# experiments/utils.py's EXPERIMENT_TYPES). Resolves the experiment's
# runner, GPU need, and SGE resource request via experiments/_resolve_job.py,
# then qsubs experiments/_submit_job.sh <id> -- which brings up a vLLM
# server in-job when the model needs one, waits for it to answer /health,
# and calls the resolved runner directly with the experiment's own config
# path. Every experiment type goes through this one script now; there is no
# separate serve-then-call step.
#
# Replaces scripts/submit.sh (extraction/ablation only, via a separate
# scripts/run_experiment.py adapter) -- this dispatches by experiment-type
# for every Tier-1 type, and runners take their own config path directly
# instead of going through an adapter that translates params into CLI flags.
#
# SGE job names cannot start with a digit, and contract ids always do
# (YYYY-MM-DD-slug-NN), so the qsub -N name is "x<id>", not "<id>" verbatim
# -- this only affects the human-readable qstat name.
#
# --dry-run prints the resolved qsub command without submitting anything.
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ] || { [ $# -eq 2 ] && [ "$2" != "--dry-run" ]; }; then
    echo "usage: bash experiments/submit.sh <id> [--dry-run]" >&2
    exit 1
fi
ID="$1"
DRY_RUN=0
[ "${2:-}" = "--dry-run" ] && DRY_RUN=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PY="$REPO_ROOT/.venv/bin/python"
[ -x "$PY" ] || PY=python3

eval "$("$PY" "$REPO_ROOT/experiments/_resolve_job.py" "$ID")"

: "${SGE_PROJECT:?SGE_PROJECT is not set -- export it in your cluster profile}"

OUT_DIR="$REPO_ROOT/experiments/experiment-configs/$DATASET/$EXPERIMENT_TYPE/$ID/out"
mkdir -p "$OUT_DIR"
JOB_NAME="x${ID}"
QSUB_ARGS=(-N "$JOB_NAME" -P "$SGE_PROJECT" -j y -o "$OUT_DIR/${ID}.log" -m e)

if [ "$GPU_NEED" = "vllm_server" ] || [ "$GPU_NEED" = "direct_gpu" ]; then
    # Local model (vLLM-served or NNsight-direct): resource request comes
    # from the model-config's resources: block (experiments/model-configs/
    # is the single source of truth for both this and the actual server
    # launch parameters in experiments/_submit_job.sh).
    QSUB_ARGS+=(
        -l "h_rt=${WALLTIME}"
        -pe omp "$OMP"
        -l gpus=1
        -l "gpu_memory=${GPU_MEMORY}"
        -l "gpu_c=${GPU_C}"
    )
    # GPU_TYPE is only emitted when the model-config's resources: block sets
    # gpu_type -- most models don't need to pin one, a gpu_c floor is enough.
    if [ -n "${GPU_TYPE:-}" ]; then
        QSUB_ARGS+=(-l "gpu_type=${GPU_TYPE}")
    fi
else
    # Frontier model, or no model at all: no GPU request. There is no
    # serve/resources section to default a walltime from, so
    # _resolve_job.py already required params.walltime to be set explicitly
    # in the experiment's own config -- inventing a default here would be
    # exactly the magic number the contract forbids.
    QSUB_ARGS+=(-l "h_rt=${WALLTIME}")
fi

echo "Submitting ${ID} as SGE job '${JOB_NAME}' (type=${EXPERIMENT_TYPE}, runner=${RUNNER}, gpu_need=${GPU_NEED})"

if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] would run: qsub ${QSUB_ARGS[*]} $REPO_ROOT/experiments/_submit_job.sh $ID"
    exit 0
fi

qsub "${QSUB_ARGS[@]}" "$REPO_ROOT/experiments/_submit_job.sh" "$ID"
