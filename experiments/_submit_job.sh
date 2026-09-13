#!/bin/bash -l
# Job body for experiments/submit.sh. Not meant to be qsub'd directly by
# hand -- submit.sh works out the GPU/walltime request first, since those
# have to be known before the job is submitted.
#
# Re-resolves the same experiment (fresh shell on the compute node), brings
# up a vLLM server in-job when the model needs one -- absorbing
# experiments/gen_serve_script.py's role (one place that knows how to build
# a vLLM launch command from a model-config, including extra_vllm_args and
# the gemma-3-27b/A100 guard) -- waits for it to answer /health, then calls
# the resolved runner directly with the experiment's own config path. If a
# server is already answering on the model's port, it's reused instead of
# starting a second one. NNsight-direct and no-model experiment types skip
# the server dance entirely.
#
# SUBMIT_JOB_DRY_RUN=1 prints the vLLM launch command (if any) and the final
# runner invocation without actually running either -- for local testing of
# this script's own dispatch logic without singularity/GPU access.
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: _submit_job.sh <id>" >&2
    exit 1
fi
ID="$1"
DRY_RUN="${SUBMIT_JOB_DRY_RUN:-0}"

# REPO_ROOT comes from experiments/submit.sh's `qsub -v REPO_ROOT=...`, not
# self-located via ${BASH_SOURCE[0]} -- SGE copies this script to
# /var/spool/sge/<node>/job_scripts/<jobid> before executing it, so
# BASH_SOURCE would resolve to the spool path, not the repo checkout.
: "${REPO_ROOT:?REPO_ROOT is not set -- run this via experiments/submit.sh, not qsub directly}"
cd "$REPO_ROOT"

PY="$REPO_ROOT/.venv/bin/python"
[ -x "$PY" ] || PY=python3

# HuggingFace weights cache must never land in $HOME (CLAUDE.local.md storage
# discipline) -- HF_CACHE is exported repo-wide via the user's shell profile,
# but HF_HOME (the var transformers/huggingface_hub actually reads) is only
# set interactively via the `hfhome` alias, which a non-interactive qsub job
# never runs. Previously only exported inside the vLLM singularity launch
# string below, so any direct_gpu (NNsight) or no-model job silently fell
# back to ~/.cache/huggingface -- discovered when the interp-judge job hit
# $HOME's quota downloading qwen-2.5-7b. Set it for every job type here.
export HF_HOME="${HF_CACHE:?HF_CACHE is not set -- export it in your cluster profile}"

eval "$("$PY" "$REPO_ROOT/experiments/_resolve_job.py" "$ID")"

SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ]; then
        echo "Stopping vLLM server (pid $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

RUNNER_ARGS=()

if [ "$GPU_NEED" = "vllm_server" ]; then
    # Known-incompatible GPU/model combinations refuse loud rather than run
    # degraded. The runner calls experiments/utils.py's
    # check_gpu_model_compatibility too (which also now raises); this is a
    # fail-fast duplicate before spending 20+ minutes bringing up a server
    # that would produce silently wrong output. Currently: gemma-3-27b AWQ
    # on A100 (see CLAUDE.local.md).
    if [ "$MODEL_ID" = "gaunernst/gemma-3-27b-it-int4-awq" ]; then
        GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
        if [[ "$GPU_NAME" == *A100* ]]; then
            echo "error: ${MODEL_ID} is known to silently degrade on A100 (${GPU_NAME}) -- refusing to serve it here." >&2
            exit 1
        fi
    fi

    HEALTH_URL="http://localhost:${SERVE_PORT}/health"

    if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
        echo "vLLM server already answering on port ${SERVE_PORT}; reusing it."
    else
        SIF_PATH="${VLLM_SIF_DIR:?VLLM_SIF_DIR is not set}/${SIF_IMAGE}"
        HF_CACHE_DIR="${HF_CACHE:?HF_CACHE is not set}"

        # Built as an interpolated string, not a bash array run through
        # exec: extra_vllm_args entries (e.g. nuextract-2.0-8b's
        # --limit-mm-per-prompt '{"image": 2}') carry their own embedded
        # shell quoting, meant to be re-parsed by the inner `bash -c` below
        # -- treating an entry as one opaque argv token would break that
        # quoting instead of preserving it. Mirrors gen_serve_script.py's
        # (and the hand-written serve_*.sh scripts') approach exactly.
        VLLM_CMD="/usr/bin/python3 -m vllm.entrypoints.openai.api_server \
            --model ${MODEL_ID} \
            --max-model-len ${MAX_MODEL_LEN} \
            --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
            --dtype ${DTYPE} \
            --host 0.0.0.0 \
            --port ${SERVE_PORT} \
            --seed 342 \
            --trust-remote-code"
        if [ -n "$QUANTIZATION" ]; then
            VLLM_CMD="$VLLM_CMD --quantization ${QUANTIZATION}"
        fi
        for arg in "${EXTRA_VLLM_ARGS[@]}"; do
            VLLM_CMD="$VLLM_CMD $arg"
        done

        LAUNCH_CMD="export TMPDIR=${TMPDIR:-/tmp} && export HF_HOME=${HF_CACHE_DIR} && $VLLM_CMD"

        if [ "$DRY_RUN" = "1" ]; then
            echo "[dry-run] would run: singularity exec --nv --bind \$SINGULARITY_BIND $SIF_PATH bash -c \"$LAUNCH_CMD\""
        else
            echo "Starting vLLM server for ${MODEL_ID} on port ${SERVE_PORT}..."
            singularity exec --nv \
                --bind "${SINGULARITY_BIND:?SINGULARITY_BIND is not set}" \
                "$SIF_PATH" \
                bash -c "$LAUNCH_CMD" &
            SERVER_PID=$!

            echo "Waiting for ${HEALTH_URL} to come up (up to 30 minutes)..."
            READY=0
            for _ in $(seq 1 120); do
                if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                    echo "vLLM server process exited before becoming healthy." >&2
                    break
                fi
                if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
                    READY=1
                    break
                fi
                sleep 15
            done

            if [ "$READY" != "1" ]; then
                echo "vLLM server never became healthy; aborting without running the experiment." >&2
                exit 1
            fi
            echo "vLLM server is healthy."
        fi
    fi

    RUNNER_ARGS+=(--api-base "http://localhost:${SERVE_PORT}/v1")
fi

echo "[submit_job] ${ID}: experiments/${RUNNER} ${CONFIG_PATH} ${RUNNER_ARGS[*]:-}"

if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] would run: $PY experiments/${RUNNER} $CONFIG_PATH ${RUNNER_ARGS[*]:-}"
    exit 0
fi

"$PY" "experiments/${RUNNER}" "$CONFIG_PATH" "${RUNNER_ARGS[@]}"
