"""
Generate SGE vLLM serve scripts from experiments/config.yaml.

For every model in config.yaml that has a ``serve`` section, writes
``experiments/serve_{model_name}.sh`` and makes it executable.

Regenerate whenever you change a model's serve section:

    python experiments/gen_serve_script.py           # all models
    python experiments/gen_serve_script.py gemma-3-27b olmocr  # specific models
    python experiments/gen_serve_script.py --dry-run  # preview without writing

The cluster section of config.yaml controls SGE directives (project, mail).
All runtime paths still come from environment variables (VLLM_SIF_DIR, etc.).
"""
from __future__ import annotations

import argparse
import stat
import sys
from pathlib import Path

import yaml

_EXPERIMENTS_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _script_name(model_key: str) -> str:
    """'gemma-3-27b' → 'gemma_3_27b' (safe for filenames and SGE job names)."""
    return model_key.replace("-", "_").replace(".", "_")


# ---------------------------------------------------------------------------
# Script generator
# ---------------------------------------------------------------------------

def _generate(model_key: str, model_cfg: dict, defaults: dict, cluster: dict) -> str:
    """Return the complete content of a serve script for one model."""
    serve = model_cfg["serve"]
    model_id: str = model_cfg["model_id"]
    name = _script_name(model_key)

    sge_project: str = cluster.get("sge_project", "")
    sge_mail: str = cluster.get("sge_mail_events", "e")

    walltime: str = serve["walltime"]
    omp: int = serve["omp"]
    gpu_memory: str = serve["gpu_memory"]
    gpu_cap: float = serve["gpu_capability"]
    port: int = serve["port"]
    max_model_len: int = serve["max_model_len"]
    gpu_mem_util: float = serve["gpu_memory_utilization"]
    quantization: str | None = serve.get("quantization")
    dtype: str | None = serve.get("dtype")
    sif_image: str = serve.get("sif_image", "vllm-openai_latest.sif")
    seed: int = defaults.get("seed", 342)

    # -----------------------------------------------------------------------
    # SGE project directive (omit the line entirely if not configured)
    # -----------------------------------------------------------------------
    sge_project_line = f"#$ -P {sge_project}\n" if sge_project else ""

    # -----------------------------------------------------------------------
    # GPU check block:
    #   AWQ marlin models → abort on A100 (silent degradation, not an error)
    #   All other models   → just capture GPU name for logging
    # -----------------------------------------------------------------------
    if quantization == "awq_marlin":
        gpu_check_block = (
            "# ---------------------------------------------------------------------------\n"
            "# A100 compatibility check\n"
            "#\n"
            f"# {model_id} uses AWQ marlin quantization, which produces silently\n"
            "# degraded output on A100 GPUs due to a kernel incompatibility.\n"
            "# Abort early rather than producing unreliable results.\n"
            "# ---------------------------------------------------------------------------\n"
            'GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)\n'
            'if echo "$GPU_NAME" | grep -qi "A100"; then\n'
            '    echo "ERROR: ${MODEL} is incompatible with A100 GPUs (AWQ marlin kernel issue)."\n'
            '    echo "GPU detected: ${GPU_NAME}"\n'
            '    echo "Resubmit on a node with H100 or newer, or switch to an FP16/FP8 variant."\n'
            '    exit 1\n'
            'fi\n'
        )
    else:
        gpu_check_block = (
            "GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)\n"
        )

    # -----------------------------------------------------------------------
    # vLLM server flags
    # -----------------------------------------------------------------------
    vllm_flags: list[str] = [
        f"        --model ${{MODEL}}",
        f"        --max-model-len ${{MAX_MODEL_LEN}}",
        f"        --gpu-memory-utilization ${{GPU_MEM_UTIL}}",
    ]
    if quantization:
        vllm_flags.append(f"        --quantization {quantization}")
    if dtype and dtype != "auto":
        vllm_flags.append(f"        --dtype {dtype}")
    vllm_flags += [
        "        --host 0.0.0.0",
        "        --port ${PORT}",
        f"        --seed {seed}",
        "        --trust-remote-code",
    ]
    vllm_cmd = " \\\n".join(vllm_flags)

    # -----------------------------------------------------------------------
    # Assemble script
    #
    # Bash ${VAR} references inside Python f-strings require {{ / }} escaping.
    # To keep the template readable, only the dynamic parts use f-string
    # interpolation; bash variables that are purely runtime (${JOB_ID},
    # ${GPU_NAME}, etc.) are written with doubled braces.
    # -----------------------------------------------------------------------
    script = f"""\
#!/bin/bash -l
{sge_project_line}\
#$ -l h_rt={walltime}
#$ -pe omp {omp}
#$ -l gpus=1
#$ -l gpu_memory={gpu_memory}
#$ -l gpu_c={gpu_cap}
#$ -o out/serve_{name}_out.txt
#$ -e out/serve_{name}_error.txt
#$ -m {sge_mail}
# Generated by gen_serve_script.py from config.yaml — do not edit manually.
# Re-run:  python experiments/gen_serve_script.py {model_key}

# ---------------------------------------------------------------------------
# All absolute paths come from environment variables.  Set these in your
# cluster profile (~/.bashrc / ~/.bash_profile) or a local .env file:
#
#   VLLM_SIF_DIR        directory containing Singularity .sif images
#   HF_CACHE            HuggingFace weights cache directory
#   SINGULARITY_BIND    bind-mount argument (e.g. /projectnb/foo:/projectnb/foo)
#   SCHOLARLM_ROOT      absolute path to this repository root
#   TMPDIR              temporary directory for Singularity jobs
#
# Model and serve parameters live in experiments/config.yaml.
# ---------------------------------------------------------------------------

SIF_IMAGE="${{VLLM_SIF_DIR:?VLLM_SIF_DIR is not set}}/{sif_image}"
HF_CACHE="${{HF_CACHE:?HF_CACHE is not set}}"
SCHOLARLM_ROOT="${{SCHOLARLM_ROOT:?SCHOLARLM_ROOT is not set}}"
TMPDIR="${{TMPDIR:-/tmp}}"

MODEL="{model_id}"
PORT={port}
MAX_MODEL_LEN={max_model_len}
GPU_MEM_UTIL={gpu_mem_util}
ENDPOINT_FILE="${{SCHOLARLM_ROOT}}/.vllm_endpoint_${{MODEL//\\//_}}.txt"

# ---------------------------------------------------------------------------
# Cleanup: remove the endpoint file when the server exits
# ---------------------------------------------------------------------------
cleanup() {{
    echo "Cleaning up endpoint file..."
    rm -f "$ENDPOINT_FILE"
}}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Write endpoint file so client jobs know which host:port to call
# ---------------------------------------------------------------------------
HOSTNAME_FULL="$(hostname)"
echo "${{HOSTNAME_FULL}}:${{PORT}}" > "$ENDPOINT_FILE"
echo "Wrote endpoint: ${{HOSTNAME_FULL}}:${{PORT}} → ${{ENDPOINT_FILE}}"

{gpu_check_block}
echo "============================================"
echo "  Job ID:      ${{JOB_ID}}"
echo "  Node:        $(hostname)"
echo "  GPU:         ${{GPU_NAME}}"
echo "  Model:       ${{MODEL}}"
echo "  Context:     ${{MAX_MODEL_LEN}} tokens"
echo "  Port:        ${{PORT}}"
echo "  SIF image:   ${{SIF_IMAGE}}"
echo "  HF cache:    ${{HF_CACHE}}"
echo "============================================"

singularity exec \\
    --nv \\
    --bind "${{SINGULARITY_BIND:?SINGULARITY_BIND is not set}}" \\
    "$SIF_IMAGE" \\
    bash -c "export TMPDIR=${{TMPDIR}} && \\
    export HF_HOME=${{HF_CACHE}} && \\
    /usr/bin/python3 -m vllm.entrypoints.openai.api_server \\
{vllm_cmd}"

echo "vLLM server has stopped."
"""
    return script


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate SGE serve scripts from config.yaml.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "models",
        nargs="*",
        metavar="MODEL_KEY",
        help="Model keys to generate (default: all models with a serve section).",
    )
    p.add_argument(
        "--config",
        default=str(_EXPERIMENTS_DIR / "config.yaml"),
        metavar="FILE",
        help="Path to config.yaml (default: experiments/config.yaml).",
    )
    p.add_argument(
        "--out-dir",
        default=str(_EXPERIMENTS_DIR),
        metavar="DIR",
        help="Directory to write serve scripts (default: experiments/).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated scripts to stdout instead of writing files.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    defaults: dict = cfg.get("defaults", {})
    cluster: dict = cfg.get("cluster", {})
    models: dict = cfg.get("models", {})

    targets = args.models if args.models else [k for k, v in models.items() if "serve" in v]

    if not targets:
        print("No models with a serve section found in config.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    generated: list[str] = []
    skipped: list[str] = []

    for key in targets:
        if key not in models:
            print(f"  [skip] '{key}' not found in config models.", file=sys.stderr)
            skipped.append(key)
            continue
        model_cfg = models[key]
        if "serve" not in model_cfg:
            print(f"  [skip] '{key}' has no serve section (frontier model?).", file=sys.stderr)
            skipped.append(key)
            continue

        content = _generate(key, model_cfg, defaults, cluster)
        script_name = f"serve_{_script_name(key)}.sh"

        if args.dry_run:
            print(f"# ===== {script_name} =====")
            print(content)
        else:
            out_path = out_dir / script_name
            out_path.write_text(content)
            out_path.chmod(out_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            print(f"  wrote {out_path}")
            generated.append(script_name)

    if not args.dry_run:
        print(f"\nGenerated {len(generated)} script(s).")
        if skipped:
            print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
