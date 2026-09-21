"""
ChatExtract baseline runner.

Runs the ChatExtract extraction baseline (Polak & Morgan, Nat. Commun. 2024;
arXiv:2303.05352) for any registered dataset, on any registered backbone model,
writing to:

    experiments/results/{dataset}/baseline_chatextract/{experiment_id}/final.json

(id-addressed, like every other Tier-1 type.)

Unlike the NuExtract baseline, ChatExtract is a *text-based* method: it reads the
OCR'd `<page>/<table>` tagged text (same input as MeasurementLM / run_extraction),
so it does NOT require `experiments/process_pdfs.py`. It is a multi-turn
conversational method — classify each sentence, gate single vs. multiple values,
extract Material/Value/Unit, then verify each field with redundant strict yes/no
questions — run once per dataset attribute (see `measurementlm_chatextract.py`).
Because it wraps an ordinary chat model, `model` selects any entry under
experiments/model-configs/extraction/, so ChatExtract can be run on the same
backbones as MeasurementLM for a fair same-model comparison.

Usage
-----
    python experiments/run_baseline_chatextract.py experiments/experiment-configs/pond/baseline_chatextract/<id>/<id>.yaml

Required params: dataset.
Optional params: model (default: gemma-3-27b), paper_subset (list), ocr_dir,
api_base, api_key, max_concurrent (default 32), extract_tables (default true),
single_verification (default false).

Available datasets: any file in experiments/dataset-configs/<name>.py that exports CONFIG.
Available models: any file in experiments/model-configs/extraction/<name>.yaml.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — make scholarlm and run_extraction importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_EXPERIMENTS_DIR))

from scholarlm.measurementlm import NumpyEncoder
from scholarlm.measurementlm_chatextract import MeasurementLMChatExtract

from run_extraction import load_dataset_config, load_papers, get_model_config
import utils as paths
from utils import set_seeds, check_gpu_model_compatibility, write_run_metadata


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_baseline_chatextract(
    dataset_config,
    model_config,
    output_dir: Path,
    paper_subset_override: list[str] | None = None,
    ocr_dir: str | None = None,
    api_base: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    max_concurrent: int = 32,
    extract_tables: bool = True,
    include_single_verification: bool = False,
) -> None:
    """Run the ChatExtract baseline for a dataset on a given backbone model.

    Writes a single `final.json` to `output_dir`, in the standard extraction
    record schema (same fields as MeasurementLM/ablation final.json output),
    so it can be loaded via `analysis.loaders.load_extraction` unmodified.
    """
    data_dir = Path(dataset_config.data_dir)

    is_frontier = model_config.api_base is not None
    if is_frontier:
        effective_api_base = model_config.api_base
        if api_key == "EMPTY":
            if "openai.com" in model_config.api_base:
                api_key = os.environ.get("OPENAI_API_KEY", "")
            else:
                api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError(
                f"API key required for frontier model '{model_config.name}'. "
                "Set OPENAI_API_KEY or GEMINI_API_KEY, or pass --api-key."
            )
    else:
        effective_api_base = api_base

    # ChatExtract reads OCR text directly. Mirroring run_extraction, an explicit
    # --ocr-dir (e.g. a pre-cleaned ocr_output_cleaned_{model} directory, whose
    # tables carry <caption> elements) takes precedence; otherwise fall back to
    # raw OCR. Unlike MeasurementLM, ChatExtract never runs its own table-cleaning
    # pass, so pass --ocr-dir to feed it the same cleaned tables MeasurementLM sees.
    effective_ocr_dir = ocr_dir or str(data_dir / "ocr_output_raw")

    print(f"\nDataset   : {dataset_config.name}")
    print(f"Model     : {model_config.name} ({model_config.model_id})")
    print(f"OCR dir   : {effective_ocr_dir}")
    print(f"Output    : {output_dir}\n")

    text, text_info = load_papers(dataset_config, effective_ocr_dir, paper_subset_override)
    print(f"Loaded {len(text_info)} papers.\n")

    # Paper titles (from directory.json metadata) form the head of each passage.
    titles = [info.get("title", "") or "" for info in text_info]

    mlm = MeasurementLMChatExtract(
        model_name=model_config.model_id,
        entity_identification_prompt=dataset_config.entity_identification_prompt,
        entity_identification_schema=dataset_config.entity_schema,
        attribute_info_dict=dataset_config.attribute_info_dict,
        attribute_property_names=dataset_config.chatextract_property_names,
        entity_noun=dataset_config.chatextract_entity_noun,
        include_single_verification=include_single_verification,
        extract_tables=extract_tables,
        measurement_event_schema=dataset_config.measurement_event_schema,
        sampling_params=model_config.sampling_params,
        api_base=effective_api_base,
        api_key=api_key,
        max_concurrent=max_concurrent,
        use_extra_body=not is_frontier,
    )

    gpu_warnings = check_gpu_model_compatibility(model_config.model_id)

    print("Running ChatExtract baseline...")
    start_time = time.time()
    data = mlm.fit(text, titles)

    dataset = [
        info | dp | {"document_id": info["document_id"], "measurement_id": i}
        for i, dp in enumerate(data)
        for info in [text_info[dp["document_id"]]]
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "final.json"
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

    write_run_metadata(
        output_dir,
        start_time=start_time,
        dataset=dataset_config.name,
        model=model_config.name,
        model_id=model_config.model_id,
        hf_revision=model_config.hf_revision,
        baseline="chatextract",
        extract_tables=extract_tables,
        include_single_verification=include_single_verification,
        gpu_compatibility_warnings=gpu_warnings,
        max_prompt_tokens=mlm.max_prompt_tokens,
        token_usage=mlm.token_usage,
        n_failed_work_items=len(mlm.failures),
        failed_work_items=mlm.failures,
    )
    print(f"\nDone. Final dataset: {out_path}")
    print(f"       Records saved: {len(dataset)}")
    print(f"       Failed work items: {len(mlm.failures)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the ChatExtract extraction baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("config", help="Path to an experiment-configs/.../<id>.yaml.")
    p.add_argument(
        "--api-base", default=None, metavar="URL",
        help="Override params.api_base (submit.sh injects the compute node's vLLM endpoint here).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config_path = Path(args.config)
    cfg = paths.load_experiment_config(config_path)
    params = cfg["params"]
    paths.require_params(params, "dataset", config_path=config_path)

    set_seeds(cfg["seed"])

    model_name = params.get("model", "gemma-3-27b")
    dataset_config = load_dataset_config(params["dataset"])
    model_config = get_model_config(model_name)
    output_dir = paths.result_dir(params["dataset"], "baseline_chatextract", cfg["id"])

    run_baseline_chatextract(
        dataset_config=dataset_config,
        model_config=model_config,
        output_dir=output_dir,
        paper_subset_override=params.get("paper_subset"),
        ocr_dir=params.get("ocr_dir"),
        api_base=args.api_base or params.get("api_base") or "http://localhost:8081/v1",
        api_key=params.get("api_key", "EMPTY"),
        max_concurrent=params.get("max_concurrent", 32),
        extract_tables=params.get("extract_tables", True),
        include_single_verification=params.get("single_verification", False),
    )


if __name__ == "__main__":
    main()
