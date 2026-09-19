"""
NuExtract3 baseline runner.

Runs the NuExtract3 extraction baseline for any registered dataset, writing to:

    experiments/results/{dataset}/baseline_nuextract3/{experiment_id}/final.json

(id-addressed, like every other Tier-1 type.)

Unlike the old NuExtract-2.0-8B baseline (`run_baseline_nuextract.py`), this
runner reads raw OCR text directly -- no `process_pdfs.py` step, no rendered
page images, no table cleaning (`MeasurementLMNuExtract3` doesn't support
`clean_tables`; see its own module docstring for why). One API call per
document, not one per (page, attribute).

Usage
-----
    python experiments/run_baseline_nuextract3.py experiments/experiment-configs/pond/baseline_nuextract3/<id>/<id>.yaml

Required params: dataset.
Optional params: model (default "nuextract3"), paper_subset (list), ocr_dir
    (default "{data_dir}/ocr_output_raw"), api_base, api_key, max_tokens (see
    MeasurementLMNuExtract3's own docstring for its fallback order -- this,
    then the model config's sampling_params.max_tokens, then 32768).

Available datasets: any file in experiments/dataset-configs/<name>.py that exports CONFIG.
"""
from __future__ import annotations

import argparse
import json
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
from scholarlm.measurementlm_nuextract3 import MeasurementLMNuExtract3

from run_extraction import load_dataset_config, load_papers
import utils as paths
from utils import set_seeds, check_gpu_model_compatibility, write_run_metadata


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_baseline_nuextract3(
    dataset_config,
    model_config,
    output_dir: Path,
    paper_subset_override: list[str] | None = None,
    ocr_dir: str | None = None,
    api_base: str = "http://localhost:8081/v1",
    api_key: str = "EMPTY",
    max_tokens: int | None = None,
) -> None:
    """Run the NuExtract3 baseline for a dataset.

    Reads `{data_dir}/ocr_output_raw/` directly -- no pre-processing step --
    unless `ocr_dir` overrides that, e.g. to point at a
    `experiments/results/{dataset}/drop_references/<id>/` directory instead
    (same override convention as `run_baseline_langextract.py`'s `ocr_dir`).

    Writes a single `final.json` to `output_dir`, in the standard extraction
    record schema (same fields as MeasurementLM/ablation final.json output),
    so it can be loaded via `analysis.loaders.load_extraction` unmodified.

    Args:
        dataset_config: Dataset configuration loaded from `experiments/dataset-configs/`.
        model_config: Model configuration from `experiments/model-configs/baseline/`.
        output_dir: Directory for the output file (created if needed).
        paper_subset_override: If provided, overrides `dataset_config.paper_subset`.
        ocr_dir: If provided, overrides the default `{data_dir}/ocr_output_raw`.
        api_base: Base URL of the vLLM OpenAI-compatible server hosting NuExtract3.
        api_key: API key for the vLLM server (any non-empty string works).
        max_tokens: Forwarded to `MeasurementLMNuExtract3`; `None` uses its own
            sampling_params/32768 fallback.
    """
    if dataset_config.direct_extraction_schema is None or dataset_config.direct_extraction_prompt is None:
        raise ValueError(
            f"Dataset '{dataset_config.name}' does not define direct_extraction_schema "
            f"and/or direct_extraction_prompt, both required for the NuExtract3 baseline "
            f"(the same values Ablation 1 uses)."
        )

    effective_ocr_dir = ocr_dir or str(Path(dataset_config.data_dir) / "ocr_output_raw")

    print(f"\nDataset   : {dataset_config.name}")
    print(f"Model     : {model_config.name} ({model_config.model_id})")
    print(f"OCR dir   : {effective_ocr_dir}")
    print(f"Output    : {output_dir}\n")

    text, text_info = load_papers(dataset_config, effective_ocr_dir, paper_subset_override)
    print(f"Loaded {len(text)} papers.\n")

    mlm = MeasurementLMNuExtract3(
        model_name=model_config.model_id,
        entity_identification_prompt=dataset_config.entity_identification_prompt,
        entity_identification_schema=dataset_config.entity_schema,
        attribute_info_dict=dataset_config.attribute_info_dict,
        direct_extraction_schema=dataset_config.direct_extraction_schema,
        direct_extraction_prompt=dataset_config.direct_extraction_prompt,
        examples=dataset_config.nuextract_examples,
        sampling_params=model_config.sampling_params,
        api_base=api_base,
        api_key=api_key,
        max_tokens=max_tokens,
    )

    gpu_warnings = check_gpu_model_compatibility(model_config.model_id)

    print("Running NuExtract3 baseline...")
    start_time = time.time()
    data = mlm.fit(text)

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
        baseline="nuextract3",
        gpu_compatibility_warnings=gpu_warnings,
        # Surfaces issue #2's fix (ISSUE-nuextract-baseline.md): a dropped call
        # or an overflowing document is no longer indistinguishable from a
        # genuine negative -- see MeasurementLMNuExtract3's module docstring.
        validation_failures=mlm.validation_failures,
        context_length_exceeded_docs=sorted(mlm.context_length_exceeded_docs),
    )
    print(f"\nDone. Final dataset: {out_path}")
    print(f"       Records saved: {len(dataset)}")
    if mlm.validation_failures:
        print(f"       Validation failures: {mlm.validation_failures}")
    if mlm.context_length_exceeded_docs:
        print(f"       Context-length-exceeded docs: {sorted(mlm.context_length_exceeded_docs)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the NuExtract3 extraction baseline.",
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

    exp_defaults = paths.load_config().get("defaults", {})
    repo_seed = exp_defaults.get("seed")
    if repo_seed is not None and cfg["seed"] != repo_seed:
        raise ValueError(
            f"{config_path}: seed ({cfg['seed']}) does not match experiments/config.yaml "
            f"defaults.seed ({repo_seed}) -- the repo's seed is a fixed, repo-wide value, "
            "not a per-run knob."
        )
    set_seeds(cfg["seed"])

    model_name = params.get("model", "nuextract3")
    dataset_config = load_dataset_config(params["dataset"])
    model_config = paths.get_model_config("baseline", model_name)
    output_dir = paths.result_dir(params["dataset"], "baseline_nuextract3", cfg["id"])

    run_baseline_nuextract3(
        dataset_config=dataset_config,
        model_config=model_config,
        output_dir=output_dir,
        paper_subset_override=params.get("paper_subset"),
        ocr_dir=params.get("ocr_dir"),
        api_base=args.api_base or params.get("api_base") or "http://localhost:8081/v1",
        api_key=params.get("api_key", "EMPTY"),
        max_tokens=params.get("max_tokens"),
    )


if __name__ == "__main__":
    main()
