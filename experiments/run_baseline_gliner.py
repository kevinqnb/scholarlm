"""
GLiNER2 baseline runner.

Runs the GLiNER2 structured-extraction baseline (Fastino AI, EMNLP 2025;
``fastino/gliner2-{base,large}-v1``) for any registered dataset, writing to:

    experiments/results/{dataset}/baseline_gliner/{experiment_id}/final.json

(id-addressed, like every other Tier-1 type -- not the old
data/experiments/{dataset}/extraction/{gliner-model}/{date}/ convention this
runner previously wrote to specifically so analysis code could load it
unmodified; that tradeoff now applies uniformly across every runner, not
just this one, see the run_ablation.py/run_table_cleaning.py commits.)

Like the ChatExtract baseline (and unlike NuExtract), GLiNER2 is a *text-based*
method: it reads the OCR'd `<page>/<table>` tagged text (same input as
MeasurementLM / run_extraction), so it does NOT require `experiments/process_pdfs.py`.
Unlike every other runner, GLiNER2 is a small local encoder model loaded directly
via `GLiNER2.from_pretrained(...)` — there is **no vLLM / OpenAI-compatible
server**, so this runner takes no api_base/api_key params. It runs one structured
schema per dataset attribute (see `measurementlm_gliner.py`), tuned by
`threshold` (precision/recall) rather than a generation temperature.

Usage
-----
    python experiments/run_baseline_gliner.py experiments/experiment-configs/pond/baseline_gliner/<id>/<id>.yaml

Required params: dataset.
Optional params: model (default: gliner-large-v1; also: gliner-base-v1),
paper_subset (list), threshold (default 0.5), batch_size (default 8), device.

Requires the optional `gliner2[local]` dependency (installed via the `gpu` extra:
`uv sync --extra gpu`, or `pip install "gliner2[local]"`).

Available datasets: any file in experiments/dataset-configs/<name>.py that exports CONFIG.
Available models: any file in experiments/model-configs/baseline/<name>.yaml
(gliner-large-v1, gliner-base-v1).
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
from scholarlm.measurementlm_gliner import MeasurementLMGliner

from run_extraction import load_dataset_config, load_papers
import utils as paths
from utils import set_seeds, check_gpu_model_compatibility, write_run_metadata


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_baseline_gliner(
    dataset_config,
    model_config,
    output_dir: Path,
    paper_subset_override: list[str] | None = None,
    threshold: float = 0.5,
    batch_size: int = 8,
    device: str | None = None,
) -> None:
    """Run the GLiNER2 baseline for a dataset with a given GLiNER model.

    Writes a single `final.json` to `output_dir`, in the standard extraction
    record schema (same fields as MeasurementLM/ablation final.json output), so
    it can be loaded via `analysis.loaders.load_extraction` unmodified.
    """
    data_dir = Path(dataset_config.data_dir)

    print(f"\nDataset   : {dataset_config.name}")
    print(f"Model     : {model_config.name} ({model_config.model_id})")
    print(f"Output    : {output_dir}\n")

    # GLiNER reads OCR text directly (same input as MeasurementLM / ChatExtract).
    ocr_dir = str(data_dir / "ocr_output_raw")
    text, text_info = load_papers(dataset_config, ocr_dir, paper_subset_override)
    print(f"Loaded {len(text_info)} papers.\n")

    mlm = MeasurementLMGliner(
        model_name=model_config.model_id,
        entity_identification_prompt=dataset_config.entity_identification_prompt,
        entity_identification_schema=dataset_config.entity_schema,
        attribute_info_dict=dataset_config.attribute_info_dict,
        gliner_property_names=(
            dataset_config.gliner_property_names
            or dataset_config.chatextract_property_names
        ),
        entity_type_description=dataset_config.entity_type_description,
        gliner_entity_description=dataset_config.gliner_entity_description,
        gliner_field_descriptions=dataset_config.gliner_field_descriptions,
        measurement_event_schema=dataset_config.measurement_event_schema,
        sampling_params=model_config.sampling_params,
        threshold=threshold,
        batch_size=batch_size,
        device=device,
    )

    gpu_warnings = check_gpu_model_compatibility(model_config.model_id)

    print("Running GLiNER baseline...")
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
        baseline="gliner",
        threshold=threshold,
        batch_size=batch_size,
        gpu_compatibility_warnings=gpu_warnings,
    )
    print(f"\nDone. Final dataset: {out_path}")
    print(f"       Records saved: {len(dataset)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the GLiNER2 structured-extraction baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("config", help="Path to an experiment-configs/.../<id>.yaml.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config_path = Path(args.config)
    cfg = paths.load_experiment_config(config_path)
    params = cfg["params"]
    paths.require_params(params, "dataset", config_path=config_path)

    set_seeds(cfg["seed"])

    model_name = params.get("model", "gliner-large-v1")
    dataset_config = load_dataset_config(params["dataset"])
    model_config = paths.get_model_config("baseline", model_name)
    output_dir = paths.result_dir(params["dataset"], "baseline_gliner", cfg["id"])

    run_baseline_gliner(
        dataset_config=dataset_config,
        model_config=model_config,
        output_dir=output_dir,
        paper_subset_override=params.get("paper_subset"),
        threshold=params.get("threshold", 0.5),
        batch_size=params.get("batch_size", 8),
        device=params.get("device"),
    )


if __name__ == "__main__":
    main()
