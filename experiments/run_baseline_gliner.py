"""
GLiNER2 baseline runner.

Runs the GLiNER2 structured-extraction baseline (Fastino AI, EMNLP 2025;
``fastino/gliner2-{base,large}-v1``) for any registered dataset, writing results
to the *standard extraction path* so it can be loaded and compared against
MeasurementLM using the existing analysis code with no modification:

    data/experiments/{dataset}/extraction/{gliner-model}/{YYYY_mm_dd}/final.json

Like the ChatExtract baseline (and unlike NuExtract), GLiNER2 is a *text-based*
method: it reads the OCR'd `<page>/<table>` tagged text (same input as
MeasurementLM / run_extraction), so it does NOT require `experiments/process_pdfs.py`.
Unlike every other runner, GLiNER2 is a small local encoder model loaded directly
via `GLiNER2.from_pretrained(...)` — there is **no vLLM / OpenAI-compatible
server**, so this runner takes no `--api-base`/`--api-key`. It runs one structured
schema per dataset attribute (see `measurementlm_gliner.py`), tuned by
`--threshold` (precision/recall) rather than a generation temperature.

Usage
-----
    # From the repo root (first run downloads the model from HuggingFace):
    python experiments/run_baseline_gliner.py --dataset pond
    python experiments/run_baseline_gliner.py --dataset pond --model gliner-base-v1
    python experiments/run_baseline_gliner.py --dataset nfix \\
        --paper-subset physical_and_chemical_limnological --threshold 0.4

Requires the optional `gliner2[local]` dependency (installed via the `gpu` extra:
`uv sync --extra gpu`, or `pip install "gliner2[local]"`).

Available datasets: any file in experiments/configs/<name>.py that exports CONFIG.
Available models: gliner-large-v1 (default), gliner-base-v1.
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
from model_registry import BASELINE_MODEL_REGISTRY
import paths
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
    p.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (must match a file in experiments/configs/<name>.py).",
    )
    p.add_argument(
        "--model",
        default="gliner-large-v1",
        choices=["gliner-large-v1", "gliner-base-v1"],
        help="GLiNER model (must match a BASELINE_MODEL_REGISTRY entry). Default: gliner-large-v1.",
    )
    p.add_argument(
        "--date",
        default=None,
        help="Output date tag YYYY_mm_dd (default: today).",
    )
    p.add_argument(
        "--paper-subset",
        nargs="+",
        default=None,
        metavar="PAPER_CODE",
        help="Override dataset paper_subset with an explicit list of paper codes.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="GLiNER confidence threshold in [0, 1] (default: 0.5). Lower = higher recall.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="GLiNER chunk batch size (default: 8).",
    )
    p.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help="Torch device / map_location for GLiNER (e.g. 'cuda', 'cpu'). Default: auto.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    from utils import load_config
    cfg = load_config()
    seed = cfg.get("defaults", {}).get("seed", 342)
    set_seeds(seed)

    dataset_config = load_dataset_config(args.dataset)
    model_config = BASELINE_MODEL_REGISTRY[args.model]
    output_dir = paths.extraction(args.dataset, args.model, args.date)

    run_baseline_gliner(
        dataset_config=dataset_config,
        model_config=model_config,
        output_dir=output_dir,
        paper_subset_override=args.paper_subset,
        threshold=args.threshold,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
