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
paper_subset (list), ocr_dir (default "{data_dir}/ocr_output_raw" -- until
2026-09-25 this param was silently ignored; see run_baseline_gliner()'s
docstring), threshold (default 0.5), batch_size (default 8),
include_qualifiers (bool, default true -- false drops the 7 qualifier/shape
fields from every per-attribute GLiNER structure entirely, mirroring
run_ablation.py's own flag. Unlike Ablation 1/NuExtract3/LangExtract, GLiNER's
quantity fields are hardcoded in measurementlm_gliner.py rather than driven by
a dataset-config schema, so this works for every dataset with no config
changes required).

`device` is NOT a params key -- GLiNER2 loads weights directly in-process (no
vLLM server), so the device to load them onto is a fixed per-model value, not
something that varies between experiments; it lives in the model config
(`experiments/model-configs/baseline/<model>.yaml`'s `device:` key, e.g.
`cuda`) alongside that file's `resources:` block. Missing it is a hard error,
not a CPU fallback -- see measurementlm_gliner.py's `__init__`.

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
    ocr_dir: str | None = None,
    threshold: float = 0.5,
    batch_size: int = 8,
    include_qualifiers: bool = True,
    *,
    device: str,
) -> None:
    """Run the GLiNER2 baseline for a dataset with a given GLiNER model.

    Writes a single `final.json` to `output_dir`, in the standard extraction
    record schema (same fields as MeasurementLM/ablation final.json output), so
    it can be loaded via `analysis.loaders.load_extraction` unmodified.

    ocr_dir: If provided, overrides the default `{data_dir}/ocr_output_raw`
        (same override convention as run_baseline_nuextract3.py/
        run_baseline_langextract.py's `ocr_dir`, e.g. a table-cleaning-based
        `experiments/results/{dataset}/drop_references/<id>/` directory).
        Until 2026-09-25 this runner accepted `params.ocr_dir` in its
        experiment configs but silently never read it -- every full run
        before that date read raw OCR regardless of what its config claimed;
        see notes/scholarlm/builds for the fix.
    include_qualifiers: False drops the 7 qualifier/shape fields
        (qualifiers/point_value/lower/upper/list_values/tolerance/
        standard_deviation) from every per-attribute GLiNER structure --
        mirrors run_ablation.py's own flag, but needs no dataset-config
        variant (see MeasurementLMGliner's own docstring).
    """
    if not isinstance(include_qualifiers, bool):
        raise ValueError(
            f"include_qualifiers must be a bool, got {include_qualifiers!r} "
            f"({type(include_qualifiers).__name__}) -- a YAML string like "
            f"'false' is truthy in Python and would silently run with "
            f"qualifiers included while claiming otherwise."
        )
    data_dir = Path(dataset_config.data_dir)

    # GLiNER reads OCR text directly (same input as MeasurementLM / ChatExtract).
    effective_ocr_dir = ocr_dir or str(data_dir / "ocr_output_raw")

    print(f"\nDataset   : {dataset_config.name}")
    print(f"Model     : {model_config.name} ({model_config.model_id})")
    print(f"OCR dir   : {effective_ocr_dir}")
    print(f"Output    : {output_dir}\n")

    text, text_info = load_papers(dataset_config, effective_ocr_dir, paper_subset_override)
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
        include_qualifiers=include_qualifiers,
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
        ocr_dir=effective_ocr_dir,
        threshold=threshold,
        batch_size=batch_size,
        include_qualifiers=include_qualifiers,
        # GLiNER2.from_pretrained is a local model, no OpenAI-compatible calls
        # (see module docstring) -- token_usage would be all-zero, indistinguishable
        # from "forgot to record", so it's marked n/a rather than omitted.
        token_accounting="n/a: local GLiNER2, no API calls",
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

    if model_config.device is None:
        raise ValueError(
            f"Model config 'baseline/{model_name}.yaml' has no 'device' key. "
            f"MeasurementLMGliner loads weights directly in-process (no vLLM "
            f"server) -- without an explicit device it silently stays on "
            f"whatever nn.Module defaults to (CPU), even inside a GPU job. "
            f"Add e.g. `device: cuda` to that model config."
        )

    run_baseline_gliner(
        dataset_config=dataset_config,
        model_config=model_config,
        output_dir=output_dir,
        paper_subset_override=params.get("paper_subset"),
        ocr_dir=params.get("ocr_dir"),
        threshold=params.get("threshold", 0.5),
        batch_size=params.get("batch_size", 8),
        include_qualifiers=params.get("include_qualifiers", True),
        device=model_config.device,
    )


if __name__ == "__main__":
    main()
