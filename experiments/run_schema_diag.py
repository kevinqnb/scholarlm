"""experiments/run_schema_diag.py <config> --api-base <url>

Manual/composite diagnostic runner -- NOT one of the contract's 14 automated
run_{type}.py entry points, not wired into experiments/submit.sh's
EXPERIMENT_TYPES dispatch (see notes/hub/conventions.md's "composite/manual
experiment-types" carve-out; this repo's existing drop_references and
probe_calibration types follow the same pattern). Invoked directly by a
hand-rolled qsub job script that brings up its own vLLM server, since this
isn't a permanent pipeline capability.

Root-causes the gpt-oss-120b _parse_quantities() failure mode found in
2026-09-21-pond-extraction-gptoss120b-full-01 (see that experiment-config's
own description, and this script's own config's description for the full
writeup): replays known-failing (value -> parse_quantities call) pairs
pulled directly from that run's final.json against 3 ParseQuantityResponse
schema variants that differ only in point_value's type, holding the
instructions and every other field/sampling-param identical to the
production _parse_quantities() call (src/scholarlm/measurementlm.py).

Writes results.json (one row per (value, variant, repeat) trial) and a
run_metadata.json to experiments/results/{dataset}/schema_diag/<id>/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from scholarlm.instruction_prompts import PARSE_QUANTITY_VALUE_ONLY_INSTRUCTIONS
from scholarlm.measurementlm import response_validator

import utils as paths
from utils import set_seeds, write_run_metadata


# ---------------------------------------------------------------------------
# Schema variants -- identical to ParseQuantityResponse
# (src/scholarlm/measurementlm.py) except for point_value's type. Every
# other field is untouched so the 3 variants differ in exactly one thing.
# ---------------------------------------------------------------------------

class VariantA(BaseModel):
    """(a) Current production schema: point_value: str | None = None."""
    explanation: str
    qualifiers: list[str]
    point_value: str | None = None
    lower: str | None = None
    upper: str | None = None
    list_values: list[str] | None = None
    tolerance: str | None = None
    standard_deviation: str | None = None


class VariantB(BaseModel):
    """(b) point_value required, non-nullable str (no Optional/anyOf)."""
    explanation: str
    qualifiers: list[str]
    point_value: str
    lower: str | None = None
    upper: str | None = None
    list_values: list[str] | None = None
    tolerance: str | None = None
    standard_deviation: str | None = None


class VariantC(BaseModel):
    """(c) point_value: str | float | None -- accepts a bare number."""
    explanation: str
    qualifiers: list[str]
    point_value: str | float | None = None
    lower: str | None = None
    upper: str | None = None
    list_values: list[str] | None = None
    tolerance: str | None = None
    standard_deviation: str | None = None


VARIANTS = {"a_current": VariantA, "b_required_str": VariantB, "c_str_or_float": VariantC}


def parses_as_float(v) -> bool:
    try:
        float(str(v))
        return True
    except (ValueError, TypeError):
        return False


def is_missing_all_qualifier_fields(d: dict) -> bool:
    return not any(
        k in d
        for k in ("qualifiers", "point_value", "lower", "upper",
                   "list_values", "tolerance", "standard_deviation")
    )


def is_corrupt_point_value(d: dict) -> bool:
    pv = d.get("point_value")
    return isinstance(pv, str) and len(pv) > 20


def load_failing_values(source_final_json: Path, n_numeric: int, n_nonnumeric: int) -> dict[str, list]:
    """Pull known-failing records from a prior run's final.json, split by
    whether `value` parses as a float. Deterministic (first N in file
    order) -- this is a diagnostic sample, not a metric, so a seeded shuffle
    would add noise without adding information."""
    with open(source_final_json) as f:
        data = json.load(f)
    failing = [d for d in data if is_missing_all_qualifier_fields(d) or is_corrupt_point_value(d)]
    numeric = [d["value"] for d in failing if parses_as_float(d.get("value"))][:n_numeric]
    nonnumeric = [d["value"] for d in failing if not parses_as_float(d.get("value"))][:n_nonnumeric]
    if len(numeric) < n_numeric or len(nonnumeric) < n_nonnumeric:
        raise ValueError(
            f"{source_final_json}: only found {len(numeric)} numeric-failing and "
            f"{len(nonnumeric)} non-numeric-failing records (wanted {n_numeric}/{n_nonnumeric})."
        )
    return {"numeric": numeric, "nonnumeric": nonnumeric}


def build_prompt(value) -> str:
    """Exact value_only-mode prompt construction from
    MeasurementLM._parse_quantities() (src/scholarlm/measurementlm.py)."""
    query = (
        f"Extracted value: {value}\n"
        f"Parse this extracted value into its structured components. "
    )
    return f"## INSTRUCTIONS:\n{PARSE_QUANTITY_VALUE_ONLY_INSTRUCTIONS}\n\n## QUERY:\n{query}"


def call_once(client: OpenAI, model_id: str, prompt: str, schema_cls: type[BaseModel],
               max_tokens: int, sampling_params: dict) -> dict:
    """One synchronous call, mirroring MeasurementLM._acall's kwargs
    construction for a vllm_server (non-frontier) model exactly: same
    temperature/top_p/top_k/seed/reasoning_effort, same response_format
    shape, same response_validator."""
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "parse_quantity_response", "schema": schema_cls.model_json_schema()},
    }
    extra_body = {}
    if "top_k" in sampling_params:
        extra_body["top_k"] = sampling_params["top_k"]
    if "repetition_penalty" in sampling_params:
        extra_body["repetition_penalty"] = sampling_params["repetition_penalty"]
    if "seed" in sampling_params:
        extra_body["seed"] = sampling_params["seed"]
    chat_template_kwargs = {}
    if "enable_thinking" in sampling_params:
        chat_template_kwargs["enable_thinking"] = sampling_params["enable_thinking"]
    if "reasoning_effort" in sampling_params:
        chat_template_kwargs["reasoning_effort"] = sampling_params["reasoning_effort"]
    if chat_template_kwargs:
        extra_body["chat_template_kwargs"] = chat_template_kwargs

    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=sampling_params.get("temperature"),
            top_p=sampling_params.get("top_p"),
            response_format=response_format,
            extra_body=extra_body,
            timeout=120,
        )
        raw = response.choices[0].message.content
    except Exception as e:
        return {"ok": False, "error": f"API call failed: {e}", "raw": None, "elapsed": round(time.time() - t0, 2)}

    try:
        response_validator(schema_cls, raw)
        return {"ok": True, "error": None, "raw": raw, "elapsed": round(time.time() - t0, 2)}
    except Exception as e:
        return {"ok": False, "error": str(e), "raw": raw, "elapsed": round(time.time() - t0, 2)}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--api-base", required=True, metavar="URL")
    p.add_argument("--api-key", default="EMPTY")
    args = p.parse_args(argv)

    config_path = Path(args.config)
    cfg = paths.load_experiment_config(config_path)
    params = cfg["params"]
    paths.require_params(params, "dataset", "model", "source_experiment_id",
                          "n_numeric_values", "n_nonnumeric_values", "n_repeats",
                          "max_tokens", config_path=config_path)

    exp_defaults = paths.load_config().get("defaults", {})
    repo_seed = exp_defaults.get("seed")
    if repo_seed is not None and cfg["seed"] != repo_seed:
        raise ValueError(
            f"{config_path}: seed ({cfg['seed']}) does not match experiments/config.yaml "
            f"defaults.seed ({repo_seed})."
        )
    set_seeds(cfg["seed"])

    model_config = paths.load_model_config("extraction", params["model"])
    sampling_params = model_config["sampling_params"]
    model_id = model_config["model_id"]

    source_final = paths.result_dir(params["dataset"], "extraction", params["source_experiment_id"]) / "final.json"
    values = load_failing_values(source_final, params["n_numeric_values"], params["n_nonnumeric_values"])

    print(f"Loaded {len(values['numeric'])} numeric-failing and {len(values['nonnumeric'])} "
          f"non-numeric-failing values from {source_final}")

    client = OpenAI(base_url=args.api_base, api_key=args.api_key)

    output_dir = paths.result_dir(params["dataset"], "schema_diag", cfg["id"])
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    start_time = time.time()
    total = len(values["numeric"] + values["nonnumeric"]) * len(VARIANTS) * params["n_repeats"]
    done = 0
    for value_type, value_list in values.items():
        for value in value_list:
            prompt = build_prompt(value)
            for variant_name, schema_cls in VARIANTS.items():
                for repeat in range(params["n_repeats"]):
                    result = call_once(client, model_id, prompt, schema_cls,
                                        params["max_tokens"], sampling_params)
                    results.append({
                        "value_type": value_type,
                        "value": value,
                        "variant": variant_name,
                        "repeat": repeat,
                        **result,
                    })
                    done += 1
                    if done % 20 == 0:
                        print(f"  {done}/{total} calls done...")

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary table: success rate per (value_type, variant).
    print("\n=== Summary: success rate per (value_type, variant) ===")
    summary = {}
    for r in results:
        key = (r["value_type"], r["variant"])
        summary.setdefault(key, [0, 0])
        summary[key][1] += 1
        if r["ok"]:
            summary[key][0] += 1
    for (value_type, variant), (ok, n) in sorted(summary.items()):
        print(f"  {value_type:12s} {variant:16s} {ok}/{n} ({100*ok/n:.0f}%)")

    write_run_metadata(
        output_dir,
        start_time=start_time,
        dataset=params["dataset"],
        model=params["model"],
        model_id=model_id,
        source_experiment_id=params["source_experiment_id"],
        n_numeric_values=params["n_numeric_values"],
        n_nonnumeric_values=params["n_nonnumeric_values"],
        n_repeats=params["n_repeats"],
        max_tokens=params["max_tokens"],
        total_calls=total,
        hostname="localhost",
    )
    print(f"\nDone. Results: {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
