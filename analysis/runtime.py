"""LaTeX tables of runtime and compute: per extraction method (``params.runtime``), or per
interp judge over one extraction run (``params.judge_runtime``). A config holds exactly one.

Extraction table
----------------
For one recovery-validity config's runs.

Columns: total time (s), input tokens, output tokens (both in millions), model weight size
(GB = 1e9 bytes), and GPU. Every value is read from the run's own ``run_metadata.json``
except model size, which is the on-disk size of the cached checkpoint.

What each column means, by experiment type (the results-dir level under the dataset):

- ``extraction`` (full pipeline): time is ``total_step_seconds``, the sum of the 7 step
  timings. All 7 must be recorded as run, so a run built on another run's checkpoints
  (e.g. a ``final``-only rerun on copied-in ``values.json``) is rejected rather than
  under-reported. A later ``--step`` rerun replaces that step's timing, but its tokens
  are *added* to the stored total (``experiments/utils.write_run_metadata``), so such a
  run over-counts tokens by the superseded step. That is not detectable from the
  metadata; list those runs under ``footnotes``.
- ``ablation`` / ``baseline_*``: time is ``runtime_seconds``, the wall clock of one
  invocation. These runners record no per-step timing, so a resumed run would be
  under-reported; check the run was a single invocation before listing it.

Neither includes vLLM server startup. Tokens: ``token_usage`` is required unless the run
records ``token_accounting`` instead (GLiNER: no LLM calls; LangExtract: makes its own
HTTP calls), which renders as "--".

Model size: weight files of the cached snapshot that ``hf_revision`` (or ``main``)
points to under ``$HF_CACHE/hub``, namely the files ``model.safetensors.index.json``
names, else the single ``model.safetensors``. Every other cached snapshot of that model
must have the same weight size, since the cache does not record which one a run loaded.

Usage
-----
    python analysis/runtime.py --config analysis/analysis-configs/recovery-validity/<id>.yaml

Table spec (``params.runtime``, every key required)::

    analysis_config: <recovery-validity config id>   # rows must be among its experiment_ids
    dataset: nfix
    decimals: {seconds: 0, tokens_millions: 1, size_gb: 1}
    gpu_labels: {"NVIDIA L40S": "L40S", ...}       # every GPU name met must be listed
    footnotes: {<experiment id>: "text"}          # marks that row's token cells; {} for none
    output: analysis/results/recovery-validity/<name>.tex
    caption: "..."
    label: tab:...
    blocks:                                       # row groups, separated by \\midrule
      - rows:
          - {label: "GLiNER-large", id: <experiment id>}

Interp-judge table
------------------
Columns: time (s), input tokens (millions), judge model weight size (GB), stored activation
size (GB: ``attention_outputs.npz`` + ``layer_outputs.npz`` as written, compressed), and GPU.

- Time is ``runtime_seconds``: the judging loop and output writes, not model loading.
  The judge runs once per invocation (no resume), but check the run was one job.
- Input tokens are not recorded by ``run_judge_interp.py``, so they are recomputed here:
  the judge's prompts are rebuilt from the extraction's ``final.json`` and the run's own
  ``ocr_dir`` with the runner's own code (``judge_prompts.prepare_chat_entries``,
  ``judgementlm.tokenize``) and the judge's tokenizer from the HF cache, then summed.
  The rebuilt prompt count must equal the run's ``responses.json`` rows, and the
  longest prompt must equal the run's recorded ``max_prompt_tokens``, so drift in the
  prompt code, OCR, input file or tokenizer since the run fails loud. One forward pass
  per prompt (``max_new_tokens: 1``), so there is no output-token column.

Table spec (``params.judge_runtime``, every key required)::

    dataset: supermat
    extraction_id: <extraction experiment id>      # every row must have judged exactly this
    decimals: {seconds: 0, tokens_millions: 1, size_gb: 1}
    gpu_labels: {"NVIDIA L40S": "L40S"}
    output: analysis/results/recovery-validity/<name>.tex
    caption: "..."
    label: tab:...
    rows:
      - {label: "\\texttt{llama-3.1-8b}", id: <judge_interp experiment id>}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))
sys.path.insert(0, str(_REPO_ROOT))

from analysis.common.config import _load_envelope, analysis_config_path, get_section  # noqa: E402
import utils as paths  # noqa: E402

SECTIONS = ("runtime", "judge_runtime")
SECTION = "runtime"
SECTION_KEYS = (
    "analysis_config", "dataset", "decimals", "gpu_labels", "footnotes", "output", "caption", "label", "blocks",
)
DECIMAL_KEYS = ("seconds", "tokens_millions", "size_gb")
PIPELINE_STEPS = ("entities", "attributes", "entity_prov", "attribute_prov", "events", "values", "final")
# Experiment types timed by one invocation's runtime_seconds.
RUNTIME_SECONDS_TYPES = (
    "ablation", "baseline_gliner", "baseline_langextract", "baseline_nuextract3", "baseline_chatextract",
)
MISSING_CELL = "--"
HEADER = (
    "Method & Time (s) & Input tok.\\ (M) & Output tok.\\ (M) & Size (GB) & GPU \\\\"
)
JUDGE_SECTION = "judge_runtime"
JUDGE_SECTION_KEYS = ("dataset", "extraction_id", "decimals", "gpu_labels", "output", "caption", "label", "rows")
ACTIVATION_FILES = ("attention_outputs.npz", "layer_outputs.npz")
JUDGE_HEADER = "Judge & Time (s) & Input tok.\\ (M) & Weights (GB) & Activations (GB) & GPU \\\\"


# ---------------------------------------------------------------------------
# Loading + validation
# ---------------------------------------------------------------------------


def load_spec(path: Path) -> dict:
    """Load and validate a runtime table spec.

    Args:
        path: Config path (must live in analysis-configs/recovery-validity/).

    Returns:
        The ``params.runtime`` section.

    Raises:
        ValueError, KeyError: Bad envelope, keys or types, a duplicate row id, or a
            footnote for an id that is not a row.
    """
    spec = _load_section(path, SECTION, SECTION_KEYS)
    _check_common(path, spec, ("analysis_config", "dataset", "output", "caption", "label"), ("gpu_labels", "footnotes"))

    blocks = spec["blocks"]
    if not isinstance(blocks, list) or not blocks:
        raise ValueError(f"{path}: blocks must be a non-empty list")
    ids: list[str] = []
    for block in blocks:
        if not isinstance(block, dict) or set(block) != {"rows"}:
            raise ValueError(f"{path}: each block needs exactly the key 'rows', got {block!r}")
        ids += _check_rows(path, block["rows"])
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"{path}: experiment id(s) in more than one row: {dupes}")
    stray = sorted(set(spec["footnotes"]) - set(ids))
    if stray:
        raise ValueError(f"{path}: footnotes for id(s) that are not table rows: {stray}")
    return spec


def load_judge_spec(path: Path) -> dict:
    """Load and validate an interp-judge runtime table spec.

    Args:
        path: Config path (must live in analysis-configs/recovery-validity/).

    Returns:
        The ``params.judge_runtime`` section.

    Raises:
        ValueError, KeyError: Bad envelope, keys or types, or a duplicate row id.
    """
    spec = _load_section(path, JUDGE_SECTION, JUDGE_SECTION_KEYS)
    _check_common(path, spec, ("dataset", "extraction_id", "output", "caption", "label"), ("gpu_labels",))
    ids = _check_rows(path, spec["rows"])
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"{path}: experiment id(s) in more than one row: {dupes}")
    return spec


def config_section(path: Path) -> str:
    """Which table a config builds: its single ``params`` key, one of SECTIONS.

    Raises:
        ValueError: params is not exactly one of SECTIONS.
    """
    params = _load_envelope(path, "recovery-validity")["params"]
    if len(params) != 1 or next(iter(params)) not in SECTIONS:
        raise ValueError(f"{path}: params keys {sorted(params)} must be exactly one of {list(SECTIONS)}")
    return next(iter(params))


def _load_section(path: Path, section: str, keys: tuple[str, ...]) -> dict:
    """The config's only params section, checked to be ``section`` with exactly ``keys``."""
    cfg = _load_envelope(path, "recovery-validity")
    if set(cfg["params"]) != {section}:
        raise ValueError(f"{path}: params keys {sorted(cfg['params'])} must be exactly [{section!r}]")
    return get_section(cfg, section, required_keys=keys)


def _check_common(path: Path, spec: dict, str_keys: tuple[str, ...], mapping_keys: tuple[str, ...]) -> None:
    """Check non-empty strings, the decimals block, and string-to-string mappings."""
    for key in str_keys:
        if not isinstance(spec[key], str) or not spec[key]:
            raise ValueError(f"{path}: {key} must be a non-empty string, got {spec[key]!r}")
    decimals = spec["decimals"]
    if not isinstance(decimals, dict) or set(decimals) != set(DECIMAL_KEYS):
        raise ValueError(f"{path}: decimals keys must be exactly {sorted(DECIMAL_KEYS)}, got {decimals!r}")
    for key, d in decimals.items():
        if isinstance(d, bool) or not isinstance(d, int) or d < 0:
            raise ValueError(f"{path}: decimals.{key} must be a non-negative int, got {d!r}")
    for key in mapping_keys:
        mapping = spec[key]
        if not isinstance(mapping, dict) or not all(
            isinstance(k, str) and isinstance(v, str) and k and v for k, v in mapping.items()
        ):
            raise ValueError(f"{path}: {key} must be a mapping of non-empty strings, got {mapping!r}")


def _check_rows(path: Path, rows) -> list[str]:
    """Check a non-empty list of ``{label, id}`` rows; return their ids in order."""
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: a row list is empty")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"label", "id"}:
            raise ValueError(f"{path}: each row needs exactly keys ['id', 'label'], got {row!r}")
        if not all(isinstance(row[k], str) and row[k] for k in ("label", "id")):
            raise ValueError(f"{path}: row label and id must be non-empty strings, got {row!r}")
    return [row["id"] for row in rows]


def check_rows_in_analysis_config(spec: dict) -> None:
    """Fail unless every row id is one of the named recovery-validity config's runs.

    Raises:
        ValueError: The config is not a recovery_validity config for ``spec['dataset']``,
            or a row id is not in its ``experiment_ids``.
    """
    config_path = analysis_config_path("recovery-validity", spec["analysis_config"])
    params = _load_envelope(config_path, "recovery-validity")["params"]
    if "recovery_validity" not in params or "experiment_ids" not in params:
        raise ValueError(f"{config_path}: not a recovery_validity analysis config")
    row_ids = [r["id"] for b in spec["blocks"] for r in b["rows"]]
    extra = sorted(set(row_ids) - set(params["experiment_ids"]))
    if extra:
        raise ValueError(f"row id(s) {extra} are not in {config_path}'s experiment_ids")


def run_stats(meta: dict, experiment_type: str, experiment_id: str) -> dict:
    """Time, tokens and GPU of one run from its run_metadata.

    Args:
        meta: Parsed ``run_metadata.json``.
        experiment_type: Results-dir type (``extraction``, ``ablation``, ``baseline_*``).
        experiment_id: For error messages.

    Returns:
        ``{seconds, prompt_tokens, completion_tokens, gpu, model_id, hf_revision}``;
        the token values are None when the run records ``token_accounting`` instead.

    Raises:
        ValueError: Unknown type, an incomplete pipeline step record, a missing or
            inconsistent time field, missing token accounting, or not exactly one GPU.
    """
    where = f"{experiment_id} ({experiment_type})"
    if experiment_type == "extraction":
        steps = meta.get("step_seconds")
        if not isinstance(steps, dict) or set(steps) != set(PIPELINE_STEPS):
            raise ValueError(
                f"{where}: step_seconds must record exactly the steps {list(PIPELINE_STEPS)}, "
                f"got {sorted(steps) if isinstance(steps, dict) else steps!r} -- built on another run's checkpoints?"
            )
        not_run = [s for s in PIPELINE_STEPS if not steps[s].get("ran") or steps[s].get("seconds") is None]
        if not_run:
            raise ValueError(f"{where}: step(s) {not_run} have no recorded run in this directory")
        seconds = sum(steps[s]["seconds"] for s in PIPELINE_STEPS)
        if round(seconds, 1) != meta["total_step_seconds"]:
            raise ValueError(f"{where}: total_step_seconds {meta['total_step_seconds']} != step sum {seconds}")
        # Present only when the last write was a full-pipeline invocation, which wraps every step.
        if "runtime_seconds" in meta and meta["runtime_seconds"] < seconds:
            raise ValueError(f"{where}: runtime_seconds {meta['runtime_seconds']} < step sum {seconds}")
    elif experiment_type in RUNTIME_SECONDS_TYPES:
        if "step_seconds" in meta:
            raise ValueError(f"{where}: has step_seconds; this type is timed by runtime_seconds -- handle it explicitly")
        if "runtime_seconds" not in meta:
            raise ValueError(f"{where}: no runtime_seconds")
        seconds = meta["runtime_seconds"]
    else:
        raise ValueError(f"{where}: unknown experiment type; known: ('extraction',) + {RUNTIME_SECONDS_TYPES}")

    if "token_usage" in meta and "token_accounting" in meta:
        raise ValueError(f"{where}: has both token_usage and token_accounting")
    if "token_usage" in meta:
        usage = meta["token_usage"]
        for key in ("prompt_tokens", "completion_tokens"):
            if not isinstance(usage.get(key), int) or usage[key] <= 0:
                raise ValueError(f"{where}: token_usage.{key} must be a positive int, got {usage.get(key)!r}")
        prompt_tokens, completion_tokens = usage["prompt_tokens"], usage["completion_tokens"]
    elif "token_accounting" in meta:
        if not str(meta["token_accounting"]).startswith("n/a"):
            raise ValueError(f"{where}: token_accounting {meta['token_accounting']!r} is not an 'n/a' note")
        prompt_tokens = completion_tokens = None
    else:
        raise ValueError(f"{where}: neither token_usage nor token_accounting recorded")

    gpus = meta.get("gpu_info")
    if not isinstance(gpus, list) or len(gpus) != 1:
        raise ValueError(f"{where}: gpu_info must list exactly one GPU, got {gpus!r}")
    for key in ("model_id", "hf_revision"):
        if key not in meta:
            raise ValueError(f"{where}: no {key} in run_metadata")
    return {
        "seconds": seconds, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "gpu": gpus[0]["name"], "model_id": meta["model_id"], "hf_revision": meta["hf_revision"],
    }


def _weight_files(snapshot: Path) -> list[Path]:
    """Weight files of one cached snapshot: those the safetensors index names, else model.safetensors."""
    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        with open(index) as f:
            names = sorted(set(json.load(f)["weight_map"].values()))
    elif (snapshot / "model.safetensors").exists():
        names = ["model.safetensors"]
    else:
        raise FileNotFoundError(f"{snapshot}: neither model.safetensors.index.json nor model.safetensors")
    files = [snapshot / n for n in names]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{snapshot}: weight file(s) not downloaded: {missing}")
    return files


def model_weight_bytes(hub: Path, model_id: str, revision: str | None) -> int:
    """On-disk size of a cached model's weight files, in bytes.

    Args:
        hub: ``$HF_CACHE/hub``.
        model_id: HuggingFace repo id.
        revision: The run's ``hf_revision``; None means ``main``.

    Returns:
        Total bytes of the snapshot's weight files (symlinks followed).

    Raises:
        FileNotFoundError: Model, ref, snapshot or weight files not in the cache.
        ValueError: Another cached snapshot of the model has a different weight size.
    """
    repo = hub / f"models--{model_id.replace('/', '--')}"
    ref = repo / "refs" / (revision or "main")
    if ref.exists():
        snapshot = repo / "snapshots" / ref.read_text().strip()
    elif revision is not None:
        snapshot = repo / "snapshots" / revision
    else:
        raise FileNotFoundError(f"{ref} not found -- is {model_id} in the cache?")
    if not snapshot.is_dir():
        raise FileNotFoundError(f"{snapshot} not found")
    size = sum(p.stat().st_size for p in _weight_files(snapshot))
    for other in sorted((repo / "snapshots").iterdir()):
        if other == snapshot:
            continue
        other_size = sum(p.stat().st_size for p in _weight_files(other))
        if other_size != size:
            raise ValueError(
                f"{model_id}: cached snapshots {snapshot.name} ({size} B) and {other.name} ({other_size} B) differ; "
                "cannot tell which one the run loaded"
            )
    return size


def collect(spec: dict, hub: Path) -> dict[str, dict]:
    """Stats for every row id: run_metadata fields plus model size.

    Raises:
        ValueError: A run's results dir is under another dataset, its metadata names
            another dataset, or a GPU name has no ``gpu_labels`` entry.
    """
    out: dict[str, dict] = {}
    sizes: dict[tuple, int] = {}
    for block in spec["blocks"]:
        for row in block["rows"]:
            exp_id = row["id"]
            result_dir = paths.find_result_dir(exp_id)
            dataset, experiment_type = result_dir.relative_to(paths.RESULTS_ROOT).parts[:2]
            meta = paths.load_run_metadata(result_dir)
            if meta is None:
                raise FileNotFoundError(f"{result_dir}: no run_metadata.json")
            if dataset != spec["dataset"] or meta.get("dataset") != spec["dataset"]:
                raise ValueError(
                    f"{exp_id}: results dir dataset {dataset!r} / metadata dataset {meta.get('dataset')!r} "
                    f"!= {spec['dataset']!r}"
                )
            stats = run_stats(meta, experiment_type, exp_id)
            if stats["gpu"] not in spec["gpu_labels"]:
                raise ValueError(f"{exp_id}: GPU {stats['gpu']!r} has no gpu_labels entry")
            key = (stats["model_id"], stats["hf_revision"])
            if key not in sizes:
                sizes[key] = model_weight_bytes(hub, *key)
            stats["weight_bytes"] = sizes[key]
            out[exp_id] = stats
    return out


def judge_run_stats(meta: dict, experiment_id: str, extraction_id: str) -> dict:
    """Time, GPU and model of one interp-judge run from its run_metadata.

    Args:
        meta: Parsed ``run_metadata.json``.
        experiment_id: For error messages.
        extraction_id: The extraction run every row must have judged.

    Returns:
        ``{seconds, gpu, model_id, max_prompt_tokens}``.

    Raises:
        ValueError: The run judged another extraction, or a field is missing / not
            exactly one GPU.
    """
    for key in ("runtime_seconds", "judge_model_id", "max_prompt_tokens", "extraction_id"):
        if key not in meta:
            raise ValueError(f"{experiment_id}: no {key} in run_metadata")
    if meta["extraction_id"] != extraction_id:
        raise ValueError(f"{experiment_id}: judged {meta['extraction_id']!r}, not {extraction_id!r}")
    gpus = meta.get("gpu_info")
    if not isinstance(gpus, list) or len(gpus) != 1:
        raise ValueError(f"{experiment_id}: gpu_info must list exactly one GPU, got {gpus!r}")
    return {
        "seconds": meta["runtime_seconds"], "gpu": gpus[0]["name"], "model_id": meta["judge_model_id"],
        "max_prompt_tokens": meta["max_prompt_tokens"],
    }


def activation_bytes(result_dir: Path) -> int:
    """Total on-disk size of a judge run's stored activations.

    Raises:
        FileNotFoundError: Either activation file is missing.
    """
    files = [result_dir / name for name in ACTIVATION_FILES]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"activation file(s) missing: {missing}")
    return sum(p.stat().st_size for p in files)


def check_prompt_lengths(lengths: list[int], n_responses: int, max_prompt_tokens: int, experiment_id: str) -> int:
    """Total input tokens, after checking the rebuilt prompts against the run's own record.

    Args:
        lengths: Rebuilt prompt length per judged row.
        n_responses: Rows in the run's ``responses.json``.
        max_prompt_tokens: The run's recorded longest prompt.
        experiment_id: For error messages.

    Returns:
        ``sum(lengths)``.

    Raises:
        ValueError: Row count or longest prompt differs from the run's record.
    """
    if len(lengths) != n_responses:
        raise ValueError(f"{experiment_id}: rebuilt {len(lengths)} prompts, run judged {n_responses} rows")
    if max(lengths) != max_prompt_tokens:
        raise ValueError(
            f"{experiment_id}: longest rebuilt prompt is {max(lengths)} tokens, run recorded {max_prompt_tokens} "
            "-- prompt code, OCR, input file or tokenizer changed since the run"
        )
    return sum(lengths)


def judge_prompt_lengths(dataset: str, experiment_id: str, model_id: str) -> list[int]:
    """Rebuild a judge run's prompts with the runner's own code and count their tokens.

    Mirrors ``run_judge_interp.run_interp_judge`` for a run over an extraction's
    ``final.json`` (not synthetic mode).

    Args:
        dataset: Dataset name.
        experiment_id: The judge_interp experiment id.
        model_id: The judge model id recorded by the run.

    Returns:
        Token count per prompt, in ``prepare_chat_entries`` order.

    Raises:
        ValueError: The experiment config is a synthetic-mode run, lacks ``ocr_dir``,
            or names a judge whose model config has another model id.
    """
    from transformers import AutoTokenizer
    from run_extraction import load_dataset_config
    from scholarlm.judgementlm import tokenize
    from scholarlm.utils import judge_prompts

    config_path = paths.experiment_config_dir(dataset, "judge_interp", experiment_id) / f"{experiment_id}.yaml"
    params = paths.load_experiment_config(config_path)["params"]
    if any(k in params for k in ("synthetic", "synthetic_file")):
        raise ValueError(f"{config_path}: synthetic-mode run; only extraction-judging runs are supported")
    paths.require_params(params, "judge", "extraction_id", "ocr_dir", config_path=config_path)
    judge_cfg = paths.load_model_config("interp_judge", params["judge"])
    if judge_cfg["model_id"] != model_id:
        raise ValueError(f"{config_path}: judge {params['judge']!r} is now {judge_cfg['model_id']!r}, run used {model_id!r}")

    dataset_config = load_dataset_config(dataset)
    with open(paths.find_result_dir(params["extraction_id"]) / "final.json") as f:
        data = json.load(f)
    documents = judge_prompts.load_documents_for_dataset(dataset_config, params["ocr_dir"])
    entries = judge_prompts.prepare_chat_entries(data, documents, dataset_config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # The runner's own defaults for these two keys (run_judge_interp.run_interp_judge).
    use_chat_template = judge_cfg.get("use_chat_template", True)
    answer_cue = judge_cfg.get("answer_cue", None)
    return [
        len(tokenize(e["system"], e["context_text"], e["user_query"], tokenizer,
                     use_chat_template=use_chat_template, answer_cue=answer_cue)[0])
        for e in entries
    ]


def collect_judges(spec: dict, hub: Path) -> dict[str, dict]:
    """Stats for every judge row: metadata fields, recounted input tokens, weight and activation sizes.

    Raises:
        ValueError: A run is not a judge_interp run of ``spec['dataset']``, or a GPU
            name has no ``gpu_labels`` entry.
    """
    out: dict[str, dict] = {}
    for row in spec["rows"]:
        exp_id = row["id"]
        result_dir = paths.find_result_dir(exp_id)
        dataset, experiment_type = result_dir.relative_to(paths.RESULTS_ROOT).parts[:2]
        if (dataset, experiment_type) != (spec["dataset"], "judge_interp"):
            raise ValueError(f"{exp_id}: is a {dataset}/{experiment_type} run, not {spec['dataset']}/judge_interp")
        meta = paths.load_run_metadata(result_dir)
        if meta is None:
            raise FileNotFoundError(f"{result_dir}: no run_metadata.json")
        stats = judge_run_stats(meta, exp_id, spec["extraction_id"])
        if stats["gpu"] not in spec["gpu_labels"]:
            raise ValueError(f"{exp_id}: GPU {stats['gpu']!r} has no gpu_labels entry")
        with open(result_dir / "responses.json") as f:
            n_responses = len(json.load(f))
        print(f"Recounting judge prompt tokens for {exp_id} ...")
        lengths = judge_prompt_lengths(dataset, exp_id, stats["model_id"])
        stats["prompt_tokens"] = check_prompt_lengths(lengths, n_responses, stats["max_prompt_tokens"], exp_id)
        stats["weight_bytes"] = model_weight_bytes(hub, stats["model_id"], None)
        stats["activation_bytes"] = activation_bytes(result_dir)
        out[exp_id] = stats
    return out


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _footnote_marks(spec: dict) -> dict[str, str]:
    """Footnote mark per footnoted id, lettered in row order."""
    ordered = [r["id"] for b in spec["blocks"] for r in b["rows"] if r["id"] in spec["footnotes"]]
    return {exp_id: chr(ord("a") + i) for i, exp_id in enumerate(ordered)}


def format_row(label: str, stats: dict, decimals: dict, gpu_label: str, mark: str | None) -> str:
    """One LaTeX table row.

    Args:
        label: Method label.
        stats: Output of ``run_stats`` plus ``weight_bytes``.
        decimals: ``{seconds, tokens_millions, size_gb}``.
        gpu_label: Display name for the GPU.
        mark: Footnote mark for the token cells, or None.

    Returns:
        ``label & time & in & out & size & gpu \\\\``.
    """
    sup = f"$^{{{mark}}}$" if mark else ""
    tokens = []
    for key in ("prompt_tokens", "completion_tokens"):
        v = stats[key]
        tokens.append(MISSING_CELL if v is None else f"{v / 1e6:,.{decimals['tokens_millions']}f}{sup}")
    cells = [
        label,
        f"{stats['seconds']:,.{decimals['seconds']}f}",
        *tokens,
        f"{stats['weight_bytes'] / 1e9:.{decimals['size_gb']}f}",
        gpu_label,
    ]
    return " & ".join(cells) + " \\\\"


def build_table(spec: dict, stats: dict[str, dict]) -> str:
    """Render the full LaTeX ``table`` environment.

    Args:
        spec: Output of ``load_spec``.
        stats: Output of ``collect``.

    Returns:
        LaTeX source.
    """
    marks = _footnote_marks(spec)
    for exp_id in marks:
        if stats[exp_id]["prompt_tokens"] is None:
            raise ValueError(f"footnote on {exp_id}, whose token cells are {MISSING_CELL!r}")
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{spec['caption']}}}",
        f"\\label{{{spec['label']}}}",
        "\\begin{tabular}{lrrrrl}",
        "\\toprule",
        HEADER,
    ]
    for block in spec["blocks"]:
        lines.append("\\midrule")
        for row in block["rows"]:
            s = stats[row["id"]]
            lines.append(format_row(
                row["label"], s, spec["decimals"], spec["gpu_labels"][s["gpu"]], marks.get(row["id"])))
    lines += ["\\bottomrule", "\\end{tabular}"]
    if marks:
        notes = " ".join(f"$^{{{m}}}${spec['footnotes'][i]}" for i, m in marks.items())
        lines.append(f"\\par\\smallskip{{\\footnotesize {notes}}}")
    lines += ["\\end{table}", ""]
    return "\n".join(lines)


def format_judge_row(label: str, stats: dict, decimals: dict, gpu_label: str) -> str:
    """One interp-judge table row: ``label & time & in & weights & activations & gpu \\\\``."""
    size = decimals["size_gb"]
    cells = [
        label,
        f"{stats['seconds']:,.{decimals['seconds']}f}",
        f"{stats['prompt_tokens'] / 1e6:,.{decimals['tokens_millions']}f}",
        f"{stats['weight_bytes'] / 1e9:.{size}f}",
        f"{stats['activation_bytes'] / 1e9:.{size}f}",
        gpu_label,
    ]
    return " & ".join(cells) + " \\\\"


def build_judge_table(spec: dict, stats: dict[str, dict]) -> str:
    """Render the interp-judge LaTeX ``table`` environment.

    Args:
        spec: Output of ``load_judge_spec``.
        stats: Output of ``collect_judges``.

    Returns:
        LaTeX source.
    """
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{spec['caption']}}}",
        f"\\label{{{spec['label']}}}",
        "\\begin{tabular}{lrrrrl}",
        "\\toprule",
        JUDGE_HEADER,
        "\\midrule",
    ]
    for row in spec["rows"]:
        s = stats[row["id"]]
        lines.append(format_judge_row(row["label"], s, spec["decimals"], spec["gpu_labels"][s["gpu"]]))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI: build the table from ``--config``, write it, and print it.

    Args:
        argv: Arguments (None = sys.argv).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="analysis-configs/recovery-validity/<id>.yaml table spec")
    args = parser.parse_args(argv)

    if "HF_CACHE" not in os.environ:
        raise KeyError("HF_CACHE is not set (needed to size model weights)")
    hub = Path(os.environ["HF_CACHE"]) / "hub"
    if config_section(args.config) == JUDGE_SECTION:
        spec = load_judge_spec(args.config)
        text = build_judge_table(spec, collect_judges(spec, hub))
    else:
        spec = load_spec(args.config)
        check_rows_in_analysis_config(spec)
        text = build_table(spec, collect(spec, hub))
    output = Path(spec["output"])
    output = output if output.is_absolute() else _REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    print(text)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
