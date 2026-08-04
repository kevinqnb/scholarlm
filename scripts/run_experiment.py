"""Adapter: run one experiment per the shared experiment contract.

Usage
-----
    python scripts/run_experiment.py configs/<id>.yaml [--api-base URL]

Reads configs/<id>.yaml (the contract's standardized envelope: id, project,
description, seed, params), translates `params` into a call to this repo's
existing entry point (experiments/run_extraction.py or run_ablation.py — it
does not reimplement either), and writes the contract's standardized output
to $RUNS_ROOT/<id>/: run.json, config.snapshot.yaml, metrics.json, log.txt,
figures/, artifacts/ (a symlink back into this repo's own
data/experiments/... tree — never copied).

Supported `params` keys
------------------------
    entry_point   "extraction" (default) | "ablation"
    dataset       required — experiments/configs/<dataset>.py must define CONFIG
    model         required — key in experiments/model_registry.py MODEL_REGISTRY
    date          optional — YYYY_mm_dd output date tag. Defaults to the date
                  encoded in the experiment id itself (YYYY-MM-DD-slug-NN), so
                  the run directory this repo already uses is deterministic
                  from the id alone.
    ablation      required if entry_point == "ablation" (one of 1-6)
    ocr_dir       optional — pre-cleaned OCR .txt directory
    paper_subset  optional — list of paper codes, overrides the dataset's default
    resume        optional bool (extraction only) — skip steps already on disk
    final_only    optional bool (extraction only) — discard intermediate files
    step          optional (extraction only) — run a single named pipeline step
    api_base      optional — vLLM server URL. The --api-base CLI flag (used by
                  submit.sh, which learns the compute node's port at run time)
                  always overrides this.

Deliberately NOT a params key: api_key. configs/<id>.yaml is committed to the
public repo; run_extraction.py already resolves OPENAI_API_KEY / GEMINI_API_KEY
from the environment for frontier models, and that is left untouched.

Assumptions (flagged per the contract's adapter instructions)
---------------------------------------------------------------
- `seed` in configs/<id>.yaml must equal experiments/config.yaml's
  `defaults.seed`. Neither run_extraction.py nor run_ablation.py accepts a
  --seed flag; the seed is a fixed property of this repo's config, not a
  per-run knob. Rather than silently ignore the contract's `seed` field, this
  adapter fails loudly if the two disagree.
- Metrics are best-effort. `recovery_rate` is computed whenever the dataset's
  ground_truth_file resolves and analysis.ablation.get_matching_rules(dataset)
  recognizes the dataset. `validity_rate` is computed ONLY when a combined
  judge run already exists for this (dataset, model, date) — with no judge
  output, validity_rate has no way to distinguish "unmatched but potentially
  correct" from "hallucinated", so it is omitted from metrics.json entirely
  rather than written as a misleadingly low number.
- figures/ is created empty. Nothing in run_extraction.py / run_ablation.py
  produces figures; that happens later in separate analysis/ scripts, out of
  scope for a single run.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_envelope(config_path: Path) -> dict:
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    missing = [k for k in ("id", "project", "description", "seed", "params") if k not in cfg]
    if missing:
        raise ValueError(f"{config_path}: missing required key(s): {missing}")
    if cfg["id"] != config_path.stem:
        raise ValueError(
            f"{config_path}: id '{cfg['id']}' does not match filename stem '{config_path.stem}'"
        )
    if not isinstance(cfg["params"], dict):
        raise ValueError(f"{config_path}: params must be a mapping")
    return cfg


def _default_date(id_: str) -> str:
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})-", id_)
    if not m:
        raise ValueError(
            f"Cannot derive a date from id '{id_}' (expected YYYY-MM-DD-slug-NN); "
            "set params.date explicitly."
        )
    return f"{m.group(1)}_{m.group(2)}_{m.group(3)}"


def _write_run_json(run_dir: Path, payload: dict) -> None:
    with open(run_dir / "run.json", "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Validation (before the expensive subprocess call)
# ---------------------------------------------------------------------------


def _validate(cfg: dict, repo_root: Path) -> tuple:
    """Validate the envelope + params against this repo's real registries.

    Returns (dataset_config, model_config, entry_point, ablation) on success.
    Raises ValueError with a specific, actionable message on failure.
    """
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root / "experiments"))

    from run_extraction import load_dataset_config, get_model_config
    from model_registry import MODEL_REGISTRY
    from utils import load_config as load_experiments_config

    params = cfg["params"]

    entry_point = params.get("entry_point", "extraction")
    if entry_point not in ("extraction", "ablation"):
        raise ValueError(f"params.entry_point must be 'extraction' or 'ablation', got {entry_point!r}")

    if "dataset" not in params:
        raise ValueError("params.dataset is required")
    if "model" not in params:
        raise ValueError("params.model is required")

    dataset_config = load_dataset_config(params["dataset"])  # raises FileNotFoundError if unknown
    model_config = get_model_config(params["model"])  # raises KeyError if unknown
    if params["model"] not in MODEL_REGISTRY:
        raise ValueError(f"model '{params['model']}' not in MODEL_REGISTRY")

    ablation = None
    if entry_point == "ablation":
        from run_ablation import ABLATION_REGISTRY

        ablation = params.get("ablation")
        if ablation is None:
            raise ValueError("params.ablation is required when entry_point == 'ablation'")
        ablation = str(ablation)
        if ablation not in ABLATION_REGISTRY:
            raise ValueError(
                f"params.ablation '{ablation}' not in ABLATION_REGISTRY "
                f"(choices: {sorted(ABLATION_REGISTRY.keys())})"
            )

    exp_defaults = load_experiments_config().get("defaults", {})
    repo_seed = exp_defaults.get("seed")
    if repo_seed is not None and cfg["seed"] != repo_seed:
        raise ValueError(
            f"configs seed ({cfg['seed']}) does not match experiments/config.yaml "
            f"defaults.seed ({repo_seed}). Neither run_extraction.py nor run_ablation.py "
            "takes a --seed flag -- the repo's seed is fixed, not a per-run parameter. "
            "Set seed to match, or change experiments/config.yaml if you actually mean "
            "to change the repo-wide seed."
        )

    # Best-effort: warn (don't fail) if metrics won't be computable later.
    try:
        os.chdir(repo_root)
        sys.path.insert(0, str(repo_root))
        from analysis.ablation import get_matching_rules

        get_matching_rules(params["dataset"])
    except Exception as e:
        print(f"[validate] warning: recovery/validity metrics will be unavailable ({e})")

    if dataset_config.ground_truth_file is not None:
        gt_path = repo_root / dataset_config.ground_truth_file
        if not gt_path.exists():
            print(f"[validate] warning: ground_truth_file not found ({gt_path}); recovery_rate will be unavailable")

    return dataset_config, model_config, entry_point, ablation


# ---------------------------------------------------------------------------
# argv construction — thin translation of params into this repo's own CLI
# ---------------------------------------------------------------------------


def _build_argv(entry_point: str, params: dict, date: str, ablation: str | None, api_base: str | None) -> list[str]:
    argv = ["--dataset", params["dataset"], "--model", params["model"], "--date", date]

    if entry_point == "ablation":
        argv += ["--ablation", ablation]

    if params.get("ocr_dir"):
        argv += ["--ocr-dir", params["ocr_dir"]]
    if params.get("paper_subset"):
        argv += ["--paper-subset", *params["paper_subset"]]

    if entry_point == "extraction":
        if params.get("resume"):
            argv.append("--resume")
        if params.get("final_only"):
            argv.append("--final-only")
        if params.get("step"):
            argv += ["--step", params["step"]]

    effective_api_base = api_base or params.get("api_base")
    if effective_api_base:
        argv += ["--api-base", effective_api_base]

    return argv


# ---------------------------------------------------------------------------
# Metrics shim — reuses analysis/{loaders,metrics,ablation}.py, never
# reimplements matching/scoring logic.
# ---------------------------------------------------------------------------


def _compute_metrics(
    dataset: str,
    model: str,
    date: str,
    entry_point: str,
    ablation: str | None,
    output_dir: Path,
    repo_root: Path,
) -> dict:
    metrics: dict = {}

    final_path = output_dir / "final.json"
    if not final_path.exists():
        print(f"[metrics] {final_path} does not exist (single --step run?) -- skipping")
        return metrics

    with open(final_path) as f:
        records = json.load(f)
    metrics["n_measurements"] = len(records)
    metrics["n_documents"] = len({r.get("document_id") for r in records})

    run_meta_path = output_dir / "run_metadata.json"
    if run_meta_path.exists():
        with open(run_meta_path) as f:
            run_meta = json.load(f)
        if "runtime_seconds" in run_meta:
            metrics["runtime_seconds"] = float(run_meta["runtime_seconds"])

    if not records:
        return metrics

    try:
        os.chdir(repo_root)
        sys.path.insert(0, str(repo_root))
        import pandas as pd
        from analysis.ablation import get_matching_rules, process_extraction_df
        from analysis.loaders import load_ground_truth, load_combined_judgements
        from analysis.metrics import recovery_rate, validity_rate
        from run_extraction import load_dataset_config
    except Exception as e:
        print(f"[metrics] skipping recovery/validity: import failed ({e})")
        return metrics

    try:
        dataset_config = load_dataset_config(dataset)
        ground_truth_df = load_ground_truth(dataset_config)
        strict_matching, fuzzy_matching, fuzzy_threshold = get_matching_rules(dataset)
    except Exception as e:
        print(f"[metrics] skipping recovery/validity for dataset '{dataset}': {e}")
        return metrics

    extraction_df = pd.DataFrame(records)
    extraction_df = process_extraction_df(extraction_df, dataset, dataset_config)
    cache_path = output_dir / "match_cache.pkl"

    try:
        recov, recov_lo, recov_hi = recovery_rate(
            ground_truth_df,
            extraction_df,
            strict_matching=strict_matching,
            fuzzy_matching=fuzzy_matching,
            fuzzy_threshold=fuzzy_threshold,
            cache_path=cache_path,
            return_ci=True,
        )
        metrics["recovery_rate"] = recov
        metrics["recovery_rate_ci_lo"] = recov_lo
        metrics["recovery_rate_ci_hi"] = recov_hi
    except Exception as e:
        print(f"[metrics] recovery_rate failed: {e}")

    try:
        judged_df = pd.DataFrame(
            load_combined_judgements(dataset, model, date, ablation=ablation)
        )
    except FileNotFoundError:
        judged_df = None
        print("[metrics] no combined judge output for this run -- omitting validity_rate "
              "(with no judge output, validity_rate would degenerate to precision, which "
              "is not the same thing and would be reported under a misleading name)")

    if judged_df is not None:
        try:
            valid, valid_lo, valid_hi = validity_rate(
                ground_truth_df,
                extraction_df,
                strict_matching=strict_matching,
                fuzzy_matching=fuzzy_matching,
                fuzzy_threshold=fuzzy_threshold,
                judged_df=judged_df,
                cache_path=cache_path,
                return_ci=True,
            )
            metrics["validity_rate"] = valid
            metrics["validity_rate_ci_lo"] = valid_lo
            metrics["validity_rate_ci_hi"] = valid_hi
        except Exception as e:
            print(f"[metrics] validity_rate failed: {e}")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="Path to configs/<id>.yaml")
    p.add_argument(
        "--api-base",
        default=None,
        metavar="URL",
        help="Override params.api_base (submit.sh uses this to inject the compute node's vLLM endpoint).",
    )
    args = p.parse_args(argv)

    config_path = Path(args.config).resolve()
    cfg = _load_envelope(config_path)
    id_ = cfg["id"]

    runs_root = os.environ.get("RUNS_ROOT")
    if not runs_root:
        print("error: RUNS_ROOT is not set", file=sys.stderr)
        return 1
    run_dir = Path(runs_root) / id_
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)

    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    from utils import get_git_info

    git_info = get_git_info()
    started_at = _now()
    job_id = os.environ.get("JOB_ID", "local")
    host = socket.gethostname()
    config_snapshot_path = run_dir / "config.snapshot.yaml"
    shutil.copy2(config_path, config_snapshot_path)

    base_run_json = {
        "id": id_,
        "project": cfg["project"],
        "git_sha": git_info["commit"],
        "git_dirty": git_info["dirty"],
        "started_at": started_at,
        "finished_at": None,
        "status": "running",
        "host": host,
        "job_id": job_id,
        "config_path": str(config_path.relative_to(REPO_ROOT)) if REPO_ROOT in config_path.parents else str(config_path),
    }
    _write_run_json(run_dir, base_run_json)

    log_path = run_dir / "log.txt"

    try:
        dataset_config, model_config, entry_point, ablation = _validate(cfg, REPO_ROOT)
    except Exception as e:
        base_run_json.update(status="failed", finished_at=_now(), error=str(e))
        _write_run_json(run_dir, base_run_json)
        with open(log_path, "a") as f:
            f.write(f"[validate] {e}\n")
        print(f"error: {e}", file=sys.stderr)
        return 1

    params = cfg["params"]
    date = params.get("date") or _default_date(id_)
    entry_script = REPO_ROOT / "experiments" / ("run_ablation.py" if entry_point == "ablation" else "run_extraction.py")
    entry_argv = _build_argv(entry_point, params, date, ablation, args.api_base)

    print(f"[run_experiment] {id_}: {entry_script.name} {' '.join(entry_argv)}")

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            [sys.executable, str(entry_script), *entry_argv],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            log_f.write(line)
        returncode = proc.wait()

    finished_at = _now()
    status = "success" if returncode == 0 else "failed"
    base_run_json.update(status=status, finished_at=finished_at)
    _write_run_json(run_dir, base_run_json)

    import paths as _paths

    if entry_point == "ablation":
        output_dir = _paths.ablation(params["dataset"], ablation, params["model"], date)
    else:
        output_dir = _paths.extraction(params["dataset"], params["model"], date)

    artifacts_link = run_dir / "artifacts"
    if artifacts_link.is_symlink() or artifacts_link.exists():
        if artifacts_link.is_symlink():
            artifacts_link.unlink()
    if output_dir.exists():
        artifacts_link.symlink_to(output_dir)

    metrics: dict = {}
    if status == "success":
        metrics = _compute_metrics(params["dataset"], params["model"], date, entry_point, ablation, output_dir, REPO_ROOT)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[run_experiment] {id_}: {status} -> {run_dir}")
    return returncode


if __name__ == "__main__":
    sys.exit(main())
