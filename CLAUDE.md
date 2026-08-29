# ScholarlM

This is research code. Its output goes into papers. The failure mode that matters is not a crash — it is code that runs cleanly and produces a number that is quietly wrong. Optimize for catching that.

## Fail loud

Defensive coding is an anti-pattern here. It converts crashes, which I would notice, into wrong results, which I would not.

- No bare `except:` and no `except Exception` that swallows and continues.
- No default values for missing config keys. A missing key is a hard error.
- No silent fallbacks — no "if the GPU isn't available, use CPU," no "if the file is missing, skip it," no substituting an empty result for a failure.
- Assert at every pipeline boundary: shapes, dtypes, row counts, ID sets, and that joins didn't drop or duplicate rows.
- If something is genuinely optional, it gets an explicit flag, not an inferred default.

## Staged gates before any full run

Never go from "code written" to "full experiment submitted." Walk the ladder, and say which rung we are on:

1. **Unit tests** on a tiny hand-built fixture where I can verify the expected output by inspection.
2. **Smoke run** — one step, one batch, smallest possible input, run directly in the current shell session — never by opening a new interactive or batch session yourself. Confirms plumbing, not results.
3. **Tiny end-to-end** whose result I can predict in advance. If it doesn't match the prediction, stop; do not scale up.
4. **Full run**, submitted through the wrapper.

Propose this sequence yourself rather than waiting for me to ask.

## Sanity controls, not just tests

Unit tests catch broken code. These catch broken _experiments_. Include them in the experiment plan, not as an afterthought:

- **Shuffled-label / permutation control** — performance should collapse to chance. If it doesn't, something is leaking.
- **Known-answer case** — an input whose correct output I can state up front.
- **Seed determinism** — two runs with the same seed produce identical output. If they don't, find out why before interpreting anything.
- **Ablation direction** — removing a component I believe matters should hurt. A component that can be deleted with no effect is either useless or unused.

## Guarding the evaluation code

Changes to metric, scoring, or evaluation logic are the highest-risk edits in the repo, because the natural debugging loop — adjust until the numbers look reasonable — is indistinguishable from fitting the metric to the hypothesis.

- Never modify eval or metric code as part of "getting the experiment to run." If the eval breaks, report it and stop.
- Any change to eval logic is a separate, standalone commit with its own justification, and it invalidates every prior number computed with the old version. Say that explicitly when proposing such a change.
- When results look surprising, the first hypothesis is a bug in _our_ code, not a real effect. Investigate in that order.

## ScholarLM

ScholarlM is a research library for extracting entity-attribute-value measurement
triplets from scientific PDFs. A seven-step LLM pipeline (entity identification →
attribute detection → provenance → event resolution → value extraction →
standardization → deduplication) runs against either local open-weight models served
via vLLM or frontier APIs (OpenAI / Gemini), and is evaluated against manually
reviewed ground truth using a separate LLM-judge pipeline for hallucination/validity
scoring.

## Experiment contract

This repo conforms to the experiment contract. When implementing an experiment, read
`notes/hub/conventions.md` first.

**No magic numbers.** Every value that would change between runs — dataset, model,
paper subset, sampling parameters that vary per-experiment, etc. — belongs in a
`configs/<id>.yaml` file's `params` block, not hardcoded in runner code. Fixed,
repo-wide values (the global seed, per-model default sampling params, SGE serve
resources) stay in `experiments/config.yaml`, which is the single source of truth for
those and is itself committed and git-tracked for reproducibility.

## Entry points

The contract-standard way to run an experiment:

```bash
python scripts/run_experiment.py configs/<id>.yaml   # run directly
bash scripts/submit.sh <id>                           # or submit to SGE
```

`scripts/run_experiment.py` is a thin adapter over this repo's own entry points below
— it does not reimplement any of them. `scripts/submit.sh` brings up a vLLM server as
part of the submitted job when the experiment's model needs one (see
`scripts/_run_experiment_job.sh`); frontier-model experiments skip that step.

The native entry points it wraps, callable directly for anything the contract's
`params` shape doesn't cover:

```bash
python experiments/run_extraction.py --dataset pond --model gemma-3-27b
python experiments/run_ablation.py --dataset pond --model gemma-3-27b --ablation 2
python experiments/run_judge_local.py ...      # vLLM judge, local
python experiments/run_judge_interp.py ...     # NNsight judge, collects activations
python experiments/run_judge_combine.py --dataset pond --extraction-model gemma-3-27b --extraction-date 2026_04_01
python experiments/run_jacobian_lens.py ...     # NNsight, Jacobian-lens j-scores (JacobianLensLM)
python experiments/run_baseline_gliner.py ...
python experiments/run_baseline_nuextract.py ...
python experiments/run_baseline_chatextract.py ...
python experiments/run_ocr.py ...
python experiments/process_pdfs.py --dataset pond
```

Run any of them with `--help` for the full flag set. Available datasets are the files
in `experiments/configs/*.py`; available models are the keys of `MODEL_REGISTRY` in
`experiments/model_registry.py`.

## Environment setup

Install with `uv sync` (add `--extra gpu` for local vLLM/nnsight inference, `--extra
dev` for the test/notebook tooling). The following environment variables must be set
in your shell profile — see `CLAUDE.local.md` (gitignored) for this machine's actual
values:

| Variable | Purpose |
|---|---|
| `RUNS_ROOT` | contract-standard run output root |
| `NOTES_ROOT` | private notes repo, symlinked in as `notes/` |
| `VLLM_SIF_DIR` | directory of Singularity images for vLLM serving |
| `HF_CACHE` | HuggingFace weights cache |
| `SINGULARITY_BIND` | bind-mount argument for `singularity exec` |
| `SCHOLARLM_ROOT` | absolute path to this repo, used by generated serve scripts |
| `SGE_PROJECT` | SGE project allocation (`-P` flag) |
| `OPENAI_API_KEY` / `GEMINI_API_KEY` | required only for frontier-model runs |

## Running tests

`pytest` is part of the `dev` extra, not the default install:

```bash
uv run --extra dev pytest
```

## Repo layout

```
src/scholarlm/          Core library — pipeline, config, probe/calibration utilities
experiments/            Runner scripts, model/dataset registries, path helpers
experiments/configs/    One DatasetConfig per dataset (pond.py, nfix.py, …)
scripts/                Experiment-contract adapter (run_experiment.py, submit.sh)
configs/                Committed configs/<id>.yaml, one per experiment
analysis/               Experiment analysis code and notebooks
examples/               Jupyter notebooks
data/experiments/       All outputs — never committed to git
```

## Key concepts

**Extraction pipeline** (`MeasurementLM` in `src/scholarlm/measurementlm.py`)
Seven sequential steps: entities → attributes → entity_prov → attribute_prov → events → values → final. Each step has a JSON checkpoint; `--resume` skips steps whose output already exists.

**Ablations** (`src/scholarlm/measurementlm_ablation{1–6}.py`)
Each ablation subclass overrides one or more pipeline steps. Run via `experiments/run_ablation.py`.

**DatasetConfig / ModelConfig** (`src/scholarlm/config.py`)
Single source of truth for dataset- and model-specific values. Config files live in `experiments/configs/`.

**Path helpers** (`experiments/paths.py`)
Every path in the output tree is constructed here. Never build paths by hand in scripts.

**Judge pipeline**
- `run_judge_interp.py` — NNsight (local, collects attention activations)
- `run_judge_local.py` — vLLM (local, fast)
- `run_judge_combine.py` — majority-vote combination of judge runs → `combined.json`

**Jacobian-lens j-scores** (`JacobianLensLM` in `src/scholarlm/jacobianlenslm.py`)
Separate from the judge pipeline above — probes context-token residuals against a
pretrained per-layer Jacobian lens rather than generating a true/false judgement.
Driven by `run_jacobian_lens.py`; see `notes/scholarlm/threads/Jacobian Lens.md`.

**Analysis utilities** (`src/scholarlm/utils/`)
- `probe.py` — logistic-regression probe on attention activations
- `calibration.py` — ECE and reliability diagram
- `unit_conversion.py` — `apply_unit_conversion(df, unit_conversion_table)` converts extracted values to standard units before ground-truth matching

**Experiment analysis** (`analysis/`)
- `loaders.py` — load experiment outputs by (dataset, model, date)
- `metrics.py` — `recovery_rate`, `validity_rate`, per-paper summaries
- `ablation.py` — dataset-specific strict/fuzzy matching rules (`get_matching_rules`) and recovery/validity computation across ablations

## Output directory schema

```
data/experiments/
  {dataset}/
    extraction/{model}/{YYYY_mm_dd}/       → 7 JSON checkpoints + final.json
    ablations/ablation{N}/{model}/{date}/   → final.json (+ judge/ subdir)
    judge/{ext_model}/{ext_date}/{judge_model}/{judge_date}/
    judge/{ext_model}/{ext_date}/combined/ → combined.json
    jacobian_lens/{ext_model}/{ext_date}/{lens_model}/{lens_date}/ → jacobian_scores.npz
    analysis/                              → CSV / NPZ outputs
    analysis/figures/                      → PDF / PNG plots
  cross_dataset/                           → cross-dataset probe CSV
```

## Adding a new dataset

1. Create `experiments/configs/{name}.py` exporting `CONFIG: DatasetConfig`.
2. Create `data/{name}/preprocessing.py` to generate `ground_truth.csv` (and `ground_truth_ten.csv` if a subset exists). Use `data/pond/preprocessing.py` or `data/nfix/preprocessing.py` as a template.
3. Set `ground_truth_file` in the config. If units vary across papers, populate `unit_conversion_table` with per-attribute `{unit: multiplier}` entries.
4. Run `python data/{name}/preprocessing.py` to generate the ground truth CSVs.
5. Run `run_extraction.py --dataset {name}` to verify the pipeline end-to-end.

## Adding a new model

Add an entry to `MODEL_REGISTRY` in `experiments/model_registry.py`.
