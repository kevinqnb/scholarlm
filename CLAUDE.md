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

## The harness

This repo is developed through the harness. Read `notes/hub/conventions.md` first —
it is the source of truth for the config standard, the experiment-ID scheme, what a
run writes, and the working loop: `/develop` then `/devlog` for building code,
`/experiment` then `/explog` then `/debrief` for running experiments.

This is a repo where the config standard is retrofitted, not native: keep a
`experiments/experiment-configs/{dataset}/{type}/<id>/<id>.yaml` for every
experiment so runs stay reproducible, but the layout doesn't have to match the
reference repo (`coastal-crawler`).

**No magic numbers.** Every value that would change between runs — dataset, model,
paper subset, per-experiment sampling parameters, and (since 2026-09-14, Phase F of
the restructure) walltime — belongs in an experiment config's `params` block, not
hardcoded in runner code: `params.walltime` is required on every experiment config
that resolves to a GPU job, no model-config fallback, since walltime is a function of
model x dataset size x experiment type, not the model alone. Fixed, repo-wide values
that genuinely don't vary per run (per-model sampling defaults, SGE serve params, the
`omp`/`gpu_memory`/`gpu_capability`/`gpu_type` resource request) live one YAML per
model under `experiments/model-configs/{kind}/<model>.yaml`; `experiments/config.yaml`
holds only the global seed (`defaults.seed`) — every experiment config's own `seed` is
checked against it at run time, with no fallback. It used to also carry per-model
sampling/serve params and SGE cluster settings; that dead content was removed
2026-09-14 (see
`notes/scholarlm/builds/2026-09-13-experiment-config-restructure-01.md`).

## Development log

`devlog/<id>.md` is the public, curated record of AI-assisted work on a build or
experiment — my prompts + a short summary + commit hashes. `<id>` is the same
contract id as the experiment config and the private note; a `devlog/` file exists only
for an `<id>` that also has a build or experiment note. Plain refactors and bug fixes
do not get one — their commit message and `Claude-Session:` trailer cover them.

`/devlog <id>` writes it at the end of a `/develop` session: a `## Session <date>`
block with that session's prompts, a 3–5 sentence summary naming the configs, and the
session's commit hashes. `/devlog` also commits the implementation code this session
produced. The `devlog/` entry itself is a trailing commit — every hash it lists
already exists and it does not list itself. Public file: same review bar as a commit
message, not a second copy of the private note. See `devlog/README.md`.

## Entry points

The contract-standard way to run an experiment:

```bash
python experiments/run_<type>.py experiments/experiment-configs/{dataset}/{type}/<id>/<id>.yaml
bash experiments/submit.sh <id> [--dry-run]     # or submit to SGE
```

Every runner takes one positional `config` argument — no override flags. There is no
adapter script: `experiments/submit.sh` resolves the experiment's runner, GPU need,
and SGE resource request via `experiments/_resolve_job.py`, then qsubs
`experiments/_submit_job.sh <id>`, which brings up a vLLM server in-job when the
model needs one (querying `/health` before reuse) and calls the resolved runner
directly with the experiment's own config path; frontier-model experiments skip the
serve step.

The 14 runners, callable directly for anything the contract's `params` shape doesn't
cover:

```bash
python experiments/run_extraction.py <config>
python experiments/run_ablation.py <config>            # ablation 1-6
python experiments/run_table_cleaning.py <config>
python experiments/run_judge_local.py <config>          # vLLM judge, local
python experiments/run_judge_interp.py <config>         # NNsight judge, collects activations
python experiments/run_judge_combine.py <config>        # explicit extraction_id/judge_ids, not directory-scanning
python experiments/run_jacobian_lens.py <config>         # NNsight, Jacobian-lens j-scores (JacobianLensLM)
python experiments/run_representation_lm.py <config>
python experiments/run_attribution.py <config>
python experiments/run_baseline_gliner.py <config>
python experiments/run_baseline_nuextract.py <config>
python experiments/run_baseline_chatextract.py <config>
python experiments/run_ocr.py <config>
python experiments/process_pdfs.py <config>
python experiments/run_probe_augment.py <config>
```

Run any of them with `--help` for the full flag set. Available datasets are the files
in `experiments/dataset-configs/*.py`; available models are the YAML files under
`experiments/model-configs/{kind}/`, one directory per experiment kind (`extraction`,
`baseline`, `vllm_judge`, `interp_judge`, `jacobian_lens`, `representation_lm`, `ocr`).

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
| `SCHOLARLM_ROOT` | absolute path to this repo, used by `experiments/_submit_job.sh` |
| `SGE_PROJECT` | SGE project allocation (`-P` flag) |
| `OPENAI_API_KEY` / `GEMINI_API_KEY` | required only for frontier-model runs |

## Running tests

`pytest` is part of the `dev` extra, not the default install:

```bash
uv run --extra dev pytest
```

## Repo layout

```
src/scholarlm/                   Core library — pipeline, config, probe/calibration utilities
experiments/                     Runner scripts, path helpers (utils.py)
experiments/dataset-configs/     One DatasetConfig per dataset (pond.py, nfix.py, …)
experiments/model-configs/       One YAML per model, by kind (extraction/, vllm_judge/, …)
experiments/experiment-configs/  {dataset}/{type}/<id>/<id>.yaml, committed, one per experiment
experiments/results/             Id-addressed run output — gitignored, never committed
analysis/                        Experiment analysis code and notebooks
examples/                        Jupyter notebooks
data/experiments/                Legacy (pre-restructure) output tree — frozen, still read by
                                  some analysis code; new runs write to experiments/results/
```

## Key concepts

**Extraction pipeline** (`MeasurementLM` in `src/scholarlm/measurementlm.py`)
Seven sequential steps: entities → attributes → entity_prov → attribute_prov → events → values → final. Each step has a JSON checkpoint; `--resume` skips steps whose output already exists.

**Ablations** (`src/scholarlm/measurementlm_ablation{1–6}.py`)
Each ablation subclass overrides one or more pipeline steps. Run via `experiments/run_ablation.py`.

**DatasetConfig / ModelConfig** (`src/scholarlm/config.py`)
Single source of truth for dataset- and model-specific values. Dataset config files
live in `experiments/dataset-configs/`; model configs in `experiments/model-configs/`
(`utils.get_model_config` bridges a model-config YAML into a `ModelConfig`).

**Path helpers** (`experiments/utils.py`)
Every path in the output tree is constructed here (`result_dir`, `find_result_dir`,
`experiment_config_dir`, `load_model_config`, `classify_gpu_need`, …) — this absorbed
the old `experiments/paths.py`. Never build paths by hand in scripts.

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

New runs write to the id-addressed tree, via `experiments/utils.py`'s `result_dir`:

```
experiments/results/
  {dataset}/{experiment-type}/<id>/        → run output (+ judge/ subdir where applicable)
  out/                                      → SGE stdout/stderr only, not pipeline output
```

Two exceptions write elsewhere and only leave a manifest under `experiments/results/`:
`run_probe_augment.py`'s real output (`probe_dataset<suffix>.json`, the diagnostic
file, `probe_augment_cache.json`) goes to `data/{dataset}/` unchanged, with only a
`run_metadata.json` recording params/seed under
`experiments/results/{dataset}/probe_augment/<id>/`; `process_pdfs.py` writes to
`data/{dataset}/processed_pdfs/`, not under `experiments/results/` at all.

The pre-restructure tree stays frozen on disk (not regenerated by new runs), still
read by some analysis code (`analysis/calibration_updated.py`, `analysis/ablation.py`)
against the exact dates they key off:

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

1. Create `experiments/dataset-configs/{name}.py` exporting `CONFIG: DatasetConfig`.
2. Create `data/{name}/preprocessing.py` to generate `ground_truth.csv` (and `ground_truth_ten.csv` if a subset exists). Use `data/pond/preprocessing.py` or `data/nfix/preprocessing.py` as a template.
3. Set `ground_truth_file` in the config. If units vary across papers, populate `unit_conversion_table` with per-attribute `{unit: multiplier}` entries.
4. Run `python data/{name}/preprocessing.py` to generate the ground truth CSVs.
5. Add an `extraction` model-config and a `experiment-configs/{name}/extraction/<id>/<id>.yaml`,
   then run `run_extraction.py` against it to verify the pipeline end-to-end.

## Adding a new model

Add `experiments/model-configs/{kind}/<model>.yaml` (`kind` matches the runner's own
model-configs subdirectory — see `utils.load_model_config`'s docstring for the full
list: `extraction`, `baseline`, `ocr`, `vllm_judge`, `interp_judge`, `jacobian_lens`,
`representation_lm`). There is no separate registry file anymore — every runner
(extraction/ablation/table-cleaning/baseline/ocr and the judge/interpretability
family: `run_judge_local.py`, `run_judge_interp.py`, `run_jacobian_lens.py`,
`run_representation_lm.py`, `run_attribution.py`) reads model params from these YAML
files via `utils.load_model_config`/`get_model_config`, and fails loud (a
`FileNotFoundError` naming the available models for that kind) on an unknown key.

A `vllm_server`- or `direct_gpu`-need model-config also needs a `resources:` block
(`gpu_memory`/`gpu_capability`/`omp`, optionally `gpu_type`) for `experiments/submit.sh`
to resolve an SGE request for it — `utils.classify_gpu_need` raises rather than
guessing if one's missing. `walltime` does **not** go here (moved out 2026-09-14,
Phase F of the restructure): it's required in the experiment config's own
`params.walltime` instead, since it varies by dataset size and experiment type, not
just model — `_resolve_job.py` fails loud if it's absent, for every experiment type.
Don't invent GPU resource numbers to close a missing `resources:` block — infer them
from prior real runs the way `interp_judge/qwen-2.5-7b.yaml`'s own `resources:`
comment does, or ask.
